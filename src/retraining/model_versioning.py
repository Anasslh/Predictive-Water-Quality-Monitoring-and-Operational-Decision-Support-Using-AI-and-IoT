"""
model_versioning.py — Timestamped model persistence with bounded version history.

DESIGN
------
Models are serialized as pickle files. Pickle works uniformly for any
ParameterModel subclass (XGBoost, RF, SVR, …) because it serializes the
entire Python object graph, including the fitted underlying estimator and
all ParameterModel attributes (_is_fitted, parameter_name, model_version).

File naming convention:
    {parameter_name}_{model_version}_{YYYYMMDD_HHMMSS}.pkl
    e.g.  EC_xgb_ec_v1_final_20260719_143052.pkl

At most max_versions files matching the pattern are kept on disk.
When a new version is saved that would exceed the limit, the oldest
files (by name, which is lexicographically sortable because of the
ISO-format timestamp) are deleted first.

ROLLBACK
--------
To roll back to a previous version, call load_specific_version() with the
timestamp string (YYYYMMDD_HHMMSS) of the desired checkpoint. All available
timestamps can be listed with list_versions().

CRASH SAFETY
------------
All JSON writes in this module use the write-to-.tmp + os.replace() pattern
(see _atomic_write_json). This guarantees that a process crash, power loss, or
exception during json.dump leaves the previous file intact rather than a
partially-written or empty file. os.replace() is an atomic rename on POSIX
filesystems: the destination either has the old content or the new content —
it can never have partial content from the new write.

CONCURRENCY LIMITATION — assumed out-of-scope, documented explicitly
--------------------------------------------------------------------
No file locking (fcntl.flock, portalocker, etc.) is implemented. This is
intentional and appropriate for the current deployment model:
  - Single process per station.
  - Single writer (one ParameterMonitor + one RetrainManager) per models_store.
  - No concurrent processes share the same models_store path.

In a multi-process scenario (e.g. two monitoring processes for the same station
running in parallel, or a web server with multiple worker processes), concurrent
writes could race even with os.replace() — the last writer wins, silently
discarding the other's update. For that context, add cross-process locking:
  POSIX:          fcntl.flock(fd, fcntl.LOCK_EX)
  Cross-platform: portalocker library (pip install portalocker)

The cost of not implementing this now is zero, because the deployment is
single-process. Do NOT add locking speculatively — it adds complexity and
a new failure mode (deadlock on crash) without benefit in the current context.
"""

import json
import os
import pickle
import warnings
from datetime import datetime, timezone
from pathlib import Path

from src.models.base import ParameterModel

# ── Current-model pointer ──────────────────────────────────────────────────────
# Written by write_current_model_pointer() after every approved promotion.
# Read by load_current_model() at startup to find the active model.
CURRENT_MODEL_FILENAME = "current_model.json"


def save_model_version(
    model: ParameterModel,
    models_store_path: str | Path,
    max_versions: int = 3,
) -> Path:
    """
    Serialize a ParameterModel to a timestamped pickle file.

    After saving, any files for the same (parameter_name, model_version) pair
    that exceed max_versions are deleted (oldest first).

    Parameters
    ----------
    model : ParameterModel
        A fitted model. model._is_fitted is preserved in the pickle.
    models_store_path : str or Path
        Directory where versioned files are stored. Created if absent.
    max_versions : int
        Maximum number of checkpoint files to retain (default 3).
        Minimum enforced: 1.

    Returns
    -------
    Path  — path of the newly saved file.
    """
    store = Path(models_store_path)
    store.mkdir(parents=True, exist_ok=True)

    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{model.parameter_name}_{model.model_version}_{ts}.pkl"
    path  = store / fname

    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Enforce max_versions — delete oldest surplus files
    _enforce_max_versions(store, model.parameter_name, model.model_version, max(1, max_versions))

    return path


def load_latest_version(
    parameter_name: str,
    model_version: str,
    models_store_path: str | Path,
) -> ParameterModel:
    """
    Load the most recently saved checkpoint for a (parameter, version) pair.

    Parameters
    ----------
    parameter_name : str
        e.g. "EC", "pH", "Turbidity"
    model_version : str
        e.g. "xgb_ec_v1_final"
    models_store_path : str or Path

    Returns
    -------
    ParameterModel  — deserialized model instance.

    Raises
    ------
    FileNotFoundError  if no matching checkpoint exists.
    """
    versions = _list_version_paths(models_store_path, parameter_name, model_version)
    if not versions:
        raise FileNotFoundError(
            f"No checkpoint found for parameter='{parameter_name}', "
            f"model_version='{model_version}' in {models_store_path}."
        )
    return _load_pickle(versions[-1])


def load_specific_version(
    timestamp: str,
    parameter_name: str,
    model_version: str,
    models_store_path: str | Path,
) -> ParameterModel:
    """
    Load a specific checkpoint by its timestamp string.

    Parameters
    ----------
    timestamp : str
        Format "YYYYMMDD_HHMMSS" — as returned by list_versions().
    parameter_name : str
    model_version : str
    models_store_path : str or Path

    Returns
    -------
    ParameterModel

    Raises
    ------
    FileNotFoundError  if the requested timestamp does not exist on disk.
    """
    store = Path(models_store_path)
    fname = f"{parameter_name}_{model_version}_{timestamp}.pkl"
    path  = store / fname
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            f"Available: {[p.name for p in _list_version_paths(store, parameter_name, model_version)]}"
        )
    return _load_pickle(path)


def list_versions(
    parameter_name: str,
    model_version: str,
    models_store_path: str | Path,
) -> list[str]:
    """
    Return timestamps of all available checkpoints, oldest first.

    Returns
    -------
    list[str]  — timestamp strings ("YYYYMMDD_HHMMSS"), empty if none.
    """
    paths = _list_version_paths(models_store_path, parameter_name, model_version)
    return [_extract_timestamp(p) for p in paths]


# ── Startup pointer — survives process restarts ───────────────────────────────

def write_current_model_pointer(
    models_store_path: str | Path,
    model_pkl_path: str | Path,
    approval_id: str,
    parameter_name: str,
    model_version: str,
) -> Path:
    """
    Write (or overwrite) the current-model pointer after a successful promotion.

    Creates {models_store_path}/current_model.json, which records which versioned
    pkl is the active production model. This file is the authoritative source of
    truth for restarts: load_current_model() reads it before falling back to the
    original bootstrap JSON.

    Called automatically by record_decision() on every approved promotion.
    Never call this manually unless you are implementing a custom promotion flow.

    Parameters
    ----------
    models_store_path : str | Path
        Root of the model store (same path used for save_model_version).
    model_pkl_path : str | Path
        Absolute path to the versioned pkl just written by save_model_version().
    approval_id : str
        Human-traceable ID linking this pointer to the approval record.
    parameter_name : str
    model_version : str

    Returns
    -------
    Path  — path of the written pointer file.
    """
    store = Path(models_store_path)
    pointer = {
        "model_path":     str(Path(model_pkl_path).resolve()),
        "approval_id":    approval_id,
        "promoted_at":    datetime.now(timezone.utc).isoformat(),
        "parameter_name": parameter_name,
        "model_version":  model_version,
    }
    pointer_path = store / CURRENT_MODEL_FILENAME
    _atomic_write_json(pointer_path, pointer, indent=2)
    return pointer_path


def load_current_model(models_store_path: str | Path) -> "ParameterModel | None":
    """
    Load the active production model from the current-model pointer, if it exists.

    This is the correct entry point for any code that needs the production model
    after a process restart. It reads {models_store_path}/current_model.json and
    deserializes the versioned pkl it points to.

    Usage pattern (startup)
    -----------------------
        model = load_current_model(models_store_path)
        if model is None:
            # No approvals on record yet — load the original bootstrap model.
            model = load_bootstrap_model(original_json_path)
        manager = RetrainManager(...)
        manager.initialize(initial_data, model)

    Parameters
    ----------
    models_store_path : str | Path
        Root of the model store. Must contain current_model.json to succeed.

    Returns
    -------
    ParameterModel | None
        The deserialized approved model, or None if no pointer file exists or
        if the pointed pkl is missing (e.g. manually deleted).
    """
    pointer_path = Path(models_store_path) / CURRENT_MODEL_FILENAME
    if not pointer_path.exists():
        return None

    with open(pointer_path, encoding="utf-8") as f:
        pointer = json.load(f)

    pkl_path = Path(pointer["model_path"])
    if not pkl_path.exists():
        warnings.warn(
            f"current_model.json points to {pkl_path} which no longer exists. "
            f"Falling back to bootstrap model. Consider re-running an approval.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    return _load_pickle(pkl_path)


def read_current_model_pointer(models_store_path: str | Path) -> dict | None:
    """
    Return the raw pointer metadata without loading the model.

    Useful for startup logging, auditing, or health checks.

    Returns
    -------
    dict | None  — the pointer JSON content, or None if no pointer exists.
    """
    pointer_path = Path(models_store_path) / CURRENT_MODEL_FILENAME
    if not pointer_path.exists():
        return None
    with open(pointer_path, encoding="utf-8") as f:
        return json.load(f)


# ── Private helpers ────────────────────────────────────────────────────────────

def _atomic_write_json(path: Path, data: dict, **dump_kwargs) -> None:
    """
    Write *data* as JSON to *path* atomically via a .tmp sibling file.

    CRASH SAFETY (the purpose of this function)
    -------------------------------------------
    Writing JSON directly to the target file is not safe: if the process is
    killed between open() and close(), the file is left empty or half-written.
    The next startup then fails to parse it, losing the pointer/counter state.

    This function writes to a sibling .tmp file first. If json.dump raises
    (disk-full, OOM, SIGKILL between open and close), the .tmp is incomplete
    but *path* still has its previous valid content. os.replace() is called
    only after the write is complete and the file handle is closed — at that
    point, the rename is atomic on POSIX (the kernel swaps inode references
    without a window where *path* is missing or partial).

    NOT A CONCURRENCY LOCK
    ----------------------
    This function does not protect against two processes writing to the same
    path simultaneously. See the module-level docstring (CONCURRENCY LIMITATION)
    for the rationale and the remedy if multi-process deployment is ever needed.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, **dump_kwargs)
    os.replace(tmp, path)


def _list_version_paths(
    models_store_path: str | Path,
    parameter_name: str,
    model_version: str,
) -> list[Path]:
    """Return matching .pkl paths sorted oldest → newest (lexicographic on timestamp)."""
    store   = Path(models_store_path)
    pattern = f"{parameter_name}_{model_version}_*.pkl"
    return sorted(store.glob(pattern))


def _enforce_max_versions(
    store: Path,
    parameter_name: str,
    model_version: str,
    max_versions: int,
) -> None:
    paths = _list_version_paths(store, parameter_name, model_version)
    surplus = paths[:max(0, len(paths) - max_versions)]
    for old in surplus:
        try:
            old.unlink()
        except OSError as exc:
            warnings.warn(f"Could not delete old model version {old}: {exc}")


def _extract_timestamp(path: Path) -> str:
    # Filename format: {param}_{version}_{YYYYMMDD_HHMMSS}.pkl
    # Timestamp is the last two '_'-separated segments before .pkl
    stem_parts = path.stem.split("_")
    return "_".join(stem_parts[-2:])   # "YYYYMMDD_HHMMSS"


def _load_pickle(path: Path) -> ParameterModel:
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Consecutive-rejection counter — survives process restarts ─────────────────
# Written on every rejection or acceptance so the count is never lost on crash.
# Stored as a sibling JSON file next to current_model.json in the same store.

REJECTION_COUNTER_FILENAME = "rejection_counter.json"


def write_rejection_counter(
    models_store_path: str | Path,
    count: int,
    parameter_name: str = "",
) -> Path:
    """
    Persist the consecutive-rejection counter for this model store.

    Parameters
    ----------
    models_store_path : str | Path
        Root of the model store — same path used by RetrainManager.
    count : int
        Current consecutive-rejection count (0 means counter was just reset).
    parameter_name : str
        Informational only; not used for logic.

    Returns
    -------
    Path  — path of the written counter file.
    """
    store = Path(models_store_path)
    store.mkdir(parents=True, exist_ok=True)
    payload = {
        "consecutive_rejections": int(count),
        "parameter_name":         parameter_name,
        "updated_at":             datetime.now(timezone.utc).isoformat(),
    }
    path = store / REJECTION_COUNTER_FILENAME
    _atomic_write_json(path, payload, indent=2)
    return path


def read_rejection_counter(models_store_path: str | Path) -> int:
    """
    Return the persisted consecutive-rejection count, or 0 if no file exists.

    Called by RetrainManager.__init__() to restore the counter after a restart.
    """
    path = Path(models_store_path) / REJECTION_COUNTER_FILENAME
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return int(data.get("consecutive_rejections", 0))
