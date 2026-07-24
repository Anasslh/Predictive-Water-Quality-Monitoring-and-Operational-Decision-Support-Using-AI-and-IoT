"""Loader tests: JSONL, status, and full parameter assembly."""

from __future__ import annotations

from pathlib import Path

from dashboard.services.loaders import (
    load_all_parameters,
    load_parameter,
    read_jsonl_records,
    read_status,
)


def test_read_valid_jsonl(exports_dir):
    records, errors = read_jsonl_records(exports_dir / "EC.jsonl")
    assert len(records) == 3
    assert errors == 0
    assert records[0].parameter_name == "EC"


def test_read_jsonl_skips_corrupt_line(exports_dir_messy):
    records, errors = read_jsonl_records(exports_dir_messy / "pH.jsonl")
    assert len(records) == 1          # one good record kept
    assert errors == 1               # one corrupt line counted, not fatal


def test_read_jsonl_missing_file(tmp_path):
    records, errors = read_jsonl_records(tmp_path / "nope.jsonl")
    assert records == [] and errors == 0


def test_read_jsonl_empty_file(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    records, errors = read_jsonl_records(p)
    assert records == [] and errors == 0


def test_read_status_valid(exports_dir):
    status = read_status(exports_dir / "EC_status.json")
    assert status is not None
    assert status.unit == "µS/cm"
    assert status.performance.rmse == 425.64


def test_read_status_missing(tmp_path):
    assert read_status(tmp_path / "nope_status.json") is None


def test_read_status_malformed(tmp_path):
    p = tmp_path / "bad_status.json"
    p.write_text("{ not json", encoding="utf-8")
    assert read_status(p) is None


def test_load_parameter_integration(exports_dir):
    pdata = load_parameter("EC", exports_dir, retention_days=3650)
    assert pdata.has_records
    assert pdata.display_unit == "µS/cm"
    assert pdata.latest_actual_record is not None
    assert pdata.status is not None


def test_load_parameter_missing_status(exports_dir_messy):
    pdata = load_parameter("pH", exports_dir_messy, retention_days=3650)
    assert pdata.status is None         # no status file for pH
    assert pdata.load_errors == 1       # corrupt line surfaced as a count
    assert pdata.has_records


def test_load_all_parameters(exports_dir_messy):
    params = load_all_parameters(["EC", "pH"], exports_dir_messy, retention_days=3650)
    assert set(params.keys()) == {"EC", "pH"}
    assert params["EC"].has_records and params["pH"].has_records
