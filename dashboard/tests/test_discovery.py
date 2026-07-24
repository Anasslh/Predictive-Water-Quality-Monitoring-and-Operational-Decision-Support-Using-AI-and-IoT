"""Parameter discovery tests."""

from __future__ import annotations

from pathlib import Path

from dashboard.services.discovery import discover_parameters


def test_discover_from_valid_dir(exports_dir):
    assert discover_parameters(exports_dir) == ["EC"]


def test_discover_ignores_status_tmp_and_subdirs(exports_dir_messy):
    # EC.jsonl + pH.jsonl are real; *_status.json, *.tmp and _canary_disabled/ ignored.
    assert discover_parameters(exports_dir_messy) == ["EC", "pH"]


def test_discover_missing_dir(tmp_path):
    assert discover_parameters(tmp_path / "does_not_exist") == []


def test_discover_empty_dir(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert discover_parameters(empty) == []


def test_discover_status_only_not_a_parameter(tmp_path):
    d = tmp_path / "exports"
    d.mkdir()
    (d / "EC_status.json").write_text("{}", encoding="utf-8")
    assert discover_parameters(d) == []
