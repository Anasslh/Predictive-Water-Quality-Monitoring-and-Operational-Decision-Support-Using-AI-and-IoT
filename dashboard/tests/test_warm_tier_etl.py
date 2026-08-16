"""Unit tests for validated and transactional Warm-tier ingestion."""

from __future__ import annotations

from datetime import datetime, timezone
import json

import pytest

import warm_tier_etl as etl


def _record(**overrides):
    payload = {
        "timestamp": "2026-07-21T12:00:00+00:00",
        "parameter_name": "NewParameter",
        "predicted_value": 10.0,
        "actual_value": None,
        "shap_top_features": [],
        "is_anomaly": None,
        "anomaly_score": None,
        "retrain_alert": None,
    }
    payload.update(overrides)
    return payload


def test_parse_success_generic_parameter_and_nulls(tmp_path):
    path = tmp_path / "archive.jsonl"
    path.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
    result = etl.parse_archive_file(path, site_id="Site-X", unit_override="mg/L")
    assert result.rejected_lines == 0
    assert len(result.rows) == 1
    row = result.rows[0]
    assert row.parameter_name == "NewParameter"
    assert row.unit == "mg/L"
    assert row.actual_value is None
    assert row.is_anomaly is None
    assert len(row.source_hash) == 32


def test_parse_duplicate_timestamp_last_whole_row_wins(tmp_path):
    path = tmp_path / "archive.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(_record(predicted_value=10.0, actual_value=11.0)),
                json.dumps(_record(predicted_value=20.0, actual_value=None)),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result = etl.parse_archive_file(path, site_id="C-1")
    assert result.duplicate_timestamps == 1
    assert len(result.rows) == 1
    assert result.rows[0].predicted_value == 20.0
    assert result.rows[0].actual_value is None


def test_corrupt_row_and_invalid_timestamp_are_rejected_independently(tmp_path):
    path = tmp_path / "archive.jsonl"
    path.write_text(
        "{not json}\n"
        + json.dumps(_record(timestamp="not-a-time"))
        + "\n"
        + json.dumps(_record(timestamp="2026-07-22T00:00:00Z"))
        + "\n",
        encoding="utf-8",
    )
    result = etl.parse_archive_file(path, site_id="C-1")
    assert result.nonempty_lines == 3
    assert result.rejected_lines == 2
    assert result.rejected_line_numbers == (1, 2)
    assert len(result.rows) == 1


def test_empty_file_is_a_clean_noop(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n", encoding="utf-8")
    result = etl.parse_archive_file(path, site_id="C-1")
    assert result.rows == []
    assert result.nonempty_lines == 0


class _Cursor:
    def __init__(self, existing_hash=None, fail_on: str | None = None):
        self.existing_hash = existing_hash
        self.fail_on = fail_on
        self.executed = []
        self.closed = False
        self.timeout = 0
        self.fast_executemany = False

    def execute(self, sql, *params):
        self.executed.append(sql)
        if self.fail_on and self.fail_on in sql:
            raise RuntimeError("simulated database failure")
        return self

    def fetchone(self):
        return None if self.existing_hash is None else (self.existing_hash,)

    def close(self):
        self.closed = True


class _Connection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.autocommit = True
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


def _connect(connection):
    def factory(connection_string, **kwargs):
        assert connection_string == "configured-secret"
        return connection

    return factory


def _row():
    payload = _record()
    return etl._parse_record(
        payload,
        site_id="C-1",
        unit_override="mg/L",
        expected_parameter=None,
        source_file="archive.jsonl",
    )


@pytest.mark.parametrize("state", ["missing", "unchanged", "changed"])
def test_ingestion_reports_insert_update_and_unchanged(state):
    row = _row()
    existing_hash = {
        "missing": None,
        "unchanged": row.source_hash,
        "changed": b"x" * 32,
    }[state]
    expected = {
        "missing": (1, 0, 0),
        "unchanged": (0, 0, 1),
        "changed": (0, 1, 0),
    }[state]
    cursor = _Cursor(existing_hash=existing_hash)
    connection = _Connection(cursor)
    result = etl.ingest_rows(
        "configured-secret", [row], connection_factory=_connect(connection)
    )
    assert (result.inserted, result.updated, result.unchanged) == expected
    assert connection.commits == 1
    assert connection.rollbacks == 0
    assert connection.closed and cursor.closed


def test_ingestion_rolls_back_entire_batch_on_failure():
    cursor = _Cursor(fail_on="INSERT dbo.Fact_WaterQuality")
    connection = _Connection(cursor)
    with pytest.raises(RuntimeError):
        etl.ingest_rows(
            "configured-secret", [_row()], connection_factory=_connect(connection)
        )
    assert connection.commits == 0
    assert connection.rollbacks == 1
    assert connection.closed and cursor.closed


def test_schema_initialization_executes_versioned_batches(tmp_path):
    schema = tmp_path / "schema.sql"
    schema.write_text("SELECT 1;\nGO\nSELECT 2;\n", encoding="utf-8")
    cursor = _Cursor()
    connection = _Connection(cursor)
    etl.apply_schema(
        "configured-secret", schema_path=schema, connection_factory=_connect(connection)
    )
    assert len(cursor.executed) == 2
    assert connection.commits == 1
