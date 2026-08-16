"""Live SQL Server integration test for schema, ETL idempotency and reads.

The test is skipped unless ``WQD_TEST_DB_CONN_STR`` targets an isolated test
database. It never uses the deployment ``DB_CONN_STR`` and deletes only rows
created under its unique site identifier.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from uuid import uuid4

import pytest

import pyodbc

from dashboard.services.db_service import fetch_historical_parameters
from warm_tier_etl import apply_schema, ingest_rows, parse_archive_file


CONNECTION_STRING = os.getenv("WQD_TEST_DB_CONN_STR", "")


@pytest.mark.skipif(
    not CONNECTION_STRING,
    reason="set WQD_TEST_DB_CONN_STR to an isolated SQL Server database",
)
def test_live_schema_ingest_duplicate_update_and_query(tmp_path):
    site_id = f"pytest-{uuid4().hex}"
    now = datetime.now(timezone.utc).replace(microsecond=0)
    source = tmp_path / "warm_integration.jsonl"
    records = [
        {
            "timestamp": (now - timedelta(days=2)).isoformat(),
            "parameter_name": "Generic-A",
            "predicted_value": 10.0,
            "actual_value": 11.0,
            "shap_top_features": [],
            "is_anomaly": True,
            "anomaly_score": 0.9,
            "retrain_alert": None,
        },
        {
            "timestamp": (now - timedelta(days=1)).isoformat(),
            "parameter_name": "Generic-A",
            "predicted_value": 12.0,
            "actual_value": None,
            "shap_top_features": [],
            "is_anomaly": None,
            "anomaly_score": None,
            "retrain_alert": None,
        },
        {
            "timestamp": (now - timedelta(hours=12)).isoformat(),
            "parameter_name": "pH",
            "predicted_value": 7.1,
            "actual_value": 7.0,
            "shap_top_features": [],
            "is_anomaly": False,
            "anomaly_score": 0.1,
            "retrain_alert": None,
        },
    ]
    source.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    apply_schema(CONNECTION_STRING)
    parsed = parse_archive_file(source, site_id=site_id, unit_override="unit-X")
    try:
        first = ingest_rows(CONNECTION_STRING, parsed.rows)
        assert (first.inserted, first.updated, first.unchanged) == (3, 0, 0)

        second = ingest_rows(CONNECTION_STRING, parsed.rows)
        assert (second.inserted, second.updated, second.unchanged) == (0, 0, 3)

        result = fetch_historical_parameters(
            CONNECTION_STRING,
            site_id,
            now - timedelta(days=7),
            now + timedelta(days=1),
        )
        assert list(result.params) == ["Generic-A", "pH"]
        generic = result.params["Generic-A"]
        assert [r.timestamp for r in generic.records] == sorted(
            r.timestamp for r in generic.records
        )
        assert generic.records[0].is_anomaly is True
        assert generic.records[1].actual_value is None
        assert generic.records[1].is_anomaly is None
        assert result.metrics.row_count == 3
        assert result.metrics.connection_ms >= 0
        assert result.metrics.query_ms >= 0
        assert result.metrics.transform_ms >= 0
    finally:
        connection = pyodbc.connect(CONNECTION_STRING, autocommit=False)
        cursor = connection.cursor()
        cursor.execute("DELETE dbo.Fact_WaterQuality WHERE SiteId = ?", site_id)
        connection.commit()
        cursor.close()
        connection.close()
