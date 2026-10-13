"""Phase 5: API contract (API_CONTRACT.md) against synthetic artifacts."""
import json

import pytest

from wfpin.serve.api import create_app


@pytest.fixture
def data_dir(tmp_path):
    risk = tmp_path / "risk"
    risk.mkdir()
    ring = [[79.0, 30.0], [79.05, 30.0], [79.05, 30.05], [79.0, 30.05], [79.0, 30.0]]
    (risk / "grid.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"cell_id": 7},
         "geometry": {"type": "Polygon", "coordinates": [ring]}}]}))
    (risk / "index.json").write_text(json.dumps(
        {"stage": "4b", "ranges": [{"season": 2024, "start": "2024-02-15", "end": "2024-06-15"}]}))
    (risk / "2024-04-12.json").write_text(json.dumps(
        {"date": "2024-04-12", "stage": "4b", "season": 2024, "cells": {"7": [0.34, False]}}))
    (tmp_path / "ablation.json").write_text(json.dumps({"stages": []}))
    return tmp_path


def test_risk_ok(data_dir):
    r = create_app(data_dir).test_client().get("/api/risk?date=2024-04-12")
    body = r.get_json()
    assert r.status_code == 200 and body["date"] == "2024-04-12"
    props = body["features"][0]["properties"]
    assert props == {"cell_id": 7, "risk": 0.34, "fire_next_day": False}
    assert body["features"][0]["geometry"]["type"] == "Polygon"


@pytest.mark.parametrize("query, status, code", [
    ("date=2025-01-01", 400, "date_out_of_range"),
    ("date=nope", 400, "bad_date"),
    ("", 400, "bad_date"),
    ("date=2024-04-13", 404, "risk_not_built"),  # in range, file missing
])
def test_risk_errors(data_dir, query, status, code):
    r = create_app(data_dir).test_client().get(f"/api/risk?{query}")
    assert r.status_code == status and r.get_json()["error"] == code


def test_passthrough_and_missing_files(data_dir):
    client = create_app(data_dir).test_client()
    assert client.get("/api/ablation").get_json() == {"stages": []}
    r = client.get("/api/methodology")
    assert r.status_code == 503 and r.get_json()["error"] == "methodology_not_built"


def test_frontend_is_served(data_dir):
    client = create_app(data_dir).test_client()
    assert b"Vanagni" in client.get("/").data
    assert client.get("/static/app.js").status_code == 200
