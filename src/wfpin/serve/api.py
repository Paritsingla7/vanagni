"""Flask API + static frontend for the 4 demo screens.

FR-7.1-7.3, API_CONTRACT.md: /api/risk?date=, /api/ablation,
/api/methodology (and /api/dates for the date picker). Read-only: every
endpoint reads a file the offline pipeline already wrote -- nothing here
trains, scores, or writes (ARCHITECTURE.md §2). A missing file is a
build-not-run error and the response says so.

    python -m wfpin.serve.api [--config config/uttarakhand.yaml] [--port 5000]
"""
from __future__ import annotations

import json
from datetime import date
from functools import lru_cache
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from wfpin.config import cli_args, paths

STATIC_DIR = Path(__file__).parent / "static"


def _error(status: int, code: str, message: str):
    return jsonify({"error": code, "message": message}), status


def create_app(data_dir: Path) -> Flask:
    app = Flask(__name__, static_folder=None)
    risk_dir = data_dir / "risk"

    @lru_cache(maxsize=1)
    def grid_features() -> dict[str, dict]:
        """cell_id -> GeoJSON geometry, read once (geometry never changes)."""
        grid = json.loads((risk_dir / "grid.geojson").read_text())
        return {str(f["properties"]["cell_id"]): f["geometry"] for f in grid["features"]}

    def _passthrough(name: str, code: str):
        path = data_dir / name
        if not path.exists():
            return _error(503, code, f"{name} does not exist. Run evaluate.py before starting "
                                     "the server.")
        return app.response_class(path.read_text(), mimetype="application/json")

    @app.get("/api/ablation")
    def ablation():
        return _passthrough("ablation.json", "ablation_not_built")

    @app.get("/api/methodology")
    def methodology():
        return _passthrough("methodology.json", "methodology_not_built")

    @app.get("/api/dates")
    def dates():
        return _passthrough("risk/index.json", "risk_not_built")

    @app.get("/api/risk")
    def risk():
        raw = request.args.get("date", "")
        try:
            day = date.fromisoformat(raw)
        except ValueError:
            return _error(400, "bad_date", f"date must be YYYY-MM-DD, got {raw!r}.")
        index_path = risk_dir / "index.json"
        if not index_path.exists():
            return _error(503, "risk_not_built", "risk/index.json does not exist. Run "
                                                 "evaluate.py before starting the server.")
        ranges = json.loads(index_path.read_text())["ranges"]
        if not any(r["start"] <= day.isoformat() <= r["end"] for r in ranges):
            valid = ", ".join(f"{r['start']} to {r['end']}" for r in ranges)
            return _error(400, "date_out_of_range",
                          f"No prediction exists for {day}. Valid dates: {valid}.")
        path = risk_dir / f"{day.isoformat()}.json"
        if not path.exists():
            return _error(404, "risk_not_built",
                          f"Risk file missing for {day} — the evaluation pipeline may not have "
                          "completed for this date. This is a build issue, not a client error.")
        payload = json.loads(path.read_text())
        geometry = grid_features()
        features = [
            {"type": "Feature", "geometry": geometry[cid],
             "properties": {"cell_id": int(cid), "risk": r, "fire_next_day": y}}
            for cid, (r, y) in payload["cells"].items()
        ]
        return jsonify({"type": "FeatureCollection", "date": payload["date"], "features": features})

    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/static/<path:name>")
    def static_files(name: str):
        return send_from_directory(STATIC_DIR, name)

    return app


def _add_args(parser) -> None:
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--data-dir", help="override the config's processed-data directory")


if __name__ == "__main__":
    cfg, args = cli_args("Vanagni demo server (read-only)", _add_args)
    data = Path(args.data_dir) if args.data_dir else paths(cfg).processed
    create_app(data).run(port=args.port, debug=False)  # AUDIT.md: never debug=True
