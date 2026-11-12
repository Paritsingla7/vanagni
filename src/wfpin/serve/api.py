"""Flask API + static frontend for the 4 demo screens.

FR-7.1-7.3, API_CONTRACT.md: /api/risk?date=, /api/ablation,
/api/methodology (and /api/dates for the date picker). Read-only: every
endpoint reads a file the offline pipeline already wrote -- nothing here
trains, scores, or writes (ARCHITECTURE.md §2). A missing file is a
build-not-run error and the response says so.

One server can carry several regions: /api/regions lists them, and every data endpoint
takes `region=<id>` (default: the first region listed).

    python -m wfpin.serve.api [--config config/uttarakhand.yaml] [--port 5000]
    python -m wfpin.serve.api --regions config/uttarakhand.yaml config/odisha.yaml
"""
from __future__ import annotations

import json
from datetime import date
from functools import lru_cache
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from wfpin.config import cli_args, load_config, paths

STATIC_DIR = Path(__file__).parent / "static"


def _error(status: int, code: str, message: str):
    return jsonify({"error": code, "message": message}), status


def region_dirs(configs: list[dict]) -> dict[str, Path]:
    """region id -> processed-data directory, in the order given (the first is the default)."""
    return {cfg["region"]["name"]: paths(cfg).processed for cfg in configs}


def region_notes(configs: list[dict]) -> dict[str, str]:
    """region id -> its `region.site_note` (a line the site shows on that region's pages)."""
    return {cfg["region"]["name"]: cfg["region"]["site_note"]
            for cfg in configs if cfg["region"].get("site_note")}


def region_list(regions: dict[str, Path], notes: dict[str, str] | None = None) -> list[dict]:
    notes = notes or {}
    return [{"id": rid, "name": rid.replace("_", " ").title(),
             **({"note": notes[rid]} if rid in notes else {})} for rid in regions]


def create_app(regions: dict[str, Path] | Path, notes: dict[str, str] | None = None) -> Flask:
    """`regions` maps region id -> its processed-data directory; a bare Path is one region.
    `notes` (region id -> text) are shown on that region's pages."""
    if isinstance(regions, Path):
        regions = {"default": regions}
    if not regions:
        raise ValueError("create_app needs at least one region")
    app = Flask(__name__, static_folder=None)
    default = next(iter(regions))

    @lru_cache(maxsize=len(regions))
    def grid_features(region: str) -> dict[str, dict]:
        """cell_id -> GeoJSON geometry, read once per region (geometry never changes)."""
        grid = json.loads((regions[region] / "risk" / "grid.geojson").read_text())
        return {str(f["properties"]["cell_id"]): f["geometry"] for f in grid["features"]}

    def _region():
        """(region id, data dir), or (None, error response) for an unknown id."""
        rid = request.args.get("region") or default
        if rid not in regions:
            return None, _error(400, "unknown_region",
                                f"No region {rid!r}. Known regions: {', '.join(regions)}.")
        return rid, regions[rid]

    def _passthrough(name: str, code: str):
        rid, data_dir = _region()
        if rid is None:
            return data_dir
        path = data_dir / name
        if not path.exists():
            return _error(503, code, f"{name} does not exist for {rid}. Run evaluate.py "
                                     "before starting the server.")
        return app.response_class(path.read_text(), mimetype="application/json")

    @app.get("/api/regions")
    def region_index():
        return jsonify({"default": default, "regions": region_list(regions, notes)})

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
        rid, data_dir = _region()
        if rid is None:
            return data_dir
        risk_dir = data_dir / "risk"
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
        geometry = grid_features(rid)
        features = [
            {"type": "Feature", "geometry": geometry[cid],
             "properties": {"cell_id": int(cid), "risk": r, "fire_next_day": y}}
            for cid, (r, y) in payload["cells"].items()
        ]
        return jsonify({"type": "FeatureCollection", "date": payload["date"], "features": features})

    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/briefing")
    def briefing():
        """The presenter's walk-through: unlinked, and not for search engines."""
        response = send_from_directory(STATIC_DIR, "briefing.html")
        response.headers["X-Robots-Tag"] = "noindex, nofollow"
        return response

    @app.get("/static/<path:name>")
    def static_files(name: str):
        return send_from_directory(STATIC_DIR, name)

    return app


def _add_args(parser) -> None:
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--data-dir", help="override the config's processed-data directory")
    add_regions_arg(parser)


def add_regions_arg(parser) -> None:
    parser.add_argument("--regions", nargs="+", metavar="CONFIG",
                        help="serve several regions, one config each (the first is the default); "
                             "overrides --config")


def regions_from_args(cfg: dict, args) -> tuple[dict[str, Path], dict[str, str]]:
    """(region id -> data dir, region id -> site note) from --regions, else from --config."""
    if args.regions:
        configs = [load_config(c) for c in args.regions]
        return region_dirs(configs), region_notes(configs)
    data = Path(args.data_dir) if args.data_dir else paths(cfg).processed
    return {cfg["region"]["name"]: data}, region_notes([cfg])


if __name__ == "__main__":
    cfg, args = cli_args("Vanagni demo server (read-only)", _add_args)
    create_app(*regions_from_args(cfg, args)).run(port=args.port, debug=False)  # never debug=True
