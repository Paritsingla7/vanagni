"""Export the demo as a static site (no server), for hosting on Vercel or any static host.

Copies the frontend and the files the Flask API would have served -- the same pre-computed
results, nothing re-scored -- into one folder:

    out/index.html            (marked data-static="1", so app.js reads files, not /api)
    out/static/...            frontend assets
    out/data/ablation.json, out/data/methodology.json
    out/data/risk/index.json, out/data/risk/grid.geojson, out/data/risk/<date>.json

    python -m wfpin.serve.export [--config config/uttarakhand.yaml] [--out site]
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from wfpin.config import cli_args, paths
from wfpin.serve.api import STATIC_DIR

REQUIRED = ["ablation.json", "methodology.json", "risk/index.json", "risk/grid.geojson"]


def export_site(data_dir: Path, out: Path) -> dict:
    missing = [name for name in REQUIRED if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"{missing} not in {data_dir}. Run evaluate.py first.")
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(STATIC_DIR, out / "static")

    html = (STATIC_DIR / "index.html").read_text()
    if '<html lang="en">' not in html:
        raise ValueError("index.html has no <html lang=\"en\"> tag to mark as static")
    (out / "index.html").write_text(html.replace('<html lang="en">',
                                                 '<html lang="en" data-static="1">', 1))

    (out / "data" / "risk").mkdir(parents=True)
    for name in ("ablation.json", "methodology.json"):
        shutil.copy2(data_dir / name, out / "data" / name)
    index = json.loads((data_dir / "risk" / "index.json").read_text())
    days = sorted(p for p in (data_dir / "risk").glob("*.json") if p.name != "index.json")
    in_range = [p for p in days
                if any(r["start"] <= p.stem <= r["end"] for r in index["ranges"])]
    for p in [data_dir / "risk" / "index.json", data_dir / "risk" / "grid.geojson", *in_range]:
        shutil.copy2(p, out / "data" / "risk" / p.name)
    # Daily files change when the pipeline re-runs; the frontend never does in between.
    (out / "vercel.json").write_text(json.dumps({
        "cleanUrls": True,
        "headers": [{"source": "/data/(.*)",
                     "headers": [{"key": "Cache-Control", "value": "public, max-age=300"}]}],
    }, indent=2))
    return {"days": len(in_range), "seasons": [r["season"] for r in index["ranges"]]}


def _add_args(parser) -> None:
    parser.add_argument("--out", default="site", help="output folder (replaced if it exists)")
    parser.add_argument("--data-dir", help="override the config's processed-data directory")


if __name__ == "__main__":
    cfg, args = cli_args("Export the Vanagni demo as a static site", _add_args)
    data = Path(args.data_dir) if args.data_dir else paths(cfg).processed
    info = export_site(data, Path(args.out))
    print(f"wrote {args.out}/: {info['days']} days, seasons {info['seasons']}")
