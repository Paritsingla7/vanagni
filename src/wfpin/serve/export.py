"""Export the demo as a static site (no server), for hosting on Vercel or any static host.

Copies the frontend and the files the Flask API would have served -- the same pre-computed
results, nothing re-scored -- into one folder:

    out/index.html            (marked data-static="1", so app.js reads files, not /api)
    out/briefing.html         the presenter's walk-through at /briefing (unlinked, noindex)
    out/static/...            frontend assets
    out/data/regions.json     the regions on the site, the first is the default (/api/regions)
    out/data/<region>/ablation.json, out/data/<region>/methodology.json
    out/data/<region>/risk/index.json, .../risk/grid.geojson, .../risk/<date>.json

    python -m wfpin.serve.export [--config config/uttarakhand.yaml] [--out site]
    python -m wfpin.serve.export --regions config/uttarakhand.yaml config/odisha.yaml
    npx vercel deploy site --prod      # first time: npx vercel link --cwd site --project vanagni
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from wfpin.config import cli_args
from wfpin.serve.api import STATIC_DIR, add_regions_arg, region_list, regions_from_args

KEEP = {".vercel", ".gitignore"}
PAGES = ["index.html", "briefing.html"]
REQUIRED = ["ablation.json", "methodology.json", "risk/index.json", "risk/grid.geojson"]


def _copy_region(data_dir: Path, out: Path) -> dict:
    """One region's API files into `out` (= data/<region>/): only days inside a test range."""
    (out / "risk").mkdir(parents=True)
    for name in ("ablation.json", "methodology.json"):
        shutil.copy2(data_dir / name, out / name)
    index = json.loads((data_dir / "risk" / "index.json").read_text())
    days = sorted(p for p in (data_dir / "risk").glob("*.json") if p.name != "index.json")
    in_range = [p for p in days
                if any(r["start"] <= p.stem <= r["end"] for r in index["ranges"])]
    for p in [data_dir / "risk" / "index.json", data_dir / "risk" / "grid.geojson", *in_range]:
        shutil.copy2(p, out / "risk" / p.name)
    return {"days": len(in_range), "seasons": [r["season"] for r in index["ranges"]]}


def export_site(regions: dict[str, Path] | Path, out: Path,
                notes: dict[str, str] | None = None) -> dict:
    """`regions` maps region id -> processed-data directory (first = default); a bare Path
    is one region. `notes` (region id -> text) are shown on that region's pages.
    Returns {region: {"days", "seasons"}}."""
    if isinstance(regions, Path):
        regions = {"default": regions}
    for rid, data_dir in regions.items():
        missing = [name for name in REQUIRED if not (data_dir / name).exists()]
        if missing:
            raise FileNotFoundError(f"{rid}: {missing} not in {data_dir}. Run evaluate.py first.")
    # Replace the build, but keep a hosting link (.vercel/) so a re-export redeploys in place.
    if out.exists():
        for child in out.iterdir():
            if child.name in KEEP:
                continue
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    shutil.copytree(STATIC_DIR, out / "static")

    for page in PAGES:   # the briefing is served at /briefing (cleanUrls), unlinked
        html = (STATIC_DIR / page).read_text()
        if '<html lang="en">' not in html:
            raise ValueError(f"{page} has no <html lang=\"en\"> tag to mark as static")
        (out / page).write_text(html.replace('<html lang="en">',
                                             '<html lang="en" data-static="1">', 1))

    info = {rid: _copy_region(data_dir, out / "data" / rid) for rid, data_dir in regions.items()}
    (out / "data" / "regions.json").write_text(json.dumps(
        {"default": next(iter(regions)), "regions": region_list(regions, notes)}, indent=2))
    # Daily files change when the pipeline re-runs; the frontend never does in between.
    (out / "vercel.json").write_text(json.dumps({
        "cleanUrls": True,
        "headers": [{"source": "/data/(.*)",
                     "headers": [{"key": "Cache-Control", "value": "public, max-age=300"}]},
                    {"source": "/briefing(.html)?",
                     "headers": [{"key": "X-Robots-Tag", "value": "noindex, nofollow"}]}],
    }, indent=2))
    return info


def _add_args(parser) -> None:
    parser.add_argument("--out", default="site", help="output folder (replaced if it exists)")
    parser.add_argument("--data-dir", help="override the config's processed-data directory")
    add_regions_arg(parser)


if __name__ == "__main__":
    cfg, args = cli_args("Export the Vanagni demo as a static site", _add_args)
    dirs, notes = regions_from_args(cfg, args)
    info = export_site(dirs, Path(args.out), notes)
    for rid, r in info.items():
        print(f"wrote {args.out}/data/{rid}/: {r['days']} days, seasons {r['seasons']}")
