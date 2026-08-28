"""Download ERA5-Land hourly weather for the California/Western-US fire region.

Chunked by year to keep each CDS request tractable. Each request returns a zip
wrapping NetCDF; we extract to era5/nc/<chunk>/. Idempotent: skips chunks whose
extracted marker already exists. Run from the repo root with the project venv.
"""
import os
import sys
import zipfile
from pathlib import Path

import cdsapi

REPO = Path(__file__).resolve().parent.parent
RC = REPO / ".cdsapirc"
OUT = REPO / "era5"
NC = OUT / "nc"
OUT.mkdir(exist_ok=True)
NC.mkdir(exist_ok=True)

AREA = [42, -125, 32, -114]  # N, W, S, E — California + Western US
VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
    "volumetric_soil_water_layer_1",
]
TIMES = ["00:00", "06:00", "12:00", "18:00", "23:00"]  # synoptic hours -> daily aggregates
ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]

# (label, year, months) — one request per MONTH; full-year requests exceed the
# CDS per-request cost cap. FIRMS archive ends 2020-06-30.
CHUNKS = (
    [(f"2018-{m:02d}", "2018", [f"{m:02d}"]) for m in range(1, 13)]
    + [(f"2019-{m:02d}", "2019", [f"{m:02d}"]) for m in range(1, 13)]
    + [(f"2020-{m:02d}", "2020", [f"{m:02d}"]) for m in range(1, 7)]
)


def read_credentials(rc: Path):
    url = key = None
    for line in rc.read_text().splitlines():
        if line.startswith("url:"):
            url = line.split(":", 1)[1].strip()
        elif line.startswith("key:"):
            key = line.split(":", 1)[1].strip()
    if not url or not key:
        sys.exit(f"Missing url/key in {rc}")
    return url, key


def main():
    url, key = read_credentials(RC)
    client = cdsapi.Client(url=url, key=key)

    for label, year, months in CHUNKS:
        marker = NC / label / ".done"
        if marker.exists():
            print(f"[{label}] already downloaded, skipping", flush=True)
            continue

        zip_path = OUT / f"era5_{label}.zip"
        print(f"[{label}] requesting {year} months={months[0]}..{months[-1]} ...", flush=True)
        client.retrieve(
            "reanalysis-era5-land",
            {
                "variable": VARIABLES,
                "year": year,
                "month": months,
                "day": ALL_DAYS,
                "time": TIMES,
                "area": AREA,
                "data_format": "netcdf",
                "download_format": "zip",
            },
            str(zip_path),
        )

        dest = NC / label
        dest.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)
        zip_path.unlink()
        marker.write_text("ok")
        extracted = sorted(p.name for p in dest.glob("*.nc"))
        print(f"[{label}] extracted: {extracted}", flush=True)

    print("ALL CHUNKS DONE", flush=True)


if __name__ == "__main__":
    main()
