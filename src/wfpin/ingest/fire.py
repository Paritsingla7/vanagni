"""FIRMS VIIRS 375 m active-fire ingestion for Uttarakhand.

FR-1.1: bulk-download detections for the configured bbox + date window,
writing a manifest (query params, row count, download date) alongside the
raw file so a re-run skips work already done.

Both VIIRS satellites are pulled and kept distinguishable by the raw
`satellite` column (`N` = Suomi-NPP, `N20` = NOAA-20). SNPP covers the whole
window; NOAA-20's standard-processing archive only starts 2018-04-01, so
mixing them roughly doubles overpass density partway through the window.
Which constellation the label uses is a panel-time decision (panel.py), not
an ingest-time one -- ingest keeps both so that choice stays reversible.

    python -m wfpin.ingest.fire [--config config/uttarakhand.yaml]
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from wfpin.config import cli_config

# Historical/standard-processing VIIRS products (DATA_SOURCES.md #1), with
# the first day each one has data -- from FIRMS's own /api/data_availability,
# so the loop doesn't spend calls on dates a sensor can't return.
SENSOR_FIRST_DAY = {
    "VIIRS_SNPP_SP": date(2012, 1, 20),
    "VIIRS_NOAA20_SP": date(2018, 4, 1),
}
AREA_API_DAY_RANGE = 5  # verified cap for VIIRS_*_SP; the API 400s above this
AREA_API_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
API_WORKERS = 4
API_RETRIES = 4
CSV_HEADER_PREFIX = "latitude,longitude"
# The MAP_KEY quota is 5,000 *transactions* per 10 min, and one Area API
# call over this bbox costs ~10 or more of them (it varies) -- so a few
# hundred calls per window, not 5,000. Stay under that with margin; when the
# quota is still hit, FIRMS answers HTTP 400 with one of QUOTA_MESSAGES
# (including a misleading "Invalid MAP_KEY"), which means "wait", not "fail".
QUOTA_WINDOW_S = 600
CALLS_PER_WINDOW = 250
QUOTA_MESSAGES = ("Invalid MAP_KEY", "Exceeding allowed transaction limit")
MAPKEY_STATUS_URL = "https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/"


class _RateLimiter:
    """At most `limit` calls in any rolling `window_s` seconds, across threads."""

    def __init__(self, limit: int, window_s: float):
        self.limit, self.window_s = limit, window_s
        self.calls: deque[float] = deque()
        self.lock = threading.Lock()

    def wait(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                while self.calls and now - self.calls[0] > self.window_s:
                    self.calls.popleft()
                if len(self.calls) < self.limit:
                    self.calls.append(now)
                    return
                sleep_s = self.window_s - (now - self.calls[0]) + 1
            time.sleep(sleep_s)


_limiter = _RateLimiter(CALLS_PER_WINDOW, QUOTA_WINDOW_S)


def _wait_for_quota(map_key: str) -> None:
    """Block until the key's rolling transaction count has room again. A key
    that is genuinely invalid has no status, which raises instead of looping."""
    while True:
        status = requests.get(MAPKEY_STATUS_URL, params={"MAP_KEY": map_key}, timeout=60).json()
        if status["current_transactions"] < status["transaction_limit"] * 0.8:
            return
        print(f"FIRMS quota at {status['current_transactions']}/"
              f"{status['transaction_limit']}, waiting...", flush=True)
        time.sleep(60)


def _area_api_chunks(start: date, end: date) -> list[tuple[date, int]]:
    """(chunk_start_date, day_count) pairs of <=AREA_API_DAY_RANGE days spanning [start, end]."""
    chunks = []
    cur = start
    while cur <= end:
        days = min(AREA_API_DAY_RANGE, (end - cur).days + 1)
        chunks.append((cur, days))
        cur += timedelta(days=days)
    return chunks


def _get_csv(map_key: str, path: str) -> str:
    """GET `AREA_API_BASE/<key>/<path>` with rate limiting and retry. FIRMS
    sometimes answers errors as plain text (even with HTTP 200), so a body
    that isn't a FIRMS CSV is an error, never an empty result. Errors name
    `path`, never the URL -- it embeds the MAP_KEY."""
    url = f"{AREA_API_BASE}/{map_key}/{path}"
    err = "no attempt made"
    for attempt in range(1, API_RETRIES + 1):
        _limiter.wait()
        try:
            resp = requests.get(url, timeout=120)
        except requests.RequestException as exc:
            err = type(exc).__name__
        else:
            if resp.ok and resp.text.startswith(CSV_HEADER_PREFIX):
                return resp.text
            err = f"HTTP {resp.status_code}: {resp.text[:200]!r}"
            if any(msg in resp.text for msg in QUOTA_MESSAGES):
                _wait_for_quota(map_key)
                continue
            if 400 <= resp.status_code < 500 and resp.status_code != 429:
                break  # a malformed request won't fix itself on retry
        if attempt < API_RETRIES:
            time.sleep(10 * attempt)
    raise RuntimeError(f"FIRMS request {path} failed: {err}")


def _fetch_chunk(map_key: str, sensor: str, coords: str, start: date, days: int, path: Path) -> int:
    """One Area API chunk, cached as its own file -- a crash on call 900 of
    1,170 loses one chunk, not the whole pull (ARCHITECTURE.md §3)."""
    if path.exists():
        return -1
    text = _get_csv(map_key, f"{sensor}/{coords}/{days}/{start.isoformat()}")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text)
    tmp.rename(path)
    rows = text.count("\n") - 1
    print(f"[{sensor} {start}+{days}d] {rows} rows", flush=True)
    return rows


def _fetch_area_api(config: dict, map_key: str, chunk_dir: Path) -> pd.DataFrame:
    bbox = config["region"]["bbox"]
    coords = f"{bbox['west']},{bbox['south']},{bbox['east']},{bbox['north']}"
    window_start = date.fromisoformat(config["window"]["start"])
    window_end = date.fromisoformat(config["window"]["end"])

    jobs = []
    for sensor, first_day in SENSOR_FIRST_DAY.items():
        sensor_dir = chunk_dir / sensor
        sensor_dir.mkdir(parents=True, exist_ok=True)
        for start, days in _area_api_chunks(max(window_start, first_day), window_end):
            jobs.append((map_key, sensor, coords, start, days, sensor_dir / f"{start}.csv"))

    with ThreadPoolExecutor(max_workers=API_WORKERS) as pool:
        list(pool.map(lambda job: _fetch_chunk(*job), jobs))

    frames = [pd.read_csv(job[-1]) for job in jobs]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_manual_bulk(config: dict, manual_dir: Path) -> pd.DataFrame | None:
    """Every CSV placed in `manual/` (DATA_SOURCES.md #1's one-time web-form
    download -- FIRMS delivers one `fire_archive_*.csv` per satellite) takes
    priority over the Area API. Bulk requests are usually by country, so they
    are clipped here to the configured bbox + window; the manifest must not
    claim a bbox/window that wasn't actually applied."""
    files = sorted(manual_dir.glob("*.csv"))
    if not files:
        return None
    bbox, window = config["region"]["bbox"], config["window"]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    in_bbox = df["latitude"].between(bbox["south"], bbox["north"]) & df["longitude"].between(
        bbox["west"], bbox["east"]
    )
    in_window = pd.to_datetime(df["acq_date"]).between(window["start"], window["end"])
    return df[in_bbox & in_window].reset_index(drop=True)


def ingest_fire(config: dict, out_dir: str = "data/raw/fire") -> Path:
    """FR-1.1: load manually-downloaded bulk CSVs if present, else the looped
    Area API. Writes the combined raw CSV (source columns only, no derived
    fields -- SCHEMA.md §1) + a manifest, gated by a `.done` marker."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    combined_path = out / "firms_viirs_uttarakhand.csv"
    manifest_path = out / "manifest.json"
    marker = out / ".done"

    if marker.exists():
        print(f"already ingested, skipping (see {manifest_path})", flush=True)
        return combined_path

    manual_dir = out / "manual"
    manual_dir.mkdir(exist_ok=True)
    df = _load_manual_bulk(config, manual_dir)
    if df is not None:
        source = f"manual bulk archive: {sorted(p.name for p in manual_dir.glob('*.csv'))}"
    else:
        map_key = os.environ.get("FIRMS_MAP_KEY")
        if not map_key:
            raise RuntimeError(
                f"No manual bulk CSV under {manual_dir} and FIRMS_MAP_KEY is not set "
                "-- see docs/DATA_SOURCES.md #1."
            )
        source = f"Area API, sensors={list(SENSOR_FIRST_DAY)}"
        df = _fetch_area_api(config, map_key, out / "chunks")

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    df.to_csv(combined_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "source": source,
                "bbox": config["region"]["bbox"],
                "window": config["window"],
                "row_count": len(df),
                "duplicates_dropped": before - len(df),
                "rows_by_satellite": df["satellite"].astype(str).value_counts().to_dict(),
                "downloaded_at": datetime.now().astimezone().isoformat(),
            },
            indent=2,
        )
    )
    marker.write_text("ok")
    return combined_path


if __name__ == "__main__":
    ingest_fire(cli_config("FR-1.1: ingest FIRMS VIIRS detections"))
