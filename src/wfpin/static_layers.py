"""Static layers, computed once and reused across every cell-day.

FR-2.1 terrain (elevation/slope/aspect from SRTM), FR-2.2 land cover
(area-weighted WorldCover class fractions, not a point sample), FR-2.3 human
proximity (road/settlement distance from OSM), FR-2.5 grid definition, plus
the join of FR-2.4's hand-compiled fire calendar (config/, not code) by
district.

Every raster input is warped onto a raster *aligned to the cell lattice*
(an integer number of pixels per cell), so a per-cell statistic is a plain
block reduction over that raster -- no per-pixel DataFrame, no zonal-stats
polygon loop, and no way for a pixel to be attributed to the wrong cell.

    python -m wfpin.static_layers [--config config/uttarakhand.yaml]
"""
from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
import requests
from affine import Affine
from pyproj import Transformer
from rasterio.warp import Resampling, reproject
from shapely.geometry import box

from wfpin.config import cli_config, paths, region_crs

SRTM_VOID = -32768  # SRTM fill/void sentinel; anything at or below this is missing data
WORLDCOVER_NODATA = 0
WORLDCOVER_TILE_DEG = 3

# The lattice is built in a projected, metre-based CRS (config `region.crs`, else
# the UTM zone of the bbox centre) rather than a raw degree grid, which keeps
# cells genuinely resolution_km x resolution_km --
# the old US pipeline's audited defect was 0.25-degree cells whose true
# area varies with latitude.
WGS84_CRS = "EPSG:4326"


@dataclass(frozen=True)
class Lattice:
    """The regular resolution_km grid in the region's projected CRS. Row 0 is
    the southernmost row; cell_id = row * n_cols + col."""

    west_m: float
    south_m: float
    res_m: float
    n_cols: int
    n_rows: int
    crs: str

    def pixels_per_cell(self, px_m: float) -> int:
        k = self.res_m / px_m
        if k != int(k):
            raise ValueError(f"{px_m} m pixels don't tile a {self.res_m} m cell")
        return int(k)

    def raster(self, px_m: float) -> tuple[Affine, tuple[int, int]]:
        """(transform, shape) of a north-up raster aligned to this lattice."""
        k = self.pixels_per_cell(px_m)
        north_m = self.south_m + self.n_rows * self.res_m
        transform = Affine(px_m, 0, self.west_m, 0, -px_m, north_m)
        return transform, (self.n_rows * k, self.n_cols * k)

    def per_cell(self, arr: np.ndarray, reducer=np.nanmean) -> np.ndarray:
        """Block-reduce a lattice-aligned raster to one value per cell,
        returned in cell_id order. Raster row 0 is north, lattice row 0 is
        south -- hence the flip."""
        k = arr.shape[0] // self.n_rows
        blocks = arr.reshape(self.n_rows, k, self.n_cols, k)[::-1]
        return reducer(blocks, axis=(1, 3)).ravel()


def lattice(config: dict) -> Lattice:
    bbox = config["region"]["bbox"]
    res_m = config["grid"]["resolution_km"] * 1000
    crs = region_crs(config)
    to_utm = Transformer.from_crs(WGS84_CRS, crs, always_xy=True)
    west_m, south_m = to_utm.transform(bbox["west"], bbox["south"])
    east_m, north_m = to_utm.transform(bbox["east"], bbox["north"])
    return Lattice(
        west_m=west_m,
        south_m=south_m,
        res_m=res_m,
        n_cols=len(np.arange(west_m, east_m, res_m)),
        n_rows=len(np.arange(south_m, north_m, res_m)),
        crs=crs,
    )


def build_grid(config: dict) -> gpd.GeoDataFrame:
    """FR-2.5: the full lattice over the configured bbox -- cell_id,
    grid_row/grid_col (for 8-neighbour lookups in panel.py), centroid
    lat/lon, state_id. `clip_grid_to_region` then keeps only cells that
    touch the state; cell_ids stay those of the full lattice, so they are
    stable for a given config."""
    lat = lattice(config)
    to_wgs84 = Transformer.from_crs(lat.crs, WGS84_CRS, always_xy=True)
    rows, cols = np.divmod(np.arange(lat.n_rows * lat.n_cols), lat.n_cols)
    x0 = lat.west_m + cols * lat.res_m
    y0 = lat.south_m + rows * lat.res_m
    lon, lat_deg = to_wgs84.transform(x0 + lat.res_m / 2, y0 + lat.res_m / 2)
    return gpd.GeoDataFrame(
        {
            "cell_id": (rows * lat.n_cols + cols).astype("int32"),
            "grid_row": rows.astype("int16"),
            "grid_col": cols.astype("int16"),
            "state_id": config["region"]["state_id"],
            "centroid_lat": lat_deg,
            "centroid_lon": lon,
        },
        geometry=[box(x, y, x + lat.res_m, y + lat.res_m) for x, y in zip(x0, y0, strict=True)],
        crs=lat.crs,
    )


def clip_grid_to_region(grid: gpd.GeoDataFrame, region: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep cells that overlap the state polygon. The bbox lattice is ~2x the
    state's area and the excess is Nepal/UP/Himachal/Tibet -- including
    crop-burning plains that would contaminate a *forest* fire label."""
    shape = region.union_all()
    frac = grid.geometry.intersection(shape).area / grid.geometry.area
    out = grid.assign(in_state_frac=frac.astype("float32"))
    return out[out["in_state_frac"] > 0].reset_index(drop=True)


# --- FR-2.1 terrain ------------------------------------------------------------------


MAX_SRTM_TILES = 120  # sanity ceiling: a bbox needing more is almost certainly a typo
# `elevation` shells out to these. If one is missing it does NOT fail: its tile step falls
# back to `touch`, leaving empty placeholder tiles in ~/.cache/elevation that then poison
# every later run. So check up front.
DEM_TOOLS = ("make", "gdal_translate", "gdalbuildvrt")


def check_dem_tools() -> None:
    missing = [tool for tool in DEM_TOOLS if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(
            f"{missing} not on PATH. Activate the conda env (`conda activate vanagni`) before "
            "downloading the DEM: without them `elevation` silently caches empty tiles."
        )


def srtm_tile_count(bbox: dict) -> int:
    """SRTM ships as 1x1-degree tiles; how many cover the bbox."""
    return (math.ceil(bbox["east"]) - math.floor(bbox["west"])) * (
        math.ceil(bbox["north"]) - math.floor(bbox["south"])
    )


def download_dem(config: dict, out_path: Path | None = None) -> Path:
    """FR-2.1: fetch SRTM 30m DEM for the configured bbox via `elevation`.

    `elevation`'s Python API handles SRTM tile discovery/download/mosaic for
    an arbitrary bbox -- see docs/DATA_SOURCES.md #5. A state-sized bbox
    needs 20-40 1x1-degree tiles, above the library's default `max_download_tiles=9`
    safety cap (meant to catch accidental huge/global requests), so it must
    be called through the Python API rather than the `eio` CLI, which has no
    flag to override that cap. The cap is set to exactly the tiles the bbox needs.
    """
    out_path = out_path or paths(config).raw / "terrain/dem.tif"
    if out_path.exists():
        return out_path
    import elevation

    check_dem_tools()
    bbox = config["region"]["bbox"]
    tiles = srtm_tile_count(bbox)
    if tiles > MAX_SRTM_TILES:
        raise ValueError(
            f"bbox needs {tiles} SRTM tiles (limit {MAX_SRTM_TILES}): check region.bbox"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    elevation.clip(
        bounds=(bbox["west"], bbox["south"], bbox["east"], bbox["north"]),
        output=str(out_path.resolve()),
        max_download_tiles=tiles,
    )
    return out_path


def compute_terrain(dem_path: Path, lat: Lattice, px_m: float) -> pd.DataFrame:
    """FR-2.1: per-cell mean elevation/slope and circular-mean aspect.

    The DEM is bilinearly warped onto a lattice-aligned UTM raster (square
    metre pixels, so gradients are true dz/dx, dz/dy). Raster row index
    increases *southward*, so the row-axis gradient is negated to get a
    north-positive dz/dy -- getting this backwards silently flips every
    aspect without producing any error.

    Aspect is an angle: 359 and 1 average to 0, not 180. It is aggregated as
    the mean of its unit vector (northness = cos, eastness = sin), which are
    also the model-usable encodings; aspect_deg is that mean vector's bearing.
    """
    transform, shape = lat.raster(px_m)
    elev = np.full(shape, np.nan, dtype="float32")
    with rasterio.open(dem_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=elev,
            src_nodata=src.nodata if src.nodata is not None else SRTM_VOID,
            dst_transform=transform,
            dst_crs=lat.crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    elev[elev <= SRTM_VOID] = np.nan

    d_row, gx = np.gradient(elev, px_m)
    gy = -d_row
    del d_row
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
    downslope = np.arctan2(-gx, -gy)  # bearing clockwise from north
    del gx, gy

    northness = lat.per_cell(np.cos(downslope))
    eastness = lat.per_cell(np.sin(downslope))
    return pd.DataFrame(
        {
            "cell_id": np.arange(lat.n_rows * lat.n_cols, dtype="int32"),
            "elevation_m": lat.per_cell(elev),
            "slope_deg": lat.per_cell(slope_deg),
            "aspect_deg": np.degrees(np.arctan2(eastness, northness)) % 360,
            "northness": northness,
            "eastness": eastness,
        }
    ).astype({c: "float32" for c in ("elevation_m", "slope_deg", "aspect_deg")})


# --- FR-2.2 land cover -----------------------------------------------------------------


def _worldcover_tiles(bbox: dict) -> list[str]:
    step = WORLDCOVER_TILE_DEG
    lats = range(math.floor(bbox["south"] / step) * step, math.ceil(bbox["north"]), step)
    lons = range(math.floor(bbox["west"] / step) * step, math.ceil(bbox["east"]), step)
    return [f"N{la:02d}E{lo:03d}" for la in lats for lo in lons]


def warp_worldcover(config: dict, lat: Lattice, out_path: Path | None = None) -> Path:
    """WorldCover v200 (2021) codes on a lattice-aligned raster in the region
    CRS, read from the public COGs' overviews (no global tile download). Kept
    on disk: panel.py needs it to check what land cover each fire detection
    sits on."""
    out_path = out_path or paths(config).processed / "landcover.tif"
    if out_path.exists():
        return out_path
    wc = config["static_layers"]["worldcover"]
    px_m = wc["pixel_m"]
    transform, shape = lat.raster(px_m)
    codes = np.zeros(shape, dtype="uint8")
    for tile in _worldcover_tiles(config["region"]["bbox"]):
        url = "/vsicurl/" + wc["url_template"].format(tile=tile)
        # Overview whose pixel (10 m * 2**(level+1)) is closest to px_m without exceeding it.
        level = max(0, int(math.log2(px_m / 10)) - 1)
        print(f"worldcover {tile} (overview level {level})", flush=True)
        with rasterio.open(url, overview_level=level) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=codes,
                src_nodata=WORLDCOVER_NODATA,
                dst_transform=transform,
                dst_crs=lat.crs,
                dst_nodata=WORLDCOVER_NODATA,
                init_dest_nodata=False,  # later tiles must not blank earlier ones
                resampling=Resampling.nearest,
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path, "w", driver="GTiff", height=shape[0], width=shape[1], count=1,
        dtype="uint8", crs=lat.crs, transform=transform, nodata=WORLDCOVER_NODATA,
        compress="deflate", tiled=True,
    ) as dst:
        dst.write(codes, 1)
    return out_path


def compute_landcover(landcover_path: Path, lat: Lattice, classes: dict) -> pd.DataFrame:
    """FR-2.2: per-cell area fraction of each fuel-relevant class (over pixels
    with data), plus the area-weighted majority class."""
    with rasterio.open(landcover_path) as src:
        codes = src.read(1)
    valid = codes != WORLDCOVER_NODATA
    n_valid = lat.per_cell(valid.astype("float32"), np.sum)
    fracs = {}
    for name, members in classes.items():
        hits = lat.per_cell((np.isin(codes, members) & valid).astype("float32"), np.sum)
        with np.errstate(invalid="ignore", divide="ignore"):
            fracs[f"frac_{name}"] = (hits / n_valid).astype("float32")
    df = pd.DataFrame({"cell_id": np.arange(lat.n_rows * lat.n_cols, dtype="int32"), **fracs})
    frac_cols = list(fracs)
    df["landcover_class"] = pd.Categorical(
        df[frac_cols].idxmax(axis=1).str.removeprefix("frac_"), categories=list(classes)
    )
    return df


# --- FR-2.3 human proximity + district boundaries (OSM) ---------------------------------


def download_osm(config: dict, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or paths(config).raw / "osm"
    url = config["static_layers"]["osm"]["pbf_url"]
    out = out_dir / url.rsplit("/", 1)[1]
    if out.exists():
        return out
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(1 << 20):
                f.write(chunk)
    tmp.rename(out)
    return out


def _in_list(field: str, values: list[str]) -> str:
    return f"{field} IN ({', '.join(repr(v) for v in values)})"


def extract_osm(config: dict, pbf: Path, out_dir: Path | None = None) -> dict:
    """State + district polygons, roads, settlements from the zone extract,
    clipped to the bbox and cached as GeoPackages in the region CRS (the pbf
    scan is the slow part; everything downstream reads the cache)."""
    out_dir = out_dir or paths(config).raw / "osm"
    crs = region_crs(config)
    osm = config["static_layers"]["osm"]
    b = config["region"]["bbox"]
    bbox = (b["west"], b["south"], b["east"], b["north"])
    queries = {
        "state": ("multipolygons", f"boundary = 'administrative' AND admin_level = '4' "
                                   f"AND name = '{osm['state_name']}'"),
        "districts": ("multipolygons", f"boundary = 'administrative' AND admin_level = "
                                       f"'{osm['district_admin_level']}'"),
        "roads": ("lines", _in_list("highway", osm["road_highway_values"])),
        "settlement_points": ("points", _in_list("place", osm["settlement_place_values"])),
        "settlement_areas": ("multipolygons", _in_list("place", osm["settlement_place_values"])),
    }
    layers = {}
    for name, (layer, where) in queries.items():
        path = out_dir / f"osm_{name}.gpkg"
        if not path.exists():
            print(f"osm: extracting {name} from {pbf.name}", flush=True)
            gdf = pyogrio.read_dataframe(pbf, layer=layer, where=where, bbox=bbox)
            gdf.to_crs(crs).to_file(path, driver="GPKG")
        layers[name] = gpd.read_file(path)
    if layers["state"].empty:
        raise RuntimeError(f"No admin_level=4 boundary named {osm['state_name']!r} in {pbf}")
    return layers


def compute_proximity(cells: gpd.GeoDataFrame, roads: gpd.GeoDataFrame,
                      settlements: gpd.GeoDataFrame) -> pd.DataFrame:
    """FR-2.3: exact centroid-to-nearest-geometry distance (m) via the
    spatial index -- distance to the road *line*, not to its nearest vertex."""
    centroids = cells.geometry.centroid
    out = {"cell_id": cells["cell_id"].to_numpy()}
    for col, layer in (("dist_road_m", roads), ("dist_settlement_m", settlements)):
        (src_idx, _), dist = layer.sindex.nearest(centroids, return_distance=True, return_all=False)
        values = np.empty(len(cells), dtype="float32")
        values[src_idx] = dist
        out[col] = values
    return pd.DataFrame(out)


def _normalise_district(name: str) -> str:
    return name.removesuffix(" District").removesuffix(" district").strip()


def assign_districts(cells: gpd.GeoDataFrame, districts: gpd.GeoDataFrame,
                     state: gpd.GeoDataFrame) -> pd.Series:
    """District of each cell centroid; a border cell whose centroid falls
    just outside the state takes the nearest in-state district."""
    in_state = districts[districts.representative_point().within(state.union_all())]
    centroids = gpd.GeoDataFrame(geometry=cells.geometry.centroid, crs=cells.crs)
    joined = gpd.sjoin_nearest(centroids, in_state[["name", "geometry"]], how="left")
    joined = joined[~joined.index.duplicated()]
    return joined["name"].map(_normalise_district).rename("district")


def join_calendar(static: pd.DataFrame, calendar_csv: str) -> pd.DataFrame:
    """FR-2.4 join. Every district in the grid must have a non-null season
    window (EXECUTION_PLAN Phase 1 done-condition) -- fail loudly otherwise."""
    cal = pd.read_csv(calendar_csv)
    cal["district"] = cal["district"].map(_normalise_district)
    out = static.merge(cal.drop(columns="state_id"), on="district", how="left")
    missing = sorted(out.loc[out["season_start_doy"].isna(), "district"].astype(str).unique())
    if missing:
        raise ValueError(f"Districts with no fire-calendar row: {missing}. Update {calendar_csv}.")
    return out.astype({"season_start_doy": "int16", "season_end_doy": "int16"})


# --- Assembly ---------------------------------------------------------------------------


def build_static_layers(config: dict, out_dir: Path | None = None) -> Path:
    """FR-2.1-2.5 end to end -> static_layers.parquet (one row per in-state
    cell, no date axis) + grid.geoparquet (cell polygons in WGS84, for
    serving and debug maps)."""
    sl = config["static_layers"]
    out_dir = out_dir or paths(config).processed
    lat = lattice(config)

    osm = extract_osm(config, download_osm(config))
    grid = clip_grid_to_region(build_grid(config), osm["state"])
    print(f"grid: {len(grid)} cells overlap {sl['osm']['state_name']}", flush=True)

    terrain = compute_terrain(download_dem(config), lat, sl["terrain"]["pixel_m"])
    landcover = compute_landcover(warp_worldcover(config, lat), lat, sl["worldcover"]["classes"])
    settlements = pd.concat(
        [osm["settlement_points"], osm["settlement_areas"].assign(geometry=lambda d: d.centroid)],
        ignore_index=True,
    )
    proximity = compute_proximity(grid, osm["roads"], settlements)

    static = (
        pd.DataFrame(grid.drop(columns="geometry"))
        .assign(district=assign_districts(grid, osm["districts"], osm["state"]).to_numpy())
        .merge(terrain, on="cell_id", how="left")
        .merge(landcover, on="cell_id", how="left")
        .merge(proximity, on="cell_id", how="left")
    )
    static = join_calendar(static, config["season"]["calendar_csv"])
    static["state_id"] = static["state_id"].astype("category")

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "static_layers.parquet"
    static.to_parquet(out, index=False)
    grid[["cell_id", "geometry"]].to_crs(WGS84_CRS).to_parquet(out_dir / "grid.geoparquet")
    print(f"wrote {out}: {len(static)} cells, {static.shape[1]} columns", flush=True)
    return out


if __name__ == "__main__":
    build_static_layers(cli_config("FR-2.1-2.5: build all static per-cell layers"))
