"""Lattice orientation + terrain sign conventions: the two places a static
layer can be silently wrong (a flipped row order or a negated gradient
produces plausible-looking numbers and no error)."""
import numpy as np
import pytest
import rasterio

from wfpin.config import paths, region_crs
from wfpin.static_layers import (
    Lattice,
    build_grid,
    check_dem_tools,
    compute_terrain,
    lattice,
    srtm_tile_count,
)

LAT = Lattice(west_m=300_000, south_m=3_200_000, res_m=5000, n_cols=3, n_rows=2, crs="EPSG:32644")
PX = 500  # 10 px per cell


def test_per_cell_maps_raster_north_to_highest_row():
    _, shape = LAT.raster(PX)
    arr = np.zeros(shape, dtype="float32")
    arr[:10, :10] = 1.0  # top-left block of the raster = north-west cell
    per_cell = LAT.per_cell(arr)
    nw_cell = (LAT.n_rows - 1) * LAT.n_cols + 0
    assert per_cell[nw_cell] == 1.0
    assert per_cell.sum() == 1.0


def test_grid_ids_match_lattice_rows():
    config = {
        "region": {"state_id": "UK", "bbox": {"west": 78.0, "south": 29.0, "east": 78.2,
                                              "north": 29.1}},
        "grid": {"resolution_km": 5},
    }
    grid = build_grid(config)
    n_cols = grid["grid_col"].max() + 1
    assert (grid["cell_id"] == grid["grid_row"] * n_cols + grid["grid_col"]).all()
    # Higher grid_row is further north.
    assert grid.groupby("grid_row")["centroid_lat"].mean().is_monotonic_increasing


def _write_dem(path, elev):
    transform, _ = LAT.raster(PX)
    with rasterio.open(path, "w", driver="GTiff", height=elev.shape[0], width=elev.shape[1],
                       count=1, dtype="float32", crs=LAT.crs, transform=transform) as dst:
        dst.write(elev, 1)


def test_terrain_aspect_faces_downhill(tmp_path):
    _, shape = LAT.raster(PX)
    cols = np.arange(shape[1], dtype="float32")
    east_down = np.tile(1000 - cols * PX * 0.1, (shape[0], 1))  # 10% grade, falling eastward
    _write_dem(tmp_path / "dem.tif", east_down)
    t = compute_terrain(tmp_path / "dem.tif", LAT, PX)
    np.testing.assert_allclose(t["aspect_deg"], 90, atol=0.5)  # faces east
    np.testing.assert_allclose(t["slope_deg"], np.degrees(np.arctan(0.1)), atol=0.1)
    np.testing.assert_allclose(t["eastness"], 1, atol=1e-3)


def test_aspect_is_circular_mean(tmp_path):
    # Left half of each cell faces ~north-west, right half ~north-east: the
    # mean must be north (0 deg), not the arithmetic-mean artefact of 180.
    _, shape = LAT.raster(PX)
    rows, cols = np.mgrid[: shape[0], : shape[1]].astype("float32")
    north_down = 1000 - (shape[0] - rows) * PX * 0.1  # falls northward
    tilt = np.where((cols % 10) < 5, cols, 10 - cols % 10 + (cols // 10) * 10) * PX * 0.01
    _write_dem(tmp_path / "dem.tif", (north_down + tilt).astype("float32"))
    t = compute_terrain(tmp_path / "dem.tif", LAT, PX)
    bearing = (t["aspect_deg"] + 180) % 360 - 180  # wrap to [-180, 180)
    assert (bearing.abs() < 20).all(), t["aspect_deg"].tolist()


# --- region-agnostic setup: a second region must be a config change only -----------------

ODISHA = {
    "region": {"name": "odisha", "state_id": "OD",
               "bbox": {"west": 81.3, "east": 87.6, "south": 17.7, "north": 22.7}},
    "grid": {"resolution_km": 5},
}
ODISHA_ALBERS = {**ODISHA, "region": {**ODISHA["region"], "crs": "ESRI:102028"}}


def test_region_crs_defaults_to_the_utm_zone_of_the_bbox_centre():
    with pytest.warns(UserWarning, match="wider than one UTM zone"):
        assert region_crs(ODISHA) == "EPSG:32645"  # centre ~84.5 E -> zone 45N
    uttarakhand = {"region": {"name": "uk", "bbox": {"west": 77.5, "east": 81.1,
                                                      "south": 28.6, "north": 31.5}}}
    assert region_crs(uttarakhand) == "EPSG:32644"  # fits in one zone: no warning
    southern = {"region": {"name": "s", "bbox": {"west": 30, "east": 31, "south": -2, "north": -1}}}
    assert region_crs(southern) == "EPSG:32736"  # southern hemisphere
    assert region_crs(ODISHA_ALBERS) == "ESRI:102028"  # an explicit setting wins, silently


def test_regions_never_share_data_directories():
    odisha = paths(ODISHA)
    assert odisha.raw.parts[-2:] == ("odisha", "raw")
    assert odisha.processed.parts[-2:] == ("odisha", "processed")
    pinned = paths({"region": {"name": "uttarakhand"}, "paths": {"raw": "data/raw",
                                                                    "processed": "data/processed"}})
    assert str(pinned.raw) == "data/raw"  # an existing layout can be pinned explicitly
    assert odisha.raw != pinned.raw and odisha.processed != pinned.processed


def test_odisha_grid_builds_from_config_alone_with_truly_equal_cell_areas():
    lat = lattice(ODISHA_ALBERS)
    assert lat.crs == "ESRI:102028"
    grid = build_grid(ODISHA_ALBERS)
    assert grid["state_id"].eq("OD").all()
    # Equal-area projection: a square projected cell has the same true area
    # everywhere, unlike a UTM zone far from its central meridian.
    np.testing.assert_allclose(grid.geometry.area, 25_000_000, rtol=1e-9)
    # The lattice is the projected rectangle enclosing the bbox, so it must
    # *cover* the bbox (and may poke slightly past it; the state polygon clips it).
    assert grid["centroid_lat"].min() < 17.8 and grid["centroid_lat"].max() > 22.6
    assert grid["centroid_lon"].min() < 81.4 and grid["centroid_lon"].max() > 87.5


def test_srtm_tile_count_matches_the_tiles_actually_downloaded_for_uttarakhand():
    assert srtm_tile_count({"west": 77.5, "east": 81.1, "south": 28.6, "north": 31.5}) == 20
    assert srtm_tile_count(ODISHA["region"]["bbox"]) == 42  # 81..88 E x 17..23 N


def test_missing_gdal_tools_fail_loudly_instead_of_poisoning_the_dem_cache(monkeypatch):
    monkeypatch.setenv("PATH", "")
    with pytest.raises(RuntimeError, match="empty tiles"):
        check_dem_tools()
