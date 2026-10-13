"""Lattice orientation + terrain sign conventions: the two places a static
layer can be silently wrong (a flipped row order or a negated gradient
produces plausible-looking numbers and no error)."""
import numpy as np
import rasterio

from wfpin.static_layers import UTM_CRS, Lattice, build_grid, compute_terrain

LAT = Lattice(west_m=300_000, south_m=3_200_000, res_m=5000, n_cols=3, n_rows=2)
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
                       count=1, dtype="float32", crs=UTM_CRS, transform=transform) as dst:
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
