"""Static layers, computed once and reused across every cell-day.

FR-2.1 terrain (slope/aspect from SRTM), FR-2.2 land cover (area-weighted
regrid, not an unweighted point sample), FR-2.3 human proximity (road/
settlement distance via cKDTree or sjoin_nearest), FR-2.5 grid definition
(cell_id, centroid_lat/lon, state_id from config/uttarakhand.yaml).

FR-2.4 (the fire calendar table) is deliberately not built here — it is a
hand-compiled CSV/YAML, not code; see docs/EXECUTION_PLAN.md Phase 1.4.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import yaml
from pyproj import CRS, Transformer
from shapely.geometry import box

# Uttarakhand sits entirely in UTM zone 44N; projecting here (rather than a
# raw degree grid) keeps cells genuinely resolution_km x resolution_km --
# the old US pipeline's audited defect was 0.25-degree cells whose true
# area varies with latitude.
UTM_CRS = "EPSG:32644"
WGS84_CRS = "EPSG:4326"


def load_config(path: str = "config/uttarakhand.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_grid(config: dict) -> gpd.GeoDataFrame:
    """FR-2.5: one row per cell -- cell_id, centroid_lat/lon, state_id.

    Grid is defined by snapping the configured bbox (reprojected to UTM
    44N) to a regular resolution_km x resolution_km lattice, so every cell
    has the same true area regardless of its latitude within the state.
    """
    bbox = config["region"]["bbox"]
    state_id = config["region"]["state_id"]
    resolution_m = config["grid"]["resolution_km"] * 1000

    to_utm = Transformer.from_crs(WGS84_CRS, UTM_CRS, always_xy=True)
    to_wgs84 = Transformer.from_crs(UTM_CRS, WGS84_CRS, always_xy=True)

    west_m, south_m = to_utm.transform(bbox["west"], bbox["south"])
    east_m, north_m = to_utm.transform(bbox["east"], bbox["north"])

    x_edges = np.arange(west_m, east_m, resolution_m)
    y_edges = np.arange(south_m, north_m, resolution_m)

    rows = []
    cell_id = 0
    for y0 in y_edges:
        for x0 in x_edges:
            cx_m, cy_m = x0 + resolution_m / 2, y0 + resolution_m / 2
            cx_lon, cy_lat = to_wgs84.transform(cx_m, cy_m)
            rows.append(
                {
                    "cell_id": cell_id,
                    "state_id": state_id,
                    "centroid_lat": cy_lat,
                    "centroid_lon": cx_lon,
                    "geometry": box(x0, y0, x0 + resolution_m, y0 + resolution_m),
                }
            )
            cell_id += 1

    return gpd.GeoDataFrame(rows, crs=CRS.from_user_input(UTM_CRS))
