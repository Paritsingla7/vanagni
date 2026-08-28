"""Static layers, computed once and reused across every cell-day.

FR-2.1 terrain (slope/aspect from SRTM), FR-2.2 land cover (area-weighted
regrid, not an unweighted point sample), FR-2.3 human proximity (road/
settlement distance via cKDTree or sjoin_nearest), FR-2.5 grid definition
(cell_id, centroid_lat/lon, state_id from config/uttarakhand.yaml).

FR-2.4 (the fire calendar table) is deliberately not built here — it is a
hand-compiled CSV/YAML, not code; see docs/EXECUTION_PLAN.md Phase 1.4.
"""
