"""ERA5-Land ingestion via cdsapi, resampled to daily per-cell features.

FR-1.2: hourly reanalysis for the configured bbox + date window, resampled
using all 24 hourly steps (not the 5-synoptic-hour subset the old US
pipeline used) and bucketed by local day (UTC+5:30 fixed shift for India),
not UTC day.
"""
