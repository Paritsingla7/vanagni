"""FIRMS VIIRS 375 m active-fire ingestion for Uttarakhand.

FR-1.1: bulk-download detections for the configured bbox + date window,
writing a manifest (query params, row count, download date) alongside the
raw file so a re-run skips work already done.
"""
