"""Feature-family builders used by panel.py.

Split out from panel.py so train.py's staged ablation (FR-4.1) can
include/exclude a family per stage without duplicating join/window logic:
fire history, weather, human-proximity/fuel/calendar, and the region-
relative seasonality feature (days-since-dry-season-onset, INDIA.md 2.2).
"""
