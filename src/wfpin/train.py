"""Staged ablation training entrypoint.

FR-4.1-4.3: `--stage {1,2,3,4}` -- persistence, +fire history, +weather,
+human-proximity/fuel/calendar. Stage 1 has no learned parameters but must
run through the same evaluation path as 2-4 so FR-5.4's comparison is
apples to apples. Every stage serializes a loadable model artifact.
"""
