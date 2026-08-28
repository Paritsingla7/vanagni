"""Dense, leakage-safe cell x day panel construction.

FR-3.1-3.4: every cell, every day in the window, quiet days included as
explicit negatives. Label is fire detected on day d+1; every feature comes
from day d or earlier. Shift before window, never a positional `.rolling()`
after a lossy join, no sentinel values reaching the feature matrix.
"""
