"""Temporal holdout evaluation.

FR-5.1-5.4: train on N-1 fire seasons, test on the most recent; PR-AUC and
precision@top-k (k from config/uttarakhand.yaml's evaluation.precision_at_k)
as headline metrics, base rate reported alongside every number, persistence
baseline always shown next to every later stage. No threshold-0.5
precision/recall/F1 as a headline result.
"""
