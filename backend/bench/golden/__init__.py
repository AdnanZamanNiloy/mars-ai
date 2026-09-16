"""Versioned golden benchmark: query sets, expectations and thresholds.

This package holds the DATA for the deterministic offline evaluator
(`bench/eval_offline.py`). It is deliberately data-only so a future golden
version (v2, v3...) is a new JSON file plus a threshold file, not a new
evaluator. `bench.golden.loader` validates and loads a version pair.

Layout
------
    queries_v1.json     the query list + per-query OFFLINE expectations
    expected_v1.json    the expectation schema / legend (machine-readable)
    thresholds_v1.json  versioned aggregate + per-category pass/fail floors

Only deterministic, offline-checkable expectations live here — never
model-specific prose. See `queries_v1.json` header for the field contract.
"""
