"""Tests for the self-contained Brinc model service."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import model_service

REAL_CSV = Path.home() / "Desktop/filter_3/dashboard_ready.csv"


def test_bundle_available_with_118_features():
    ok, message = model_service.available()
    assert ok, message
    scorer = model_service.load_scorer()
    assert scorer["n_features"] == 118
    assert len(scorer["num_cols"]) == 104
    assert len(scorer["cat_cols"]) == 14
    assert scorer["model_dir"].endswith("model")


def test_score_with_model_on_real_slice():
    if not REAL_CSV.exists():
        return
    raw = pd.read_csv(REAL_CSV, encoding="utf-8-sig").head(20)
    scored = model_service.score_with_model(raw)
    assert len(scored) == 20
    assert scored["accept_probability"].between(0, 1).all()
    assert scored["model_rank"].tolist() == list(range(1, 21))
    assert scored["prediction_threshold"].iloc[0] == 0.125
    assert "identity" in scored.columns


def test_score_is_deterministic():
    if not REAL_CSV.exists():
        return
    raw = pd.read_csv(REAL_CSV, encoding="utf-8-sig").head(20)
    first = model_service.score_with_model(raw)["accept_probability"].tolist()
    second = model_service.score_with_model(raw)["accept_probability"].tolist()
    assert first == second


def main():
    tests = [
        test_bundle_available_with_118_features,
        test_score_with_model_on_real_slice,
        test_score_is_deterministic,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
