"""Tests for the hybrid Brinc-model + agent evaluator (pure helpers only)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hybrid_evaluate as he


def _frame():
    return pd.DataFrame(
        {
            "year": [2024] * 8,
            "project_name": [f"P{i}" for i in range(8)],
            "date_of_birth": ["1990-01-01"] * 8,
            "problem_en": [f"problem {i}" for i in range(8)],
            "solution_en": [f"solution {i}" for i in range(8)],
            "problem": ["مشكلة"] * 8,
            "solution": ["حل"] * 8,
            "outcome_clean": ["Accepted", "Rejected"] * 4,
            "outcome": ["Yes", "No"] * 4,
            "is_winner": [True, False] * 4,
        }
    )


def test_strip_outcomes_removes_every_label_column():
    frame = _frame()
    stripped = he.strip_outcomes(frame)
    assert not any(column in he.OUTCOME_COLS for column in stripped.columns)
    assert "problem_en" in stripped.columns
    assert "outcome_clean" not in stripped.columns


def test_idea_text_prefers_english_and_falls_back_to_arabic():
    frame = _frame()
    english = he.idea_text_for(frame.loc[0])
    assert "problem 0" in english and "solution 0" in english
    only_arabic = frame.copy()
    only_arabic.loc[0, ["problem_en", "solution_en"]] = None
    text = he.idea_text_for(only_arabic.loc[0])
    assert "مشكلة" in text and "حل" in text


def test_stratified_sample_dedupes_and_balances():
    frame = _frame()
    frame["identity"] = (
        frame["project_name"] + "|" + frame["date_of_birth"]
    )
    labels = (
        frame["outcome_clean"].eq("Accepted").astype(int)
    )
    # Duplicate every identity to prove dedupe.
    duplicated = pd.concat([frame, frame], ignore_index=True)
    duplicated["identity"] = (
        duplicated["project_name"] + "|" + duplicated["date_of_birth"]
    )
    selected = he.stratified_sample(duplicated, labels.reindex(duplicated.index), 2, 7)
    assert len(selected) == 4
    assert len(set(selected)) == 4
    assert int(labels.reindex(selected).sum()) == 2
    assert duplicated.loc[selected, "identity"].is_unique


def test_blend_and_metrics_rank_correctly():
    model = pd.Series([0.9, 0.8, 0.4, 0.1])
    agent = pd.Series([70.0, 60.0, 50.0, 40.0])
    label = pd.Series([1, 1, 0, 0])
    for weight in (0.0, 0.5, 1.0):
        score = he.blend_scores(he.rank_pct(model), he.rank_pct(agent), weight)
        metrics = he.prediction_metrics(score, label)
        assert metrics["auc"] == 1.0
        assert metrics["spearman"] > 0.85


def test_metrics_handle_missing_scores():
    label = pd.Series([1, 0, 1, 0, 0])
    score = pd.Series([0.9, np.nan, 0.2, np.nan, 0.1])
    metrics = he.prediction_metrics(score, label)
    assert metrics["auc"] is not None


def test_model_top_k_picks_highest_probability_identities():
    engineered = pd.DataFrame(
        {
            "identity": ["a", "a", "b", "c"],
            "project_name": ["P1", "P1", "P2", "P3"],
            "year": [2024, 2024, 2025, 2025],
        },
        index=[0, 1, 2, 3],
    )
    labels = pd.Series([0, 1, 1, 0], index=[0, 1, 2, 3])
    oof = np.array([0.4, 0.9, 0.7, 0.1])
    selected = he.model_top_k(engineered, labels, oof, top_k=2)
    # Identity "a" should keep its best row (index 1), then "b" (index 2).
    assert list(selected) == [1, 2]


def test_cascade_metrics_reports_shortlist_and_agent_increment():
    engineered = pd.DataFrame(
        {
            "identity": ["a", "b", "c", "d"],
            "project_name": ["A", "B", "C", "D"],
            "year": [2024, 2024, 2025, 2025],
        },
        index=[0, 1, 2, 3],
    )
    labels = pd.Series([1, 0, 1, 0], index=[0, 1, 2, 3])
    oof = np.array([0.9, 0.6, 0.4, 0.1])
    cache = {
        "a": {"score": 40.0},
        "b": {"score": 60.0},
        "c": {"score": 55.0},
        "d": {"score": 35.0},
    }
    result = he.run_cascade_metrics(engineered, labels, oof, cache, top_k=4, weight=0.6)
    assert result["top_k"] == 4
    assert result["model_precision_at_k"] == 0.5
    assert result["agent_scored_in_top_k"] == 4
    assert result["model_recall_at_k"] == 1.0
    assert "shortlist" in result and len(result["shortlist"]) == 4


def main():
    tests = [
        test_strip_outcomes_removes_every_label_column,
        test_idea_text_prefers_english_and_falls_back_to_arabic,
        test_stratified_sample_dedupes_and_balances,
        test_blend_and_metrics_rank_correctly,
        test_metrics_handle_missing_scores,
        test_model_top_k_picks_highest_probability_identities,
        test_cascade_metrics_reports_shortlist_and_agent_increment,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
