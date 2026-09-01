"""Unit tests for the similar-idea helpers in similarity.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import similarity


def _synthetic_cache() -> tuple[np.ndarray, pd.DataFrame, dict]:
    matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.99, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.8],
        ],
        dtype=np.float32,
    )
    metadata = pd.DataFrame(
        {
            "project_name": ["A", "A again", "B", "C"],
            "year": ["2024", "2024", "2024", "2025"],
            "cohort_id": ["2024-1", "2024-2", "2024-1", "2025-1"],
            "sector": ["Tech", "Tech", "Food", "Tech"],
            "text": [
                "Project: A\nProblem: problem one\nSolution: solution one",
                "Project: A again\nProblem: problem one again\n"
                "Solution: solution one again",
                "Project: B\nProblem: problem two\nSolution: solution two",
                "Project: C\nProblem: problem three\nSolution: solution three",
            ],
        }
    )
    index = {"model": "voyage-4-large", "dimensions": 4, "input_type": "document"}
    return matrix, metadata, index


def _duplicate_fixture() -> tuple[np.ndarray, pd.DataFrame, dict]:
    """Index with exact, same-name, and near-identical duplicate rows."""
    matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.99, 0.1, 0.0, 0.0],
            [0.88, 0.35, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    metadata = pd.DataFrame(
        {
            "project_name": ["Alpha", "Alpha", "ALPHA", "Beta", "Gamma"],
            "year": ["2024", "2024", "2024", "2024", "2024"],
            "cohort_id": ["2024-1", "2024-1", "2024-1", "2024-2", "2024-2"],
            "sector": ["Tech", "Tech", "Tech", "Tech", "Other"],
            "text": [
                "Project: Alpha\nProblem: shared problem\n"
                "Solution: solution one",
                "Project: Alpha\nProblem: shared problem\n"
                "Solution: solution one",
                "Project: ALPHA\nProblem: shared problem with extra words\n"
                "Solution: solution one with extra words",
                "Project: Beta\nProblem: shared problem\n"
                "Solution: solution two",
                "Project: Gamma\nProblem: other problem\n"
                "Solution: other solution",
            ],
        }
    )
    index = {"model": "voyage-4-large", "dimensions": 4, "input_type": "document"}
    return matrix, metadata, index


def test_similarity_level_boundaries():
    assert similarity.similarity_level(0.99) == "Nearly identical"
    assert similarity.similarity_level(0.93) == "Very similar"
    assert similarity.similarity_level(0.88) == "Similar"
    assert similarity.similarity_level(0.82) == "Somewhat similar"
    assert similarity.similarity_level(0.75) == "Not similar"


def test_query_similarity_level_boundaries():
    assert similarity.query_similarity_level(0.70) == "Very similar"
    assert similarity.query_similarity_level(0.60) == "Similar"
    assert similarity.query_similarity_level(0.52) == "Somewhat similar"
    assert similarity.query_similarity_level(0.45) == "Not similar"


def test_strictness_thresholds_are_ordered():
    assert similarity.STRICTNESS_THRESHOLDS["Very similar"] > (
        similarity.STRICTNESS_THRESHOLDS["Similar"]
    )
    assert similarity.STRICTNESS_THRESHOLDS["Similar"] > (
        similarity.STRICTNESS_THRESHOLDS["Somewhat similar"]
    )


def test_document_sections_parses_problem_and_solution():
    text = (
        "Project: ilmTech\n"
        "Problem:  multi-line  problem collapsed to one line\n"
        "Solution: the solution\n"
        "Differentiation: something different"
    )
    sections = similarity.document_sections(text)
    assert sections["Project"] == "ilmTech"
    assert sections["Problem"] == "multi-line problem collapsed to one line"
    assert sections["Solution"] == "the solution"
    assert sections["Differentiation"] == "something different"


def test_top_similar_pairs_includes_identical_and_includes_sections(monkeypatch):
    monkeypatch.setattr(similarity, "load_index", _duplicate_fixture)
    pairs = similarity.top_similar_pairs("Very similar")
    assert len(pairs) == 6
    top = pairs.iloc[0]
    assert top["left_project"] == "Alpha"
    assert top["right_project"] == "Alpha"
    assert top["level"] == "Nearly identical"
    assert set(pairs["level"]) == {"Nearly identical", "Very similar"}
    similar = pairs[pairs["level"] == "Very similar"].iloc[0]
    assert similar["left_project"] == "ALPHA"
    assert similar["right_project"] == "Beta"
    assert similar["left_problem"] == "shared problem with extra words"
    assert similar["right_problem"] == "shared problem"
    assert similar["left_solution"] == "solution one with extra words"
    assert similar["right_solution"] == "solution two"


def test_relation_band_boundaries():
    assert similarity.relation_band(0.97) == "Very similar (likely same idea)"
    assert similarity.relation_band(0.92) == "Very similar (likely same idea)"
    assert similarity.relation_band(0.89) == (
        "Similar (different idea, same concept)"
    )
    assert similarity.relation_band(0.87) == (
        "Similar (different idea, same concept)"
    )


def test_similar_clusters_groups_related_ideas_once(monkeypatch):
    monkeypatch.setattr(similarity, "load_index", _duplicate_fixture)
    clusters, edges = similarity.similar_clusters()
    assert set(clusters["cluster_id"]) == {0}
    assert set(clusters["project_name"]) == {"ALPHA", "Beta"}
    assert len(edges) == 1
    edge = edges.iloc[0]
    assert edge["band"] == "Very similar (likely same idea)"
    assert 0.87 <= edge["similarity"] < 0.97
    assert sorted([edge["left_row_index"], edge["right_row_index"]]) == [0, 1]


def test_similar_clusters_respects_threshold(monkeypatch):
    monkeypatch.setattr(similarity, "load_index", _duplicate_fixture)
    clusters, edges = similarity.similar_clusters(threshold=0.97)
    assert clusters.empty
    assert edges.empty
    clusters, edges = similarity.similar_clusters(threshold=0.0)
    assert set(clusters["project_name"]) == {"ALPHA", "Beta", "Gamma"}
    assert len(edges) == 3


def test_similar_clusters_caps_group_members(monkeypatch):
    monkeypatch.setattr(similarity, "load_index", _duplicate_fixture)
    clusters, edges = similarity.similar_clusters(
        threshold=0.0,
        max_group_size=2,
    )
    assert set(clusters["project_name"]) == {"ALPHA", "Beta"}
    assert len(edges) == 1


def test_deduplicate_matrix_collapses_exact_same_name_and_near_identical():
    matrix, metadata, _ = _duplicate_fixture()
    dedup_matrix, dedup_metadata = similarity.deduplicate_matrix(matrix, metadata)
    assert len(dedup_metadata) == 3
    assert set(dedup_metadata["project_name"]) == {"ALPHA", "Beta", "Gamma"}
    assert len(dedup_matrix) == 3


def test_names_share_identity_uses_whole_words():
    assert similarity.names_share_identity("NOSH Cafe", "Nosh")
    assert similarity.names_share_identity("Daribny / دربني", "Daribny")
    assert not similarity.names_share_identity(
        "Indoor Football Pitch", "Dome Indoor Football Pitches"
    )
    assert not similarity.names_share_identity(
        "Pearls of Bahrain", "Pearls bay"
    )


def test_rank_matches_orders_by_similarity_and_applies_min_score():
    matrix, metadata, _ = _synthetic_cache()
    query = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    matches = similarity._rank_matches(matrix, metadata, query, top_k=3, min_score=0.5)
    assert [match["project_name"] for match in matches] == ["A", "A again"]
    assert matches[0]["level"] == "Very similar"
    assert matches[0]["problem"] == "problem one"
    assert matches[0]["solution"] == "solution one"

    strict = similarity._rank_matches(matrix, metadata, query, top_k=3, min_score=1.0)
    assert [match["project_name"] for match in strict] == ["A"]


def test_search_similar_returns_empty_without_api_key(monkeypatch):
    monkeypatch.setattr(similarity, "load_index", _synthetic_cache)
    monkeypatch.setattr(similarity, "_voyage_api_key", lambda: None)
    assert similarity.search_similar("unique query without a key 123") == []


def test_search_similar_embeds_query_as_document(monkeypatch):
    captured: dict = {}

    def _fake_embed(texts, api_key, model, dimensions, input_type, batch_size, token_budget):
        captured.update(
            {
                "texts": texts,
                "api_key": api_key,
                "model": model,
                "dimensions": dimensions,
                "input_type": input_type,
            }
        )
        return [[1.0, 0.0, 0.0, 0.0]]

    monkeypatch.setattr(similarity, "load_index", _synthetic_cache)
    monkeypatch.setattr(similarity, "_voyage_api_key", lambda: "test-key")
    monkeypatch.setattr(similarity.idea_search, "embed_many", _fake_embed)

    matches = similarity.search_similar("another unique query 456", top_k=3)
    assert captured["api_key"] == "test-key"
    assert captured["input_type"] == "document"
    assert captured["texts"] == ["another unique query 456"]
    assert [match["project_name"] for match in matches] == ["A again"]
    assert matches[0]["problem"] == "problem one again"
