"""Tests for the repeated-ideas detection helper."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit_app


def _raw_frame() -> pd.DataFrame:
    """Small full-schema frame with one repeated idea and one single idea."""
    return pd.DataFrame(
        {
            "year": [2024, 2024, 2024, 2025, 2025],
            "cohort_id": [
                "2024-1",
                "2024-2",
                "2024-2",
                "2025-1",
                "2025-2",
            ],
            "cohort": [
                "English",
                "Arabic",
                "Arabic",
                "English",
                "Arabic",
            ],
            "project_name": [
                "Repeat Idea",
                "Repeat Idea",
                "Single Idea",
                "Repeat Idea",
                "Repeat Idea",
            ],
            "date_of_birth": [
                "1990-01-01",
                "1990-01-01",
                "1991-02-02",
                "1989-03-03",
                "1989-03-03",
            ],
            "Sector": [
                "Technology & IT",
                "Technology & IT",
                "Food & Beverage",
                "Food & Beverage",
                "Food & Beverage",
            ],
            "outcome_clean": [
                "Rejected",
                "Accepted",
                "Rejected",
                "Rejected",
                "Rejected",
            ],
            "Business Stage": ["Idea", "MVP", "Idea", "Idea", "Idea"],
            "individual_or_team": [
                "Individual",
                "Individual",
                "Team",
                "Individual",
                "Individual",
            ],
            "problem_en": [
                "First problem",
                "Second problem",
                "Problem",
                "2025 problem",
                "2025 second problem",
            ],
            "solution_en": [
                "First solution",
                "Second solution",
                "Solution",
                "2025 solution",
                "2025 second solution",
            ],
        }
    )


def test_two_cohort_ideas_one_row_per_repeated_identity():
    ideas = streamlit_app._two_cohort_ideas(_raw_frame())
    assert len(ideas) == 2
    repeated = ideas[ideas["year"] == 2024].iloc[0]
    assert repeated["project_name"] == "Repeat Idea"
    assert repeated["year"] == 2024
    assert repeated["cohort_1"] == "2024-1"
    assert repeated["cohort_2"] == "2024-2"
    assert repeated["outcome_1"] == "Rejected"
    assert repeated["outcome_2"] == "Accepted"
    assert repeated["sector_1"] == "Technology & IT"
    assert repeated["applicant_type"] == "Individual"
    assert "Second problem" in repeated["idea_text"]
    assert "Second solution" in repeated["idea_text"]
    assert ideas["year"].tolist() == [2024, 2025]


def test_two_cohort_ideas_requires_two_cohorts_in_same_year():
    raw = pd.DataFrame(
        {
            "year": [2024, 2024, 2023],
            "cohort_id": ["2024-1", "2024-1", "2023-1"],
            "project_name": ["Only Once", "Only Once", "Across Years"],
            "date_of_birth": ["1990-01-01", "1990-01-01", "1990-01-01"],
        }
    )
    assert streamlit_app._two_cohort_ideas(raw).empty


def test_two_cohort_ideas_different_years_do_not_merge():
    raw = pd.DataFrame(
        {
            "year": [2024, 2025],
            "cohort_id": ["2024-1", "2025-1"],
            "project_name": ["Same Name", "Same Name"],
            "date_of_birth": ["1990-01-01", "1990-01-01"],
        }
    )
    assert streamlit_app._two_cohort_ideas(raw).empty


def test_two_cohort_ideas_falls_back_to_project_name_without_dob():
    raw = pd.DataFrame(
        {
            "year": [2024, 2024],
            "cohort_id": ["2024-1", "2024-2"],
            "project_name": ["Name Only", "Name Only"],
        }
    )
    ideas = streamlit_app._two_cohort_ideas(raw)
    assert len(ideas) == 1
    assert ideas.iloc[0]["project_name"] == "Name Only"


def test_two_cohort_ideas_respects_years_sectors_outcomes_types():
    ideas = streamlit_app._two_cohort_ideas(
        _raw_frame(),
        years=[2024],
        sectors=["Technology & IT"],
    )
    assert len(ideas) == 1
    assert ideas.iloc[0]["project_name"] == "Repeat Idea"
    assert ideas.iloc[0]["year"] == 2024
    by_outcome = streamlit_app._two_cohort_ideas(
        _raw_frame(), years=[2024], outcomes=["Accepted"]
    )
    assert len(by_outcome) == 1
    assert by_outcome.iloc[0]["outcome_2"] == "Accepted"
    by_type = streamlit_app._two_cohort_ideas(
        _raw_frame(), years=[2024], types=["Individual"]
    )
    assert len(by_type) == 1
    assert by_type.iloc[0]["applicant_type"] == "Individual"
    assert streamlit_app._two_cohort_ideas(
        _raw_frame(), years=[2024], types=["Team"]
    ).empty
    assert streamlit_app._two_cohort_ideas(
        _raw_frame(), years=[2024], sectors=["Food & Beverage"]
    ).empty


def test_two_cohort_ideas_missing_required_columns_returns_empty():
    assert streamlit_app._two_cohort_ideas(
        pd.DataFrame({"project_name": ["x"]})
    ).empty
