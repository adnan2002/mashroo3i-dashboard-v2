"""Tests for auto-vertical categorical axis labels on the dashboard charts."""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit_app


def test_crowded_sector_labels_become_vertical():
    figure = go.Figure(
        go.Bar(x=[f"Sector {index}" for index in range(12)], y=[1] * 12)
    )
    streamlit_app._auto_vertical_axis_labels(figure)
    assert figure.layout.xaxis.tickangle == 90
    assert figure.layout.xaxis.automargin is True


def test_sparse_labels_stay_horizontal():
    figure = go.Figure(go.Bar(x=["A", "B", "C"], y=[1, 2, 3]))
    streamlit_app._auto_vertical_axis_labels(figure)
    assert figure.layout.xaxis.tickangle != 90


def test_five_categories_rotate_on_small_screen_budget():
    figure = go.Figure(
        go.Bar(x=["2023-1", "2024-1", "2024-2", "2025-1", "2025-2"], y=[1] * 5)
    )
    streamlit_app._auto_vertical_axis_labels(figure)
    assert figure.layout.xaxis.tickangle == 90


def test_four_short_labels_stay_horizontal():
    figure = go.Figure(go.Bar(x=["A", "B", "C", "D"], y=[1, 2, 3, 4]))
    streamlit_app._auto_vertical_axis_labels(figure)
    assert figure.layout.xaxis.tickangle != 90


def test_six_short_numeric_labels_stay_horizontal():
    figure = go.Figure(go.Bar(x=["1", "2", "3", "4", "5", "5+"], y=[1] * 6))
    streamlit_app._auto_vertical_axis_labels(figure)
    assert figure.layout.xaxis.tickangle != 90


def test_numeric_year_axis_untouched():
    figure = go.Figure(go.Bar(x=[2023, 2024, 2025], y=[1, 2, 3]))
    streamlit_app._auto_vertical_axis_labels(figure)
    assert figure.layout.xaxis.tickangle != 90


def test_single_long_label_becomes_vertical():
    figure = go.Figure(
        go.Bar(
            x=["A very long category label exceeding 14 characters"],
            y=[1],
        )
    )
    streamlit_app._auto_vertical_axis_labels(figure)
    assert figure.layout.xaxis.tickangle == 90


def test_explicit_category_axis_rotates_when_long():
    figure = go.Figure(go.Bar(x=["Short", "Also short"], y=[1, 2]))
    figure.update_xaxes(
        type="category",
        categoryarray=["Short", "Also short"],
        ticktext=["A very long label for a crowded axis", "Equally long label"],
    )
    streamlit_app._auto_vertical_axis_labels(figure)
    assert figure.layout.xaxis.tickangle == 90


def test_opt_out_flag_keeps_labels_horizontal():
    figure = go.Figure(
        go.Bar(x=[f"Sector {index}" for index in range(12)], y=[1] * 12)
    )
    figure._no_auto_vertical_labels = True
    streamlit_app._auto_vertical_axis_labels(figure)
    assert figure.layout.xaxis.tickangle != 90


def main():
    tests = [
        test_crowded_sector_labels_become_vertical,
        test_sparse_labels_stay_horizontal,
        test_five_categories_rotate_on_small_screen_budget,
        test_four_short_labels_stay_horizontal,
        test_six_short_numeric_labels_stay_horizontal,
        test_numeric_year_axis_untouched,
        test_single_long_label_becomes_vertical,
        test_explicit_category_axis_rotates_when_long,
        test_opt_out_flag_keeps_labels_horizontal,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
