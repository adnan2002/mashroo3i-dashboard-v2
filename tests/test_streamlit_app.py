"""Tests for the Streamlit two-file loading and attendance join logic.

Run directly with ``python tests/test_streamlit_app.py`` or via pytest.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit_app

FIXTURES = ROOT / "tests" / "fixtures"
APPLICATIONS_CSV = FIXTURES / "applications.csv"
ATTENDANCE_CSV = FIXTURES / "attendance.csv"


def _load_pair():
    apps = streamlit_app.load_applications(
        APPLICATIONS_CSV.read_bytes(), APPLICATIONS_CSV.name
    )
    attendance = streamlit_app.load_attendance(
        ATTENDANCE_CSV.read_bytes(), ATTENDANCE_CSV.name
    )
    return apps, attendance


def test_loaders_read_fixture_files():
    apps, attendance = _load_pair()
    assert len(apps) == 14
    assert "project_name" in apps.columns
    assert all(column in apps.columns for column in streamlit_app.APPLICATION_COLUMNS)
    assert len(attendance) == 5
    assert all(
        column in attendance.columns
        for column in streamlit_app.dash_app.ATTENDANCE_COLUMNS
    )
    assert "matched_project_name" in attendance.columns


def test_join_links_matched_attendance_to_applications():
    apps, attendance = _load_pair()
    joined = streamlit_app.join_attendance(attendance, apps)
    assert len(joined) == 5
    expected_matched = int(attendance["matched_project_name"].notna().sum())
    assert joined["Sector"].notna().sum() == expected_matched
    assert joined["cohort_id"].notna().sum() == expected_matched
    unmatched = joined[joined["matched_project_name"].isna()]
    assert unmatched["Sector"].isna().all()
    assert unmatched["outcome_clean"].isna().all()
    assert unmatched["cohort_id"].isna().all()
    # No fan-out: matched rows are not duplicated by the join.
    assert len(joined) == len(attendance)


def test_attendance_page_renders_with_joined_data():
    apps, attendance = _load_pair()
    joined = streamlit_app.join_attendance(attendance, apps)
    rendered = streamlit_app.dash_app.update_page(
        "page5", None, None, None, None, None, None, df_input=joined
    )
    assert rendered is not None and len(rendered) > 0


def test_filters_drop_unmatched_attendance_when_outcome_selected():
    apps, attendance = _load_pair()
    joined = streamlit_app.join_attendance(attendance, apps)
    filtered = streamlit_app._apply_filters(
        joined, None, None, ["Accepted"], None, None
    )
    assert filtered["Sector"].notna().all()
    assert len(filtered) == int(joined["outcome_clean"].eq("Accepted").sum())


def test_attendance_page_available_only_when_loaded():
    assert "Attendance" not in streamlit_app.available_pages(False)
    assert "Attendance" in streamlit_app.available_pages(True)


def test_invalid_attendance_file_raises_value_error():
    apps, _ = _load_pair()
    csv_bytes = apps.to_csv(index=False).encode("utf-8-sig")
    try:
        streamlit_app.load_attendance(csv_bytes, "applications.csv")
    except ValueError:
        return
    raise AssertionError("expected ValueError for a file without attendance columns")


def test_applicant_type_derived_from_individual_or_team():
    apps, _ = _load_pair()
    real_like = apps.drop(columns=["applicant_type"])
    csv_bytes = real_like.to_csv(index=False).encode("utf-8-sig")
    loaded = streamlit_app.load_applications(csv_bytes, "real_like.csv")
    assert "applicant_type" in loaded.columns
    assert loaded["applicant_type"].eq(loaded["individual_or_team"]).all()


def test_selection_raw_filter_maps_type_and_skips_missing_columns():
    raw = pd.DataFrame(
        {
            "year": [2024, 2024, 2025, 2025],
            "cohort": ["English", "Arabic", "English", "Arabic"],
            "outcome_clean": ["Accepted", "Rejected", "Accepted", "Accepted"],
            "Sector": ["Technology & IT", "Food & Beverage", "Technology & IT", "Education"],
            "individual_or_team": ["Team", "Individual", "Team", "Individual"],
        }
    )
    filtered = streamlit_app._filter_raw_for_selection(
        raw, years=[2024], sectors=["Technology & IT"], types=["Team"]
    )
    assert len(filtered) == 1
    assert filtered.iloc[0]["year"] == 2024
    assert filtered.iloc[0]["individual_or_team"] == "Team"
    assert len(streamlit_app._filter_raw_for_selection(raw)) == 4
    assert len(streamlit_app._filter_raw_for_selection(raw, outcomes=["Maybe"])) == 0
    # Missing filter columns are skipped rather than erroring.
    no_sector = raw.drop(columns=["Sector"])
    assert len(streamlit_app._filter_raw_for_selection(no_sector, sectors=["X"])) == 4


def test_selection_filter_key_changes_with_selections():
    empty = streamlit_app._selection_filter_key([], [], [], [], [])
    assert empty == streamlit_app._selection_filter_key([], [], [], [], [])
    assert empty != streamlit_app._selection_filter_key([2024], [], [], [], [])


def test_attendance_cohort_chart_uses_cohort_id_and_year_is_ascending():
    apps, attendance = _load_pair()
    joined = streamlit_app.join_attendance(attendance, apps)
    rendered = streamlit_app.dash_app.update_page(
        "page5", None, None, None, None, None, None, df_input=joined
    )
    rows = streamlit_app._render_rows(rendered)
    assert rows is not None
    cards = streamlit_app._extract_cards(rows[0])

    cohort_fig = cards[0][1]
    cohort_labels = [str(label) for label in cohort_fig.data[0].x]
    assert all(label.startswith("20") for label in cohort_labels)
    assert not any(label in ("Arabic", "English") for label in cohort_labels)
    assert cohort_labels == sorted(cohort_labels)

    year_fig = cards[1][1]
    years = [int(label) for label in year_fig.data[0].x]
    assert years == sorted(years)


def test_accepted_applications_over_years_uses_unique_years_and_no_y_grid():
    apps, _ = _load_pair()
    rendered = streamlit_app.dash_app.update_page(
        "page1", None, None, None, None, None, None, df_input=apps
    )
    rows = streamlit_app._render_rows(rendered)
    assert rows is not None
    cards = streamlit_app._extract_cards(rows[0])
    fig = next(
        fig
        for title, fig, *_ in cards
        if "Accepted Applications Over Years" in title
    )
    xaxis = fig.layout.xaxis
    assert xaxis.type == "category"
    assert list(xaxis.categoryarray) == [2023, 2024, 2025]
    yaxis = fig.layout.yaxis
    assert yaxis.showgrid is False
    assert yaxis.visible is False


def test_applications_by_year_and_cohort_shows_totals():
    apps, _ = _load_pair()
    rendered = streamlit_app.dash_app.update_page(
        "page1", None, None, None, None, None, None, df_input=apps
    )
    rows = streamlit_app._render_rows(rendered)
    assert rows is not None
    cards = streamlit_app._extract_cards(rows[0])
    fig = next(
        fig
        for title, fig, *_ in cards
        if "Applications by Year & Cohort" in title
    )
    assert fig.layout.barmode == "group"
    assert not fig.layout.annotations
    assert fig.layout.title.text is None
    expected_totals = [
        f"{year}: {len(apps[apps['year'] == year])}"
        for year in sorted(apps["year"].unique())
    ]
    totals_trace = next(
        trace
        for trace in fig.data
        if getattr(trace, "mode", None) == "text" and trace.showlegend is False
    )
    assert list(totals_trace.text) == expected_totals
    assert len(set(totals_trace.y)) == 1
    assert [int(value) for value in totals_trace.x] == [
        int(year) for year in sorted(apps["year"].unique())
    ]


def test_cards_expose_missing_note():
    apps, _ = _load_pair()
    rendered = streamlit_app.dash_app.update_page(
        "page2", None, None, None, None, None, None, df_input=apps
    )
    rows = streamlit_app._render_rows(rendered)
    assert rows is not None
    cards = streamlit_app._extract_cards(rows[0])
    assert len(cards[0]) == 3
    assert "not specified" in cards[0][1].layout.annotations[0].text


def test_education_chart_is_vertical_with_cleaned_labels():
    apps, _ = _load_pair()
    rendered = streamlit_app.dash_app.update_page(
        "page2", None, None, None, None, None, None, df_input=apps
    )
    rows = streamlit_app._render_rows(rendered)
    cards = []
    for row in rows:
        cards.extend(streamlit_app._extract_cards(row))
    figure = next(fig for title, fig, *_ in cards if title == "Education")
    assert figure.data[0].orientation == "h"
    labels = [str(label) for label in figure.data[0].y]
    assert "Higher Education" in labels
    assert not any("/" in label for label in labels)
    values = [int(value) for value in figure.data[0].x]
    assert values == sorted(values)
    streamlit_app._auto_vertical_axis_labels(figure)
    assert figure.layout.xaxis.tickangle != 90
    assert figure.layout.annotations
    assert figure.layout.annotations[0].y == 0.0
    assert figure.layout.annotations[0].yanchor == "top"
    assert figure.layout.margin.b >= 40


def test_vertical_chart_missing_note_sits_at_top():
    apps, _ = _load_pair()
    rendered = streamlit_app.dash_app.update_page(
        "page2", None, None, None, None, None, None, df_input=apps
    )
    rows = streamlit_app._render_rows(rendered)
    cards = []
    for row in rows:
        cards.extend(streamlit_app._extract_cards(row))
    figure = next(fig for title, fig, *_ in cards if title == "Age Group")
    annotations = figure.layout.annotations
    assert annotations and annotations[0].yref == "paper"
    assert annotations[0].y == 1.0
    assert annotations[0].yanchor == "top"


def test_major_chart_horizontal_original_sort_note_at_top():
    apps, _ = _load_pair()
    rendered = streamlit_app.dash_app.update_page(
        "page2", None, None, None, None, None, None, df_input=apps
    )
    rows = streamlit_app._render_rows(rendered)
    cards = []
    for row in rows:
        cards.extend(streamlit_app._extract_cards(row))
    figure = next(fig for title, fig, *_ in cards if title == "Major")
    assert figure.data[0].orientation == "h"
    values = [int(value) for value in figure.data[0].x]
    assert values == sorted(values)
    assert figure.layout.annotations
    assert figure.layout.annotations[0].y == 0.0
    assert figure.layout.annotations[0].yanchor == "top"
    assert figure.layout.margin.b >= 40


def main():
    tests = [
        test_loaders_read_fixture_files,
        test_join_links_matched_attendance_to_applications,
        test_attendance_page_renders_with_joined_data,
        test_filters_drop_unmatched_attendance_when_outcome_selected,
        test_attendance_page_available_only_when_loaded,
        test_invalid_attendance_file_raises_value_error,
        test_applicant_type_derived_from_individual_or_team,
        test_selection_raw_filter_maps_type_and_skips_missing_columns,
        test_selection_filter_key_changes_with_selections,
        test_attendance_cohort_chart_uses_cohort_id_and_year_is_ascending,
        test_accepted_applications_over_years_uses_unique_years_and_no_y_grid,
        test_applications_by_year_and_cohort_shows_totals,
        test_cards_expose_missing_note,
        test_education_chart_is_vertical_with_cleaned_labels,
        test_vertical_chart_missing_note_sits_at_top,
        test_major_chart_horizontal_original_sort_note_at_top,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
