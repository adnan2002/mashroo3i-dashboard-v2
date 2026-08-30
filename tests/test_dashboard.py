"""Smoke tests for the Mashroo3i dashboard (no browser required).

Run directly with ``python tests/test_dashboard.py`` or via pytest.
"""

import base64
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app as dashboard

FIXTURES = ROOT / "tests" / "fixtures"
APPLICATIONS_CSV = FIXTURES / "applications.csv"
ATTENDANCE_CSV = FIXTURES / "attendance.csv"


def _combined_fixture() -> pd.DataFrame:
    """One DataFrame with application + attendance columns (old combined shape)."""
    apps = pd.read_csv(APPLICATIONS_CSV).copy()
    attendance = pd.read_csv(ATTENDANCE_CSV).copy()
    apps["_project_norm"] = apps["project_name"].astype(str).str.strip().str.lower()
    attendance["_project_norm"] = (
        attendance["matched_project_name"].astype(str).str.strip().str.lower()
    )
    combined = apps.merge(
        attendance.drop(columns=["year", "cohort"]),
        on="_project_norm",
        how="left",
    ).drop(columns=["_project_norm"])
    return combined


def _upload_df(df: pd.DataFrame) -> str:
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    return "data:text/csv;base64," + base64.b64encode(csv_bytes).decode()


def _card_figures(rendered):
    """Return (card title, Plotly figure) pairs from a rendered page shell."""
    figures = []
    for row in rendered[2].children:
        for card in row.children:
            title = card.children[0].children
            figure = card.children[1].children[0].figure
            figures.append((title, figure))
    return figures


def test_index_page_serves():
    client = dashboard.server.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"Mashroo3i" in response.data


def test_upload_builds_filter_options():
    (
        status,
        year_opts,
        cohort_opts,
        outcome_opts,
        sector_opts,
        type_opts,
    ) = dashboard.handle_upload(_upload_df(_combined_fixture()), "combined.csv")
    assert status is not None
    assert dashboard.DF_GLOBAL is not None and len(dashboard.DF_GLOBAL) > 0
    assert year_opts and all(isinstance(opt["value"], int) for opt in year_opts)
    assert cohort_opts and all("Arabic" in str(opt["label"]) or "English" in str(opt["label"]) for opt in cohort_opts)
    assert outcome_opts and sector_opts and type_opts
    assert "team_attendance_rate" in dashboard.DF_GLOBAL.columns


def test_all_pages_render_after_upload():
    status, *_ = dashboard.handle_upload(_upload_df(_combined_fixture()), "combined.csv")
    for page in ("page1", "page2", "page3", "page4", "page5"):
        rendered = dashboard.update_page(page, status, None, None, None, None, None)
        assert rendered is not None and len(rendered) > 0, f"{page} produced no content"


def test_filters_produce_content():
    status, *_ = dashboard.handle_upload(_upload_df(_combined_fixture()), "combined.csv")
    rendered = dashboard.update_page(
        "page1",
        status,
        years=[2024],
        cohorts=["Arabic"],
        outcomes=["Accepted"],
        sectors=None,
        types=["Individual"],
    )
    assert rendered is not None and len(rendered) > 0


def test_empty_data_uploads_render_a_prompt():
    dashboard.DF_GLOBAL = None
    rendered = dashboard.update_page("page1", None, None, None, None, None, None)
    assert rendered is not None and len(rendered) == 1


def test_switch_page():
    # `switch_page` reads dash.ctx, which only exists inside a live callback.
    # Simulate that callback context explicitly.
    import dash._callback_context as callback_context
    from dash._utils import AttributeDict

    token = callback_context.context_value.set(
        AttributeDict(
            triggered_inputs=[{"prop_id": "btn-p5.n_clicks", "value": 1}],
        )
    )
    try:
        page, s1, s2, s3, s4, s5, value_style = dashboard.switch_page(
            0, 0, 0, 0, 1
        )
    finally:
        callback_context.context_value.reset(token)

    assert page == "page5"
    assert s5["background"] == dashboard.C_ORANGE
    assert s1["background"] == "white"
    assert value_style == {"display": "none"}


def test_no_page_word_in_buttons():
    buttons = [
        child
        for child in dashboard.app.layout.children[0].children
        if getattr(child, "id", None) in {
            "btn-p1",
            "btn-p2",
            "btn-p3",
            "btn-p4",
            "btn-p5",
        }
    ]
    assert len(buttons) == 5
    assert all("Page" not in child.children for child in buttons)


def test_member_attendance_is_not_inflated_by_team_presence():
    df = pd.DataFrame(
        {
            "year": [2024],
            "cohort": ["English"],
            "Sector": ["Services & Consulting"],
            "sessions_scheduled": [5],
            "attendance_member_rows": [2],
            "member_days_present": [5],
            "member_days_virtual": [0],
            "team_days_present": [5],
            "team_days_virtual": [0],
            "team_attendance_rate": [1.0],
            "member_attendance_rate": [0.5],
        }
    )
    summary, missing = dashboard._attendance_summary(df, "Sector")
    assert missing == 0
    assert summary is not None and not summary.empty
    assert summary["attendance_projects"].iloc[0] == 1
    assert summary["sessions_scheduled"].iloc[0] == 5
    assert summary["member_attendance_rate"].iloc[0] == 50.0

    fig = dashboard._attendance_bar_fig(summary, "Sector")
    assert list(fig.data[0].x) == ["Services & Consulting"]
    assert list(fig.data[0].y) == [50.0]
    assert list(fig.data[0].text) == ["50.0%"]


def test_team_size_chart_uses_5_plus_and_not_specified():
    df = pd.DataFrame(
        {
            "team_member_count": [1.0, 2.0, 3.0, 5.0, 6.0, "5+", None],
            "team_size_from_attendance": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        }
    )
    fig = dashboard._team_size_fig(df)
    assert list(fig.data[0].x) == ["2", "3", "4", "5", "5+"]
    assert list(fig.data[0].y) == [1, 1, 0, 1, 2]
    assert list(fig.data[0].text) == ["1", "1", "0", "1", "2"]
    assert fig.layout.xaxis.type == "category"
    assert list(fig.layout.xaxis.categoryarray) == [
        "2",
        "3",
        "4",
        "5",
        "5+",
    ]
    assert fig.layout.xaxis.autorange != "reversed"
    assert fig.layout.annotations[0].text == "1 Not Specified"


def test_distribution_charts_support_percent_mode():
    df = _combined_fixture()
    rendered = dashboard.update_page(
        "page2",
        None,
        None,
        None,
        None,
        None,
        None,
        value_mode="percent",
        df_input=df,
    )
    fig = dict(_card_figures(rendered))["Business Stage"]
    text = list(fig.data[0].text)
    values = list(fig.data[0].y)
    assert "Not Specified" not in list(fig.data[0].x)
    assert text and all(str(item).endswith("%") for item in text)
    assert all(0 <= value <= 100 for value in values)
    assert abs(sum(values) - 100.0) < 0.2
    missing = int((df["Business Stage"] == "Not Specified").sum())
    assert fig.layout.annotations[0].text == (
        f"{missing / len(df) * 100:.1f}% Not Specified"
    )

    default_rendered = dashboard.update_page(
        "page2", None, None, None, None, None, None, df_input=df
    )
    default_fig = dict(_card_figures(default_rendered))["Business Stage"]
    default_text = list(default_fig.data[0].text)
    assert "Not Specified" not in list(default_fig.data[0].x)
    assert all(not str(item).endswith("%") for item in default_text)
    assert default_text == [str(int(value)) for value in default_fig.data[0].y]
    assert default_fig.layout.annotations[0].text == (
        f"{missing} Not Specified"
    )


def test_team_size_chart_percent_mode():
    df = pd.DataFrame(
        {"team_member_count": [1.0, 2.0, 3.0, 5.0, 6.0, "5+", None]}
    )
    fig = dashboard._team_size_fig(df, as_percent=True)
    assert all(str(item).endswith("%") for item in fig.data[0].text)
    assert abs(sum(fig.data[0].y) - 100.0) < 0.2
    assert fig.layout.annotations[0].text == "16.7% Not Specified"


def test_top_sectors_excludes_not_specified_and_keeps_top_five():
    df = _combined_fixture()
    rendered = dashboard.update_page(
        "page3", None, None, None, None, None, None, df_input=df
    )
    fig = dict(_card_figures(rendered))["Top Sectors"]
    sectors = [str(value) for value in fig.data[0].y]
    cnt_sec, _ = dashboard._split_missing(df["Sector"].value_counts())
    expected_sectors = list(
        cnt_sec.sort_values(ascending=False, kind="mergesort")
        .head(5)
        .sort_values()
        .index
    )
    assert sectors == expected_sectors
    assert len(sectors) == 5
    assert "Not Specified" not in sectors
    missing = int((df["Sector"] == "Not Specified").sum())
    assert fig.layout.annotations[0].text == f"{missing} Not Specified"
    assert fig.layout.annotations[0].y == -0.09


def test_sector_vs_outcome_stacks_counts_and_percent():
    df = _combined_fixture()
    count_rendered = dashboard.update_page(
        "page3", None, None, None, None, None, None, df_input=df
    )
    rows = count_rendered[2].children
    assert [card.children[0].children for card in rows[0].children] == [
        "Applicant Type vs Outcome",
        "Sector vs Outcome",
    ]
    assert [card.children[0].children for card in rows[1].children] == [
        "Top Sectors",
        "Applicant Type Over Years",
    ]

    figures = dict(_card_figures(count_rendered))
    fig = figures["Sector vs Outcome"]
    sectors = [str(value) for value in fig.layout.yaxis.categoryarray]
    cnt_sec, _ = dashboard._split_missing(df["Sector"].value_counts())
    expected_sectors = list(
        cnt_sec.sort_values(ascending=False, kind="mergesort")
        .head(5)
        .sort_values()
        .index
    )
    assert sectors == expected_sectors
    assert len(sectors) == 5
    assert fig.layout.annotations[0].y == -0.09
    for sector in sectors:
        per_trace = [dict(zip(trace.y, trace.x)) for trace in fig.data]
        segments = [
            float(mapping.get(sector, 0)) for mapping in per_trace
        ]
        assert abs(sum(segments) - len(df[df["Sector"] == sector])) < 0.01

    percent_rendered = dashboard.update_page(
        "page3", None, None, None, None, None, None, "percent", df_input=df
    )
    percent_fig = dict(_card_figures(percent_rendered))["Sector vs Outcome"]
    for trace in percent_fig.data:
        assert all(str(text).endswith("%") for text in trace.text)
    for sector in sectors:
        per_trace = [dict(zip(trace.y, trace.x)) for trace in percent_fig.data]
        segments = [
            float(mapping.get(sector, 0)) for mapping in per_trace
        ]
        assert abs(sum(segments) - 100.0) < 0.2


def test_dist_fig_switches_between_counts_and_percent():
    counts = pd.Series([3, 1], index=["A", "B"])
    percent_fig = dashboard._dist_fig(
        counts.index, counts.values, as_percent=True, total=4
    )
    assert list(percent_fig.data[0].text) == ["75.0%", "25.0%"]
    count_fig = dashboard._dist_fig(
        counts.index, counts.values, as_percent=False, total=4
    )
    assert list(count_fig.data[0].text) == ["3", "1"]


def test_notes_sit_outside_plot_area():
    counts = pd.Series([3, 1], index=["A", "B"])
    vertical = dashboard._dist_fig(
        counts.index, counts.values, note="2 Not Specified"
    )
    assert vertical.layout.annotations[0].y == -0.09
    assert vertical.layout.annotations[0].yanchor == "top"
    assert vertical.layout.margin.b >= 56

    horizontal = dashboard._dist_fig(
        counts.index, counts.values, orientation="h", note="2 Not Specified"
    )
    assert horizontal.layout.annotations[0].y == -0.09
    assert horizontal.layout.annotations[0].yanchor == "top"
    assert horizontal.layout.margin.b >= 56


def test_attendance_bar_fig_shows_rate_with_note():
    summary = pd.DataFrame(
        {
            "year": [2023, 2024],
            "attendance_projects": [5, 6],
            "sessions_scheduled": [10, 12],
            "member_opportunities": [100, 120],
            "member_days_attended": [80, 90],
            "member_attendance_rate": [80.0, 75.0],
        }
    )
    figure = dashboard._attendance_bar_fig(
        summary, "year", sort_by="label", note="2 Not Specified"
    )
    assert all(str(text).endswith("%") for text in figure.data[0].text)
    assert list(figure.data[0].y) == [80.0, 75.0]
    assert figure.layout.annotations[0].text == "2 Not Specified"


def test_attendance_sector_chart_stays_percent_when_counts_requested():
    df = _combined_fixture()
    rate_rendered = dashboard.update_page(
        "page5", None, None, None, None, None, None,
        value_mode="percent", df_input=df,
    )
    count_rendered = dashboard.update_page(
        "page5", None, None, None, None, None, None,
        value_mode="counts", df_input=df,
    )
    rate_figures = dict(_card_figures(rate_rendered))
    count_figures = dict(_card_figures(count_rendered))

    sector_rate = rate_figures["Member Attendance by Sector"]
    sector_count = count_figures["Member Attendance by Sector"]
    assert all(str(text).endswith("%") for text in sector_rate.data[0].text)
    assert all(str(text).endswith("%") for text in sector_count.data[0].text)
    assert list(sector_count.data[0].x) == list(sector_rate.data[0].x)
    assert list(sector_count.data[0].y) == list(sector_rate.data[0].y)
    assert list(sector_count.data[0].text) == list(sector_rate.data[0].text)

    for title in ("Member Attendance Rate by Cohort", "Member Attendance Rate by Year"):
        assert count_figures[title].data[0].text == rate_figures[title].data[0].text
        assert all(
            str(text).endswith("%")
            for text in count_figures[title].data[0].text
        )


def test_missing_note_returns_none_for_zero_count():
    assert dashboard._missing_note({"not specified": 0}) is None


def test_attendance_summary_excludes_missing_groups():
    df = pd.DataFrame(
        {
            "year": [2024] * 5,
            "Sector": ["Tech", "Food", None, "Not Specified", "Edu"],
            "sessions_scheduled": [2] * 5,
            "attendance_member_rows": [3] * 5,
            "member_days_present": [4, 5, 6, 4, 5],
            "member_days_virtual": [1, 2, 1, 2, 1],
        }
    )
    summary, missing = dashboard._attendance_summary(df, "Sector")
    assert missing == 2
    assert set(summary["Sector"]) == {"Tech", "Food", "Edu"}


def test_cohort_page_removes_top_sectors_and_expands_applicant_type():
    df = _combined_fixture()
    rendered = dashboard.update_page(
        "page4", None, None, None, None, None, None, df_input=df
    )
    rows = rendered[2].children
    assert len(rows) == 3
    first_row_titles = [card.children[0].children for card in rows[0].children]
    second_row_titles = [card.children[0].children for card in rows[1].children]
    assert first_row_titles == [
        "Language Cohort vs Outcome",
        "Cohort Size by Language Cohort",
    ]
    assert second_row_titles == [
        "Accepted vs Rejected by Cohort",
        "Applicant Type by Language Cohort",
    ]
    titles = first_row_titles + second_row_titles
    assert "Top Sectors by Cohort" not in titles
    assert "Acceptance Rate by Language Cohort" not in titles
    cohort_size_fig = rows[1].children[0].children[1].children[0].figure
    expected_ids = sorted(
        str(value) for value in df["cohort_id"].dropna().unique()
    )
    assert cohort_size_fig.layout.barmode == "stack"
    assert cohort_size_fig.data[0].orientation == "h"
    assert [
        str(value) for value in cohort_size_fig.layout.yaxis.categoryarray
    ] == expected_ids
    assert [
        str(value) for value in cohort_size_fig.layout.yaxis.ticktext
    ] == [
        dashboard._cohort_size_label(value) for value in expected_ids
    ]
    assert {trace.name for trace in cohort_size_fig.data} >= {
        "Accepted",
        "Rejected",
    }
    assert [
        card.children[0].children for card in rows[2].children
    ] == ["Accepted Applications Over Years"]


def test_cohort_language_charts_stack_counts_and_percent():
    df = _combined_fixture()
    count_rendered = dashboard.update_page(
        "page4", None, None, None, None, None, None, df_input=df
    )
    figures = dict(_card_figures(count_rendered))
    cohort_counts = (
        df["cohort"]
        .value_counts()
        .sort_values(ascending=False, kind="mergesort")
    )
    expected_desc = list(cohort_counts.index)
    expected_visual = list(cohort_counts.sort_values().index)

    outcome_fig = figures["Language Cohort vs Outcome"]
    assert outcome_fig.data[0].orientation == "h"
    assert outcome_fig.layout.barmode == "stack"
    assert [
        str(value) for value in outcome_fig.layout.yaxis.categoryarray
    ] == [str(value) for value in expected_visual]
    for cohort in expected_desc:
        segments = [
            float(dict(zip(trace.y, trace.x)).get(cohort, 0))
            for trace in outcome_fig.data
        ]
        assert abs(sum(segments) - len(df[df["cohort"] == cohort])) < 0.01

    size_fig = figures["Cohort Size by Language Cohort"]
    assert size_fig.data[0].orientation == "h"
    assert [trace.name for trace in size_fig.data] == [
        str(cohort) for cohort in expected_visual
    ]
    assert size_fig.data[0].marker.color == dashboard.C_ORANGE_LIGHT
    assert size_fig.data[1].marker.color == dashboard.C_ORANGE_DARK
    assert size_fig.data[0].legendrank < size_fig.data[1].legendrank
    assert [str(value) for value in size_fig.layout.yaxis.categoryarray] == [
        str(cohort) for cohort in expected_visual
    ]
    for trace in size_fig.data:
        cohort = str(trace.y[0])
        assert float(trace.x[0]) == int(cohort_counts[cohort])

    percent_rendered = dashboard.update_page(
        "page4", None, None, None, None, None, None, "percent", df_input=df
    )
    percent_figures = dict(_card_figures(percent_rendered))
    percent_outcome = percent_figures["Language Cohort vs Outcome"]
    for trace in percent_outcome.data:
        assert all(str(text).endswith("%") for text in trace.text)
    for cohort in expected_desc:
        segments = [
            float(dict(zip(trace.y, trace.x)).get(cohort, 0))
            for trace in percent_outcome.data
        ]
        assert abs(sum(segments) - 100.0) < 0.2

    percent_size = percent_figures["Cohort Size by Language Cohort"]
    for trace in percent_size.data:
        cohort = str(trace.y[0])
        assert str(trace.text[0]).endswith("%")
        assert abs(
            float(trace.x[0])
            - round(int(cohort_counts[cohort]) / len(df) * 100, 1)
        ) < 0.01


def test_overview_and_sector_pages_use_uniform_grid():
    dashboard.handle_upload(_upload_df(_combined_fixture()), "combined.csv")
    for page in ("page1", "page3"):
        rendered = dashboard.update_page(page, None, None, None, None, None, None)
        rows = rendered[2].children
        assert rows
        cards = []

        def collect_cards(node):
            if getattr(node, "className", None) == "chart-card":
                cards.append(node.style)
            children = getattr(node, "children", None)
            if children is None:
                return
            if not isinstance(children, (list, tuple)):
                children = [children]
            for child in children:
                collect_cards(child)

        for row in rows:
            assert "chart-grid" in row.className
            assert row.style.get("display") == "grid"
            collect_cards(row)
        assert cards
        assert all(card_style.get("flex") == "1" for card_style in cards)


def test_overview_total_applications_over_years_matches_grouped_counts():
    df = _combined_fixture()
    rendered = dashboard.update_page(
        "page1",
        None,
        None,
        None,
        None,
        None,
        None,
        df_input=df,
    )
    first_row = rendered[2].children[0]
    assert [card.children[0].children for card in first_row.children] == [
        "Applications by Year & Cohort",
        "Total Applications Over Years",
    ]

    fig = dict(_card_figures(rendered))["Total Applications Over Years"]
    expected_years = sorted(df["year"].unique())
    assert fig.layout.xaxis.type == "category"
    assert list(fig.data[0].x) == expected_years
    assert list(fig.data[0].y) == [
        int(len(df[df["year"] == year]))
        for year in expected_years
    ]
    assert fig.layout.yaxis.showgrid is False
    assert fig.layout.yaxis.visible is False
    assert getattr(fig, "_dashboard_height", None) == 360


def test_upload_derives_applicant_type_from_individual_or_team():
    df = _combined_fixture().drop(columns=["applicant_type"])
    (
        status,
        _year_opts,
        _cohort_opts,
        _outcome_opts,
        _sector_opts,
        type_opts,
    ) = dashboard.handle_upload(_upload_df(df), "combined.csv")
    assert status is not None
    assert "applicant_type" in dashboard.DF_GLOBAL.columns
    assert dashboard.DF_GLOBAL["applicant_type"].notna().all()
    assert type_opts
    assert all(opt["value"] in ("Individual", "Team") for opt in type_opts)


def main():
    tests = [
        test_index_page_serves,
        test_upload_builds_filter_options,
        test_all_pages_render_after_upload,
        test_filters_produce_content,
        test_empty_data_uploads_render_a_prompt,
        test_switch_page,
        test_upload_derives_applicant_type_from_individual_or_team,
        test_member_attendance_is_not_inflated_by_team_presence,
        test_team_size_chart_uses_5_plus_and_not_specified,
        test_distribution_charts_support_percent_mode,
        test_team_size_chart_percent_mode,
        test_dist_fig_switches_between_counts_and_percent,
        test_notes_sit_outside_plot_area,
        test_top_sectors_excludes_not_specified_and_keeps_top_five,
        test_sector_vs_outcome_stacks_counts_and_percent,
        test_attendance_bar_fig_shows_rate_with_note,
        test_attendance_sector_chart_stays_percent_when_counts_requested,
        test_cohort_page_removes_top_sectors_and_expands_applicant_type,
        test_cohort_language_charts_stack_counts_and_percent,
        test_overview_and_sector_pages_use_uniform_grid,
        test_overview_total_applications_over_years_matches_grouped_counts,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
