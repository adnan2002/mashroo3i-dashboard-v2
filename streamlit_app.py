"""Streamlit version of the Mashroo3i dashboard.

This app reuses the chart construction from ``app.py`` so the Dash and
Streamlit versions stay visually aligned. It runs on Streamlit Community
Cloud with the same CSV/Excel upload and filters.
"""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st
from dash import dcc as dash_dcc

import app as dash_app


PAGE_LABELS = {
    "Overview": "page1",
    "Applicant Profile": "page2",
    "Sectors & Applicant Type": "page3",
    "Cohort Comparison": "page4",
    "Attendance": "page5",
}

# Applications columns the dashboard needs (the Dash app's REQUIRED_COLUMNS
# minus the attendance-specific columns).
APPLICATION_COLUMNS = [
    column
    for column in dash_app.REQUIRED_COLUMNS
    if column not in dash_app.ATTENDANCE_COLUMNS
]

# Attendance columns kept from the attendance file, plus the join/filter keys.
ATTENDANCE_KEEP_COLUMNS = list(dash_app.ATTENDANCE_COLUMNS) + [
    "year",
    "cohort",
    "matched_project_name",
    "matched_application_id",
]


def _children(value) -> list:
    """Return a component's children as a flat list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _find_graph(node):
    """Find the first Plotly Graph component inside a Dash node."""
    if isinstance(node, dash_dcc.Graph):
        return node
    for child in _children(getattr(node, "children", None)):
        found = _find_graph(child)
        if found is not None:
            return found
    return None


def _extract_cards(row):
    """Convert a Dash chart row into (title, figure, flex) tuples."""
    cards = []
    for card in _children(getattr(row, "children", None)):
        if not hasattr(card, "children"):
            continue
        node_children = _children(getattr(card, "children", None))
        title_node = node_children[0] if node_children else None
        if hasattr(title_node, "children"):
            title_node = title_node.children
        title = " ".join(str(item) for item in _children(title_node)) if isinstance(title_node, list) else str(title_node or "")
        graph = _find_graph(card)
        if graph is None:
            continue
        flex = getattr(card, "style", {}) or {}
        cards.append((title, graph.figure, flex.get("flex", "1")))
    return cards


def _render_rows(rendered):
    """Return the chart row list from a Dash page render."""
    if len(rendered) < 3:
        return None
    container = rendered[2]
    if not hasattr(container, "children"):
        return None
    rows = _children(getattr(container, "children", None))
    return rows or None


def _read_matching_sheet(
    file_bytes: bytes,
    filename: str,
    candidate_columns: list[str],
    min_match: int = 3,
) -> tuple[pd.DataFrame, str]:
    """Read a CSV or the first Excel sheet whose columns match a dataset type.

    Raises ValueError when no sheet contains enough matching columns.
    """
    if filename.lower().endswith(".csv"):
        sheets = {
            "<csv>": pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8-sig")
        }
    else:
        sheets = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
    for sheet_name, sheet in sheets.items():
        overlap = [column for column in candidate_columns if column in sheet.columns]
        if len(overlap) >= min_match:
            return sheet, sheet_name
    raise ValueError(
        "No sheet with the expected columns was found in "
        f"{filename} (looked for: {', '.join(candidate_columns[:6])}, ...)."
    )


def _normalize_shared(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same year/cohort/applicant_type normalization as the apps."""
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    if "cohort" in df.columns:
        df["cohort"] = df["cohort"].map(dash_app._clean_cohort)
    if "applicant_type" in df.columns:
        df["applicant_type"] = df["applicant_type"].map(dash_app._clean_applicant_type)
    return df


@st.cache_data(show_spinner=False)
def load_applications(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Load the applications sheet/file and keep the dashboard columns."""
    raw, _sheet = _read_matching_sheet(file_bytes, filename, APPLICATION_COLUMNS)
    keep = [
        column
        for column in APPLICATION_COLUMNS + ["project_name", "individual_or_team"]
        if column in raw.columns
    ]
    df = _normalize_shared(raw[keep].copy())
    # Real data uses individual_or_team (per dataframe_schemas.json), while the
    # dashboard filters on applicant_type. Derive the alias when missing.
    if "applicant_type" not in df.columns and "individual_or_team" in df.columns:
        df["applicant_type"] = df["individual_or_team"].map(
            dash_app._clean_applicant_type
        )
    return df


@st.cache_data(show_spinner=False)
def load_attendance(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Load the attendance sheet/file and keep attendance + join columns."""
    raw, _sheet = _read_matching_sheet(
        file_bytes, filename, dash_app.ATTENDANCE_COLUMNS
    )
    keep = [column for column in ATTENDANCE_KEEP_COLUMNS if column in raw.columns]
    return _normalize_shared(raw[keep].copy())


def available_pages(attendance_loaded: bool) -> list[str]:
    """Pages to show in navigation; Attendance only when data is loaded."""
    pages = list(PAGE_LABELS)
    if not attendance_loaded:
        pages.remove("Attendance")
    return pages


def join_attendance(
    attendance: pd.DataFrame, applications: pd.DataFrame
) -> pd.DataFrame:
    """Left-join attendance rows onto applications by normalized project name.

    Unmatched attendance rows keep their attendance metrics and get blank
    application attributes, so outcome/sector/type filters exclude them.
    """
    if (
        "project_name" not in applications.columns
        or "matched_project_name" not in attendance.columns
    ):
        return attendance.copy()

    app = applications.copy()
    app["_project_norm"] = app["project_name"].astype(str).str.strip().str.lower()
    app = app.drop_duplicates(subset=["_project_norm"])

    att = attendance.copy()
    att["_project_norm"] = (
        att["matched_project_name"].astype(str).str.strip().str.lower()
    )

    app_cols = [
        column
        for column in app.columns
        if column not in ("year", "cohort", "project_name")
    ]
    joined = att.merge(app[app_cols], on="_project_norm", how="left")
    return joined.drop(columns=["_project_norm"])


def _filter_options(series: pd.Series) -> list:
    values = series.dropna().unique()
    if pd.api.types.is_integer_dtype(series):
        return sorted(int(value) for value in values)
    return sorted(str(value) for value in values)


def _apply_filters(df: pd.DataFrame, years, cohorts, outcomes, sectors, types) -> pd.DataFrame:
    filtered = df
    if years:
        filtered = filtered[filtered["year"].isin(years)]
    if cohorts:
        filtered = filtered[filtered["cohort"].isin(cohorts)]
    if outcomes:
        filtered = filtered[filtered["outcome_clean"].isin(outcomes)]
    if sectors:
        filtered = filtered[filtered["Sector"].isin(sectors)]
    if types:
        filtered = filtered[filtered["applicant_type"].isin(types)]
    return filtered


def _kpcs(df: pd.DataFrame) -> tuple[int, int, float, float]:
    total = len(df)
    accepted = len(df[df["outcome_clean"] == "Accepted"]) if "outcome_clean" in df else 0
    rate = round(accepted / total * 100, 1) if total else 0
    bahraini = len(df[df["nationality"].astype(str).str.contains("bahrain", case=False, na=False)])
    bah_rate = round(bahraini / total * 100, 1) if total else 0
    return total, accepted, rate, bah_rate


def _render_kpis(df: pd.DataFrame) -> None:
    total, accepted, rate, bah_rate = _kpcs(df)
    values = [
        ("👥", "Total Applicants", total),
        ("✅", "Accepted", accepted),
        ("📈", "Acceptance Rate", f"{rate}%"),
        ("🇧🇭", "Bahraini Nationals", f"{bah_rate}%"),
    ]
    columns = st.columns(4)
    for column, (icon, label, value) in zip(columns, values):
        with column:
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-icon">{icon}</div>
                  <div class="kpi-value">{value}</div>
                  <div class="kpi-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_plotly(figure, key: str = None) -> None:
    """Render a Plotly figure with the same compact look as the Dash app."""
    figure.update_layout(height=320)
    st.plotly_chart(
        figure,
        width="stretch",
        config={"displayModeBar": False},
        key=key,
    )


def _inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
          --ms-orange: #ff6b2e;
          --ms-orange-dark: #e6531f;
          --ms-peach: #fddcc9;
          --ms-peach-light: #fff4ee;
          --ms-bg: #fbf6f2;
          --ms-text: #2d2926;
          --ms-muted: #8d7d74;
          --ms-border: #f2e2d9;
          --ms-shadow: 0 10px 28px rgba(99, 51, 18, 0.07);
        }
        .stApp {
          background: var(--ms-bg);
        }
        .stApp [data-testid="stMain"] {
          padding-top: 0 !important;
        }
        .stApp [data-testid="stMainBlockContainer"] {
          padding-top: 0 !important;
          margin-top: 0 !important;
        }
        .stApp [data-testid="stAppViewContainer"] {
          padding-top: 0 !important;
        }
        [data-testid="stHeader"] {
          background: transparent;
        }
        [data-testid="stSidebar"] {
          background: #ffffff;
          border-right: 1px solid var(--ms-peach);
          box-shadow: 8px 0 30px rgba(99, 51, 18, 0.035);
        }
        [data-testid="stSidebarContent"] {
          padding-top: 1.2rem;
        }
        [data-testid="stMetric"] {
          background: #ffffff;
          border: 1px solid var(--ms-border);
          border-radius: 15px;
          padding: 1rem;
          box-shadow: var(--ms-shadow);
        }
        [data-testid="stPlotlyChart"] {
          border: 1px solid var(--ms-border);
          border-radius: 16px;
          background: #ffffff;
          padding: .5rem .3rem .3rem;
          box-shadow: var(--ms-shadow);
        }
        .sidebar-brand {
          color: var(--ms-orange);
          font-size: 1.35rem;
          font-weight: 800;
          letter-spacing: -.3px;
          margin-bottom: -.2rem;
        }
        .sidebar-caption {
          color: var(--ms-muted);
          font-size: .78rem;
          font-weight: 600;
          margin-bottom: .7rem;
        }
        .page-title {
          color: var(--ms-text);
          font-size: 1.55rem;
          font-weight: 800;
          letter-spacing: -.3px;
          margin-bottom: .1rem;
        }
        .page-subtitle {
          color: var(--ms-muted);
          font-size: .82rem;
          font-weight: 600;
          margin-bottom: .5rem;
        }
        .kpi-card {
          min-width: 0;
          border: 1px solid var(--ms-border);
          border-radius: 15px;
          background: #ffffff;
          box-shadow: var(--ms-shadow);
          padding: .9rem 1rem;
          display: flex;
          flex-direction: column;
          gap: .35rem;
        }
        .kpi-icon {
          width: 32px;
          height: 32px;
          display: flex;
          align-items: center;
          justify-content: center;
          border-radius: 10px;
          background: var(--ms-peach-light);
          font-size: 1.05rem;
        }
        .kpi-value {
          color: var(--ms-text);
          font-size: 1.45rem;
          font-weight: 800;
          line-height: 1.1;
          letter-spacing: -.5px;
        }
        .kpi-label {
          color: var(--ms-orange-dark);
          font-size: .62rem;
          font-weight: 800;
          letter-spacing: .2px;
          text-transform: uppercase;
        }
        .chart-title {
          position: relative;
          display: flex;
          align-items: center;
          min-height: 44px;
          padding: .7rem .7rem .2rem .95rem;
          color: var(--ms-text);
          font-size: .78rem;
          font-weight: 800;
        }
        .chart-title::before {
          content: "";
          position: absolute;
          top: .55rem;
          bottom: .55rem;
          left: 0;
          width: 4px;
          border-radius: 0 4px 4px 0;
          background: linear-gradient(180deg, var(--ms-orange), #ffb27a);
        }
        .upload-status {
          border: 1px solid var(--ms-peach);
          border-radius: 10px;
          padding: .5rem .65rem;
          background: var(--ms-peach-light);
          color: var(--ms-orange-dark);
          font-size: .78rem;
          font-weight: 700;
          text-align: center;
        }
        div[data-testid="stFileUploaderDropzone"] {
          border: 1.5px dashed #f3b38f;
          border-radius: 13px;
          background: var(--ms-peach-light);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Mashroo3i Dashboard",
        page_icon="🏗️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_theme()

    with st.sidebar:
        st.markdown('<div class="sidebar-brand">🏗️ Mashroo3i</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-caption">Applicant &amp; cohort insights</div>',
            unsafe_allow_html=True,
        )
        uploaded_apps = st.file_uploader(
            "Applications (required)",
            type=["csv", "xlsx", "xls"],
            key="applications-uploader",
        )
        uploaded_attendance = st.file_uploader(
            "Attendance (optional)",
            type=["csv", "xlsx", "xls"],
            key="attendance-uploader",
            help="Upload the attendance file to unlock the Attendance page.",
        )

        if uploaded_apps is not None:
            try:
                st.session_state["applications"] = load_applications(
                    uploaded_apps.getvalue(), uploaded_apps.name
                )
            except ValueError as exc:
                st.warning(f"Applications file not loaded: {exc}")

        if uploaded_attendance is not None:
            try:
                st.session_state["attendance"] = load_attendance(
                    uploaded_attendance.getvalue(), uploaded_attendance.name
                )
            except ValueError as exc:
                st.warning(f"Attendance file not loaded: {exc}")

        applications = st.session_state.get("applications")
        attendance = st.session_state.get("attendance")
        if applications is None:
            st.info("Upload the applications file (CSV or Excel) to start.")
            st.stop()

        st.markdown(
            f'<div class="upload-status">✅ {len(applications)} applications</div>',
            unsafe_allow_html=True,
        )
        if attendance is not None:
            joined_preview = join_attendance(attendance, applications)
            matched = (
                int(joined_preview["Sector"].notna().sum())
                if "Sector" in joined_preview.columns
                else 0
            )
            st.markdown(
                f'<div class="upload-status">✅ {len(attendance)} attendance records '
                f"({matched} matched to applications)</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("Attendance not uploaded yet — the Attendance page stays hidden.")

        st.divider()
        page_name = st.sidebar.radio(
            "Page",
            available_pages(attendance is not None),
            label_visibility="collapsed",
        )
        display_mode = st.radio(
            "Show values as",
            ["Counts", "Percentages"],
            horizontal=True,
            key="value_mode",
        )
        value_mode = "percent" if display_mode == "Percentages" else "counts"
        st.markdown("**Filters**")
        years = st.multiselect("Years", _filter_options(applications["year"]), placeholder="All Years")
        cohorts = st.multiselect("Cohorts", _filter_options(applications["cohort"]), placeholder="All Cohorts")
        outcomes = st.multiselect("Outcomes", _filter_options(applications["outcome_clean"]), placeholder="All Outcomes")
        sectors = st.multiselect("Sectors", _filter_options(applications["Sector"]), placeholder="All Sectors")
        types = st.multiselect(
            "Applicant type",
            _filter_options(applications["applicant_type"]),
            placeholder="Individual or Team",
        )

    st.markdown(f'<div class="page-title">{page_name}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Live insights from the uploaded applications '
        "and attendance files</div>",
        unsafe_allow_html=True,
    )

    filtered = _apply_filters(applications, years, cohorts, outcomes, sectors, types)
    if len(filtered) == 0:
        st.info("No data matches the current filters. Try removing one or more filters.")
        return

    _render_kpis(filtered)
    st.write("")

    if page_name == "Attendance":
        joined = join_attendance(attendance, applications)
        filtered_att = _apply_filters(
            joined, years, cohorts, outcomes, sectors, types
        )
        if len(filtered_att) == 0:
            st.info(
                "No attendance data matches the current filters. "
                "Try removing one or more filters."
            )
            return
        rendered = dash_app.update_page(
            "page5",
            None,
            years,
            cohorts,
            outcomes,
            sectors,
            types,
            value_mode=value_mode,
            df_input=filtered_att,
        )
    else:
        rendered = dash_app.update_page(
            PAGE_LABELS[page_name],
            None,
            years,
            cohorts,
            outcomes,
            sectors,
            types,
            value_mode=value_mode,
            df_input=filtered,
        )
    rows = _render_rows(rendered)
    if rows is None:
        st.info("This page has no data in the uploaded file.")
        return

    for row_index, row in enumerate(rows):
        cards = _extract_cards(row)
        if not cards:
            continue
        widths = [float(card[2]) for card in cards]
        columns = st.columns(widths)
        for card_index, (column, (title, figure, _)) in enumerate(zip(columns, cards)):
            with column:
                st.markdown(f'<div class="chart-title">{title}</div>', unsafe_allow_html=True)
                render_plotly(
                    figure,
                    key=f"page-{page_name}-row-{row_index}-card-{card_index}",
                )


if __name__ == "__main__":
    main()
