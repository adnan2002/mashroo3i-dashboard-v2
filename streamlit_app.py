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
import idea_agent
import model_service
from hybrid_evaluate import idea_text_for, strip_outcomes


PAGE_LABELS = {
    "Overview": "page1",
    "Applicant Profile": "page2",
    "Sectors & Applicant Type": "page3",
    "Cohort Comparison": "page4",
    "Attendance": "page5",
}
AGENT_PAGE = "Idea Validator"
HYBRID_PAGE = "Selection Advisor"
SUMMARY_PAGE = "Dashboard Summary"

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

# Columns that only the full Brinc applicant file has; used to decide whether
# a sidebar upload can feed the Selection Advisor model directly.
MODEL_INPUT_KEYS = (
    "project_name",
    "date_of_birth",
    "individual_or_team",
    "problem",
    "solution",
    "in_two_cohorts",
    "sector_all",
)


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
    # Real data uses individual_or_team, while the dashboard filters on
    # applicant_type. Derive the alias when missing.
    if "applicant_type" not in df.columns and "individual_or_team" in df.columns:
        df["applicant_type"] = df["individual_or_team"].map(
            dash_app._clean_applicant_type
        )
    return df


def load_raw_upload(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Load the full uploaded file, preferring a sheet with the model schema."""
    if filename.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8-sig")
    sheets = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
    for sheet in sheets.values():
        if _has_model_schema(sheet):
            return sheet
    return next(iter(sheets.values()))


def _has_model_schema(df: pd.DataFrame) -> bool:
    return all(key in df.columns for key in MODEL_INPUT_KEYS)


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
    pages.extend(ai_pages())
    return pages


def dashboard_pages(attendance_loaded: bool) -> list[str]:
    """Dashboard section pages; Attendance only when data is loaded."""
    pages = list(PAGE_LABELS)
    if not attendance_loaded:
        pages.remove("Attendance")
    return pages


def ai_pages() -> list[str]:
    """AI-assisted section pages."""
    return [AGENT_PAGE, SUMMARY_PAGE, HYBRID_PAGE]


def _agent_dimension_rows(score: dict) -> pd.DataFrame:
    """Turn a score payload into a small, render-friendly DataFrame."""
    rows = []
    weights = dict(idea_agent.SCORING_RUBRIC)
    for name, dimension in (score.get("dimensions") or {}).items():
        rows.append(
            {
                "Dimension": name,
                "Score": f"{dimension.get('score')} / 5",
                "Weight": f"{weights.get(name, 0):.0%}",
                "Rationale": dimension.get("rationale", ""),
                "Evidence": " | ".join(dimension.get("evidence") or []),
            }
        )
    return pd.DataFrame(rows)


def _render_agent_report(report: dict) -> None:
    """Render the idea-validation score report (no dashboard summary)."""
    score = report.get("score") or {}
    if not score:
        st.info("Idea validation is not available for this run.")
        return
    total = score.get("total_score")
    metric_columns = st.columns(3)
    metric_columns[0].metric(
        "Selection Score",
        f"{total}/25" if total is not None else "N/A",
    )
    metric_columns[1].metric("Verdict", score.get("verdict") or "Unknown")
    metric_columns[2].metric("Web sources", len(score.get("sources") or []))
    if score.get("low_evidence") or score.get("evidence_note"):
        st.warning(
            score.get("evidence_note")
            or "Web research was thin, so this score is low-confidence."
        )
    if score.get("bahrain_impact"):
        st.markdown(f"**Bahrain impact:** {score['bahrain_impact']}")

    dimension_rows = _agent_dimension_rows(score)
    if len(dimension_rows):
        st.markdown("### Rubric breakdown")
        st.dataframe(dimension_rows, use_container_width=True, hide_index=True)

    risks = score.get("risks") or []
    recommendations = score.get("recommendations") or []
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Risks")
        if risks:
            for risk in risks:
                st.markdown(f"- {risk}")
        else:
            st.caption("No risks identified.")
    with col2:
        st.markdown("### Recommendations")
        if recommendations:
            for item in recommendations:
                st.markdown(f"- {item}")
        else:
            st.caption("No recommendations available.")

    sources = score.get("sources") or []
    if sources:
        st.markdown("### Sources")
        for source in sources:
            title = source.get("title") or source.get("url")
            url = source.get("url") or ""
            if url:
                st.markdown(f"- [{title}]({url})")
            else:
                st.markdown(f"- {title}")


def _render_agent_page(applications: pd.DataFrame, attendance) -> None:
    """Render the AI Agent page: idea input -> validation + dashboard insights."""
    st.markdown(f'<div class="page-title">{AGENT_PAGE}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Validate an idea with web research and '
        "score it against the official /25 selection rubric."
        "</div>",
        unsafe_allow_html=True,
    )

    ready, message = idea_agent.agent_ready()
    if not ready:
        st.warning(message)
    else:
        st.caption(message)

    idea = st.text_area(
        "Idea / problem",
        key="agent_idea",
        height=96,
        placeholder="e.g. A platform that connects farmers with local restaurants...",
        help="This is the only input the agent needs besides your uploaded data.",
    )
    description = st.text_area(
        "Description",
        key="agent_description",
        height=180,
        placeholder="Describe the problem, who it affects, and how you plan to solve it.",
    )
    if st.button(
        "Validate & Analyze",
        key="agent_run",
        type="primary",
        use_container_width=True,
    ):
        if not idea.strip() and not description.strip():
            st.warning("Please enter an idea or a problem first.")
        else:
            idea_text = (
                f"Problem: {idea.strip()}\nDescription: {description.strip()}"
            )
            status_box = st.empty()
            status_box.markdown("**Evaluating...**")

            def _status(message: str) -> None:
                status_box.markdown(f"**{message}**")

            try:
                report = idea_agent.run_agent_stream(
                    idea_text=idea_text,
                    applications=applications,
                    attendance=attendance,
                    include_dashboard=False,
                    on_status=_status,
                )
                st.session_state["agent_report"] = report.to_dict()
                status_box.markdown("**Evaluation complete**")
            except Exception as exc:
                st.error(f"Agent run failed: {exc}")
                status_box.markdown("**Evaluation failed**")

    report = st.session_state.get("agent_report") or {}
    if report:
        errors = report.get("errors") or []
        if errors:
            for error in errors:
                st.warning(error)
        _render_agent_report(report)


def _render_dashboard_summary(summary: dict) -> None:
    """Render an appealing, card-based dashboard summary."""
    kpis = summary.get("kpis") or {}
    card_specs = [
        ("Applications", kpis.get("applications"), "👥"),
        ("Accepted", kpis.get("accepted"), "✅"),
        ("Acceptance rate", kpis.get("acceptance_rate_pct"), "📈"),
    ]
    present = [(label, value, icon) for label, value, icon in card_specs if value is not None]
    if present:
        columns = st.columns(len(present))
        for column, (label, value, icon) in zip(columns, present):
            suffix = "%" if "share" in label or "rate" in label else ""
            with column:
                st.markdown(
                    f"""
                    <div class="kpi-card">
                      <div class="kpi-icon">{icon}</div>
                      <div class="kpi-value">{value}{suffix}</div>
                      <div class="kpi-label">{label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    narrative = (summary.get("summary") or "").strip()
    if narrative:
        st.markdown(
            f'<div class="summary-card">{narrative}</div>',
            unsafe_allow_html=True,
        )

    insights = summary.get("insights") or []
    if insights:
        st.markdown("### Highlights")
        cards = "".join(
            f'<div class="insight-card"><b>✦ Insight</b>'
            f'<span class="insight-text">{text}</span></div>'
            for text in insights
        )
        st.markdown(
            f'<div class="insight-grid">{cards}</div>',
            unsafe_allow_html=True,
        )

    snapshot = summary.get("snapshot") or {}
    top_categories = snapshot.get("top_categories") or {}
    if top_categories:
        st.markdown("### Top categories")
        category_titles = {
            "Sector": "Sector",
            "applicant_type": "Applicant Type",
            "outcome_clean": "Outcome",
            "cohort": "Language Cohort",
            "cohort_id": "Cohort ID",
            "Business Stage": "Business Stage",
        }
        category_columns = st.columns(min(3, len(top_categories)))
        category_items = list(top_categories.items())
        per_column = max(1, len(category_items) // len(category_columns) + 1)
        for index, (column_name, counts) in enumerate(category_items):
            with category_columns[index % len(category_columns)]:
                title = category_titles.get(column_name, column_name)
                st.markdown(f"**{title}**")
                st.dataframe(
                    pd.DataFrame(
                        [{"Category": key, "Count": value} for key, value in counts.items()]
                    ),
                    use_container_width=True,
                    hide_index=True,
                    height=180,
                )

    yearly = snapshot.get("yearly") or []
    if yearly:
        st.markdown("### Acceptance by year")
        st.dataframe(
            pd.DataFrame(yearly),
            use_container_width=True,
            hide_index=True,
        )

def _render_dashboard_summary_page(
    applications: pd.DataFrame,
    attendance,
    years=None,
    cohorts=None,
    outcomes=None,
    sectors=None,
    types=None,
) -> None:
    """Dedicated AI page: an appealing summary of the uploaded dashboards."""
    st.markdown(
        f'<div class="page-title">{SUMMARY_PAGE}</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="page-subtitle">An AI-written overview of your uploaded '
        "applications and attendance data</div>",
        unsafe_allow_html=True,
    )
    agent_ok, agent_message = idea_agent.agent_ready()
    if not agent_ok:
        st.warning(agent_message)
    else:
        st.caption(agent_message)

    filtered = _apply_filters(
        applications, years, cohorts, outcomes, sectors, types
    )
    if len(filtered) == 0:
        st.info("No applicants match the current filters.")
        return
    st.caption(
        f"Loaded {len(applications)} applicants, filtered to "
        f"{len(filtered)} by the sidebar filters."
    )
    active_filter_key = _selection_filter_key(
        years, cohorts, outcomes, sectors, types
    )

    if st.button(
        "Generate dashboard summary",
        key="dash-summary-run",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner("Analyzing your dashboards..."):
            try:
                agent = idea_agent.IdeaValidationAgent(filtered, attendance)
                insights = agent.summarize_dashboards()
                st.session_state["dash_summary"] = insights.to_dict()
                st.session_state["dash_filter_key"] = active_filter_key
            except Exception as exc:
                st.error(f"Dashboard summary failed: {exc}")

    summary = st.session_state.get("dash_summary")
    if summary:
        if st.session_state.get("dash_filter_key") != active_filter_key:
            st.caption(
                "Filters changed since the last summary - press Generate again."
            )
        _render_dashboard_summary(summary)


def _render_candidate_body(row: pd.Series, result: dict) -> None:
    """Render the structured agent result for one candidate."""
    score = result.get("score") or {}
    if result.get("error"):
        st.error(result["error"])
        return
    if score.get("bahrain_impact"):
        st.markdown(f"**Bahrain impact:** {score['bahrain_impact']}")
    dimensions = score.get("dimensions") or {}
    if dimensions:
        st.markdown("**Rubric breakdown**")
        for name, dimension in dimensions.items():
            rationale = str(dimension.get("rationale") or "")
            st.markdown(
                f"- **{name}**: {dimension.get('score')}/5 "
                f"- {rationale[:220]}"
            )
    if score.get("evidence_note"):
        st.info(score["evidence_note"])
    if score.get("risks"):
        st.markdown("**Risks**")
        for risk in score["risks"]:
            st.markdown(f"- {risk}")
    if score.get("recommendations"):
        st.markdown("**Recommendations**")
        for item in score["recommendations"]:
            st.markdown(f"- {item}")
    sources = score.get("sources") or []
    if sources:
        st.markdown("**Sources**")
        for source in sources:
            url = source.get("url") or ""
            title = source.get("title") or url
            st.markdown(f"- [{title}]({url})" if url else f"- {title}")


def _render_one_candidate_block(row: pd.Series, result: dict) -> None:
    """Render one agent-reviewed candidate (used for streaming and cache)."""
    score = result.get("score") or {}
    label = (
        f"{row.get('model_rank')}. {row.get('project_name')} "
        f"- Selection {score.get('total_score')}/25 "
        f"({score.get('verdict')})"
    )
    with st.expander(label):
        _render_candidate_body(row, result)


def _render_agent_result_expanders(results: dict, ranked: pd.DataFrame) -> None:
    """Show one expander per agent-reviewed candidate from the cache."""
    for _index, row in ranked.iterrows():
        identity = str(row.get("identity"))
        result = results.get(identity)
        if result:
            _render_one_candidate_block(row, result)


def _render_hybrid_page(
    years=None,
    cohorts=None,
    outcomes=None,
    sectors=None,
    types=None,
) -> None:
    """Model-first shortlist + agent second-stage evaluation page."""
    st.markdown(
        f'<div class="page-title">{HYBRID_PAGE}</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="page-subtitle">Model ranks candidates first and then the '
        "agent streams a /25 rubric review of the shortlist.</div>",
        unsafe_allow_html=True,
    )

    model_ok, model_message = model_service.available()
    agent_ok, agent_message = idea_agent.agent_ready()
    if not model_ok:
        st.warning(model_message)
    if not agent_ok:
        st.warning(agent_message)
    if model_ok:
        st.caption(model_message)
    if agent_ok:
        st.caption(agent_message)

    raw = st.session_state.get("brinc_raw")
    if raw is None:
        reused = st.session_state.get("raw_applications")
        if reused is not None and _has_model_schema(reused):
            st.session_state["brinc_raw"] = reused
            raw = reused

    uploaded = st.file_uploader(
        "Applications CSV",
        type=["csv"],
        key="brinc-uploader",
        help="Optional: replace the data already available. The model needs "
        "the full 54-column schema (problem/solution text, sector, DOB, "
        "etc.).",
    )
    real_path = model_service.REAL_DATA_CSV
    use_real = False
    if real_path.exists():
        use_real = st.checkbox(
            "Use the real dashboard_ready.csv instead",
            value=False,
            key="brinc-use-real",
            help=f"Loads {real_path}.",
        )
    if uploaded is not None:
        try:
            st.session_state["brinc_raw"] = pd.read_csv(
                io.BytesIO(uploaded.getvalue()), encoding="utf-8-sig"
            )
        except Exception as exc:
            st.error(f"Could not read the uploaded CSV: {exc}")
    elif use_real:
        st.session_state["brinc_raw"] = pd.read_csv(
            real_path, encoding="utf-8-sig"
        )

    if raw is None:
        candidate = st.session_state.get("raw_applications")
        if candidate is not None and not _has_model_schema(candidate):
            st.info(
                "The uploaded applications file does not include the full "
                "model schema. Upload the complete Brinc applicant CSV below."
            )
        elif real_path.exists():
            st.info(
                "Upload the full applicant CSV (or tick the real-data option) "
                "to start."
            )
        else:
            st.info("Upload the full applicant CSV to start.")
        return

    filtered_raw = _filter_raw_for_selection(
        raw, years, cohorts, outcomes, sectors, types
    )
    if len(filtered_raw) == 0:
        st.info("No applicants match the current filters.")
        return
    st.caption(
        f"Loaded {len(raw)} applicant rows, filtered to "
        f"{len(filtered_raw)} by the sidebar filters."
    )
    active_filter_key = _selection_filter_key(
        years, cohorts, outcomes, sectors, types
    )
    if st.button(
        "Rank candidates with the classifier model",
        key="brinc-rank",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner("Scoring applicants with the model..."):
            try:
                st.session_state["brinc_ranked"] = model_service.score_with_model(
                    filtered_raw
                )
                st.session_state["brinc_filter_key"] = active_filter_key
            except Exception as exc:
                st.error(f"Model scoring failed: {exc}")

    ranked = st.session_state.get("brinc_ranked")
    if ranked is None:
        st.info("Press the button above to produce the primary shortlist.")
        return
    if st.session_state.get("brinc_filter_key") != active_filter_key:
        st.caption("Filters changed since ranking - press Rank again to refresh.")

    predicted = int(ranked["predicted_accepted"].sum())
    threshold = float(ranked["prediction_threshold"].iloc[0])
    kpi_columns = st.columns(3)
    kpi_columns[0].metric("Applicants scored", len(ranked))
    kpi_columns[1].metric(
        "Predicted Accepted",
        predicted,
        help=f"At saved threshold {threshold:.3f}",
    )
    kpi_columns[2].metric(
        "Top probability", f"{ranked['accept_probability'].iloc[0]:.1%}"
    )

    if predicted == 0:
        st.info("No applicants are Predicted Accepted at the current threshold.")
        return
    slider_min = min(5, max(1, predicted))
    slider_max = max(1, predicted)
    shown = st.slider(
        "Shortlist size",
        slider_min,
        slider_max,
        min(10, slider_max),
        key="brinc-show",
        help=f"Maximum equals the {predicted} Predicted Accepted applicants.",
    )
    st.markdown("### Primary shortlist by classifier model")
    primary = ranked.head(shown)[
        [
            "model_rank",
            "project_name",
            "year",
            "Sector",
            "accept_probability",
            "predicted_accepted",
        ]
    ].copy()
    primary["predicted_accepted"] = primary["predicted_accepted"].map(
        {1: "Accepted", 0: "Rejected"}
    )
    st.dataframe(
        primary,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.markdown("### Streaming agent review (secondary evaluation)")
    max_agent = min(30, len(ranked))
    count = st.slider(
        "Top candidates for agent review",
        1,
        max_agent,
        5,
        key="brinc-agent-count",
        help="Each candidate takes roughly one minute (web search + LLM).",
    )
    streamed_this_run = False
    if st.button(
        f"Run agent on top {count} candidates",
        key="brinc-run-agent",
        use_container_width=True,
    ):
        results = st.session_state.setdefault("brinc_agent_results", {})
        stripped = strip_outcomes(filtered_raw)
        top = ranked.head(count).copy()
        progress = st.progress(0)
        status = st.empty()
        stream_area = st.container()
        for index, (_position, row) in enumerate(top.iterrows(), 1):
            identity = str(row["identity"])
            status.write(f"Reviewing {index}/{count}: {row['project_name']}")
            if not (identity in results and results[identity].get("score")):
                with st.expander(
                    f"{row.get('model_rank')}. {row.get('project_name')} "
                    "- Evaluating..."
                ):
                    candidate_status = st.empty()
                    candidate_status.markdown("**Evaluating...**")

                    def _candidate_status(message: str) -> None:
                        candidate_status.markdown(f"**{message}**")

                    try:
                        report = idea_agent.run_agent_stream(
                            idea_text=idea_text_for(row),
                            applications=stripped,
                            include_dashboard=False,
                            on_status=_candidate_status,
                        )
                        results[identity] = report.to_dict()
                    except Exception as exc:
                        results[identity] = {"error": str(exc)[:300]}
                    candidate_status.markdown("**Evaluation complete**")
                    _render_candidate_body(row, results[identity])
            else:
                # Cached result: render instantly.
                with stream_area:
                    _render_one_candidate_block(row, results[identity])
            progress.progress(index / count)
        status.write("Agent deep-dive complete.")
        streamed_this_run = True

    results = st.session_state.get("brinc_agent_results") or {}
    if results:
        review_rows = []
        for _position, row in ranked.head(count).iterrows():
            identity = str(row["identity"])
            result = results.get(identity) or {}
            score = result.get("score") or {}
            review_rows.append(
                {
                    "Model rank": row.get("model_rank"),
                    "Project": row.get("project_name"),
                    "Model prob.": round(float(row["accept_probability"]), 3),
                    "Innovation": score.get("total_score"),
                    "Verdict": score.get("verdict"),
                    "Sources": len(score.get("sources") or []),
                    "Status": "Error" if result.get("error") else "Done",
                }
            )
        st.markdown("### Combined shortlist after agent review")
        st.dataframe(
            pd.DataFrame(review_rows),
            use_container_width=True,
            hide_index=True,
        )
        if not streamed_this_run:
            # Returning to the page: show cached detail blocks instantly.
            _render_agent_result_expanders(results, ranked.head(count))
        st.caption(
            "The classifier model rank is the primary signal and the agent score is a "
            "second-opinion annotation (evidence, risks, selection rubric) for "
            "human review."
        )


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


def _filter_raw_for_selection(
    raw: pd.DataFrame,
    years=None,
    cohorts=None,
    outcomes=None,
    sectors=None,
    types=None,
) -> pd.DataFrame:
    """Apply the dashboard sidebar filters to the full-schema Brinc CSV."""
    filtered = raw
    if years and "year" in filtered.columns:
        filtered = filtered[filtered["year"].isin(years)]
    if cohorts and "cohort" in filtered.columns:
        filtered = filtered[filtered["cohort"].isin(cohorts)]
    if outcomes and "outcome_clean" in filtered.columns:
        filtered = filtered[filtered["outcome_clean"].isin(outcomes)]
    if sectors and "Sector" in filtered.columns:
        filtered = filtered[filtered["Sector"].isin(sectors)]
    if types:
        type_column = (
            "applicant_type"
            if "applicant_type" in filtered.columns
            else "individual_or_team"
        )
        if type_column in filtered.columns:
            filtered = filtered[filtered[type_column].isin(types)]
    return filtered


def _selection_filter_key(years, cohorts, outcomes, sectors, types) -> tuple:
    """Hash the active filters to detect when ranking is stale."""
    return (
        tuple(years or ()),
        tuple(cohorts or ()),
        tuple(outcomes or ()),
        tuple(sectors or ()),
        tuple(types or ()),
    )


def _kpcs(df: pd.DataFrame) -> tuple[int, int, float]:
    total = len(df)
    accepted = len(df[df["outcome_clean"] == "Accepted"]) if "outcome_clean" in df else 0
    rate = round(accepted / total * 100, 1) if total else 0
    return total, accepted, rate


def _render_kpis(df: pd.DataFrame) -> None:
    total, accepted, rate = _kpcs(df)
    values = [
        ("👥", "Total Applicants", total),
        ("✅", "Accepted", accepted),
        ("📈", "Acceptance Rate", f"{rate}%"),
    ]
    columns = st.columns(3)
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


def _numeric_value(value) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _categorical_x_labels(figure) -> list[str]:
    """Return categorical x tick labels, or [] for numeric/date axes."""
    xaxis = figure.layout.xaxis
    axis_type = getattr(xaxis, "type", None)
    explicit_category = axis_type in ("category", "multicategory")
    if axis_type not in (None, "-", "", "category", "multicategory"):
        return []  # explicit linear/log/date/etc. axis

    labels = [str(item) for item in (getattr(xaxis, "ticktext", None) or [])]
    if not labels:
        seen: set[str] = set()
        for trace in figure.data:
            values = getattr(trace, "x", None)
            if values is None:
                continue
            for value in values:
                text = str(value)
                if text not in seen:
                    seen.add(text)
                    labels.append(text)
    if not labels:
        return []
    numeric_only = all(_numeric_value(item) for item in labels)
    if numeric_only and not explicit_category:
        return []  # e.g. year axes
    return labels


def _labels_at_overlap_risk(labels: list[str]) -> bool:
    if not labels:
        return False
    longest = max(len(label) for label in labels)
    estimated_width = sum(len(label) for label in labels) * 7
    # Rotate when a few long-ish category names could touch on a narrow
    # container; keep trivial short labels (e.g. "1 2 3 4 5 5+") horizontal.
    return (
        (len(labels) > 4 and estimated_width > 120)
        or longest > 12
        or estimated_width > 450
    )


def _auto_vertical_axis_labels(figure) -> None:
    """Rotate crowded categorical x labels vertical (90deg), otherwise keep horizontal."""
    if getattr(figure, "_no_auto_vertical_labels", False):
        return
    labels = _categorical_x_labels(figure)
    if labels and _labels_at_overlap_risk(labels):
        figure.update_xaxes(tickangle=90, automargin=True)


def render_plotly(figure, key: str = None) -> None:
    """Render a Plotly figure with the same compact look as the Dash app."""
    _auto_vertical_axis_labels(figure)
    figure.update_layout(height=getattr(figure, "_dashboard_height", 320))
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
        .summary-card {
          border: 1px solid var(--ms-border);
          border-radius: 16px;
          background: #ffffff;
          box-shadow: var(--ms-shadow);
          padding: 1rem 1.15rem;
          margin: .35rem 0 .7rem;
          color: var(--ms-text);
          line-height: 1.55;
        }
        .insight-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
          gap: .8rem;
          margin: .4rem 0 .9rem;
        }
        .insight-card {
          display: flex;
          flex-direction: column;
          border: 1px solid var(--ms-border);
          border-radius: 14px;
          background: #ffffff;
          box-shadow: var(--ms-shadow);
          padding: .8rem .9rem;
          min-height: 112px;
          height: 100%;
          color: var(--ms-text);
          font-size: .86rem;
          line-height: 1.5;
        }
        .insight-card b {
          color: var(--ms-orange-dark);
        }
        .insight-text {
          display: block;
          margin-top: .35rem;
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
                st.session_state["raw_applications"] = load_raw_upload(
                    uploaded_apps.getvalue(), uploaded_apps.name
                )
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
        section = st.sidebar.radio(
            "Section",
            ["Dashboard", "AI"],
            horizontal=True,
            key="app_section",
        )
        if section == "Dashboard":
            page_name = st.sidebar.radio(
                "Page",
                dashboard_pages(attendance is not None),
                label_visibility="collapsed",
                key="dashboard_page",
            )
        else:
            page_name = st.sidebar.radio(
                "Page",
                ai_pages(),
                label_visibility="collapsed",
                key="ai_page",
            )
        if page_name != "Attendance":
            display_mode = st.radio(
                "Show values as",
                ["Counts", "Percentages"],
                horizontal=True,
                key="value_mode",
            )
            value_mode = (
                "percent" if display_mode == "Percentages" else "counts"
            )
        else:
            # All attendance charts show rates, so the counts/percent toggle
            # is hidden and cannot alter the sector chart.
            value_mode = "percent"
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

    if page_name == AGENT_PAGE:
        _render_agent_page(applications, attendance)
        return
    if page_name == SUMMARY_PAGE:
        _render_dashboard_summary_page(
            applications, attendance, years, cohorts, outcomes, sectors, types
        )
        return
    if page_name == HYBRID_PAGE:
        _render_hybrid_page(years, cohorts, outcomes, sectors, types)
        return

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
