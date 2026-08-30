"""Mashroo3i applicants dashboard.

Converted from ``Dashboard_Mashroo3i.ipynb`` into a plain Python application.

Run locally::

    python app.py

Serve with a production WSGI server::

    gunicorn app:server
"""

from __future__ import annotations

import argparse
import base64
import io
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html


# ---------------------------------------------------------------------------
# Global application state
# ---------------------------------------------------------------------------

DF_GLOBAL = None

C_ORANGE = "#FF6B2E"
C_ORANGE_DARK = "#E6531F"
C_ORANGE_LIGHT = "#FF8C42"
C_YELLOW = "#FFC96B"
C_ORANGE_SOFT = "#FFB27A"
C_PEACH = "#FDDCC9"
C_PEACH_LIGHT = "#FFF4EE"
C_GRAY = "#D0D0D0"
C_TEXT = "#4B4038"
CHART_FONT = "'Inter', 'Segoe UI', system-ui, sans-serif"
COHORT_COLORS = {"Arabic": C_YELLOW, "English": C_ORANGE}
AGE_ORDER = ["18-24", "25-34", "35-44", "45+", "Not Specified"]

# Columns the dashboard needs from an uploaded file (kept even if the source
# file contains many more columns).
REQUIRED_COLUMNS = [
    "year",
    "cohort",
    "cohort_id",
    "Sector",
    "outcome_clean",
    "Business Stage",
    "Age Group",
    "applicant_type",
    "team_member_count",
    "employment_status",
    "has_commercial_registration",
    "education",
    "major",
    "nationality",
]

ATTENDANCE_COLUMNS = [
    "sessions_scheduled",
    "team_days_present",
    "team_days_virtual",
    "team_attendance_rate",
    "member_attendance_rate",
    "attendance_member_rows",
    "member_days_present",
    "member_days_virtual",
    "team_size_from_attendance",
]
REQUIRED_COLUMNS.extend(ATTENDANCE_COLUMNS)

MEMBER_ATTENDANCE_COLUMNS = {
    "sessions_scheduled",
    "attendance_member_rows",
    "member_days_present",
    "member_days_virtual",
}

btn_w = {
    "background": "white",
    "color": "#4B4038",
    "border": f"1px solid {C_PEACH}",
    "borderRadius": "12px",
    "padding": "11px 12px",
    "fontWeight": "700",
    "cursor": "pointer",
    "fontSize": "12px",
    "textAlign": "left",
    "transition": "all .18s ease",
    "boxShadow": "0 1px 2px rgba(64, 41, 24, .04)",
}
btn_a = {
    "background": C_ORANGE,
    "color": "white",
    "border": "none",
    "borderRadius": "12px",
    "padding": "11px 12px",
    "fontWeight": "700",
    "cursor": "pointer",
    "fontSize": "12px",
    "textAlign": "left",
    "boxShadow": "0 8px 18px rgba(255, 107, 46, .22)",
}
VALUE_MODE_STYLE = {
    "fontSize": "12px",
    "fontWeight": "700",
    "color": C_TEXT,
}


def _clean_cohort(value):
    """Normalize cohort labels to 'Arabic' / 'English' when recognizable."""
    if pd.isna(value):
        return value
    text = str(value).strip().lower()
    if "arab" in text:
        return "Arabic"
    if "eng" in text:
        return "English"
    return str(value).strip()


def _clean_applicant_type(value):
    """Normalize applicant_type values to 'Individual' / 'Team'."""
    if pd.isna(value):
        return value
    text = str(value).strip().lower().replace("_", " ")
    if "solo" in text or "individual" in text or text == "ind":
        return "Individual"
    if "team" in text or "group" in text:
        return "Team"
    return str(value).strip().replace("_", " ").title()


def _attendance_pct(series):
    """Convert attendance rates to percentages (0-1 values become 0-100)."""
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().any() and values.max() <= 1.0:
        values = values * 100
    return values


def _attendance_summary(df, by):
    """Summarize individual member attendance for one grouping column.

    ``team_attendance_rate`` only measures whether at least one member
    attended a session, so a team with six members and only one attendee can
    appear to have 100% attendance. This helper instead measures the total
    member-days attended against the total member-session opportunities:

    (member_days_present + member_days_virtual)
    / (sessions_scheduled * attendance_member_rows)
    """
    if not MEMBER_ATTENDANCE_COLUMNS.issubset(df.columns) or by not in df.columns:
        return None

    attendance = df.copy()
    attendance["_sessions"] = pd.to_numeric(
        attendance["sessions_scheduled"], errors="coerce"
    ).fillna(0)
    attendance["_members"] = pd.to_numeric(
        attendance["attendance_member_rows"], errors="coerce"
    ).fillna(0)
    attendance["_present"] = pd.to_numeric(
        attendance["member_days_present"], errors="coerce"
    ).fillna(0)
    attendance["_virtual"] = pd.to_numeric(
        attendance["member_days_virtual"], errors="coerce"
    ).fillna(0)

    attendance["_member_opportunities"] = (
        attendance["_sessions"] * attendance["_members"]
    )
    attendance["_member_days_attended"] = (
        attendance["_present"] + attendance["_virtual"]
    )
    attendance["_has_attendance"] = (
        attendance["_member_opportunities"] > 0
    )
    attendance["_group_missing"] = attendance[by].map(_is_missing_group)
    attended = attendance[
        attendance["_has_attendance"] & ~attendance["_group_missing"]
    ].copy()
    missing_count = int(
        (attendance["_has_attendance"] & attendance["_group_missing"]).sum()
    )
    if attended.empty:
        return attended, missing_count

    summary = (
        attended.groupby(by, dropna=False)
        .agg(
            attendance_projects=("_has_attendance", "sum"),
            sessions_scheduled=("_sessions", "sum"),
            member_opportunities=("_member_opportunities", "sum"),
            member_days_attended=("_member_days_attended", "sum"),
        )
        .reset_index()
    )
    summary["member_attendance_rate"] = (
        summary["member_days_attended"] / summary["member_opportunities"] * 100
    )
    return summary, missing_count


def _is_missing_group(value) -> bool:
    """True for NaN, blank, or 'Not Specified'-style group values."""
    if pd.isna(value):
        return True
    text = str(value).strip().lower()
    return text in {"", "nan", "none", "null", "not specified", "blanks", "blank"}


def _age_order(index):
    """Order age-group labels youngest-first, with unknown labels after."""
    known = [age for age in AGE_ORDER if age in index]
    unknown = sorted([age for age in index if age not in AGE_ORDER])
    return known + unknown


def build_layout():
    """Build the complete Dash layout."""
    return html.Div(
        className="app-shell",
        style={
            "fontFamily": "'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif",
            "display": "flex",
            "minHeight": "100vh",
            "background": "#FBF6F2",
        },
        children=[
            html.Div(
                className="sidebar",
                style={
                    "background": "white",
                    "padding": "20px 16px",
                    "display": "flex",
                    "flexDirection": "column",
                    "gap": "8px",
                },
                children=[
                    html.Div(
                        "🏗️ Mashroo3i",
                        className="sidebar-brand",
                        style={
                            "color": C_ORANGE,
                            "fontSize": "22px",
                            "fontWeight": "800",
                            "letterSpacing": "-.3px",
                            "padding": "2px 4px 4px",
                        },
                    ),
                    html.Div(
                        "Applicant & cohort insights",
                        style={
                            "color": "#9B8B82",
                            "fontSize": "11px",
                            "fontWeight": "600",
                            "padding": "0 4px 10px",
                            "marginTop": "-7px",
                        },
                    ),
                    dcc.Upload(
                        id="upload-csv",
                        children=html.Div(
                            ["⬆️ Upload CSV"],
                            style={
                                "textAlign": "center",
                                "fontWeight": "800",
                                "fontSize": "13px",
                                "color": C_ORANGE_DARK,
                            },
                        ),
                        className="upload-zone",
                        style={
                            "width": "100%",
                            "height": "44px",
                            "lineHeight": "44px",
                            "borderWidth": "1.5px",
                            "borderStyle": "dashed",
                            "borderColor": "#F3B38F",
                            "borderRadius": "13px",
                            "background": C_PEACH_LIGHT,
                            "cursor": "pointer",
                        },
                        multiple=False,
                    ),
                    html.Div(id="upload-status"),
                    html.Hr(style={"margin": "10px 0 8px", "border": "0", "borderTop": f"1px solid {C_PEACH}"}),
                    html.Button("1 - Overview", id="btn-p1", n_clicks=0, style=btn_a),
                    html.Button("2 - Applicant Profile", id="btn-p2", n_clicks=0, style=btn_w),
                    html.Button("3 - Sectors & Applicant Type", id="btn-p3", n_clicks=0, style=btn_w),
                    html.Button("4 - Cohort Comparison", id="btn-p4", n_clicks=0, style=btn_w),
                    html.Button("5 - Attendance", id="btn-p5", n_clicks=0, style=btn_w),
                    html.Hr(style={"margin": "10px 0 8px", "border": "0", "borderTop": f"1px solid {C_PEACH}"}),
                    html.Div(
                        "Filters",
                        style={
                            "fontSize": "10px",
                            "fontWeight": "800",
                            "letterSpacing": "1.2px",
                            "textTransform": "uppercase",
                            "color": "#A18F86",
                            "padding": "2px 4px",
                        },
                    ),
                    dcc.Dropdown(id="f-year", options=[], multi=True, placeholder="All Years"),
                    dcc.Dropdown(id="f-cohort", options=[], multi=True, placeholder="All Cohorts"),
                    dcc.Dropdown(id="f-outcome", options=[], multi=True, placeholder="All Outcomes"),
                    dcc.Dropdown(id="f-sector", options=[], multi=True, placeholder="All Sectors"),
                    dcc.Dropdown(id="f-type", options=[], multi=True, placeholder="Individual or Team"),
                    html.Div(
                        "Values",
                        style={
                            "fontSize": "10px",
                            "fontWeight": "800",
                            "letterSpacing": "1.2px",
                            "textTransform": "uppercase",
                            "color": "#A18F86",
                            "padding": "10px 4px 2px",
                        },
                    ),
                    dcc.RadioItems(
                        id="value-mode",
                        options=[
                            {"label": "Counts", "value": "counts"},
                            {"label": "%", "value": "percent"},
                        ],
                        value="counts",
                        inline=True,
                        labelStyle={"marginRight": "14px"},
                        style=VALUE_MODE_STYLE,
                    ),
                ],
            ),
            html.Div(
                className="main-area",
                style={
                    "flex": "1",
                    "minWidth": "0",
                    "background": "#FBF6F2",
                    "boxSizing": "border-box",
                    "display": "flex",
                    "flexDirection": "column",
                },
                children=[
                    dcc.Loading(
                        className="page-loader",
                        type="circle",
                        color=C_ORANGE,
                        style={"flex": "1", "minWidth": "0", "display": "flex", "flexDirection": "column"},
                        children=html.Div(
                            id="page-content",
                            className="page-content",
                            style={"width": "100%", "minWidth": "0"},
                        ),
                    )
                ],
            ),
            dcc.Store(id="current-page", data="page1"),
        ],
    )


app = Dash(__name__)
app.title = "Mashroo3i Dashboard"
app.layout = build_layout()
server = app.server


@app.callback(
    Output("current-page", "data"),
    Output("btn-p1", "style"),
    Output("btn-p2", "style"),
    Output("btn-p3", "style"),
    Output("btn-p4", "style"),
    Output("btn-p5", "style"),
    Output("value-mode", "style"),
    Input("btn-p1", "n_clicks"),
    Input("btn-p2", "n_clicks"),
    Input("btn-p3", "n_clicks"),
    Input("btn-p4", "n_clicks"),
    Input("btn-p5", "n_clicks"),
)
def switch_page(b1, b2, b3, b4, b5):
    from dash import ctx

    if not ctx.triggered:
        return "page1", btn_a, btn_w, btn_w, btn_w, btn_w, VALUE_MODE_STYLE
    pages = {
        "btn-p1": "page1",
        "btn-p2": "page2",
        "btn-p3": "page3",
        "btn-p4": "page4",
        "btn-p5": "page5",
    }
    page = pages.get(ctx.triggered_id, "page1")
    return (
        page,
        btn_a if page == "page1" else btn_w,
        btn_a if page == "page2" else btn_w,
        btn_a if page == "page3" else btn_w,
        btn_a if page == "page4" else btn_w,
        btn_a if page == "page5" else btn_w,
        {"display": "none"} if page == "page5" else VALUE_MODE_STYLE,
    )


@app.callback(
    Output("upload-status", "children"),
    Output("f-year", "options"),
    Output("f-cohort", "options"),
    Output("f-outcome", "options"),
    Output("f-sector", "options"),
    Output("f-type", "options"),
    Input("upload-csv", "contents"),
    State("upload-csv", "filename"),
)
def handle_upload(contents, filename):
    global DF_GLOBAL
    if contents is None:
        return "", [], [], [], [], []
    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    if filename.lower().endswith(".csv"):
        df_up = pd.read_csv(io.BytesIO(decoded), encoding="utf-8-sig")
    else:
        df_up = pd.read_excel(io.BytesIO(decoded))
    keep = [
        c for c in REQUIRED_COLUMNS + ["individual_or_team"] if c in df_up.columns
    ]
    DF_GLOBAL = df_up[keep].copy()

    # Years must be integers, never floats.
    if "year" in DF_GLOBAL.columns:
        DF_GLOBAL["year"] = pd.to_numeric(DF_GLOBAL["year"], errors="coerce").astype("Int64")
    if "cohort" in DF_GLOBAL.columns:
        DF_GLOBAL["cohort"] = DF_GLOBAL["cohort"].map(_clean_cohort)
    # Real data uses individual_or_team, while the dashboard filters on
    # applicant_type. Derive the alias when missing.
    if "applicant_type" not in DF_GLOBAL.columns and "individual_or_team" in DF_GLOBAL.columns:
        DF_GLOBAL["applicant_type"] = DF_GLOBAL["individual_or_team"].map(_clean_applicant_type)
    if "applicant_type" in DF_GLOBAL.columns:
        DF_GLOBAL["applicant_type"] = DF_GLOBAL["applicant_type"].map(_clean_applicant_type)

    msg = html.Div(
        f"✅ {len(DF_GLOBAL)} rows",
        className="upload-status",
        style={
            "background": C_PEACH_LIGHT,
            "borderRadius": "10px",
            "padding": "8px 10px",
            "border": f"1px solid {C_PEACH}",
            "fontSize": "11px",
            "fontWeight": "700",
            "textAlign": "center",
            "color": C_ORANGE_DARK,
        },
    )
    year_opts = [
        {"label": str(int(y)), "value": int(y)}
        for y in sorted(DF_GLOBAL["year"].dropna().unique())
    ]
    cohort_opts = [
        {"label": str(c), "value": c} for c in DF_GLOBAL["cohort"].dropna().unique()
    ]
    outcome_opts = [
        {"label": str(o), "value": o} for o in DF_GLOBAL["outcome_clean"].dropna().unique()
    ]
    sector_opts = [
        {"label": str(s), "value": s} for s in DF_GLOBAL["Sector"].dropna().unique()
    ]
    type_opts = [
        {"label": str(t), "value": t} for t in DF_GLOBAL["applicant_type"].dropna().unique()
    ]
    return msg, year_opts, cohort_opts, outcome_opts, sector_opts, type_opts


def _bar_fig(x, y, orientation="v", text=None, color=C_ORANGE, radius=14):
    """Compact single-color bar chart that fills its container."""
    trace = {
        "marker": dict(color=color, cornerradius=radius),
    }
    if text is not None:
        trace["text"] = text
        trace["textposition"] = "outside"
        trace["cliponaxis"] = False
    fig = go.Figure(go.Bar(x=x, y=y, orientation=orientation, **trace))
    if orientation == "h":
        fig.update_layout(
            margin=dict(l=10, r=80, t=30, b=10),
            xaxis=dict(title="", visible=False, tickangle=0),
            yaxis=dict(
                title="",
                showgrid=False,
                tickangle=0,
                tickfont=dict(family=CHART_FONT, size=11, color=C_TEXT),
            ),
        )
    else:
        label_texts = [str(label) for label in x] if x is not None else []
        max_label_len = max((len(text) for text in label_texts), default=0)
        fig.update_layout(
            margin=dict(l=10, r=60, t=32, b=10),
            xaxis=dict(
                title="",
                showgrid=False,
                tickangle=(
                    -90
                    if (len(label_texts) > 5 and max_label_len > 3)
                    or max_label_len > 20
                    else 0
                ),
                tickfont=dict(family=CHART_FONT, size=11, color=C_TEXT),
            ),
            yaxis=dict(title="", visible=False, tickangle=0),
        )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family=CHART_FONT,
            size=11,
            color=C_TEXT,
        ),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor=C_PEACH,
            font=dict(family=CHART_FONT, size=11, color="#2D2926"),
        ),
        showlegend=False,
    )
    return fig


_MISSING_LABELS = {
    "not specified": "Not Specified",
    "blanks": "Blanks",
}


def _split_missing(counts: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    """Separate 'Not Specified'/'Blanks' counts from a distribution series."""
    visible = []
    missing: dict[str, int] = {}
    for label, value in counts.items():
        canonical = _MISSING_LABELS.get(str(label).strip().lower())
        if canonical is not None:
            missing[canonical] = missing.get(canonical, 0) + int(value)
        else:
            visible.append((label, value))
    return pd.Series(dict(visible), name=counts.name), missing


def _clean_slash_label(value):
    """Keep the descriptive part of labels like 'University / Higher Education'."""
    if isinstance(value, str) and "/" in value:
        cleaned = value.split("/")[-1].strip()
        return cleaned if cleaned else value
    return value


def _missing_note(
    missing: dict[str, int],
    as_percent: bool = False,
    full_total: int | None = None,
    label: str | None = None,
) -> str | None:
    """Build the side-note text for excluded blanks/not-specified counts."""
    if not missing:
        return None
    count = sum(missing.values())
    if count <= 0:
        return None
    if label:
        label = label
    elif len(missing) > 1:
        label = "blank or not specified"
    else:
        label = next(iter(missing)).lower()
    if as_percent and full_total:
        return f"{count / full_total * 100:.1f}% {label}"
    return f"{count} {label}"


def _add_missing_note(fig, note, bottom=False):
    """Add the missing-category note outside the plot area."""
    if not note:
        return fig
    if bottom:
        fig.update_layout(
            margin=dict(b=max(40, fig.layout.margin.b or 0)),
        )
        y, yanchor, yshift = 0, "top", -6
    else:
        fig.update_layout(
            margin=dict(t=max(40, fig.layout.margin.t or 0)),
        )
        y, yanchor, yshift = 1, "bottom", 6
    fig.add_annotation(
        text=note,
        xref="paper",
        yref="paper",
        x=1,
        y=y,
        xanchor="right",
        yanchor=yanchor,
        showarrow=False,
        align="right",
        xshift=-6,
        yshift=yshift,
        font=dict(family=CHART_FONT, size=10, color="#9B8B82"),
    )
    return fig


def _dist_fig(
    labels,
    values,
    orientation="v",
    as_percent=False,
    total=None,
    precision=1,
    note=None,
    note_bottom=None,
):
    """Single-variable distribution bar chart showing counts or % of total."""
    values = pd.Series(values, index=labels, dtype=float)
    if total is None:
        total = float(values.sum())
    if as_percent and total:
        shown = (values / total * 100).round(precision)
        text = [f"{value:.{precision}f}%" for value in shown]
    else:
        shown = values.astype(int)
        text = [str(int(value)) for value in shown]
    if orientation == "h":
        fig = _bar_fig(shown.values, labels, orientation="h", text=text)
    else:
        fig = _bar_fig(labels, shown.values, orientation="v", text=text)
    if note_bottom is None:
        note_bottom = orientation == "h"
    return _add_missing_note(fig, note, bottom=note_bottom)


def _order_legend_colors(fig):
    """Show orange legend entries before red ones in multi-color legends."""
    color_order = {
        C_ORANGE: 0,
        C_ORANGE_LIGHT: 1,
        C_ORANGE_SOFT: 2,
        C_ORANGE_DARK: 10,
        C_YELLOW: 20,
        C_GRAY: 30,
    }
    for trace in fig.data:
        if not getattr(trace, "showlegend", True):
            continue
        color = getattr(getattr(trace, "marker", None), "color", None)
        if isinstance(color, str) and color in color_order:
            trace.legendrank = color_order[color]


def _multi_h_bar_fig(
    df,
    y_col,
    color_col,
    text_col,
    color_map,
    category_orders,
    legend_title,
    radius=14,
    barmode="stack",
    textposition="inside",
):
    """Horizontal multi-series bar chart with consistent styling."""
    fig = px.bar(
        df,
        y=y_col,
        x="Total",
        color=color_col,
        orientation="h",
        barmode=barmode,
        color_discrete_map=color_map,
        text=text_col,
        category_orders=category_orders,
    )
    fig.update_traces(
        marker=dict(cornerradius=radius),
        textposition=textposition,
        cliponaxis=textposition == "outside",
    )
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=list(category_orders[y_col]),
    )
    _order_legend_colors(fig)
    fig.update_layout(
        margin=dict(l=10, r=60, t=40, b=10),
        legend_title_text=legend_title,
        legend=dict(
            orientation="h",
            y=1.2,
            x=0.5,
            xanchor="center",
            font=dict(family=CHART_FONT, size=10),
            title=dict(side="top center"),
        ),
        xaxis=dict(title="", visible=False, tickangle=0),
        yaxis=dict(
            title="",
            showgrid=False,
            tickangle=0,
            tickfont=dict(family=CHART_FONT, size=11, color=C_TEXT),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=CHART_FONT, size=11, color=C_TEXT),
    )
    return fig


def _attendance_bar_fig(
    summary,
    by,
    horizontal=False,
    max_items=None,
    sort_by="value",
    note=None,
):
    """Build a member-attendance rate (%) chart with an optional note."""
    if summary is None or summary.empty:
        return _bar_fig([], [], orientation="h" if horizontal else "v")

    if sort_by == "label":
        summary = summary.sort_values(by)
    indexed = summary.set_index(by)
    values = indexed["member_attendance_rate"].round(1)
    if max_items:
        values = values.sort_values().tail(max_items)
    elif horizontal:
        values = values.sort_values()
    elif sort_by != "label":
        values = values.sort_values(ascending=False)

    labels = [f"{value}%" for value in values.values]

    if horizontal:
        fig = _bar_fig(
            values.values,
            values.index,
            orientation="h",
            text=labels,
        )
    else:
        fig = _bar_fig(
            values.index,
            values.values,
            orientation="v",
            text=labels,
        )
        fig.update_xaxes(type="category")
    return _add_missing_note(fig, note, bottom=horizontal)


def _team_size_bucket(value):
    """Map a raw team-member count to a display category (1 is excluded)."""
    if pd.isna(value):
        return "Blanks"
    text = str(value).strip()
    if text in {"", "-", "nan", "None", "null"}:
        return "Blanks"
    if text == "5+":
        return "5+"
    try:
        number = float(text)
    except (TypeError, ValueError):
        return "Blanks"
    if pd.isna(number) or number <= 0:
        return "Blanks"
    if number <= 1:
        return None
    if number > 5:
        return "5+"
    return str(int(number))


def _team_size_fig(df, as_percent=False):
    """Build a team-size distribution with 2-5, 5+, and a Blanks side note."""
    if "team_member_count" in df.columns:
        size_col = "team_member_count"
    elif "team_size_from_attendance" in df.columns:
        size_col = "team_size_from_attendance"
    elif "attendance_member_rows" in df.columns:
        size_col = "attendance_member_rows"
    else:
        return _bar_fig([], [])

    buckets = {"2": 0, "3": 0, "4": 0, "5": 0, "5+": 0, "Blanks": 0}
    for value in df[size_col]:
        bucket = _team_size_bucket(value)
        if bucket is not None:
            buckets[bucket] += 1

    labels = ["2", "3", "4", "5", "5+"]
    counts = pd.Series(buckets).reindex(labels).astype(int)
    blanks = int(buckets["Blanks"])
    if as_percent:
        total_visible = float(counts.sum())
        values = (
            (counts / total_visible * 100).round(1) if total_visible else counts
        )
        text = [f"{value:.1f}%" for value in values]
    else:
        values = counts
        text = [str(int(value)) for value in counts.values]
    fig = _bar_fig(
        labels,
        values.values,
        text=text,
    )
    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=["2", "3", "4", "5", "5+"],
    )
    note = _missing_note(
        {"Blanks": blanks},
        as_percent=as_percent,
        full_total=counts.sum() + blanks,
        label="blank or not specified",
    )
    return _add_missing_note(fig, note)


def _card(title, fig, flex="1", height=320):
    """Modern card with a consistent chart height, so nothing gets clipped."""
    setattr(fig, "_dashboard_height", height)
    return html.Div(
        className="chart-card",
        style={
            "flex": flex,
            "minWidth": "0",
            "background": "white",
            "borderRadius": "16px",
            "border": f"1px solid {C_PEACH}",
            "boxShadow": "0 10px 24px rgba(99, 51, 18, .06)",
            "overflow": "hidden",
            "display": "flex",
            "flexDirection": "column",
        },
        children=[
            html.Div(
                title,
                className="chart-card-title",
                style={"flexShrink": "0"},
            ),
            html.Div(
                className="chart-card-body",
                style={"flex": "1", "minHeight": "0"},
                children=[
                    dcc.Graph(
                        figure=fig,
                        style={"height": f"{height}px", "width": "100%"},
                        config={"displayModeBar": False},
                    )
                ],
            ),
        ],
    )


def _row(*cards):
    """One responsive chart row."""
    return html.Div(
        className="chart-row",
        style={"display": "flex", "gap": "16px", "minWidth": "0"},
        children=list(cards),
    )


def _grid_row(*cards):
    """One chart row with equal-width cards in a CSS grid."""
    return html.Div(
        className=f"chart-row chart-grid chart-cols-{len(cards)}",
        style={"display": "grid", "gap": "16px", "minWidth": "0"},
        children=list(cards),
    )


def _page_shell(title, kpis, *rows, subtitle="Live insights from the uploaded dataset"):
    return [
        html.Div(
            className="page-heading",
            style={"padding": "0 2px 16px"},
            children=[
                html.Div(title, className="page-title"),
                html.Div(
                    subtitle,
                    className="page-subtitle",
                ),
            ],
        ),
        kpis,
        html.Div(
            className="chart-rows",
            style={
                "display": "flex",
                "flexDirection": "column",
                "gap": "16px",
                "minWidth": "0",
            },
            children=list(rows),
        ),
    ]


def _empty_state(icon, title, copy=None):
    """Shared empty state used before upload and when filters match nothing."""
    return html.Div(
        className="empty-state",
        children=[
            html.Div(icon, className="empty-icon"),
            html.Div(title, className="empty-title"),
            html.Div(copy or "", className="empty-copy") if copy else None,
        ],
    )


@app.callback(
    Output("page-content", "children"),
    Input("current-page", "data"),
    # Re-render immediately after an upload by watching the upload status.
    Input("upload-status", "children"),
    Input("f-year", "value"),
    Input("f-cohort", "value"),
    Input("f-outcome", "value"),
    Input("f-sector", "value"),
    Input("f-type", "value"),
    Input("value-mode", "value"),
)
def update_page(
    page,
    _upload_trigger,
    years,
    cohorts,
    outcomes,
    sectors,
    types,
    value_mode=None,
    df_input=None,
):
    global DF_GLOBAL
    dff = DF_GLOBAL if df_input is None else df_input
    if dff is None:
        return [
            _empty_state(
                "📤",
                "Upload your dataset to get started",
                "Drop a CSV or Excel file into the upload area to unlock the dashboard.",
            )
        ]

    if years:
        dff = dff[dff["year"].isin(years)]
    if cohorts:
        dff = dff[dff["cohort"].isin(cohorts)]
    if outcomes:
        dff = dff[dff["outcome_clean"].isin(outcomes)]
    if sectors:
        dff = dff[dff["Sector"].isin(sectors)]
    if types:
        dff = dff[dff["applicant_type"].isin(types)]
    if len(dff) == 0:
        return [
            _empty_state(
                "🔍",
                "No data for this filter",
                "Try removing one or more filters to see matching applicant records.",
            )
        ]

    as_percent = value_mode == "percent"
    total = len(dff)
    accepted = len(dff[dff["outcome_clean"] == "Accepted"])
    rate = round(accepted / total * 100, 1) if total else 0
    bah_rate = (
        round(
            len(
                dff[dff["nationality"].astype(str).str.contains("bahrain", case=False, na=False)]
            )
            / total
            * 100,
            1,
        )
        if total
        else 0
    )

    kpis = html.Div(
        className="kpi-grid",
        children=[
            html.Div(
                className="kpi-card",
                style={"padding": "14px 16px", "display": "flex", "flexDirection": "column", "gap": "7px"},
                children=[
                    html.Div("👥", className="kpi-icon"),
                    html.Div(f"{total}", className="kpi-value"),
                    html.Div("Total Applicants", className="kpi-label"),
                ],
            ),
            html.Div(
                className="kpi-card",
                style={"padding": "14px 16px", "display": "flex", "flexDirection": "column", "gap": "7px"},
                children=[
                    html.Div("✅", className="kpi-icon"),
                    html.Div(f"{accepted}", className="kpi-value"),
                    html.Div("Accepted", className="kpi-label"),
                ],
            ),
            html.Div(
                className="kpi-card",
                style={"padding": "14px 16px", "display": "flex", "flexDirection": "column", "gap": "7px"},
                children=[
                    html.Div("📈", className="kpi-icon"),
                    html.Div(f"{rate}%", className="kpi-value"),
                    html.Div("Acceptance Rate", className="kpi-label"),
                ],
            ),
            html.Div(
                className="kpi-card",
                style={"padding": "14px 16px", "display": "flex", "flexDirection": "column", "gap": "7px"},
                children=[
                    html.Img(
                        src="https://flagcdn.com/w80/bh.png",
                        className="kpi-icon",
                        style={"width": "32px", "height": "21px", "borderRadius": "4px", "objectFit": "cover"},
                    ),
                    html.Div(f"{bah_rate}%", className="kpi-value"),
                    html.Div("Bahraini Nationals", className="kpi-label"),
                ],
            ),
        ],
    )

    if page == "page1":
        df_yc = dff.groupby(["year", "cohort"]).size().reset_index(name="Total")
        fig_y = px.bar(
            df_yc,
            x="year",
            y="Total",
            color="cohort",
            barmode="group",
            color_discrete_map=COHORT_COLORS,
            text="Total",
            category_orders={"cohort": ["Arabic", "English"]},
        )
        fig_y.update_traces(marker=dict(cornerradius=12), textposition="outside", cliponaxis=False)
        year_totals = dff.groupby("year").size().reset_index(name="Total")
        max_total = int(year_totals["Total"].max())
        fig_y.update_layout(
            margin=dict(l=10, r=40, t=40, b=10),
            legend_title_text="Cohort",
            legend=dict(
                orientation="h",
                y=1.2,
                x=0.5,
                xanchor="center",
                font=dict(family=CHART_FONT, size=10),
                title=dict(side="top center"),
            ),
            xaxis=dict(
                title="",
                showgrid=False,
                type="category",
                categoryarray=sorted(dff["year"].unique()),
                tickangle=0,
                tickfont=dict(family=CHART_FONT, size=11, color=C_TEXT),
            ),
            yaxis=dict(title="", visible=False, range=[0, max_total * 1.22]),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family=CHART_FONT, size=11, color=C_TEXT),
        )
        fig_y.add_trace(
            go.Scatter(
                x=[int(year) for year in year_totals["year"]],
                y=[max_total * 1.10 for _ in year_totals["year"]],
                mode="text",
                text=[
                    f"{int(year)}: {int(total)}"
                    for year, total in zip(year_totals["year"], year_totals["Total"])
                ],
                textfont=dict(
                    family=CHART_FONT, size=13, color=C_TEXT, weight="bold"
                ),
                textposition="middle center",
                showlegend=False,
                hoverinfo="skip",
            )
        )

        cnt_type, missing_type = _split_missing(dff["applicant_type"].value_counts())
        cnt_type = cnt_type.sort_values(ascending=False)
        note_type = _missing_note(missing_type, as_percent, total)
        fig_type = _dist_fig(
            cnt_type.index,
            cnt_type.values,
            as_percent=as_percent,
            note=note_type,
        )

        cnt_nat, missing_nat = _split_missing(dff["nationality"].value_counts())
        cnt_nat = cnt_nat.head(5).sort_values()
        note_nat = _missing_note(missing_nat, as_percent, total)
        fig_nat = _dist_fig(
            cnt_nat.index,
            cnt_nat.values,
            orientation="h",
            as_percent=as_percent,
            note=note_nat,
        )
        fig_team_size = _team_size_fig(dff, as_percent=as_percent)

        return _page_shell(
            "Overview",
            kpis,
            _grid_row(
                _card("Applications by Year & Cohort", fig_y),
            ),
            _grid_row(
                _card("Applicant Type Breakdown", fig_type),
                _card("Top Nationalities", fig_nat),
            ),
            _grid_row(
                _card("Team Member Size", fig_team_size),
            ),
        )

    elif page == "page2":
        cnt_stage, missing_stage = _split_missing(dff["Business Stage"].value_counts())
        cnt_stage = cnt_stage.sort_values(ascending=False)
        note_stage = _missing_note(missing_stage, as_percent, total)
        fig_stage = _dist_fig(
            cnt_stage.index,
            cnt_stage.values,
            as_percent=as_percent,
            note=note_stage,
        )

        cnt_age, missing_age = _split_missing(dff["Age Group"].value_counts())
        cnt_age = cnt_age.reindex(_age_order(dff["Age Group"].dropna().unique())).dropna()
        note_age = _missing_note(missing_age, as_percent, total)
        fig_age = _dist_fig(
            cnt_age.index,
            cnt_age.values,
            as_percent=as_percent,
            note=note_age,
        )

        cnt_out, missing_out = _split_missing(dff["outcome_clean"].value_counts())
        cnt_out = cnt_out.sort_values(ascending=False)
        note_out = _missing_note(missing_out, as_percent, total)
        fig_out = _dist_fig(
            cnt_out.index,
            cnt_out.values,
            as_percent=as_percent,
            note=note_out,
        )

        cnt_cr, missing_cr = _split_missing(dff["has_commercial_registration"].value_counts())
        cnt_cr = cnt_cr.sort_values(ascending=False)
        note_cr = _missing_note(missing_cr, as_percent, total)
        fig_cr = _dist_fig(
            cnt_cr.index,
            cnt_cr.values,
            as_percent=as_percent,
            note=note_cr,
        )

        cnt_emp, missing_emp = _split_missing(dff["employment_status"].value_counts())
        cnt_emp = cnt_emp.sort_values(ascending=False)
        note_emp = _missing_note(missing_emp, as_percent, total)
        fig_emp = _dist_fig(
            cnt_emp.index,
            cnt_emp.values,
            as_percent=as_percent,
            note=note_emp,
        )

        cnt_edu, missing_edu = _split_missing(
            dff["education"].map(_clean_slash_label).value_counts()
        )
        cnt_edu = cnt_edu.sort_values().tail(6)
        note_edu = _missing_note(missing_edu, as_percent, total)
        fig_edu = _dist_fig(
            cnt_edu.index,
            cnt_edu.values,
            orientation="h",
            as_percent=as_percent,
            note=note_edu,
        )

        cnt_major, missing_major = _split_missing(
            dff["major"].map(_clean_slash_label).value_counts()
        )
        cnt_major = cnt_major.sort_values().tail(6)
        note_major = _missing_note(missing_major, as_percent, total)
        fig_major = _dist_fig(
            cnt_major.index,
            cnt_major.values,
            orientation="h",
            as_percent=as_percent,
            note=note_major,
        )

        return _page_shell(
            "Applicant Profile",
            kpis,
            _row(
                _card("Business Stage", fig_stage),
                _card("Age Group", fig_age),
                _card("Outcome", fig_out),
            ),
            _row(
                _card("Commercial Registration", fig_cr),
                _card("Employment Status", fig_emp),
                _card("Education", fig_edu),
            ),
            _row(_card("Major", fig_major)),
        )

    elif page == "page3":
        df_type_out = dff.groupby(["applicant_type", "outcome_clean"]).size().reset_index(name="Total")
        fig_type_out = px.bar(
            df_type_out,
            y="applicant_type",
            x="Total",
            color="outcome_clean",
            orientation="h",
            barmode="stack",
            color_discrete_map={
                "Accepted": C_ORANGE_LIGHT,
                "Rejected": C_ORANGE_DARK,
                "Not Specified": C_GRAY,
            },
            text="Total",
        )
        fig_type_out.update_traces(marker=dict(cornerradius=14), textposition="inside")
        _order_legend_colors(fig_type_out)
        fig_type_out.update_layout(
            margin=dict(l=10, r=30, t=40, b=10),
            legend_title_text="Outcome",
            legend=dict(
                orientation="h",
                y=1.2,
                x=0.5,
                xanchor="center",
                font=dict(family=CHART_FONT, size=10),
                title=dict(side="top center"),
            ),
            xaxis=dict(title="", visible=False, tickangle=0),
            yaxis=dict(
                title="",
                showgrid=False,
                tickangle=0,
                tickfont=dict(family=CHART_FONT, size=11, color=C_TEXT),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family=CHART_FONT, size=11, color=C_TEXT),
        )

        cnt_sec, missing_sec = _split_missing(dff["Sector"].value_counts())
        top_sec_desc = cnt_sec.sort_values(
            ascending=False, kind="mergesort"
        ).head(5)
        top_sec_asc = top_sec_desc.sort_values()
        top_sec = top_sec_desc.index
        note_sec = _missing_note(missing_sec, as_percent, total)
        fig_sec = _dist_fig(
            top_sec_asc.index,
            top_sec_asc.values,
            orientation="h",
            as_percent=as_percent,
            note=note_sec,
            note_bottom=True,
        )

        cnt_y_type = dff.groupby(["year", "applicant_type"]).size().reset_index(name="Total")
        fig_y_type = px.bar(
            cnt_y_type,
            x="year",
            y="Total",
            color="applicant_type",
            barmode="group",
            color_discrete_map={"Individual": C_ORANGE_DARK, "Team": C_ORANGE_LIGHT},
            text="Total",
        )
        fig_y_type.update_traces(marker=dict(cornerradius=12), textposition="outside", cliponaxis=False)
        _order_legend_colors(fig_y_type)
        fig_y_type.update_layout(
            margin=dict(l=10, r=50, t=40, b=10),
            legend_title_text="Applicant Type",
            legend=dict(
                orientation="h",
                y=1.2,
                x=0.5,
                xanchor="center",
                font=dict(family=CHART_FONT, size=10),
                title=dict(side="top center"),
            ),
            xaxis=dict(
                title="",
                showgrid=False,
                tickangle=0,
                tickfont=dict(family=CHART_FONT, size=11, color=C_TEXT),
                tickformat="d",
            ),
            yaxis=dict(title="", visible=False),
            bargap=0.4,
            bargroupgap=0.25,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family=CHART_FONT, size=11, color=C_TEXT),
        )

        df_sec_out = (
            dff[dff["Sector"].isin(top_sec)]
            .groupby(["Sector", "outcome_clean"], observed=True)
            .size()
            .reset_index(name="Total")
        )
        if as_percent:
            sector_total = df_sec_out.groupby("Sector")["Total"].transform("sum")
            df_sec_out["Total"] = (df_sec_out["Total"] / sector_total * 100).round(1)
            df_sec_out["Text"] = df_sec_out["Total"].astype(str) + "%"
        else:
            df_sec_out["Text"] = df_sec_out["Total"].astype(str)

        fig_sec_out = px.bar(
            df_sec_out,
            y="Sector",
            x="Total",
            color="outcome_clean",
            orientation="h",
            barmode="stack",
            color_discrete_map={
                "Accepted": C_ORANGE_LIGHT,
                "Rejected": C_ORANGE_DARK,
                "Not Specified": C_GRAY,
            },
            text="Text",
            category_orders={
                "Sector": list(top_sec_asc.index),
                "outcome_clean": ["Accepted", "Rejected", "Not Specified"],
            },
        )
        fig_sec_out.update_traces(marker=dict(cornerradius=14), textposition="inside")
        fig_sec_out.update_yaxes(
            categoryorder="array",
            categoryarray=list(top_sec_asc.index),
        )
        _order_legend_colors(fig_sec_out)
        fig_sec_out.update_layout(
            margin=dict(l=10, r=30, t=40, b=10),
            legend_title_text="Outcome",
            legend=dict(
                orientation="h",
                y=1.2,
                x=0.5,
                xanchor="center",
                font=dict(family=CHART_FONT, size=10),
                title=dict(side="top center"),
            ),
            xaxis=dict(title="", visible=False, tickangle=0),
            yaxis=dict(
                title="",
                showgrid=False,
                tickangle=0,
                tickfont=dict(family=CHART_FONT, size=11, color=C_TEXT),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family=CHART_FONT, size=11, color=C_TEXT),
        )
        fig_sec_out = _add_missing_note(fig_sec_out, note_sec, bottom=True)

        acc_by_year = (
            dff[dff["outcome_clean"] == "Accepted"]
            .groupby("year")
            .size()
            .reset_index(name="Accepted")
        )
        fig_acc = go.Figure(
            go.Scatter(
                x=acc_by_year["year"],
                y=acc_by_year["Accepted"],
                mode="lines+markers+text",
                text=acc_by_year["Accepted"],
                textposition="top center",
                line=dict(color=C_ORANGE, width=3),
                marker=dict(size=9, color=C_ORANGE),
            )
        )
        acc_years = sorted(acc_by_year["year"].unique())
        fig_acc.update_layout(
            margin=dict(l=10, r=30, t=35, b=10),
            xaxis=dict(
                title="",
                showgrid=False,
                type="category",
                categoryarray=acc_years,
                tickangle=0,
                tickfont=dict(family=CHART_FONT, size=11, color=C_TEXT),
            ),
            yaxis=dict(title="", showgrid=False, visible=False, zeroline=False),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family=CHART_FONT, size=11, color=C_TEXT),
        )

        return _page_shell(
            "Sectors & Applicant Type",
            kpis,
            _grid_row(
                _card("Applicant Type vs Outcome", fig_type_out, height=360),
                _card("Sector vs Outcome", fig_sec_out, height=360),
            ),
            _grid_row(
                _card("Top Sectors", fig_sec, height=360),
                _card("Applicant Type Over Years", fig_y_type, height=360),
            ),
            _grid_row(
                _card("Accepted Applications Over Years", fig_acc, height=720),
            ),
        )

    elif page == "page4":
        cohort_totals = (
            dff.groupby("cohort", observed=True)
            .size()
            .sort_values(ascending=False, kind="mergesort")
        )
        cohort_order_visual = list(cohort_totals.sort_values().index)

        df_cohort_outcome = (
            dff.groupby(["cohort", "outcome_clean"], observed=True)
            .size()
            .reset_index(name="Total")
        )
        if as_percent:
            cohort_total = df_cohort_outcome.groupby("cohort")["Total"].transform("sum")
            df_cohort_outcome["Total"] = (
                df_cohort_outcome["Total"] / cohort_total * 100
            ).round(1)
            df_cohort_outcome["Text"] = df_cohort_outcome["Total"].astype(str) + "%"
        else:
            df_cohort_outcome["Text"] = df_cohort_outcome["Total"].astype(str)
        fig_cohort_outcome = _multi_h_bar_fig(
            df_cohort_outcome,
            y_col="cohort",
            color_col="outcome_clean",
            text_col="Text",
            color_map={
                "Accepted": C_ORANGE_LIGHT,
                "Rejected": C_ORANGE_DARK,
                "Not Specified": C_GRAY,
            },
            category_orders={
                "cohort": cohort_order_visual,
                "outcome_clean": ["Accepted", "Rejected", "Not Specified"],
            },
            legend_title="Outcome",
        )

        cohort_colors = {
            cohort: C_ORANGE_LIGHT if index == 0 else C_ORANGE_DARK
            for index, cohort in enumerate(cohort_order_visual)
        }
        df_cohort_size = pd.DataFrame(
            {
                "cohort": cohort_order_visual,
                "Total": [
                    int(cohort_totals[cohort])
                    for cohort in cohort_order_visual
                ],
            }
        )
        if as_percent:
            df_cohort_size["Total"] = (
                df_cohort_size["Total"] / total * 100
            ).round(1)
            df_cohort_size["Text"] = df_cohort_size["Total"].astype(str) + "%"
        else:
            df_cohort_size["Text"] = df_cohort_size["Total"].astype(str)
        fig_cohort_size = _multi_h_bar_fig(
            df_cohort_size,
            y_col="cohort",
            color_col="cohort",
            text_col="Text",
            color_map=cohort_colors,
            category_orders={"cohort": cohort_order_visual},
            legend_title="Cohort",
            barmode="stack",
            textposition="inside",
        )

        cohort_size_col = (
            "cohort_id"
            if "cohort_id" in dff.columns and dff["cohort_id"].notna().any()
            else "cohort"
        )
        cnt_cohort, missing_cohort = _split_missing(
            dff[cohort_size_col].value_counts()
        )
        cnt_cohort = cnt_cohort.sort_index()
        note_cohort = _missing_note(missing_cohort, as_percent, total)
        fig_cohort = _dist_fig(
            cnt_cohort.index,
            cnt_cohort.values,
            as_percent=as_percent,
            note=note_cohort,
        )
        fig_cohort.update_xaxes(type="category")

        df_type_cohort = dff.groupby(["cohort", "applicant_type"]).size().reset_index(name="Total")
        fig_type_cohort = px.bar(
            df_type_cohort,
            x="cohort",
            y="Total",
            color="applicant_type",
            barmode="group",
            text="Total",
            color_discrete_map={"Individual": C_ORANGE_DARK, "Team": C_ORANGE_LIGHT},
            category_orders={"cohort": ["Arabic", "English"]},
        )
        fig_type_cohort.update_traces(marker=dict(cornerradius=10), textposition="outside", cliponaxis=False)
        _order_legend_colors(fig_type_cohort)
        fig_type_cohort.update_layout(
            margin=dict(l=10, r=50, t=45, b=10),
            legend_title_text="Applicant Type",
            bargap=0.4,
            bargroupgap=0.25,
            legend=dict(
                orientation="h",
                y=1.35,
                x=0.5,
                xanchor="center",
                font=dict(family=CHART_FONT, size=10),
                title=dict(side="top center"),
            ),
            xaxis=dict(
                title="",
                showgrid=False,
                tickangle=0,
                tickfont=dict(family=CHART_FONT, size=11, color=C_TEXT),
            ),
            yaxis=dict(title="", visible=False),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family=CHART_FONT, size=11, color=C_TEXT),
        )

        return _page_shell(
            "Cohort Comparison",
            kpis,
            _grid_row(
                _card("Language Cohort vs Outcome", fig_cohort_outcome),
                _card("Cohort Size by Language Cohort", fig_cohort_size),
            ),
            _grid_row(
                _card("Cohort Size", fig_cohort),
                _card("Applicant Type by Language Cohort", fig_type_cohort),
            ),
        )

    else:  # page5 - Attendance
        if not any(col in dff.columns for col in ATTENDANCE_COLUMNS):
            return [
                html.Div(
                    "Attendance data is not included in the uploaded file",
                    style={
                        "flex": "1",
                        "background": "white",
                        "borderRadius": "20px",
                        "display": "flex",
                        "alignItems": "center",
                        "justifyContent": "center",
                        "fontWeight": "800",
                    },
                )
            ]

        if "member_attendance_rate" in dff.columns:
            attendance_projects = int(
                dff["member_attendance_rate"].notna().sum()
            )
        else:
            attendance_projects = 0
        attendance_note = (
            f"Member attendance is shown for {attendance_projects} "
            f"attendance-matched projects out of {len(dff)} applicant rows "
            "in the current filter."
        )

        yearly_attendance, year_missing = _attendance_summary(dff, "year")
        if yearly_attendance is not None and not yearly_attendance.empty:
            cohort_by = "cohort_id" if "cohort_id" in dff.columns else "cohort"
            cohort_attendance, cohort_missing = _attendance_summary(
                dff, cohort_by
            )
            sector_attendance, sector_missing = _attendance_summary(dff, "Sector")
            attendance_projects = int(
                yearly_attendance["attendance_projects"].sum()
            )
            overall_rate = (
                yearly_attendance["member_days_attended"].sum()
                / yearly_attendance["member_opportunities"].sum()
                * 100
            )
            attendance_note = (
                f"Member attendance is shown for {attendance_projects} "
                f"attendance-matched projects out of {len(dff)} applicant rows "
                "in the current filter."
            )
            attendance_note += f" Overall: {overall_rate:.1f}%."
            # Show a note only where a group actually has missing values
            # (e.g. Sector); charts with zero missing stay clean.
            note_year = _missing_note({"not specified": year_missing})
            note_cohort = _missing_note({"not specified": cohort_missing})
            note_sector = _missing_note({"not specified": sector_missing})
            fig_att_cohort = _attendance_bar_fig(
                cohort_attendance, cohort_by, sort_by="label", note=note_cohort,
            )
            fig_att_year = _attendance_bar_fig(
                yearly_attendance, "year", sort_by="label", note=note_year,
            )
            fig_att_year.update_layout(
                xaxis=dict(
                    title="",
                    showgrid=False,
                    tickangle=0,
                    tickfont=dict(family=CHART_FONT, size=11, color=C_TEXT),
                    tickformat="d",
                )
            )
            fig_att_sector = _attendance_bar_fig(
                sector_attendance, "Sector", horizontal=True, max_items=6,
                note=note_sector,
            )
        else:
            rate_col = (
                "member_attendance_rate"
                if "member_attendance_rate" in dff.columns
                else "team_attendance_rate"
                if "team_attendance_rate" in dff.columns
                else None
            )
            if rate_col:
                att_series = _attendance_pct(dff[rate_col])
                group_col = "cohort_id" if "cohort_id" in dff.columns else "cohort"
                cohort_ok = dff[group_col].map(
                    lambda value: not _is_missing_group(value)
                )
                year_ok = dff["year"].map(
                    lambda value: not _is_missing_group(value)
                )
                att_by_cohort = (
                    att_series[cohort_ok].groupby(dff.loc[cohort_ok, group_col])
                    .mean()
                    .round(1)
                    .sort_index()
                )
                fig_att_cohort = _bar_fig(
                    att_by_cohort.index,
                    att_by_cohort.values,
                    text=att_by_cohort.values.astype(str) + "%",
                )
                att_by_year = (
                    att_series[year_ok]
                    .groupby(dff.loc[year_ok, "year"])
                    .mean()
                    .round(1)
                    .sort_index()
                )
                fig_att_year = _bar_fig(
                    att_by_year.index,
                    att_by_year.values,
                    text=att_by_year.values.astype(str) + "%",
                )
                fig_att_year.update_layout(
                    xaxis=dict(
                        title="",
                        showgrid=False,
                        tickangle=0,
                        tickfont=dict(family=CHART_FONT, size=11, color=C_TEXT),
                        tickformat="d",
                    )
                )
                sector_ok = dff["Sector"].map(
                    lambda value: not _is_missing_group(value)
                ) & att_series.notna()
                att_by_sector = (
                    att_series[sector_ok]
                    .groupby(dff.loc[sector_ok, "Sector"])
                    .mean()
                    .round(1)
                    .sort_values()
                    .tail(6)
                )
                fig_att_sector = _bar_fig(
                    att_by_sector.values,
                    att_by_sector.index,
                    orientation="h",
                    text=att_by_sector.values.astype(str) + "%",
                )
            else:
                fig_att_cohort = _bar_fig([], [])
                fig_att_year = _bar_fig([], [])
                fig_att_sector = _bar_fig([], [])

        return _page_shell(
            "Attendance",
            kpis,
            _row(
                _card("Member Attendance Rate by Cohort", fig_att_cohort),
                _card("Member Attendance Rate by Year", fig_att_year),
            ),
            _grid_row(
                _card("Member Attendance by Sector", fig_att_sector),
            ),
            subtitle=attendance_note,
        )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description="Mashroo3i analytics dashboard")
    parser.add_argument("--host", default=os.environ.get("DASH_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("DASH_PORT", "8050")))
    parser.add_argument("--debug", action="store_true", default=bool(os.environ.get("DASH_DEBUG")))
    args = parser.parse_args(argv)

    display_host = "localhost" if args.host in ("0.0.0.0", "") else args.host
    print(f"Local dashboard: http://{display_host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)


if __name__ == "__main__":
    main()
