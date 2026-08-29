"""Streamlit smoke tests for token streaming on the Idea Validator page."""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import idea_agent
import streamlit_app

FIXTURES = ROOT / "tests" / "fixtures"
APPLICATIONS_CSV = FIXTURES / "applications.csv"
ATTENDANCE_CSV = FIXTURES / "attendance.csv"


def _applications():
    return streamlit_app.load_applications(
        APPLICATIONS_CSV.read_bytes(), APPLICATIONS_CSV.name
    )


def _fake_stream(**kwargs):
    """Stub the streamed agent so no LLM call is made."""
    on_status = kwargs.get("on_status")
    on_token = kwargs.get("on_token")
    if on_status:
        on_status("Evaluating...")
        on_status("Generating your /25 selection score...")
    if on_token:
        on_token('{"dimensions": {')
        on_token('"Problem / Need": {"score": 3}}')
    score = idea_agent.InnovationScore(
        dimensions={
            name: idea_agent.ScoreDimension(score=3, rationale="test")
            for name, _weight in idea_agent.SCORING_RUBRIC
        },
        total_score=15.0,
        verdict="Promising",
        bahrain_impact="Supports local jobs.",
        sources=[
            idea_agent.SearchSource(
                title="Test source", url="https://example.com", snippet="ok"
            )
        ],
    )
    dashboard = idea_agent.DashboardInsights(
        summary="Test dashboard summary.",
        insights=["Test insight."],
        kpis={},
        snapshot={},
    )
    return idea_agent.AgentReport(
        idea_text="test", score=score, dashboard=dashboard
    )


def test_idea_validator_streams_and_stores_report():
    original = idea_agent.run_agent_stream
    idea_agent.run_agent_stream = _fake_stream
    try:
        at = AppTest.from_file(str(ROOT / "streamlit_app.py")).run()
        at.session_state["applications"] = _applications()
        at.run()
        for radio in at.sidebar.radio:
            if "AI" in (radio.options or []):
                radio.set_value("AI").run()
                break
        for radio in at.sidebar.radio:
            if "Idea Validator" in (radio.options or []):
                radio.set_value("Idea Validator").run()
                break
        assert not at.exception
        at.text_area(key="agent_idea").set_value("A local idea for Bahrain.")
        at.text_area(key="agent_description").set_value("A clear description.")
        at.button(key="agent_run").click().run()
        assert not at.exception
        report = at.session_state["agent_report"]
        assert report["score"]["total_score"] == 15.0
        assert report["score"]["bahrain_impact"] == "Supports local jobs."
    finally:
        idea_agent.run_agent_stream = original


def test_dashboard_summary_page_renders():
    original = idea_agent.IdeaValidationAgent.summarize_dashboards

    def _fake_summary(self, question=None, idea_text=None):
        return idea_agent.DashboardInsights(
            summary="This quarter shows steady growth with strong Bahraini participation.",
            insights=[
                "Technology & IT is the leading sector.",
                "Acceptance improved in the latest cohort.",
                "Team applications are more likely to be accepted.",
            ],
            kpis={
                "applications": len(self.applications),
                "accepted": 200,
                "acceptance_rate_pct": 20.8,
                "bahraini_share_pct": 60.1,
            },
            snapshot={
                "top_categories": {
                    "Sector": {"Technology & IT": 229, "Food & Beverage": 159},
                    "applicant_type": {"Team": 371, "Individual": 1112},
                    "outcome_clean": {"Accepted": 309, "Rejected": 1174},
                    "cohort": {"English": 751, "Arabic": 732},
                    "cohort_id": {"2024-1": 321, "2025-1": 190},
                    "Business Stage": {"Idea": 600, "MVP": 400},
                },
                "yearly": [],
            },
        )

    idea_agent.IdeaValidationAgent.summarize_dashboards = _fake_summary
    try:
        at = AppTest.from_file(str(ROOT / "streamlit_app.py")).run()
        at.session_state["applications"] = _applications()
        at.run()
        for radio in at.sidebar.radio:
            if "AI" in (radio.options or []):
                radio.set_value("AI").run()
                break
        for radio in at.sidebar.radio:
            if "Dashboard Summary" in (radio.options or []):
                radio.set_value("Dashboard Summary").run()
                break
        assert not at.exception
        apps = at.session_state["applications"]
        expected_2024 = int((apps["year"] == 2024).sum())
        for choice in at.sidebar.multiselect:
            if choice.label == "Years":
                choice.set_value([2024]).run()
                break
        assert not at.exception
        at.button(key="dash-summary-run").click().run()
        assert not at.exception
        assert "dash_summary" in at.session_state
        summary = at.session_state["dash_summary"]
        assert summary["kpis"]["applications"] == expected_2024
        assert any("Highlights" in str(markdown.value) for markdown in at.markdown)
        assert any("insight-grid" in str(markdown.value) for markdown in at.markdown)
        assert not any("Team share" in str(markdown.value) for markdown in at.markdown)
        assert not any(
            "team_share" in str(markdown.value) for markdown in at.markdown
        )
        markdown_text = " ".join(str(m.value) for m in at.markdown)
        for title in ("Applicant Type", "Outcome", "Language Cohort", "Cohort ID"):
            assert title in markdown_text, f"missing category title {title}"
        assert not any(
            "Agent data snapshot" in str(expander.label)
            for expander in at.expander
        )
    finally:
        idea_agent.IdeaValidationAgent.summarize_dashboards = original


def test_attendance_page_hides_values_toggle():
    at = AppTest.from_file(str(ROOT / "streamlit_app.py")).run()
    at.session_state["applications"] = _applications()
    attendance = streamlit_app.load_attendance(
        ATTENDANCE_CSV.read_bytes(), ATTENDANCE_CSV.name
    )
    at.session_state["attendance"] = attendance
    at.run()
    for radio in at.sidebar.radio:
        if "Attendance" in (radio.options or []):
            radio.set_value("Attendance").run()
            break
    assert not at.exception
    labels = [radio.label for radio in at.sidebar.radio]
    # The toggle should be hidden on the Attendance page because every
    # attendance chart is a rate and cannot be changed to counts.
    assert "Show values as" not in labels


def main():
    test_idea_validator_streams_and_stores_report()
    test_dashboard_summary_page_renders()
    test_attendance_page_hides_values_toggle()
    print("PASS streaming, summary, attendance toggle tests")


if __name__ == "__main__":
    main()
