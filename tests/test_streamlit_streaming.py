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
    test_attendance_page_hides_values_toggle()
    print("PASS streaming + attendance toggle tests")


if __name__ == "__main__":
    main()
