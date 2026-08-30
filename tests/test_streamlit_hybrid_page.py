"""Streamlit AppTest smoke for the Model + Agent (hybrid) page."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import model_service
import streamlit_app

FIXTURES = ROOT / "tests" / "fixtures"
APPLICATIONS_CSV = FIXTURES / "applications.csv"
REAL_CSV = Path.home() / "Desktop/filter_3/dashboard_ready.csv"


def _applications() -> pd.DataFrame:
    return streamlit_app.load_applications(
        APPLICATIONS_CSV.read_bytes(), APPLICATIONS_CSV.name
    )


def _open_ai_page(at: AppTest, page_name: str) -> AppTest:
    for radio in at.sidebar.radio:
        if "AI" in (radio.options or []):
            radio.set_value("AI").run()
            break
    for radio in at.sidebar.radio:
        if page_name in (radio.options or []):
            radio.set_value(page_name).run()
            break
    return at


def test_hybrid_page_ranks_with_real_model():
    if not REAL_CSV.exists():
        return  # model fixture not present on this machine
    raw = pd.read_csv(REAL_CSV, encoding="utf-8-sig").head(20)
    ranked = model_service.score_with_model(raw)
    assert len(ranked) == 20
    assert ranked["accept_probability"].between(0, 1).all()
    assert ranked["model_rank"].tolist() == list(range(1, 21))

    at = AppTest.from_file(str(ROOT / "streamlit_app.py")).run()
    at.session_state["applications"] = _applications()
    at.run()
    assert not at.exception
    at = _open_ai_page(at, "Selection Advisor")
    assert not at.exception
    at.session_state["brinc_raw"] = raw
    at.session_state["brinc_ranked"] = ranked
    at.run()
    assert not at.exception
    assert len(at.session_state["brinc_ranked"]) == 20
    assert any("Primary shortlist" in str(value.value) for value in at.markdown)
    primary_frames = [
        element.value
        for element in at.dataframe
        if hasattr(element.value, "columns")
        and "predicted_accepted" in element.value.columns
    ]
    assert primary_frames, "primary shortlist table not rendered"
    labels = set(
        primary_frames[0]["predicted_accepted"].astype(str).unique()
    )
    assert labels <= {"Accepted", "Rejected"}, labels
    metric_labels = [metric.label for metric in at.metric]
    assert "Applicants scored" in metric_labels
    assert "Predicted Accepted" in metric_labels
    assert "Top probability" not in metric_labels


def test_hybrid_page_applies_year_filter_to_shortlist():
    if not REAL_CSV.exists():
        return
    raw = pd.read_csv(REAL_CSV, encoding="utf-8-sig")
    raw = raw[raw["year"].isin([2024, 2025])].head(30)
    expected = int((raw["year"] == 2024).sum())
    assert expected > 0

    at = AppTest.from_file(str(ROOT / "streamlit_app.py")).run()
    at.session_state["applications"] = _applications()
    at.run()
    at = _open_ai_page(at, "Selection Advisor")
    assert not at.exception
    at.session_state["brinc_raw"] = raw
    at.run()

    selected_filter = False
    for choice in at.sidebar.multiselect:
        if choice.label == "Years":
            choice.set_value([2024]).run()
            selected_filter = True
            break
    assert selected_filter
    assert not at.exception
    at.button(key="brinc-rank").click().run()
    assert not at.exception
    ranked = at.session_state["brinc_ranked"]
    assert len(ranked) == expected
    assert set(ranked["year"]) == {2024}
    accepted_count = int(ranked["predicted_accepted"].sum())
    shortlist_sliders = [s for s in at.slider if s.label == "Shortlist size"]
    assert shortlist_sliders, "shortlist slider not rendered"
    assert shortlist_sliders[0].max == accepted_count


def test_real_dataset_option_hidden_when_file_missing():
    original = model_service.REAL_DATA_CSV
    model_service.REAL_DATA_CSV = Path(
        "/tmp/does-not-exist-dashboard_ready.csv"
    )
    try:
        at = AppTest.from_file(str(ROOT / "streamlit_app.py")).run()
        assert not at.exception
        at = _open_ai_page(at, "Selection Advisor")
        assert not at.exception
        assert not any(
            checkbox.label == "Use the real dashboard_ready.csv instead"
            for checkbox in at.checkbox
        )
        warning_text = " ".join(
            str(element.value) for element in at.warning
        )
        assert "Real dataset not found" not in warning_text
    finally:
        model_service.REAL_DATA_CSV = original


def test_sidebar_full_csv_reused_by_selection_advisor():
    if not REAL_CSV.exists():
        return
    at = AppTest.from_file(str(ROOT / "streamlit_app.py")).run()
    app_uploader = next(
        uploader
        for uploader in at.file_uploader
        if uploader.label == "Applications (required)"
    )
    app_uploader.upload(REAL_CSV.name, REAL_CSV.read_bytes())
    at.run()
    at = _open_ai_page(at, "Selection Advisor")
    assert not at.exception
    assert any(
        button.label == "Rank candidates with the classifier model"
        for button in at.button
    ), "model CSV upload should be reused from the sidebar"
    info_text = " ".join(str(element.value) for element in at.info)
    assert "Upload the full applicant CSV" not in info_text
    assert "does not include the full model schema" not in info_text


def test_sidebar_partial_csv_still_requires_model_upload():
    fixture = ROOT / "tests" / "fixtures" / "applications.csv"
    at = AppTest.from_file(str(ROOT / "streamlit_app.py")).run()
    app_uploader = next(
        uploader
        for uploader in at.file_uploader
        if uploader.label == "Applications (required)"
    )
    app_uploader.upload(fixture.name, fixture.read_bytes())
    at.run()
    at = _open_ai_page(at, "Selection Advisor")
    assert not at.exception
    assert not any(
        button.label == "Rank candidates with the classifier model"
        for button in at.button
    )
    info_text = " ".join(str(element.value) for element in at.info)
    assert "does not include the full model schema" in info_text


def test_candidate_review_rows_use_selection_score_label():
    ranked = pd.DataFrame(
        {
            "identity": ["a", "b"],
            "model_rank": [1, 2],
            "project_name": ["Alpha", "Beta"],
            "accept_probability": [0.7, 0.5],
        }
    )
    results = {
        "a": {
            "score": {
                "total_score": 19.0,
                "verdict": "Strong",
                "sources": [],
            }
        },
        "b": {"error": "boom"},
    }
    rows = streamlit_app._candidate_review_rows(ranked, results, 2)
    assert rows[0]["Selection score"] == "19.0/25"
    assert rows[0]["Verdict"] == "Strong"
    assert rows[0]["Status"] == "Done"
    assert "Innovation" not in rows[0]
    assert rows[1]["Selection score"] == "N/A"
    assert rows[1]["Status"] == "Error"


def main():
    test_hybrid_page_ranks_with_real_model()
    test_hybrid_page_applies_year_filter_to_shortlist()
    test_real_dataset_option_hidden_when_file_missing()
    test_sidebar_full_csv_reused_by_selection_advisor()
    test_sidebar_partial_csv_still_requires_model_upload()
    test_candidate_review_rows_use_selection_score_label()
    print("PASS hybrid page ranking + filter tests")


if __name__ == "__main__":
    main()
