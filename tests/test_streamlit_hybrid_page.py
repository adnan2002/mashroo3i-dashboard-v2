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


def main():
    test_hybrid_page_ranks_with_real_model()
    test_hybrid_page_applies_year_filter_to_shortlist()
    print("PASS hybrid page ranking + filter tests")


if __name__ == "__main__":
    main()
