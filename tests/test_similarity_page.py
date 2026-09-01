"""Streamlit AppTest smoke tests for similar-idea flagging."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit_app

FIXTURES = ROOT / "tests" / "fixtures"
APPLICATIONS_CSV = FIXTURES / "applications.csv"


def _applications():
    return streamlit_app.load_applications(
        APPLICATIONS_CSV.read_bytes(), APPLICATIONS_CSV.name
    )


def _open_ai_page(at: AppTest, page_name: str) -> AppTest:
    for radio in at.sidebar.radio:
        if "AI" in (radio.options or []):
            radio.set_value("AI").run(timeout=60)
            break
    for radio in at.sidebar.radio:
        if page_name in (radio.options or []):
            radio.set_value(page_name).run(timeout=60)
            break
    return at


def test_similar_ideas_page_renders_offline_pairs():
    at = AppTest.from_file(str(ROOT / "streamlit_app.py")).run()
    at.session_state["applications"] = _applications()
    at.run()
    assert not at.exception
    at = _open_ai_page(at, streamlit_app.SIMILAR_IDEAS_PAGE)
    assert not at.exception

    metric_labels = [metric.label for metric in at.metric]
    assert "Related idea groups" in metric_labels
    markdown_text = " ".join(str(markdown.value) for markdown in at.markdown)
    assert "Group of" in markdown_text
    assert "Most similar to" in markdown_text
    assert "%" in markdown_text
    assert "Problem:" in markdown_text
    expander_labels = [expander.label for expander in at.expander]
    assert "See full descriptions" in expander_labels


def test_similar_ideas_page_has_discovery_tab_only():
    at = AppTest.from_file(str(ROOT / "streamlit_app.py")).run()
    at.session_state["applications"] = _applications()
    at.run()
    assert not at.exception
    at = _open_ai_page(at, streamlit_app.SIMILAR_IDEAS_PAGE)
    assert not at.exception

    tab_labels = [tab.label for tab in at.tabs]
    assert tab_labels == ["Similar ideas", "Search"]
    selectbox_labels = [selectbox.label for selectbox in at.selectbox]
    assert "How similar should the matches be?" not in selectbox_labels


def test_similar_ideas_threshold_slider_passes_choice(monkeypatch):
    captured: dict[str, float] = {}

    def fake_clusters(threshold=0.70, max_group_size=12):
        captured["threshold"] = threshold
        clusters = pd.DataFrame(
            {
                "cluster_id": [0, 0],
                "row_index": [0, 1],
                "project_name": ["Alpha", "Beta"],
                "year": ["2024", "2024"],
                "cohort_id": ["2024-1", "2024-1"],
                "sector": ["Tech", "Tech"],
                "snippet": ["one", "two"],
                "problem": ["Problem one", "Problem two"],
                "solution": ["Solution one", "Solution two"],
            }
        )
        edges = pd.DataFrame(
            {
                "cluster_id": [0],
                "left_row_index": [0],
                "right_row_index": [1],
                "similarity": [0.88],
                "band": ["Similar (different idea, same concept)"],
            }
        )
        return clusters, edges

    monkeypatch.setattr(
        streamlit_app.similarity,
        "similar_clusters",
        fake_clusters,
    )

    at = AppTest.from_file(str(ROOT / "streamlit_app.py")).run(timeout=60)
    at.session_state["applications"] = _applications()
    at.run(timeout=60)
    at = _open_ai_page(at, streamlit_app.SIMILAR_IDEAS_PAGE)
    at.slider(key="similar_discovery_threshold").set_value(80).run(timeout=60)
    assert not at.exception
    assert captured["threshold"] == 0.80
    assert at.slider(key="similar_discovery_threshold").value == 80


def test_idea_validator_shows_auto_similarity_note(monkeypatch):
    canned = [
        {
            "level": "Very similar",
            "project_name": "Coffeesheep",
            "year": "2024",
            "cohort_id": "2024-1",
            "sector": "Food & Beverage",
            "problem": "Fresh coffee beans distribution through supermarkets.",
            "solution": "Wide distribution of coffee products.",
        }
    ]
    monkeypatch.setattr(
        streamlit_app.similarity,
        "search_similar",
        lambda *args, **kwargs: canned,
    )
    monkeypatch.setattr(
        streamlit_app.similarity,
        "deduplicated_index",
        lambda: ([], pd.DataFrame(), {}),
    )

    at = AppTest.from_file(str(ROOT / "streamlit_app.py")).run()
    at.session_state["applications"] = _applications()
    at.run()
    at = _open_ai_page(at, streamlit_app.AGENT_PAGE)
    assert not at.exception

    idea_text = "A coffee truck that sells freshly roasted beans"
    description_text = (
        "Distribute fresh coffee through supermarkets across Bahrain."
    )
    at.text_area(key="agent_idea").set_value(idea_text)
    at.text_area(key="agent_description").set_value(description_text)
    # Simulate the user having paused before this rerun so the debounce lets
    # the automatic similarity check run.
    at.session_state["agent_sim_text"] = (idea_text, description_text)
    at.session_state["agent_sim_last_edit"] = 0.0
    at.run()
    assert not at.exception

    markdown_text = " ".join(str(markdown.value) for markdown in at.markdown)
    assert "Heads up" in markdown_text
    assert "this idea looks similar" in markdown_text
    assert "your idea" not in markdown_text
    assert "Cohort 2024-1" in markdown_text
    assert "Coffeesheep" in markdown_text
    expander_labels = [expander.label for expander in at.expander]
    assert any("Compare with these past ideas" in label for label in expander_labels)


def test_idea_validator_forces_similarity_check_on_validate(monkeypatch):
    """Submitting before the debounce should still show the similarity note."""
    canned = [
        {
            "level": "Very similar",
            "project_name": "Coffeesheep",
            "year": "2024",
            "cohort_id": "2024-1",
            "problem": "Fresh coffee beans distribution through supermarkets.",
            "solution": "Wide distribution of coffee products.",
        }
    ]
    searches: list[str] = []

    def fake_search(query, **kwargs):
        searches.append(query)
        return canned

    class FakeReport:
        def to_dict(self):
            return {}

    monkeypatch.setattr(streamlit_app.similarity, "search_similar", fake_search)
    monkeypatch.setattr(
        streamlit_app.idea_agent,
        "run_agent_stream",
        lambda **kwargs: FakeReport(),
    )
    # Keep the debounce from ever elapsing so the forced check is what fires.
    monkeypatch.setattr(streamlit_app.time, "monotonic", lambda: 0.0)

    at = AppTest.from_file(str(ROOT / "streamlit_app.py")).run()
    at.session_state["applications"] = _applications()
    at.run()
    at = _open_ai_page(at, streamlit_app.AGENT_PAGE)
    assert not at.exception

    at.text_area(key="agent_idea").set_value(
        "A coffee truck that sells freshly roasted beans"
    )
    at.text_area(key="agent_description").set_value(
        "Distribute fresh coffee through supermarkets across Bahrain."
    )
    at.button(key="agent_run").click().run(timeout=60)
    assert not at.exception

    assert len(searches) == 1
    markdown_text = " ".join(str(markdown.value) for markdown in at.markdown)
    assert "Heads up" in markdown_text
    assert "this idea looks similar" in markdown_text
    assert "your idea" not in markdown_text
    assert "Cohort 2024-1" in markdown_text
    assert "Coffeesheep" in markdown_text


def _search(at: AppTest, query: str) -> AppTest:
    at.text_input(key="similar_search_query").set_value(query).run(timeout=60)
    at.button(key="similar_search_submit").click().run(timeout=60)
    return at


def _search_result_frames(at: AppTest) -> list:
    """Return search-result tables (which always include a Level column)."""
    return [
        element.value
        for element in at.dataframe
        if hasattr(element.value, "columns")
        and "Level" in element.value.columns
    ]


def test_similar_ideas_search_renders_semantic_results(monkeypatch):
    canned = [
        {
            "level": "Very similar",
            "project_name": f"Past Idea {index}",
            "year": "2024",
            "cohort_id": "2024-1",
            "sector": "Technology & IT",
            "problem": "A platform that connects students and tutors.",
            "solution": "Matching and scheduling for tutoring sessions.",
        }
        for index in range(3)
    ]
    monkeypatch.setattr(
        streamlit_app.similarity,
        "search_similar",
        lambda *args, **kwargs: canned,
    )
    monkeypatch.setattr(
        streamlit_app.similarity,
        "deduplicated_index",
        lambda: ([], pd.DataFrame(), {}),
    )

    at = AppTest.from_file(str(ROOT / "streamlit_app.py")).run()
    at.session_state["applications"] = _applications()
    at.run()
    at = _open_ai_page(at, streamlit_app.SIMILAR_IDEAS_PAGE)
    at = _search(at, "tutoring platform")
    assert not at.exception

    metric_labels = [metric.label for metric in at.metric]
    assert "Matches found" in metric_labels
    frames = _search_result_frames(at)
    assert frames, "Search results table not rendered"
    assert list(frames[0].columns) == [
        "Project",
        "Year",
        "Cohort",
        "Sector",
        "Level",
        "Score (%)",
        "Problem",
        "Solution",
    ]
    assert set(frames[0]["Project"]) == {"Past Idea 0", "Past Idea 1", "Past Idea 2"}
    assert set(frames[0]["Level"]) == {"Very similar"}
    detail_labels = [expander.label for expander in at.expander]
    assert not any(label.startswith("View ") for label in detail_labels)


def test_similar_ideas_search_caps_results_at_ten(monkeypatch):
    canned = [
        {
            "level": "Similar",
            "project_name": f"Past Idea {index}",
            "year": "2024",
            "cohort_id": "2024-1",
            "sector": "Technology & IT",
            "problem": "A platform for students.",
            "solution": "Matching students with tutors.",
        }
        for index in range(12)
    ]
    monkeypatch.setattr(
        streamlit_app.similarity,
        "search_similar",
        lambda *args, **kwargs: canned,
    )
    monkeypatch.setattr(
        streamlit_app.similarity,
        "deduplicated_index",
        lambda: ([], pd.DataFrame(), {}),
    )

    at = AppTest.from_file(str(ROOT / "streamlit_app.py")).run()
    at.session_state["applications"] = _applications()
    at.run()
    at = _open_ai_page(at, streamlit_app.SIMILAR_IDEAS_PAGE)
    at = _search(at, "tutoring")
    assert not at.exception

    matches = [metric for metric in at.metric if metric.label == "Matches found"]
    assert matches and int(matches[0].value) == 10
    frames = _search_result_frames(at)
    assert frames, "Search results table not rendered"
    assert len(frames[0]) == 10
    detail_labels = [
        expander.label
        for expander in at.expander
        if expander.label.startswith("View ")
    ]
    assert detail_labels == []


def test_similar_ideas_search_uses_keyword_first(monkeypatch):
    def _no_semantic(*args, **kwargs):
        raise AssertionError(
            "Semantic search should not run when keyword matches exist"
        )

    monkeypatch.setattr(streamlit_app.similarity, "search_similar", _no_semantic)

    at = AppTest.from_file(str(ROOT / "streamlit_app.py")).run()
    at.session_state["applications"] = _applications()
    at.run()
    at = _open_ai_page(at, streamlit_app.SIMILAR_IDEAS_PAGE)
    at = _search(at, "coffee")
    assert not at.exception

    metric_labels = [metric.label for metric in at.metric]
    assert "Matches found" in metric_labels
    frames = _search_result_frames(at)
    assert frames, "Keyword fallback table not rendered"
    assert set(frames[0]["Level"]) == {"Keyword match"}
    assert 0 < len(frames[0]) <= 10
    detail_labels = [
        expander.label
        for expander in at.expander
        if expander.label.startswith("View ")
    ]
    assert detail_labels == []


def test_similar_ideas_search_empty_query_shows_no_results():
    at = AppTest.from_file(str(ROOT / "streamlit_app.py")).run()
    at.session_state["applications"] = _applications()
    at.run()
    at = _open_ai_page(at, streamlit_app.SIMILAR_IDEAS_PAGE)
    at.button(key="similar_search_submit").click().run(timeout=60)
    assert not at.exception

    info_text = " ".join(str(info.value) for info in at.info)
    assert "No matching ideas found" in info_text
    frames = _search_result_frames(at)
    assert frames == []
    detail_labels = [
        expander.label
        for expander in at.expander
        if expander.label.startswith("View ")
    ]
    assert detail_labels == []


def test_two_cohort_page_removed_but_similar_ideas_stays_ai():
    assert "Two-Cohort Ideas" not in streamlit_app.dashboard_pages(True)
    assert "Two-Cohort Ideas" not in streamlit_app.available_pages(True)
    assert streamlit_app.SIMILAR_IDEAS_PAGE in streamlit_app.ai_pages()
