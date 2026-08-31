"""Tests for the DeepSeek + Tavily idea-validation agent.

Run directly with ``python tests/test_idea_agent.py`` or via pytest.
These tests never call the network; the LLM and search are faked.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import idea_agent


SCORE_PAYLOAD = {
    "dimensions": {
        "Problem / Need": {"score": 4, "rationale": "Clear need.", "evidence": ["A"]},
        "Solution / Idea": {"score": 4, "rationale": "Clear solution.", "evidence": []},
        "Innovation / Differentiation": {
            "score": 3,
            "rationale": "Somewhat different.",
            "evidence": [],
        },
        "Market Potential": {"score": 4, "rationale": "Growing market.", "evidence": []},
        "Feasibility": {"score": 3, "rationale": "Buildable in Bahrain.", "evidence": []},
    },
    "total_score": 18.0,
    "verdict": "Promising",
    "bahrain_impact": "Creates local jobs and supports Bahrain's food-sector goals.",
    "risks": ["Competition"],
    "recommendations": ["Talk to customers"],
}


class FakeLLMClient:
    """Returns scripted completions and records how it was called."""

    def __init__(self, responses: list[dict] | None = None):
        self.responses = list(responses or [])
        self.calls: list[dict] = []

    def complete(
        self,
        messages,
        tools=None,
        json_mode=False,
        temperature=0.2,
        max_tokens=2500,
    ):
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "json_mode": json_mode,
            }
        )
        if self.responses:
            return self.responses.pop(0)
        return {
            "content": json.dumps(
                {
                    "innovation_validation": dict(SCORE_PAYLOAD),
                    "dashboard_insights": {
                        "summary": "The dashboard shows strong growth.",
                        "insights": ["Sector A dominates."],
                        "kpis": {},
                        "snapshot": {},
                    },
                }
            ),
            "tool_calls": [],
        }


class StubSearcher:
    def __init__(self, sources=None):
        self.sources = sources or [
            idea_agent.SearchSource(
                title="Example competitor",
                url="https://example.com/competitor",
                snippet="A similar product exists in the market.",
            )
        ]
        self.queries = []

    def available(self):
        return True

    def search(self, query, max_results=3):
        self.queries.append(query)
        return self.sources


class StreamFakeLLM:
    """Fake streaming client: emits chunks and records calls."""

    def __init__(self):
        self.calls = 0

    def complete(self, messages, tools=None, json_mode=False, temperature=0.2, max_tokens=6000):
        return {"content": json.dumps(SCORE_PAYLOAD), "tool_calls": []}

    def stream_complete(
        self,
        messages,
        tools=None,
        json_mode=False,
        temperature=0.2,
        max_tokens=6000,
        on_token=None,
    ):
        self.calls += 1
        text = json.dumps(SCORE_PAYLOAD)
        for index in range(0, len(text), 12):
            piece = text[index : index + 12]
            if on_token:
                on_token(piece)
        return text


class ScriptedLoopLLM:
    """Emits both native tool calls, then a partial final answer."""

    def __init__(self):
        self.calls = []

    def complete(
        self,
        messages,
        tools=None,
        json_mode=False,
        temperature=0.2,
        max_tokens=2500,
    ):
        self.calls.append({"tools": tools, "json_mode": json_mode})
        if tools:
            if any(message.get("role") == "tool" for message in messages):
                # Final orchestration answer is partial on purpose.
                return {
                    "content": json.dumps(
                        {
                            "innovation_validation": {"verdict": "Promising"},
                            "dashboard_insights": {"summary": ""},
                        }
                    ),
                    "tool_calls": [],
                }
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "name": "score_idea",
                        "arguments": json.dumps(
                            {"idea_text": "Problem: x\nDescription: y"}
                        ),
                    },
                    {
                        "id": "call_2",
                        "name": "summarize_dashboards",
                        "arguments": json.dumps(
                            {"question": "Summarize everything."}
                        ),
                    },
                ],
            }
        if json_mode:
            if any("IDEA:" in (message.get("content") or "") for message in messages):
                return {"content": json.dumps(SCORE_PAYLOAD), "tool_calls": []}
            return {
                "content": json.dumps(
                    {
                        "summary": "Real dashboard summary.",
                        "insights": ["Real insight."],
                        "kpis": {"applications": 4},
                        "snapshot": {},
                    }
                ),
                "tool_calls": [],
            }
        return {"content": "", "tool_calls": []}


def _apps() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": [2024, 2024, 2025, 2025],
            "cohort": ["English", "Arabic", "English", "Arabic"],
            "outcome_clean": ["Accepted", "Rejected", "Accepted", "Accepted"],
            "Sector": ["Technology & IT", "Food & Beverage", "Technology & IT", "Education"],
            "applicant_type": ["Team", "Individual", "Team", "Individual"],
            "nationality": ["Bahrain", "Egypt", "Bahrain", "Jordan"],
            "Business Stage": ["Idea", "MVP", "Operating", "Idea"],
        }
    )


def _custom_criteria():
    return [
        idea_agent.SelectionCriterion(
            name="Alpha",
            guidance="custom alpha guidance",
            weight=0.75,
        ),
        idea_agent.SelectionCriterion(
            name="Beta",
            guidance="custom beta guidance",
            weight=0.25,
        ),
    ]


def test_criteria_file_round_trip():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "criteria.json"
        idea_agent.save_selection_criteria(_custom_criteria(), path)
        loaded = idea_agent.load_selection_criteria(path)
    assert [criterion.name for criterion in loaded] == ["Alpha", "Beta"]
    assert [criterion.guidance for criterion in loaded] == [
        "custom alpha guidance",
        "custom beta guidance",
    ]
    assert sum(criterion.weight for criterion in loaded) == 1.0


def test_default_criteria_match_file():
    loaded = idea_agent.load_selection_criteria()
    assert [criterion.guidance for criterion in loaded] == [
        "Is there a clear and relevant problem being addressed?",
        "Is the proposed solution clear and does it effectively address the problem?",
        "Is the idea innovative, different, or offering a new approach?",
        "Is there a clear target customer/market and potential for the idea to grow?",
        "Is the idea realistic and achievable, considering the technology, resources, regulations in Bahrain and capabilities required?",
    ]


def test_criteria_validation_blocks_bad_input():
    assert not idea_agent.validate_selection_criteria([])[0]
    assert not idea_agent.validate_selection_criteria(
        [idea_agent.SelectionCriterion(name="Alpha", weight=1.5)]
    )[0]
    assert not idea_agent.validate_selection_criteria(
        [idea_agent.SelectionCriterion(name="Alpha", weight=float("nan"))]
    )[0]
    assert not idea_agent.validate_selection_criteria(
        [
            idea_agent.SelectionCriterion(name="Alpha", weight=0.2),
            idea_agent.SelectionCriterion(name="alpha", weight=0.2),
        ]
    )[0]


def test_criteria_weights_are_normalised():
    normalised = idea_agent.normalise_weights(
        [
            idea_agent.SelectionCriterion(
                name="Alpha", guidance="a", weight=2
            ),
            idea_agent.SelectionCriterion(
                name="Beta", guidance="b", weight=1
            ),
        ]
    )
    assert [criterion.weight for criterion in normalised] == [
        round(2 / 3, 6),
        round(1 / 3, 6),
    ]
    equal = idea_agent.normalise_weights(
        [
            idea_agent.SelectionCriterion(name="Alpha", weight=0),
            idea_agent.SelectionCriterion(name="Beta", weight=0),
        ]
    )
    assert [criterion.weight for criterion in equal] == [0.5, 0.5]


def test_custom_criteria_scoring_and_prompt():
    payload = {
        "dimensions": {
            "Alpha": {"score": 4, "rationale": "good"},
            "Beta": {"score": 2, "rationale": "ok"},
        },
        "bahrain_impact": "Local jobs.",
        "risks": [],
        "recommendations": [],
    }
    score = idea_agent._score_from_payload(payload, [], _custom_criteria())
    assert set(score.dimensions) == {"Alpha", "Beta"}
    assert score.total_score == 17.5
    messages = idea_agent._score_prompt(
        "Problem: test", "context", _custom_criteria()
    )
    system = messages[0]["content"]
    assert "Alpha (max 5, weight 75%): custom alpha guidance" in system
    assert "Beta (max 5, weight 25%): custom beta guidance" in system
    assert '"Alpha"' in system
    assert '"Beta"' in system
    assert "impact on Bahrain" in system


def test_score_payload_does_not_truncate_agent_output():
    payload = {
        "dimensions": {
            "Problem / Need": {
                "score": 4,
                "rationale": "A complete rationale.",
                "evidence": [f"evidence {index}" for index in range(6)],
            }
        },
        "risks": [f"risk {index}" for index in range(12)],
        "recommendations": [
            f"recommendation {index}" for index in range(12)
        ],
    }
    score = idea_agent._score_from_payload(payload, [])
    assert len(score.dimensions["Problem / Need"].evidence) == 6
    assert len(score.risks) == 12
    assert len(score.recommendations) == 12


def test_score_payload_math_clamps_and_weighs():
    score = idea_agent._score_from_payload(SCORE_PAYLOAD, [])
    assert set(score.dimensions) == {
        name for name, _weight in idea_agent.SCORING_RUBRIC
    }
    # 4+4+3+4+3 = 18/25 (equal weights, scale x5).
    assert score.total_score == 18.0
    assert score.verdict == "Promising"
    assert score.low_evidence is True


def test_score_idea_splices_search_evidence():
    agent = idea_agent.IdeaValidationAgent(
        applications=_apps(),
        client=FakeLLMClient(
            [
                {
                    "content": json.dumps(SCORE_PAYLOAD),
                    "tool_calls": [],
                }
            ]
        ),
        searcher=StubSearcher(),
    )
    score = agent.score_idea("Problem: food waste\nDescription: reuse leftovers")
    assert score.sources and score.sources[0].url.startswith("http")
    assert len(agent.last_tool_calls) == 0
    assert score.total_score == 18.0


def test_run_agent_stream_emits_status_and_tokens():
    stream_fake = StreamFakeLLM()
    statuses: list[str] = []
    tokens: list[str] = []
    agent = idea_agent.IdeaValidationAgent(
        applications=_apps(),
        client=stream_fake,
        searcher=StubSearcher(),
    )
    report = agent.run_stream(
        "Problem: x\nDescription: y",
        include_dashboard=False,
        on_status=statuses.append,
        on_token=tokens.append,
    )
    assert any("Searching the web" in item for item in statuses)
    assert any("Generating your /25" in item for item in statuses)
    assert len(tokens) > 3
    assert report.score.total_score == 18.0
    assert "Bahrain" in report.score.bahrain_impact
    assert report.dashboard is None
    assert stream_fake.calls == 1


def test_run_agent_stream_does_not_overwrite_criteria_file():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "criteria.json"
        original_path = idea_agent.SELECTION_CRITERIA_PATH
        idea_agent.SELECTION_CRITERIA_PATH = path
        try:
            agent = idea_agent.IdeaValidationAgent(
                applications=_apps(),
                client=StreamFakeLLM(),
                searcher=StubSearcher(),
                criteria=_custom_criteria(),
            )
            agent.run_stream(
                "Problem: x\nDescription: y",
                include_dashboard=False,
            )
        finally:
            idea_agent.SELECTION_CRITERIA_PATH = original_path
        assert not path.exists()


def test_dashboard_analyzer_is_read_only_and_rejects_unknown_columns():
    apps = _apps()
    analyzer = idea_agent.DashboardAnalyzer(apps)
    snapshot = analyzer.snapshot()
    assert snapshot["kpis"]["applications"] == 4
    assert snapshot["kpis"]["acceptance_rate_pct"] == 75.0
    assert snapshot["kpis"]["years"] == [2024, 2025]
    assert snapshot["top_categories"]["Sector"]["Technology & IT"] == 2
    # The original frame must remain untouched.
    assert len(apps) == 4
    assert apps["year"].tolist() == [2024, 2024, 2025, 2025]
    for bad_column in ("__class__", "exec", "df.drop"):
        try:
            analyzer.top_categories(bad_column)
        except idea_agent.AgentError:
            continue
        raise AssertionError(f"expected AgentError for {bad_column!r}")


def test_dashboard_summary_falls_back_when_model_is_garbage():
    agent = idea_agent.IdeaValidationAgent(
        applications=_apps(),
        client=FakeLLMClient([{"content": "not json at all", "tool_calls": []}]),
        searcher=StubSearcher(),
    )
    insights = agent.summarize_dashboards()
    assert insights.summary
    assert insights.insights
    assert insights.kpis["applications"] == 4


def test_run_loop_accepts_final_report_without_native_tool_calls():
    agent = idea_agent.IdeaValidationAgent(
        applications=_apps(),
        client=FakeLLMClient(),
        searcher=StubSearcher(),
    )
    report = agent.run("Problem: x\nDescription: y")
    assert report.score is not None
    assert report.dashboard is not None
    assert report.dashboard.summary
    assert report.dashboard.insights
    assert not report.errors


def test_run_loop_guarantees_both_capabilities_when_model_stays_silent():
    agent = idea_agent.IdeaValidationAgent(
        applications=_apps(),
        client=FakeLLMClient([{"content": "", "tool_calls": []}]),
        searcher=StubSearcher(),
    )
    report = agent.run("Problem: x\nDescription: y")
    assert report.score is not None
    assert report.dashboard is not None
    assert report.used_fallback is True


def test_complete_tool_results_are_not_clobbered_by_partial_final_answer():
    agent = idea_agent.IdeaValidationAgent(
        applications=_apps(),
        client=ScriptedLoopLLM(),
        searcher=StubSearcher(),
    )
    report = agent.run("Problem: x\nDescription: y")
    assert report.score is not None
    assert report.score.total_score == 18.0
    assert report.score.sources
    assert report.dashboard is not None
    assert report.dashboard.kpis["applications"] == 4
    assert not report.errors


def test_parse_json_object_handles_fences_and_whitespace():
    text = '```json\n{"a": {"b": 1}}\n```'
    assert idea_agent.parse_json_object(text) == {"a": {"b": 1}}
    try:
        idea_agent.parse_json_object("{broken")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unbalanced JSON")


def _fake_message_response(text="hello"):
    message = SimpleNamespace(content=text, tool_calls=[])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _fake_stream_chunk(text="hello"):
    delta = SimpleNamespace(content=text)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


class _RecordingCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if kwargs.get("stream"):
            return [_fake_stream_chunk()]
        return _fake_message_response()


class _MockOpenAI:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_RecordingCompletions())


def _new_client(seed=None, temperature=None):
    client = idea_agent.DeepSeekClient(
        api_key="test-key", seed=seed, temperature=temperature
    )
    mock = _MockOpenAI()
    client._client = mock
    return client, mock.chat.completions


def test_deepseek_client_defaults_to_deterministic_sampling():
    client, completions = _new_client()
    client.complete([{"role": "user", "content": "hi"}])
    assert completions.kwargs["temperature"] == idea_agent.AGENT_TEMPERATURE
    assert completions.kwargs["seed"] == idea_agent.AGENT_SEED
    assert completions.kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


def test_deepseek_client_explicit_overrides_win():
    client, completions = _new_client()
    client.complete(
        [{"role": "user", "content": "hi"}],
        temperature=0.5,
        seed=7,
    )
    assert completions.kwargs["temperature"] == 0.5
    assert completions.kwargs["seed"] == 7
    assert completions.kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


def test_deepseek_client_stream_forwards_deterministic_sampling():
    client, completions = _new_client()
    result = client.stream_complete([{"role": "user", "content": "hi"}])
    assert completions.kwargs["stream"] is True
    assert completions.kwargs["temperature"] == idea_agent.AGENT_TEMPERATURE
    assert completions.kwargs["seed"] == idea_agent.AGENT_SEED
    assert completions.kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
    assert result == "hello"


def main():
    tests = [
        test_criteria_file_round_trip,
        test_default_criteria_match_file,
        test_criteria_validation_blocks_bad_input,
        test_criteria_weights_are_normalised,
        test_custom_criteria_scoring_and_prompt,
        test_score_payload_does_not_truncate_agent_output,
        test_score_payload_math_clamps_and_weighs,
        test_score_idea_splices_search_evidence,
        test_run_agent_stream_emits_status_and_tokens,
        test_run_agent_stream_does_not_overwrite_criteria_file,
        test_dashboard_analyzer_is_read_only_and_rejects_unknown_columns,
        test_dashboard_summary_falls_back_when_model_is_garbage,
        test_run_loop_accepts_final_report_without_native_tool_calls,
        test_run_loop_guarantees_both_capabilities_when_model_stays_silent,
        test_complete_tool_results_are_not_clobbered_by_partial_final_answer,
        test_parse_json_object_handles_fences_and_whitespace,
        test_deepseek_client_defaults_to_deterministic_sampling,
        test_deepseek_client_explicit_overrides_win,
        test_deepseek_client_stream_forwards_deterministic_sampling,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
