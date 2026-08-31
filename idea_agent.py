"""AI agent for startup-idea validation and dashboard-data insights.

The agent is backed by DeepSeek (OpenAI-compatible API) for reasoning and
Tavily for web search. It exposes exactly two capabilities:

1. ``score_idea`` -- search the web and score how innovative the idea is.
2. ``summarize_dashboards`` -- read-only analysis of uploaded CSV/DataFrames.

Keys are loaded from Streamlit secrets (``DEEPSEEK_API_KEY``,
``TAVILY_API_KEY``) with an environment-variable fallback. The model comes
from ``AGENT_MODEL`` and defaults to ``deepseek-v4-flash``.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd


def _secret(name: str, default: str | None = None) -> str | None:
    """Read a Streamlit secret, falling back to an environment variable."""
    try:
        import streamlit as st

        value = st.secrets.get(name)
        if value is not None and str(value).strip():
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)


DEEPSEEK_BASE_URL = _secret(
    "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
)
DEFAULT_MODEL = "deepseek-v4-flash"
AGENT_MODEL = _secret("AGENT_MODEL", DEFAULT_MODEL)
AGENT_TEMPERATURE = 0.0
AGENT_SEED = 42
AGENT_THINKING_DISABLED = True

# (criterion name, weight). Official selection rubric: five criteria, each
# scored 0-5, equally weighted into a 0-25 total.
SCORING_RUBRIC: tuple[tuple[str, float], ...] = (
    ("Problem / Need", 0.20),
    ("Solution / Idea", 0.20),
    ("Innovation / Differentiation", 0.20),
    ("Market Potential", 0.20),
    ("Feasibility", 0.20),
)
SELECTION_CRITERIA_PATH = Path(
    os.getenv(
        "BRINC_SELECTION_CRITERIA_PATH",
        str(Path(__file__).resolve().parent / "selection_criteria.json"),
    )
)

MAX_TOOL_ITERATIONS = 6
MAX_SEARCH_QUERIES = 6
MAX_SNIPPET_CHARS = 700
MAX_RESEARCH_CHARS = 9000
MAX_IDEA_CHARS = 6000

DEFAULT_DASHBOARD_QUESTION = (
    "Summarize the current dashboards and highlight the most valuable insights."
)


class AgentError(RuntimeError):
    """Raised when the agent cannot complete a requested capability."""


@dataclass(frozen=True)
class SelectionCriterion:
    """One user-editable selection criterion.

    ``weight`` is a 0-1 fraction (20% is stored as 0.20) and is normalised to
    sum to 1.0 when used for scoring.
    """

    name: str
    guidance: str = ""
    weight: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_SELECTION_CRITERIA: list[SelectionCriterion] = [
    SelectionCriterion(
        name="Problem / Need",
        guidance="Is there a clear and relevant problem being addressed?",
        weight=0.20,
    ),
    SelectionCriterion(
        name="Solution / Idea",
        guidance=(
            "Is the proposed solution clear and does it effectively address "
            "the problem?"
        ),
        weight=0.20,
    ),
    SelectionCriterion(
        name="Innovation / Differentiation",
        guidance="Is the idea innovative, different, or offering a new approach?",
        weight=0.20,
    ),
    SelectionCriterion(
        name="Market Potential",
        guidance=(
            "Is there a clear target customer/market and potential for the "
            "idea to grow?"
        ),
        weight=0.20,
    ),
    SelectionCriterion(
        name="Feasibility",
        guidance=(
            "Is the idea realistic and achievable, considering the technology, "
            "resources, regulations in Bahrain and capabilities required?"
        ),
        weight=0.20,
    ),
]


def _default_criteria_copy() -> list[SelectionCriterion]:
    return [
        SelectionCriterion(
            name=criterion.name,
            guidance=criterion.guidance,
            weight=criterion.weight,
        )
        for criterion in DEFAULT_SELECTION_CRITERIA
    ]


def normalise_weights(
    criteria: list[SelectionCriterion],
) -> list[SelectionCriterion]:
    """Return criteria with non-negative weights that sum to 1.0.

    Invalid/repair weights are treated as zero; when every weight is zero the
    criteria fall back to equal weights.
    """
    cleaned = [criterion for criterion in criteria if criterion.name.strip()]
    if not cleaned:
        return []
    weights: list[float] = []
    for criterion in cleaned:
        try:
            value = float(criterion.weight)
        except (TypeError, ValueError):
            value = 0.0
        if not math.isfinite(value) or value < 0:
            value = 0.0
        weights.append(value)
    total = sum(weights)
    if total > 0:
        weights = [value / total for value in weights]
    else:
        weights = [1.0 / len(weights)] * len(weights)
    return [
        SelectionCriterion(
            name=criterion.name.strip(),
            guidance=criterion.guidance.strip(),
            weight=round(value, 6),
        )
        for criterion, value in zip(cleaned, weights)
    ]


def load_selection_criteria(
    path: str | Path | None = None,
) -> list[SelectionCriterion]:
    """Load criteria from JSON, falling back to the built-in defaults."""
    criteria_path = Path(path or SELECTION_CRITERIA_PATH)
    try:
        raw = json.loads(criteria_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return _default_criteria_copy()
    if not isinstance(raw, list):
        return _default_criteria_copy()

    loaded: list[SelectionCriterion] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        try:
            weight = float(item.get("weight") or 0.0)
        except (TypeError, ValueError):
            weight = 0.0
        if not math.isfinite(weight) or weight < 0:
            weight = 0.0
        loaded.append(
            SelectionCriterion(
                name=name,
                guidance=str(item.get("guidance") or "").strip(),
                weight=weight,
            )
        )
    if not loaded:
        return _default_criteria_copy()
    return normalise_weights(loaded)


def save_selection_criteria(
    criteria: list[SelectionCriterion],
    path: str | Path | None = None,
) -> None:
    """Persist the current in-memory criteria, overwriting the JSON file."""
    if not criteria:
        raise ValueError("At least one selection criterion is required.")
    criteria_path = Path(path or SELECTION_CRITERIA_PATH)
    data = [
        criterion.to_dict()
        for criterion in normalise_weights(criteria)
    ]
    criteria_path.parent.mkdir(parents=True, exist_ok=True)
    criteria_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_selection_criteria(
    criteria: list[SelectionCriterion],
) -> tuple[bool, str]:
    """Validate editable criteria before generation or saving."""
    if not criteria:
        return False, "Add at least one selection criterion."
    seen: set[str] = set()
    for index, criterion in enumerate(criteria, 1):
        name = criterion.name.strip()
        if not name:
            return False, f"Row {index}: criterion name is required."
        folded = name.casefold()
        if folded in seen:
            return False, f"Row {index}: duplicate criterion '{name}'."
        seen.add(folded)
        try:
            weight = float(criterion.weight)
        except (TypeError, ValueError):
            return False, (
                f"Row {index}: '{name}' needs a weight between 0% and 100%."
            )
        if not math.isfinite(weight) or weight < 0 or weight > 1:
            return False, (
                f"Row {index}: '{name}' weight must be between 0% and 100%."
            )
    return True, ""


def criteria_signature(
    criteria: list[SelectionCriterion],
) -> tuple[tuple[str, str, float], ...]:
    """Return a stable signature for detecting in-memory criteria changes."""
    return tuple(
        (criterion.name, criterion.guidance, round(criterion.weight, 6))
        for criterion in normalise_weights(criteria)
    )


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SearchSource:
    title: str
    url: str
    snippet: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class ScoreDimension:
    score: float
    rationale: str
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InnovationScore:
    dimensions: dict[str, ScoreDimension]
    total_score: float | None
    verdict: str
    bahrain_impact: str = ""
    risks: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    sources: list[SearchSource] = field(default_factory=list)
    low_evidence: bool = False
    evidence_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimensions": {
                name: dimension.to_dict()
                for name, dimension in self.dimensions.items()
            },
            "total_score": self.total_score,
            "verdict": self.verdict,
            "bahrain_impact": self.bahrain_impact,
            "risks": self.risks,
            "recommendations": self.recommendations,
            "sources": [source.to_dict() for source in self.sources],
            "low_evidence": self.low_evidence,
            "evidence_note": self.evidence_note,
        }


@dataclass
class DashboardInsights:
    summary: str
    insights: list[str]
    kpis: dict[str, Any]
    snapshot: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentReport:
    idea_text: str
    score: InnovationScore | None
    dashboard: DashboardInsights | None
    used_fallback: bool = False
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "idea_text": self.idea_text,
            "score": self.score.to_dict() if self.score else None,
            "dashboard": self.dashboard.to_dict() if self.dashboard else None,
            "used_fallback": self.used_fallback,
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# LLM / search clients
# ---------------------------------------------------------------------------


class DeepSeekClient:
    """Minimal OpenAI-compatible chat wrapper for DeepSeek."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 120,
        max_retries: int = 2,
        seed: int | None = None,
        temperature: float | None = None,
    ) -> None:
        self.api_key = api_key or _secret("DEEPSEEK_API_KEY")
        self.model = model or AGENT_MODEL
        self.timeout = timeout
        self.max_retries = max_retries
        self.seed = AGENT_SEED if seed is None else seed
        self.temperature = (
            AGENT_TEMPERATURE if temperature is None else temperature
        )
        if not self.api_key:
            raise AgentError(
                "DEEPSEEK_API_KEY is not set. Add it to Streamlit secrets "
                "or the environment."
            )
        from openai import OpenAI  # imported lazily so tests can avoid the SDK

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=DEEPSEEK_BASE_URL,
            timeout=timeout,
            max_retries=0,
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int = 6000,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Return ``{"content": ..., "tool_calls": [...]}`` for one turn."""
        resolved_seed = self.seed if seed is None else seed
        resolved_temperature = (
            self.temperature if temperature is None else temperature
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._complete_once(
                    messages,
                    tools=tools,
                    json_mode=json_mode,
                    temperature=resolved_temperature,
                    max_tokens=max_tokens,
                    seed=resolved_seed,
                )
            except Exception as exc:  # retry transient/rate-limit/model errors
                last_error = exc
                if isinstance(exc, AgentError):
                    raise
                if "json" in str(exc).lower() and json_mode:
                    # Some free models reject response_format; retry without it.
                    json_mode = False
                    continue
                if attempt < self.max_retries:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise AgentError(f"DeepSeek request failed: {exc}") from exc
        assert last_error is not None
        raise AgentError(f"DeepSeek request failed: {last_error}")

    def _complete_once(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        json_mode: bool,
        temperature: float,
        max_tokens: int,
        seed: int | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            kwargs["seed"] = seed
        if AGENT_THINKING_DISABLED:
            # deepseek-v4-* defaults to thinking mode, where temperature and
            # seed are accepted but have no effect; disable it for reproducible
            # greedy sampling.
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        tool_calls = []
        for call in message.tool_calls or []:
            tool_calls.append(
                {
                    "id": call.id,
                    "name": call.function.name,
                    "arguments": call.function.arguments or "{}",
                }
            )
        return {"content": message.content or "", "tool_calls": tool_calls}

    def stream_complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int = 6000,
        seed: int | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        """Stream content deltas, calling ``on_token`` and returning the text.

        If the stream fails before any token is produced, falls back once to
        the non-streaming ``complete`` so callers still get a result.
        """
        resolved_seed = self.seed if seed is None else seed
        resolved_temperature = (
            self.temperature if temperature is None else temperature
        )
        accumulated: list[str] = []
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": resolved_temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if resolved_seed is not None:
            kwargs["seed"] = resolved_seed
        if AGENT_THINKING_DISABLED:
            # See _complete_once: thinking mode ignores sampling controls.
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            stream = self._client.chat.completions.create(**kwargs)
            for chunk in stream:
                if not getattr(chunk, "choices", None):
                    continue
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None) or ""
                if text:
                    accumulated.append(text)
                    if on_token:
                        on_token(text)
        except Exception:
            if accumulated:
                return "".join(accumulated)
            response = self.complete(
                messages,
                tools=tools,
                json_mode=json_mode,
                temperature=resolved_temperature,
                max_tokens=max_tokens,
                seed=resolved_seed,
            )
            return response["content"]
        return "".join(accumulated)


class TavilySearch:
    """Thin wrapper around the Tavily search SDK."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or _secret("TAVILY_API_KEY")
        self._client = None
        if self.api_key:
            from tavily import TavilyClient

            self._client = TavilyClient(api_key=self.api_key)

    def available(self) -> bool:
        return self._client is not None

    def search(self, query: str, max_results: int = 3) -> list[SearchSource]:
        if self._client is None:
            return []
        try:
            response = self._client.search(
                query=query,
                search_depth="basic",
                max_results=max_results,
                include_answer=False,
                timeout=45,
            )
        except Exception:
            return []
        sources: list[SearchSource] = []
        for item in response.get("results", []) or []:
            url = str(item.get("url") or "").strip()
            if not url.lower().startswith(("http://", "https://")):
                continue
            sources.append(
                SearchSource(
                    title=str(item.get("title") or url),
                    url=url,
                    snippet=str(item.get("content") or "")[
                        : MAX_SNIPPET_CHARS
                    ],
                )
            )
        return sources


# ---------------------------------------------------------------------------
# Web research + scoring
# ---------------------------------------------------------------------------


def _compact_idea(idea_text: str, limit: int = MAX_IDEA_CHARS) -> str:
    return re.sub(r"\s+", " ", idea_text).strip()[:limit]


def _search_queries(idea_text: str) -> list[str]:
    short = _compact_idea(idea_text, 180)
    return [
        f'"{short}" startup OR product',
        f'"{short}" competitors OR similar existing solutions',
        f'"{short}" market size OR market demand OR problem',
        f'"{short}" trend OR innovation OR emerging 2024 2025',
        f'"{short}" Bahrain market OR local demand',
        f'"Bahrain" "{short}" regulation OR policy OR national strategy',
    ][:MAX_SEARCH_QUERIES]


def research_idea(
    idea_text: str, searcher: TavilySearch
) -> tuple[list[SearchSource], list[str]]:
    """Run a small set of Tavily searches and return deduplicated sources."""
    queries = _search_queries(idea_text)
    sources: dict[str, SearchSource] = {}
    for query in queries:
        for source in searcher.search(query, max_results=3):
            if source.url not in sources:
                sources[source.url] = source
    return list(sources.values()), queries


def _research_context(sources: list[SearchSource]) -> str:
    chunks: list[str] = []
    used = 0
    for source in sources:
        block = (
            f"- {source.title}\n  URL: {source.url}\n  "
            f"Excerpt: {source.snippet}"
        )
        if used + len(block) > MAX_RESEARCH_CHARS:
            break
        chunks.append(block)
        used += len(block)
    return "\n".join(chunks)


def parse_json_object(text: str) -> dict[str, Any]:
    """Extract and parse the first balanced JSON object from a string."""
    if not text:
        raise ValueError("Empty model response")
    text = text.strip()
    # Strip markdown code fences if present.
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in model response")
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : index + 1])
    raise ValueError("Unbalanced JSON object in model response")


def _complete_json(
    client: DeepSeekClient, messages: list[dict[str, Any]]
) -> dict[str, Any]:
    """Request a JSON object, retrying with larger budgets if parsing fails."""
    last_payload: dict[str, Any] = {}
    for json_mode, max_tokens in (
        (True, 6000),
        (True, 8000),
        (False, 8000),
    ):
        response = client.complete(
            messages, json_mode=json_mode, max_tokens=max_tokens
        )
        try:
            payload = parse_json_object(response["content"])
        except ValueError:
            last_payload = {}
            continue
        if payload:
            return payload
        last_payload = {}
    return last_payload


def _normalise_dimension_name(
    name: str,
    criteria: list[SelectionCriterion] | None = None,
) -> str | None:
    normalized = re.sub(r"[^a-z]+", "", name.lower())
    active = criteria or load_selection_criteria()
    for criterion in active:
        if normalized == re.sub(r"[^a-z]+", "", criterion.name.lower()):
            return criterion.name
    return None


def _score_from_payload(
    payload: dict[str, Any],
    sources: list[SearchSource],
    criteria: list[SelectionCriterion] | None = None,
) -> InnovationScore:
    """Parse/normalise an LLM score payload into a validated InnovationScore."""
    active = normalise_weights(criteria or load_selection_criteria())
    if not active:
        return InnovationScore(
            dimensions={},
            total_score=None,
            verdict="Insufficient Evidence",
        )
    dimensions: dict[str, ScoreDimension] = {}
    raw_dimensions = payload.get("dimensions") or {}
    if not isinstance(raw_dimensions, dict):
        raw_dimensions = {}
    for criterion in active:
        canonical = criterion.name
        raw = None
        for key, value in raw_dimensions.items():
            if _normalise_dimension_name(str(key), active) == canonical:
                raw = value
                break
        if isinstance(raw, dict):
            try:
                score = float(raw.get("score", 0))
            except (TypeError, ValueError):
                score = 0.0
            score = max(0.0, min(5.0, round(score, 1)))
            evidence = raw.get("evidence") or []
            if not isinstance(evidence, list):
                evidence = [str(evidence)] if evidence else []
            dimensions[canonical] = ScoreDimension(
                score=score,
                rationale=str(raw.get("rationale") or "").strip(),
                evidence=[str(item) for item in evidence],
            )
        else:
            dimensions[canonical] = ScoreDimension(
                score=0.0,
                rationale="The model did not provide a score for this dimension.",
            )

    if dimensions:
        # Each criterion is 0-5; weights sum to 1.0, so scale to 0-25.
        total = round(
            5
            * sum(
                dimensions[name].score * weight
                for name, weight in (
                    (criterion.name, criterion.weight)
                    for criterion in active
                )
            ),
            1,
        )
    else:
        total = None

    if total is None:
        verdict = "Insufficient Evidence"
    elif total >= 19:
        verdict = "Strong"
    elif total >= 13:
        verdict = "Promising"
    else:
        verdict = "Weak"

    risks_raw = payload.get("risks") or []
    recommendations_raw = payload.get("recommendations") or []
    risks = [str(item) for item in risks_raw] if isinstance(
        risks_raw, list
    ) else []
    recommendations = (
        [str(item) for item in recommendations_raw]
        if isinstance(recommendations_raw, list)
        else []
    )

    low_evidence = not sources
    evidence_note = (
        "Web research returned no usable sources, so this score is "
        "low-confidence and should be treated as a first-pass estimate."
        if low_evidence
        else ""
    )
    return InnovationScore(
        dimensions=dimensions,
        total_score=total,
        verdict=verdict,
        bahrain_impact=str(payload.get("bahrain_impact") or "").strip(),
        risks=risks,
        recommendations=recommendations,
        sources=sources,
        low_evidence=low_evidence,
        evidence_note=evidence_note,
    )


def _score_prompt(
    idea_text: str,
    context: str,
    criteria: list[SelectionCriterion] | None = None,
) -> list[dict[str, Any]]:
    active = normalise_weights(criteria or load_selection_criteria())
    if not active:
        active = _default_criteria_copy()
    dimension_lines = "\n".join(
        f"- {criterion.name} (max 5, weight {criterion.weight:.0%}): "
        f"{criterion.guidance or 'Use the rubric guidance for this criterion.'}"
        for criterion in active
    )
    dimension_examples = ",\n".join(
        f'        "{json.dumps(criterion.name, ensure_ascii=False)[1:-1]}": '
        '{"score": 0-5, "rationale": "...", "evidence": ["source title"]}'
        for criterion in active
    )
    json_shape = (
        '{"dimensions": {\n'
        f"{dimension_examples}\n"
        '}, '
        '"bahrain_impact": "one short sentence on the impact of the idea on '
        'Bahrain (jobs, economy, policy fit)", '
        '"total_score": number (0-25), "verdict": "Strong|Promising|Weak", '
        '"risks": ["..."], "recommendations": ["..."]}'
    )
    system = (
        "You are a startup selection analyst. Score the idea strictly on the "
        "web research provided against the active selection rubric. Each "
        "criterion is scored 0-5 and the weighted criteria combine into a "
        "total out of 25. Score only what the evidence supports.\n"
        "Return ONLY valid JSON with this shape:\n"
        f"{json_shape}\n"
        f"Scoring rubric (each criterion 0-5, total /25):\n{dimension_lines}\n"
        "Interpret 'Feasibility' as realistic and achievable given the "
        "technology, resources, Bahrain regulations, and the capabilities "
        "required. Score Market Potential and Feasibility with the idea's "
        "impact on Bahrain front of mind.\n"
        "Keep each rationale short and evidence-grounded. Verdicts: Strong "
        ">=19, Promising >=13, Weak <13.\n"
        "Do not invent facts that are not in the research. If evidence is thin, "
        "still return the structure and lower the relevant scores. "
        "No markdown, no commentary."
    )
    user = (
        f"IDEA:\n{_compact_idea(idea_text)}\n\n"
        f"WEB RESEARCH ({len(context)} chars):\n{context or 'NO RESULTS'}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Strict, read-only dashboard analysis
# ---------------------------------------------------------------------------


class DashboardAnalyzer:
    """Safe, read-only summariser for uploaded applications/attendance data."""

    CATEGORY_COLUMNS = (
        "Sector",
        "outcome_clean",
        "Business Stage",
        "applicant_type",
        "cohort",
        "cohort_id",
    )

    def __init__(
        self,
        applications: pd.DataFrame,
        attendance: pd.DataFrame | None = None,
    ) -> None:
        # Frozen copies: the agent can never mutate caller-owned data.
        self.applications = applications.copy()
        self.attendance = (
            attendance.copy() if attendance is not None and len(attendance) else None
        )

    def _require_column(self, df: pd.DataFrame, column: Any) -> str:
        if not isinstance(column, str) or column not in df.columns:
            raise AgentError(f"Unknown column: {column!r}")
        return column

    def schema(self, limit: int = 14) -> list[dict[str, Any]]:
        records = []
        for column in self.applications.columns[:limit]:
            series = self.applications[column]
            records.append(
                {
                    "column": column,
                    "dtype": str(series.dtype),
                    "nulls": int(series.isna().sum()),
                    "unique": int(series.nunique(dropna=True)),
                }
            )
        return records

    def kpis(self) -> dict[str, Any]:
        df = self.applications
        total = int(len(df))
        accepted = (
            int(df["outcome_clean"].eq("Accepted").sum())
            if "outcome_clean" in df.columns
            else 0
        )
        acceptance_rate = round(accepted / total * 100, 1) if total else 0.0
        bahraini = (
            int(
                df["nationality"]
                .astype(str)
                .str.contains("bahrain", case=False, na=False)
                .sum()
            )
            if "nationality" in df.columns
            else 0
        )
        teams = (
            int(df["applicant_type"].eq("Team").sum())
            if "applicant_type" in df.columns
            else 0
        )
        years = sorted(
            [int(value) for value in df["year"].dropna().unique()]
            if "year" in df.columns
            else []
        )
        return {
            "applications": total,
            "accepted": accepted,
            "acceptance_rate_pct": acceptance_rate,
            "bahraini": bahraini,
            "bahraini_share_pct": round(bahraini / total * 100, 1)
            if total
            else 0.0,
            "teams": teams,
            "team_share_pct": round(teams / total * 100, 1)
            if total
            else 0.0,
            "years": years,
        }

    def top_categories(self, column: str, n: int = 5) -> dict[str, int]:
        column = self._require_column(self.applications, column)
        counts = (
            self.applications[column]
            .astype(str)
            .replace({"nan": "Not Specified", "None": "Not Specified"})
            .value_counts()
            .head(n)
        )
        return {str(key): int(value) for key, value in counts.items()}

    def yearly_breakdown(self) -> list[dict[str, Any]]:
        column = self._require_column(self.applications, "year")
        rows: list[dict[str, Any]] = []
        for year, group in self.applications.groupby(column, sort=True):
            total = int(len(group))
            accepted = (
                int(group["outcome_clean"].eq("Accepted").sum())
                if "outcome_clean" in group.columns
                else 0
            )
            rows.append(
                {
                    "year": int(year),
                    "applications": total,
                    "accepted": accepted,
                    "acceptance_rate_pct": round(accepted / total * 100, 1)
                    if total
                    else 0.0,
                }
            )
        return rows

    def snapshot(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "schema": self.schema(),
            "kpis": self.kpis(),
            "yearly": self.yearly_breakdown(),
            "top_categories": {},
        }
        for column in self.CATEGORY_COLUMNS:
            if column in self.applications.columns:
                snapshot["top_categories"][column] = self.top_categories(
                    column, n=5
                )
        if "keywords" in self.applications.columns:
            snapshot["top_keywords"] = self.top_categories("keywords", n=8)
        if self.attendance is not None:
            snapshot["attendance"] = {
                "records": int(len(self.attendance)),
                "columns": list(self.attendance.columns),
            }
        return snapshot


def _dashboard_prompt(
    snapshot: dict[str, Any],
    question: str,
    idea_text: str | None = None,
) -> list[dict[str, Any]]:
    system = (
        "You are a data analyst for a startup accelerator dashboard. Using the "
        "data snapshot ONLY (never invent numbers), write a concise summary and "
        "3-6 decision-ready insights. The summary MUST state the exact "
        "applications count and acceptance rate in the 'kpis' object, and every "
        "insight must come directly from the snapshot. Return ONLY valid JSON:\n"
        '{"summary": "...", "insights": ["..."], "kpis": {...}}\n'
        "No markdown, no commentary."
    )
    content = (
        "DATA SNAPSHOT:\n"
        f"{json.dumps(snapshot, ensure_ascii=False, default=str)}\n\n"
        f"QUESTION: {question}\n"
        + (
            f"\nIDEA BEING EVALUATED (relevant context only): {idea_text}"
            if idea_text
            else ""
        )
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]


def _fallback_dashboard(
    analyzer: DashboardAnalyzer, question: str
) -> DashboardInsights:
    snapshot = analyzer.snapshot()
    kpis = snapshot["kpis"]
    insights: list[str] = []
    for column, counts in (snapshot.get("top_categories") or {}).items():
        top = ", ".join(f"{key} ({value})" for key, value in list(counts.items())[:3])
        if top:
            insights.append(f"Top {column}: {top}.")
    for row in snapshot.get("yearly", []):
        insights.append(
            f"{row['year']}: {row['applications']} applications, "
            f"{row['acceptance_rate_pct']}% accepted."
        )
    summary = (
        f"The dashboard contains {kpis.get('applications', 0)} application "
        f"records. Acceptance rate is {kpis.get('acceptance_rate_pct', 0)}%; "
        f"{kpis.get('bahraini_share_pct', 0)}% are Bahraini and "
        f"{kpis.get('team_share_pct', 0)}% are teams."
    )
    return DashboardInsights(
        summary=summary,
        insights=insights[:6],
        kpis=kpis,
        snapshot=snapshot,
    )


# ---------------------------------------------------------------------------
# Agent orchestration
# ---------------------------------------------------------------------------


def _tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "score_idea",
                "description": (
                    "Search the web for competitors, market need and trends, "
                    "then score the innovation level of the idea on a weighted "
                    "0-25 rubric."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "idea_text": {
                            "type": "string",
                            "description": "The idea's problem and description.",
                        }
                    },
                    "required": ["idea_text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "summarize_dashboards",
                "description": (
                    "Summarize the uploaded dashboard CSVs and extract "
                    "valuable insights using read-only analysis only."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The data question to answer.",
                        }
                    },
                    "required": ["question"],
                },
            },
        },
    ]


def _parse_tool_request(content: str) -> dict[str, Any] | None:
    """Parse the JSON fallback the model may emit instead of tool_calls."""
    try:
        payload = parse_json_object(content)
    except ValueError:
        return None
    tool = payload.get("tool")
    if not isinstance(tool, str) or tool not in {"score_idea", "summarize_dashboards"}:
        return None
    args = payload.get("args")
    return {"tool": tool, "args": args if isinstance(args, dict) else {}}


class IdeaValidationAgent:
    """Agent with exactly two tools: web-scored innovation validation and
    read-only dashboard summarisation."""

    def __init__(
        self,
        applications: pd.DataFrame,
        attendance: pd.DataFrame | None = None,
        client: DeepSeekClient | None = None,
        searcher: TavilySearch | None = None,
        max_tool_iterations: int = MAX_TOOL_ITERATIONS,
        criteria: list[SelectionCriterion] | None = None,
    ) -> None:
        self.applications = applications
        self.attendance = attendance
        self.client = client
        self.searcher = searcher or TavilySearch()
        self.analyzer = DashboardAnalyzer(applications, attendance)
        self.max_tool_iterations = max_tool_iterations
        self.criteria = normalise_weights(
            criteria or load_selection_criteria()
        )
        self.last_tool_calls: list[dict[str, Any]] = []

    # -- capabilities ------------------------------------------------------

    def score_idea(self, idea_text: str) -> InnovationScore:
        sources, _queries = research_idea(idea_text, self.searcher)
        client = self.client or DeepSeekClient()
        messages = _score_prompt(
            idea_text, _research_context(sources), self.criteria
        )
        payload = _complete_json(client, messages)
        score = _score_from_payload(payload, sources, self.criteria)
        if not score.risks:
            score.risks = [
                "The model did not identify specific risks; treat this score "
                "as a rough first pass."
            ]
        if not score.recommendations:
            score.recommendations = [
                "Validate demand with target customers before building."
            ]
        return score

    def summarize_dashboards(
        self,
        question: str = DEFAULT_DASHBOARD_QUESTION,
        idea_text: str | None = None,
    ) -> DashboardInsights:
        snapshot = self.analyzer.snapshot()
        client = self.client or DeepSeekClient()
        messages = _dashboard_prompt(snapshot, question, idea_text)
        payload = _complete_json(client, messages)
        summary = str(payload.get("summary") or "").strip()
        insights_raw = payload.get("insights")
        insights = (
            [str(item) for item in insights_raw][:8]
            if isinstance(insights_raw, list)
            else []
        )
        if not summary or not insights:
            fallback = _fallback_dashboard(self.analyzer, question)
            return DashboardInsights(
                summary=summary or fallback.summary,
                insights=insights or fallback.insights,
                kpis=snapshot["kpis"],
                snapshot=snapshot,
            )
        return DashboardInsights(
            summary=summary,
            insights=insights,
            kpis=snapshot["kpis"],
            snapshot=snapshot,
        )

    # -- agent loop --------------------------------------------------------

    def _dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "score_idea":
            result = self.score_idea(str(args.get("idea_text") or ""))
            return {"score": result.to_dict()}
        if name == "summarize_dashboards":
            question = str(args.get("question") or DEFAULT_DASHBOARD_QUESTION)
            result = self.summarize_dashboards(question, self._idea_text)
            return {"dashboard": result.to_dict()}
        raise AgentError(f"Unknown tool: {name}")

    def _run_tool_loop(
        self, idea_text: str, question: str
    ) -> tuple[AgentReport, bool]:
        self._idea_text = idea_text
        client = self.client or DeepSeekClient()
        system = (
            "You are an idea-validation and data-analysis agent. You have "
            "exactly two tools: score_idea and summarize_dashboards. Use both "
            "tools to validate the idea and summarize the uploaded dashboards, "
            "then answer with a final JSON object containing "
            '"innovation_validation" and "dashboard_insights". '
            "Do not use any other capability."
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"IDEA: {_compact_idea(idea_text, 3000)}\n\n"
                    f"DASHBOARD QUESTION: {question}\n"
                    "Use both tools before responding."
                ),
            },
        ]
        score = None
        dashboard = None
        used_fallback = False
        tool_calls_log: list[dict[str, Any]] = []

        for _ in range(self.max_tool_iterations):
            response = client.complete(messages, tools=_tool_definitions())
            tool_calls = response.get("tool_calls") or []
            content = response.get("content") or ""
            if tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": content or None,
                        "tool_calls": [
                            {
                                "id": call["id"],
                                "type": "function",
                                "function": {
                                    "name": call["name"],
                                    "arguments": call["arguments"],
                                },
                            }
                            for call in tool_calls
                        ],
                    }
                )
                for call in tool_calls:
                    self.last_tool_calls.append(call)
                    tool_calls_log.append(call)
                    try:
                        args = parse_json_object(call["arguments"])
                    except ValueError:
                        args = {}
                    result = self._dispatch(call["name"], args)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )
                    if "score" in result:
                        score = _score_from_dict(
                            result["score"], self.criteria
                        )
                    if "dashboard" in result:
                        dashboard = _dashboard_from_dict(result["dashboard"])
                continue

            # No native tool calls: try the JSON ToolRequest fallback.
            request = _parse_tool_request(content)
            if request:
                used_fallback = True
                messages.append({"role": "assistant", "content": content})
                result = self._dispatch(request["tool"], request["args"])
                messages.append(
                    {
                        "role": "user",
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
                if "score" in result:
                    score = _score_from_dict(result["score"], self.criteria)
                if "dashboard" in result:
                    dashboard = _dashboard_from_dict(result["dashboard"])
                continue

            # Plain text (possibly a final report JSON); stop the loop.
            try:
                payload = parse_json_object(content)
            except ValueError:
                payload = None
            if payload:
                # The model's final answer may be partial; never let it clobber
                # a complete tool result with an incomplete blob.
                if "innovation_validation" in payload and score is None:
                    candidate = _score_from_dict(
                        payload["innovation_validation"], self.criteria
                    )
                    if candidate.dimensions:
                        score = candidate
                if "dashboard_insights" in payload and dashboard is None:
                    candidate = _dashboard_from_dict(
                        payload["dashboard_insights"]
                    )
                    if candidate.summary:
                        dashboard = candidate
            break

        errors: list[str] = []
        if score is None:
            used_fallback = True
            try:
                score = self.score_idea(idea_text)
            except Exception as exc:
                errors.append(f"score_idea failed: {exc}")
        if dashboard is None:
            used_fallback = True
            try:
                dashboard = self.summarize_dashboards(question, idea_text)
            except Exception as exc:
                errors.append(f"summarize_dashboards failed: {exc}")
        return (
            AgentReport(
                idea_text=idea_text,
                score=score,
                dashboard=dashboard,
                used_fallback=used_fallback,
                errors=errors,
            ),
            False,
        )

    def run(
        self,
        idea_text: str,
        question: str = DEFAULT_DASHBOARD_QUESTION,
    ) -> AgentReport:
        """Run both capabilities and return a complete, serialisable report."""
        report, _ = self._run_tool_loop(idea_text, question)
        return report

    def run_stream(
        self,
        idea_text: str,
        question: str = DEFAULT_DASHBOARD_QUESTION,
        include_dashboard: bool = True,
        on_status: Callable[[str], None] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> AgentReport:
        """Streamed execution: web search progress, then live LLM tokens."""
        self._idea_text = idea_text
        client = self.client or DeepSeekClient()
        searcher = self.searcher or TavilySearch()

        def status(text: str) -> None:
            if on_status:
                on_status(text)

        status("Evaluating...")
        queries = _search_queries(idea_text)
        sources: dict[str, SearchSource] = {}
        for index, query in enumerate(queries, 1):
            status(f"Searching the web... ({index}/{len(queries)})")
            for source in searcher.search(query, max_results=3):
                if source.url not in sources:
                    sources[source.url] = source
        source_list = list(sources.values())

        status("Generating your /25 selection score...")
        score_messages = _score_prompt(
            idea_text, _research_context(source_list), self.criteria
        )
        score_text = client.stream_complete(
            score_messages, json_mode=True, on_token=on_token
        )
        try:
            payload = parse_json_object(score_text)
        except ValueError:
            try:
                payload = _complete_json(client, score_messages)
            except AgentError:
                payload = {}
        score = _score_from_payload(payload, source_list, self.criteria)
        if not score.risks:
            score.risks = [
                "The model did not identify specific risks; treat this score "
                "as a rough first pass."
            ]
        if not score.recommendations:
            score.recommendations = [
                "Validate demand with target customers before building."
            ]

        dashboard = None
        if include_dashboard:
            status("Generating dashboard insights...")
            snapshot = self.analyzer.snapshot()
            dash_messages = _dashboard_prompt(snapshot, question, idea_text)
            dash_text = client.stream_complete(
                dash_messages, json_mode=True, on_token=on_token
            )
            try:
                dash_payload = parse_json_object(dash_text)
            except ValueError:
                dash_payload = {}
            summary = str(dash_payload.get("summary") or "").strip()
            insights_raw = dash_payload.get("insights")
            insights = (
                [str(item) for item in insights_raw][:8]
                if isinstance(insights_raw, list)
                else []
            )
            if not summary or not insights:
                fallback = _fallback_dashboard(self.analyzer, question)
                dashboard = DashboardInsights(
                    summary=summary or fallback.summary,
                    insights=insights or fallback.insights,
                    kpis=snapshot["kpis"],
                    snapshot=snapshot,
                )
            else:
                dashboard = DashboardInsights(
                    summary=summary,
                    insights=insights,
                    kpis=snapshot["kpis"],
                    snapshot=snapshot,
                )
        status("Evaluation complete")
        return AgentReport(
            idea_text=idea_text,
            score=score,
            dashboard=dashboard,
            used_fallback=False,
            errors=[],
        )


def _score_from_dict(
    payload: dict[str, Any],
    criteria: list[SelectionCriterion] | None = None,
) -> InnovationScore:
    active = normalise_weights(criteria or load_selection_criteria())
    dimensions: dict[str, ScoreDimension] = {}
    raw_dimensions = payload.get("dimensions") or {}
    if isinstance(raw_dimensions, dict):
        for criterion in active:
            raw = None
            for key, value in raw_dimensions.items():
                if _normalise_dimension_name(str(key), active) == criterion.name:
                    raw = value
                    break
            if not isinstance(raw, dict):
                continue
            dimensions[criterion.name] = ScoreDimension(
                score=float(raw.get("score", 0) or 0),
                rationale=str(raw.get("rationale") or ""),
                evidence=[str(item) for item in raw.get("evidence") or []],
            )
    sources = []
    for raw in payload.get("sources") or []:
        if isinstance(raw, dict):
            sources.append(
                SearchSource(
                    title=str(raw.get("title") or ""),
                    url=str(raw.get("url") or ""),
                    snippet=str(raw.get("snippet") or ""),
                )
            )
    total = payload.get("total_score")
    if total is None and dimensions:
        total = round(
            5
            * sum(
                dimensions[criterion.name].score * criterion.weight
                for criterion in active
            ),
            1,
        )
    verdict = str(payload.get("verdict") or "")
    if not verdict and total is not None:
        if total >= 19:
            verdict = "Strong"
        elif total >= 13:
            verdict = "Promising"
        else:
            verdict = "Weak"
    return InnovationScore(
        dimensions=dimensions,
        total_score=float(total) if total is not None else None,
        verdict=verdict or "Insufficient Evidence",
        bahrain_impact=str(payload.get("bahrain_impact") or ""),
        risks=[str(item) for item in payload.get("risks") or []],
        recommendations=[
            str(item) for item in payload.get("recommendations") or []
        ],
        sources=sources,
        low_evidence=bool(payload.get("low_evidence", not sources)),
        evidence_note=str(payload.get("evidence_note") or ""),
    )


def _dashboard_from_dict(payload: dict[str, Any]) -> DashboardInsights:
    return DashboardInsights(
        summary=str(payload.get("summary") or ""),
        insights=[str(item) for item in payload.get("insights") or []],
        kpis=payload.get("kpis") or {},
        snapshot=payload.get("snapshot") or {},
    )


def run_agent(
    idea_text: str,
    applications: pd.DataFrame,
    attendance: pd.DataFrame | None = None,
    question: str = DEFAULT_DASHBOARD_QUESTION,
    client: DeepSeekClient | None = None,
    searcher: TavilySearch | None = None,
    criteria: list[SelectionCriterion] | None = None,
) -> AgentReport:
    """Convenience entrypoint used by the Streamlit page and the evaluator."""
    agent = IdeaValidationAgent(
        applications=applications,
        attendance=attendance,
        client=client,
        searcher=searcher,
        criteria=criteria,
    )
    return agent.run(idea_text, question)


def run_agent_stream(
    idea_text: str,
    applications: pd.DataFrame,
    attendance: pd.DataFrame | None = None,
    question: str = DEFAULT_DASHBOARD_QUESTION,
    include_dashboard: bool = True,
    on_status: Callable[[str], None] | None = None,
    on_token: Callable[[str], None] | None = None,
    client: DeepSeekClient | None = None,
    searcher: TavilySearch | None = None,
    criteria: list[SelectionCriterion] | None = None,
) -> AgentReport:
    """Streaming entrypoint used by the Streamlit pages."""
    agent = IdeaValidationAgent(
        applications=applications,
        attendance=attendance,
        client=client,
        searcher=searcher,
        criteria=criteria,
    )
    return agent.run_stream(
        idea_text,
        question=question,
        include_dashboard=include_dashboard,
        on_status=on_status,
        on_token=on_token,
    )


def agent_ready() -> tuple[bool, str]:
    """Return (ok, message) describing whether the required keys are present."""
    missing = [
        name
        for name in ("DEEPSEEK_API_KEY", "TAVILY_API_KEY")
        if not _secret(name)
    ]
    if missing:
        return (
            False,
            "Missing secret keys: "
            + ", ".join(missing)
            + ". Add them to Streamlit secrets or the environment.",
        )
    return True, f""
