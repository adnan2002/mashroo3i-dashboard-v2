"""Evaluation harness for the idea-validation agent.

Reads the real dashboard CSV, filters to 2024-2025 (2023 excluded), samples
real submitted ideas, runs the agent end-to-end, and reports pass/fail per
criterion.

Usage::

    python evaluate_agent.py                    # live, 10 samples
    python evaluate_agent.py --offline          # fake LLM/search, no network
    python evaluate_agent.py --samples 6 --out /tmp/mashroo3i_eval.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import time
from pathlib import Path
from typing import Any

import pandas as pd

import idea_agent


ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
DEFAULT_CSV = os.path.expanduser("~/Desktop/filter_3/dashboard_ready.csv")

CURATED_CASES = [
    {
        "label": "generic food delivery app",
        "language": "en",
        "expected": "weak",
        "text": (
            "Project: QuickEats\nProblem: Hungry people want food delivered.\n"
            "Description: A mobile app where users browse local restaurants "
            "and order meals for home delivery."
        ),
    },
    {
        "label": "another photo social network",
        "language": "en",
        "expected": "weak",
        "text": (
            "Project: SnapPic\nProblem: People like sharing photos.\n"
            "Description: A social media app for posting and liking photos "
            "with friends and being followed by others."
        ),
    },
    {
        "label": "ordinary coffee shop",
        "language": "en",
        "expected": "weak",
        "text": (
            "Project: Daily Brew\nProblem: People need coffee.\n"
            "Description: Open a specialty coffee shop serving espresso and "
            "pastries in a busy Bahrain neighborhood."
        ),
    },
    {
        "label": "generic online marketplace",
        "language": "en",
        "expected": "weak",
        "text": (
            "Project: ShopAll\nProblem: Buyers want to shop online.\n"
            "Description: A marketplace website where sellers list products "
            "and buyers compare prices and order them."
        ),
    },
    {
        "label": "commodity water delivery",
        "language": "en",
        "expected": "weak",
        "text": (
            "Project: AquaGo\nProblem: Offices need drinking water.\n"
            "Description: A bottled-water home and office delivery service "
            "with ordering by phone and WhatsApp."
        ),
    },
    {
        "label": "Arabic generic restaurant delivery",
        "language": "ar",
        "expected": "weak",
        "text": (
            "المشروع: تطبيق توصيل مطاعم\n"
            "المشكلة: الناس يريدون طلب الطعام من المطاعم.\n"
            "الوصف: تطبيق جوال لعرض المطاعم المحلية وطلب الوجبات "
            "وتوصيلها إلى المنزل."
        ),
    },
    {
        "label": "vague non-idea input",
        "language": "en",
        "expected": "edge",
        "text": "Project: Innovation\nProblem: Some problem.\nDescription: Do something innovative.",
    },
    {
        "label": "differentiated circular-construction tech",
        "language": "en",
        "expected": "strong",
        "text": (
            "Project: CircularBuilt\nProblem: Construction waste is burned or "
            "landfilled, while building materials are imported at high cost.\n"
            "Description: A platform that sorts demolition waste on site, "
            "converts it into 3D-printable low-carbon building blocks with "
            "verified life-cycle certificates, and matches output to local "
            "developers through a digital material marketplace."
        ),
    },
    {
        "label": "differentiated agri-iot freshness chain",
        "language": "en",
        "expected": "strong",
        "text": (
            "Project: FreshTrace\nProblem: Small farms lose 30% of produce "
            "because cold-chain and demand signals are invisible.\n"
            "Description: A low-cost solar sensor and AI demand-prediction "
            "system that bundles coop harvests, optimizes routing to "
            "restaurants, and issues traceable freshness credits per batch."
        ),
    },
]


def build_ideas(
    frame: pd.DataFrame, n_english: int, n_arabic: int, seed: int
) -> list[dict[str, str]]:
    """Sample real 2024-2025 ideas as problem+description prompts."""
    frame = frame[frame["year"].isin([2024, 2025])].reset_index(drop=True)
    english = frame[
        frame["problem_en"].notna() & frame["solution_en"].notna()
    ]
    arabic_pool = frame[
        frame["problem"].fillna("").astype(str).map(
            lambda value: bool(ARABIC_RE.search(value))
        )
        & frame["solution"].notna()
    ]
    rng = random.Random(seed)
    english_ids = rng.sample(
        english.index.tolist(), min(n_english, len(english))
    )
    arabic_ids = rng.sample(
        arabic_pool.index.tolist(), min(n_arabic, len(arabic_pool))
    )
    ideas: list[dict[str, str]] = []
    for index in english_ids:
        row = frame.loc[index]
        ideas.append(
            {
                "language": "en",
                "project": str(row.get("project_name") or "")[:80],
                "text": (
                    f"Project: {row.get('project_name', '')}\n"
                    f"Problem: {row['problem_en']}\n"
                    f"Description: {row['solution_en']}"
                ),
            }
        )
    for index in arabic_ids:
        row = frame.loc[index]
        ideas.append(
            {
                "language": "ar",
                "project": str(row.get("project_name") or "")[:80],
                "text": (
                    f"المشروع: {row.get('project_name', '')}\n"
                    f"المشكلة: {row['problem']}\n"
                    f"الوصف: {row['solution']}"
                ),
            }
        )
    return ideas


def build_curated_ideas() -> list[dict[str, str]]:
    return [dict(case) for case in CURATED_CASES]


class OfflineLLM:
    """Scripted fake LLM used only for harness smoke checks."""

    def complete(
        self,
        messages,
        tools=None,
        json_mode=False,
        temperature=0.2,
        max_tokens=2500,
    ):
        payload = {
            "innovation_validation": {
                "dimensions": {
                    name: {
                        "score": 4.0,
                        "rationale": "Offline fixture rationale.",
                        "evidence": [],
                    }
                    for name, _weight in idea_agent.SCORING_RUBRIC
                },
                "total_score": 20.0,
                "verdict": "Promising",
                "sources": [
                    {
                        "title": "Offline fixture source",
                        "url": "https://example.com/offline",
                        "snippet": "Offline fixture snippet.",
                    }
                ],
                "risks": [],
                "recommendations": [],
            },
            "dashboard_insights": {
                "summary": "Offline fixture dashboard summary.",
                "insights": ["Offline fixture insight."],
                "kpis": {},
                "snapshot": {},
            },
        }
        return {"content": json.dumps(payload), "tool_calls": []}


class OfflineSearcher:
    def available(self):
        return True

    def search(self, query, max_results=3):
        return [
            idea_agent.SearchSource(
                title="Offline fixture source",
                url="https://example.com/offline",
                snippet="Offline fixture snippet.",
            )
        ]


def ground_truth(analyzer: idea_agent.DashboardAnalyzer) -> dict[str, Any]:
    return {
        "kpis": analyzer.kpis(),
        "top_sector": analyzer.top_categories("Sector", 1)
        if "Sector" in analyzer.applications.columns
        else {},
    }


def check_report(
    report: idea_agent.AgentReport, ground: dict[str, Any]
) -> dict[str, bool]:
    """Return pass/fail flags for every evaluation criterion."""
    checks: dict[str, bool] = {
        "report_parse": report.score is not None and report.dashboard is not None,
        "dimensions_complete": False,
        "scores_in_range": False,
        "weighted_consistent": False,
        "sources_valid": False,
        "dashboard_nonempty": False,
        "dashboard_kpis_match": False,
        "dashboard_narrative_grounded": False,
        "no_errors": not report.errors,
    }
    score = report.score
    dashboard = report.dashboard
    if score is not None:
        dimensions = score.dimensions or {}
        names = {name for name, _weight in idea_agent.SCORING_RUBRIC}
        checks["dimensions_complete"] = set(dimensions) == names and all(
            dimension.rationale
            and "did not provide" not in dimension.rationale.lower()
            for dimension in dimensions.values()
        )
        checks["scores_in_range"] = all(
            0 <= dimension.score <= 5 for dimension in dimensions.values()
        )
        if score.total_score is not None and dimensions:
            expected = round(
                5
                * sum(
                    dimensions[name].score * weight
                    for name, weight in idea_agent.SCORING_RUBRIC
                ),
                1,
            )
            checks["weighted_consistent"] = abs(
                score.total_score - expected
            ) <= 0.6
        checks["sources_valid"] = bool(score.sources) and all(
            source.url.lower().startswith(("http://", "https://"))
            for source in score.sources
        )
    if dashboard is not None:
        checks["dashboard_nonempty"] = (
            len(dashboard.summary or "") > 20 and bool(dashboard.insights)
        )
        checks["dashboard_kpis_match"] = bool(ground["kpis"]) and (
            dashboard.kpis == ground["kpis"]
        )
        narrative = (dashboard.summary or "") + " " + " ".join(
            dashboard.insights or []
        )
        expected_numbers = [
            str(ground["kpis"]["applications"]),
            str(ground["kpis"]["acceptance_rate_pct"]),
        ]
        checks["dashboard_narrative_grounded"] = any(
            number in narrative for number in expected_numbers
        )
    return checks


def run(
    csv_path: str,
    samples: int,
    seed: int,
    offline: bool,
    sleep: float,
    max_retries: int,
) -> dict[str, Any]:
    frame = pd.read_csv(csv_path, encoding="utf-8-sig")
    subset = frame[frame["year"].isin([2024, 2025])]
    # Idea sampling excludes 2023 and dashboard analysis runs on 2024-2025 only.
    ideas = build_ideas(frame, max(0, samples - 2), 2, seed)
    analyzer = idea_agent.DashboardAnalyzer(subset)
    ground = ground_truth(analyzer)

    if offline:
        client = OfflineLLM()
        searcher = OfflineSearcher()
    else:
        client = idea_agent.DeepSeekClient(max_retries=max_retries)
        searcher = idea_agent.TavilySearch()

    rows: list[dict[str, Any]] = []
    criteria: dict[str, list[bool]] = {}
    print(
        f"Evaluating {len(ideas)} real ideas from "
        f"{len(subset)} rows (2024-2025 only, 2023 excluded)..."
    )
    for index, sample in enumerate(ideas, 1):
        started = time.monotonic()
        try:
            report = idea_agent.run_agent(
                idea_text=sample["text"],
                applications=subset,
                client=client,
                searcher=searcher,
            )
            checks = check_report(report, ground)
            error = None
        except Exception as exc:
            checks = {
                "report_parse": False,
                "dimensions_complete": False,
                "scores_in_range": False,
                "weighted_consistent": False,
                "sources_valid": False,
                "dashboard_nonempty": False,
                "dashboard_kpis_match": False,
                "dashboard_narrative_grounded": False,
                "no_errors": False,
            }
            error = str(exc)[:300]
        elapsed = round(time.monotonic() - started, 2)
        for key, value in checks.items():
            criteria.setdefault(key, []).append(value)
        rows.append(
            {
                "index": index,
                "language": sample["language"],
                "project": sample["project"],
                "total_score": (
                    report.score.total_score if "report" in locals() and report.score else None
                ),
                "verdict": (
                    report.score.verdict if "report" in locals() and report.score else None
                ),
                "sources": (
                    len(report.score.sources) if "report" in locals() and report.score else 0
                ),
                "seconds": elapsed,
                "checks": checks,
                "error": error,
            }
        )
        passed = sum(checks.values())
        print(
            f"[{index}/{len(ideas)}] {sample['language']} "
            f"{passed}/{len(checks)} checks | "
            f"score={rows[-1]['total_score']} | {elapsed}s"
        )
        if not offline and sleep:
            time.sleep(sleep)

    summary = {
        "csv": csv_path,
        "years_included": [2024, 2025],
        "years_excluded": [2023],
        "subset_rows": int(len(subset)),
        "samples": len(rows),
        "model": idea_agent.AGENT_MODEL,
        "live": not offline,
        "ground_truth": ground,
        "criteria_pass_rate": {
            key: f"{sum(values)}/{len(values)}"
            for key, values in criteria.items()
        },
        "latency_seconds": {
            "mean": round(statistics.mean(row["seconds"] for row in rows), 2)
            if rows
            else None,
            "p50": round(statistics.median(row["seconds"] for row in rows), 2)
            if rows
            else None,
            "max": max((row["seconds"] for row in rows), default=0),
        },
        "samples": rows,
    }
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print("\n=== EVALUATION SUMMARY ===")
    print(
        f"Data: {summary['subset_rows']} rows from {summary['years_included']} "
        f"(2023 excluded); {summary['samples']} sampled ideas; "
        f"model={summary['model']}; live={summary['live']}"
    )
    for criterion, rate in summary["criteria_pass_rate"].items():
        print(f"  {criterion:<32} {rate}")
    latency = summary["latency_seconds"]
    print(
        f"  latency (s): mean={latency['mean']} p50={latency['p50']} "
        f"max={latency['max']}"
    )


def run_cases(
    csv_path: str,
    offline: bool,
    sleep: float,
    max_retries: int,
) -> dict[str, Any]:
    """Evaluate curated good/bad/edge ideas as controls."""
    frame = pd.read_csv(csv_path, encoding="utf-8-sig")
    subset = frame[frame["year"].isin([2024, 2025])]
    analyzer = idea_agent.DashboardAnalyzer(subset)
    ground = ground_truth(analyzer)
    client: Any = OfflineLLM() if offline else idea_agent.DeepSeekClient(
        max_retries=max_retries
    )
    searcher: Any = OfflineSearcher() if offline else idea_agent.TavilySearch()

    rows: list[dict[str, Any]] = []
    print(f"Evaluating {len(CURATED_CASES)} curated control cases ...")
    for index, case in enumerate(build_curated_ideas(), 1):
        started = time.monotonic()
        try:
            report = idea_agent.run_agent(
                idea_text=case["text"],
                applications=subset,
                client=client,
                searcher=searcher,
            )
            checks = check_report(report, ground)
            total = report.score.total_score if report.score else None
            expected = case["expected"]
            if expected == "weak":
                direction_ok = total is not None and total <= 12
            elif expected == "strong":
                direction_ok = total is not None and total >= 19
            else:
                direction_ok = checks["report_parse"]
            checks["direction_expected"] = direction_ok
            error = None
        except Exception as exc:
            checks = {
                "report_parse": False,
                "dimensions_complete": False,
                "scores_in_range": False,
                "weighted_consistent": False,
                "sources_valid": False,
                "dashboard_nonempty": False,
                "dashboard_kpis_match": False,
                "dashboard_narrative_grounded": False,
                "no_errors": False,
                "direction_expected": False,
            }
            total = None
            error = str(exc)[:300]
        elapsed = round(time.monotonic() - started, 2)
        rows.append(
            {
                "index": index,
                "label": case["label"],
                "language": case["language"],
                "expected": case["expected"],
                "total_score": total,
                "verdict": (
                    report.score.verdict
                    if "report" in locals()
                    and report.score is not None
                    else None
                ),
                "sources": (
                    len(report.score.sources)
                    if "report" in locals()
                    and report.score is not None
                    else 0
                ),
                "seconds": elapsed,
                "checks": checks,
                "error": error,
            }
        )
        passed = sum(checks.values())
        print(
            f"[{index}/{len(CURATED_CASES)}] {case['label']} "
            f"({case['expected']}) {passed}/{len(checks)} checks | "
            f"score={total} | {elapsed}s"
        )
        if not offline and sleep:
            time.sleep(sleep)

    criteria: dict[str, list[bool]] = {}
    for row in rows:
        for key, value in row["checks"].items():
            criteria.setdefault(key, []).append(value)
    return {
        "csv": csv_path,
        "mode": "curated-cases",
        "model": idea_agent.AGENT_MODEL,
        "live": not offline,
        "cases": rows,
        "criteria_pass_rate": {
            key: f"{sum(values)}/{len(values)}"
            for key, values in criteria.items()
        },
    }


def print_cases_summary(summary: dict[str, Any]) -> None:
    print("\n=== CURATED CASE SUMMARY ===")
    for criterion, rate in summary["criteria_pass_rate"].items():
        print(f"  {criterion:<28} {rate}")
    print("\n  case                                    score  verdict")
    for case in summary["cases"]:
        print(
            f"  {case['label']:<40} {str(case['total_score']):>5}  "
            f"{case['verdict']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--suite",
        choices=["random", "cases"],
        default="random",
        help="random=sampled real ideas (default), cases=curated controls.",
    )
    parser.add_argument("--sleep", type=float, default=2.5)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.suite == "cases":
        summary = run_cases(
            csv_path=args.csv,
            offline=args.offline,
            sleep=args.sleep,
            max_retries=args.max_retries,
        )
        print_cases_summary(summary)
        if args.out:
            Path(args.out).write_text(
                json.dumps(
                    summary, ensure_ascii=False, indent=2, default=str
                )
            )
            print(f"\nSaved full results to {args.out}")
        return
    summary = run(
        csv_path=args.csv,
        samples=args.samples,
        seed=args.seed,
        offline=args.offline,
        sleep=args.sleep,
        max_retries=args.max_retries,
    )
    print_summary(summary)
    if args.out:
        Path(args.out).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str)
        )
        print(f"\nSaved full results to {args.out}")


if __name__ == "__main__":
    main()
