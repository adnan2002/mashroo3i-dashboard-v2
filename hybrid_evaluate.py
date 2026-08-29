"""Hybrid evaluator: Brinc CatBoost acceptance model + innovation agent.

Measures how well the saved Brinc model (prob. an applicant is Accepted)
combined with the agent's web-researched innovation score (0-25) predicts
the real acceptance label.

The evaluation removes every outcome column from the inputs and uses honest
out-of-fold probabilities (never the leaky in-sample ``predictions.csv``).

Run with the model environment (has catboost/sklearn) after installing the
agent packages there::

    ~/Desktop/filter_3/venv/bin/python hybrid_evaluate.py --stage all

Stages::

    --stage model    build/load OOF model probabilities
    --stage agent    score the sampled ideas with the LLM agent
    --stage metrics  combine and report (requires both stages done)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FILTER2_ROOT = Path(__file__).resolve().parent
FILTER3_ROOT = Path(os.path.expanduser("~/Desktop/filter_3"))
for path in (FILTER2_ROOT, FILTER3_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

MODEL_DIR = FILTER3_ROOT / "artifacts" / "brinc_model_full"
DATA_CSV = FILTER3_ROOT / "dashboard_ready.csv"
OOF_CACHE = Path("/tmp/hybrid_oof.npz")
AGENT_CACHE = Path("/tmp/hybrid_agent_scores.json")
RESULT_CACHE = Path("/tmp/hybrid_eval.json")
OUTCOME_COLS = ("outcome", "outcome_clean", "is_winner")
DEFAULT_WEIGHT = 0.6


# ---------------------------------------------------------------------------
# Pure helpers (no model/Llibs required - unit tested from either env)
# ---------------------------------------------------------------------------


def strip_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with every known outcome column removed."""
    return frame.drop(columns=[c for c in OUTCOME_COLS if c in frame.columns])


def _text_value(value: Any) -> str:
    if value is None or isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in ("nan", "none") else text


def idea_text_for(row: pd.Series) -> str:
    """Build the problem+description prompt from English (or Arabic) fields."""
    problem_en = _text_value(row.get("problem_en"))
    solution_en = _text_value(row.get("solution_en"))
    if problem_en or solution_en:
        return (
            f"Project: {_text_value(row.get('project_name'))}\n"
            f"Problem: {problem_en or _text_value(row.get('problem'))}\n"
            f"Description: {solution_en or _text_value(row.get('solution'))}"
        )
    return (
        f"Project: {_text_value(row.get('project_name'))}\n"
        f"Problem: {_text_value(row.get('problem'))}\n"
        f"Description: {_text_value(row.get('solution'))}"
    )


def stratified_sample(
    features: pd.DataFrame,
    labels: pd.Series,
    per_class: int = 15,
    seed: int = 42,
) -> pd.Index:
    """Dedupe by identity, then sample per_class positives and negatives."""
    uniq = features.drop_duplicates(subset="identity", keep="first")
    labels = labels.loc[uniq.index]
    rng = np.random.RandomState(seed)
    indices = []
    for label in (1, 0):
        pool = labels.index[labels == label].to_numpy()
        if len(pool) == 0:
            continue
        take = min(per_class, len(pool))
        indices.extend(rng.choice(pool, size=take, replace=False).tolist())
    selected = pd.Index(indices)
    if len(selected) != len(set(selected)):
        raise AssertionError("Stratified sample contains duplicates")
    return selected


def rank_pct(series: pd.Series) -> pd.Series:
    return series.rank(pct=True)


def blend_scores(model_rank: pd.Series, agent_rank: pd.Series, weight: float) -> pd.Series:
    return weight * model_rank + (1.0 - weight) * agent_rank


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks (1-based) for a numeric array, breaking ties by mean."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and sorted_values[end] == sorted_values[index]:
            end += 1
        ranks[order[index:end]] = (index + end - 1) / 2.0 + 1.0
        index = end
    return ranks


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney ROC AUC with tie handling (pure numpy)."""
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = _average_ranks(scores)
    sum_pos_ranks = ranks[labels == 1].sum()
    return float(
        (sum_pos_ranks - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def _pr_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Average precision (pure numpy)."""
    order = np.argsort(scores, kind="mergesort")
    labels = labels[order]
    scores = scores[order]
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    change = np.flatnonzero(scores[1:] != scores[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(scores)]
    precision: list[float] = []
    recall: list[float] = []
    tp = 0
    fp = 0
    # Walk thresholds from highest score down; ties count together, like
    # sklearn's precision_recall_curve.
    for start, end in zip(starts[::-1], ends[::-1]):
        tp += int(labels[start:end].sum())
        fp += int((1 - labels[start:end]).sum())
        precision.append(tp / (tp + fp))
        recall.append(tp / positives)
    precision = np.array(precision)
    recall = np.array(recall)
    # sklearn reports the curve from "all samples positive" upward.
    precision = precision[::-1]
    recall = recall[::-1]
    # Add sklearn's sentinel (precision=1 at recall=0) so the final step
    # (recall 1 -> 0) is counted with precision 1.
    precision = np.concatenate([precision, [1.0]])
    recall = np.concatenate([recall, [0.0]])
    return float(np.sum((recall[:-1] - recall[1:]) * precision[:-1]))


def _spearman(labels: np.ndarray, scores: np.ndarray) -> float:
    """Pearson correlation between rank(scores) and the binary labels."""
    ranks = _average_ranks(scores)
    if ranks.std() == 0 or labels.std() == 0:
        return float("nan")
    return float(np.corrcoef(ranks, labels)[0, 1])


def prediction_metrics(score: pd.Series, label: pd.Series) -> dict[str, float]:
    """ROC AUC, PR AUC, and Spearman correlation for a score vs binary labels."""
    mask = score.notna() & label.notna()
    y = label[mask].astype(int)
    s = score[mask]
    if len(y) < 3 or y.nunique() < 2 or s.nunique() < 2:
        return {"auc": None, "pr_auc": None, "spearman": None}
    y_values = y.to_numpy(dtype=float)
    s_values = s.to_numpy(dtype=float)
    return {
        "auc": _roc_auc(y_values, s_values),
        "pr_auc": _pr_auc(y_values, s_values),
        "spearman": _spearman(y_values, s_values),
    }


def load_agent_cache() -> dict[str, dict[str, Any]]:
    if AGENT_CACHE.exists():
        return json.loads(AGENT_CACHE.read_text())
    return {}


def save_agent_cache(cache: dict[str, dict[str, Any]]) -> None:
    AGENT_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Model / agent stages (require the right environments)
# ---------------------------------------------------------------------------


def load_model_stack():
    """Lazy import of the heavy modeling module (filter_3 environment)."""
    from brinc_modeling import (  # noqa: PLC0415
        SEED,
        build_estimator,
        engineer_features,
        pos_weight,
    )

    return SEED, build_estimator, engineer_features, pos_weight


def grouped_oof(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    repeats: int = 1,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, list[float]]:
    """Group-CV OOF probabilities (same strategy as the Brinc finalize run)."""
    from copy import deepcopy

    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    all_oof: list[np.ndarray] = []
    aucs: list[float] = []
    for rep in range(repeats):
        cv = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed + rep * 101
        )
        oof = np.zeros(len(X))
        for train_idx, val_idx in cv.split(X, y, groups):
            model = deepcopy(estimator)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
        all_oof.append(oof)
        aucs.append(float(roc_auc_score(y, oof)))
    combined = np.mean(np.vstack(all_oof), axis=0) if all_oof else np.zeros(len(X))
    return combined, aucs


def run_model_stage(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Engineer 2024-25 features and return row-aligned OOF probabilities."""
    SEED, build_estimator, engineer_features, pos_weight = load_model_stack()
    config = json.loads((MODEL_DIR / "config.json").read_text())
    feature_cols = json.loads((MODEL_DIR / "feature_columns.json").read_text())
    num_cols, cat_cols = feature_cols["num"], feature_cols["cat"]

    engineered, _y, _sets, cat_from_model, all_num, _groups = engineer_features(
        raw
    )
    # The outcome column is only used to derive y; never part of the model X.
    X = engineered[num_cols + cat_cols]
    assert not any(column in OUTCOME_COLS for column in X.columns)
    y = pd.Series(_y, index=engineered.index)
    groups = engineered["identity"]

    estimator = build_estimator(
        "catboost",
        dict(config["params"]),
        impute="median",
        scale="standard",
        num_cols=num_cols,
        cat_cols=cat_cols,
        weighted=True,
        sample=None,
        n_jobs=-1,
        seed=SEED,
        pw=pos_weight(y),
    )
    oof, fold_aucs = grouped_oof(estimator, X, y, groups)
    np.savez_compressed(
        OOF_CACHE,
        index=engineered.index.to_numpy(),
        oof=oof,
        y=y.to_numpy(),
    )
    print(f"model OOF folds AUC={[round(a, 3) for a in fold_aucs]}")
    return engineered, y, oof


def run_agent_stage(
    engineered: pd.DataFrame,
    raw_no_outcomes: pd.DataFrame,
    selected_index: pd.Index,
) -> dict[str, dict[str, Any]]:
    """Score sampled ideas with the DeepSeek+Tavily agent, resumable."""
    import idea_agent  # noqa: PLC0415

    cache = load_agent_cache()
    client = idea_agent.DeepSeekClient(max_retries=3)
    searcher = idea_agent.TavilySearch()
    for position, index in enumerate(selected_index, 1):
        row = engineered.loc[index]
        identity = str(row["identity"])
        if identity in cache and cache[identity].get("score") is not None:
            print(f"[{position}/{len(selected_index)}] cached {identity}")
            continue
        text = idea_text_for(row)
        print(f"[{position}/{len(selected_index)}] scoring {identity[:60]} ...")
        report = idea_agent.run_agent(
            idea_text=text,
            applications=raw_no_outcomes,
            client=client,
            searcher=searcher,
        )
        score = report.score
        cache[identity] = {
            "project": str(row.get("project_name") or ""),
            "year": int(row.get("year") or 0),
            "score": score.total_score if score else None,
            "verdict": score.verdict if score else None,
            "sources": len(score.sources) if score else 0,
            "errors": report.errors,
        }
        save_agent_cache(cache)
    return cache


def run_metrics(
    engineered: pd.DataFrame,
    y: pd.Series,
    oof: np.ndarray,
    agent_cache: dict[str, dict[str, Any]],
    weight: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    oof_by_index = pd.Series(oof, index=engineered.index)
    for index in engineered.index:
        identity = str(engineered.loc[index, "identity"])
        agent = agent_cache.get(identity)
        rows.append(
            {
                "identity": identity,
                "project": str(engineered.loc[index, "project_name"] or ""),
                "year": int(engineered.loc[index, "year"]),
                "label": int(y.loc[index]),
                "model_prob": float(oof_by_index.loc[index]),
                "agent_score": agent["score"] if agent else None,
            }
        )
    frame = pd.DataFrame(rows)
    frame["model_rank"] = rank_pct(frame["model_prob"])
    frame["agent_rank"] = rank_pct(frame["agent_score"])
    frame["hybrid"] = blend_scores(
        frame["model_rank"], frame["agent_rank"], weight
    )

    model_m = prediction_metrics(frame["model_prob"], frame["label"])
    sampled = frame[frame["agent_score"].notna()]
    model_sampled_m = prediction_metrics(sampled["model_prob"], sampled["label"])
    agent_m = prediction_metrics(frame["agent_score"], frame["label"])
    hybrid_m = prediction_metrics(frame["hybrid"], frame["label"])
    sweep = {}
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        blended = blend_scores(frame["model_rank"], frame["agent_rank"], w)
        sweep[str(w)] = prediction_metrics(blended, frame["label"])["auc"]

    return {
        "weight": weight,
        "model_only": model_m,
        "model_only_scored_rows": model_sampled_m,
        "agent_only": agent_m,
        "hybrid": hybrid_m,
        "weight_sweep_auc": sweep,
        "rows": len(frame),
        "scored_rows": int(frame["agent_score"].notna().sum()),
        "samples": rows,
    }


def model_top_k(
    engineered: pd.DataFrame,
    y: pd.Series,
    oof: np.ndarray,
    top_k: int,
) -> pd.Index:
    """Shortlist: dedupe identities, keep the row with the best OOF prob."""
    frame = pd.DataFrame(
        {
            "label": y,
            "model_prob": pd.Series(oof, index=engineered.index),
            "identity": engineered["identity"],
        }
    )
    best = frame.sort_values("model_prob", ascending=False).drop_duplicates(
        "identity"
    )
    return pd.Index(best.head(top_k).index)


def run_cascade_metrics(
    engineered: pd.DataFrame,
    y: pd.Series,
    oof: np.ndarray,
    cache: dict[str, dict[str, Any]],
    top_k: int,
    weight: float,
) -> dict[str, Any]:
    """Evaluate the model-first -> agent-second cascade on the shortlist."""
    frame = pd.DataFrame(
        {
            "label": y,
            "model_prob": pd.Series(oof, index=engineered.index),
            "identity": engineered["identity"],
            "project": engineered["project_name"],
            "year": engineered["year"],
        }
    )
    frame["agent_score"] = frame["identity"].map(
        lambda identity: (cache.get(str(identity)) or {}).get("score")
    )
    best = frame.sort_values("model_prob", ascending=False).drop_duplicates(
        "identity"
    )
    top = best.head(top_k).copy().reset_index(drop=True)
    top["model_rank"] = np.arange(1, len(top) + 1)

    positives_above = int(y.sum())
    in_top_accepted = int(top["label"].sum())
    scored = top[top["agent_score"].notna()].copy()
    held_out = top["agent_score"].isna().sum()

    model_half = top.sort_values("model_prob", ascending=False).head(
        max(1, len(top) // 2)
    )
    agent_half = scored.sort_values(
        "agent_score", ascending=False
    ).head(max(1, len(top) // 2))

    # Lexicographic cascade: model rank primary, agent rank secondary.
    within_model = prediction_metrics(top["model_prob"], top["label"])
    within_agent = prediction_metrics(scored["agent_score"], scored["label"])
    combined = blend_scores(
        rank_pct(top["model_prob"]),
        rank_pct(top["agent_score"]),
        weight,
    )
    within_hybrid = prediction_metrics(combined, top["label"])

    return {
        "mode": "cascade",
        "primary_top_k": len(top),
        "top_k": top_k,
        "accepted_in_top_k": in_top_accepted,
        "total_positives_in_scope": positives_above,
        "model_precision_at_k": float(in_top_accepted / len(top)),
        "model_recall_at_k": float(
            in_top_accepted / positives_above if positives_above else 0.0
        ),
        "baseline_positive_rate": float(y.mean()),
        "primary_ranked_precision_top_half": float(model_half["label"].mean()),
        "secondary_agent_rerank_precision_top_half": float(
            agent_half["label"].mean()
        ),
        "auc_within_shortlist_model": within_model,
        "auc_within_shortlist_agent": within_agent,
        "auc_within_shortlist_hybrid": within_hybrid,
        "agent_scored_in_top_k": len(scored),
        "agent_missing_in_top_k": int(held_out),
        "shortlist": top[
            [
                "identity",
                "project",
                "year",
                "label",
                "model_prob",
                "agent_score",
                "model_rank",
            ]
        ].to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "model", "agent", "metrics"])
    parser.add_argument("--rows-per-class", type=int, default=15)
    parser.add_argument("--mode", choices=["hybrid", "cascade"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight", type=float, default=DEFAULT_WEIGHT)
    parser.add_argument("--out", default=str(RESULT_CACHE))
    args = parser.parse_args()
    stage = args.stage or "all"

    raw = pd.read_csv(DATA_CSV, encoding="utf-8-sig")
    subset = raw[raw["year"].isin([2024, 2025])].copy()
    raw_no_outcomes = strip_outcomes(subset)

    engineered = y = oof = None
    if stage in ("model", "agent", "all"):
        engineered, y, oof = run_model_stage(subset)
    else:
        loaded = np.load(OOF_CACHE, allow_pickle=True)
        index = pd.Index(loaded["index"])
        engineered = pd.DataFrame(index=index)
        y = pd.Series(loaded["y"], index=index)
        oof = loaded["oof"]
        # Rebuild needed columns for sampling/agent text from the raw subset.
        engineered = subset.loc[index].copy()
        engineered["identity"] = (
            engineered["project_name"].fillna("")
            + "|"
            + engineered["date_of_birth"].fillna("")
        )

    if stage in ("model", "all"):
        from sklearn.metrics import roc_auc_score

        print(
            "model-only AUC (full 2024-25 OOF): %.3f"
            % roc_auc_score(y, oof)
        )

    if args.mode == "cascade":
        selected = model_top_k(engineered, y, oof, args.top_k)
        print(
            f"cascade shortlist: top {len(selected)} by model OOF probability"
        )
    else:
        selected = stratified_sample(engineered, y, args.rows_per_class, args.seed)
    print(
        f"selected {len(selected)} rows "
        f"(positive={int(y.loc[selected].sum())}, "
        f"negative={int((1 - y.loc[selected]).sum())})"
    )

    if stage in ("agent", "all"):
        run_agent_stage(engineered, raw_no_outcomes, selected)

    cache = load_agent_cache()
    if args.mode == "cascade":
        result = run_cascade_metrics(
            engineered, y, oof, cache, args.top_k, args.weight
        )
    else:
        result = run_metrics(engineered, y, oof, cache, args.weight)
    result.update(
        {
            "scope": {"years": [2024, 2025], "rows": len(subset)},
            "model": "catboost (config.json params, OOF group CV)",
            "agent_model": "deepseek-v4-flash + Tavily",
            "outcome_columns_removed": list(OUTCOME_COLS),
            "seed": args.seed,
            "rows_per_class": args.rows_per_class,
            "mode": args.mode,
        }
    )
    Path(args.out).write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str)
    )
    if args.mode == "cascade":
        keys = [
            "model_precision_at_k",
            "model_recall_at_k",
            "baseline_positive_rate",
            "primary_ranked_precision_top_half",
            "secondary_agent_rerank_precision_top_half",
            "auc_within_shortlist_model",
            "auc_within_shortlist_agent",
            "auc_within_shortlist_hybrid",
            "agent_scored_in_top_k",
            "agent_missing_in_top_k",
        ]
        print(json.dumps({key: result[key] for key in keys}, indent=2))
    else:
        print(json.dumps(
            {
                "model_only": result["model_only"],
                "agent_only": result["agent_only"],
                "hybrid": result["hybrid"],
                "weight_sweep_auc": result["weight_sweep_auc"],
                "scored_rows": result["scored_rows"],
            },
            indent=2,
        ))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
