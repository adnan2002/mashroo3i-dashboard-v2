"""Production scorer for the Brinc CatBoost acceptance model.

Self-contained deployment: the project's ``model/`` folder holds
``final_model.joblib``, ``final_threshold.json``, and the vendored
``features_build.py`` (feature engineering). No dependency on
``~/Desktop/filter_3`` at runtime.

The model directory is configurable with ``BRINC_MODEL_DIR``; it defaults to
``<project>/model`` and falls back to the old filter_3 artifact dir for dev.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
FILTER3_ROOT = Path(os.path.expanduser("~/Desktop/filter_3"))
DEV_MODEL_DIR = FILTER3_ROOT / "artifacts" / "brinc_model_full"
REAL_DATA_CSV = FILTER3_ROOT / "dashboard_ready.csv"
MODEL_DIR = Path(os.getenv("BRINC_MODEL_DIR", str(PROJECT_ROOT / "model")))
if not (MODEL_DIR / "final_model.joblib").exists() and (
    DEV_MODEL_DIR / "final_model.joblib"
).exists():
    MODEL_DIR = DEV_MODEL_DIR  # dev fallback only

OUTCOME_COLS = ("outcome", "outcome_clean", "is_winner")


class ModelUnavailableError(RuntimeError):
    """Raised when the Brinc model artifacts or dependencies are missing."""


def _require(path: Path) -> Path:
    if not path.exists():
        raise ModelUnavailableError(f"Missing Brinc artifact: {path}")
    return path


def _ensure_feature_module() -> None:
    """Make the vendored ``features_build`` importable from the model dir."""
    if str(MODEL_DIR) not in sys.path:
        sys.path.insert(0, str(MODEL_DIR))


def _extract_columns(model) -> tuple[list[str], list[str]]:
    """Recover num/cat column order from the saved sklearn Pipeline."""
    if not hasattr(model, "named_steps") or "pre" not in model.named_steps:
        raise ModelUnavailableError(
            "final_model.joblib is not the expected Brinc sklearn Pipeline."
        )
    pre = model.named_steps["pre"]
    if not hasattr(pre, "transformers"):
        raise ModelUnavailableError(
            "final_model.joblib does not expose a ColumnTransformer."
        )
    num_cols = [
        column
        for name, _transformer, columns in pre.transformers
        if name == "num"
        for column in columns
    ]
    cat_cols = [
        column
        for name, _transformer, columns in pre.transformers
        if name == "cat"
        for column in columns
    ]
    return num_cols, cat_cols


def load_scorer() -> dict[str, Any]:
    """Load the model, threshold, and pipeline feature columns once."""
    try:
        import joblib  # noqa: PLC0415
    except ImportError as exc:
        raise ModelUnavailableError(
            "joblib is not installed; run: pip install joblib"
        ) from exc

    _ensure_feature_module()
    try:
        from features_build import engineer_features  # noqa: PLC0415, F401
    except ImportError as exc:
        raise ModelUnavailableError(
            "model/features_build.py is not importable; the model bundle is "
            "incomplete."
        ) from exc

    model_path = _require(MODEL_DIR / "final_model.joblib")
    threshold = json.loads(
        _require(MODEL_DIR / "final_threshold.json").read_text()
    )["threshold"]
    model = joblib.load(model_path)
    num_cols, cat_cols = _extract_columns(model)
    return {
        "model": model,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "threshold": float(threshold),
        "model_dir": str(MODEL_DIR),
        "n_features": len(num_cols) + len(cat_cols),
    }


def score_with_model(raw: pd.DataFrame) -> pd.DataFrame:
    """Score applicants with the saved Brinc model (auto-engineered input).

    Returns one row per input row with identity, accept probability,
    predicted label at the saved threshold, and a model rank.
    """
    scorer = load_scorer()
    _ensure_feature_module()
    from features_build import engineer_features  # noqa: PLC0415

    frame = raw.copy()
    # The restored engineer_features derives y internally; a placeholder
    # keeps it working when outcome columns were intentionally removed.
    if "outcome_clean" not in frame.columns:
        frame["outcome_clean"] = ""
    # Arrow-backed strings reject unicode regexes like \u0600; force the
    # modeling code down the plain-Python (object) string path.
    string_columns = frame.select_dtypes(include=["string", "object"]).columns
    frame[string_columns] = frame[string_columns].astype(object)
    engineered, _y, _sets, _cats, _nums, _groups = engineer_features(frame)

    X = engineered[scorer["num_cols"] + scorer["cat_cols"]]
    assert not any(column in OUTCOME_COLS for column in X.columns), (
        "Outcome columns must not enter the model feature matrix."
    )
    proba = scorer["model"].predict_proba(X)[:, 1]

    out = pd.DataFrame(
        {
            "identity": engineered["identity"],
            "project_name": frame.get("project_name"),
            "year": frame.get("year"),
            "Sector": frame.get("Sector"),
            "date_of_birth": frame.get("date_of_birth"),
            "problem_en": frame.get("problem_en"),
            "solution_en": frame.get("solution_en"),
            "problem": frame.get("problem"),
            "solution": frame.get("solution"),
            "accept_probability": proba,
            "predicted_accepted": (proba >= scorer["threshold"]).astype(int),
        },
        index=frame.index,
    )
    out = out.sort_values(
        "accept_probability", ascending=False, kind="mergesort"
    ).reset_index(drop=True)
    out["model_rank"] = range(1, len(out) + 1)
    out["prediction_threshold"] = scorer["threshold"]
    return out


def available() -> tuple[bool, str]:
    """Report whether the bundled model is ready to score."""
    _ensure_feature_module()
    try:
        from features_build import engineer_features  # noqa: PLC0415, F401
    except ImportError as exc:
        return False, f"Model bundle incomplete (features_build): {exc}"
    missing = [
        path
        for path in (
            MODEL_DIR / "final_model.joblib",
            MODEL_DIR / "final_threshold.json",
        )
        if not path.exists()
    ]
    if missing:
        return False, "Missing Brinc artifacts: " + ", ".join(
            str(path) for path in missing
        )
    try:
        scorer = load_scorer()
    except ModelUnavailableError as exc:
        return False, str(exc)
    return (
        True,
        f"Model is ready!",
    )
