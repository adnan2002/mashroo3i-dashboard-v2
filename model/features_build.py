#!/usr/bin/env python3
"""
Brinc accelerator selection modeling on dashboard_ready.csv.

Constraint: only columns/values from dashboard_ready.csv are used.
No external APIs, LLMs, embeddings, fuzzy company lookups, or pretrained models.

Goal: recall-optimized (false-negative-focused) binary classification
of `outcome_clean == "Accepted"` using tree boosters, with a broad
trial-and-error search over features, imputation, scaling, resampling,
hyperparameters, and ensembles.

Run:
    ./venv/bin/python features_build.py            # full search
    ./venv/bin/python features_build.py --quick    # fast validation of the code path
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


SEED = 42
DATA_PATH = Path("dashboard_ready.csv")
OUT_DIR = Path("artifacts/brinc_model")
POSITIVE_LABEL = "Accepted"


# ---------------------------------------------------------------------------
# Small text/theme dictionaries (CSV-only, no external knowledge base)
# ---------------------------------------------------------------------------

TECH_TERMS = [
    "tech", "ai", "artificial intelligence", "software", "app", "application",
    "platform", "digital", "online", "e-commerce", "ecommerce", "automation",
    "data", "cloud", "smart", "iot", "mobile", "fintech", "robotics", "cyber",
    "web", "saas", "machine learning", "blockchain", "software", "system",
    "database", "website", "technology",
]

COMMON_SECTORS = {
    "Food & Beverage",
    "Retail & Consumer Goods",
    "Fashion, Textiles & Beauty",
    "Entertainment & Events",
    "Tourism & Hospitality",
    "Sports & Fitness",
    "Media, Design & Creative",
    "Services & Consulting",
    "Social & Community",
    "Transportation & Automotive",
}

THEME_KEYWORDS = {
    "food": ["food", "restaurant", "cafe", "coffee", "bakery", "meal", "cuisine",
             "catering", "recipe", "kitchen", "snack", "grocery", "beverage",
             "juice", "dessert", "healthy food"],
    "beauty": ["beauty", "hair", "salon", "makeup", "nail", "skincare", "cosmetic",
               "fashion", "clothing", "jewelry", "dress", "perfume", "apparel",
               "eyewear", "modest"],
    "tech": ["app", "platform", "software", "website", "online", "digital", "ai",
             "artificial intelligence", "data", "cloud", "automation", "mobile",
             "iot", "blockchain", "fintech", "cyber", "saas", "system", "technology",
             "smart", "application"],
    "health": ["health", "medical", "clinic", "doctor", "fitness", "wellness",
               "pharma", "therapy", "hospital", "nutrition", "care", "mental"],
    "education": ["education", "training", "course", "learning", "school", "teach",
                  "academy", "workshop", "student", "elearning", "tutor",
                  "curriculum", "skills"],
    "marketing": ["marketing", "social media", "advertising", "brand", "promotion",
                  "content", "influencer", "media", "design", "photography",
                  "seo", "campaign"],
    "events": ["event", "entertainment", "booking", "ticket", "festival",
               "concert", "tourism", "travel", "hotel", "experience", "leisure"],
    "transport": ["transport", "logistics", "delivery", "car", "vehicle",
                  "shipping", "ride", "driver", "fleet", "auto", "mobility",
                  "gps", "tracking"],
    "finance": ["finance", "funding", "investment", "bank", "insurance",
                "payment", "loan", "capital", "revenue", "profit", "money",
                "fintech", "financial"],
    "home": ["home", "real estate", "property", "interior", "furniture",
             "construction", "renovation", "rent", "housing", "clinic"],
    "environment": ["environment", "sustainability", "renewable", "solar",
                    "recycling", "green", "energy", "waste", "agriculture",
                    "farm", "water", "climate"],
    "services": ["service", "consulting", "maintenance", "cleaning", "repair",
                 "support", "agency", "professional", "concierge", "managed"],
}

TEXT_COLS_EN = [
    "problem_en", "solution_en", "differentiation_en", "sales_approach_en",
    "impact_en", "inspiration_en", "why_join_en", "program_expectations_en",
    "team_fit_en",
]

TEXT_COLS_RAW = [
    "problem", "solution", "differentiation", "sales_approach", "impact",
    "inspiration", "why_join", "program_expectations", "team_fit",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[brinc-model] {msg}", flush=True)


def safe_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def split_list(s) -> list[str]:
    if pd.isna(s) or not str(s).strip():
        return []
    return [p.strip() for p in re.split(r"\s*[;,]\s*", str(s)) if p.strip()]


def has_any(text, terms) -> bool:
    text = str(text).lower()
    return any(re.search(rf"\b{re.escape(t)}\b", text) for t in terms)


def word_count(s) -> int:
    if pd.isna(s):
        return 0
    return len(str(s).split())


def unique_ratio(s) -> float:
    words = [w for w in re.findall(r"[A-Za-z0-9\u0600-\u06FF]+", str(s).lower()) if w]
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def jaccard(a, b) -> float:
    wa = set(re.findall(r"[A-Za-z0-9\u0600-\u06FF]+", str(a).lower()))
    wb = set(re.findall(r"[A-Za-z0-9\u0600-\u06FF]+", str(b).lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def extract_business_age(text, app_year):
    text = str(text).lower()
    year_match = re.search(
        r"(?:since|from|in|est\.?|established|founded|started|operating|running|"
        r"launched|registered|opened)\s*(?:in|on|since)?\s*(20\d{2})",
        text,
    )
    if year_match:
        return max(0, int(app_year) - int(year_match.group(1)))
    duration = re.search(
        r"(?:operating|running|in business|established|founded|started|launched|"
        r"trading|open|selling)\s*(?:for|since|over)?\s*(\d{1,2})\s*(?:years?|yrs?)\b",
        text,
    )
    if duration:
        years = int(duration.group(1))
        return years if years <= 30 else np.nan
    return np.nan


# ---------------------------------------------------------------------------
# Metrics and CV
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_prob):
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        fbeta_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "fn": int(fn),
        "fn_rate": fn / max(1, fn + tp),
        "fp": int(fp),
        "fp_rate": fp / max(1, fp + tn),
    }


def run_cv(estimator, X, y, groups, n_splits=5, n_jobs=-1):
    from sklearn.model_selection import StratifiedGroupKFold

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    fold_metrics = []
    for train_idx, val_idx in cv.split(X, y, groups):
        model = deepcopy(estimator)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = model.predict_proba(X.iloc[val_idx])[:, 1]
        oof[val_idx] = proba
        fold_metrics.append(compute_metrics(y.iloc[val_idx], proba >= 0.5, proba))
    mean_metrics = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
    return mean_metrics, oof


def threshold_rows(oof, y, lo=0.05, hi=0.75, step=0.025, recall_floor=0.85):
    rows = []
    for t in np.arange(lo, hi + 1e-9, step):
        m = compute_metrics(y.to_numpy(), (oof >= t), oof)
        rows.append({**m, "threshold": float(t)})
    df_t = pd.DataFrame(rows)
    good = df_t.loc[df_t["recall"] >= recall_floor]
    if len(good):
        best = good.sort_values(["f2", "recall"], ascending=[False, False]).iloc[0]
    else:
        best = df_t.sort_values(["f2", "recall"], ascending=[False, False]).iloc[0]
    return best.to_dict(), df_t


def summarize_cv(m):
    return {
        "cv_accuracy": round(m["accuracy"], 4),
        "cv_precision": round(m["precision"], 4),
        "cv_recall": round(m["recall"], 4),
        "cv_f1": round(m["f1"], 4),
        "cv_f2": round(m["f2"], 4),
        "cv_roc_auc": round(m["roc_auc"], 4),
        "cv_pr_auc": round(m["pr_auc"], 4),
        "cv_fn_rate": round(m["fn_rate"], 4),
    }


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def pos_weight(y):
    counts = y.value_counts()
    return float(counts.get(0, 1) / max(1, counts.get(1, 1)))


def make_model(name, params=None, weighted=True, n_jobs=-1, seed=SEED, pw=1.0):
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
        VotingClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier

    params = dict(params or {})
    if name == "xgboost":
        model = XGBClassifier(
            n_estimators=params.pop("n_estimators", 500),
            learning_rate=params.pop("learning_rate", 0.05),
            max_depth=params.pop("max_depth", 5),
            min_child_weight=params.pop("min_child_weight", 3),
            subsample=params.pop("subsample", 0.85),
            colsample_bytree=params.pop("colsample_bytree", 0.85),
            reg_alpha=params.pop("reg_alpha", 0.1),
            reg_lambda=params.pop("reg_lambda", 1.0),
            gamma=params.pop("gamma", 0.0),
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=n_jobs,
            random_state=seed,
            **params,
        )
        if weighted:
            model.set_params(scale_pos_weight=pw)
        return model
    if name == "lightgbm":
        model = LGBMClassifier(
            n_estimators=params.pop("n_estimators", 500),
            learning_rate=params.pop("learning_rate", 0.05),
            num_leaves=params.pop("num_leaves", 31),
            min_child_samples=params.pop("min_child_samples", 20),
            subsample=params.pop("subsample", 0.85),
            colsample_bytree=params.pop("colsample_bytree", 0.85),
            reg_alpha=params.pop("reg_alpha", 0.1),
            reg_lambda=params.pop("reg_lambda", 1.0),
            n_jobs=n_jobs,
            random_state=seed,
            verbose=-1,
            **params,
        )
        if weighted:
            model.set_params(class_weight="balanced")
        return model
    if name == "catboost":
        model = CatBoostClassifier(
            iterations=params.pop("iterations", 500),
            learning_rate=params.pop("learning_rate", 0.05),
            depth=params.pop("depth", 6),
            l2_leaf_reg=params.pop("l2_leaf_reg", 3),
            bagging_temperature=params.pop("bagging_temperature", 1),
            random_strength=params.pop("random_strength", 1),
            auto_class_weights="Balanced" if weighted else None,
            random_seed=seed,
            thread_count=n_jobs if n_jobs and n_jobs > 0 else -1,
            verbose=0,
            allow_writing_files=False,
            **params,
        )
        return model
    if name == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.06,
            random_state=seed,
            **params,
        )
        if weighted:
            model.set_params(class_weight="balanced")
        return model
    if name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=3,
            max_features="sqrt",
            n_jobs=n_jobs,
            random_state=seed,
            **params,
        )
        if weighted:
            model.set_params(class_weight="balanced")
        return model
    if name == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=3,
            max_features="sqrt",
            n_jobs=n_jobs,
            random_state=seed,
            **params,
        )
        if weighted:
            model.set_params(class_weight="balanced")
        return model
    if name == "logistic_regression":
        model = LogisticRegression(max_iter=3000, random_state=seed, n_jobs=n_jobs, **params)
        if weighted:
            model.set_params(class_weight="balanced")
        return model
    raise ValueError(name)


def build_preprocessor(num_cols, cat_cols, impute="median", scale="standard"):
    from sklearn.compose import ColumnTransformer
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (
        MinMaxScaler,
        OneHotEncoder,
        RobustScaler,
        StandardScaler,
    )

    if impute == "none":
        num_pipe = [
            ("impute", "passthrough"),
        ]
    else:
        if impute == "mean":
            imp = SimpleImputer(strategy="mean")
        elif impute == "median":
            imp = SimpleImputer(strategy="median")
        elif impute == "knn":
            imp = KNNImputer(n_neighbors=5)
        elif impute == "iterative":
            imp = IterativeImputer(
                max_iter=5, random_state=SEED, n_nearest_features=10,
                sample_posterior=False,
            )
        else:
            raise ValueError(impute)
        num_pipe = [
            ("impute", imp),
        ]
    if impute != "none":
        if scale == "standard":
            num_pipe.append(("scale", StandardScaler()))
        elif scale == "robust":
            num_pipe.append(("scale", RobustScaler()))
        elif scale == "minmax":
            num_pipe.append(("scale", MinMaxScaler()))

    cat_pipe = [
        ("fill", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
    return ColumnTransformer(
        [
            ("num", Pipeline(num_pipe), num_cols),
            ("cat", Pipeline(cat_pipe), cat_cols),
        ]
    )


def build_estimator(name, params, impute="median", scale="standard",
                    num_cols=None, cat_cols=None, weighted=True, sample=None,
                    n_jobs=-1, seed=SEED, pw=1.0):
    from imblearn.combine import SMOTEENN
    from imblearn.over_sampling import SMOTE, SMOTENC
    from imblearn.pipeline import make_pipeline as make_imb_pipeline
    from imblearn.under_sampling import RandomUnderSampler
    from sklearn.pipeline import Pipeline

    model = make_model(name, params, weighted=weighted, n_jobs=n_jobs, seed=seed, pw=pw)
    if sample:
        pre = build_preprocessor(num_cols, cat_cols, impute=impute, scale=scale)
        sampler = {
            "smote": SMOTE(random_state=seed),
            "smoteenn": SMOTEENN(random_state=seed),
            "random_under": RandomUnderSampler(random_state=seed),
        }[sample]
        return make_imb_pipeline(pre, sampler, model)
    return Pipeline([("pre", build_preprocessor(num_cols, cat_cols, impute, scale)), ("model", model)])


# ---------------------------------------------------------------------------
# Feature engineering (CSV only)
# ---------------------------------------------------------------------------

def engineer_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    df = raw.copy()
    y = (df["outcome_clean"] == POSITIVE_LABEL).astype(int).to_numpy()

    # --- identifiers / grouping -------------------------------------------
    df["identity"] = df["project_name"].fillna("") + "|" + df["date_of_birth"].fillna("")
    df["app_year"] = pd.to_numeric(df["year"], errors="coerce")

    # --- demographic ---------------------------------------------------------
    dob_year = pd.to_datetime(df["date_of_birth"], errors="coerce").dt.year
    df["age"] = df["app_year"] - dob_year
    df["age_missing_flag"] = df["age"].isnull().astype(int)
    df["team_member_count_num"] = (
        df["team_member_count"].replace("5+", "6")
    )
    df["team_member_count_num"] = pd.to_numeric(df["team_member_count_num"], errors="coerce")
    df["team_missing_flag"] = df["team_member_count_num"].isnull().astype(int)
    df["is_team"] = (df["individual_or_team"] == "Team").astype(int)
    df["nationality_bin"] = np.where(df["nationality"] == "Bahrain", "Bahrain", "Other")
    df["gender_f"] = df["gender"].fillna("Unknown")
    df["employment_f"] = df["employment_status"].fillna("Unknown")
    df["education_f"] = df["education"].fillna("Unknown")
    df["how_heard_f"] = df["how_heard"].fillna("Unknown")
    df["prior_program_f"] = df["prior_program_experience"].fillna("Unknown")
    df["prev_participant_f"] = df["previous_mashroo3i_participant"].fillna("Unknown")
    df["cohort_f"] = df["cohort"].fillna("Unknown")

    df["cr"] = (df["has_commercial_registration"] == "Yes").astype(int)
    df["cr_and_team"] = df["cr"] * df["is_team"]
    df["in_two_cohorts"] = df["in_two_cohorts"].astype(int)

    prev_year = pd.to_numeric(df["previous_mashroo3i_year"], errors="coerce")
    df["prev_year_recency"] = (df["app_year"] - prev_year).clip(lower=0)
    df["prev_year_missing_flag"] = prev_year.isnull().astype(int)

    # --- stage/sector/market ------------------------------------------------
    df["stage_f"] = df["Business Stage"].fillna("Not Specified")
    df["sector_f"] = df["Sector"].fillna("Not Specified")
    df["sector_missing_flag"] = df["sector"].isnull().astype(int)
    df["stage_missing_flag"] = df["stage"].isnull().astype(int)
    df["age_group_f"] = df["Age Group"].fillna("Not Specified")
    df["stage_maturity"] = df["stage_f"].map(
        {"Idea": 0, "MVP": 1, "Revenue Generating": 2, "Operating": 3}
    ).fillna(-1).astype(int)
    df["cr_established"] = ((df["cr"] == 1) & (df["stage_maturity"] >= 1)).astype(int)

    # multi-label sector_all / subsector_all / how_heard_all
    sector_parts = df["sector_all"].map(split_list)
    subsector_parts = df["subsector_all"].map(split_list)
    how_heard_parts = df["how_heard_all"].map(split_list)
    target_parts = df["target_customers_all"].map(split_list)
    prior_program_parts = df["prior_program_experience_all"].map(split_list)

    df["sector_count"] = sector_parts.apply(len)
    df["subsector_count"] = subsector_parts.apply(len)
    df["how_heard_count"] = how_heard_parts.apply(len)
    df["target_count"] = target_parts.apply(len)
    df["prior_program_count"] = prior_program_parts.apply(len)

    df["sector_all_tech"] = sector_parts.apply(lambda parts: any("Technology & IT" in p for p in parts)).astype(int)
    df["sector_all_common"] = sector_parts.apply(
        lambda parts: any(p in COMMON_SECTORS for p in parts)
    ).astype(int)
    df["is_common_sector"] = (
        df["sector_f"].isin(COMMON_SECTORS) | (df["sector_all_common"] == 1)
    ).astype(int)
    df["sector_tech_named"] = (df["sector_f"] == "Technology & IT").astype(int)

    for sector in sorted(COMMON_SECTORS):
        df[f"sector_flag_{re.sub(r'[^a-z0-9]+', '_', sector.lower()).strip('_')}"] = (
            sector_parts.apply(lambda parts, s=sector: s in parts)
        ).astype(int)

    top_segments = [
        "Businesses & Companies", "Government & Public Sector", "General Public",
        "Women", "Parents & Families", "Youth & Teens", "Students",
        "Travelers & Expats", "Health & Medical", "Professionals & Employees",
        "Children & Kids", "Athletes & Fitness", "Car Owners & Drivers",
        "Property Owners & Real Estate", "Food & Beverage Customers",
        "Education Institutions", "Tech & Online Users", "Investors & Funders",
    ]
    df["target_all_text"] = df["target_customers_all"].fillna("") + ";" + df["target_customers_normalized"].fillna("")
    for seg in top_segments:
        df[f"target_flag_{re.sub(r'[^a-z0-9]+', '_', seg.lower()).strip('_')}"] = (
            df["target_all_text"].str.contains(re.escape(seg), regex=True, na=False).astype(int)
        )
    df["has_target_customers"] = (df["target_customers"].notna() | (df["target_all_text"].str.len() > 0)).astype(int)

    top_channels = [
        "Instagram", "Email", "Friends & Family", "Tamkeen (Direct)",
        "University / Education", "Other Online / Social Media",
        "Word of Mouth / Referral", "Programs & Organizations", "LinkedIn",
        "Media & Advertising", "TikTok", "Government / Ministries",
    ]
    how_all = df["how_heard_all"].fillna("") + ";" + df["how_heard"].fillna("")
    for ch in top_channels:
        df[f"channel_flag_{re.sub(r'[^a-z0-9]+', '_', ch.lower()).strip('_')}"] = (
            how_all.str.contains(re.escape(ch), regex=True, na=False).astype(int)
        )

    # --- submission timing ---------------------------------------------------
    submitted = pd.to_datetime(df["submitted_at"], errors="coerce")
    first_sub = df.groupby("cohort_id")["submitted_at"].transform("min")
    first_sub = pd.to_datetime(first_sub, errors="coerce")
    df["submission_dayofyear"] = submitted.dt.dayofyear
    df["submission_weekday"] = submitted.dt.weekday
    df["submission_hour"] = submitted.dt.hour
    df["submission_missing_flag"] = submitted.isnull().astype(int)
    df["days_since_round_start"] = (submitted - first_sub).dt.days

    # --- business age (regex only, no LLM) -----------------------------------
    age_text = (
        df["project_name"].fillna("").astype(str) + " | " +
        df["keywords"].fillna("").astype(str) + " | " +
        df["problem_en"].fillna("").astype(str) +
        df["solution_en"].fillna("").astype(str) +
        df["impact_en"].fillna("").astype(str)
    )
    df["business_age_regex"] = [
        extract_business_age(t, yr) for t, yr in zip(age_text, df["app_year"])
    ]
    df["business_age_missing_flag"] = df["business_age_regex"].isnull().astype(int)

    # --- text features ---------------------------------------------------------
    for col in TEXT_COLS_EN:
        df[f"{col}_has"] = df[col].notna().astype(int)
        df[f"{col}_words"] = df[col].map(word_count)
        df[f"{col}_unique"] = df[col].map(unique_ratio)
    for col in TEXT_COLS_RAW:
        df[f"{col}_raw_words"] = df[col].map(word_count)

    df["text_word_total"] = df[[c + "_words" for c in TEXT_COLS_EN]].sum(axis=1)
    df["text_raw_total"] = df[[c + "_raw_words" for c in TEXT_COLS_RAW]].sum(axis=1)
    df["n_text_fields_present"] = df[[c + "_has" for c in TEXT_COLS_EN]].sum(axis=1)
    df["problem_solution_jaccard"] = [
        jaccard(a, b) for a, b in zip(df["problem_en"], df["solution_en"])
    ]
    df["impact_inspiration_jaccard"] = [
        jaccard(a, b) for a, b in zip(df["impact_en"], df["inspiration_en"])
    ]
    df["solution_has_numbers"] = df["solution_en"].fillna("").str.contains(r"\d", regex=True).astype(int)
    df["solution_has_currency"] = df["solution_en"].fillna("").str.contains(
        r"\b(bhd|dinar|dinars|bd|usd|\$)\b", regex=True, flags=re.I
    ).astype(int)
    df["problem_has_question"] = df["problem_en"].fillna("").str.contains(r"\?", regex=True).astype(int)

    # --- keywords --------------------------------------------------------------
    kw_lists = df["keywords"].map(lambda s: split_list(s) if isinstance(s, str) else split_list(""))
    kw_series = kw_lists.explode().dropna()
    kw_freq = kw_series[kw_series.str.lower() != "uncategorized"].str.lower().value_counts()
    top_keywords = kw_freq.head(30).index.tolist()

    df["keyword_count"] = kw_lists.apply(len)
    df["keywords_uncategorized"] = df["keywords"].fillna("").str.lower().str.contains("uncategorized", regex=False).astype(int)
    df["keywords_lower"] = df["keywords"].fillna("").str.lower()
    for kw in top_keywords:
        col = "kw_" + re.sub(r"[^a-z0-9]+", "_", kw).strip("_")
        df[col] = df["keywords_lower"].str.contains(re.escape(kw), regex=True).astype(int)
    df["kw_mean_freq"] = kw_lists.apply(
        lambda keys: np.mean([kw_freq.get(k.lower(), 0) for k in keys]) if keys else 0
    )
    df["kw_min_freq"] = kw_lists.apply(
        lambda keys: min([kw_freq.get(k.lower(), 0) for k in keys]) if keys else 0
    )
    df["kw_novel_share"] = kw_lists.apply(
        lambda keys: np.mean([kw_freq.get(k.lower(), 0) <= 3 for k in keys]) if keys else 0
    )
    df = df.drop(columns=["keywords_lower"])

    # --- theme / tech flags ------------------------------------------------------
    all_text = (
        df["keywords"].fillna("").astype(str) + " | " +
        df["problem_en"].fillna("").astype(str) + " | " +
        df["solution_en"].fillna("").astype(str) + " | " +
        df["project_name"].fillna("").astype(str) + " | " +
        df["impact_en"].fillna("").astype(str)
    )
    theme_flags = []
    for theme, terms in THEME_KEYWORDS.items():
        col = "theme_" + theme
        df[col] = all_text.apply(lambda t: has_any(t, terms)).astype(int)
        theme_flags.append(col)
    df["theme_count"] = df[theme_flags].sum(axis=1)

    df["tech_coupled"] = (
        (df["sector_all_tech"] == 1) |
        df["keywords"].fillna("").apply(lambda t: has_any(t, TECH_TERMS)) |
        (df["problem_en"].fillna("") + df["solution_en"].fillna("")).apply(lambda t: has_any(t, TECH_TERMS)) |
        df["project_name"].fillna("").apply(lambda t: has_any(t, TECH_TERMS))
    ).astype(int)
    df["common_sector_without_tech"] = (df["is_common_sector"] & (1 - df["tech_coupled"])).astype(int)
    df["impact_tech_proxy"] = df["impact_en_words"] * df["tech_coupled"]
    df["tech_and_team"] = df["tech_coupled"] * df["is_team"]
    df["tech_and_stage"] = df["tech_coupled"] * (df["stage_maturity"] + 1)

    df["text_local"] = all_text.apply(lambda t: has_any(t, ["bahrain", "local", "community", "kingdom", "citizens", "national"])).astype(int)
    df["text_jobs"] = all_text.apply(lambda t: has_any(t, ["job", "employ", "hire", "workforce", "career", "recruit", "opportunities"])).astype(int)
    df["text_finance"] = all_text.apply(lambda t: has_any(t, ["funding", "investment", "revenue", "profit", "capital", "loan", "sales", "income"])).astype(int)
    df["text_scale"] = all_text.apply(lambda t: has_any(t, ["global", "expand", "scale", "export", "region", "gulf", "middle east", "international"])).astype(int)

    # --- criteria rubric proxies (sheet.xlsx; deterministic, CSV-only) --------
    df["crit_problem"] = (
        (df["problem_en_words"] >= 20).astype(int)
        + (df["problem_en_words"] >= 60).astype(int)
        + (df["problem_has_question"] == 1).astype(int)
        + (df["problem_en_unique"] >= 0.5).astype(int)
    )
    df["crit_solution"] = (
        (df["solution_en_words"] >= 20).astype(int)
        + (df["solution_en_words"] >= 60).astype(int)
        + (df["solution_has_numbers"] == 1).astype(int)
        + (df["problem_solution_jaccard"] >= 0.35).astype(int)
    )
    df["crit_innovation"] = (
        (df["differentiation_en_has"] == 1).astype(int)
        + (df["differentiation_en_words"] >= 15).astype(int)
        + (df["kw_novel_share"] >= 0.5).astype(int)
        + (df["keyword_count"] >= 4).astype(int)
    )
    df["crit_market"] = (
        (df["has_target_customers"] == 1).astype(int)
        + (df["target_count"] >= 1).astype(int)
        + (df["is_common_sector"] == 1).astype(int)
        + (df["impact_en_words"] >= 15).astype(int)
    )
    df["crit_feasibility"] = (
        (df["cr"] == 1).astype(int)
        + (df["stage_maturity"] >= 1).astype(int)
        + (df["team_member_count_num"] >= 2).fillna(0).astype(int)
        + (df["business_age_regex"].notna().astype(int))
    )
    df["crit_total"] = (
        df["crit_problem"] + df["crit_solution"] + df["crit_innovation"]
        + df["crit_market"] + df["crit_feasibility"]
    )
    df["crit_tech_market"] = df["crit_market"] * df["tech_coupled"]
    df["crit_solution_feasibility"] = df["crit_solution"] * df["crit_feasibility"]

    # --- name quality ------------------------------------------------------------
    df["name_len"] = df["project_name"].fillna("").str.len()
    df["name_has_digit"] = df["project_name"].fillna("").str.contains(r"\d", regex=True).astype(int)
    df["name_has_arabic"] = df["project_name"].fillna("").str.contains(r"[\u0600-\u06FF]", regex=True).astype(int)

    # --- categories / numerics -----------------------------------------------
    CAT_COLS = [
        "year", "cohort_id", "gender_f", "employment_f", "education_f",
        "how_heard_f", "prior_program_f", "prev_participant_f", "cohort_f",
        "nationality_bin", "stage_f", "sector_f", "subsector", "age_group_f",
    ]

    numeric_text_cols = []
    for col in TEXT_COLS_EN:
        numeric_text_cols += [f"{col}_has", f"{col}_words", f"{col}_unique"]
    for col in TEXT_COLS_RAW:
        numeric_text_cols += [f"{col}_raw_words"]

    df["sector_tech_named"] = df["sector_tech_named"].astype(int)

    kw_flag_cols = [c for c in df.columns if c.startswith("kw_") and c not in ("kw_mean_freq", "kw_min_freq", "kw_novel_share")]
    theme_flag_cols = ["theme_" + t for t in THEME_KEYWORDS]
    tech_flag_cols = ["tech_coupled", "common_sector_without_tech", "impact_tech_proxy", "tech_and_team", "tech_and_stage"]
    signal_flag_cols = ["text_local", "text_jobs", "text_finance", "text_scale"]
    criteria_proxy_cols = [
        "crit_problem", "crit_solution", "crit_innovation", "crit_market",
        "crit_feasibility", "crit_total", "crit_tech_market",
        "crit_solution_feasibility",
    ]
    sector_all_flag_cols = [c for c in df.columns if c.startswith("sector_flag_")]
    target_flag_cols = [c for c in df.columns if c.startswith("target_flag_")]
    channel_flag_cols = [c for c in df.columns if c.startswith("channel_flag_")]

    num_feature_groups = {
        "base": [
            "age", "team_member_count_num", "app_year", "stage_maturity", "cr",
            "cr_and_team", "in_two_cohorts", "prev_year_recency", "name_len",
            "age_missing_flag", "team_missing_flag", "sector_missing_flag",
            "stage_missing_flag", "prev_year_missing_flag",
        ],
        "text": numeric_text_cols,
        "keywords": kw_flag_cols + [
            "keyword_count", "kw_mean_freq", "kw_min_freq", "kw_novel_share",
            "keywords_uncategorized",
        ] + theme_flag_cols + ["theme_count"],
        "tech": tech_flag_cols + signal_flag_cols + sector_all_flag_cols,
        "criteria": [
            "business_age_regex", "business_age_missing_flag",
            "sector_count", "subsector_count", "how_heard_count",
            "target_count", "prior_program_count", "has_target_customers",
            "problem_solution_jaccard", "impact_inspiration_jaccard",
            "solution_has_numbers", "solution_has_currency", "problem_has_question",
            "text_word_total", "text_raw_total", "n_text_fields_present",
        ] + target_flag_cols + channel_flag_cols,
        "temporal": [
            "submission_dayofyear", "submission_weekday", "submission_hour",
            "days_since_round_start", "submission_missing_flag",
        ],
        "criteria_proxy": criteria_proxy_cols,
        "quality": [
            "name_has_digit", "name_has_arabic",
        ],
    }

    all_num_ordered = []
    for key in ["base", "text", "keywords", "tech", "criteria", "criteria_proxy", "temporal", "quality"]:
        for c in num_feature_groups[key]:
            if c not in all_num_ordered:
                all_num_ordered.append(c)

    feature_sets = {
        "base": {"num": num_feature_groups["base"], "cat": CAT_COLS},
        "base_text": {
            "num": num_feature_groups["base"] + num_feature_groups["text"],
            "cat": CAT_COLS,
        },
        "base_keywords": {
            "num": num_feature_groups["base"] + num_feature_groups["text"] + num_feature_groups["keywords"],
            "cat": CAT_COLS,
        },
        "base_criteria": {
            "num": num_feature_groups["base"] + num_feature_groups["text"] + num_feature_groups["criteria"] + num_feature_groups["criteria_proxy"],
            "cat": CAT_COLS,
        },
        "full": {
            "num": all_num_ordered,
            "cat": CAT_COLS,
        },
    }

    df_out = df.copy()
    df_out["y"] = y
    num_groups = num_feature_groups
    return df_out, y, feature_sets, CAT_COLS, all_num_ordered, num_groups


# ---------------------------------------------------------------------------
# Main experiment driver
# ---------------------------------------------------------------------------

def main():
    import joblib
    from sklearn.model_selection import train_test_split

    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"loading {DATA_PATH}")
    raw = pd.read_csv(DATA_PATH)
    log(f"rows={len(raw)} target_yes={(raw['outcome_clean'] == POSITIVE_LABEL).sum()}")

    df, y, feature_sets, cat_cols, all_num, num_feature_groups = engineer_features(raw)
    y = pd.Series(y, index=df.index)
    log(f"engineered {len(all_num)} numeric / {len(cat_cols)} categorical features")

    # --- group-aware split -------------------------------------------------
    group_y = df.groupby("identity")["y"].max()
    train_groups, test_groups = train_test_split(
        group_y.index, test_size=0.2, stratify=group_y, random_state=SEED
    )
    train_mask = df["identity"].isin(train_groups)
    test_mask = df["identity"].isin(test_groups)
    X = df[all_num + cat_cols]
    X_train, X_test = X.loc[train_mask], X.loc[test_mask]
    y_train, y_test = y.loc[train_mask], y.loc[test_mask]
    groups_train = pd.factorize(df.loc[train_mask, "identity"])[0]

    log(f"train={len(X_train)} test={len(X_test)} train_pos={y_train.mean():.3f} "
        f"test_pos={y_test.mean():.3f} overlap={len(set(train_groups) & set(test_groups))}")

    pw = pos_weight(y_train)
    log(f"pos_weight={pw:.2f}")

    results = []
    n_splits = 3 if args.quick else 5

    def checkpoint():
        pd.DataFrame(results).sort_values(
            ["oof_f2_at_best", "oof_recall_at_best"], ascending=False
        ).to_csv(OUT_DIR / "experiments.csv", index=False)

    if args.quick:
        models = ["xgboost", "lightgbm", "catboost"]
        imputes = ["median", "none"]
        scales = ["standard"]
        samples = [None]
        feature_keys = ["base_keywords", "base_criteria", "full"]
        n_random = 3
    else:
        models = [
            "catboost", "lightgbm", "xgboost", "hist_gradient_boosting",
            "random_forest", "extra_trees", "logistic_regression",
        ]
        imputes = ["median", "mean", "knn", "iterative", "none"]
        scales = ["standard", "robust", "minmax"]
        samples = [None, "smote", "smoteenn", "random_under"]
        feature_keys = ["base", "base_text", "base_keywords", "base_criteria", "full"]
        n_random = 15

    # Phase 1: model family baselines on full feature set
    log("Phase 1: baseline model families (full features, median+standard, weighted)")
    for name in models:
        est = build_estimator(
            name, {}, impute="median", scale="standard",
            num_cols=feature_sets["full"]["num"], cat_cols=cat_cols,
            weighted=True, n_jobs=-1, pw=pw,
        )
        metrics, oof = run_cv(est, X_train, y_train, groups_train, n_splits=n_splits)
        best_thr, thr_df = threshold_rows(oof, y_train)
        row = {
            **summarize_cv(metrics),
            "model": name, "impute": "median", "scale": "standard",
            "sample": "class_weight", "feature_set": "full",
            "best_threshold": round(float(best_thr["threshold"]), 3),
            "oof_f2_at_best": round(float(best_thr["f2"]), 4),
            "oof_recall_at_best": round(float(best_thr["recall"]), 4),
            "oof_fn_rate_at_best": round(float(best_thr["fn_rate"]), 4),
        }
        results.append(row)
        checkpoint()
        log(f"  {name}: cv_f2={row['cv_f2']} oof_f2={row['oof_f2_at_best']} "
            f"recall={row['oof_recall_at_best']} fn_rate={row['oof_fn_rate_at_best']}")

    # Phase 2: feature set ablations on the top booster
    top_booster = "catboost"
    log(f"Phase 2: feature-set ablations with {top_booster}")
    for fkey in feature_keys:
        num_cols = feature_sets[fkey]["num"]
        est = build_estimator(
            top_booster, {}, impute="median", scale="standard",
            num_cols=num_cols, cat_cols=cat_cols, weighted=True, n_jobs=-1, pw=pw,
        )
        metrics, oof = run_cv(est, X_train, y_train, groups_train, n_splits=n_splits)
        best_thr, _ = threshold_rows(oof, y_train)
        row = {
            **summarize_cv(metrics),
            "model": top_booster, "impute": "median", "scale": "standard",
            "sample": "class_weight", "feature_set": fkey,
            "best_threshold": round(float(best_thr["threshold"]), 3),
            "oof_f2_at_best": round(float(best_thr["f2"]), 4),
            "oof_recall_at_best": round(float(best_thr["recall"]), 4),
            "oof_fn_rate_at_best": round(float(best_thr["fn_rate"]), 4),
        }
        results.append(row)
        checkpoint()
        log(f"  {fkey}: cv_f2={row['cv_f2']} oof_f2={row['oof_f2_at_best']} "
            f"recall={row['oof_recall_at_best']} fn_rate={row['oof_fn_rate_at_best']}")

    # Phase 3: imputation x scaling on best feature set
    best_feature_key = "full"
    log(f"Phase 3: imputation x scaling ({best_feature_key}) with {top_booster}")
    candidate_combos = []
    for imp in imputes:
        for sc in scales:
            if imp == "none" and sc != "standard":
                continue
            candidate_combos.append((imp, sc))
    for imp, sc in candidate_combos:
        est = build_estimator(
            top_booster, {}, impute=imp, scale=sc,
            num_cols=feature_sets[best_feature_key]["num"],
            cat_cols=cat_cols, weighted=True, n_jobs=-1, pw=pw,
        )
        metrics, oof = run_cv(est, X_train, y_train, groups_train, n_splits=n_splits)
        best_thr, _ = threshold_rows(oof, y_train)
        row = {
            **summarize_cv(metrics),
            "model": top_booster, "impute": imp, "scale": sc,
            "sample": "class_weight", "feature_set": best_feature_key,
            "best_threshold": round(float(best_thr["threshold"]), 3),
            "oof_f2_at_best": round(float(best_thr["f2"]), 4),
            "oof_recall_at_best": round(float(best_thr["recall"]), 4),
            "oof_fn_rate_at_best": round(float(best_thr["fn_rate"]), 4),
        }
        results.append(row)
        checkpoint()
        log(f"  {imp}/{sc}: cv_f2={row['cv_f2']} oof_f2={row['oof_f2_at_best']} "
            f"recall={row['oof_recall_at_best']} fn_rate={row['oof_fn_rate_at_best']}")

    # choose best imputation/scaling from phase 3 rows
    phase3_rows = [
        r for r in results
        if r["model"] == top_booster and r["feature_set"] == best_feature_key
        and r["sample"] == "class_weight"
    ]
    best_prep_row = max(phase3_rows, key=lambda r: (r["oof_f2_at_best"], r["oof_recall_at_best"]))
    chosen_impute, chosen_scale = best_prep_row["impute"], best_prep_row["scale"]
    log(f"chosen impute={chosen_impute} scale={chosen_scale}")

    # Phase 4: tuning boosters on best feature set
    log("Phase 4: randomized tuning (catboost/lightgbm/xgboost)")
    import numpy as _np
    rng = _np.random.default_rng(SEED)

    def random_params(model_name):
        if model_name == "catboost":
            return {
                "iterations": int(rng.choice([300, 500, 800])),
                "learning_rate": float(rng.choice([0.02, 0.05, 0.1])),
                "depth": int(rng.choice([4, 6, 8])),
                "l2_leaf_reg": float(rng.choice([1, 3, 5, 10])),
                "bagging_temperature": float(rng.choice([0, 1, 2])),
                "random_strength": float(rng.choice([0, 1])),
            }
        if model_name == "lightgbm":
            return {
                "n_estimators": int(rng.choice([300, 500, 800])),
                "learning_rate": float(rng.choice([0.02, 0.05, 0.1])),
                "num_leaves": int(rng.choice([15, 31, 63])),
                "min_child_samples": int(rng.choice([10, 20, 40])),
                "subsample": float(rng.choice([0.7, 0.85, 1.0])),
                "colsample_bytree": float(rng.choice([0.7, 0.85, 1.0])),
                "reg_alpha": float(rng.choice([0, 0.1, 1])),
                "reg_lambda": float(rng.choice([0, 0.1, 1, 5])),
            }
        return {
            "n_estimators": int(rng.choice([300, 500, 800])),
            "learning_rate": float(rng.choice([0.02, 0.05, 0.1])),
            "max_depth": int(rng.choice([3, 5, 7])),
            "min_child_weight": int(rng.choice([1, 3, 7])),
            "subsample": float(rng.choice([0.7, 0.85, 1.0])),
            "colsample_bytree": float(rng.choice([0.7, 0.85, 1.0])),
            "reg_alpha": float(rng.choice([0, 0.1, 1])),
            "reg_lambda": float(rng.choice([0, 0.1, 1, 5])),
        }

    tuned = {}
    for model_name in ["catboost", "lightgbm", "xgboost"]:
        best = None
        for i in range(n_random):
            params = random_params(model_name)
            est = build_estimator(
                model_name, params, impute=chosen_impute, scale=chosen_scale,
                num_cols=feature_sets[best_feature_key]["num"],
                cat_cols=cat_cols, weighted=True, n_jobs=-1, pw=pw,
            )
            metrics, oof = run_cv(est, X_train, y_train, groups_train, n_splits=n_splits)
            best_thr, _ = threshold_rows(oof, y_train)
            row = {
                **summarize_cv(metrics),
                "model": model_name, "impute": chosen_impute, "scale": chosen_scale,
                "sample": "class_weight", "feature_set": best_feature_key,
                "tuned": True, "n_random": i + 1,
                "params": json.dumps(params),
                "best_threshold": round(float(best_thr["threshold"]), 3),
                "oof_f2_at_best": round(float(best_thr["f2"]), 4),
                "oof_recall_at_best": round(float(best_thr["recall"]), 4),
                "oof_fn_rate_at_best": round(float(best_thr["fn_rate"]), 4),
            }
            results.append(row)
            checkpoint()
            log(f"  {model_name} #{i+1}: f2={row['oof_f2_at_best']} rec={row['oof_recall_at_best']}")
            if best is None or row["oof_f2_at_best"] > best["oof_f2_at_best"]:
                best = row
        tuned[model_name] = (params if best is None else json.loads(best["params"]))

    # Phase 5: sampling variants on best tuned model
    log("Phase 5: resampling variants")
    best_tuned_row = max(
        [r for r in results if r.get("tuned")],
        key=lambda r: (r["oof_f2_at_best"], r["oof_recall_at_best"]),
        default=None,
    )
    sample_model = best_tuned_row["model"] if best_tuned_row else "catboost"
    sample_params = json.loads(best_tuned_row["params"]) if best_tuned_row else {}
    for sample in samples:
        if sample is None:
            continue
        est = build_estimator(
            sample_model, sample_params, impute=chosen_impute, scale=chosen_scale,
            num_cols=feature_sets[best_feature_key]["num"],
            cat_cols=cat_cols, weighted=False, sample=sample, n_jobs=-1, pw=pw,
        )
        metrics, oof = run_cv(est, X_train, y_train, groups_train, n_splits=n_splits)
        best_thr, _ = threshold_rows(oof, y_train)
        row = {
            **summarize_cv(metrics),
            "model": sample_model, "impute": chosen_impute, "scale": chosen_scale,
            "sample": sample, "feature_set": best_feature_key,
            "best_threshold": round(float(best_thr["threshold"]), 3),
            "oof_f2_at_best": round(float(best_thr["f2"]), 4),
            "oof_recall_at_best": round(float(best_thr["recall"]), 4),
            "oof_fn_rate_at_best": round(float(best_thr["fn_rate"]), 4),
        }
        results.append(row)
        checkpoint()
        log(f"  {sample}: f2={row['oof_f2_at_best']} rec={row['oof_recall_at_best']}")

    # Phase 6: bagged booster (multi-seed average) on the best config
    log("Phase 6: multi-seed bagged booster")
    top_single = max(
        [r for r in results if r.get("oof_f2_at_best")],
        key=lambda r: (r["oof_f2_at_best"], r["oof_recall_at_best"]),
    )
    bagged_fits = []
    bag_oofs = []
    bag_num = feature_sets[top_single["feature_set"]]["num"]
    bag_params = json.loads(top_single["params"]) if top_single.get("params") else {}
    bag_seeds = [42, 7, 2024]
    for seed in bag_seeds:
        est = build_estimator(
            top_single["model"], bag_params, impute=top_single["impute"],
            scale=top_single["scale"], num_cols=bag_num, cat_cols=cat_cols,
            weighted=(top_single["sample"] == "class_weight"),
            sample=(None if top_single["sample"] == "class_weight" else top_single["sample"]),
            n_jobs=-1, pw=pw, seed=seed,
        )
        metrics, oof = run_cv(est, X_train, y_train, groups_train, n_splits=n_splits)
        bagged_fits.append(est)
        bag_oofs.append(oof)
    bag_oof = np.mean(bag_oofs, axis=0)
    bag_best_thr, _ = threshold_rows(bag_oof, y_train)
    row = {
        **summarize_cv(metrics),
        "model": "bagged_" + top_single["model"],
        "impute": top_single["impute"], "scale": top_single["scale"],
        "sample": top_single["sample"], "feature_set": top_single["feature_set"],
        "bagged": True, "n_seeds": len(bag_seeds),
        "best_threshold": round(float(bag_best_thr["threshold"]), 3),
        "oof_f2_at_best": round(float(bag_best_thr["f2"]), 4),
        "oof_recall_at_best": round(float(bag_best_thr["recall"]), 4),
        "oof_fn_rate_at_best": round(float(bag_best_thr["fn_rate"]), 4),
    }
    results.append(row)
    checkpoint()
    log(f"  bagged: f2={row['oof_f2_at_best']} rec={row['oof_recall_at_best']}")

    # Phase 7: final selection
    candidates = [r for r in results if r.get("oof_f2_at_best")]
    final = max(candidates, key=lambda r: (r["oof_f2_at_best"], r["oof_recall_at_best"]))
    log(f"final config: {final['model']} sample={final['sample']} f2={final['oof_f2_at_best']}")

    results_df = pd.DataFrame(results).sort_values(["oof_f2_at_best", "oof_recall_at_best"], ascending=False)
    results_df.to_csv(OUT_DIR / "experiments.csv", index=False)

    # Retrain final on train, threshold from OOF, evaluate test
    final_num = feature_sets[final["feature_set"]]["num"]
    final_params = json.loads(final["params"]) if final.get("params") else {}
    if final.get("bagged"):
        final_ests = []
        for seed in bag_seeds:
            est = build_estimator(
                final["model"].replace("bagged_", ""), final_params,
                impute=final["impute"], scale=final["scale"],
                num_cols=final_num, cat_cols=cat_cols,
                weighted=(final["sample"] == "class_weight"),
                sample=(None if final["sample"] == "class_weight" else final["sample"]),
                n_jobs=-1, pw=pw, seed=seed,
            )
            est.fit(X_train, y_train)
            final_ests.append(est)
        final_est = final_ests
        test_proba = np.mean([e.predict_proba(X_test)[:, 1] for e in final_ests], axis=0)
    else:
        final_est = build_estimator(
            final["model"], final_params, impute=final["impute"], scale=final["scale"],
            num_cols=final_num, cat_cols=cat_cols,
            weighted=(final["sample"] == "class_weight"),
            sample=(None if final["sample"] == "class_weight" else final["sample"]),
            n_jobs=-1, pw=pw,
        )
        final_est.fit(X_train, y_train)
        test_proba = final_est.predict_proba(X_test)[:, 1]
    test_threshold = float(final["best_threshold"])
    test_pred = test_proba >= test_threshold
    test_metrics = compute_metrics(y_test.to_numpy(), test_pred, test_proba)

    log(f"test: f2={test_metrics['f2']:.4f} recall={test_metrics['recall']:.4f} "
        f"fn_rate={test_metrics['fn_rate']:.4f} pr_auc={test_metrics['pr_auc']:.4f}")

    safe_json(OUT_DIR / "feature_columns.json", {"num": final_num, "cat": cat_cols})
    safe_json(OUT_DIR / "final_threshold.json", {"threshold": test_threshold})
    safe_json(OUT_DIR / "test_metrics.json", test_metrics)
    joblib.dump(final_est, OUT_DIR / "final_model.joblib")
    pd.DataFrame({"y_true": y_test.to_numpy(), "y_prob": test_proba, "y_pred": test_pred.astype(int)}).to_csv(
        OUT_DIR / "test_predictions.csv", index=False
    )
    log(f"saved artifacts to {OUT_DIR}")


if __name__ == "__main__":
    main()
