"""Similar-idea helpers for the Streamlit app.

Loads the prebuilt Voyage index (``idea_embeddings.npy``,
``ideas_metadata.csv``, ``ideas_index.json``) copied into this repository and
exposes two capabilities:

* ``top_similar_pairs`` - flag near-duplicate ideas already in the index
  (works fully offline, no API key required).
* ``search_similar`` - embed a new idea with the Voyage API and rank the
  closest past submissions.

Raw cosine scores are translated into friendly labels so the UI never shows
numbers to non-technical users.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import streamlit as st

import idea_search


LEVEL_THRESHOLDS: tuple[tuple[str, float], ...] = (
    ("Nearly identical", 0.97),
    ("Very similar", 0.92),
    ("Similar", 0.87),
    ("Somewhat similar", 0.80),
)

STRICTNESS_THRESHOLDS: dict[str, float] = {
    "Very similar": 0.92,
    "Similar": 0.87,
    "Somewhat similar": 0.80,
}

QUERY_LEVEL_THRESHOLDS: tuple[tuple[str, float], ...] = (
    ("Very similar", 0.65),
    ("Similar", 0.58),
    ("Somewhat similar", 0.50),
)

DEFAULT_STRICTNESS = "Very similar"
DEFAULT_PAIR_LIMIT = 25
DEFAULT_TOP_K = 3
DEFAULT_MIN_SCORE = 0.50
DEFAULT_CLUSTER_THRESHOLD = 0.70
DEFAULT_GROUP_SIZE = 12
SNIPPET_LIMIT = 160
DUPLICATE_MERGE_SIMILARITY = 0.97
MIN_NAME_TOKEN_CHARS = 3
DOCUMENT_LABELS = (
    "Project",
    "Problem",
    "Solution",
    "Differentiation",
    "Impact",
    "Inspiration",
    "Keywords",
)
_NON_WORD_RE = re.compile(r"[^\w]+")


def similarity_level(score: float) -> str:
    """Translate a cosine score into a plain-language label."""
    for label, threshold in LEVEL_THRESHOLDS:
        if score >= threshold:
            return label
    return "Not similar"


def relation_band(score: float) -> str:
    """Label a direct similarity between two stored ideas."""
    if score >= 0.92:
        return "Very similar (likely same idea)"
    return "Similar (different idea, same concept)"


def query_similarity_level(score: float) -> str:
    """Label a query-vs-document score (lower scale than doc-vs-doc)."""
    for label, threshold in QUERY_LEVEL_THRESHOLDS:
        if score >= threshold:
            return label
    return "Not similar"


def _voyage_api_key() -> str | None:
    """Read the Voyage key from Streamlit secrets, env, or a local .env."""
    try:
        value = st.secrets.get("VOYAGE_AI_API_KEY") or st.secrets.get(
            "VOYAGE_API_KEY"
        )
        if value and str(value).strip():
            return str(value).strip()
    except Exception:
        pass
    for name in ("VOYAGE_AI_API_KEY", "VOYAGE_API_KEY"):
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    idea_search.load_dotenv()
    for name in ("VOYAGE_AI_API_KEY", "VOYAGE_API_KEY"):
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return None


@st.cache_data(show_spinner=False)
def load_index() -> tuple[np.ndarray, pd.DataFrame, dict] | None:
    """Return the cached (matrix, metadata, index) triple, or None."""
    return idea_search.load_cache()


def _snippet(text: object, limit: int = SNIPPET_LIMIT) -> str:
    clean = idea_search.clean_text(text)
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "..."


def document_sections(text: object) -> dict[str, str]:
    """Split a stored idea document back into its labeled sections."""
    sections: dict[str, str] = {}
    label_pattern = "|".join(re.escape(label) for label in DOCUMENT_LABELS)
    pattern = re.compile(rf"^({label_pattern}):\s*(.*)$")
    for raw_line in str(text or "").splitlines():
        line = idea_search.clean_text(raw_line)
        match = pattern.match(line)
        if match:
            sections[match.group(1)] = match.group(2).strip()
    return sections


def normalize_name(value: object) -> str:
    """Lowercase a project name and reduce punctuation to single spaces."""
    return _NON_WORD_RE.sub(" ", idea_search.clean_text(value).lower()).strip()


def normalize_text(value: object) -> str:
    """Collapse a document to one lowercased line for exact-duplicate checks."""
    return re.sub(
        r"\s+", " ", idea_search.clean_text(value)
    ).strip().lower()


def deduplicate_matrix(
    matrix: np.ndarray,
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Collapse duplicate submissions into one representative per idea.

    Three passes keep the most complete entry and drop the rest:
    1. identical documents (duplicate CSV rows),
    2. the same normalized project name (case/translation variants),
    3. near-identical documents (reworded repeats of one submission).
    """
    if len(metadata) == 0:
        return matrix, metadata
    meta = metadata.copy().reset_index(drop=True)
    meta["_norm_name"] = meta["project_name"].map(normalize_name)
    meta["_norm_text"] = meta["text"].map(normalize_text)
    meta["_text_len"] = meta["text"].map(
        lambda value: len(idea_search.clean_text(value))
    )
    meta["_order"] = np.arange(len(meta), dtype=int)

    # 1. Identical documents.
    meta = meta.drop_duplicates("_norm_text", keep="first")

    # 2. Same project name: keep the most complete submission, preferring the
    #    latest round; unnamed rows are never collapsed together.
    named = meta[meta["_norm_name"] != ""]
    unnamed = meta[meta["_norm_name"] == ""]
    named = named.sort_values(
        ["_norm_name", "_text_len", "year", "cohort_id"],
        ascending=[True, False, False, False],
        kind="mergesort",
    ).drop_duplicates("_norm_name", keep="first")
    meta = pd.concat([named, unnamed], ignore_index=True)
    meta = meta.sort_values("_order", kind="mergesort").reset_index(drop=True)
    kept = meta["_order"].astype(int).tolist()
    dedup_matrix = np.asarray(matrix)[kept]

    # 3. Near-identical documents (reworded repeats): greedily keep the
    #    longer document of every cluster at or above the merge similarity.
    stored = idea_search.normalized_matrix(dedup_matrix)
    similarity = stored @ stored.T
    row_indices, col_indices = np.triu_indices(len(stored), k=1)
    scores = similarity[row_indices, col_indices]
    dropped: set[int] = set()
    for position in np.argsort(scores)[::-1]:
        score = float(scores[position])
        if score < DUPLICATE_MERGE_SIMILARITY:
            break
        left, right = int(row_indices[position]), int(col_indices[position])
        if left in dropped or right in dropped:
            continue
        keep, kill = (
            (left, right)
            if meta.iloc[left]["_text_len"] >= meta.iloc[right]["_text_len"]
            else (right, left)
        )
        dropped.add(kill)

    survivors = [index for index in range(len(stored)) if index not in dropped]
    final_matrix = dedup_matrix[survivors]
    final_metadata = meta.iloc[survivors].reset_index(drop=True)
    return final_matrix, final_metadata


def deduplicated_index() -> tuple[np.ndarray, pd.DataFrame, dict] | None:
    """Return the cached index with duplicate submissions collapsed."""
    cached = load_index()
    if cached is None:
        return None
    matrix, metadata, index = cached
    dedup_matrix, dedup_metadata = deduplicate_matrix(matrix, metadata)
    return dedup_matrix, dedup_metadata, index


def _pair_row(metadata: pd.DataFrame, left: int, right: int, score: float) -> dict:
    left_row = metadata.iloc[left]
    right_row = metadata.iloc[right]
    left_sections = document_sections(left_row["text"])
    right_sections = document_sections(right_row["text"])
    return {
        "similarity": score,
        "level": similarity_level(score),
        "left_project": left_row["project_name"],
        "left_year": left_row["year"],
        "left_cohort_id": left_row["cohort_id"],
        "left_sector": left_row["sector"],
        "left_snippet": _snippet(left_row["text"]),
        "left_problem": left_sections.get("Problem", ""),
        "left_solution": left_sections.get("Solution", ""),
        "right_project": right_row["project_name"],
        "right_year": right_row["year"],
        "right_cohort_id": right_row["cohort_id"],
        "right_sector": right_row["sector"],
        "right_snippet": _snippet(right_row["text"]),
        "right_problem": right_sections.get("Problem", ""),
        "right_solution": right_sections.get("Solution", ""),
    }


def names_share_identity(left_name: object, right_name: object) -> bool:
    """True when every word of the shorter name appears in the longer name."""
    left_tokens = normalize_name(left_name).split()
    right_tokens = normalize_name(right_name).split()
    if not left_tokens or not right_tokens:
        return False
    shorter, longer = sorted((left_tokens, right_tokens), key=len)
    if not any(len(token) >= MIN_NAME_TOKEN_CHARS for token in shorter):
        return False
    return set(shorter) <= set(longer)


def _top_similar_pairs(
    matrix: np.ndarray,
    metadata: pd.DataFrame,
    threshold: float,
    limit: int = DEFAULT_PAIR_LIMIT,
) -> pd.DataFrame:
    """Return the highest-similarity pairs at or above ``threshold``."""
    stored = idea_search.normalized_matrix(matrix)
    similarity = stored @ stored.T
    count = len(stored)
    row_indices, col_indices = np.triu_indices(count, k=1)
    scores = similarity[row_indices, col_indices]

    rows: list[dict] = []
    for position in np.argsort(scores)[::-1]:
        score = float(scores[position])
        if score < threshold:
            break
        left, right = int(row_indices[position]), int(col_indices[position])
        rows.append(
            _pair_row(
                metadata,
                left,
                right,
                score,
            )
        )
        if len(rows) >= limit:
            break
    columns = [
        "similarity",
        "level",
        "left_project",
        "left_year",
        "left_cohort_id",
        "left_sector",
        "left_snippet",
        "left_problem",
        "left_solution",
        "right_project",
        "right_year",
        "right_cohort_id",
        "right_sector",
        "right_snippet",
        "right_problem",
        "right_solution",
    ]
    return pd.DataFrame(rows, columns=columns)


def top_similar_pairs(
    level: str = DEFAULT_STRICTNESS,
    limit: int = DEFAULT_PAIR_LIMIT,
) -> pd.DataFrame:
    """Flag the closest pairs already in the index, including identical ones."""
    cached = load_index()
    if cached is None:
        return pd.DataFrame()
    matrix, metadata, _ = cached
    threshold = STRICTNESS_THRESHOLDS.get(level, STRICTNESS_THRESHOLDS[DEFAULT_STRICTNESS])
    return _top_similar_pairs(matrix, metadata, threshold, limit)


def similar_clusters(
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
    max_group_size: int = DEFAULT_GROUP_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group ideas by a shared anchor so each idea appears only once.

    The deduplicated index collapses identical, same-name, and near-identical
    repeats before clustering. Ideas are connected when their cosine similarity
    is at least ``threshold``. Groups are built greedily: the idea with the
    most direct matches becomes the anchor, takes its closest unassigned
    matches (capped at ``max_group_size`` members total), and all assigned
    ideas are removed from later groups. Every undirected relationship inside a
    group appears exactly once in the edge frame, so the UI never shows the
    same pairing twice.
    """
    cluster_columns = [
        "cluster_id",
        "row_index",
        "project_name",
        "year",
        "cohort_id",
        "sector",
        "snippet",
        "problem",
        "solution",
    ]
    edge_columns = [
        "cluster_id",
        "left_row_index",
        "right_row_index",
        "similarity",
        "band",
    ]
    empty = (
        pd.DataFrame(columns=cluster_columns),
        pd.DataFrame(columns=edge_columns),
    )
    cached = deduplicated_index()
    if cached is None:
        return empty
    matrix, metadata, _ = cached
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or len(matrix) == 0:
        return empty
    stored = idea_search.normalized_matrix(matrix)
    scores = np.asarray(stored @ stored.T, dtype=np.float32)
    count = len(metadata)
    row_indices, col_indices = np.triu_indices(count, k=1)
    edge_scores = scores[row_indices, col_indices]
    kept = edge_scores >= threshold
    edge_pairs = [
        (int(left), int(right), float(score))
        for left, right, score in zip(
            row_indices[kept],
            col_indices[kept],
            edge_scores[kept],
        )
    ]

    degree: dict[int, int] = {}
    best: dict[int, float] = {}
    for left, right, score in edge_pairs:
        degree[left] = degree.get(left, 0) + 1
        degree[right] = degree.get(right, 0) + 1
        best[left] = max(best.get(left, 0.0), score)
        best[right] = max(best.get(right, 0.0), score)

    if not edge_pairs:
        return empty

    order = sorted(
        range(count),
        key=lambda index: (
            -degree.get(index, 0),
            -best.get(index, 0.0),
        ),
    )
    clusters_rows: list[dict] = []
    edges_rows: list[dict] = []
    assigned: set[int] = set()
    cluster_id = 0
    for anchor in order:
        if anchor in assigned:
            continue
        neighbor_scores: list[tuple[int, float]] = []
        for left, right, score in edge_pairs:
            if left == anchor and right not in assigned:
                neighbor_scores.append((right, score))
            elif right == anchor and left not in assigned:
                neighbor_scores.append((left, score))
        neighbor_scores.sort(key=lambda item: -item[1])
        members = [anchor]
        members.extend(index for index, _score in neighbor_scores[: max_group_size - 1])
        if len(members) < 2:
            continue
        assigned.update(members)

        for index in members:
            row = metadata.iloc[index]
            sections = document_sections(row["text"])
            clusters_rows.append(
                {
                    "cluster_id": cluster_id,
                    "row_index": index,
                    "project_name": row["project_name"],
                    "year": row["year"],
                    "cohort_id": row["cohort_id"],
                    "sector": row["sector"],
                    "snippet": _snippet(row["text"]),
                    "problem": sections.get("Problem", ""),
                    "solution": sections.get("Solution", ""),
                }
            )
        member_set = set(members)
        for left, right, score in edge_pairs:
            if left in member_set and right in member_set:
                edges_rows.append(
                    {
                        "cluster_id": cluster_id,
                        "left_row_index": left,
                        "right_row_index": right,
                        "similarity": score,
                        "band": relation_band(score),
                    }
                )
        cluster_id += 1

    clusters_df = pd.DataFrame(clusters_rows, columns=cluster_columns)
    edges_df = pd.DataFrame(edges_rows, columns=edge_columns)
    if not edges_df.empty:
        edges_df = edges_df.sort_values(
            ["cluster_id", "similarity"],
            ascending=[True, False],
            kind="mergesort",
        ).reset_index(drop=True)
    return clusters_df, edges_df


def _rank_matches(
    matrix: np.ndarray,
    metadata: pd.DataFrame,
    query_vector: np.ndarray,
    top_k: int,
    min_score: float,
) -> list[dict]:
    """Rank stored documents against one normalized query vector."""
    stored = idea_search.normalized_matrix(matrix)
    query_norm = query_vector / max(float(np.linalg.norm(query_vector)), 1e-12)
    similarities = np.asarray(stored @ query_norm.T, dtype=np.float32).reshape(-1)

    matches: list[dict] = []
    for position in np.argsort(similarities)[::-1][:top_k]:
        score = float(similarities[position])
        if score < min_score:
            break
        row = metadata.iloc[position]
        sections = document_sections(row["text"])
        matches.append(
            {
                "similarity": score,
                "level": query_similarity_level(score),
                "project_name": row["project_name"],
                "year": row["year"],
                "cohort_id": row["cohort_id"],
                "sector": row["sector"],
                "snippet": _snippet(row["text"]),
                "problem": sections.get("Problem", ""),
                "solution": sections.get("Solution", ""),
            }
        )
    return matches


@st.cache_data(show_spinner=False, ttl=3600)
def search_similar(
    query_text: str,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[dict]:
    """Find past ideas similar to ``query_text`` using Voyage embeddings."""
    text = (query_text or "").strip()
    if not text:
        return []
    cached = deduplicated_index()
    if cached is None:
        return []
    matrix, metadata, index = cached
    api_key = _voyage_api_key()
    if not api_key:
        return []

    query_vectors = idea_search.embed_many(
        [text],
        api_key,
        index["model"],
        int(index["dimensions"]),
        # Embedding the query as a document keeps it in the same space as the
        # stored index, giving comparable scores (the `query` input type is
        # systematically lower against this document index).
        "document",
        idea_search.DEFAULT_BATCH_SIZE,
        idea_search.DEFAULT_TOKEN_BUDGET,
    )
    if not query_vectors:
        return []
    query_vector = np.asarray(query_vectors[0], dtype=np.float32).reshape(1, -1)
    return _rank_matches(matrix, metadata, query_vector, top_k, min_score)


def index_available() -> bool:
    """Return whether the copied index assets can be loaded."""
    return load_index() is not None
