#!/usr/bin/env python3
"""Find semantically similar ideas using the Voyage embeddings API.

Examples
--------
Build the idea index::

    venv/bin/python idea_search.py build

Search for past ideas similar to new text::

    venv/bin/python idea_search.py search \
        "A platform that helps students find affordable tutoring" --top-k 5

Find very similar pairs already in the index::

    venv/bin/python idea_search.py pairs --threshold 0.85 --limit 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "dashboard_ready.csv"
ENV_PATH = ROOT / ".env"
EMBEDDINGS_PATH = ROOT / "idea_embeddings.npy"
METADATA_PATH = ROOT / "ideas_metadata.csv"
INDEX_PATH = ROOT / "ideas_index.json"

API_URL = "https://api.voyageai.com/v1/embeddings"
DEFAULT_MODEL = "voyage-4-large"
DEFAULT_DIMENSIONS = 1024
DEFAULT_BATCH_SIZE = 64
DEFAULT_TOKEN_BUDGET = 90_000
API_RETRIES = 3

ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")

# (label, English translation column, original column)
TEXT_FIELDS = [
    ("Project", "project_name", None),
    ("Problem", "problem_en", "problem"),
    ("Solution", "solution_en", "solution"),
    ("Differentiation", "differentiation_en", "differentiation"),
    ("Impact", "impact_en", "impact"),
    ("Inspiration", "inspiration_en", "inspiration"),
    ("Keywords", "keywords", None),
]

METADATA_COLUMNS = [
    "idea_id",
    "row_index",
    "year",
    "cohort",
    "cohort_id",
    "project_name",
    "sector",
    "stage",
    "outcome_clean",
    "in_two_cohorts",
    "text",
]


def load_dotenv(path: Path = ENV_PATH) -> None:
    """Load simple KEY=VALUE entries from .env without printing secrets."""

    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)


def get_api_key() -> str:
    load_dotenv()
    key = (
        os.getenv("VOYAGE_AI_API_KEY")
        or os.getenv("VOYAGE_API_KEY")
        or ""
    ).strip()
    if not key:
        raise RuntimeError(
            "Voyage API key not found. Set VOYAGE_AI_API_KEY in .env "
            "or in the environment."
        )
    return key


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def has_arabic(text: str) -> bool:
    return bool(ARABIC_RE.search(text))


def make_document(row: pd.Series) -> str:
    """Build one semantic document from the idea-defining text fields."""

    parts: list[str] = []
    seen: set[str] = set()

    for label, english_col, original_col in TEXT_FIELDS:
        english = clean_text(row.get(english_col))
        original = clean_text(row.get(original_col)) if original_col else ""

        pieces: list[str] = []
        if english:
            pieces.append(english)
            if original and original != english and has_arabic(original):
                pieces.append(f"Original: {original}")
        elif original:
            pieces.append(original)

        if pieces:
            text = " ".join(pieces)
            if text not in seen:
                parts.append(f"{label}: {text}")
                seen.add(text)

    if parts:
        return "\n".join(parts)
    return "Unnamed idea"


def make_idea_id(row: pd.Series, row_index: int) -> str:
    """Return a stable, human-independent id for one application row."""

    key = "|".join(
        [
            clean_text(row.get("year")),
            clean_text(row.get("cohort_id")),
            clean_text(row.get("project_name")),
            clean_text(row.get("problem")),
            clean_text(row.get("solution")),
        ]
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def iter_batches(
    texts: Sequence[str],
    batch_size: int,
    token_budget: int,
) -> Iterable[list[str]]:
    """Yield batches bounded by count and an estimated token budget."""

    batch: list[str] = []
    estimated_tokens = 0
    for text in texts:
        estimate = max(1, len(text) // 4)
        if batch and (
            len(batch) >= batch_size
            or estimated_tokens + estimate > token_budget
        ):
            yield batch
            batch = []
            estimated_tokens = 0
        batch.append(text)
        estimated_tokens += estimate
    if batch:
        yield batch


def extract_embeddings(payload: dict) -> list[list[float]]:
    records = payload.get("data")
    if records is None:
        records = payload.get("embeddings")
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"Unexpected Voyage response: {payload}")
    if isinstance(records[0], dict):
        return [record["embedding"] for record in records]
    return records


def post_embeddings(
    api_key: str,
    texts: list[str],
    model: str,
    dimensions: int | None,
    input_type: str,
) -> list[list[float]]:
    """POST one batch to Voyage with bounded retries."""

    payload: dict = {
        "input": texts,
        "model": model,
        "input_type": input_type,
    }
    if dimensions is not None:
        payload["output_dimension"] = dimensions

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    with requests.Session() as session:
        last_error: Exception | None = None
        for attempt in range(API_RETRIES):
            try:
                response = session.post(
                    API_URL,
                    headers=headers,
                    json=payload,
                    timeout=90,
                )
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise requests.HTTPError(
                        f"Voyage returned HTTP {response.status_code}",
                        response=response,
                    )
                response.raise_for_status()
                return extract_embeddings(response.json())
            except (requests.RequestException, RuntimeError) as exc:
                last_error = exc
                if isinstance(exc, requests.HTTPError) and exc.response is not None:
                    detail = exc.response.text[:300]
                    last_error = RuntimeError(
                        f"Voyage request failed with HTTP {exc.response.status_code}: {detail}"
                    )
                if attempt < API_RETRIES - 1:
                    time.sleep(min(2**attempt * 2, 20))

        raise RuntimeError(f"Voyage request failed after retries: {last_error}")


def embed_many(
    texts: Sequence[str],
    api_key: str,
    model: str,
    dimensions: int | None,
    input_type: str,
    batch_size: int,
    token_budget: int,
    dry_run: bool = False,
) -> list[list[float]] | None:
    """Embed all texts, returning None in dry-run mode."""

    batches = list(iter_batches(texts, batch_size, token_budget))
    print(
        f"Embedding {len(texts)} texts in {len(batches)} batches "
        f"with {model} ({input_type})"
    )
    if dry_run:
        print("Dry run: no API call or file write performed.")
        return None

    vectors: list[list[float]] = []
    for index, texts_to_send in enumerate(batches, start=1):
        print(
            f"  Batch {index}/{len(batches)} "
            f"({len(texts_to_send)} texts, "
            f"~{sum(len(text) for text in texts_to_send) // 4:,} tokens)"
        )
        vectors.extend(
            post_embeddings(
                api_key,
                texts_to_send,
                model,
                dimensions,
                input_type,
            )
        )
    return vectors


def read_input_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def cache_is_current(metadata: pd.DataFrame, docs: Sequence[str]) -> bool:
    if not metadata.empty and "text_hash" in metadata.columns:
        current_hashes = [
            hashlib.sha256(doc.encode("utf-8")).hexdigest() for doc in docs
        ]
        stored_hashes = metadata["text_hash"].astype(str).tolist()
        return stored_hashes == current_hashes
    return False


def build_index(args: argparse.Namespace) -> None:
    path = Path(args.csv)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    df = read_input_csv(path)
    docs: list[str] = []
    rows: list[dict] = []
    seen_idea_ids: set[str] = set()

    for row_index, (_, row) in enumerate(df.iterrows()):
        text = make_document(row)
        docs.append(text)
        idea_id = make_idea_id(row, row_index)
        if idea_id in seen_idea_ids:
            idea_id = f"{idea_id}-{row_index:04d}"
        seen_idea_ids.add(idea_id)
        rows.append(
            {
                "idea_id": idea_id,
                "row_index": row_index,
                "year": clean_text(row.get("year")),
                "cohort": clean_text(row.get("cohort")),
                "cohort_id": clean_text(row.get("cohort_id")),
                "project_name": clean_text(row.get("project_name")),
                "sector": clean_text(row.get("sector")),
                "stage": clean_text(row.get("stage")),
                "outcome_clean": clean_text(row.get("outcome_clean")),
                "in_two_cohorts": clean_text(row.get("in_two_cohorts")),
                "text": text,
                "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )

    metadata = pd.DataFrame(rows)
    content_hash = hashlib.sha256(
        "\n".join(docs).encode("utf-8")
    ).hexdigest()

    if (
        not args.dry_run
        and not args.force
        and EMBEDDINGS_PATH.exists()
        and METADATA_PATH.exists()
        and INDEX_PATH.exists()
    ):
        try:
            stored_metadata = pd.read_csv(METADATA_PATH, dtype=str, keep_default_na=False)
            stored_index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
            if (
                stored_index.get("model") == args.model
                and stored_index.get("content_hash") == content_hash
                and cache_is_current(stored_metadata, docs)
            ):
                print(
                    "Idea index is up to date "
                    f"({len(stored_metadata)} ideas). Use --force to rebuild."
                )
                return
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    if not args.dry_run:
        api_key = get_api_key()
        vectors = embed_many(
            docs,
            api_key,
            args.model,
            args.dimensions,
            "document",
            args.batch_size,
            args.token_budget,
            dry_run=False,
        )
        if vectors is None:
            raise RuntimeError("No embeddings returned from Voyage.")
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.shape[0] != len(docs):
            raise RuntimeError(
                f"Expected {len(docs)} embeddings but got {matrix.shape[0]}."
            )
        if args.dimensions is not None and matrix.shape[1] != args.dimensions:
            raise RuntimeError(
                f"Expected {args.dimensions} dimensions but got {matrix.shape[1]}."
            )
        np.save(EMBEDDINGS_PATH, matrix)
        metadata.to_csv(METADATA_PATH, index=False, encoding="utf-8-sig")
        index = {
            "model": args.model,
            "dimensions": int(matrix.shape[1]),
            "input_type": "document",
            "content_hash": content_hash,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "rows": len(docs),
            "csv": str(path),
            "text_fields": [label for label, _, _ in TEXT_FIELDS],
        }
        INDEX_PATH.write_text(
            json.dumps(index, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            f"Saved {len(docs)} embeddings "
            f"({matrix.shape[1]} dims) to {EMBEDDINGS_PATH.name}, "
            f"{METADATA_PATH.name}, and {INDEX_PATH.name}."
        )
    else:
        embed_many(
            docs,
            "",
            args.model,
            args.dimensions,
            "document",
            args.batch_size,
            args.token_budget,
            dry_run=True,
        )
        print(f"Would save {len(docs)} rows to {METADATA_PATH.name}.")


def load_cache() -> tuple[np.ndarray, pd.DataFrame, dict] | None:
    if not (EMBEDDINGS_PATH.exists() and METADATA_PATH.exists() and INDEX_PATH.exists()):
        return None
    try:
        matrix = np.load(EMBEDDINGS_PATH)
        metadata = pd.read_csv(METADATA_PATH, dtype=str, keep_default_na=False)
        index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if matrix.shape[0] != len(metadata):
        return None
    return matrix.astype(np.float32), metadata, index


def normalized_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def format_rows(rows: pd.DataFrame) -> str:
    output: list[str] = []
    for _, row in rows.iterrows():
        text = clean_text(row.get("text"))
        if len(text) > 180:
            text = text[:177] + "..."
        output.append(
            f"  similarity={float(row['similarity']):.4f}  "
            f"id={row['idea_id']}  {row['year']}/{row['cohort_id']}  "
            f"project={row['project_name'] or '(unnamed)'}  "
            f"sector={row['sector'] or '-'}  "
            f"repeat_applicant={row['in_two_cohorts']}"
        )
        output.append(f"    {text}")
    return "\n".join(output)


def search(args: argparse.Namespace) -> None:
    cached = load_cache()
    if cached is None:
        raise RuntimeError(
            "No idea index found. Run `python idea_search.py build` first."
        )
    matrix, metadata, index = cached
    query = (args.query or args.query_text or "").strip()
    if not query:
        raise ValueError("Search requires a non-empty query.")

    api_key = get_api_key()
    query_vectors = embed_many(
        [query],
        api_key,
        index["model"],
        int(index["dimensions"]),
        "query",
        args.batch_size,
        args.token_budget,
    )
    if query_vectors is None:
        raise RuntimeError("No query embedding returned.")

    query_vector = np.asarray(query_vectors[0], dtype=np.float32).reshape(1, -1)
    stored = normalized_matrix(matrix)
    query_norm = query_vector / max(float(np.linalg.norm(query_vector)), 1e-12)
    similarities = np.asarray(
        stored @ query_norm.T,
        dtype=np.float32,
    ).reshape(-1)
    top_indices = np.argsort(similarities)[::-1][: args.top_k]
    result = metadata.iloc[top_indices].copy()
    result["similarity"] = similarities[top_indices]
    result = result.sort_values("similarity", ascending=False).reset_index(drop=True)

    print(f"Top {len(result)} similar ideas:")
    print(format_rows(result))


def find_pairs(args: argparse.Namespace) -> None:
    cached = load_cache()
    if cached is None:
        raise RuntimeError(
            "No idea index found. Run `python idea_search.py build` first."
        )
    matrix, metadata, _ = cached
    stored = normalized_matrix(matrix)
    similarity = stored @ stored.T
    row_indices, col_indices = np.triu_indices(len(stored), k=1)
    pair_scores = similarity[row_indices, col_indices]

    order = np.argsort(pair_scores)[::-1]
    rows: list[dict] = []
    for pos in order:
        left, right = int(row_indices[pos]), int(col_indices[pos])
        score = float(pair_scores[pos])
        if args.threshold is not None and score < args.threshold:
            break
        left_row = metadata.iloc[left]
        right_row = metadata.iloc[right]
        rows.append(
            {
                "similarity": score,
                "left_id": left_row["idea_id"],
                "left_project": left_row["project_name"],
                "left_year": left_row["year"],
                "left_cohort_id": left_row["cohort_id"],
                "left_repeat_applicant": left_row["in_two_cohorts"],
                "right_id": right_row["idea_id"],
                "right_project": right_row["project_name"],
                "right_year": right_row["year"],
                "right_cohort_id": right_row["cohort_id"],
                "right_repeat_applicant": right_row["in_two_cohorts"],
            }
        )
        if len(rows) >= args.limit:
            break

    if not rows:
        print("No pairs found at or above the selected threshold.")
        return

    print(f"Most similar {len(rows)} pairs:")
    for pair in rows:
        print(
            f"  {pair['similarity']:.4f}  "
            f"{pair['left_year']}/{pair['left_cohort_id']} "
            f"{pair['left_project'] or '(unnamed)'} "
            f"<=> {pair['right_year']}/{pair['right_cohort_id']} "
            f"{pair['right_project'] or '(unnamed)'}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Semantic idea search with Voyage embeddings."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_build = subparsers.add_parser("build", help="Embed all ideas and save the index.")
    p_build.add_argument("--csv", default=str(DEFAULT_CSV), help="Input application CSV.")
    p_build.add_argument("--model", default=DEFAULT_MODEL, help="Voyage model name.")
    p_build.add_argument(
        "--dimensions",
        type=int,
        default=DEFAULT_DIMENSIONS,
        help="Output embedding dimensions.",
    )
    p_build.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Maximum texts per API request.",
    )
    p_build.add_argument(
        "--token-budget",
        type=int,
        default=DEFAULT_TOKEN_BUDGET,
        help="Approximate token budget per API request.",
    )
    p_build.add_argument(
        "--force", action="store_true", help="Rebuild even if the cache is current."
    )
    p_build.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview batching without calling the API or writing files.",
    )
    p_build.set_defaults(func=build_index)

    p_search = subparsers.add_parser("search", help="Search the index with new idea text.")
    p_search.add_argument("query", nargs="?", help="New idea text.")
    p_search.add_argument("--query", dest="query_text", help="New idea text.")
    p_search.add_argument("--top-k", type=int, default=10, help="Number of results.")
    p_search.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Maximum texts per API request.",
    )
    p_search.add_argument(
        "--token-budget",
        type=int,
        default=DEFAULT_TOKEN_BUDGET,
        help="Approximate token budget per API request.",
    )
    p_search.set_defaults(func=search)

    p_pairs = subparsers.add_parser(
        "pairs", help="Find the most similar pairs inside the index."
    )
    p_pairs.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Minimum cosine similarity; use 0 to show the top pairs.",
    )
    p_pairs.add_argument("--limit", type=int, default=100, help="Maximum pairs to print.")
    p_pairs.set_defaults(func=find_pairs)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
