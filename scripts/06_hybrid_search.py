# scripts/06_hybrid_search.py
import os
import re

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Constants and initialization
load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"

RETRIEVAL_TOP_K = 10
DISPLAY_TOP_K = 5
DEFAULT_RRF_K = 60

TEST_QUERIES = [
    "BERT fine-tuning",
    "Yann LeCun convolutional networks",
    "making computers understand human emotions from text",
]

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)

df = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)


# Helper functions
def normalize_whitespace(text: str) -> str:
    """Replace repeated whitespace and line breaks with single spaces."""
    return " ".join(text.split())


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 lexical retrieval."""
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


def prepare_bm25_index(data: pd.DataFrame) -> BM25Okapi:
    """Build a BM25 index from article titles and abstracts."""
    corpus = (
        data["title"].map(normalize_whitespace)
        + " "
        + data["abstract"].map(normalize_whitespace)
    ).tolist()

    tokenized_corpus = [tokenize(text) for text in corpus]

    return BM25Okapi(tokenized_corpus)


def bm25_search(
    query: str,
    bm25: BM25Okapi,
    top_k: int = RETRIEVAL_TOP_K,
) -> list[dict]:
    """Run BM25 lexical search and return ranked document results."""
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)
    ranked_indices = np.argsort(scores)[::-1]

    results = []

    for idx in ranked_indices:
        score = float(scores[idx])

        if score <= 0:
            continue

        row = df.iloc[idx]

        results.append(
            {
                "arxiv_id": row["id"],
                "title": normalize_whitespace(row["title"]),
                "category": row["category"],
                "year": int(row["year"]),
                "abstract": normalize_whitespace(row["abstract"]),
                "score": score,
            }
        )

        if len(results) >= top_k:
            break

    return results


def encode_query(query: str) -> np.ndarray:
    """Encode and normalize a query for vector retrieval."""
    return model.encode(query, normalize_embeddings=True)


def vector_search(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
) -> list[dict]:
    """Run vector search in Pinecone and return results in a common format."""
    query_vector = encode_query(query)

    matches = index.query(
        vector=query_vector.tolist(),
        top_k=top_k,
        include_metadata=True,
    ).matches

    return [
        {
            "arxiv_id": match.metadata["arxiv_id"],
            "title": normalize_whitespace(match.metadata["title"]),
            "category": match.metadata["category"],
            "year": int(match.metadata["year"]),
            "abstract": normalize_whitespace(match.metadata["abstract"]),
            "score": float(match.score),
        }
        for match in matches
    ]

def reciprocal_rank_fusion(
    bm25_results: list[dict],
    vector_results: list[dict],
    rrf_k: int = DEFAULT_RRF_K,
    top_k: int = DISPLAY_TOP_K,
) -> list[dict]:
    """Fuse BM25 and vector ranked lists using Reciprocal Rank Fusion."""
    fused = {}

    for rank, result in enumerate(bm25_results, start=1):
        arxiv_id = result["arxiv_id"]

        if arxiv_id not in fused:
            fused[arxiv_id] = {
                **result,
                "score": 0.0,
                "bm25_rank": None,
                "vector_rank": None,
            }

        fused[arxiv_id]["score"] += 1 / (rrf_k + rank)
        fused[arxiv_id]["bm25_rank"] = rank

    for rank, result in enumerate(vector_results, start=1):
        arxiv_id = result["arxiv_id"]

        if arxiv_id not in fused:
            fused[arxiv_id] = {
                **result,
                "score": 0.0,
                "bm25_rank": None,
                "vector_rank": None,
            }

        fused[arxiv_id]["score"] += 1 / (rrf_k + rank)
        fused[arxiv_id]["vector_rank"] = rank

    return sorted(
        fused.values(),
        key=lambda result: result["score"],
        reverse=True,
    )[:top_k]


def print_results(title: str, query: str, results: list[dict], score_name: str) -> None:
    """Print ranked search results."""
    print(f"\n{title}")
    print(f"Query: '{query}'\n")

    if not results:
        print("No results found.\n")
        return

    for rank, result in enumerate(results[:DISPLAY_TOP_K], start=1):
        print(
            f"{rank}. ID: paper_{result['arxiv_id']} | "
            f"{score_name}: {result['score']:.4f}"
        )
        print(f"   Title: {result['title']}")
        print(f"   Category: {result['category']}")
        print(f"   Year: {result['year']}")
        print(f"   Abstract: {result['abstract'][:200]}...")
        print()

def print_hybrid_results(title: str, query: str, results: list[dict]) -> None:
    """Print RRF-fused results with source rankings."""
    print(f"\n{title}")
    print(f"Query: '{query}'\n")

    if not results:
        print("No results found.\n")
        return

    for rank, result in enumerate(results[:DISPLAY_TOP_K], start=1):
        bm25_rank = result["bm25_rank"] if result["bm25_rank"] is not None else "-"
        vector_rank = (
            result["vector_rank"] if result["vector_rank"] is not None else "-"
        )

        print(
            f"{rank}. ID: paper_{result['arxiv_id']} | "
            f"RRF score: {result['score']:.6f} | "
            f"BM25 rank: {bm25_rank} | "
            f"Vector rank: {vector_rank}"
        )
        print(f"   Title: {result['title']}")
        print(f"   Category: {result['category']}")
        print(f"   Year: {result['year']}")
        print(f"   Abstract: {result['abstract'][:200]}...")
        print()

def run_query_experiment(
    query: str,
    bm25: BM25Okapi,
    rrf_k: int = DEFAULT_RRF_K,
) -> None:
    """Run BM25, vector, and hybrid search for one query."""
    bm25_results = bm25_search(query, bm25)
    vector_results = vector_search(query)

    hybrid_results = reciprocal_rank_fusion(
        bm25_results,
        vector_results,
        rrf_k=rrf_k,
    )

    print("\n" + "=" * 100)
    print(f"QUERY EXPERIMENT: '{query}'")
    print("=" * 100)

    print_results("BM25 search", query, bm25_results, "BM25 score")
    print_results("Vector search", query, vector_results, "Vector score")
    print_hybrid_results(
        f"Hybrid search with RRF (k={rrf_k})",
        query,
        hybrid_results,
    )

def compare_rrf_k(query: str, bm25: BM25Okapi) -> None:
    """Compare hybrid ranking for different RRF smoothing parameters."""
    bm25_results = bm25_search(query, bm25)
    vector_results = vector_search(query)

    print("\n" + "=" * 100)
    print(f"RRF K COMPARISON: '{query}'")
    print("=" * 100)

    for rrf_k in [1, 10, 60, 100]:
        hybrid_results = reciprocal_rank_fusion(
            bm25_results,
            vector_results,
            rrf_k=rrf_k,
        )
        print_hybrid_results(
            f"Hybrid search with RRF (k={rrf_k})",
            query,
            hybrid_results,
        )

def main() -> None:
    """Run hybrid search experiments."""
    bm25 = prepare_bm25_index(df)

    for query in TEST_QUERIES:
        run_query_experiment(query, bm25)

    compare_rrf_k(TEST_QUERIES[2], bm25)

if __name__ == "__main__":
    main()