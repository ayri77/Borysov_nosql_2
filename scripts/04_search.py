# scripts/04_search.py
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Constants and initialization
load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")  # Metadata and full abstracts for local search

# Helper functions:
def encode_query(query: str) -> np.ndarray:
    """Encode and normalize a text query using the embedding model."""
    return model.encode(query, normalize_embeddings=True)


def pinecone_search(
    query: str,
    metadata_filter: dict | None = None,
    top_k: int = TOP_K,
):
    """Run semantic search in Pinecone with an optional metadata filter."""
    query_vector = encode_query(query)

    query_args = {
        "vector": query_vector.tolist(),
        "top_k": top_k,
        "include_metadata": True,
    }

    if metadata_filter is not None:
        query_args["filter"] = metadata_filter

    return index.query(**query_args)


def print_pinecone_results(title: str, query: str, results) -> None:
    """Print ranked Pinecone search results with metadata."""
    print(f"\n{title}")
    print(f"Query: '{query}'\n")

    if not results.matches:
        print("No results found.\n")
        return

    for rank, match in enumerate(results.matches, start=1):
        print(f"{rank}. ID: {match.id} | Score: {match.score:.4f}")
        print(f"   Title: {match.metadata['title']}")
        print(f"   Category: {match.metadata['category']}")
        print(f"   Year: {match.metadata['year']}")
        print(f"   Abstract: {match.metadata['abstract'][:200]}...")
        print()

def local_metric_search(
    query: str,
    metric: str,
    embeddings: np.ndarray,
    top_k: int = TOP_K,
) -> tuple[np.ndarray, np.ndarray]:
    """Run exact local search using the selected similarity metric."""
    query_vector = encode_query(query)

    if metric == "cosine":
        scores = (embeddings @ query_vector) / (
                np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vector)
        )
        top_indices = np.argsort(scores)[::-1][:top_k]

    elif metric == "dot":
        scores = embeddings @ query_vector
        top_indices = np.argsort(scores)[::-1][:top_k]

    elif metric == "l2":
        scores = np.linalg.norm(embeddings - query_vector, axis=1)
        top_indices = np.argsort(scores)[:top_k]

    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return top_indices, scores[top_indices]

def print_local_results(
    title: str,
    query: str,
    indices: np.ndarray,
    scores: np.ndarray,
    metric: str,
) -> None:
    """Print ranked results produced by local exact search."""
    value_label = "Distance" if metric == "l2" else "Score"

    print(f"\n{title}")
    print(f"Query: '{query}'\n")

    for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
        row = df.iloc[idx]

        print(f"{rank}. ID: paper_{row['id']} | {value_label}: {score:.4f}")
        print(f"   Title: {row['title']}")
        print(f"   Category: {row['category']}")
        print(f"   Year: {row['year']}")
        print(f"   Abstract: {row['abstract'][:200]}...")
        print()

# 3. Pure semantic search
semantic_query = "teaching machines to recognize objects in pictures"
semantic_results = pinecone_search(semantic_query)
print_pinecone_results("Pure semantic search", semantic_query, semantic_results)

# 4. Semantic search with metadata filtering
filtered_query = "reinforcement learning"

unfiltered_results = pinecone_search(filtered_query)
print_pinecone_results(
    "Unfiltered search for comparison",
    filtered_query,
    unfiltered_results,
)

results_a = pinecone_search(
    filtered_query,
    metadata_filter={
        "$and": [
            {"year": {"$gte": 2021}},
            {"category": {"$eq": "cs.LG"}},
        ]
    },
)
print_pinecone_results("Filtered search A: recent cs.LG papers", filtered_query, results_a)

results_b = pinecone_search(
    filtered_query,
    metadata_filter={"year": {"$lt": 2015}},
)
print_pinecone_results("Filtered search B: papers before 2015", filtered_query, results_b)

# 5. Local metric comparison
embeddings = np.load("embeddings/embeddings.npy")

for metric in ["cosine", "dot", "l2"]:
    indices, scores = local_metric_search(semantic_query, metric, embeddings)
    print_local_results(
        f"Local search: {metric}",
        semantic_query,
        indices,
        scores,
        metric,
    )