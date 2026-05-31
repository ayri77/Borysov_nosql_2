# scripts/05_chunking.py
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

FIXED_INDEX_NAME = "arxiv-chunks-fixed"
SEMANTIC_INDEX_NAME = "arxiv-chunks-semantic"

TOP_LONGEST_ARTICLES = 30
CHUNK_SIZE = 100
OVERLAP = 20
MAX_SEMANTIC_CHUNK_WORDS = 100
BATCH_SIZE = 200
TOP_K = 5
METRIC = "cosine"

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")

# Helper functions:
def select_longest_articles(df: pd.DataFrame, n: int = TOP_LONGEST_ARTICLES) -> pd.DataFrame:
    """Select papers with the longest abstracts."""
    articles = df.copy()
    articles["abstract_word_count"] = articles["abstract"].str.split().str.len()

    return (
        articles
        .sort_values("abstract_word_count", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )

def fixed_size_chunk(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into fixed-size overlapping word chunks."""
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    words = text.split()
    step = chunk_size - overlap

    return [
        " ".join(words[start:start + chunk_size])
        for start in range(0, len(words), step)
        if words[start:start + chunk_size]
    ]

def sentence_based_chunk(text: str, max_words: int) -> list[str]:
    """Build chunks from complete sentences up to a target word limit."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    chunks = []
    current_sentences = []
    current_word_count = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_word_count = len(sentence.split())

        if current_sentences and current_word_count + sentence_word_count > max_words:
            chunks.append(" ".join(current_sentences))
            current_sentences = []
            current_word_count = 0

        current_sentences.append(sentence)
        current_word_count += sentence_word_count

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks

def normalize_whitespace(text: str) -> str:
    """Replace repeated whitespace and line breaks with single spaces."""
    return " ".join(text.split())

def build_chunk_records(articles: pd.DataFrame, strategy: str) -> list[dict]:
    """Create chunk records with text and metadata for a chunking strategy."""
    if strategy not in {"fixed", "semantic"}:
        raise ValueError(f"Unsupported chunking strategy: {strategy}")

    records = []

    for _, row in articles.iterrows():
        if strategy == "fixed":
            chunks = fixed_size_chunk(
                row["abstract"],
                chunk_size=CHUNK_SIZE,
                overlap=OVERLAP,
            )
        else:
            chunks = sentence_based_chunk(
                row["abstract"],
                max_words=MAX_SEMANTIC_CHUNK_WORDS,
            )

        for chunk_number, chunk_text in enumerate(chunks):
            chunk_text = normalize_whitespace(chunk_text)
            title = normalize_whitespace(row["title"])

            records.append(
                {
                    "id": f"{strategy}_{row['id']}_{chunk_number}",
                    "text": f"{title} [SEP] {chunk_text}",
                    "metadata": {
                        "arxiv_id": row["id"],
                        "title": title,
                        "chunk_text": chunk_text,
                        "chunk_number": chunk_number,
                        "year": int(row["year"]),
                        "category": row["category"],
                        "strategy": strategy,
                    },
                }
            )

    return records

def create_index_if_needed(index_name: str) -> None:
    """Create a Pinecone index for chunk embeddings if it does not exist."""
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=VECTOR_DIM,
            metric=METRIC,
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

def upload_chunks(index_name: str, records: list[dict]) -> None:
    """Encode and upload chunk records to Pinecone in batches."""
    index = pc.Index(index_name)

    texts = [record["text"] for record in records]

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    for start in tqdm(
        range(0, len(records), BATCH_SIZE),
        desc=f"Uploading to {index_name}",
    ):
        end = min(start + BATCH_SIZE, len(records))

        vectors = [
            {
                "id": records[i]["id"],
                "values": embeddings[i].tolist(),
                "metadata": records[i]["metadata"],
            }
            for i in range(start, end)
        ]

        index.upsert(vectors=vectors)

    stats = index.describe_index_stats()
    print(f"Total vectors in '{index_name}': {stats['total_vector_count']}")

def search_chunks(index_name: str, query: str, top_k: int = TOP_K):
    """Search for relevant chunks in a selected Pinecone index."""
    index = pc.Index(index_name)
    query_vector = model.encode(query, normalize_embeddings=True).tolist()

    return index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
    )


def print_chunk_results(title: str, query: str, results) -> None:
    """Print ranked chunk search results with metadata."""
    print(f"\n{title}")
    print(f"Query: '{query}'\n")

    if not results.matches:
        print("No results found.\n")
        return

    for rank, match in enumerate(results.matches, start=1):
        metadata = match.metadata

        print(f"{rank}. ID: {match.id} | Score: {match.score:.4f}")
        print(f"   Title: {metadata['title']}")
        print(f"   Category: {metadata['category']}")
        print(f"   Chunk number: {metadata['chunk_number']}")
        print(f"   Strategy: {metadata['strategy']}")
        print(f"   Chunk: {metadata['chunk_text'][:300]}...")
        print()

def print_chunking_summary(articles: pd.DataFrame) -> None:
    """Print aggregate statistics and a boundary example for both chunking strategies."""
    fixed_sizes = []
    semantic_sizes = []
    fixed_total = 0
    semantic_total = 0

    for _, row in articles.iterrows():
        fixed_chunks = [
            normalize_whitespace(chunk)
            for chunk in fixed_size_chunk(row["abstract"], CHUNK_SIZE, OVERLAP)
        ]
        semantic_chunks = [
            normalize_whitespace(chunk)
            for chunk in sentence_based_chunk(
                row["abstract"],
                MAX_SEMANTIC_CHUNK_WORDS,
            )
        ]

        fixed_total += len(fixed_chunks)
        semantic_total += len(semantic_chunks)

        fixed_sizes.extend(len(chunk.split()) for chunk in fixed_chunks)
        semantic_sizes.extend(len(chunk.split()) for chunk in semantic_chunks)

    print("\nChunking summary for 30 longest abstracts")
    print(f"Fixed-size chunks: {fixed_total}")
    print(f"Sentence-based chunks: {semantic_total}")
    print(
        f"Fixed-size chunk words: "
        f"min={min(fixed_sizes)}, "
        f"max={max(fixed_sizes)}, "
        f"mean={np.mean(fixed_sizes):.1f}"
    )
    print(
        f"Sentence-based chunk words: "
        f"min={min(semantic_sizes)}, "
        f"max={max(semantic_sizes)}, "
        f"mean={np.mean(semantic_sizes):.1f}"
    )

    row = articles.iloc[0]
    fixed_chunks = fixed_size_chunk(row["abstract"], CHUNK_SIZE, OVERLAP)
    semantic_chunks = sentence_based_chunk(
        row["abstract"],
        MAX_SEMANTIC_CHUNK_WORDS,
    )

    print("\nFixed-size boundary example")
    print(f"End of chunk 1: ...{fixed_chunks[0][-150:]}")
    print(f"Start of chunk 2: {fixed_chunks[1][:150]}...")

    print("\nSentence-based boundary example")
    print(f"End of chunk 1: ...{semantic_chunks[0][-150:]}")
    print(f"Start of chunk 2: {semantic_chunks[1][:150]}...")

def main() -> None:
    """Run chunking, indexing, and chunk-search comparison."""
    longest_articles = select_longest_articles(df)
    print_chunking_summary(longest_articles)

    fixed_records = build_chunk_records(longest_articles, "fixed")
    semantic_records = build_chunk_records(longest_articles, "semantic")

    print("\nChunk records summary")
    print(f"Fixed records: {len(fixed_records)}")
    print(f"Semantic records: {len(semantic_records)}")

    create_index_if_needed(FIXED_INDEX_NAME)
    create_index_if_needed(SEMANTIC_INDEX_NAME)

    upload_chunks(FIXED_INDEX_NAME, fixed_records)
    upload_chunks(SEMANTIC_INDEX_NAME, semantic_records)

    chunk_query = "observations of supernova explosions and gamma-ray bursts"

    fixed_results = search_chunks(FIXED_INDEX_NAME, chunk_query)
    semantic_results = search_chunks(SEMANTIC_INDEX_NAME, chunk_query)

    print_chunk_results("Fixed-size chunking search", chunk_query, fixed_results)
    print_chunk_results("Sentence-based chunking search", chunk_query, semantic_results)


if __name__ == "__main__":
    main()