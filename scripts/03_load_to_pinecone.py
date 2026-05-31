# scripts/03_load_to_pinecone.py
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

load_dotenv()

INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200
METRIC = "cosine"

# Initialise client
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Create index if it does not exist
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric=METRIC,
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        ),
    )

index = pc.Index(INDEX_NAME)

# Load prepared data and embeddings
df = pd.read_parquet(INPUT_PARQUET)
embeddings = np.load(INPUT_EMBEDDINGS)

assert len(df) == len(embeddings), "Dataset and embeddings lengths do not match."
assert embeddings.shape[1] == VECTOR_DIM, "Unexpected embedding dimension."

# Upload vectors in batches
for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Uploading batches"):
    end = min(start + BATCH_SIZE, len(df))
    vectors = []

    for i in range(start, end):
        row = df.iloc[i]

        vectors.append(
            {
                "id": f"paper_{row['id']}",
                "values": embeddings[i].tolist(),
                "metadata": {
                    "arxiv_id": row["id"],
                    "title": row["title"],
                    "abstract": row["abstract"][:500],
                    "authors": row["authors"][:200],
                    "year": int(row["year"]),
                    "category": row["category"],
                },
            }
        )

    index.upsert(vectors=vectors)

stats = index.describe_index_stats()
print(f"Total vectors in index: {stats['total_vector_count']}")