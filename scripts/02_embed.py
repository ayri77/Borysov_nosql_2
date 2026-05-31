# scripts/02_embed.py
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
import numpy as np

# Constants
INPUT_FILE  = "data/arxiv_subset.parquet"
OUTPUT_FILE = "embeddings/embeddings.npy"
MODEL_NAME = "allenai/specter2_base"
BATCH_SIZE = 64

# create output directory if needed
os.makedirs("embeddings", exist_ok=True)

# read parquet dataset
df = pd.read_parquet(INPUT_FILE)
# build texts:
# title + " [SEP] " + abstract
texts = (
    df["title"].str.strip()
    + " [SEP] "
    + df["abstract"].str.strip()
).tolist()


# load SentenceTransformer model
model = SentenceTransformer(MODEL_NAME)

# generate embeddings:
# - texts
# - batch_size=64
# - show_progress_bar=True
# - normalize_embeddings=True
embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True,
)

# print:
# - number of texts
# - embeddings shape
# - norm of the first embedding
print(f"Number of encoded texts: {len(texts)}")
print(f"Embeddings shape: {embeddings.shape}")
print(f"Norm of the first embedding: {np.linalg.norm(embeddings[0]):.6f}")

# save embeddings as .npy
np.save(OUTPUT_FILE, embeddings)
print(f"Embeddings saved to: {OUTPUT_FILE}")