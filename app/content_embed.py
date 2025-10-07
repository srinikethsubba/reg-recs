import pandas as pd
import numpy as np
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer

DATA_MOVIES = Path("data/movies.csv")
INDEX_DIR   = Path("index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

def main():
    if not DATA_MOVIES.exists():
        raise FileNotFoundError("data/movies.csv not found!")

    print(">> Loading movies.csv …")
    movies = pd.read_csv(DATA_MOVIES)
    movies = movies[["movieId","title"]].copy()

    # For now we’ll just use the title as text (you could extend with overviews/genres)
    texts = movies["title"].fillna("").tolist()

    print(">> Loading embedding model …")
    model = SentenceTransformer("all-MiniLM-L6-v2")  # lightweight, fast
    vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    print(">> Embeddings shape:", vecs.shape)

    # Build FAISS index
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs.astype("float32"))
    faiss.write_index(index, str(INDEX_DIR / "faiss_content.idx"))
    print(">> Wrote index/faiss_content.idx with", index.ntotal, "movies")

    # Save mapping (movieId <-> title)
    map_path = INDEX_DIR / "content_map.csv"
    movies.to_csv(map_path, index=False)
    print(">> Wrote", map_path)

if __name__ == "__main__":
    main()
