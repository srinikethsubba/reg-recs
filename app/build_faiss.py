import numpy as np
import faiss
from pathlib import Path

MODELS_DIR = Path("models")
INDEX_DIR  = Path("index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

def main():
    vecs = np.load(MODELS_DIR / "item_vectors.npy").astype("float32")
    print(">> Loaded item_vectors.npy", vecs.shape)

    index = faiss.IndexFlatIP(vecs.shape[1])  # inner product search
    faiss.normalize_L2(vecs)  # normalize for cosine similarity
    index.add(vecs)

    faiss.write_index(index, str(INDEX_DIR / "faiss_items.idx"))
    print(">> Wrote index/faiss_items.idx with", index.ntotal, "items")

if __name__ == "__main__":
    main()
