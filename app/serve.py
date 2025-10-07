# ================================================================
# app/serve.py — FastAPI Movie Recommender with RAG Explanations
# ================================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # fixes OpenMP warning on macOS

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pathlib import Path
import torch, faiss, pandas as pd
from torch import nn

# ------------------ Initialize FastAPI app ------------------
app = FastAPI(title="Movie Recommender (PyTorch + FAISS)")

# ------------------ Two-Tower Model Definition ------------------
class TwoTower(torch.nn.Module):
    def __init__(self, n_users, n_items, d=64):
        super().__init__()
        self.user_emb = torch.nn.Embedding(n_users, d)
        self.item_emb = torch.nn.Embedding(n_items, d)

# ------------------ Global Objects ------------------
_ckpt = None
_model = None
_index_items = None
_content_map = None
_inv_m2i = None

# ------------------ Load Model and Index on Startup ------------------
@app.on_event("startup")
def _load():
    global _ckpt, _model, _index_items, _content_map, _inv_m2i
    ckpt_path = Path("models/two_tower.pt")
    idx_path = Path("index/faiss_items.idx")
    cmap_path = Path("index/content_map.csv")

    if not ckpt_path.exists():
        raise RuntimeError("Missing model checkpoint")
    if not idx_path.exists():
        raise RuntimeError("Missing FAISS index")
    if not cmap_path.exists():
        raise RuntimeError("Missing content map")

    # Fix for PyTorch 2.6+ (allows loading dicts)
    import numpy as np
    import torch.serialization
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    _ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    _model = TwoTower(_ckpt["n_users"], _ckpt["n_items"], d=64)
    _model.load_state_dict(_ckpt["state_dict"])
    _model.eval()

    _index_items = faiss.read_index(str(idx_path))
    _content_map = pd.read_csv(cmap_path)
    _inv_m2i = {v: k for k, v in _ckpt["m2i"].items()}
    print(">> Core recommender model and FAISS index loaded.")

# ------------------ Health Check Endpoint ------------------
@app.get("/health")
def health():
    ok = all(x is not None for x in [_ckpt, _model, _index_items, _content_map])
    return {"status": "ok" if ok else "not_ready"}

# ------------------ Helper: Get User Vector ------------------
def _user_vec(user_id: int):
    i = _ckpt["u2i"].get(user_id)
    if i is None:
        return None
    with torch.no_grad():
        u = torch.tensor([i]).long()
        v = nn.functional.normalize(_model.user_emb(u), dim=1).numpy().astype("float32")
    return v

# ------------------ Main Recommender Endpoint ------------------
@app.get("/recommend")
def recommend(user_id: int, k: int = 10):
    v = _user_vec(user_id)
    if v is None:
        return JSONResponse({"error": "cold-start: user not in training data"}, status_code=400)

    scores, idxs = _index_items.search(v, k)
    internal = idxs[0].tolist()
    mids = [_inv_m2i[i] for i in internal]

    titles = (
        _content_map.set_index("movieId")
        .loc[mids]["title"]
        .reindex(mids)
        .fillna("")
        .tolist()
    )

    return {
        "items": [
            {"movieId": int(m), "title": t, "score": float(s)}
            for m, t, s in zip(mids, titles, scores[0].tolist())
        ]
    }

# ================================================================
# 🧠 RAG EXPLANATION ENDPOINT
# ================================================================
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np

_llm = None
_content_index = None
_content_embed_model = None
_content_map_df = None

@app.on_event("startup")
def _load_content_index():
    global _content_index, _content_embed_model, _content_map_df
    content_index_path = Path("index/faiss_content.idx")
    if content_index_path.exists():
        _content_index = faiss.read_index(str(content_index_path))
        _content_embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        _content_map_df = pd.read_csv("index/content_map.csv")
        print(">> Content FAISS index loaded for RAG explanations.")
    else:
        print(">> Warning: content FAISS index not found; RAG explanations disabled.")

@app.get("/recommend_explained")
def recommend_explained(user_id: int, k: int = 5):
    v = _user_vec(user_id)
    if v is None:
        return JSONResponse({"error": "cold-start: user not in training data"}, status_code=400)
    if _content_index is None:
        return JSONResponse({"error": "RAG index not loaded"}, status_code=500)

    # Step 1: Get top movies from CF
    scores, idxs = _index_items.search(v, k)
    internal = idxs[0].tolist()
    mids = [_inv_m2i[i] for i in internal]
    titles = _content_map.set_index("movieId").loc[mids]["title"].tolist()

    # Step 2: Retrieve similar movies from content index
    query_vecs = _content_embed_model.encode(titles, convert_to_numpy=True, normalize_embeddings=True)
    D, I = _content_index.search(query_vecs.astype("float32"), 3)

    # Step 3: Generate explanations using LLM
    global _llm
    if _llm is None:
        _llm = pipeline("text2text-generation", model="google/flan-t5-small")

    explanations = []
    for title, related_ids in zip(titles, I):
        related_titles = _content_map_df.iloc[related_ids]["title"].tolist()
        prompt = f"Explain why someone who liked {related_titles[0]} might also enjoy {title}."
        try:
            response = _llm(prompt, max_new_tokens=40, temperature=0.7)
            explanation = response[0]["generated_text"]
        except Exception as e:
            explanation = f"(Failed to generate: {e})"
        explanations.append({"title": title, "reason": explanation})

    return {"recommendations": explanations}

