# 🎬 Movie Recommender with RAG (Retrieval-Augmented Generation)

This project is an **AI-powered Movie Recommendation System** that uses **PyTorch**, **FAISS**, and **FastAPI** to deliver intelligent, explainable recommendations.

## 🚀 Features
- **Collaborative Filtering** (Two-Tower PyTorch Model)
- **FAISS Vector Search** for similarity retrieval
- **RAG pipeline** for explainable recommendations
- **FastAPI REST API** backend
- **Content-based embeddings** via SentenceTransformers

## 🧠 Tech Stack
- PyTorch
- FAISS
- SentenceTransformers
- Transformers (FLAN-T5)
- FastAPI + Uvicorn

## ⚙️ Setup
```bash
git clone https://github.com/srinikethsubba/reg-recs.git
cd reg-recs
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.serve:app --reload

