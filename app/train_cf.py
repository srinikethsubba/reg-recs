# app/train_cf.py — Train a simple two-tower CF model and export item vectors
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

DATA_RATINGS = Path("data/ratings.csv")
MODELS_DIR   = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

class Interactions(Dataset):
    def __init__(self, ratings, u2i, m2i):
        self.u = ratings["userId"].map(u2i).astype(np.int64).values
        self.m = ratings["movieId"].map(m2i).astype(np.int64).values
    def __len__(self): return len(self.u)
    def __getitem__(self, i): return int(self.u[i]), int(self.m[i])

class TwoTower(nn.Module):
    def __init__(self, n_users, n_items, d=64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, d)
        self.item_emb = nn.Embedding(n_items, d)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
    def forward(self, u, it):
        u_vec = nn.functional.normalize(self.user_emb(u), dim=1)
        i_vec = nn.functional.normalize(self.item_emb(it), dim=1)
        score = (u_vec * i_vec).sum(dim=1)
        return score, u_vec, i_vec

def bpr_loss(pos, neg):
    return -torch.log(torch.sigmoid(pos - neg) + 1e-8).mean()

def main():
    if not DATA_RATINGS.exists():
        raise FileNotFoundError("data/ratings.csv not found. Put MovieLens ratings.csv in data/")
    print(">> Loading ratings…")
    ratings = pd.read_csv(DATA_RATINGS, usecols=["userId","movieId","rating"])
    ratings = ratings.loc[ratings["rating"] >= 4.0, ["userId","movieId"]].copy()
    print(f">> Positive interactions: {len(ratings):,}")

    users = ratings["userId"].unique()
    movies = ratings["movieId"].unique()
    u2i = {u:i for i,u in enumerate(users)}
    m2i = {m:i for i,m in enumerate(movies)}
    print(f">> Users: {len(users):,} | Items: {len(movies):,}")

    ds = Interactions(ratings, u2i, m2i)
    dl = DataLoader(ds, batch_size=4096, shuffle=True, drop_last=True)

    model = TwoTower(len(users), len(movies), d=64)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    epochs = 6
    for ep in range(1, epochs+1):
        model.train()
        total, n = 0.0, 0
        for u, it in dl:
            u = u.long(); it = it.long()
            pos, _, _ = model(u, it)
            it_neg = torch.randint(0, len(m2i), (len(it),), dtype=torch.long)
            neg, _, _ = model(u, it_neg)
            loss = bpr_loss(pos, neg)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.item()) * len(u); n += len(u)
        print(f"epoch {ep}/{epochs}  bpr_loss={total/n:.4f}")

    ckpt = {"state_dict": model.state_dict(), "n_users": len(users), "n_items": len(movies),
            "u2i": u2i, "m2i": m2i}
    torch.save(ckpt, MODELS_DIR / "two_tower.pt")
    print(">> Saved models/two_tower.pt")

    with torch.no_grad():
        item_vecs = nn.functional.normalize(model.item_emb.weight, dim=1).cpu().numpy().astype("float32")
    np.save(MODELS_DIR / "item_vectors.npy", item_vecs)
    print(">> Saved models/item_vectors.npy  shape", item_vecs.shape)

if __name__ == "__main__":
    main()
