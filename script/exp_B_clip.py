"""
Experiment B: posting data WITH image  ->  CLIP features + classifier
Extracts a 512-dim CLIP embedding per post image, then trains a simple
classifier for each target. Embeddings are cached to clip_emb.npz.

Install:  pip install torch torchvision ftfy regex tqdm
          pip install git+https://github.com/openai/CLIP.git
Run:      DATA_ROOT=/path/to/extracted python3 exp_B_clip.py

NOTE: ResNet/ViT variant = swap the encoder in embed_image() for a
torchvision model with its classification head removed; the rest is identical.
"""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, clip
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from common import load_master, find_image_path, TARGETS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = os.environ.get("RESULT_DIR", "../result/exp_B")
os.makedirs(OUT_DIR, exist_ok=True)
CACHE = os.path.join(OUT_DIR, "clip_emb.npz")

def build_embeddings(posts):
    """Return (embeddings [N,512], row index array) for posts with a real image."""
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        return d["emb"], d["idx"]
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    model.eval()
    embs, idxs = [], []
    for i, fn in tqdm(list(enumerate(posts["img_file"])), desc="CLIP"):
        p = find_image_path(fn)
        if p is None:
            continue
        try:
            img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                v = model.encode_image(img).squeeze().cpu().numpy()
            embs.append(v); idxs.append(i)
        except Exception:
            continue
    emb = np.array(embs, dtype=np.float32); idx = np.array(idxs)
    np.savez(CACHE, emb=emb, idx=idx)
    return emb, idx

posts = load_master()
emb, idx = build_embeddings(posts)
print(f"images embedded: {len(idx)}")
print(f"{'Target':10}{'N':>7}{'accuracy':>11}{'macroF1':>10}")

rows = []
for t in TARGETS:
    y_all = posts[t].values[idx]
    keep = np.array([v is not None and str(v) != "nan" for v in y_all])
    X = emb[keep]; y = LabelEncoder().fit_transform(y_all[keep].astype(str))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr, ytr)
    p = clf.predict(Xte)
    acc = accuracy_score(yte, p); f1 = f1_score(yte, p, average="macro")
    print(f"{t:10}{len(y):>7}{acc:>11.3f}{f1:>10.3f}")
    rows.append({"target": t, "N": len(y),
                 "accuracy": round(acc, 4), "macro_f1": round(f1, 4)})

out_csv = os.path.join(OUT_DIR, "exp_B_results.csv")
pd.DataFrame(rows).to_csv(out_csv, index=False)
print(f"\nsaved -> {out_csv}")