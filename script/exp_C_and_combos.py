"""
Experiment C, A+C, B+C  ->  LightGBM
  C   = qmap API-spotdata aggregated features only
  A+C = tag features + spot features
  B+C = CLIP image embeddings + spot features
Requires clip_emb.npz produced by exp_B_clip.py (only for B+C).
Run:  DATA_ROOT=/path/to/extracted python3 exp_C_and_combos.py
"""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from common import load_master, build_tag_features, build_spot_features, TARGETS

N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
OUT_DIR = os.environ.get("RESULT_DIR", "../result/exp_C")
os.makedirs(OUT_DIR, exist_ok=True)
# CLIP embeddings produced by exp_B (needed only for B+C)
CLIP_CACHE = os.environ.get("CLIP_CACHE", "../result/exp_B/clip_emb.npz")

posts = load_master()
spot = build_spot_features(posts)              # block C, aligned to posts rows
rows = []

def evaluate(X, y, name, target):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=42, stratify=y)
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                           num_leaves=31, n_jobs=N_JOBS, verbose=-1)
    m.fit(Xtr, ytr); p = m.predict(Xte)
    acc = accuracy_score(yte, p); f1 = f1_score(yte, p, average="macro")
    print(f"  {name:6}{acc:>11.3f}{f1:>10.3f}")
    rows.append({"target": target, "experiment": name, "N": len(y),
                 "accuracy": round(acc, 4), "macro_f1": round(f1, 4)})

# optional CLIP embeddings for B+C
emb = idx = None
if os.path.exists(CLIP_CACHE):
    d = np.load(CLIP_CACHE); emb, idx = d["emb"], d["idx"]
else:
    print(f"(no CLIP cache at {CLIP_CACHE} -> skipping B+C; run exp_B first)")

for t in TARGETS:
    print(f"\n{t}   {'(exp)':6}{'accuracy':>11}{'macroF1':>10}")
    mask = posts[t].notna().values
    y = LabelEncoder().fit_transform(posts.loc[mask, t])

    # C only
    evaluate(spot[mask].values, y, "C", t)

    # A + C
    tag = build_tag_features(posts, t)[mask]
    ac = pd.concat([tag.reset_index(drop=True),
                    spot[mask].reset_index(drop=True)], axis=1)
    evaluate(ac.values, y, "A+C", t)

    # B + C  (only rows that have an image)
    if emb is not None:
        img_mask = np.zeros(len(posts), dtype=bool); img_mask[idx] = True
        both = mask & img_mask
        emb_row = {r: emb[k] for k, r in enumerate(idx)}
        idxs = np.where(both)[0]
        Xb = np.array([emb_row[r] for r in idxs])
        Xc = spot.values[idxs]
        Xbc = np.concatenate([Xb, Xc], axis=1)
        yb = LabelEncoder().fit_transform(posts.loc[both, t])
        evaluate(Xbc, yb, "B+C", t)

out_csv = os.path.join(OUT_DIR, "exp_C_results.csv")
pd.DataFrame(rows).to_csv(out_csv, index=False)
print(f"\nsaved -> {out_csv}")