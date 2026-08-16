"""
Experiment A: posting data WITHOUT image  ->  LightGBM
Features = coords + other tag columns (block A only).
Run:  DATA_ROOT=/path/to/extracted python3 exp_A_lgbm.py
"""
import os, warnings; warnings.filterwarnings("ignore")
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from common import load_master, build_tag_features, TARGETS

# use all CPU cores Slurm gave us (falls back to all local cores)
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))

# where to save results
OUT_DIR = os.environ.get("RESULT_DIR", "../result/exp_A")
os.makedirs(OUT_DIR, exist_ok=True)

posts = load_master()
print(f"{'Target':10}{'N':>7}{'cls':>5}{'accuracy':>11}{'macroF1':>10}")

rows = []
for t in TARGETS:
    X = build_tag_features(posts, t)
    mask = posts[t].notna()
    X = X[mask.values]
    y = LabelEncoder().fit_transform(posts.loc[mask, t])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, n_jobs=N_JOBS, verbose=-1)
    m.fit(Xtr, ytr)
    p = m.predict(Xte)
    acc = accuracy_score(yte, p)
    f1 = f1_score(yte, p, average="macro")
    n = len(y); k = posts.loc[mask, t].nunique()
    print(f"{t:10}{n:>7}{k:>5}{acc:>11.3f}{f1:>10.3f}")
    rows.append({"target": t, "N": n, "classes": k, "accuracy": round(acc, 4), "macro_f1": round(f1, 4)})

# save results table
out_csv = os.path.join(OUT_DIR, "exp_A_results.csv")
pd.DataFrame(rows).to_csv(out_csv, index=False)
print(f"\nsaved -> {out_csv}")