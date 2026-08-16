"""
common.py - shared data preparation used by all experiments.

Builds the master posting-data table (A), the label targets (Y1/Y2/Y3),
and the qmap API-spotdata features (C). Import this from every experiment.

IMPORTANT: email / any personal info is dropped immediately and never used.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ----- edit this to point at your top-level data folder -----
# Layout (from the screenshot):
#   <ROOT>/app_data/posting_data/QuestPOST_csv/ Posts.csv + photo/*.png
#   <ROOT>/app_data/posting_data/gnavi_db_csv/ Posts.csv + photo/*.png
#   <ROOT>/app_data/api_spotdata/qmap_db_silk_csv/ *.csv
#   <ROOT>/app_data/api_spotdata/qmap_db_experiment_csv/ *.csv
# Scripts live in script/, so default ROOT is ../data (one level up).
ROOT = os.environ.get("DATA_ROOT", "../data")

POSTING_DIR = os.path.join(ROOT, "app_data", "posting_data")
SPOT_DIR = os.path.join(ROOT, "app_data", "api_spotdata")

QUEST_DIR = os.path.join(POSTING_DIR, "QuestPOST_csv")
GNAVI_DIR = os.path.join(POSTING_DIR, "gnavi_db_csv")
SILK_DIR  = os.path.join(SPOT_DIR, "qmap_db_silk_csv")

# Label column -> target meaning
#   label2 = sound, label3 = smell, label5 = mood
TARGETS = {"Y1_mood": "label5", "Y2_sound": "label2", "Y3_smell": "label3"}


def _load_posts(path, source):
    """Load one Posts.csv, drop personal info, keep numeric coords."""
    df = pd.read_csv(path, on_bad_lines="skip")
    df = df.rename(columns={"longitude": "lng"})
    df["source"] = source
    # never leak email / user identifiers
    df = df.drop(columns=[c for c in ["email"] if c in df.columns])
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lng"] = pd.to_numeric(df["lng"], errors="coerce")
    df = df.dropna(subset=["lat", "lng"]).reset_index(drop=True)
    # local image filename (strip server URL / any path prefix)
    df["img_file"] = df["image_path"].astype(str).str.split("/").str[-1]
    return df


def clean_label(series, topk=6):
    """Keep only the first tag, strip '#', collapse long tail into 'other'."""
    s = series.astype(str).str.replace("#", "", regex=False)
    s = s.str.split(",").str[0].str.strip()
    s = s.replace(["nan", "None", ""], np.nan)
    top = s.value_counts().index[:topk]
    return s.where(s.isin(top), other="other")


def load_master():
    """Return the merged, cleaned posting table with Y1/Y2/Y3 targets."""
    a = _load_posts(os.path.join(QUEST_DIR, "Posts.csv"), "quest")
    b = _load_posts(os.path.join(GNAVI_DIR, "Posts.csv"), "gnavi")
    posts = pd.concat([a, b], ignore_index=True)
    for y, col in TARGETS.items():
        posts[y] = clean_label(posts[col])
    return posts


def build_tag_features(posts, target):
    """Feature block A: coords + other tag columns (excludes the target column)."""
    feat = posts[["lat", "lng"]].copy()
    exclude = TARGETS[target]  # don't feed the target's own source column
    tag_cols = [c for c in ["label1", "label2", "label3", "label4", "label5", "label6"] if c != exclude]
    for c in tag_cols:
        feat[c] = LabelEncoder().fit_transform(posts[c].astype(str))
    feat["source"] = LabelEncoder().fit_transform(posts["source"])
    return feat


def build_spot_features(posts):
    """
    Feature block C: qmap API-spotdata aggregated per location.

    We join by a rounded lat/lng grid because posts and qmap quests share the
    same physical locations. eval_q gives per-location sensory eval scores.
    """
    eval_q = pd.read_csv(os.path.join(SILK_DIR, "eval_q.csv"), on_bad_lines="skip")
    quest  = pd.read_csv(os.path.join(SILK_DIR, "quest.csv"),  on_bad_lines="skip")

    # align key dtypes before merge
    for df_ in (eval_q, quest):
        df_["quest_id"] = pd.to_numeric(df_["quest_id"], errors="coerce")
    eval_q = eval_q.dropna(subset=["quest_id"])
    quest = quest.dropna(subset=["quest_id"])
    quest["lat"] = pd.to_numeric(quest["lat"], errors="coerce")
    quest["lng"] = pd.to_numeric(quest["lng"], errors="coerce")

    # per-quest eval stats
    ev = (eval_q.groupby("quest_id")["eval"]
          .agg(spot_eval_mean="mean", spot_eval_max="max", spot_eval_cnt="count")
          .reset_index())
    q = quest[["quest_id", "lat", "lng"]].merge(ev, on="quest_id", how="left")
    q["gkey"] = (q["lat"].round(3).astype(str) + "_" + q["lng"].round(3).astype(str))
    spot = (q.groupby("gkey")[["spot_eval_mean", "spot_eval_max", "spot_eval_cnt"]]
            .mean().reset_index())

    posts = posts.copy()
    posts["gkey"] = (posts["lat"].round(3).astype(str) + "_" +
                     posts["lng"].round(3).astype(str))
    merged = posts.merge(spot, on="gkey", how="left")
    cols = ["spot_eval_mean", "spot_eval_max", "spot_eval_cnt"]
    merged[cols] = merged[cols].fillna(0)
    return merged[cols].reset_index(drop=True)


def find_image_path(img_file):
    """Locate a post image by filename in either photo folder. None if missing."""
    for d in (QUEST_DIR, GNAVI_DIR):
        p = os.path.join(d, "photo", img_file)
        if os.path.exists(p):
            return p
    return None