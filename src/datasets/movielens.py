import os
import zipfile
from pathlib import Path

import pandas as pd

from .common import _download

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ML20M_URL = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"

def prepare_movielens(version="1m", out_csv="data/movielens.csv"):
    if version == "1m":
        url = ML1M_URL; zpath = "data/raw_ml1m.zip"
    elif version == "20m":
        url = ML20M_URL; zpath = "data/raw_ml20m.zip"
    else:
        raise ValueError("version must be '1m' or '20m'")
    _download(url, zpath)
    with zipfile.ZipFile(zpath, "r") as z:
        if version == "1m":
            with z.open("ml-1m/ratings.dat") as f:
                df = pd.read_csv(f, sep="::", header=None, engine="python",
                                 names=["user_id","item_id","rating","timestamp"])
        else:
            with z.open("ml-20m/ratings.csv") as f:
                df = pd.read_csv(f)
    df = df.rename(columns={"userId": "user_id", "movieId": "item_id"})
    if "rating" in df.columns:
        df = df[df["rating"] >= 3.5]
    df2 = df[["user_id", "item_id", "timestamp"]].copy()
    df2["user_id"] = df2["user_id"].astype(str)
    df2["item_id"] = df2["item_id"].astype(str)
    Path(os.path.dirname(out_csv) or ".").mkdir(parents=True, exist_ok=True)
    df2.to_csv(out_csv, index=False); return out_csv
