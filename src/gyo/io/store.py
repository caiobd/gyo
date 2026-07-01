from pathlib import Path
import numpy as np
import pandas as pd


def save_embeddings(path, emb: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb.astype(np.float32, copy=False))


def load_embeddings(path) -> np.ndarray:
    return np.load(path).astype(np.float32, copy=False)


def save_table(path, df: pd.DataFrame) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_table(path) -> pd.DataFrame:
    return pd.read_parquet(path)
