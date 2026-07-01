import numpy as np
import pandas as pd
from gyo.io.store import save_embeddings, load_embeddings, save_table, load_table


def test_embeddings_roundtrip(tmp_path):
    emb = np.random.rand(5, 4).astype(np.float32)
    p = tmp_path / "emb.npy"
    save_embeddings(p, emb)
    loaded = load_embeddings(p)
    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded, emb)


def test_table_roundtrip(tmp_path):
    df = pd.DataFrame({"idx": [0, 1], "path": ["a.png", "b.png"], "label": ["t", "u"]})
    p = tmp_path / "meta.parquet"
    save_table(p, df)
    loaded = load_table(p)
    pd.testing.assert_frame_equal(loaded, df)
