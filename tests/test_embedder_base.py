import numpy as np
import pandas as pd
from PIL import Image
from gyo.embedders.base import DummyEmbedder, l2_normalize, list_images


def _make_images(folder, n=3):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("L", (28, 28), color=i * 10).save(folder / f"img_{i}.png")


def test_l2_normalize_unit_rows():
    x = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    out = l2_normalize(x)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0, 1.0], atol=1e-6)


def test_dummy_embedder_shapes_and_meta(tmp_path):
    _make_images(tmp_path, 3)
    pd.DataFrame({"path": ["img_0.png", "img_1.png"], "label": ["a", "b"]}).to_csv(
        tmp_path / "labels.csv", index=False
    )
    emb, meta = DummyEmbedder(dim=8).embed_folder(tmp_path)
    assert emb.shape == (3, 8)
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), np.ones(3), atol=1e-6)
    assert list(meta.columns) == ["idx", "path", "label"]
    assert meta.loc[0, "label"] == "a" and meta.loc[2, "label"] == ""


def test_list_images_sorted(tmp_path):
    _make_images(tmp_path, 3)
    files = list_images(tmp_path)
    assert [f.name for f in files] == ["img_0.png", "img_1.png", "img_2.png"]


import pytest


@pytest.mark.slow
def test_mobileclip_smoke(tmp_path):
    from gyo.embedders.mobileclip import MobileCLIPEmbedder

    _make_images(tmp_path, 2)
    emb, meta = MobileCLIPEmbedder().embed_folder(tmp_path)
    assert emb.shape[0] == 2
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), np.ones(2), atol=1e-5)
    assert list(meta.columns) == ["idx", "path", "label"]
