import numpy as np
import pandas as pd
from PIL import Image
from fastapi.testclient import TestClient
from gyo.api.server import create_app


def _seed_run(data_dir):
    img_dir = data_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        Image.new("L", (28, 28), color=i * 30).save(img_dir / f"{i:06d}.png")
    pd.DataFrame(
        {
            "idx": [0, 1, 2],
            "path": ["000000.png", "000001.png", "000002.png"],
            "label": ["A", "A", "B"],
        }
    ).to_parquet(data_dir / "meta.parquet", index=False)
    pd.DataFrame(
        {
            "idx": [0, 1, 2],
            "c_0": [0, 0, 1],
            "c_1": [0, 1, 0],
            "j": [0, 0, 0],
            "r_0": [1.0, 1.0, 1.0],
            "r_1": [0.5, 0.5, 0.5],
            "final_residual": [0.1, 0.2, 0.9],
        }
    ).to_parquet(data_dir / "codes.parquet", index=False)


def test_tree_endpoint(tmp_path):
    _seed_run(tmp_path)
    client = TestClient(create_app(str(tmp_path)))
    r = client.get("/api/tree?level=2")
    assert r.status_code == 200
    body = r.json()
    assert body["num_levels"] == 2
    prefixes = [tuple(n["prefix"]) for n in body["nodes"]]
    assert () in prefixes and (0,) in prefixes and (0, 1) in prefixes
    root = next(n for n in body["nodes"] if n["prefix"] == [])
    assert root["occupancy"] == 3


def test_node_items_and_thumb(tmp_path):
    _seed_run(tmp_path)
    client = TestClient(create_app(str(tmp_path)))
    r = client.get("/api/node/0")
    assert r.status_code == 200
    items = r.json()["items"]
    assert {it["idx"] for it in items} == {0, 1}
    t = client.get("/thumb/0")
    assert t.status_code == 200 and t.headers["content-type"].startswith("image/")


def test_root_serves_html(tmp_path):
    _seed_run(tmp_path)
    client = TestClient(create_app(str(tmp_path)))
    r = client.get("/")
    assert r.status_code == 200 and "<html" in r.text.lower()


def test_node_metrics_endpoint(tmp_path):
    _seed_run(tmp_path)
    client = TestClient(create_app(str(tmp_path)))
    r = client.get("/api/node/0/metrics")
    assert r.status_code == 200
    body = r.json()
    assert "occupancy" in body
    assert "mean_residual" in body
    assert "label_distribution" in body
    assert body["occupancy"] == 2
