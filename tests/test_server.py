import json

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
    np.save(
        data_dir / "embeddings.npy",
        np.asarray([[0.0, 0.0], [0.0, 1.0], [2.0, 0.0]], dtype=np.float32),
    )
    codebook_dir = data_dir / "codebooks" / "v1"
    codebook_dir.mkdir(parents=True, exist_ok=True)
    np.save(codebook_dir / "level_0.npy", np.asarray([[0.0, 0.0], [2.0, 0.0]]))
    np.save(codebook_dir / "level_1.npy", np.asarray([[0.0, 0.0], [0.0, 1.0]]))
    (codebook_dir / "config.json").write_text(
        json.dumps(
            {
                "num_levels": 2,
                "codebook_size": 2,
                "dim": 2,
                "proj_dim": None,
                "seed": 0,
            }
        )
    )


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


def test_atlas_root_contract(tmp_path):
    _seed_run(tmp_path)
    body = TestClient(create_app(str(tmp_path))).get("/api/atlas/root").json()

    assert body["focus"]["prefix"] == []
    assert {tuple(child["prefix"]) for child in body["children"]} == {(0,), (1,)}
    assert all(len(child["position"]) == 2 for child in body["children"])
    assert body["projection"] == {
        "method": "metric-mds",
        "metric": "euclidean",
        "stress": body["projection"]["stress"],
        "warning": body["projection"]["stress"] > 0.10,
    }
    for node in [body["focus"], *body["children"]]:
        assert set(node["samples"]) == {"representative", "outliers"}
        assert all(
            set(item) == {"idx", "path", "label"}
            for item in node["samples"]["representative"]
        )
    assert "embeddings" not in body


def test_atlas_internal_prefix_and_leaf(tmp_path):
    _seed_run(tmp_path)
    client = TestClient(create_app(str(tmp_path)))

    internal = client.get("/api/atlas/0")
    assert internal.status_code == 200
    assert internal.json()["focus"]["prefix"] == [0]
    assert {tuple(child["prefix"]) for child in internal.json()["children"]} == {
        (0, 0),
        (0, 1),
    }
    assert all(
        "parent_distance" in child and "token_norm" in child
        for child in internal.json()["children"]
    )

    leaf = client.get("/api/atlas/0,1")
    assert leaf.status_code == 200
    assert leaf.json()["focus"]["prefix"] == [0, 1]
    assert leaf.json()["children"] == []
    assert leaf.json()["projection"]["stress"] == 0.0


def test_atlas_rejects_bad_or_unknown_prefix(tmp_path):
    _seed_run(tmp_path)
    client = TestClient(create_app(str(tmp_path)))

    assert client.get("/api/atlas/not-an-int").status_code == 400
    assert client.get("/api/atlas/0,,1").status_code == 400
    assert client.get("/api/atlas/9").status_code == 404


def test_atlas_missing_inputs_are_conflicts(tmp_path):
    _seed_run(tmp_path)
    (tmp_path / "embeddings.npy").unlink()

    response = TestClient(create_app(str(tmp_path))).get("/api/atlas/root")
    assert response.status_code == 409
    assert "embeddings.npy" in response.json()["detail"]

    _seed_run(tmp_path)
    (tmp_path / "codebooks" / "v1" / "level_1.npy").unlink()
    response = TestClient(create_app(str(tmp_path))).get("/api/atlas/root")
    assert response.status_code == 409
    assert "level_1.npy" in response.json()["detail"]


def test_atlas_joins_metadata_by_item_id(tmp_path):
    _seed_run(tmp_path)
    codes = pd.read_parquet(tmp_path / "codes.parquet")
    codes["idx"] = [10, 20, 30]
    codes.to_parquet(tmp_path / "codes.parquet", index=False)
    pd.DataFrame(
        {
            "idx": [30, 10, 20],
            "path": ["id-30.png", "id-10.png", "id-20.png"],
            "label": ["thirty", "ten", "twenty"],
        }
    ).to_parquet(tmp_path / "meta.parquet", index=False)

    response = TestClient(create_app(str(tmp_path))).get("/api/atlas/root")
    assert response.status_code == 200
    items = response.json()["focus"]["samples"]["representative"]
    assert {(item["idx"], item["path"], item["label"]) for item in items} == {
        (10, "id-10.png", "ten"),
        (20, "id-20.png", "twenty"),
        (30, "id-30.png", "thirty"),
    }


def test_atlas_rejects_codes_without_metadata(tmp_path):
    _seed_run(tmp_path)
    meta = pd.read_parquet(tmp_path / "meta.parquet")
    meta.iloc[:2].to_parquet(tmp_path / "meta.parquet", index=False)

    response = TestClient(create_app(str(tmp_path))).get("/api/atlas/root")
    assert response.status_code == 409
    assert "metadata" in response.json()["detail"]
    assert "2" in response.json()["detail"]
