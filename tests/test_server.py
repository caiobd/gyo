import json
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from gyo.api.server import create_app, _dataset_fingerprint
from gyo.api.server import _level_columns, _validated_final_residual


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
    assert body["dataset_id"] == _dataset_fingerprint(tmp_path)
    assert body["num_levels"] == 2
    assert body["projection"] == {
        "method": "metric-mds",
        "metric": "euclidean",
        "raw_stress": body["projection"]["raw_stress"],
        "stress": body["projection"]["stress"],
        "distances": body["projection"]["distances"],
    }
    distances = np.asarray(body["projection"]["distances"])
    assert distances.shape == (len(body["children"]), len(body["children"]))
    np.testing.assert_allclose(distances, distances.T)
    np.testing.assert_allclose(np.diag(distances), 0)
    for node in [body["focus"], *body["children"]]:
        assert set(node["samples"]) == {"representative", "outliers"}
        assert all(
            set(item) == {"idx", "path", "label"}
            for item in node["samples"]["representative"]
        )
    assert "embeddings" not in body


def test_dataset_fingerprint_is_stable_and_changes_with_input(tmp_path):
    _seed_run(tmp_path)
    first = _dataset_fingerprint(tmp_path)
    assert _dataset_fingerprint(tmp_path) == first
    values = np.load(tmp_path / "codebooks/v1/level_0.npy")
    values[0, 0] = 9
    np.save(tmp_path / "codebooks/v1/level_0.npy", values)
    assert _dataset_fingerprint(tmp_path) != first


def test_atlas_mutation_invalidates_loaded_run(tmp_path):
    _seed_run(tmp_path)
    client = TestClient(create_app(str(tmp_path)))
    first = client.get("/api/atlas/root").json()
    values = np.load(tmp_path / "codebooks/v1/level_0.npy")
    values[1] = [5, 0]
    np.save(tmp_path / "codebooks/v1/level_0.npy", values)
    second = client.get("/api/atlas/root").json()
    assert second["dataset_id"] != first["dataset_id"]
    assert second["projection"]["distances"] != first["projection"]["distances"]


def test_geometry_persists_across_app_instances(tmp_path, monkeypatch):
    _seed_run(tmp_path)
    import gyo.api.server as server
    calls = 0
    original = server.metric_mds
    def counted(distances):
        nonlocal calls
        calls += 1
        return original(distances)
    monkeypatch.setattr(server, "metric_mds", counted)
    TestClient(server.create_app(str(tmp_path))).get("/api/atlas/root")
    TestClient(server.create_app(str(tmp_path))).get("/api/atlas/root")
    assert calls == 1


def test_corrupt_geometry_cache_is_rebuilt(tmp_path):
    _seed_run(tmp_path)
    cache = tmp_path / "atlas/v1/geometry.json"
    cache.parent.mkdir(parents=True)
    cache.write_text("not json")
    response = TestClient(create_app(str(tmp_path))).get("/api/atlas/root")
    assert response.status_code == 200
    assert json.loads(cache.read_text())["dataset_id"] == response.json()["dataset_id"]


@pytest.mark.parametrize("saved", [
    {"positions": [[2.0, 0.0], [0.0, 0.0]], "raw_stress": 0.1, "child_prefixes": [[0], [1]]},
    {"positions": [[1.0 + np.finfo(float).eps, 0.0], [0.0, 0.0]], "raw_stress": 0.1, "child_prefixes": [[0], [1]]},
    {"positions": [[0.0, 0.0], [0.0, 0.0]], "raw_stress": -0.1, "child_prefixes": [[0], [1]]},
    {"positions": [[0.0, 0.0], [0.0, 0.0]], "raw_stress": 0.1, "child_prefixes": [[1], [0]]},
])
def test_invalid_persisted_geometry_is_rebuilt(tmp_path, monkeypatch, saved):
    _seed_run(tmp_path)
    import gyo.api.server as server
    client = TestClient(server.create_app(str(tmp_path)))
    dataset_id = client.get("/api/dataset").json()["dataset_id"]
    cache = tmp_path / "atlas/v1/geometry.json"; cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"dataset_id": dataset_id, "version": 1, "geometries": {"root": saved}}))
    calls = 0; original = server.metric_mds
    def counted(distances):
        nonlocal calls; calls += 1; return original(distances)
    monkeypatch.setattr(server, "metric_mds", counted)
    response = TestClient(server.create_app(str(tmp_path))).get("/api/atlas/root")
    assert response.status_code == 200 and calls == 1
    assert all(abs(coordinate) <= 1.0 for child in response.json()["children"] for coordinate in child["position"])
    rebuilt = json.loads(cache.read_text())["geometries"]["root"]
    assert rebuilt["child_prefixes"] == [[0], [1]]


def test_unwritable_geometry_cache_falls_back_to_memory(tmp_path, monkeypatch):
    _seed_run(tmp_path)
    import gyo.api.server as server
    monkeypatch.setattr(server.tempfile, "NamedTemporaryFile", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("read only")))
    client = TestClient(server.create_app(str(tmp_path)))
    first = client.get("/api/atlas/root")
    second = client.get("/api/atlas/root")
    assert first.status_code == second.status_code == 200
    assert first.json()["children"] == second.json()["children"]


def test_all_data_endpoints_revalidate_after_dataset_mutation(tmp_path):
    _seed_run(tmp_path)
    client = TestClient(create_app(str(tmp_path)))
    first_tree = client.get("/api/tree").json()
    first_thumb = client.get("/thumb/0")
    codes = pd.read_parquet(tmp_path / "codes.parquet")
    codes["c_0"] = 1
    codes.to_parquet(tmp_path / "codes.parquet", index=False)
    atlas = client.get("/api/atlas/root")
    second_tree = client.get("/api/tree").json()
    second_thumb = client.get("/thumb/0")
    assert atlas.status_code == 200
    assert second_tree["dataset_id"] != first_tree["dataset_id"]
    assert second_tree["nodes"] != first_tree["nodes"]
    assert second_thumb.headers["x-dataset-id"] != first_thumb.headers["x-dataset-id"]
    assert client.get("/api/dataset").json()["dataset_id"] == second_tree["dataset_id"]


def test_metadata_only_mutation_changes_identity_and_reloads_consumers(tmp_path):
    _seed_run(tmp_path)
    client = TestClient(create_app(str(tmp_path)))
    first_id = client.get("/api/dataset").json()["dataset_id"]
    client.get("/api/atlas/root")
    first_node = client.get("/api/node/0").json()
    first_thumb = client.get("/thumb/0").content

    meta = pd.read_parquet(tmp_path / "meta.parquet")
    meta["label"] = ["New A", "New B", "New C"]
    meta.loc[meta["idx"] == 0, "path"] = "000002.png"
    meta.to_parquet(tmp_path / "meta.parquet", index=False)

    second_id = client.get("/api/dataset").json()["dataset_id"]
    atlas = client.get("/api/atlas/root").json()
    second_node = client.get("/api/node/0").json()
    second_thumb = client.get("/thumb/0").content
    assert second_id != first_id
    assert second_node["items"] != first_node["items"]
    assert second_node["items"][0]["label"] == "New A"
    assert second_node["items"][0]["path"] == "000002.png"
    assert second_thumb != first_thumb
    atlas_items = [item for node in [atlas["focus"], *atlas["children"]] for item in node["samples"]["representative"]]
    assert any(item["idx"] == 0 and item["label"] == "New A" and item["path"] == "000002.png" for item in atlas_items)


def test_concurrent_apps_merge_persisted_prefix_geometry(tmp_path):
    _seed_run(tmp_path)
    clients = [TestClient(create_app(str(tmp_path))) for _ in range(2)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda pair: pair[0].get(pair[1]), zip(clients, ["/api/atlas/root", "/api/atlas/0"])))
    assert all(response.status_code == 200 for response in responses)
    saved = json.loads((tmp_path / "atlas/v1/geometry.json").read_text())
    assert set(saved["geometries"]) >= {"root", "0"}


def test_atlas_rejects_focus_above_sibling_safety_limit(tmp_path):
    _seed_run(tmp_path)
    from gyo.api.server import MAX_ATLAS_SIBLINGS
    count = MAX_ATLAS_SIBLINGS + 1
    pd.DataFrame({
        "idx": range(count), "c_0": range(count), "final_residual": np.zeros(count),
    }).to_parquet(tmp_path / "codes.parquet", index=False)
    pd.DataFrame({
        "idx": range(count), "path": ["000000.png"] * count, "label": ["A"] * count,
    }).to_parquet(tmp_path / "meta.parquet", index=False)
    np.save(tmp_path / "embeddings.npy", np.zeros((count, 2)))
    codebooks = tmp_path / "codebooks/v1"
    np.save(codebooks / "level_0.npy", np.column_stack((np.arange(count), np.zeros(count))))
    (codebooks / "level_1.npy").unlink()
    (codebooks / "config.json").write_text(json.dumps({"num_levels": 1, "codebook_size": count, "dim": 2}))
    response = TestClient(create_app(str(tmp_path))).get("/api/atlas/root")
    assert response.status_code == 409
    assert "supported per-focus maximum" in response.json()["detail"]
    assert "smaller codebook" in response.json()["detail"]
    assert TestClient(create_app(str(tmp_path))).get("/api/tree").status_code == 200


def test_atlas_accepts_exact_sibling_product_limit(tmp_path, monkeypatch):
    _seed_run(tmp_path)
    import gyo.api.server as server
    count = server.MAX_ATLAS_SIBLINGS
    pd.DataFrame({"idx": range(count), "c_0": range(count), "final_residual": np.zeros(count)}).to_parquet(tmp_path / "codes.parquet", index=False)
    pd.DataFrame({"idx": range(count), "path": ["000000.png"] * count, "label": ["A"] * count}).to_parquet(tmp_path / "meta.parquet", index=False)
    np.save(tmp_path / "embeddings.npy", np.zeros((count, 2)))
    codebooks = tmp_path / "codebooks/v1"
    np.save(codebooks / "level_0.npy", np.column_stack((np.arange(count), np.zeros(count))))
    (codebooks / "level_1.npy").unlink()
    (codebooks / "config.json").write_text(json.dumps({"num_levels": 1, "codebook_size": count, "dim": 2}))
    monkeypatch.setattr(server, "metric_mds", lambda distances: SimpleNamespace(positions=np.zeros((count, 2)), stress=0.0))
    response = TestClient(server.create_app(str(tmp_path))).get("/api/atlas/root")
    assert response.status_code == 200 and len(response.json()["children"]) == count


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


def test_level_columns_are_numeric_and_contiguous():
    columns = [f"c_{level}" for level in range(11)]
    assert _level_columns(reversed(columns)) == columns
    with pytest.raises(ValueError, match="contiguous"):
        _level_columns(["c_0", "c_2"])


def test_final_residual_validation_rejects_shape_count_and_nonfinite():
    for values, rows in [([[1.0], [2.0]], 2), ([1.0], 2), ([1.0, np.nan], 2)]:
        with pytest.raises(ValueError, match="final_residual"):
            _validated_final_residual(values, rows)


def test_atlas_rejects_fractional_codes(tmp_path):
    _seed_run(tmp_path)
    codes = pd.read_parquet(tmp_path / "codes.parquet")
    codes["c_0"] = codes["c_0"].astype(float)
    codes.loc[0, "c_0"] = 0.5
    codes.to_parquet(tmp_path / "codes.parquet", index=False)

    response = TestClient(create_app(str(tmp_path))).get("/api/atlas/root")
    assert response.status_code == 409
    assert "integral" in response.json()["detail"]


@pytest.mark.parametrize(
    "mutation, detail",
    [
        ("malformed-config", "config"),
        ("missing-config-key", "config"),
        ("corrupt-embedding", "embeddings.npy"),
        ("corrupt-codebook", "level_0.npy"),
        ("codebook-shape", "2D"),
        ("codebook-nonfinite", "finite"),
    ],
)
def test_atlas_reports_malformed_serialized_inputs_as_conflicts(
    tmp_path, mutation, detail
):
    _seed_run(tmp_path)
    config_path = tmp_path / "codebooks" / "v1" / "config.json"
    if mutation == "malformed-config":
        config_path.write_text("{")
    elif mutation == "missing-config-key":
        config_path.write_text(json.dumps({"num_levels": 2}))
    elif mutation == "corrupt-embedding":
        (tmp_path / "embeddings.npy").write_bytes(b"not-npy")
    elif mutation == "corrupt-codebook":
        (config_path.parent / "level_0.npy").write_bytes(b"not-npy")
    elif mutation == "codebook-shape":
        np.save(config_path.parent / "level_0.npy", np.zeros(2))
    else:
        codebook = np.load(config_path.parent / "level_0.npy")
        codebook[0, 0] = np.inf
        np.save(config_path.parent / "level_0.npy", codebook)

    response = TestClient(create_app(str(tmp_path))).get("/api/atlas/root")
    assert response.status_code == 409
    assert detail in response.json()["detail"]


@pytest.mark.parametrize("target", ["embeddings", "codebook"])
def test_atlas_rejects_complex_vectors(tmp_path, target):
    _seed_run(tmp_path)
    if target == "embeddings":
        path = tmp_path / "embeddings.npy"
    else:
        path = tmp_path / "codebooks" / "v1" / "level_0.npy"
    np.save(path, np.load(path).astype(np.complex64))

    response = TestClient(create_app(str(tmp_path))).get("/api/atlas/root")
    assert response.status_code == 409
    assert "real numeric" in response.json()["detail"]


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
    for item_id in (10, 20, 30):
        Image.new("L", (2, 2), color=item_id).save(
            tmp_path / "images" / f"id-{item_id}.png"
        )

    client = TestClient(create_app(str(tmp_path)))
    response = client.get("/api/atlas/root")
    assert response.status_code == 200
    items = response.json()["focus"]["samples"]["representative"]
    assert {(item["idx"], item["path"], item["label"]) for item in items} == {
        (10, "id-10.png", "ten"),
        (20, "id-20.png", "twenty"),
        (30, "id-30.png", "thirty"),
    }
    node_items = client.get("/api/node/0").json()["items"]
    assert {(item["idx"], item["path"]) for item in node_items} == {
        (10, "id-10.png"),
        (20, "id-20.png"),
    }
    for item in items:
        thumb = client.get(f"/thumb/{item['idx']}")
        assert thumb.status_code == 200
        assert Image.open(BytesIO(thumb.content)).getpixel((0, 0)) == item["idx"]


def test_atlas_rejects_codes_without_metadata(tmp_path):
    _seed_run(tmp_path)
    meta = pd.read_parquet(tmp_path / "meta.parquet")
    meta.iloc[:2].to_parquet(tmp_path / "meta.parquet", index=False)

    response = TestClient(create_app(str(tmp_path))).get("/api/atlas/root")
    assert response.status_code == 409
    assert "metadata" in response.json()["detail"]
    assert "2" in response.json()["detail"]


@pytest.mark.parametrize(
    "mutation, detail",
    [
        ("row-mismatch", "rows"),
        ("dim-mismatch", "dimension"),
        ("invalid-code", "code index"),
        ("nonfinite", "finite"),
        ("duplicate-code-id", "duplicate codes"),
        ("duplicate-meta-id", "duplicate metadata"),
    ],
)
def test_atlas_validates_run_inputs(tmp_path, mutation, detail):
    _seed_run(tmp_path)
    if mutation == "row-mismatch":
        np.save(tmp_path / "embeddings.npy", np.zeros((2, 2), dtype=np.float32))
    elif mutation == "dim-mismatch":
        np.save(tmp_path / "embeddings.npy", np.zeros((3, 3), dtype=np.float32))
    elif mutation == "invalid-code":
        codes = pd.read_parquet(tmp_path / "codes.parquet")
        codes.loc[0, "c_0"] = 2
        codes.to_parquet(tmp_path / "codes.parquet", index=False)
    elif mutation == "nonfinite":
        embeddings = np.load(tmp_path / "embeddings.npy")
        embeddings[0, 0] = np.nan
        np.save(tmp_path / "embeddings.npy", embeddings)
    elif mutation == "duplicate-code-id":
        codes = pd.read_parquet(tmp_path / "codes.parquet")
        codes["idx"] = [0, 0, 2]
        codes.to_parquet(tmp_path / "codes.parquet", index=False)
    else:
        meta = pd.read_parquet(tmp_path / "meta.parquet")
        meta["idx"] = [0, 0, 2]
        meta.to_parquet(tmp_path / "meta.parquet", index=False)

    response = TestClient(create_app(str(tmp_path))).get("/api/atlas/root")
    assert response.status_code == 409
    assert detail in response.json()["detail"]


def test_atlas_caches_stats_and_rankings(tmp_path, monkeypatch):
    _seed_run(tmp_path)
    import gyo.api.server as server

    counts = {"stats": 0, "ranked": 0}
    real_stats = server.node_stats
    real_ranked = server.ranked_samples

    def counting_stats(*args, **kwargs):
        counts["stats"] += 1
        return real_stats(*args, **kwargs)

    def counting_ranked(*args, **kwargs):
        counts["ranked"] += 1
        return real_ranked(*args, **kwargs)

    monkeypatch.setattr(server, "node_stats", counting_stats)
    monkeypatch.setattr(server, "ranked_samples", counting_ranked)
    client = TestClient(server.create_app(str(tmp_path)))
    client.get("/api/atlas/root")
    first = counts.copy()
    client.get("/api/atlas/root")
    assert counts == first
    assert counts["stats"] == 1


def test_atlas_purity_ignores_missing_labels(tmp_path):
    _seed_run(tmp_path)
    meta = pd.read_parquet(tmp_path / "meta.parquet")
    meta["label"] = [None, None, "B"]
    meta.to_parquet(tmp_path / "meta.parquet", index=False)
    client = TestClient(create_app(str(tmp_path)))

    assert client.get("/api/atlas/0").json()["focus"]["purity"] is None
    assert client.get("/api/atlas/root").json()["focus"]["purity"] == 1.0


def test_thumb_rejects_metadata_path_traversal(tmp_path):
    _seed_run(tmp_path)
    meta = pd.read_parquet(tmp_path / "meta.parquet")
    meta.loc[0, "path"] = "../codes.parquet"
    meta.to_parquet(tmp_path / "meta.parquet", index=False)

    assert TestClient(create_app(str(tmp_path))).get("/thumb/0").status_code == 404
