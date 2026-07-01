import numpy as np
import pandas as pd
import pytest
from PIL import Image
from typer.testing import CliRunner
from fastapi.testclient import TestClient
from gyo.cli import app
from gyo.data.fashion_mnist import prepare_fashion_mnist
from gyo.api.server import create_app
from gyo.io.store import save_table

runner = CliRunner()


def _fake_fmnist(n=120):
    return [(Image.new("L", (28, 28), color=(i * 7) % 255), i % 10) for i in range(n)]


def test_full_pipeline_dummy(tmp_path):
    prepare_fashion_mnist(tmp_path, n=120, dataset=_fake_fmnist(120))
    assert (tmp_path / "labels.csv").exists()

    assert (
        runner.invoke(
            app, ["extract", "--data-dir", str(tmp_path), "--embedder", "dummy"]
        ).exit_code
        == 0
    )
    assert (
        runner.invoke(
            app,
            [
                "fit-rq",
                "--data-dir",
                str(tmp_path),
                "--levels",
                "3",
                "--codebook-size",
                "8",
                "--iters",
                "10",
            ],
        ).exit_code
        == 0
    )
    assert runner.invoke(app, ["encode", "--data-dir", str(tmp_path)]).exit_code == 0

    codes = pd.read_parquet(tmp_path / "codes.parquet")
    assert len(codes) == 120
    assert {"c_0", "c_1", "c_2", "j", "final_residual"} <= set(codes.columns)

    client = TestClient(create_app(str(tmp_path)))
    body = client.get("/api/tree?level=3").json()
    assert body["num_levels"] == 3
    root = next(n for n in body["nodes"] if n["prefix"] == [])
    assert root["occupancy"] == 120
    for n in body["nodes"]:
        assert set(n.keys()) == {
            "prefix",
            "level",
            "occupancy",
            "mean_residual",
            "size_norm",
            "residual_norm",
            "purity",
        }


def test_thumb_content_type_matches_extension_and_no_cache(tmp_path):
    """/thumb must report the real image type (jpg != png) and forbid caching,
    otherwise the browser serves stale thumbnails across datasets."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    Image.new("RGB", (8, 8), (123, 50, 200)).save(img_dir / "000000.jpg")
    Image.new("L", (8, 8), 200).save(img_dir / "000001.png")
    save_table(
        tmp_path / "meta.parquet",
        pd.DataFrame(
            [
                {"idx": 0, "path": "000000.jpg", "label": "x"},
                {"idx": 1, "path": "000001.png", "label": "y"},
            ]
        ),
    )
    save_table(
        tmp_path / "codes.parquet",
        pd.DataFrame({"idx": [0, 1], "c_0": [0, 0], "final_residual": [0.0, 0.0]}),
    )

    client = TestClient(create_app(str(tmp_path)))

    jpg = client.get("/thumb/0")
    assert jpg.headers["content-type"] == "image/jpeg"
    assert "no-cache" in jpg.headers.get("cache-control", "")

    png = client.get("/thumb/1")
    assert png.headers["content-type"] == "image/png"


def test_global_node_residual_is_internal_average(tmp_path):
    """Coloring relies on aggregate nodes carrying the mean residual of all
    items beneath them, and residual_norm being a normalized [0,1] value."""
    prepare_fashion_mnist(tmp_path, n=120, dataset=_fake_fmnist(120))
    runner.invoke(app, ["extract", "--data-dir", str(tmp_path), "--embedder", "dummy"])
    runner.invoke(
        app,
        ["fit-rq", "--data-dir", str(tmp_path), "--levels", "3",
         "--codebook-size", "8", "--iters", "10"],
    )
    runner.invoke(app, ["encode", "--data-dir", str(tmp_path)])

    codes = pd.read_parquet(tmp_path / "codes.parquet")
    expected_root_residual = float(codes["final_residual"].mean())

    body = TestClient(create_app(str(tmp_path))).get("/api/tree?level=3").json()
    root = next(n for n in body["nodes"] if n["prefix"] == [])

    # Most global node's residual == average residual over every item.
    assert root["mean_residual"] == pytest.approx(expected_root_residual)
    # residual_norm is a normalized color input for every node, root included.
    for n in body["nodes"]:
        assert 0.0 <= n["residual_norm"] <= 1.0
