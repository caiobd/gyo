import pandas as pd
from PIL import Image
from typer.testing import CliRunner
from fastapi.testclient import TestClient
from gyo.cli import app
from gyo.data.fashion_mnist import prepare_fashion_mnist
from gyo.api.server import create_app

runner = CliRunner()


def _fake(n=120):
    return [(Image.new("L", (28, 28), color=(i * 7) % 255), i % 10) for i in range(n)]


def test_api_node_returns_items_for_leaf_and_internal(tmp_path):
    prepare_fashion_mnist(tmp_path, n=120, dataset=_fake(120))
    runner.invoke(app, ["extract", "--data-dir", str(tmp_path), "--embedder", "dummy"])
    runner.invoke(app, ["fit-rq", "--data-dir", str(tmp_path), "--levels", "3",
                        "--codebook-size", "8", "--iters", "10"])
    runner.invoke(app, ["encode", "--data-dir", str(tmp_path)])
    client = TestClient(create_app(str(tmp_path)))

    tree = client.get("/api/tree?level=3").json()
    internal = next(n for n in tree["nodes"] if len(n["prefix"]) == 1)
    pfx = ",".join(str(c) for c in internal["prefix"])

    body = client.get(f"/api/node/{pfx}").json()
    assert body["occupancy"] == internal["occupancy"]
    assert len(body["items"]) == min(200, internal["occupancy"])
    assert set(body["items"][0]) == {"idx", "path", "label"}

    root = client.get("/api/node/root").json()
    assert root["occupancy"] == 120


def test_js_module_route(tmp_path, monkeypatch):
    (tmp_path / "codes.parquet")  # not needed; route is static
    import gyo.api.server as srv
    js_dir = srv.WEB / "js"
    js_dir.mkdir(parents=True, exist_ok=True)
    (js_dir / "ping.js").write_text("export const ping = 1;\n")
    client = TestClient(create_app(str(tmp_path)))
    r = client.get("/js/ping.js")
    assert r.status_code == 200
    assert "javascript" in r.headers["content-type"]
    assert "export const ping" in r.text
