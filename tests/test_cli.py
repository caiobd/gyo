import numpy as np
import pandas as pd
from PIL import Image
from typer.testing import CliRunner
from gyo.cli import app

runner = CliRunner()


def _seed_images(data_dir, n=6):
    img_dir = data_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("L", (28, 28), color=i * 9).save(img_dir / f"{i:06d}.png")
    pd.DataFrame(
        {
            "path": [f"{i:06d}.png" for i in range(n)],
            "label": ["A" if i % 2 else "B" for i in range(n)],
        }
    ).to_csv(data_dir / "labels.csv", index=False)


def test_extract_fit_encode_pipeline(tmp_path):
    _seed_images(tmp_path, 6)
    r1 = runner.invoke(
        app, ["extract", "--data-dir", str(tmp_path), "--embedder", "dummy"]
    )
    assert r1.exit_code == 0, r1.output
    assert (tmp_path / "embeddings.npy").exists()
    assert (tmp_path / "meta.parquet").exists()

    r2 = runner.invoke(
        app,
        [
            "fit-rq",
            "--data-dir",
            str(tmp_path),
            "--levels",
            "2",
            "--codebook-size",
            "3",
        ],
    )
    assert r2.exit_code == 0, r2.output
    assert (tmp_path / "codebooks" / "v1" / "config.json").exists()

    r3 = runner.invoke(app, ["encode", "--data-dir", str(tmp_path)])
    assert r3.exit_code == 0, r3.output
    codes = pd.read_parquet(tmp_path / "codes.parquet")
    assert {"idx", "c_0", "c_1", "j", "final_residual"} <= set(codes.columns)
    assert len(codes) == 6
