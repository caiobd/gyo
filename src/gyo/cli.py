from pathlib import Path
import pandas as pd
import typer

from gyo.io.store import save_embeddings, load_embeddings, save_table, load_table
from gyo.rq.quantizer import ResidualQuantizer

app = typer.Typer(help="RQ embedding-space inspector")


def _embedder(name: str):
    if name == "dummy":
        from gyo.embedders.base import DummyEmbedder

        return DummyEmbedder()
    from gyo.embedders.mobileclip import MobileCLIPEmbedder

    return MobileCLIPEmbedder()


@app.command("prep-data")
def prep_data(
    data_dir: str = "run",
    dataset: str = "fashion_mnist",
    n: int = 10000,
    split: str = "train",
):
    if dataset == "fashion_mnist":
        from gyo.data.fashion_mnist import prepare_fashion_mnist

        prepare_fashion_mnist(data_dir, n=n, split=split)
    elif dataset == "cub200":
        from gyo.data.cub200 import prepare_cub200

        prepare_cub200(data_dir, n=n)
    elif dataset == "flowers102":
        from gyo.data.flowers102 import prepare_flowers102

        prepare_flowers102(data_dir, n=n)
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    typer.echo(f"prepared {dataset} ({n} images) in {data_dir}/images")


@app.command()
def extract(data_dir: str = "run", embedder: str = "mobileclip"):
    emb, meta = _embedder(embedder).embed_folder(Path(data_dir) / "images")
    save_embeddings(Path(data_dir) / "embeddings.npy", emb)
    save_table(Path(data_dir) / "meta.parquet", meta)
    typer.echo(f"extracted {emb.shape[0]} embeddings dim={emb.shape[1]}")


@app.command("fit-rq")
def fit_rq(
    data_dir: str = "run",
    levels: int = 3,
    codebook_size: int = 256,
    iters: int = 10,
    seed: int = 0,
):
    emb = load_embeddings(Path(data_dir) / "embeddings.npy")
    rq = ResidualQuantizer(num_levels=levels, codebook_size=codebook_size, seed=seed)
    rq.fit(emb, iters=iters)
    rq.save(Path(data_dir) / "codebooks" / "v1")
    typer.echo(f"fit RQ L={levels} K={codebook_size} on {emb.shape[0]} points")


@app.command()
def encode(data_dir: str = "run"):
    emb = load_embeddings(Path(data_dir) / "embeddings.npy")
    rq = ResidualQuantizer.load(Path(data_dir) / "codebooks" / "v1")
    res = rq.encode(emb)
    cols = {"idx": range(len(emb))}
    for lvl in range(rq.num_levels):
        cols[f"c_{lvl}"] = res.codes[:, lvl]
    cols["j"] = res.tie_index
    for lvl in range(rq.num_levels):
        cols[f"r_{lvl}"] = res.residuals[:, lvl]
    cols["final_residual"] = res.final_residual
    save_table(Path(data_dir) / "codes.parquet", pd.DataFrame(cols))
    typer.echo(f"encoded {len(emb)} items")


@app.command()
def serve(data_dir: str = "run", port: int = 8000):
    import uvicorn
    from gyo.api.server import create_app

    uvicorn.run(create_app(data_dir), host="127.0.0.1", port=port)


if __name__ == "__main__":
    app()
