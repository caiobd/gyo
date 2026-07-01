from pathlib import Path
import json
import mimetypes

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse

from collections import Counter

from gyo.io.store import load_table
from gyo.tree.build import build_tree, node_at
from gyo.tree.signals import node_stats, dead_codeword_counts

WEB = Path(__file__).resolve().parent.parent / "web"


def create_app(data_dir: str) -> FastAPI:
    data_dir = Path(data_dir)
    app = FastAPI(title="gyo")

    def _load():
        codes_df = load_table(data_dir / "codes.parquet")
        meta_df = load_table(data_dir / "meta.parquet")
        level_cols = [c for c in codes_df.columns if c.startswith("c_")]
        codes = codes_df[sorted(level_cols)].to_numpy(np.int64)
        final_res = codes_df["final_residual"].to_numpy(np.float32)
        labels = meta_df["label"].astype(str).tolist()
        return codes, final_res, labels, meta_df

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (WEB / "index.html").read_text()

    @app.get("/style.css")
    def style():
        return FileResponse(WEB / "style.css", media_type="text/css")

    @app.get("/js/{name}")
    def js_module(name: str):
        path = WEB / "js" / name
        if ".." in name or not path.exists():
            raise HTTPException(404, "module not found")
        return FileResponse(
            path,
            media_type="application/javascript",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/api/tree")
    def tree(level: int = 99):
        codes, final_res, labels, _ = _load()
        cfg_path = data_dir / "codebooks" / "v1" / "config.json"
        if cfg_path.exists():
            codebook_size = json.loads(cfg_path.read_text())["codebook_size"]
        else:
            codebook_size = int(codes.max()) + 1
        root = build_tree(codes, final_res, labels)
        stats = [s for s in node_stats(root, labels) if s.level <= level]
        return {
            "num_levels": codes.shape[1],
            "dead_codewords": dead_codeword_counts(codes, codebook_size),
            "nodes": [
                {
                    "prefix": list(s.prefix),
                    "level": s.level,
                    "occupancy": s.occupancy,
                    "mean_residual": s.mean_residual,
                    "size_norm": s.size_norm,
                    "residual_norm": s.residual_norm,
                    "purity": s.purity,
                }
                for s in stats
            ],
        }

    @app.get("/api/node/{prefix}")
    def node(prefix: str):
        codes, final_res, labels, meta_df = _load()
        root = build_tree(codes, final_res, labels)
        pfx = () if prefix == "root" else tuple(int(p) for p in prefix.split(","))
        target = node_at(root, pfx)
        if target is None:
            raise HTTPException(404, "prefix not found")
        items = [
            {
                "idx": int(i),
                "path": str(meta_df.loc[i, "path"]),
                "label": str(meta_df.loc[i, "label"]),
            }
            for i in target.item_indices[:200]
        ]
        return {"items": items, "occupancy": target.occupancy}

    @app.get("/api/node/{prefix}/metrics")
    def node_metrics(prefix: str):
        codes, final_res, labels, meta_df = _load()
        root = build_tree(codes, final_res, labels)
        pfx = () if prefix == "root" else tuple(int(p) for p in prefix.split(","))
        target = node_at(root, pfx)
        if target is None:
            raise HTTPException(404, "prefix not found")
        all_stats = node_stats(root, labels)
        node_stat = next((s for s in all_stats if s.prefix == pfx), None)
        if node_stat is None:
            raise HTTPException(404, "node stats not found")
        label_dist = {}
        if labels:
            counts = Counter(labels[i] for i in target.item_indices)
            label_dist = dict(counts.most_common(20))
        return {
            "prefix": list(pfx),
            "level": node_stat.level,
            "occupancy": node_stat.occupancy,
            "mean_residual": node_stat.mean_residual,
            "residual_norm": node_stat.residual_norm,
            "size_norm": node_stat.size_norm,
            "purity": node_stat.purity,
            "label_distribution": label_dist,
        }

    @app.get("/thumb/{idx}")
    def thumb(idx: int):
        _, _, _, meta_df = _load()
        if idx < 0 or idx >= len(meta_df):
            raise HTTPException(404, "idx out of range")
        path = data_dir / "images" / str(meta_df.loc[idx, "path"])
        if not path.exists():
            raise HTTPException(404, "image missing")
        media_type, _ = mimetypes.guess_type(path.name)
        return FileResponse(
            path,
            media_type=media_type or "application/octet-stream",
            headers={"Cache-Control": "no-cache"},
        )

    return app
