from pathlib import Path
import json
import mimetypes

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse

from collections import Counter

from gyo.atlas import (
    metric_mds,
    prefix_vector,
    ranked_samples,
    sibling_distance_matrix,
)
from gyo.io.store import load_table
from gyo.rq.quantizer import ResidualQuantizer
from gyo.tree.build import build_tree, node_at
from gyo.tree.signals import node_stats, dead_codeword_counts

WEB = Path(__file__).resolve().parent.parent / "web"


def create_app(data_dir: str) -> FastAPI:
    data_dir = Path(data_dir)
    app = FastAPI(title="gyo")
    data_cache = None
    identity_cache = None
    atlas_cache = None

    def _load():
        nonlocal data_cache, identity_cache
        if data_cache is not None:
            return data_cache
        codes_df = load_table(data_dir / "codes.parquet")
        meta_df = load_table(data_dir / "meta.parquet")
        level_cols = [c for c in codes_df.columns if c.startswith("c_")]
        codes = codes_df[sorted(level_cols)].to_numpy(np.int64)
        final_res = codes_df["final_residual"].to_numpy(np.float32)
        item_ids = (
            [int(value) for value in codes_df["idx"]]
            if "idx" in codes_df
            else list(range(len(codes_df)))
        )
        meta_ids = (
            [int(value) for value in meta_df["idx"]]
            if "idx" in meta_df
            else list(range(len(meta_df)))
        )
        meta_by_id = {
            item_id: meta_df.iloc[position]
            for position, item_id in enumerate(meta_ids)
        }
        labels = []
        for item_id in item_ids:
            row = meta_by_id.get(item_id)
            if (
                row is None
                or "label" not in row
                or bool(row[["label"]].isna().iloc[0])
            ):
                labels.append(None)
            else:
                labels.append(str(row["label"]))
        identity_cache = (tuple(item_ids), meta_by_id)
        data_cache = (codes, final_res, labels, meta_df)
        return data_cache

    def _load_atlas():
        nonlocal atlas_cache
        if atlas_cache is not None:
            return atlas_cache

        codes, final_res, labels, _ = _load()
        embeddings_path = data_dir / "embeddings.npy"
        config_path = data_dir / "codebooks" / "v1" / "config.json"
        if not embeddings_path.exists():
            raise HTTPException(409, "required Atlas input missing: embeddings.npy")
        if not config_path.exists():
            raise HTTPException(409, "required Atlas input missing: codebooks/v1/config.json")

        config = json.loads(config_path.read_text())
        num_levels = int(config.get("num_levels", codes.shape[1]))
        for level in range(num_levels):
            path = config_path.parent / f"level_{level}.npy"
            if not path.exists():
                raise HTTPException(409, f"required Atlas input missing: {path.name}")
        rq = ResidualQuantizer.load(config_path.parent)
        embeddings = np.load(embeddings_path)
        item_ids, meta_by_id = identity_cache
        missing_ids = [item_id for item_id in item_ids if item_id not in meta_by_id]
        if missing_ids:
            missing = ", ".join(str(item_id) for item_id in missing_ids[:20])
            raise HTTPException(409, f"metadata missing for item ids: {missing}")
        root = build_tree(codes, final_res, labels)
        atlas_cache = (root, labels, embeddings, rq, item_ids, meta_by_id)
        return atlas_cache

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

    @app.get("/api/atlas/{prefix}")
    def atlas(prefix: str):
        if prefix == "root":
            pfx = ()
        else:
            try:
                parts = prefix.split(",")
                if not parts or any(part == "" for part in parts):
                    raise ValueError
                pfx = tuple(int(part) for part in parts)
            except ValueError as exc:
                raise HTTPException(400, "prefix must be 'root' or CSV integers") from exc

        root, labels, embeddings, rq, item_ids, meta_by_id = _load_atlas()
        codebooks = rq.codebooks
        target = node_at(root, pfx)
        if target is None:
            raise HTTPException(404, "prefix not found")

        stats_by_prefix = {stat.prefix: stat for stat in node_stats(root, labels)}

        def item(position):
            item_id = item_ids[position]
            row = meta_by_id[item_id]
            label = labels[position]
            return {
                "idx": item_id,
                "path": str(row["path"]),
                "label": label,
            }

        def node_payload(tree_node):
            stat = stats_by_prefix[tree_node.prefix]
            center = prefix_vector(tree_node.prefix, codebooks)
            samples = ranked_samples(embeddings, tree_node.item_indices, center)
            return {
                "prefix": list(stat.prefix),
                "level": stat.level,
                "occupancy": stat.occupancy,
                "mean_residual": stat.mean_residual,
                "purity": stat.purity,
                "residual_norm": stat.residual_norm,
                "samples": {
                    "representative": [item(index) for index in samples.representative],
                    "outliers": [item(index) for index in samples.outliers],
                },
            }

        child_nodes = list(target.children.values())
        child_prefixes = [child.prefix for child in child_nodes]
        distances = sibling_distance_matrix(child_prefixes, codebooks)
        projection = metric_mds(distances)
        parent_vector = prefix_vector(pfx, codebooks)
        children = []
        for child, position in zip(child_nodes, projection.positions):
            child_vector = prefix_vector(child.prefix, codebooks)
            token = codebooks[child.level - 1][child.prefix[-1]]
            payload = node_payload(child)
            payload.update(
                {
                    "position": position.tolist(),
                    "parent_distance": float(np.linalg.norm(child_vector - parent_vector)),
                    "token_norm": float(np.linalg.norm(token)),
                    "has_children": bool(child.children),
                }
            )
            children.append(payload)

        return {
            "focus": node_payload(target),
            "children": children,
            "projection": {
                "method": "metric-mds",
                "metric": "euclidean",
                "stress": projection.stress,
                "warning": projection.stress > 0.10,
            },
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
