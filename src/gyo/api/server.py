from pathlib import Path
import hashlib
import json
import mimetypes
import os
import re
import tempfile
import threading
from functools import wraps

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

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
ATLAS_LAYOUT_VERSION = "metric-mds-v1:euclidean"
MAX_ATLAS_SIBLINGS = 256


def _dataset_files(data_dir):
    root = Path(data_dir)
    codebooks = root / "codebooks" / "v1"
    return [
        root / "codes.parquet",
        root / "meta.parquet",
        root / "embeddings.npy",
        *sorted(codebooks.glob("*.npy")),
        codebooks / "config.json",
    ]


def _dataset_fingerprint(data_dir):
    """Hash all geometry-bearing inputs and the layout algorithm contract."""
    digest = hashlib.sha256(ATLAS_LAYOUT_VERSION.encode())
    for path in _dataset_files(data_dir):
        digest.update(str(path.relative_to(data_dir)).encode())
        try:
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
        except FileNotFoundError:
            digest.update(b"<missing>")
        except OSError as exc:
            raise HTTPException(409, f"unable to read Atlas input: {path.name}") from exc
    return digest.hexdigest()


def _level_columns(columns):
    levels = []
    for column in columns:
        match = re.fullmatch(r"c_(\d+)", str(column))
        if match:
            levels.append((int(match.group(1)), column))
    levels.sort()
    if [level for level, _ in levels] != list(range(len(levels))):
        raise ValueError("Atlas level columns must be contiguous from c_0")
    return [column for _, column in levels]


def _validated_codes(values):
    raw = np.asarray(values)
    if raw.ndim != 2 or raw.dtype.kind not in "iuf":
        raise ValueError("Atlas codes must be a real numeric 2D array")
    if not np.isfinite(raw).all():
        raise ValueError("Atlas codes must be finite")
    if not np.equal(raw, np.floor(raw)).all():
        raise ValueError("Atlas codes must contain integral values")
    if np.any(raw < -(2**63)) or np.any(raw >= 2**63):
        raise ValueError("Atlas codes must fit in int64")
    return raw.astype(np.int64)


def _validated_final_residual(values, num_rows):
    raw = np.asarray(values)
    if (
        raw.ndim != 1
        or raw.shape[0] != num_rows
        or raw.dtype.kind not in "iuf"
    ):
        raise ValueError("Atlas final_residual must be numeric, 1D, and match codes rows")
    if not np.isfinite(raw).all():
        raise ValueError("Atlas final_residual must be finite")
    result = raw.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Atlas final_residual must be finite float32 values")
    return result


def _config_int(config, key):
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Atlas config {key} must be a positive integer")
    return value


def create_app(data_dir: str) -> FastAPI:
    data_dir = Path(data_dir)
    app = FastAPI(title="gyo")
    data_cache = None
    identity_cache = None
    atlas_cache = None
    dataset_cache = None
    fingerprint_stats = None
    geometry_cache = {}
    active_dataset_id = None
    cache_lock = threading.RLock()

    def synchronized(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            with cache_lock:
                return function(*args, **kwargs)
        return wrapper

    def _fingerprint():
        nonlocal dataset_cache, fingerprint_stats
        files = _dataset_files(data_dir)
        try:
            stats = tuple(
                (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                for path in files
                for stat in (path.stat(),)
            )
        except OSError:
            return _dataset_fingerprint(data_dir)
        if fingerprint_stats != stats:
            dataset_cache = _dataset_fingerprint(data_dir)
            fingerprint_stats = stats
        return dataset_cache

    def _geometry_for(dataset_id):
        nonlocal geometry_cache
        if geometry_cache.get("dataset_id") == dataset_id:
            return geometry_cache.setdefault("geometries", {})
        path = data_dir / "atlas" / "v1" / "geometry.json"
        try:
            loaded = json.loads(path.read_text())
            if (
                loaded.get("dataset_id") == dataset_id
                and loaded.get("version") == 1
                and isinstance(loaded.get("geometries"), dict)
            ):
                geometry_cache = loaded
            else:
                geometry_cache = {"dataset_id": dataset_id, "version": 1, "geometries": {}}
        except (OSError, UnicodeError, json.JSONDecodeError, AttributeError):
            geometry_cache = {"dataset_id": dataset_id, "version": 1, "geometries": {}}
        return geometry_cache["geometries"]

    def _persist_geometry():
        nonlocal geometry_cache
        path = data_dir / "atlas" / "v1" / "geometry.json"
        lock_path = path.with_suffix(".lock")
        temporary = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open("a+") as lock_stream:
                if fcntl is not None:
                    fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
                try:
                    try:
                        current = json.loads(path.read_text())
                    except (OSError, UnicodeError, json.JSONDecodeError):
                        current = {}
                    if (
                        current.get("dataset_id") == geometry_cache["dataset_id"]
                        and current.get("version") == 1
                        and isinstance(current.get("geometries"), dict)
                    ):
                        merged = dict(current["geometries"])
                        merged.update(geometry_cache["geometries"])
                        geometry_cache["geometries"] = merged
                    with tempfile.NamedTemporaryFile(
                        "w", dir=path.parent, prefix=".geometry-",
                        suffix=".tmp", delete=False,
                    ) as stream:
                        temporary = Path(stream.name)
                        json.dump(geometry_cache, stream, separators=(",", ":"), allow_nan=False)
                        stream.flush()
                        os.fsync(stream.fileno())
                    temporary.replace(path)
                    try:
                        directory_fd = os.open(path.parent, os.O_RDONLY)
                        try:
                            os.fsync(directory_fd)
                        finally:
                            os.close(directory_fd)
                    except OSError:
                        pass
                finally:
                    if fcntl is not None:
                        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
        except (OSError, TypeError, ValueError):
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    def _ensure_dataset():
        nonlocal data_cache, identity_cache, atlas_cache, geometry_cache
        nonlocal active_dataset_id
        with cache_lock:
            dataset_id = _fingerprint()
            if active_dataset_id != dataset_id:
                data_cache = identity_cache = atlas_cache = None
                geometry_cache = {}
                active_dataset_id = dataset_id
            return dataset_id

    def _load():
        nonlocal data_cache, identity_cache
        if data_cache is not None:
            return data_cache
        codes_df = load_table(data_dir / "codes.parquet")
        meta_df = load_table(data_dir / "meta.parquet")
        level_cols = _level_columns(codes_df.columns)
        if not level_cols:
            raise ValueError("Atlas codes require contiguous level columns from c_0")
        codes = _validated_codes(codes_df[level_cols].to_numpy())
        if "final_residual" not in codes_df:
            raise ValueError("Atlas final_residual column is required")
        final_res = _validated_final_residual(
            codes_df["final_residual"].to_numpy(), len(codes)
        )
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
        identity_cache = (tuple(item_ids), tuple(meta_ids), meta_by_id)
        data_cache = (codes, final_res, labels, meta_df)
        return data_cache

    def _load_atlas():
        nonlocal atlas_cache
        if atlas_cache is not None:
            return atlas_cache

        try:
            codes, final_res, labels, _ = _load()
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(409, f"invalid Atlas table input: {exc}") from exc
        embeddings_path = data_dir / "embeddings.npy"
        config_path = data_dir / "codebooks" / "v1" / "config.json"
        if not embeddings_path.exists():
            raise HTTPException(409, "required Atlas input missing: embeddings.npy")
        if not config_path.exists():
            raise HTTPException(409, "required Atlas input missing: codebooks/v1/config.json")

        try:
            config = json.loads(config_path.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise HTTPException(409, f"invalid Atlas config: {exc}") from exc
        if not isinstance(config, dict):
            raise HTTPException(409, "invalid Atlas config: expected a JSON object")
        try:
            num_levels = _config_int(config, "num_levels")
            codebook_size = _config_int(config, "codebook_size")
            dim = _config_int(config, "dim")
        except ValueError as exc:
            raise HTTPException(409, f"invalid Atlas config: {exc}") from exc
        for level in range(num_levels):
            path = config_path.parent / f"level_{level}.npy"
            if not path.exists():
                raise HTTPException(409, f"required Atlas input missing: {path.name}")
        rq = ResidualQuantizer(
            num_levels=num_levels,
            codebook_size=codebook_size,
            dim=dim,
            proj_dim=config.get("proj_dim"),
            seed=config.get("seed", 0),
        )
        rq.codebooks = []
        for level in range(num_levels):
            path = config_path.parent / f"level_{level}.npy"
            try:
                rq.codebooks.append(np.load(path, allow_pickle=False))
            except (OSError, ValueError) as exc:
                raise HTTPException(409, f"invalid Atlas input {path.name}: {exc}") from exc
        try:
            embeddings = np.load(embeddings_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise HTTPException(
                409, f"invalid Atlas input embeddings.npy: {exc}"
            ) from exc
        item_ids, meta_ids, meta_by_id = identity_cache

        if (
            codes.ndim != 2
            or embeddings.ndim != 2
            or embeddings.dtype.kind not in "iuf"
        ):
            raise HTTPException(
                409, "Atlas codes and embeddings must be real numeric 2D arrays"
            )
        if codes.shape[0] != embeddings.shape[0]:
            raise HTTPException(409, "Atlas codes and embeddings rows must match")
        if codes.shape[1] != rq.num_levels:
            raise HTTPException(409, "Atlas codes levels must match the quantizer")
        if len(rq.codebooks) != rq.num_levels:
            raise HTTPException(409, "Atlas codebook levels must match the quantizer")
        if len(set(item_ids)) != len(item_ids):
            raise HTTPException(409, "duplicate codes item ids")
        if len(set(meta_ids)) != len(meta_ids):
            raise HTTPException(409, "duplicate metadata item ids")
        missing_ids = [item_id for item_id in item_ids if item_id not in meta_by_id]
        if missing_ids:
            missing = ", ".join(str(item_id) for item_id in missing_ids[:20])
            raise HTTPException(409, f"metadata missing for item ids: {missing}")
        for level, codebook in enumerate(rq.codebooks):
            if codebook.ndim != 2 or codebook.dtype.kind not in "iuf":
                raise HTTPException(
                    409, f"Atlas codebook level {level} must be a real numeric 2D array"
                )
            if not np.isfinite(codebook).all():
                raise HTTPException(409, f"Atlas codebook level {level} must be finite")
            if codebook.shape[0] != rq.codebook_size:
                raise HTTPException(409, f"Atlas codebook level {level} size mismatch")
            if codebook.shape[1] != embeddings.shape[1] or (
                rq.dim is not None and codebook.shape[1] != rq.dim
            ):
                raise HTTPException(409, "Atlas embedding and codebook dimension mismatch")
            if np.any(codes[:, level] < 0) or np.any(
                codes[:, level] >= rq.codebook_size
            ):
                raise HTTPException(409, f"Atlas code index out of range at level {level}")
        if not np.isfinite(embeddings).all():
            raise HTTPException(409, "Atlas embeddings must be finite")
        root = build_tree(codes, final_res, labels)
        stats_by_prefix = {stat.prefix: stat for stat in node_stats(root, labels)}
        atlas_cache = (
            root,
            labels,
            embeddings,
            rq,
            item_ids,
            meta_by_id,
            stats_by_prefix,
            {},
            _fingerprint(),
        )
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
    @synchronized
    def tree(level: int = 99):
        dataset_id = _ensure_dataset()
        codes, final_res, labels, _ = _load()
        cfg_path = data_dir / "codebooks" / "v1" / "config.json"
        if cfg_path.exists():
            codebook_size = json.loads(cfg_path.read_text())["codebook_size"]
        else:
            codebook_size = int(codes.max()) + 1
        root = build_tree(codes, final_res, labels)
        stats = [s for s in node_stats(root, labels) if s.level <= level]
        return {
            "dataset_id": dataset_id,
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
    @synchronized
    def node(prefix: str):
        dataset_id = _ensure_dataset()
        codes, final_res, labels, _ = _load()
        root = build_tree(codes, final_res, labels)
        pfx = () if prefix == "root" else tuple(int(p) for p in prefix.split(","))
        target = node_at(root, pfx)
        if target is None:
            raise HTTPException(404, "prefix not found")
        item_ids, _, meta_by_id = identity_cache
        items = []
        for position in target.item_indices[:200]:
            item_id = item_ids[position]
            row = meta_by_id.get(item_id)
            if row is None:
                continue
            label = (
                None
                if "label" not in row or row[["label"]].isna().iloc[0]
                else str(row["label"])
            )
            items.append({"idx": item_id, "path": str(row["path"]), "label": label})
        return {"dataset_id": dataset_id, "items": items, "occupancy": target.occupancy}

    @app.get("/api/node/{prefix}/metrics")
    @synchronized
    def node_metrics(prefix: str):
        dataset_id = _ensure_dataset()
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
            "dataset_id": dataset_id,
            "prefix": list(pfx),
            "level": node_stat.level,
            "occupancy": node_stat.occupancy,
            "mean_residual": node_stat.mean_residual,
            "residual_norm": node_stat.residual_norm,
            "size_norm": node_stat.size_norm,
            "purity": node_stat.purity,
            "label_distribution": label_dist,
        }

    @app.get("/api/dataset")
    @synchronized
    def dataset():
        return {"dataset_id": _ensure_dataset()}

    @app.get("/api/atlas/{prefix}")
    @synchronized
    def atlas(prefix: str):
        dataset_id = _ensure_dataset()
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

        (
            root,
            labels,
            embeddings,
            rq,
            item_ids,
            meta_by_id,
            stats_by_prefix,
            samples_by_prefix,
            _cached_dataset_id,
        ) = _load_atlas()
        codebooks = rq.codebooks
        target = node_at(root, pfx)
        if target is None:
            raise HTTPException(404, "prefix not found")

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
            samples = samples_by_prefix.get(tree_node.prefix)
            if samples is None:
                samples = ranked_samples(embeddings, tree_node.item_indices, center)
                samples_by_prefix[tree_node.prefix] = samples
            labeled = [
                labels[position]
                for position in tree_node.item_indices
                if labels[position] is not None
            ]
            purity = None
            if labeled:
                purity = Counter(labeled).most_common(1)[0][1] / len(labeled)
            return {
                "prefix": list(stat.prefix),
                "level": stat.level,
                "occupancy": stat.occupancy,
                "mean_residual": stat.mean_residual,
                "purity": purity,
                "residual_norm": stat.residual_norm,
                "samples": {
                    "representative": [item(index) for index in samples.representative],
                    "outliers": [item(index) for index in samples.outliers],
                },
            }

        child_nodes = list(target.children.values())
        if len(child_nodes) > MAX_ATLAS_SIBLINGS:
            raise HTTPException(
                409,
                f"Atlas focus has {len(child_nodes)} siblings; limit is "
                f"{MAX_ATLAS_SIBLINGS}; this dataset exceeds the supported per-focus maximum. "
                "Train with a smaller codebook (or retrain with lower per-level cardinality).",
            )
        child_prefixes = [child.prefix for child in child_nodes]
        distances = sibling_distance_matrix(child_prefixes, codebooks)
        geometries = _geometry_for(dataset_id)
        geometry_key = "root" if not pfx else ",".join(map(str, pfx))
        saved = geometries.get(geometry_key)
        try:
            positions = np.asarray(saved["positions"], dtype=np.float64)
            raw_stress = float(saved["raw_stress"])
            if (
                positions.shape != (len(child_nodes), 2)
                or not np.isfinite(positions).all()
                or not np.isfinite(raw_stress)
                or raw_stress < 0
                or np.any(positions < -1.0)
                or np.any(positions > 1.0)
                or saved.get("child_prefixes") != [list(prefix) for prefix in child_prefixes]
            ):
                raise ValueError
        except (TypeError, KeyError, ValueError, OverflowError):
            projection = metric_mds(distances)
            positions, raw_stress = projection.positions, projection.stress
            geometries[geometry_key] = {"positions": positions.tolist(), "raw_stress": raw_stress, "child_prefixes": [list(prefix) for prefix in child_prefixes]}
            _persist_geometry()
        parent_vector = prefix_vector(pfx, codebooks)
        children = []
        for child, position in zip(child_nodes, positions):
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
            "dataset_id": dataset_id,
            "num_levels": rq.num_levels,
            "focus": node_payload(target),
            "children": children,
            "projection": {
                "method": "metric-mds",
                "metric": "euclidean",
                "raw_stress": raw_stress,
                "stress": raw_stress,
                "distances": distances.tolist(),
            },
        }

    @app.get("/thumb/{idx}")
    @synchronized
    def thumb(idx: int):
        dataset_id = _ensure_dataset()
        _load()
        _, _, meta_by_id = identity_cache
        row = meta_by_id.get(idx)
        if row is None:
            raise HTTPException(404, "idx out of range")
        images_root = (data_dir / "images").resolve()
        path = (images_root / str(row["path"])).resolve()
        if not path.is_relative_to(images_root):
            raise HTTPException(404, "image path outside images directory")
        if not path.exists():
            raise HTTPException(404, "image missing")
        media_type, _ = mimetypes.guess_type(path.name)
        return FileResponse(
            path,
            media_type=media_type or "application/octet-stream",
            # Image bytes are intentionally outside dataset identity; thumbnails
            # are always revalidated instead of being retained by the browser.
            headers={"Cache-Control": "no-cache", "X-Dataset-ID": dataset_id},
        )

    return app
