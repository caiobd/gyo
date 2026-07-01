---
name: gyo-pipeline
description: Use when setting up gyo from scratch — preparing data, extracting real embeddings with MobileCLIP, training RQ codebooks, encoding, and serving.
---

# gyo Pipeline

## Overview

Run the full gyo pipeline end-to-end: download Fashion-MNIST → extract MobileCLIP embeddings → fit RQ codebooks → encode → serve.

**Assumes:** repo cloned, Python 3.12 pinned, `uv sync` done.

## Steps

Execute in order:

```bash
# 1. Prepare Fashion-MNIST images (2000)
uv run gyo prep-data --n 2000

# 2. Extract real embeddings with MobileCLIP (512-dim)
uv run gyo extract --embedder mobileclip

# 3. Train RQ codebooks (5 levels, 32 codewords each)
uv run gyo fit-rq --levels 5 --codebook-size 32

# 4. Encode embeddings into codes
uv run gyo encode

# 5. Launch server
uv run gyo serve --port 8001
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `no module 'gyo'` | Run from repo root or `uv run` |
| Port 8001 busy | Use `--port 8002` or kill old process: `lsof -ti:8001 \| xargs kill` |
| MobileCLIP OOM | Add `--batch-size 32` or use CPU: add `--device cpu` to `extract` |

## Commonly Useful

```bash
# Clean and restart (removes old codes/codebooks)
rm -rf run/codebooks run/codes.parquet
uv run gyo extract --embedder mobileclip
uv run gyo fit-rq --levels 5 --codebook-size 32
uv run gyo encode
uv run gyo serve --port 8001
```
