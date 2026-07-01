from pathlib import Path

import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image

from gyo.embedders.base import list_images, l2_normalize, _load_labels


def _reparameterize(model):
    try:
        from timm.utils import reparameterize_model
    except Exception:  # pragma: no cover - fallback path
        from open_clip.model import reparameterize_model  # type: ignore
    return reparameterize_model(model)


class MobileCLIPEmbedder:
    def __init__(
        self,
        model_name="MobileCLIP2-S0",
        pretrained="dfndr2b",
        device="cpu",
        batch_size=64,
    ):
        self.device = device
        self.batch_size = batch_size
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        model.eval()  # MANDATORY before reparameterize (batchnorm)
        model = _reparameterize(model)  # MANDATORY for reparameterizable blocks
        self.model = model.to(device)
        self.preprocess = preprocess

    @torch.no_grad()
    def embed_folder(self, folder) -> tuple[np.ndarray, pd.DataFrame]:
        folder = Path(folder)
        files = list_images(folder)
        labels = _load_labels(folder)
        out = []
        for start in range(0, len(files), self.batch_size):
            batch = files[start : start + self.batch_size]
            tensors = [self.preprocess(Image.open(f).convert("RGB")) for f in batch]
            x = torch.stack(tensors).to(self.device)
            feats = self.model.encode_image(x).cpu().numpy()
            out.append(feats)
        emb = np.concatenate(out, 0) if out else np.zeros((0, 512), np.float32)
        emb = l2_normalize(emb)
        meta = pd.DataFrame(
            [
                {"idx": i, "path": f.name, "label": labels.get(f.name, "")}
                for i, f in enumerate(files)
            ],
            columns=["idx", "path", "label"],
        )
        return emb, meta
