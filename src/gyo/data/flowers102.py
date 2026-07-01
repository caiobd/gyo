from pathlib import Path
import pandas as pd


def prepare_flowers102(out_dir, n=5000, seed=0) -> Path:
    from torchvision.datasets import Flowers102
    from PIL import Image
    import random

    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    ds = Flowers102(root=str(out_dir / "_torchvision"), split="train", download=True)
    ds_test = Flowers102(
        root=str(out_dir / "_torchvision"), split="test", download=True
    )

    # Collect all images and labels
    all_items = []
    for i in range(len(ds)):
        img, label = ds[i]
        all_items.append((img, ds.classes[label]))
    for i in range(len(ds_test)):
        img, label = ds_test[i]
        all_items.append((img, ds_test.classes[label]))

    indices = list(range(len(all_items)))
    random.seed(seed)
    random.shuffle(indices)
    indices = indices[:n]

    rows = []
    for idx, i in enumerate(indices):
        img, label = all_items[i]
        fname = f"{idx:06d}.jpg"
        img.save(img_dir / fname)
        rows.append({"path": fname, "label": label})

    pd.DataFrame(rows, columns=["path", "label"]).to_csv(
        out_dir / "labels.csv", index=False
    )
    return img_dir
