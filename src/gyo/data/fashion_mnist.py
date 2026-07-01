from pathlib import Path
import pandas as pd

FASHION_CLASSES = [
    "T-shirt_top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle_boot",
]


def _default_dataset(out_dir: Path, split: str):
    from torchvision.datasets import FashionMNIST

    train = split == "train"
    return FashionMNIST(root=str(out_dir / "_torchvision"), train=train, download=True)


def prepare_fashion_mnist(
    out_dir, n=10000, split="train", seed=0, dataset=None
) -> Path:
    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    if dataset is None:
        dataset = _default_dataset(out_dir, split)
    rows = []
    for i in range(min(n, len(dataset))):
        img, label_int = dataset[i]
        fname = f"{i:06d}.png"
        img.convert("L").save(img_dir / fname)
        rows.append({"path": fname, "label": FASHION_CLASSES[label_int]})
    pd.DataFrame(rows, columns=["path", "label"]).to_csv(
        out_dir / "labels.csv", index=False
    )
    return img_dir
