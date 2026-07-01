import pandas as pd
from PIL import Image
from gyo.data.fashion_mnist import prepare_fashion_mnist, FASHION_CLASSES


def _fake_dataset(n=5):
    return [(Image.new("L", (28, 28), color=i * 5), i % 10) for i in range(n)]


def test_prepare_dumps_images_and_labels(tmp_path):
    img_dir = prepare_fashion_mnist(tmp_path, n=4, dataset=_fake_dataset(10))
    pngs = sorted(img_dir.glob("*.png"))
    assert len(pngs) == 4
    labels = pd.read_csv(tmp_path / "labels.csv")
    assert list(labels.columns) == ["path", "label"]
    assert len(labels) == 4
    assert labels.loc[0, "label"] in FASHION_CLASSES
    assert (img_dir / labels.loc[0, "path"]).exists()
