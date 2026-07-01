from pathlib import Path
import pandas as pd

CUB_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"


def prepare_cub200(out_dir, n=5000, seed=0) -> Path:
    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    import tarfile
    import urllib.request
    import random

    tgz_path = out_dir / "CUB_200_2011.tgz"
    if not tgz_path.exists():
        print("downloading CUB-200-2011...")
        urllib.request.urlretrieve(CUB_URL, tgz_path)

    extract_dir = out_dir / "_cub_raw"
    if not extract_dir.exists():
        print("extracting...")
        with tarfile.open(tgz_path) as f:
            f.extractall(out_dir)
        (out_dir / "CUB_200_2011").rename(extract_dir)

    # Read labels
    labels = {}
    with open(extract_dir / "image_class_labels.txt") as f:
        for line in f:
            img_id, cls_id = line.strip().split()
            labels[int(img_id)] = int(cls_id) - 1

    # Read class names
    class_names = {}
    with open(extract_dir / "classes.txt") as f:
        for line in f:
            cls_id, name = line.strip().split(" ", 1)
            class_names[int(cls_id) - 1] = name.split(".")[-1]

    # Read image paths
    images = []
    with open(extract_dir / "images.txt") as f:
        for line in f:
            img_id, rel_path = line.strip().split()
            images.append((int(img_id), rel_path))

    # Use train_test_split if available, else all
    try:
        is_train = {}
        with open(extract_dir / "train_test_split.txt") as f:
            for line in f:
                img_id, flag = line.strip().split()
                is_train[int(img_id)] = int(flag)
    except FileNotFoundError:
        is_train = {img_id: 1 for img_id, _ in images}

    train_images = [
        (img_id, path) for img_id, path in images if is_train.get(img_id, 1)
    ]
    train_images.sort()
    random.seed(seed)
    train_images = train_images[:n]

    rows = []
    from PIL import Image

    for img_id, rel_path in train_images:
        src = extract_dir / "images" / rel_path
        dst_name = f"{img_id:06d}.jpg"
        img = Image.open(src).convert("RGB")
        img.save(img_dir / dst_name)
        label_idx = labels[img_id]
        rows.append(
            {"path": dst_name, "label": class_names.get(label_idx, str(label_idx))}
        )

    pd.DataFrame(rows, columns=["path", "label"]).to_csv(
        out_dir / "labels.csv", index=False
    )
    return img_dir
