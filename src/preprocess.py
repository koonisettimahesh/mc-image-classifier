import os
import random
from torchvision.datasets import Caltech101

RAW_DIR = "data_raw"
OUT_DIR = "data"
NUM_CLASSES = 10
TRAIN_SPLIT = 0.8

dataset = Caltech101(root=RAW_DIR, download=True)

selected_classes = dataset.categories[:NUM_CLASSES]

for split in ["train", "val"]:
    for cls in selected_classes:
        os.makedirs(os.path.join(OUT_DIR, split, cls), exist_ok=True)

for idx, (img, label) in enumerate(dataset):
    cls_name = dataset.categories[label]
    if cls_name not in selected_classes:
        continue

    split = "train" if random.random() < TRAIN_SPLIT else "val"
    img.save(os.path.join(OUT_DIR, split, cls_name, f"{idx}.jpg"))

print("Preprocessing completed.")
