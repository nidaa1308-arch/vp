"""Vision Park: PKLot occupancy fine-tuning with pretrained EfficientNet-B0.

Run in Google Colab on a GPU with the prior VisionPark_AI Drive dataset.
Labels are 0=vacant and 1=occupied. This is a balanced pilot experiment.
"""
from __future__ import annotations

import json
import random
import shutil
import subprocess
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm

SEED = 42
SIZE = 224
PER_CLASS = 1000
BATCH_SIZE = 32
HEAD_EPOCHS = 5
FINETUNE_EPOCHS = 5
CLASSES = ["Vacant", "Occupied"]

DRIVE = Path("/content/drive/MyDrive/VisionPark_AI")
PROCESSED = DRIVE / "Processed_Dataset"
PILOT = PROCESSED / "Pilot"
MODELS = DRIVE / "Trained_Models" / "EfficientNet_B0_Pilot_v2"
RESULTS = DRIVE / "Results" / "EfficientNet_B0_Pilot_v2"
ARCHIVE = DRIVE / "Dataset" / "PKLot.tar.gz"
EXTRACT = Path("/content/parking_dataset")


def setup():
    try:
        from google.colab import drive
    except ImportError as exc:
        raise RuntimeError("This entry point is designed for Google Colab.") from exc
    drive.mount("/content/drive")
    for directory in (PILOT, MODELS, RESULTS):
        directory.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("Enable Runtime > Change runtime type > T4 GPU in Colab.")
    print("GPU:", torch.cuda.get_device_name(0))
    return torch.device("cuda")


def pilot_is_valid(path):
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            x, y = data["X"], data["y"]
            good = (x.ndim == 4 and x.shape[1:] == (SIZE, SIZE, 3)
                    and x.dtype == np.uint8 and len(x) == len(y)
                    and len(x) > 0 and set(np.unique(y).tolist()) == {0, 1})
            if good:
                counts = np.bincount(y.astype(np.int64), minlength=2)
                print(f"Cached {path.name}: vacant={counts[0]} occupied={counts[1]}")
            return bool(good)
    except (OSError, KeyError, ValueError, EOFError) as exc:
        print("Invalid cached pilot:", path, exc)
        return False


def crop_parking_space(path, polygon):
    points = np.asarray(json.loads(polygon), dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        return None
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    points = points[np.argsort(angles)]
    points = np.roll(points, -int(np.argmin(points.sum(axis=1))), axis=0)
    if cv2.contourArea(points) < 12:
        return None
    image = cv2.imread(path)
    if image is None:
        return None
    tl, tr, br, bl = points
    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if width < 2 or height < 2:
        return None
    dst = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    image = cv2.warpPerspective(image, cv2.getPerspectiveTransform(points, dst), (width, height))
    scale = min(SIZE / width, SIZE / height)
    w, h = max(1, min(SIZE, round(width * scale))), max(1, min(SIZE, round(height * scale)))
    image = cv2.resize(image, (w, h))
    padded = np.zeros((SIZE, SIZE, 3), np.uint8)
    x0, y0 = (SIZE - w) // 2, (SIZE - h) // 2
    padded[y0:y0 + h, x0:x0 + w] = image
    return cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)


def make_splits(df):
    filenames = {part: PROCESSED / f"{part}_images.csv"
                 for part in ("train", "validation", "test")}
    if all(path.is_file() for path in filenames.values()):
        groups = {part: set(pd.read_csv(path)["image_path"].dropna())
                  for part, path in filenames.items()}
        assert not (groups["train"] & groups["validation"]), "Train/validation image overlap"
        assert not (groups["train"] & groups["test"]), "Train/test image overlap"
        assert not (groups["validation"] & groups["test"]), "Validation/test image overlap"
        return {part: df[df["image_path"].isin(images)] for part, images in groups.items()}
    first = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    tv_i, test_i = next(first.split(df, groups=df["image_path"]))
    tv = df.iloc[tv_i]
    second = GroupShuffleSplit(n_splits=1, test_size=0.17647, random_state=SEED)
    train_i, val_i = next(second.split(tv, groups=tv["image_path"]))
    splits = {"train": tv.iloc[train_i], "validation": tv.iloc[val_i], "test": df.iloc[test_i]}
    for part, frame in splits.items():
        frame[["image_path"]].drop_duplicates().to_csv(filenames[part], index=False)
    return splits


def make_pilot(frame, part):
    rng = np.random.default_rng(SEED + {"train": 0, "validation": 1, "test": 2}[part])
    images, labels = [], []
    for label in (0, 1):
        subset = frame[frame["occupancy"] == label]
        if len(subset) < PER_CLASS:
            raise RuntimeError(f"Insufficient class {label} examples in {part}")
        found = 0
        for index in tqdm(rng.permutation(len(subset)), desc=f"{part}: {CLASSES[label]}", leave=False):
            row = subset.iloc[int(index)]
            image = crop_parking_space(str(row["image_path"]), str(row["polygon"]))
            if image is None:
                continue
            images.append(image)
            labels.append(label)
            found += 1
            if found == PER_CLASS:
                break
        if found < PER_CLASS:
            raise RuntimeError(f"Only {found} valid crops in {part} class {label}")
    order = rng.permutation(len(labels))
    local_file = Path("/content") / f"visionpark_{part}.npz"
    np.savez_compressed(local_file, X=np.stack(images)[order],
                        y=np.asarray(labels, dtype=np.int64)[order])
    shutil.copyfile(local_file, PILOT / f"{part}.npz")
    local_file.unlink()
    print(f"Created {part} pilot: {len(labels)} crops")


def ensure_pilot_data():
    needed = [part for part in ("train", "validation", "test")
              if not pilot_is_valid(PILOT / f"{part}.npz")]
    if not needed:
        print("All pilot archives found: skipping full PKLot extraction.")
        return
    csv = PROCESSED / "parking_annotations.csv"
    if not csv.is_file():
        raise FileNotFoundError(
            f"Missing {csv}. Create annotations using the original ai/vptraining.py first."
        )
    df = pd.read_csv(csv, usecols=["image_path", "occupancy", "polygon"])
    splits = make_splits(df)
    first_photo = Path(str(splits[needed[0]].iloc[0]["image_path"]))
    if not first_photo.is_file():
        if not ARCHIVE.is_file():
            raise FileNotFoundError(f"Missing {ARCHIVE}; restore your PKLot archive in Google Drive.")
        EXTRACT.mkdir(parents=True, exist_ok=True)
        print("Extracting PKLot from Google Drive into ephemeral Colab storage...")
        subprocess.run(["tar", "-xzf", str(ARCHIVE), "-C", str(EXTRACT)], check=True)
    for part in needed:
        make_pilot(splits[part], part)
    for part in ("train", "validation", "test"):
        if not pilot_is_valid(PILOT / f"{part}.npz"):
            raise RuntimeError(f"Invalid pilot {part}.npz")


class ParkingDataset(Dataset):
    def __init__(self, path, training=False):
        with np.load(path, allow_pickle=False) as data:
            self.images = data["X"]
            self.labels = data["y"].astype(np.int64)
        augmentation = ([transforms.RandomHorizontalFlip(0.5),
                         transforms.ColorJitter(0.12, 0.12, 0.08, 0.02)] if training else [])
        self.transform = transforms.Compose([
            transforms.ToPILImage(), *augmentation, transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.transform(self.images[index]), int(self.labels[index])


def build_loaders():
    return {part: DataLoader(
        ParkingDataset(PILOT / f"{part}.npz", training=part == "train"),
        batch_size=BATCH_SIZE, shuffle=part == "train", num_workers=2,
        pin_memory=torch.cuda.is_available())
        for part in ("train", "validation", "test")}


def epoch(model, loader, device, stage, optimizer=None):
    training = optimizer is not None
    model.train(training)
    if training and stage == "head":
        model.features.eval()  # Keep frozen BatchNorm and stochastic depth unchanged.
    elif training:
        for layer in model.features.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()  # Avoid drifting running statistics on a small pilot.
    criterion = nn.CrossEntropyLoss()
    loss_sum = correct = count = 0
    for images, labels in tqdm(loader, leave=False, desc=f"{stage}: {'train' if training else 'val'}"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        size = labels.size(0)
        loss_sum += float(loss.item()) * size
        correct += int((logits.argmax(1) == labels).sum().item())
        count += size
    return loss_sum / count, correct / count


def train_stage(model, loaders, device, stage, epochs, optimizer, history):
    best = None
    path = MODELS / f"{stage}_best.pt"
    for number in range(1, epochs + 1):
        train_loss, train_acc = epoch(model, loaders["train"], device, stage, optimizer)
        val_loss, val_acc = epoch(model, loaders["validation"], device, stage)
        history.append({"stage": stage, "epoch": number, "train_loss": train_loss,
                        "train_accuracy": train_acc, "val_loss": val_loss,
                        "val_accuracy": val_acc})
        pd.DataFrame(history).to_csv(RESULTS / "history.csv", index=False)
        print(f"{stage} {number}/{epochs} | train={train_acc:.2%} | val={val_acc:.2%} "
              f"| val_loss={val_loss:.4f}")
        if best is None or val_acc > best["accuracy"] or (
                val_acc == best["accuracy"] and val_loss < best["loss"]):
            best = {"accuracy": val_acc, "loss": val_loss}
            torch.save({"model_state_dict": model.state_dict(), "stage": stage,
                        "epoch": number, "val_accuracy": val_acc, "val_loss": val_loss,
                        "class_names": CLASSES, "image_size": SIZE, "seed": SEED}, path)
            print("Saved checkpoint:", path)
    return path, best


def evaluate(model, loader, device):
    model.eval()
    actual, predicted = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Final test", leave=False):
            result = model(images.to(device))
            predicted.extend(result.argmax(1).cpu().tolist())
            actual.extend(labels.tolist())
    cm = confusion_matrix(actual, predicted, labels=[0, 1])
    metrics = {"accuracy": float(accuracy_score(actual, predicted)),
               "precision_occupied": float(precision_score(actual, predicted, zero_division=0)),
               "recall_occupied": float(recall_score(actual, predicted, zero_division=0)),
               "f1_occupied": float(f1_score(actual, predicted, zero_division=0)),
               "confusion_matrix": cm.tolist(), "test_crops": len(actual)}
    print("\nFINAL PILOT TEST\n", classification_report(
        actual, predicted, labels=[0, 1], target_names=CLASSES, zero_division=0))
    print("Confusion matrix (actual rows, predicted columns):\n", cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], CLASSES)
    ax.set_yticks([0, 1], CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Vision Park: EfficientNet-B0 pilot")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "confusion_matrix.png", dpi=160)
    plt.show()
    return metrics


def main():
    device = setup()
    ensure_pilot_data()
    loaders = build_loaders()

    # ImageNet-pretrained features, fresh two-class output layer.
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.to(device)

    history = []
    print("\nSTAGE 1: train classification head")
    head_path, head_best = train_stage(
        model, loaders, device, "head", HEAD_EPOCHS,
        torch.optim.Adam(model.classifier.parameters(), lr=1e-3), history)

    # Fine-tuning starts from the best head epoch, never the last head epoch.
    head = torch.load(head_path, map_location=device, weights_only=True)
    model.load_state_dict(head["model_state_dict"])
    for parameter in model.features[-2:].parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.Adam([
        {"params": model.features[-2:].parameters(), "lr": 1e-5},
        {"params": model.classifier.parameters(), "lr": 1e-4},
    ])
    print("\nSTAGE 2: fine-tune last two feature blocks")
    tuned_path, tuned_best = train_stage(
        model, loaders, device, "finetuned", FINETUNE_EPOCHS, optimizer, history)

    tuned_wins = tuned_best["accuracy"] > head_best["accuracy"] or (
        tuned_best["accuracy"] == head_best["accuracy"] and
        tuned_best["loss"] < head_best["loss"])
    selected = tuned_path if tuned_wins else head_path
    final_path = MODELS / "VisionPark_EfficientNet_B0_Pilot_v2.pt"
    shutil.copyfile(selected, final_path)
    checkpoint = torch.load(final_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Best validation model: {checkpoint['stage']} "
          f"({checkpoint['val_accuracy']:.2%})")

    # Test set is evaluated only after validation-based checkpoint selection.
    metrics = evaluate(model, loaders["test"], device)
    metrics.update({"selected_stage": checkpoint["stage"],
                    "selected_val_accuracy": float(checkpoint["val_accuracy"]),
                    "head_best_val_accuracy": float(head_best["accuracy"]),
                    "finetuned_best_val_accuracy": float(tuned_best["accuracy"]),
                    "model_path": str(final_path),
                    "note": "Image-level split; adjacent sessions may still leak."})
    with open(RESULTS / "test_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print("\nModel saved to:", final_path)
    print("Results saved to:", RESULTS)
    print("This pilot is not an independent camera/site deployment benchmark.")


if __name__ == "__main__":
    main()
