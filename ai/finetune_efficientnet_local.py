"""Train Vision Park's EfficientNet-B0 pilot on a Mac or local workstation.

Copy the existing Google Drive Pilot/*.npz files into:
    ~/VisionPark_AI/Processed_Dataset/Pilot/

To override the data root, set VISIONPARK_DATA_DIR before running this script.
Uses MPS (Apple GPU), CUDA (NVIDIA GPU), or CPU.
"""
from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm

SEED = 42
SIZE = 224
BATCH_SIZE = int(os.environ.get("VISIONPARK_BATCH_SIZE", "16"))
HEAD_EPOCHS = 5
FINETUNE_EPOCHS = 5
CLASSES = ["Vacant", "Occupied"]
DRIVE = Path(os.environ.get(
    "VISIONPARK_DATA_DIR", str(Path.home() / "VisionPark_AI")
)).expanduser()
PILOT = DRIVE / "Processed_Dataset" / "Pilot"
MODELS = DRIVE / "Trained_Models" / "EfficientNet_B0_Pilot_local"
RESULTS = DRIVE / "Results" / "EfficientNet_B0_Pilot_local"


def setup():
    for folder in (MODELS, RESULTS):
        folder.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}. Pilot data folder: {PILOT}")
    if device.type == "cpu":
        print("GPU acceleration unavailable; training may be slow.")
    return device


def ensure_pilot_data():
    expected = [PILOT / f"{name}.npz" for name in ("train", "validation", "test")]
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Copy all three pilot archives from your Google Drive "
            "VisionPark_AI/Processed_Dataset/Pilot folder into "
            f"{PILOT}. Missing: {', '.join(p.name for p in missing)}"
        )
    for path in expected:
        try:
            with np.load(path, allow_pickle=False) as data:
                x, y = data["X"], data["y"]
                if x.ndim != 4 or x.shape[1:] != (SIZE, SIZE, 3) or len(x) != len(y):
                    raise ValueError("Expected RGB 224x224 crops and matching labels")
                labels = np.unique(y)
                if not set(labels.tolist()).issubset({0, 1}) or len(labels) != 2:
                    raise ValueError(f"Expected both class labels 0 and 1, got {labels}")
                print(f"{path.name}: {len(y)} crops, labels {labels.tolist()}")
        except (OSError, EOFError, KeyError, ValueError) as exc:
            raise RuntimeError(f"Invalid pilot archive {path}: {exc}") from exc


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
