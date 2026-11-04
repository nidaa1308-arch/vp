"""Evaluate a full model checkpoint by running fresh inference on real test crops.

Do not accept test files containing only precomputed predictions.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from inference_bridge import EfficientNetOccupancy


class TestCrops(Dataset):
    def __init__(self, dataset_path, transform):
        with np.load(dataset_path, allow_pickle=False) as data:
            if "X" not in data.files or "y" not in data.files:
                raise ValueError(
                    "Test data must contain actual X image arrays and y ground-truth labels; "
                    "stored predictions are not independent model evaluation."
                )
            self.images = data["X"]
            self.labels = data["y"]
        if (self.images.ndim != 4 or self.images.shape[1:] != (224, 224, 3) or
                len(self.images) != len(self.labels) or not len(self.labels)):
            raise ValueError("Expected nonempty N x 224 x 224 x 3 test RGB images")
        if not set(np.unique(self.labels).tolist()).issubset({0, 1}):
            raise ValueError("Test labels must be 0 (vacant) or 1 (occupied)")
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.transform(self.images[index]), int(self.labels[index]), index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test", required=True, help="Actual test.npz with X and y")
    parser.add_argument("--output", default="ai/outputs/verified_evaluation")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    classifier = EfficientNetOccupancy(args.checkpoint)
    dataset = TestCrops(args.test, classifier.transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0)
    rows = []
    classifier.model.eval()
    with torch.inference_mode():
        for images, labels, indices in tqdm(loader, desc="Fresh inference"):
            logits = classifier.model(images.to(classifier.device))
            probs = logits.softmax(dim=1)[:, 1].cpu().numpy()
            for index, truth, prob in zip(indices.tolist(), labels.tolist(), probs.tolist()):
                rows.append({"sample_index": index, "actual": truth,
                             "predicted": int(prob >= 0.5),
                             "occupied_probability": round(float(prob), 8)})
    actual = [row["actual"] for row in rows]
    predicted = [row["predicted"] for row in rows]
    matrix = confusion_matrix(actual, predicted, labels=[0, 1]).tolist()
    metrics = {
        "test_samples": len(rows),
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision_occupied": float(precision_score(actual, predicted, zero_division=0)),
        "recall_occupied": float(recall_score(actual, predicted, zero_division=0)),
        "f1_occupied": float(f1_score(actual, predicted, zero_division=0)),
        "confusion_matrix": matrix,
        "note": "Fresh checkpoint inference on X images. Image-level splits may still leak nearby frames.",
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(metrics, indent=2))
    print("Saved:", output / "metrics.json", "and", output / "predictions.csv")


if __name__ == "__main__":
    main()
