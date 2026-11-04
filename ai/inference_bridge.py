"""Verified Vision Park EfficientNet-B0 checkpoint loading and image inference.

Accepts only a complete two-class EfficientNet-B0 state dictionary.
Does not treat stored class predictions as model evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms

CLASS_NAMES = ("vacant", "occupied")
IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def image_transform():
    # Exactly matches the validation/test transform in finetune_efficientnet_local.py.
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def order_points(points: np.ndarray) -> np.ndarray:
    """Match the original PKLot pilot crop ordering; reject ambiguous quads."""
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("A parking space must have four finite [x,y] corners")
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).ravel()
    indices = [int(np.argmin(sums)), int(np.argmin(diffs)),
               int(np.argmax(sums)), int(np.argmax(diffs))]
    if len(set(indices)) != 4:
        raise ValueError("Ambiguous parking-space corner ordering")
    ordered = points[indices]
    if cv2.contourArea(ordered) < 12:
        raise ValueError("Parking polygon is too small or degenerate")
    return ordered


def perspective_crop(image_bgr: np.ndarray, polygon, size: int = IMAGE_SIZE):
    """Rectify and black-pad a slot like the existing PKLot crop pipeline.

    Coordinates must match the resolution of image_bgr exactly.
    """
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected a BGR color camera image")
    if isinstance(polygon, str):
        polygon = json.loads(polygon)
    tl, tr, br, bl = points = order_points(polygon)
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < 2 or height < 2:
        raise ValueError("Parking-space crop has invalid dimensions")
    destination = np.float32([
        [0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]
    ])
    matrix = cv2.getPerspectiveTransform(points, destination)
    crop = cv2.warpPerspective(image_bgr, matrix, (width, height))
    scale = min(size / width, size / height)
    w = max(1, min(size, int(width * scale)))
    h = max(1, min(size, int(height * scale)))
    resized = cv2.resize(crop, (w, h))
    padded = np.zeros((size, size, 3), dtype=np.uint8)
    x0, y0 = (size - w) // 2, (size - h) // 2
    padded[y0:y0 + h, x0:x0 + w] = resized
    return cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)


class EfficientNetOccupancy:
    def __init__(self, checkpoint_path, *, threshold=0.80, device=None):
        if not (0.5 < threshold <= 1):
            raise ValueError("threshold must be greater than 0.5 and at most 1")
        self.threshold = threshold
        self.device = torch.device(device) if device else choose_device()
        checkpoint_path = Path(checkpoint_path).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"No trained model at {checkpoint_path}. Train the local pilot first."
            )
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
        except Exception as exc:
            raise ValueError(f"Could not read checkpoint {checkpoint_path}: {exc}") from exc
        if not isinstance(checkpoint, dict):
            raise ValueError("Unsupported checkpoint. Expected a state dictionary.")
        state = checkpoint.get("model_state_dict", checkpoint)
        if not isinstance(state, dict):
            raise ValueError("Missing model_state_dict")
        if "classifier.1.weight" not in state or "features.0.0.weight" not in state:
            raise ValueError(
                "Incomplete checkpoint: missing EfficientNet feature weights. "
                "An 11 KB classifier-only file cannot be used. Retrain or locate the full model."
            )
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise ValueError(f"Checkpoint is not a complete EfficientNet-B0: {exc}") from exc
        classes = checkpoint.get("class_names", ["Vacant", "Occupied"])
        if [str(c).lower() for c in classes] != list(CLASS_NAMES):
            raise ValueError("Checkpoint class ordering must be Vacant=0, Occupied=1")
        self.model = model.to(self.device).eval()
        self.transform = image_transform()

    @torch.inference_mode()
    def predict_rgb(self, crop_rgb: np.ndarray):
        if crop_rgb.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
            raise ValueError("Expected one RGB 224x224 parking-space crop")
        batch = self.transform(crop_rgb).unsqueeze(0).to(self.device)
        probability = float(self.model(batch).softmax(dim=1)[0, 1].item())
        confidence = max(probability, 1 - probability)
        status = ("occupied" if probability >= 0.5 else "free") \
            if confidence >= self.threshold else "unknown"
        return {"status": status, "confidence": round(confidence, 6),
                "occupied_probability": round(probability, 6)}

    def predict_space(self, image_bgr: np.ndarray, polygon):
        return self.predict_rgb(perspective_crop(image_bgr, polygon))


def publish_observation(api_base, api_key, slot_code, prediction, *, timeout=5):
    """Explicit opt-in for a known, non-unknown observation only."""
    if prediction["status"] not in ("occupied", "free"):
        raise ValueError("Unknown/uncertain predictions must not be published as free")
    if not api_key:
        raise ValueError("Set VISIONPARK_DEVICE_API_KEY in the environment")
    url = api_base.rstrip("/") + "/api/iot/observation"
    parsed = urlsplit(url)
    if parsed.scheme != "https" and not (
        parsed.scheme == "http" and parsed.hostname in ("localhost", "127.0.0.1")
    ):
        raise ValueError("Use HTTPS except when connecting to a local test backend")
    body = json.dumps({
        "slot_code": slot_code, "source": "camera",
        "status": prediction["status"], "confidence": prediction["confidence"]
    }).encode("utf-8")
    request = Request(url, data=body, method="POST", headers={
        "Content-Type": "application/json", "x-device-key": api_key
    })
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description="Classify configured parking slots in one image")
    parser.add_argument("--image", required=True, help="Local photo path")
    parser.add_argument("--config", required=True, help="JSON camera slot coordinates")
    parser.add_argument("--checkpoint", required=True, help="Full trained EfficientNet .pt")
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--send", action="store_true", help="Explicitly POST confident results")
    parser.add_argument("--api-base", default="http://127.0.0.1:3000")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    camera_id, spaces = config["camera_id"], config["spaces"]
    frame = cv2.imread(args.image)
    if frame is None:
        parser.error(f"Cannot read image: {args.image}")
    classifier = EfficientNetOccupancy(args.checkpoint, threshold=args.threshold)
    observations = []
    for slot_code, polygon in spaces.items():
        try:
            result = classifier.predict_space(frame, polygon)
        except ValueError as exc:
            result = {"status": "unknown", "error": str(exc)}
        observation = {"camera_id": camera_id, "slot_code": slot_code, **result}
        observations.append(observation)
        if args.send and result["status"] in ("free", "occupied"):
            observation["backend_response"] = publish_observation(
                args.api_base, os.getenv("VISIONPARK_DEVICE_API_KEY"),
                slot_code, result
            )
    print(json.dumps(observations, indent=2))


if __name__ == "__main__":
    main()
