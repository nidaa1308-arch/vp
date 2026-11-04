"""Build Vision Park's small PKLot pilot directly from PKLot.tar.gz.

Reads the archive sequentially twice: XML annotations first, selected JPEGs second.
Does NOT extract the full 19+ GB archive. The resulting three .npz files can be
used by ai/finetune_efficientnet_local.py.

Existing Colab split CSVs are preferred when all three are present; otherwise
we deterministically split by (camera, weather, capture-day) to reduce near-frame
leakage. Never compare accuracy from different split strategies as though equal.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import tarfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

SPLITS = ("train", "validation", "test")
LABELS = (0, 1)
SIZE = 224
SEED = 42


def relative_pklot_path(name: str) -> str | None:
    """Normalize archive paths and old absolute Colab CSV paths to a shared key."""
    text = str(name).replace("\\", "/").lstrip("./")
    for marker in ("PKLot/PKLot/",):
        if marker in text:
            return text.split(marker, 1)[1]
    for prefix in ("PUCPR/", "UFPR04/", "UFPR05/"):
        if text.startswith(prefix):
            return text
    return None


def get_existing_assignments(folder: Path):
    paths = [folder / f"{split}_images.csv" for split in SPLITS]
    if not any(path.exists() for path in paths):
        return None
    if not all(path.exists() for path in paths):
        raise RuntimeError(
            "Only some saved split CSVs were found. Keep all three together, "
            "or move them aside to use a new day/session split."
        )
    assigned = {}
    for split, path in zip(SPLITS, paths):
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "image_path" not in (reader.fieldnames or []):
                raise ValueError(f"{path} has no image_path column")
            for row in reader:
                key = relative_pklot_path(row["image_path"])
                if not key:
                    continue
                if key in assigned and assigned[key] != split:
                    raise ValueError(f"Image appears in multiple splits: {key}")
                assigned[key] = split
    if not assigned:
        raise ValueError("Saved split CSVs contained no usable PKLot image paths")
    return assigned


def deterministic_day_split(relative_path: str) -> str:
    # PKLot's canonical path: CAMERA/WEATHER/CAPTURE-DAY/FILENAME.jpg
    parts = relative_path.split("/")
    if len(parts) < 4:
        raise ValueError(f"Unexpected PKLot image path: {relative_path}")
    group = "/".join(parts[:3])
    bucket = int.from_bytes(hashlib.sha256(group.encode()).digest()[:8], "big") % 100
    return "train" if bucket < 70 else ("validation" if bucket < 85 else "test")


def ordered_quad(raw_points):
    points = np.asarray(raw_points, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        return None
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).ravel()
    ids = [int(np.argmin(sums)), int(np.argmin(diffs)),
           int(np.argmax(sums)), int(np.argmax(diffs))]
    if len(set(ids)) < 4:
        return None
    ordered = points[ids]
    return ordered if cv2.contourArea(ordered) >= 12 else None


def warped_crop(image, raw_points):
    # Matches the original Vision Park Colab preprocessing (BGR -> RGB + padding).
    points = ordered_quad(raw_points)
    if points is None:
        return None
    tl, tr, br, bl = points
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < 2 or height < 2:
        return None
    dst = np.float32([[0, 0], [width - 1, 0],
                      [width - 1, height - 1], [0, height - 1]])
    crop = cv2.warpPerspective(
        image, cv2.getPerspectiveTransform(points, dst), (width, height))
    scale = min(SIZE / width, SIZE / height)
    w, h = max(1, min(SIZE, int(width * scale))), max(1, min(SIZE, int(height * scale)))
    crop = cv2.resize(crop, (w, h))
    output = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    x, y = (SIZE - w) // 2, (SIZE - h) // 2
    output[y:y + h, x:x + w] = crop
    return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)


def scan_xml_candidates(archive: Path, assigned, per_class: int, seed: int):
    capacity = max(per_class + 300, int(per_class * 1.3))
    rng = random.Random(seed)
    buckets = {(split, label): [] for split in SPLITS for label in LABELS}
    seen = {key: 0 for key in buckets}
    xml_files = valid_spaces = 0
    print("Pass 1/2: reading parking-space XML annotations (no image extraction)...", flush=True)

    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.lower().endswith(".xml"):
                continue
            relative = relative_pklot_path(member.name)
            if relative is None:
                continue
            image_key = relative[:-4] + ".jpg"
            split = assigned.get(image_key) if assigned is not None else deterministic_day_split(image_key)
            if split is None:
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            try:
                root = ET.fromstring(handle.read())
            except ET.ParseError:
                continue
            xml_files += 1
            for space in root.findall(".//space"):
                raw_label = space.get("occupied")
                if raw_label not in ("0", "1"):
                    continue
                contour = space.find("contour")
                if contour is None:
                    continue
                try:
                    points = [[float(p.attrib["x"]), float(p.attrib["y"])]
                              for p in contour.findall("point")]
                except (KeyError, ValueError, TypeError):
                    continue
                if ordered_quad(points) is None:
                    continue
                label = int(raw_label)
                valid_spaces += 1
                key = (split, label)
                seen[key] += 1
                bucket = buckets[key]
                candidate = (image_key, points)
                if len(bucket) < capacity:
                    bucket.append(candidate)
                else:
                    index = rng.randrange(seen[key])
                    if index < capacity:
                        bucket[index] = candidate
            if xml_files % 2000 == 0:
                print(f"  XMLs {xml_files:,}; labelled valid spaces {valid_spaces:,}", flush=True)
    print(f"Annotations read: {xml_files:,} XMLs; {valid_spaces:,} valid spaces")
    for key, total in seen.items():
        print(f"  {key[0]} / {'vacant' if key[1] == 0 else 'occupied'}: {total:,} available")
        if total < per_class:
            raise RuntimeError(
                f"Insufficient {key} samples ({total}) for a balanced pilot of {per_class}. "
                "Use --samples-per-class with a smaller value if needed."
            )
    return buckets


def build_crops(archive, buckets, per_class):
    by_image = defaultdict(list)
    for (split, label), candidates in buckets.items():
        for image_key, points in candidates:
            by_image[image_key].append((split, label, points))
    gathered = {(split, label): [] for split in SPLITS for label in LABELS}
    processed_images = 0
    print("Pass 2/2: reading only selected JPEG image bytes...", flush=True)

    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.lower().endswith(".jpg"):
                continue
            relative = relative_pklot_path(member.name)
            if relative not in by_image:
                continue
            requested = by_image.pop(relative)
            if all(len(gathered[(split, label)]) >= per_class
                   for split, label, _ in requested):
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            image = cv2.imdecode(np.frombuffer(handle.read(), np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                continue
            processed_images += 1
            for split, label, points in requested:
                collected = gathered[(split, label)]
                if len(collected) >= per_class:
                    continue
                crop = warped_crop(image, points)
                if crop is not None:
                    collected.append(crop)
            if processed_images % 100 == 0:
                counts = {f"{s}-{l}": len(v) for (s, l), v in gathered.items()}
                print(f"  Selected JPEGs {processed_images:,}; {counts}", flush=True)
            if all(len(images) == per_class for images in gathered.values()):
                break

    for key, images in gathered.items():
        if len(images) != per_class:
            raise RuntimeError(
                f"Only {len(images)} usable crops for {key}; expected {per_class}. "
                "No output files were overwritten. Check archive contents and annotations."
            )
    print(f"Built {per_class * 6:,} crops from {processed_images:,} source JPEGs.")
    return gathered


def save_pilots(output_dir, gathered, per_class, archive, split_mode, seed):
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, split in enumerate(SPLITS):
        rng = np.random.default_rng(seed + i)
        x = np.stack(gathered[(split, 0)] + gathered[(split, 1)])
        y = np.asarray([0] * per_class + [1] * per_class, dtype=np.int64)
        order = rng.permutation(len(y))
        output_file = output_dir / f"{split}.npz"
        temporary = output_dir / f".{split}.npz"
        np.savez_compressed(temporary, X=x[order], y=y[order])
        temporary.replace(output_file)
        print(f"Saved {output_file} with {len(y)} images")
    metadata = {
        "source": archive.name,
        "source_size_bytes": archive.stat().st_size,
        "split_strategy": split_mode,
        "seed": seed,
        "samples_per_class_per_split": per_class,
        "labels": {"0": "Vacant", "1": "Occupied"},
        "preprocessing": "224x224 four-point homography, aspect-preserving black padding, RGB",
        "caution": "Evaluate toy-car images separately; PKLot contains real vehicles.",
    }
    (output_dir / "pilot_manifest.json").write_text(json.dumps(metadata, indent=2))
    print("Pilot preparation completed. Now run: python ai/finetune_efficientnet_local.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True,
                        help="Path to existing PKLot.tar.gz (about 4.56 GiB)")
    parser.add_argument("--output-dir", default="~/VisionPark_AI/Processed_Dataset/Pilot")
    parser.add_argument("--split-dir", default="~/VisionPark_AI/Processed_Dataset",
                        help="Folder with optional train_images.csv, validation_images.csv, test_images.csv")
    parser.add_argument("--samples-per-class", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite ALL three existing pilot files together")
    args = parser.parse_args()

    archive = Path(args.archive).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    if not archive.is_file():
        parser.error(f"Archive not found: {archive}")
    if args.samples_per_class < 1:
        parser.error("samples-per-class must be positive")
    existing = [output_dir / f"{split}.npz" for split in SPLITS]
    if any(path.exists() for path in existing) and not args.overwrite:
        parser.error("Pilot .npz already exists. Back it up first; then use --overwrite to replace all three.")
    assigned = get_existing_assignments(Path(args.split_dir).expanduser())
    mode = "saved-image-CSVs" if assigned is not None else "camera-weather-day-hash"
    print(f"Splitting by: {mode}")
    buckets = scan_xml_candidates(archive, assigned, args.samples_per_class, args.seed)
    gathered = build_crops(archive, buckets, args.samples_per_class)
    save_pilots(output_dir, gathered, args.samples_per_class, archive, mode, args.seed)


if __name__ == "__main__":
    main()
