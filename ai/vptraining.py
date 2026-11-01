
# ============================================================
# VISION PARK - COMPLETE AI DEVELOPMENT PIPELINE
# PKLot + EfficientNet-B0 Transfer Learning
# ============================================================

# STEP 1: INSTALL LIBRARIES

!pip install -q ultralytics opencv-python matplotlib pandas numpy tqdm scikit-learn torch torchvision

# STEP 2: IMPORT LIBRARIES

import os
import glob
import json
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import GroupShuffleSplit

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# STEP 3: CONFIGURE DEVICE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# STEP 4: MOUNT GOOGLE DRIVE

from google.colab import drive

drive.mount('/content/drive')

# STEP 5: CREATE PROJECT FOLDERS

main_dir = "/content/drive/MyDrive/VisionPark_AI"

folders = [
    "Dataset",
    "Processed_Dataset",
    "Trained_Models",
    "Results"
]

for folder in folders:
    os.makedirs(os.path.join(main_dir, folder), exist_ok=True)

dataset_dir = os.path.join(main_dir, "Dataset")
save_dir = os.path.join(main_dir, "Processed_Dataset")
model_dir = os.path.join(main_dir, "Trained_Models")
results_dir = os.path.join(main_dir, "Results")

print("Vision Park folders ready!")

# STEP 6: DOWNLOAD PKLOT IF NOT ALREADY PRESENT

archive_path = os.path.join(dataset_dir, "PKLot.tar.gz")

if not os.path.exists(archive_path):
    !wget -c -O "{archive_path}" "https://huggingface.co/datasets/teenygrad/pklot/resolve/main/PKLot.tar.gz"
else:
    print("PKLot archive already exists.")

# STEP 7: EXTRACT DATASET IF NEEDED

extract_path = "/content/parking_dataset"

os.makedirs(extract_path, exist_ok=True)

if not os.path.exists(os.path.join(extract_path, "PKLot")):
    !tar -xzf "{archive_path}" -C "{extract_path}"
else:
    print("Dataset already extracted.")

base_path = os.path.join(extract_path, "PKLot", "PKLot")

# STEP 8: VERIFY DATASET

xml_files = glob.glob(
    base_path + "/**/*.xml",
    recursive=True
)

jpg_files = glob.glob(
    base_path + "/**/*.jpg",
    recursive=True
)

print("XML files:", len(xml_files))
print("JPG images:", len(jpg_files))

if len(xml_files) == 0 or len(jpg_files) == 0:
    raise RuntimeError("Dataset missing. Check extraction.")

# STEP 9: LOAD EXISTING ANNOTATIONS OR GENERATE THEM

annotation_path = os.path.join(
    save_dir,
    "parking_annotations.csv"
)

if os.path.exists(annotation_path):

    print("Loading saved annotations...")

    df = pd.read_csv(annotation_path)

else:

    print("Generating parking-space annotations...")

    records = []
    missing_images = 0
    missing_labels = 0

    for xml_path in tqdm(xml_files):

        image_path = os.path.splitext(xml_path)[0] + ".jpg"

        if not os.path.exists(image_path):
            missing_images += 1
            continue

        parts = xml_path.split(os.sep)

        location = parts[-4]
        weather = parts[-3]

        tree = ET.parse(xml_path)
        root = tree.getroot()

        for space in root.findall(".//space"):

            occupancy = space.get("occupied")

            if occupancy not in ["0", "1"]:
                missing_labels += 1
                continue

            contour = space.find("contour")

            if contour is None:
                continue

            points = []

            for point in contour.findall("point"):

                x = float(point.get("x"))
                y = float(point.get("y"))

                points.append([x, y])

            if len(points) < 3:
                continue

            records.append({
                "image_path": image_path,
                "space_id": space.get("id"),
                "occupancy": int(occupancy),
                "status": "occupied" if occupancy == "1" else "vacant",
                "location": location,
                "weather": weather,
                "polygon": json.dumps(points)
            })

    df = pd.DataFrame(records)

    df.to_csv(annotation_path, index=False)

    print("Missing images:", missing_images)
    print("Missing labels:", missing_labels)

print("Total labelled spaces:", len(df))
print("Occupied:", (df["occupancy"] == 1).sum())
print("Vacant:", (df["occupancy"] == 0).sum())

# STEP 10: CREATE IMAGE-LEVEL TRAIN/VALIDATION/TEST SPLITS

split_paths = {
    "train": os.path.join(save_dir, "train_images.csv"),
    "validation": os.path.join(save_dir, "validation_images.csv"),
    "test": os.path.join(save_dir, "test_images.csv")
}

if all(os.path.exists(p) for p in split_paths.values()):

    print("Loading saved image splits...")

    train_images = pd.read_csv(split_paths["train"])["image_path"]
    val_images = pd.read_csv(split_paths["validation"])["image_path"]
    test_images = pd.read_csv(split_paths["test"])["image_path"]

    train = df[df["image_path"].isin(train_images)].copy()
    val = df[df["image_path"].isin(val_images)].copy()
    test = df[df["image_path"].isin(test_images)].copy()

else:

    print("Creating image-level splits...")

    split1 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.15,
        random_state=42
    )

    train_val_idx, test_idx = next(
        split1.split(df, groups=df["image_path"])
    )

    train_val = df.iloc[train_val_idx]
    test = df.iloc[test_idx]

    split2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.17647,
        random_state=42
    )

    train_idx, val_idx = next(
        split2.split(train_val, groups=train_val["image_path"])
    )

    train = train_val.iloc[train_idx].copy()
    val = train_val.iloc[val_idx].copy()

    test = test.copy()

    for name, data in [
        ("train", train),
        ("validation", val),
        ("test", test)
    ]:

        data["image_path"].drop_duplicates().to_csv(
            split_paths[name],
            index=False,
            header=["image_path"]
        )

print("Training spaces:", len(train))
print("Validation spaces:", len(val))
print("Testing spaces:", len(test))

# STEP 11: CHECK POLYGON CORNERS

point_counts = Counter(
    len(json.loads(p))
    for p in train["polygon"].sample(
        n=min(5000, len(train)),
        random_state=42
    )
)

print("Polygon corner distribution:", point_counts)

# STEP 12: ORDER FOUR CORNERS

def order_points(points):

    points = np.array(points, dtype="float32")

    rect = np.zeros((4, 2), dtype="float32")

    s = points.sum(axis=1)

    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]

    diff = np.diff(points, axis=1)

    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]

    return rect

# STEP 13: PERSPECTIVE-CORRECTED CROP EXTRACTION

def extract_parking_crop(image_path, polygon, size=224):

    image = cv2.imread(image_path)

    if image is None:
        return None

    points = np.array(
        json.loads(polygon),
        dtype="float32"
    )

    if points.shape != (4, 2):
        return None

    rect = order_points(points)

    tl, tr, br, bl = rect

    width = int(max(
        np.linalg.norm(br - bl),
        np.linalg.norm(tr - tl)
    ))

    height = int(max(
        np.linalg.norm(tr - br),
        np.linalg.norm(tl - bl)
    ))

    if width < 2 or height < 2:
        return None

    destination = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(
        rect,
        destination
    )

    crop = cv2.warpPerspective(
        image,
        matrix,
        (width, height)
    )

    scale = min(size / width, size / height)

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    crop = cv2.resize(
        crop,
        (new_width, new_height)
    )

    final_image = np.zeros(
        (size, size, 3),
        dtype=np.uint8
    )

    x_offset = (size - new_width) // 2
    y_offset = (size - new_height) // 2

    final_image[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = crop

    return cv2.cvtColor(
        final_image,
        cv2.COLOR_BGR2RGB
    )

# STEP 14: PREVIEW EXTRACTED PARKING SPACES

samples = train.sample(n=6, random_state=42)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for ax, (_, row) in zip(axes.flatten(), samples.iterrows()):

    crop = extract_parking_crop(
        row["image_path"],
        row["polygon"]
    )

    if crop is not None:
        ax.imshow(crop)

    ax.set_title(row["status"].upper())
    ax.axis("off")

plt.tight_layout()
plt.show()

# STEP 15: PREVIEW ORIGINAL POLYGONS AND CROPS

samples = train.sample(n=3, random_state=42)

fig, axes = plt.subplots(3, 2, figsize=(14, 15))

for i, (_, row) in enumerate(samples.iterrows()):

    image = cv2.imread(row["image_path"])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    points = np.array(
        json.loads(row["polygon"]),
        dtype=np.int32
    )

    preview = image.copy()

    cv2.polylines(
        preview,
        [points],
        True,
        (255, 0, 0),
        5
    )

    crop = extract_parking_crop(
        row["image_path"],
        row["polygon"]
    )

    axes[i, 0].imshow(preview)
    axes[i, 0].set_title("Original + parking polygon")

    axes[i, 1].imshow(crop)
    axes[i, 1].set_title("Extracted crop")

    axes[i, 0].axis("off")
    axes[i, 1].axis("off")

plt.tight_layout()
plt.show()

# STEP 16: CREATE PILOT DATASET FOLDER

pilot_dir = os.path.join(save_dir, "Pilot")

os.makedirs(pilot_dir, exist_ok=True)

# STEP 17: GENERATE BALANCED PILOT DATA

def prepare_pilot(data, split_name, samples_per_class=1000):

    output_file = os.path.join(
        pilot_dir,
        f"{split_name}.npz"
    )

    # Reuse completed pilot files
    if os.path.exists(output_file):

        print(split_name, "pilot file already exists. Skipping.")

        with np.load(output_file) as saved:
            print("Images:", len(saved["y"]))

        return

    vacant = data[data["occupancy"] == 0]
    occupied = data[data["occupancy"] == 1]

    if len(vacant) < samples_per_class or len(occupied) < samples_per_class:
        raise ValueError("Insufficient samples for balanced pilot.")

    selected = pd.concat([
        vacant.sample(n=samples_per_class, random_state=42),
        occupied.sample(n=samples_per_class, random_state=42)
    ]).sample(frac=1, random_state=42)

    images = []
    labels = []

    for _, row in tqdm(
        selected.iterrows(),
        total=len(selected),
        desc=split_name
    ):

        crop = extract_parking_crop(
            row["image_path"],
            row["polygon"]
        )

        if crop is not None:

            images.append(crop)
            labels.append(int(row["occupancy"]))

    X = np.asarray(images, dtype=np.uint8)
    y = np.asarray(labels, dtype=np.int64)

    np.savez_compressed(
        output_file,
        X=X,
        y=y
    )

    print(split_name.upper(), "completed")
    print("Images:", len(X))
    print("Vacant:", np.sum(y == 0))
    print("Occupied:", np.sum(y == 1))

# STEP 18: GENERATE ALL THREE PILOT SPLITS

prepare_pilot(train, "train")
prepare_pilot(val, "validation")
prepare_pilot(test, "test")

# STEP 19: LOAD PILOT DATA

class ParkingDataset(Dataset):

    def __init__(self, file_path, training=False):

        data = np.load(file_path)

        self.images = data["X"]
        self.labels = data["y"]

        self.training = training

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        image = self.images[index]
        label = int(self.labels[index])

        image = self.transform(image)

        return image, label

# STEP 20: CREATE DATA LOADERS

train_dataset = ParkingDataset(
    os.path.join(pilot_dir, "train.npz"),
    training=True
)

val_dataset = ParkingDataset(
    os.path.join(pilot_dir, "validation.npz")
)

test_dataset = ParkingDataset(
    os.path.join(pilot_dir, "test.npz")
)

batch_size = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

print("Data loaders created!")

# STEP 21: LOAD PRETRAINED EFFICIENTNET-B0

weights = models.EfficientNet_B0_Weights.DEFAULT

model = models.efficientnet_b0(weights=weights)

# Replace original 1000-class head with 2-class head

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2
)

model = model.to(device)

print("Pretrained EfficientNet-B0 loaded!")

# STEP 22: FREEZE FEATURE EXTRACTOR

for param in model.features.parameters():
    param.requires_grad = False

print("Feature extractor frozen.")
print("Only classification head will initially train.")

# STEP 23: TRAINING CONFIGURATION

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=0.001
)

# STEP 24: TRAINING AND VALIDATION FUNCTION

def run_epoch(loader, training=False):

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):

            outputs = model(images)

            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total

# STEP 25: TRAIN CLASSIFICATION HEAD

epochs = 5

best_val_accuracy = 0

for epoch in range(epochs):

    train_loss, train_acc = run_epoch(
        train_loader,
        training=True
    )

    val_loss, val_acc = run_epoch(
        val_loader,
        training=False
    )

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Train loss: {train_loss:.4f} | "
        f"Train accuracy: {train_acc:.4f} | "
        f"Validation loss: {val_loss:.4f} | "
        f"Validation accuracy: {val_acc:.4f}"
    )

    if val_acc > best_val_accuracy:

        best_val_accuracy = val_acc

        torch.save(
            model.state_dict(),
            os.path.join(
                model_dir,
                "efficientnet_b0_head_best.pth"
            )
        )

# STEP 26: FINE-TUNE SELECTED FEATURE LAYERS

# Unfreeze later feature blocks
for param in model.features[-2:].parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001
)

print("Selected feature layers unfrozen.")

# STEP 27: FINE-TUNE MODEL

fine_tune_epochs = 5

for epoch in range(fine_tune_epochs):

    train_loss, train_acc = run_epoch(
        train_loader,
        training=True
    )

    val_loss, val_acc = run_epoch(
        val_loader,
        training=False
    )

    print(
        f"Fine-tuning Epoch {epoch+1}/{fine_tune_epochs} | "
        f"Train accuracy: {train_acc:.4f} | "
        f"Validation accuracy: {val_acc:.4f}"
    )

    if val_acc > best_val_accuracy:

        best_val_accuracy = val_acc

        torch.save(
            model.state_dict(),
            os.path.join(
                model_dir,
                "efficientnet_b0_best.pth"
            )
        )

# STEP 28: LOAD BEST MODEL FOR TESTING

best_model_path = os.path.join(
    model_dir,
    "efficientnet_b0_best.pth"
)

if not os.path.exists(best_model_path):

    best_model_path = os.path.join(
        model_dir,
        "efficientnet_b0_head_best.pth"
    )

model.load_state_dict(
    torch.load(best_model_path, map_location=device)
)

model.eval()

# STEP 29: FINAL TEST EVALUATION

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )

print("\nFINAL TEST RESULTS")

print("Accuracy:",
      accuracy_score(all_labels, all_predictions))

print("Precision:",
      precision_score(all_labels, all_predictions))

print("Recall:",
      recall_score(all_labels, all_predictions))

print("F1 Score:",
      f1_score(all_labels, all_predictions))

print("\nClassification Report:")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=["Vacant", "Occupied"]
    )
)

# STEP 30: CONFUSION MATRIX

cm = confusion_matrix(
    all_labels,
    all_predictions
)

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Vision Park - EfficientNet-B0 Confusion Matrix")

plt.xticks([0, 1], ["Vacant", "Occupied"])
plt.yticks([0, 1], ["Vacant", "Occupied"])

plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()
plt.show()

# STEP 31: SAVE FINAL MODEL

final_model_path = os.path.join(
    model_dir,
    "VisionPark_EfficientNet_B0.pth"
)

torch.save(
    model.state_dict(),
    final_model_path
)

print("Final model saved!")
print(final_model_path)

print("\nVISION PARK PILOT TRAINING PIPELINE COMPLETED!")
