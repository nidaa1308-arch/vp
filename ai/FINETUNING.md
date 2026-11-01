# Vision Park — EfficientNet-B0 pilot fine-tuning

Run **[finetune_efficientnet.py](finetune_efficientnet.py)** in **Google Colab with a T4 GPU**. This experiment fine-tunes ImageNet-pretrained EfficientNet-B0 on balanced, perspective-corrected PKLot parking-space crops. Classes: **0 = vacant; 1 = occupied**.

## Start in Colab

1. Open [Google Colab](https://colab.research.google.com/), create a notebook and choose **Runtime → Change runtime type → T4 GPU**.
2. Copy these **two cells** into Colab and run in order.

**Cell 1 — fetch the current source file from this repository:**

~~~python
!wget -q -O /content/finetune_efficientnet.py https://raw.githubusercontent.com/nidaa1308-arch/vp/main/ai/finetune_efficientnet.py
!python -m py_compile /content/finetune_efficientnet.py
print("Source downloaded and syntax checked.")
~~~

**Cell 2 — run in the notebook kernel (Drive authorization):**

~~~python
%run /content/finetune_efficientnet.py
~~~

The script mounts Google Drive and trains both stages. **Do not run the entire legacy \`vptraining.py\` script first**; the new script handles its own training loop.

## Required Google Drive files

The script expects the previously established Drive root:

\`/content/drive/MyDrive/VisionPark_AI/\`

**Fast path:** all three files exist:

- \`Processed_Dataset/Pilot/train.npz\`
- \`Processed_Dataset/Pilot/validation.npz\`
- \`Processed_Dataset/Pilot/test.npz\`

Each compressed NPZ must contain \`X\` (uint8 RGB arrays, dimensions N×224×224×3) and \`y\` (0/1 class labels). The code validates cached arrays before using them.

**If a pilot archive is missing**, the script builds that split from:

- \`Processed_Dataset/parking_annotations.csv\`
- \`Processed_Dataset/train_images.csv\`, \`validation_images.csv\`, \`test_images.csv\` (when present)
- \`Dataset/PKLot.tar.gz\` if the original JPEGs are no longer extracted in Colab

The PKLot archive is **large**. Preserve the Drive dataset from earlier preprocessing; GitHub is only for source code, not full datasets.

## What happens

- **Stage 1:** freeze feature extractor, train the new 2-class classifier for 5 epochs, save \`head_best.pt\`.
- **Stage 2:** restore **best head checkpoint**, unfreeze final two feature blocks and train them with a smaller learning rate for 5 epochs; save \`finetuned_best.pt\`.
- Select the better checkpoint by **validation accuracy** (validation loss breaks ties). Only then evaluate on the **test** pilot split.
- Store final checkpoint at \`Trained_Models/EfficientNet_B0_Pilot_v2/VisionPark_EfficientNet_B0_Pilot_v2.pt\`.
- Store \`history.csv\`, \`test_metrics.json\` and \`confusion_matrix.png\` under \`Results/EfficientNet_B0_Pilot_v2/\`.

## Important limitations

- No accuracy is claimed before Colab produces the metrics.
- Image-level split prevents **identical images** from crossing splits, but **nearby frames and camera sessions can still leak**. For a research-quality estimate, later evaluate with capture-day/session or camera/site-held-out splits.
- Pilot crops only support **vacant/occupied classification**. PKLot does **not** supply per-vehicle segmentation masks; YOLO segmentation is a separate component.
- This pilot is not a production-ready camera inference system.
