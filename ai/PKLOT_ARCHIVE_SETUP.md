# Build the pilot dataset from the original 4.56 GiB PKLot archive

The Google Drive file \`PKLot.tar.gz\` was confirmed through its metadata
(4,898,276,304 bytes). You do **not** need to extract the full archive.
\`prepare_pilot_from_archive.py\` opens the gzip TAR sequentially, reads PKLot
XMLs in pass one, samples slot annotations, and only decodes the JPEG images
selected for 6,000 balanced parking-space crops in pass two. The full tarball
is still read twice, so allow time for disk I/O. You'll need enough disk space
for the archive and the resulting small pilot arrays, but not for a 19+ GB
extracted PKLot directory.

## On Mac / VS Code

After the feature PR is merged, open a terminal inside the \`vp\` repo:

~~~bash
git pull origin main
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r ai/requirements.txt

# Put PKLot.tar.gz anywhere on your Mac that has sufficient disk space.
# Replace this example path with the actual path after downloading it:
python ai/prepare_pilot_from_archive.py \
  --archive "$HOME/Downloads/PKLot.tar.gz"

# Only once all three .npz files exist:
python ai/finetune_efficientnet_local.py
python ai/evaluate_checkpoint.py \
  --checkpoint "$HOME/VisionPark_AI/Trained_Models/EfficientNet_B0_Pilot_local/VisionPark_EfficientNet_B0_Pilot_v2.pt" \
  --test "$HOME/VisionPark_AI/Processed_Dataset/Pilot/test.npz"
~~~

## Optional: reuse previous Colab image splits

If you still have \`train_images.csv\`, \`validation_images.csv\`,
\`test_images.csv\` in Google Drive, download all **three** to
\`~/VisionPark_AI/Processed_Dataset/\` **before** running the script.
The archived JPEG names are matched to the saved Colab absolute image paths
using the canonical PKLot relative path.

When split CSVs are not present, a deterministic camera/weather/capture-day
hash split keeps frames from the same recorded day in a single partition
(70%/15%/15% by hashed groups, approximately). It is **not** equivalent to
a strict held-out entire camera/site test. Save \`pilot_manifest.json\` to
record the resulting split strategy, and don't compare it directly with metrics
from the previous image-level split.

## Common issues

- \`FileNotFoundError: archive not found\`: replace the path after \`--archive\`
  with the actual downloaded archive location.
- No archive download is needed if the tarball is already on your Mac.
- Memory: creating a full set of 6,000 RGB 224×224 images uses roughly 0.9 GB
  for the raw arrays, plus temporary candidate and compression memory.
- Already have some .npz files? Back up all three before using
  \`--overwrite\`. The script refuses accidental replacement by default.
- No confirmed training accuracy can be claimed until the actual checkpoint
  has been trained and evaluated.
- For the **toy-car ESP32-CAM prototype**, PKLot is only a baseline:
  eventually collect new toy-car observations from both camera viewpoints and
  evaluate on separate capture sessions.
