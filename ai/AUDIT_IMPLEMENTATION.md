# Vision Park audit implementation — first verified increment

This is **not** a claim that all items in the pasted project audit are complete.
The current GitHub repository and the referenced ZIP audit are different snapshots.
The GitHub tree inspected for this change did not contain an 11 KB checkpoint,
an old inference_bridge.py or a test.npz. Therefore the audit's claimed **98.67%**
was **not independently established** here and must not be presented as validated.

## Included in this change

- **Full-checkpoint loading:** ai/inference_bridge.py builds a 2-class EfficientNet-B0
  and loads the complete checkpoint *strictly*. It rejects classifier-only files.
- **Matching image preprocessing:** perspective warping, padding to 224 × 224,
  ImageNet normalization. Compare preview crops with the training data.
- **Real evaluation:** ai/evaluate_checkpoint.py requires actual X images and
  ground-truth y labels in test.npz, runs the checkpoint on those crops and saves
  fresh metrics plus per-example probabilities. It does not reuse saved predictions.
- **Single-image camera bridge:** ai/inference_bridge.py can load a real photo
  and calibrated slot polygons and produce per-slot predictions. Unknown/low
  confidence results are not forwarded to the existing backend.
- **Optional backend delivery:** the image CLI's --send explicitly uses
  POST /api/iot/observation with x-device-key; DEMO_MODE=true rejects writes.
  Only enable with the real MySQL configuration and a properly secured device key.
- **Research prototype fusion:** ai/occupancy_fusion.py combines fresh observations
  from multiple cameras using visibility, image quality and age weights.
  It marks strong conflicts unknown and gates status updates over consecutive
  observations. This is a separately testable prototype, **not wired to live video**.

## On your MacBook Air M4: one-block setup / verification

Open a terminal in your cloned **nidaa1308-arch/vp** folder.
Copy train.npz, validation.npz and test.npz to
\`~/VisionPark_AI/Processed_Dataset/Pilot/\` first.
The data must have X arrays of real 224×224 RGB crops and y labels (0/1).

~~~bash
git pull origin main
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision opencv-python numpy pandas matplotlib scikit-learn tqdm
python -m unittest discover -s ai/tests -v
python ai/finetune_efficientnet_local.py
python ai/evaluate_checkpoint.py \
  --checkpoint ~/VisionPark_AI/Trained_Models/EfficientNet_B0_Pilot_local/VisionPark_EfficientNet_B0_Pilot_v2.pt \
  --test ~/VisionPark_AI/Processed_Dataset/Pilot/test.npz
~~~

**Do not present any test accuracy as confirmed until the command above completes.**
For more robust experimental results, do not keep re-tuning against the same test
set. Evaluate a held-out day, session and camera/site next.

## Real-image prototype after training

1. Make a copy of ai/camera.example.json and replace example polygons with
   **actual pixel coordinates** from your camera photo. They must correspond to
   the camera image resolution and the parking slot IDs already in MySQL.
2. Run:

~~~bash
python ai/inference_bridge.py \
  --image /absolute/path/to/parking_photo.jpg \
  --config /absolute/path/to/my_camera.json \
  --checkpoint ~/VisionPark_AI/Trained_Models/EfficientNet_B0_Pilot_local/VisionPark_EfficientNet_B0_Pilot_v2.pt
~~~

This prints observed statuses/probabilities locally without contacting the backend.
Once database mode and keys are configured, add --send to explicitly submit
confident results. The current backend accepts only "free" or "occupied" and
does not persist camera_id, occlusion visibility or timestamps separately.
That schema/API enhancement remains outstanding.

## Still outstanding (requires implementation and real evidence)

- Inspect any *external* model checkpoint and dataset referenced in the audit;
  original trained weights were not provided in this repository.
- Run the training/evaluation on real images and verify confusion matrix,
  accuracy, F1, latency, performance under different lighting and occlusion.
- Validate the MySQL schema, reservation concurrency, persistence after restart,
  API authentication, deployment and Socket.IO updates on the real environment.
- Build and validate continuous live camera acquisition (video streams), monitor
  connectivity/reconnects, map overlapping slots and measure processing latency.
- Extend the backend schema for camera_id, time, visibility, confidence, source
  and freshness; integrate multi-camera reconciliation and test camera failures.
- Replace demo booking with actual authenticated reservations and verify
  dashboard statistics under live database mode.
- Perform camera/site/day-held-out experiments, prior-art review, comparison to
  single-camera methods, reproducibility analysis and final documentation.
- ESP32 work is explicitly outside this scope.

Do not call a system "research-grade complete" based only on prepared code.
