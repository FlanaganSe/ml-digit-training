# Model Training Guide

End-to-end guide for training a custom YOLOv11n tile detection model and deploying it into the Superbuilders app. Covers both digit-only (10-class) and full character (36-class: digits 0-9 + letters A-Z) models.

---

## Current Model

**train3** — YOLOv11n, 36 classes (0-9 + A-Z), trained 150 epochs on Kaggle P100.

| Metric | Value |
|---|---|
| mAP50 | 0.995 |
| mAP50-95 | 0.973 |
| Precision | 0.993 |
| Recall | 0.998 |

ONNX: 10.6 MB, output shape `[1, 40, 8400]`, opset 17, float32. Deployed to `~/proj/superbuilders/public/models/digit-tiles.onnx`.

### Training curves

![Training results](train3-results.png)

### Confusion matrix (validation set)

![Confusion matrix](train3-confusion-matrix.png)

---

## Table of Contents

1. [One-Time Setup](#1-one-time-setup)
2. [Prepare Physical Tiles](#2-prepare-physical-tiles)
3. [Capture Training Data](#3-capture-training-data)
4. [Extract Frames from Video](#4-extract-frames-from-video)
5. [Filter Frames](#5-filter-frames)
6. [Annotate Images](#6-annotate-images)
7. [Roboflow: Upload, Review, Version](#7-roboflow-upload-review-version)
8. [Train the Model](#8-train-the-model)
9. [Evaluate Training Results](#9-evaluate-training-results)
10. [Export to ONNX](#10-export-to-onnx)
11. [Integrate into the App](#11-integrate-into-the-app)
12. [Test Locally in Browser](#12-test-locally-in-browser)
13. [Test on iPad / iPhone](#13-test-on-ipad--iphone)
14. [Deploy](#14-deploy)
15. [Retraining (Adding More Data)](#15-retraining-adding-more-data)

---

## 1. One-Time Setup

These steps only need to be done once.

### 1a. Install system dependencies

```bash
brew install ffmpeg
brew install cloudflared
```

### 1b. Create a Python training environment

Keep this separate from the app project — it's a different toolchain.

```bash
mkdir -p ~/proj/digit-training/frames
cd ~/proj/digit-training
python3 -m venv .venv
source .venv/bin/activate
pip install ultralytics albumentations
```

Verify it works:

```bash
yolo version
```

### 1c. Install annotation pipeline dependencies

```bash
cd ~/proj/digit-training
source .venv/bin/activate
pip install openai imagehash supervision roboflow python-dotenv
```

### 1d. Create a Roboflow account and project

1. Go to [app.roboflow.com](https://app.roboflow.com) and sign up (free tier works)
2. Click **Create New Project**
3. Project name: **digit-tiles**
4. Project type: **Object Detection**
5. Annotation group: leave default
6. Click **Create**

### 1e. Add classes

For digit-only (10-class): `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`

For full character (36-class): digits 0-9 plus `A` through `Z` (class IDs 10-35).

---

## 2. Prepare Physical Tiles

Print number tiles 0–9 (and letter tiles A-Z if training 36-class). You need at least 2 copies of each digit (for multi-digit answers like 12, 17).

**Specifications:**
- Size: 3 × 4 inches per tile
- Font: large, bold, high contrast (dark on light background)
- Finish: matte laminate (reduces glare under camera)
- Print on cardstock or laminate for durability

You can design these in any tool (Canva, Google Slides, etc.) and print at home or at a print shop.

---

## 3. Capture Training Data

### Capture device

**Use the iPad that runs the app as the primary capture device.** The camera module, focal length, color profile, and auto-exposure behavior of the deployment device are what the model needs to learn. iPhone footage is useful as supplementary data for domain diversity, but should be <30% of total.

### Target volume

| Class count | Target per class | Total raw frames |
|---|---|---|
| 10 (digits) | 150–300 | 1500–3000 |
| 36 (digits + letters) | 100–200 | 3600–7200 |

### Scene composition targets

| Bucket | Description | % of data |
|---|---|---|
| Single clean tile | One tile, centered, clear | 40% |
| Multi-tile | 2-5 tiles with gaps, touching pairs, separated pairs | 40% |
| Hard negatives | Empty surface, hands, shadows, glare, clutter | 10% |
| Hard positives | Confusable pairs (6/9, 0/O, 1/I), extreme lighting, edge of frame, far distance | 10% |

### Filming protocol

Record **many short clips** (10-15 seconds each), not one long video sweep. Keep lighting stable within a clip and change lighting between clips. This gives clean session boundaries for grouped splits.

**Lighting variety** (record each bucket under all of these, in separate clips):
1. Bright overhead (ceiling light directly above)
2. Natural daylight (near window)
3. Warm/dim (lamp, evening)
4. Side-lit (strong directional light creating shadows)

**Background variety**: Record on 3+ surfaces — the actual play surface, a white surface, a dark surface. The model can learn background shortcuts if all examples of a class are on one surface.

**Confusable-class sessions**: Dedicate filming to confusable pairs placed next to each other:
- 6 and 9 (underline marks visible)
- 0 and O (if training 36-class)
- 1 and I, 8 and B, 5 and S

### Naming convention

Name videos by content for split grouping later:

```
single_digit0_overhead.MOV      # single tile, digit 0, overhead light
single_digit0_daylight.MOV      # single tile, digit 0, daylight
multi_digits_overhead.MOV       # multi-tile digits, overhead light
multi_letters_dim.MOV           # multi-tile letters, dim light
negative_empty_overhead.MOV     # empty surface, overhead light
hard_6v9_sidelit.MOV            # confusable pair, side-lit
```

### Transfer videos to your Mac

**AirDrop** each video from your iPad to your MacBook. They'll land in `~/Downloads/`.

---

## 4. Extract Frames from Video

Each 10-second video at 2 frames per second produces ~20 images.

```bash
cd ~/proj/digit-training

# Extract from all videos at 2fps
for f in ~/Downloads/*.MOV; do
  name=$(basename "$f" .MOV)
  ffmpeg -i "$f" -vf fps=2 "frames/${name}_%04d.jpg"
done

# For negative/background videos, 1fps is enough
for f in ~/Downloads/negative_*.MOV; do
  name=$(basename "$f" .MOV)
  ffmpeg -i "$f" -vf fps=1 "frames/${name}_%04d.jpg"
done
```

Check how many frames you got:

```bash
ls frames/*.jpg | wc -l
```

---

## 5. Filter Frames

Use the annotation pipeline to remove blurry and near-duplicate frames:

```bash
cd ~/proj/digit-training
source .venv/bin/activate
python -m scripts.filter
```

This uses Laplacian variance (threshold=3.0) for blur detection and perceptual hashing (hamming distance=8) for deduplication within each video prefix. See `scripts/filter.py` for details.

---

## 6. Annotate Images

### Automated annotation with Gemini

The annotation pipeline uses Gemini Flash Lite via OpenRouter to auto-annotate frames. Set up your API key:

```bash
echo "OPENROUTER_API_KEY=your_key_here" >> .env
```

Run the pipeline scripts in order:
1. `python -m scripts.filter` — blur + dedup (moves rejected frames to `frames/rejected/`)
2. `python -m scripts.annotate` — Gemini annotation (saves results to `auto_labels/batch/`)
3. `python -m scripts.convert` — Gemini JSON → YOLO label format
4. `python -m scripts.qa` — visual QA + class distribution
5. `python -m scripts.upload` — upload images + labels to Roboflow

See `scripts/config.py` for the 36-class mapping, annotation prompt, and thresholds.

### Manual annotation in Roboflow

Alternatively, annotate directly in Roboflow's UI — see `scripts/upload.py` for batch upload with pre-computed annotations (`is_prediction=True` for review mode).

---

## 7. Roboflow: Upload, Review, Version

### Upload

```bash
echo "ROBOFLOW_API_KEY=your_key_here" >> .env
python -m scripts.upload
```

### Review annotations in Roboflow UI

Auto-annotations appear as suggestions. Accept correct ones, fix incorrect ones.

### Generate dataset version

**CRITICAL SETTINGS:**

#### Split: grouped by video source, NOT random

Roboflow's random split leaks near-duplicate frames across train/val/test (sequential frames from the same video are nearly identical). Instead:
- Assign entire video clips to splits manually in Roboflow
- Or split locally before uploading
- Target: 70% train / 20% valid / 10% test
- Ensure val/test contain frames from multiple video sources and lighting conditions
- Ensure val/test contain all class types you're training on

#### Preprocessing

- **Auto-Orient:** On
- **Resize:** **Fit (white edges)** to 640×640 — **NOT Stretch**. Stretching 9:16 portrait frames to 1:1 distorts character proportions.

#### Augmentations (training set only)

**Enable these:**

| Augmentation | Setting |
|---|---|
| Brightness | -15% to +15% |
| Exposure | -10% to +10% |
| Blur | up to 1.5px |
| Noise | up to 3% |
| Rotation | -10° to +10° |

**Do NOT enable these** (they create invalid character representations):

| Do NOT enable | Reason |
|---|---|
| Horizontal Flip | A flipped "3", "7", "J", "Z" etc. are not valid |
| Vertical Flip | Upside-down characters are nonsensical |
| 90° Rotation | Kids don't rotate tiles 90° |
| Cutout | Can obscure the character entirely |

**Ensure augmentations are applied to training images only**, not validation or test. Augmented val/test sets inflate metrics.

**Why Roboflow augmentation is required:** With 36 classes and ~1200 raw images (~33 per class), the dataset is too small for effective training without augmentation. A 3x augmentation multiplier produces ~3600-5000 training images, which is sufficient. YOLO also applies its own online augmentation during training (mosaic, HSV, scale, etc.) — this supplements Roboflow augmentation but does not replace it. Both are needed.

### Export

1. Format: **YOLOv8** (the folder structure Ultralytics expects)
2. Download zip and extract:

```bash
cd ~/proj/digit-training
rm -rf dataset
unzip ~/Downloads/digit-tiles-*.zip -d dataset
```

Verify the structure:

```bash
ls dataset/
# Should contain: data.yaml  train/  valid/  test/
```

---

## 8. Train the Model

```bash
cd ~/proj/digit-training
source .venv/bin/activate

yolo detect train \
  data=dataset/data.yaml \
  model=yolo11n.pt \
  epochs=100 \
  imgsz=640 \
  device=mps \
  fliplr=0.0 \
  flipud=0.0 \
  degrees=10 \
  hsv_v=0.5 \
  close_mosaic=10
```

**What these flags mean:**

| Flag | Value | Meaning |
|---|---|---|
| `data` | `dataset/data.yaml` | Points to your exported Roboflow dataset |
| `model` | `yolo11n.pt` | Start from pretrained YOLOv11 nano (auto-downloads first time) |
| `epochs` | `100` | Number of training passes. Check loss curves — increase if still improving. |
| `imgsz` | `640` | Input resolution — matches what the app uses |
| `device` | `mps` | Use Apple Silicon GPU (Metal Performance Shaders) |
| `fliplr` | `0.0` | **Disable horizontal flip.** YOLO default is 0.5 — flipped characters are invalid training data. |
| `flipud` | `0.0` | Disable vertical flip (already default, but explicit for safety) |
| `degrees` | `10` | Online rotation augmentation ±10° (applied dynamically each epoch, unlike Roboflow's static copies) |
| `hsv_v` | `0.5` | Brightness variation ±50% for lighting robustness (default 0.4) |
| `close_mosaic` | `10` | Disable mosaic augmentation for last 10 epochs (improves final precision) |

### Training time estimates (M3 Pro)

| Dataset size | Classes | Time per epoch | 50 epochs | 100 epochs |
|---|---|---|---|---|
| ~500 images | 10 | ~30s | ~25 min | ~50 min |
| ~3700 images | 36 | ~4.5 min | ~3.8 hours | ~7.5 hours |

Results are saved to: `runs/detect/train/` (or `train2/`, `train3/`, etc.)

---

## 9. Evaluate Training Results

### Check the metrics

```bash
open runs/detect/train/results.png
```

Key metrics:

| Metric | Target | Meaning |
|---|---|---|
| **mAP50** | > 0.85 | Mean Average Precision at 50% IoU — primary quality metric |
| **mAP50-95** | > 0.50 | Stricter metric across multiple IoU thresholds |
| **box_loss** | Decreasing | Bounding box regression loss — should trend down |
| **cls_loss** | Decreasing | Classification loss — should trend down |

**Important:** These metrics are only meaningful if your validation set is properly constructed (grouped split, representative of all classes, not augmented). If val only contains a few classes or one surface, the numbers lie.

### Visual sanity check

```bash
yolo detect predict \
  model=runs/detect/train/weights/best.pt \
  source=dataset/valid/images
open runs/detect/predict/
```

**What to look for:**
- Are bounding boxes on the right tiles?
- Are the class labels correct? (e.g., a "7" labeled as `7`, not `1`)
- Are there false positives (boxes on non-digit things)?
- Are there missed detections (tiles with no box)?

### If results are bad

| Problem | Fix |
|---|---|
| Low mAP (< 0.60) | More training data, more epochs (100–150), check annotations for errors |
| Confuses similar characters (6/9, 1/I, 0/O) | Add more varied examples of those specific characters, especially confusable pairs side by side |
| False positives on background | Add more negative examples (empty background, hands, shadows) |
| Boxes are offset/wrong size | Check that annotations are tight around tiles |
| Good on one surface, bad on another | Need more background/surface diversity in training data |
| Good in one lighting, bad in another | Need more lighting diversity across filming sessions |

---

## 10. Export to ONNX

```bash
cd ~/proj/digit-training
source .venv/bin/activate

yolo export \
  model=runs/detect/train/weights/best.pt \
  format=onnx \
  imgsz=640 \
  opset=17 \
  half=False \
  batch=1
```

This creates `runs/detect/train/weights/best.onnx` (~10 MB).

| Flag | Value | Meaning |
|---|---|---|
| `format` | `onnx` | Export for ONNX Runtime (used in browser via WASM) |
| `imgsz` | `640` | Must match training size |
| `opset` | `17` | ONNX operator set — 17 is compatible with ORT Web 1.24 |
| `half` | `False` | Full float32 precision (half/float16 not supported by WASM) |
| `batch` | `1` | Single image inference (the app sends one frame at a time) |

**Expected output tensor shapes:**
- 10-class: `[1, 14, 8400]` — 4 box coords + 10 class scores
- 36-class: `[1, 40, 8400]` — 4 box coords + 36 class scores

---

## 11. Integrate into the App

Copy the ONNX model to the app's public directory:

```bash
cp runs/detect/train/weights/best.onnx ~/proj/superbuilders/public/models/digit-tiles.onnx
```

The app uses a **StaleWhileRevalidate** caching strategy (not CacheFirst), so the browser fetches a fresh copy in the background on the next load. No versioned filename is needed — overwriting `digit-tiles.onnx` is sufficient.

The inference worker reads class count from the ONNX output tensor dimensions automatically — no other code change needed when swapping models.

### Model size check

```bash
ls -lh ~/proj/superbuilders/public/models/digit-tiles.onnx
```

Should be ~10–12 MB. The PWA caches models up to 30 MB, so this is fine.

---

## 12. Test Locally in Browser

```bash
cd ~/proj/superbuilders
pnpm dev
```

Open `https://localhost:5173?debug=true` in Chrome for initial debugging. **Important:** Final validation must happen in Safari/WebKit — Chrome's ONNX WASM behavior can differ from Safari, and the target platform is iPad Safari.

### What to check:

1. **Network tab:** Does the model file load with status 200 (or 304)?
2. **Network tab:** Does `ort-wasm-simd-threaded.mjs` and `.wasm` load without 404?
3. **Console:** No red errors about WASM, SharedArrayBuffer, or ORT?
4. **Debug HUD** (bottom-left overlay): Does it show model loaded, inference latency, detections?

### Test with camera:

1. Tap the start/camera button to grant camera permission
2. Hold a digit tile in front of your webcam
3. The debug HUD should show `detections: 1` (or however many tiles are visible)
4. The detected digit should register as an answer in the game

### Test without camera:

Add `?recognition=mock` to the URL to use keyboard input instead. Type digit keys to simulate tile detection.

---

## 13. Test on iPad / iPhone

### Start the tunnel

```bash
# Terminal 1 — dev server
cd ~/proj/superbuilders
pnpm dev

# Terminal 2 — public HTTPS tunnel
cloudflared tunnel --url https://localhost:5173
```

Cloudflared prints a URL like `https://some-random-words.trycloudflare.com`. No account needed.

### On your device

1. Open Safari on your iPad/iPhone
2. Go to the tunnel URL with `?debug=true` appended
3. Tap to start — grant camera permission when prompted
4. Hold digit tiles in front of the camera
5. Watch the debug HUD for detections and latency

### Remote debugging (optional but useful)

To see the console/network from your phone on your Mac:

1. **On iPhone/iPad:** Settings → Safari → Advanced → turn on **Web Inspector**
2. **On Mac:** Open Safari → **Develop** menu → select your device → select the page
3. You now have full DevTools for the phone's Safari

### What to check on device:

| Check | Pass | Fail |
|---|---|---|
| Model downloads | Network shows 200/304 for `.onnx` | 404 or timeout |
| WASM loads | No console errors | Errors about WASM or SharedArrayBuffer |
| Inference runs | Debug HUD shows latency < 120ms | Latency > 120ms or crashes |
| Detection works | Tiles are recognized, game responds | No detections or wrong digits |

**If inference latency > 120ms on iPad:** Consider retraining at `imgsz=320` (both training and export) for faster inference at the cost of some accuracy.

---

## 14. Deploy

Once everything works locally and on device:

```bash
cd ~/proj/superbuilders
pnpm build    # Typecheck + production build
pnpm preview  # Verify the production build locally
```

Push to GitHub and the CI/CD pipeline deploys to Cloudflare Pages automatically.

The PWA service worker caches the model with a StaleWhileRevalidate strategy — cached devices serve the existing model instantly while fetching the updated version in the background. The new model will be used on the next visit.

---

## 15. Retraining (Adding More Data)

When you need to improve the model (misdetections, new tile designs, different lighting conditions):

### Add new training data

1. Record new videos following the capture protocol in §3
2. Extract frames: `ffmpeg -i new_video.MOV -vf fps=2 frames/new_%04d.jpg`
3. Run the filter pipeline: `python -m scripts.filter`
4. Annotate (automated or manual)
5. Upload to the same Roboflow project
6. Generate a **new version** in Roboflow with proper grouped splits
7. Export as YOLOv8 and download

### Retrain

```bash
cd ~/proj/digit-training
source .venv/bin/activate

# Remove old dataset, extract new one
rm -rf dataset
unzip ~/Downloads/digit-tiles-*.zip -d dataset

# Train with corrected args
yolo detect train \
  data=dataset/data.yaml \
  model=yolo11n.pt \
  epochs=100 \
  imgsz=640 \
  device=mps \
  fliplr=0.0 \
  flipud=0.0 \
  degrees=10 \
  hsv_v=0.5 \
  close_mosaic=10

# Evaluate
yolo detect predict \
  model=runs/detect/train/weights/best.pt \
  source=dataset/valid/images

# Export and deploy
yolo export \
  model=runs/detect/train/weights/best.pt \
  format=onnx \
  imgsz=640 \
  opset=17 \
  half=False \
  batch=1

cp runs/detect/train/weights/best.onnx ~/proj/superbuilders/public/models/digit-tiles.onnx
```

### Tips for iterative improvement

- **Confusing 6 and 9?** Add more examples of each, filmed next to each other
- **Missing tiles at distance?** Record videos from farther away
- **False positives on hands/table?** Add more negative examples (empty background, hands, shadows)
- **Poor in certain lighting?** Record in that specific lighting condition
- **Good on one surface, bad on another?** Record on multiple surfaces
- Focus on the failure cases — don't just add more of what already works
- Check the confusion matrix (`runs/detect/train/confusion_matrix.png`) to identify which classes are being confused

---

## Known Pitfalls

| Pitfall | What happens | How to avoid |
|---|---|---|
| `fliplr` not set to 0.0 | YOLO default is 0.5 — mirrored "3", "7", "9", "J", "Z" corrupt training | Always pass `fliplr=0.0` explicitly |
| Roboflow resize set to "Stretch" | 9:16 portrait frames squashed to 1:1, distorting characters | Use "Fit (white edges)" |
| Random Roboflow split | Sequential video frames leak across train/val/test, inflating metrics | Split by video source/session |
| Augmented val/test sets | Near-duplicate augmented copies inflate mAP | Apply augmentation to training set only |
| Model not updating on device | Browser serves cached copy until next visit | StaleWhileRevalidate fetches update in background — reload once more |
| Training on iPhone, deploying on iPad | Domain gap from different cameras | Film primarily on the deployment device |
| Insufficient negatives | False positives on hands, surfaces, clutter | Include 10% hard negatives |
| Severe class imbalance | Rare classes have low recall | Balance with targeted filming |

---

## Quick Reference: Full Command Sequence

```bash
# === EXTRACT FRAMES ===
cd ~/proj/digit-training
for f in ~/Downloads/*.MOV; do
  name=$(basename "$f" .MOV)
  ffmpeg -i "$f" -vf fps=2 "frames/${name}_%04d.jpg"
done

# === FILTER ===
source .venv/bin/activate
python -m scripts.filter

# === ANNOTATE (automated) ===
# Ensure OPENROUTER_API_KEY is in .env
python -m scripts.annotate
python -m scripts.convert
python -m scripts.qa

# === UPLOAD TO ROBOFLOW ===
# Ensure ROBOFLOW_API_KEY is in .env
python -m scripts.upload

# === AFTER ROBOFLOW REVIEW + EXPORT ===
rm -rf dataset
unzip ~/Downloads/digit-tiles-*.zip -d dataset

# === TRAIN ===
yolo detect train \
  data=dataset/data.yaml \
  model=yolo11n.pt \
  epochs=100 \
  imgsz=640 \
  device=mps \
  fliplr=0.0 \
  flipud=0.0 \
  degrees=10 \
  hsv_v=0.5 \
  close_mosaic=10

# === EVALUATE ===
yolo detect predict model=runs/detect/train/weights/best.pt source=dataset/valid/images
open runs/detect/predict/
open runs/detect/train/results.png

# === EXPORT + DEPLOY ===
yolo export model=runs/detect/train/weights/best.pt format=onnx imgsz=640 opset=17 half=False batch=1
cp runs/detect/train/weights/best.onnx ~/proj/superbuilders/public/models/digit-tiles.onnx

# === TEST ON DEVICE ===
cd ~/proj/superbuilders
pnpm dev
# (in another terminal)
cloudflared tunnel --url https://localhost:5173
# Open tunnel URL on iPad with ?debug=true
```
