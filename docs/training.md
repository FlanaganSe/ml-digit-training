# Model Training Guide

End-to-end guide for training a custom YOLOv11n digit-tile detection model and deploying it into the Superbuilders app. Every step is included — from printing physical tiles to a working model on iPad.

---

## Table of Contents

1. [One-Time Setup](#1-one-time-setup)
2. [Prepare Physical Tiles](#2-prepare-physical-tiles)
3. [Capture Training Data](#3-capture-training-data)
4. [Extract Frames from Video](#4-extract-frames-from-video)
5. [Roboflow: Upload Images](#5-roboflow-upload-images)
6. [Roboflow: Annotate Images](#6-roboflow-annotate-images)
7. [Roboflow: Generate Dataset Version](#7-roboflow-generate-dataset-version)
8. [Roboflow: Export Dataset](#8-roboflow-export-dataset)
9. [Train the Model](#9-train-the-model)
10. [Evaluate Training Results](#10-evaluate-training-results)
11. [Export to ONNX](#11-export-to-onnx)
12. [Integrate into the App](#12-integrate-into-the-app)
13. [Test Locally in Browser](#13-test-locally-in-browser)
14. [Test on iPad / iPhone](#14-test-on-ipad--iphone)
15. [Deploy](#15-deploy)
16. [Retraining (Adding More Data)](#16-retraining-adding-more-data)

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

### 1c. Create a Roboflow account and project

1. Go to [app.roboflow.com](https://app.roboflow.com) and sign up (free tier works)
2. Click **Create New Project**
3. Project name: **digit-tiles**
4. Project type: **Object Detection**
5. Annotation group: leave default
6. Click **Create**

### 1d. Add the 10 digit classes

1. In your project, click **Classes & Tags** in the left sidebar
2. Add 10 classes, one at a time: `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`

---

## 2. Prepare Physical Tiles

Print number tiles 0–9. You need at least 2 copies of each digit (for multi-digit answers like 12, 17).

**Specifications:**
- Size: 3 × 4 inches per tile
- Font: large, bold, high contrast (dark on light background)
- Finish: matte laminate (reduces glare under camera)
- Print on cardstock or laminate for durability

You can design these in any tool (Canva, Google Slides, etc.) and print at home or at a print shop.

---

## 3. Capture Training Data

The goal is **50–100 images per digit** (500–1000 total). The fastest method is short videos, not individual photos.

### What to record

For each digit (0–9), record a **10–15 second video** on your iPhone. During each video, vary:

- **Angles** — straight on, slight tilt left, slight tilt right, slight tilt forward/back
- **Lighting** — move to different spots (overhead lamp, natural daylight, slightly dim, side-lit)
- **Distance** — close up, arm's length, roughly iPad camera distance
- **Background** — the actual surface you'll play on (table, play mat)

Also record **multi-tile compositions:**
- Place 2 tiles side by side (like "1" and "3" to form 13)
- Place 3 tiles in frame at varying distances
- These are critical — the model must detect individual digits when multiple tiles are present

Also record **5–10 seconds of empty background** (no tiles at all). These become negative examples that teach the model what "no digit" looks like.

### How to record

1. Open the iPhone Camera app
2. Record in standard 1080p video (default is fine)
3. Hold the phone roughly where the iPad camera would be during gameplay
4. Slowly move the tile around, tilt it, shift your hand — keep it natural
5. Save each video named by digit: `digit_0.MOV`, `digit_1.MOV`, ..., `digit_9.MOV`
6. Save multi-tile videos as: `multi_01.MOV`, `multi_12.MOV`, etc.
7. Save empty background video as: `background.MOV`

### Transfer videos to your Mac

**AirDrop** each video from your iPhone to your MacBook. They'll land in `~/Downloads/`.

Verify they're there:

```bash
ls ~/Downloads/digit_*.MOV
```

---

## 4. Extract Frames from Video

Each 10-second video at 2 frames per second produces ~20 images. This is much faster than taking individual photos.

```bash
cd ~/proj/digit-training

# Extract frames from all digit videos
for i in 0 1 2 3 4 5 6 7 8 9; do
  ffmpeg -i ~/Downloads/digit_$i.MOV -vf fps=2 frames/digit${i}_%04d.jpg
done

# Extract frames from multi-tile videos (if you have them)
for f in ~/Downloads/multi_*.MOV; do
  name=$(basename "$f" .MOV)
  ffmpeg -i "$f" -vf fps=2 "frames/${name}_%04d.jpg"
done

# Extract frames from background video (negative examples)
ffmpeg -i ~/Downloads/background.MOV -vf fps=1 frames/background_%04d.jpg
```

Check how many frames you got:

```bash
ls frames/*.jpg | wc -l
```

You should have 200–500+ images. Skim through them in Finder — delete any that are blurry, completely off-frame, or useless.

---

## 5. Roboflow: Upload Images

1. Go to [app.roboflow.com](https://app.roboflow.com) and open your **digit-tiles** project
2. Click **Upload Data** in the left sidebar
3. Drag the entire `~/proj/digit-training/frames/` folder from Finder into the upload zone
   - Or click **Select Files** and select all the `.jpg` files
4. Click **Save and Continue**
5. The images will appear in the **Unassigned** bucket on the Annotate page

---

## 6. Roboflow: Annotate Images

This is the most time-consuming step. You are drawing bounding boxes around every digit in every image and labeling them.

1. Click **Annotate** in the left sidebar
2. Click the **Unassigned** batch
3. Select all images → **Assign to: yourself**
4. Click into the first image to open the annotation editor

### For each image:

1. Click the **bounding box tool** (or press `B`)
2. Draw a tight rectangle around each digit tile in the image
3. Select the correct class label (`0`, `1`, `2`, ..., `9`)
4. If there are multiple tiles in the image, draw a box around each one separately
5. Press **Enter** or the **right arrow** to save and move to the next image

### Speed tips:

- The filename tells you the digit: `digit3_0012.jpg` = it's a `3`
- Use **keyboard shortcuts** to select classes quickly (check Roboflow's shortcut bar)
- For single-digit videos, every frame in that batch is the same label — you just need to adjust the box position
- For background images (no tiles), just skip them — leave them with no annotations. Roboflow treats unannotated images as negative examples.
- Use Roboflow's **Smart Polygon** or **Label Assist** if available — it can auto-suggest boxes

### When done:

All images should be in the **Dataset** column (annotated images move there automatically after approval). The **Unassigned** and **Annotating** columns should be empty.

---

## 7. Roboflow: Generate Dataset Version

1. Click **Versions** in the left sidebar
2. Click **Generate New Version**

### Configure the split:

- **Train:** 70%
- **Valid:** 20%
- **Test:** 10%

Roboflow shuffles and splits automatically.

### Configure preprocessing:

- **Auto-Orient:** On (applied by default)
- **Resize:** Stretch to **640×640** (matches YOLO input size)

### Configure augmentations:

**Enable these** (they multiply your dataset and improve generalization):

| Augmentation | Setting |
|---|---|
| Brightness | -15% to +15% |
| Exposure | -10% to +10% |
| Blur | up to 1.5px |
| Noise | up to 3% |
| Rotation | -10° to +10° |

**Do NOT enable these** (they create unrealistic digit orientations):

| Do NOT enable | Reason |
|---|---|
| Horizontal Flip | A flipped "3" is not a valid "3" |
| Vertical Flip | Upside-down digits are nonsensical |
| 90° Rotation | Kids don't rotate tiles 90° |
| Cutout | Can obscure the digit entirely |

### Generate:

Click **Generate**. Roboflow creates an augmented version of your dataset. The total image count will be 2–3× your original count.

---

## 8. Roboflow: Export Dataset

1. Click **Versions** → select your generated version
2. Click **Download Dataset**
3. Format: **YOLOv8** (this is the folder structure Ultralytics expects)
4. Click **Download zip**
5. The zip downloads to `~/Downloads/`

Extract it:

```bash
cd ~/proj/digit-training
unzip ~/Downloads/digit-tiles-*.zip -d dataset
```

Verify the structure:

```bash
ls dataset/
# Should contain: data.yaml  train/  valid/  test/

ls dataset/train/
# Should contain: images/  labels/
```

---

## 9. Train the Model

```bash
cd ~/proj/digit-training
source .venv/bin/activate

yolo detect train \
  data=dataset/data.yaml \
  model=yolo11n.pt \
  epochs=50 \
  imgsz=640 \
  device=mps
```

**What these flags mean:**

| Flag | Value | Meaning |
|---|---|---|
| `data` | `dataset/data.yaml` | Points to your exported Roboflow dataset |
| `model` | `yolo11n.pt` | Start from pretrained YOLOv11 nano (auto-downloads first time) |
| `epochs` | `50` | Number of training passes (increase to 100–150 if results are poor) |
| `imgsz` | `640` | Input resolution — matches what the app uses |
| `device` | `mps` | Use Apple Silicon GPU (Metal Performance Shaders) |

**Training takes ~15–30 minutes on M3 Pro.**

The terminal shows live progress: epoch number, loss values, and mAP metrics on each validation pass.

Results are saved to: `runs/detect/train/`

> **Note:** If you train multiple times, subsequent runs go to `runs/detect/train2/`, `train3/`, etc. The most recent is always the highest number.

---

## 10. Evaluate Training Results

### Check the metrics

Open the results plot:

```bash
open runs/detect/train/results.png
```

This shows loss curves and mAP over time. Key metrics:

| Metric | Target | Meaning |
|---|---|---|
| **mAP50** | > 0.80 | Mean Average Precision at 50% IoU — primary quality metric |
| **mAP50-95** | > 0.50 | Stricter metric across multiple IoU thresholds |
| **box_loss** | Decreasing | Bounding box regression loss — should trend down |
| **cls_loss** | Decreasing | Classification loss — should trend down |

### Visual sanity check

Run predictions on validation images to see the model in action:

```bash
yolo detect predict \
  model=runs/detect/train/weights/best.pt \
  source=dataset/valid/images
```

This saves annotated images (with bounding boxes drawn) to `runs/detect/predict/`. Open them:

```bash
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
| Confuses similar digits (6/9, 1/7) | Add more varied examples of those specific digits |
| False positives on background | Add more negative examples (empty background images) |
| Boxes are offset/wrong size | Check that annotations are tight around tiles, not sloppy |

---

## 11. Export to ONNX

Once you're satisfied with the model quality:

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

**What these flags mean:**

| Flag | Value | Meaning |
|---|---|---|
| `format` | `onnx` | Export for ONNX Runtime (used in browser via WASM) |
| `imgsz` | `640` | Must match training size |
| `opset` | `17` | ONNX operator set — 17 is compatible with ORT Web 1.24 |
| `half` | `False` | Full float32 precision (half/float16 not supported by WASM) |
| `batch` | `1` | Single image inference (the app sends one frame at a time) |

---

## 12. Integrate into the App

Copy the exported model into the Superbuilders project:

```bash
cp runs/detect/train/weights/best.onnx ~/proj/superbuilders/public/models/digit-tiles.onnx
```

The app loads this model from `public/models/`. The inference worker reads class count from the ONNX output tensor dimensions automatically — no code change needed when swapping models.

**Expected output tensor shape:** `[1, 14, 8400]` — where 14 = 4 box coordinates + 10 class scores, and 8400 = number of anchor points at 640×640.

### Model size check

```bash
ls -lh ~/proj/superbuilders/public/models/digit-tiles.onnx
```

Should be ~10–12 MB. The PWA caches models up to 30 MB, so this is fine.

---

## 13. Test Locally in Browser

```bash
cd ~/proj/superbuilders
pnpm dev
```

Open `https://localhost:5173?debug=true` in Chrome for initial debugging. **Important:** Final validation must happen in Safari/WebKit — Chrome's ONNX WASM behavior can differ from Safari, and the target platform is iPad Safari.

### What to check:

1. **Network tab:** Does `digit-tiles.onnx` load with status 200 (or 304)?
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

## 14. Test on iPad / iPhone

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

## 15. Deploy

Once everything works locally and on device:

```bash
cd ~/proj/superbuilders
pnpm build    # Typecheck + production build
pnpm preview  # Verify the production build locally
```

Push to GitHub and the CI/CD pipeline deploys to Cloudflare Pages automatically.

The PWA service worker caches `digit-tiles.onnx` with a CacheFirst strategy, so after the first load on a device, the model loads from cache instantly on subsequent visits.

---

## 16. Retraining (Adding More Data)

When you need to improve the model (misdetections, new tile designs, different lighting conditions):

### Add new training data

1. Record new videos of the problem cases
2. Extract frames: `ffmpeg -i new_video.MOV -vf fps=2 frames/new_%04d.jpg`
3. Upload to the same Roboflow project (Upload Data → drag frames)
4. Annotate the new images
5. Generate a **new version** in Roboflow (it includes all previous + new images)
6. Export as YOLOv8 and download

### Retrain

```bash
cd ~/proj/digit-training
source .venv/bin/activate

# Remove old dataset, extract new one
rm -rf dataset
unzip ~/Downloads/digit-tiles-*.zip -d dataset

# Train again (starts from pretrained weights, not your previous run)
yolo detect train \
  data=dataset/data.yaml \
  model=yolo11n.pt \
  epochs=50 \
  imgsz=640 \
  device=mps

# Evaluate
yolo detect predict \
  model=runs/detect/train*/weights/best.pt \
  source=dataset/valid/images

# Export and deploy
yolo export \
  model=runs/detect/train*/weights/best.pt \
  format=onnx \
  imgsz=640 \
  opset=17 \
  half=False \
  batch=1

cp runs/detect/train*/weights/best.onnx ~/proj/superbuilders/public/models/digit-tiles.onnx
```

> **Note:** Each training run creates a new `runs/detect/trainN/` directory. Use the latest one, or specify the exact path.

### Tips for iterative improvement

- **Confusing 6 and 9?** Add more examples of each at various rotations
- **Missing tiles at distance?** Record videos from farther away
- **False positives on hands/table?** Add more background-only images
- **Poor in certain lighting?** Record in that specific lighting condition
- Focus on the failure cases — don't just add more of what already works

---

## Quick Reference: Full Command Sequence

For when you just need the commands without explanation:

```bash
# === EXTRACT FRAMES ===
cd ~/proj/digit-training
for i in 0 1 2 3 4 5 6 7 8 9; do
  ffmpeg -i ~/Downloads/digit_$i.MOV -vf fps=2 frames/digit${i}_%04d.jpg
done

# === AFTER ROBOFLOW ANNOTATION + EXPORT ===
unzip ~/Downloads/digit-tiles-*.zip -d dataset

# === TRAIN ===
source .venv/bin/activate
yolo detect train data=dataset/data.yaml model=yolo11n.pt epochs=50 imgsz=640 device=mps

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
