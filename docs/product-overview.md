# Product Overview

## What This Is

An end-to-end machine learning training pipeline that produces a YOLOv11n object detection model for recognizing 36 alphanumeric characters (digits 0-9 and letters A-Z) on physical tiles. The trained model exports to ONNX (~10 MB) and runs in-browser via ONNX Runtime WASM, powering real-time tile detection in [TileSight](https://github.com/FlanaganSe/superbuilders-numberGame) — a demo iPad app where kids place physical tiles to answer math questions. TileSight is functional but was never fully polished for production; the training pipeline in this repository represents the more complete and reusable work.

This repository is not the app itself. It is the **training toolkit** — a set of Python scripts and documentation that automate the full lifecycle from raw video capture to a deployable ONNX model: frame extraction, quality filtering, AI-powered annotation, dataset management, stratified splitting, model training, evaluation, and export.

## Why It Was Built

Manual annotation is slow. Annotating 100 tile images by hand in Roboflow takes roughly an hour. This pipeline uses Gemini Flash Lite (via OpenRouter) with structured JSON output to annotate ~2000 images in under 30 minutes at equivalent accuracy. The pipeline also solves several non-obvious ML pitfalls that arise when training character detection models — most critically, data leakage from naive train/val splits on sequential video frames, and corrupted training data from horizontal flip augmentation on asymmetric characters.

## Stack

| Layer | Technology | Role |
|---|---|---|
| Detection model | [YOLOv11n](https://docs.ultralytics.com/) (Ultralytics) | 2.6M parameter nano model, 640×640 input, pretrained on COCO |
| Annotation | [Gemini Flash Lite](https://ai.google.dev/) via [OpenRouter](https://openrouter.ai/) | Structured JSON bounding box annotation with provider-pinned routing |
| Dataset management | [Roboflow](https://roboflow.com/) | Annotation review, preprocessing, augmentation (4x multiplier), versioned export |
| Frame filtering | OpenCV Laplacian variance + [imagehash](https://github.com/JohannesBuchner/imagehash) | Blur detection and perceptual hash deduplication |
| Training compute | [Kaggle P100 GPU](https://www.kaggle.com/) (primary) or Apple Silicon MPS (fallback) | ~3.3 hours on P100 vs ~11 hours on M3 Pro for 150 epochs |
| Inference runtime | [ONNX Runtime Web](https://onnxruntime.ai/) (WASM) | Browser-based inference on iPad Safari, <120ms per frame |
| Deployment | Cloudflare Pages | Hosts the TileSight PWA; model cached via StaleWhileRevalidate |
| Language | Python 3.14 | All pipeline scripts |
| Video processing | ffmpeg | Frame extraction from iPad video at 2 fps |

### Python dependencies

**Training:** `ultralytics`, `albumentations`

**Annotation pipeline** (`scripts/requirements.txt`): `openai` (OpenRouter SDK), `imagehash`, `supervision`, `roboflow`, `python-dotenv`

## Architecture

The pipeline is a linear sequence of standalone CLI scripts, each reading the previous step's output. There is no orchestration framework — scripts are run manually in order, with Roboflow UI as a manual checkpoint between annotation and training.

```
iPad video capture
    ↓
ffmpeg frame extraction (2 fps)
    ↓
scripts/filter.py — blur detection + perceptual hash dedup
    ↓
scripts/annotate.py — Gemini Flash Lite structured annotation
    ↓
scripts/convert.py — Gemini JSON → YOLO label format
    ↓
scripts/qa.py — automated QA checks + visual validation
    ↓
scripts/upload.py — batch upload to Roboflow
    ↓
[Roboflow UI] — manual review, preprocessing, augmentation, versioned export
    ↓
scripts/split.py — stratified grouped split by video prefix
    ↓
YOLO training (Kaggle P100 or local MPS)
    ↓
ONNX export → deployed to TileSight app
```

### Why this architecture

Each script is a pure Python module runnable via `python -m scripts.<name>`. No DAG runner, no config files beyond `.env` — the pipeline is small enough that explicit sequential execution is simpler and more debuggable than a framework. The manual Roboflow checkpoint is intentional: annotation review is a human judgment step where you fix misclassified labels and misaligned bounding boxes before they enter training.

## Directory Structure

```
digit-training/
├── scripts/                        # Core pipeline (7 modules, ~1700 LOC)
│   ├── config.py                   #   Class definitions, API config, annotation prompt, thresholds
│   ├── filter.py                   #   Laplacian blur detection + perceptual hash dedup
│   ├── annotate.py                 #   Gemini annotation via OpenRouter (structured JSON)
│   ├── convert.py                  #   Gemini [y_min,x_min,y_max,x_max] → YOLO [cx,cy,w,h]
│   ├── qa.py                       #   5 automated QA checks + bbox visualization
│   ├── upload.py                   #   Roboflow batch upload with labelmap generation
│   ├── split.py                    #   3-phase stratified grouped split algorithm
│   └── requirements.txt            #   Pipeline-specific dependencies
├── docs/
│   ├── training.md                 #   Comprehensive 15-section training guide (~730 lines)
│   ├── product-overview.md         #   This file
│   ├── train3-results.png          #   Training loss/mAP curves over 150 epochs
│   └── train3-confusion-matrix.png #   Normalized 36×36 class confusion matrix
├── kaggle/
│   └── train-digit-tiles.ipynb     #   Ready-to-run Kaggle P100 training notebook
├── frames/                         #   Extracted video frames (git-ignored)
│   └── rejected/                   #   Blurry/duplicate frames moved here by filter.py
├── auto_labels/                    #   Annotation pipeline outputs (git-ignored)
│   └── batch/
│       ├── raw_detections.json     #     Gemini raw structured JSON
│       ├── labels/                 #     YOLO .txt label files
│       └── qa/                     #     QA images with drawn bounding boxes
├── dataset/                        #   Roboflow-exported YOLOv8 dataset (git-ignored)
│   ├── data.yaml                   #     YOLO dataset config (36 classes)
│   ├── train/                      #     ~3000-5000 images (after augmentation + split)
│   ├── valid/                      #     ~250-850 images
│   └── test/                       #     ~130-420 images
├── runs/                           #   YOLO training outputs (git-ignored)
│   └── detect/
│       ├── train/, train2/, train3/#     Training run artifacts (weights, metrics, plots)
│       ├── val/, val2/, val3/      #     Validation outputs
│       └── predict/                #     Inference test outputs
├── .env.example                    #   API key template (OPENROUTER_API_KEY, ROBOFLOW_API_KEY)
└── .gitignore                      #   Excludes: .env, frames/, auto_labels/, dataset/, runs/, *.pt, *.onnx
```

**What's tracked in git:** `scripts/`, `docs/`, `kaggle/`, `README.md`, `.env.example`, `.gitignore`

**What's git-ignored (local/generated):** `frames/`, `auto_labels/`, `dataset/`, `runs/`, model weights (`.pt`, `.onnx`), `.env`, `.venv/`

## Core Concepts

### 36-class character set

Classes 0-9 map to digits, 10-35 map to letters A-Z. Defined in `scripts/config.py` as `CLASS_NAMES` and `CLASS_MAP`. Every script in the pipeline references this single source of truth.

### Physical tiles as training subjects

The model detects printed tiles (3×4 inches, bold black characters on white background, matte laminate). Training data is captured by filming these tiles on various surfaces under varied lighting using the iPad that runs the app — so the model learns the exact camera characteristics of the deployment device.

### Video prefix grouping

Each video file produces a series of sequential frames (e.g., `IMG_0118_0001.jpg` through `IMG_0118_0042.jpg`). These frames are near-identical and must stay in the same train/val/test split to avoid data leakage. The video prefix (e.g., `IMG_0118`) is the grouping key used throughout the pipeline — in deduplication (`filter.py`), cross-frame QA checks (`qa.py`), and stratified splitting (`split.py`).

### Gemini structured output

Annotation uses Gemini's structured JSON output mode (via OpenRouter's `json_schema` response format with `strict: true`). Each detection is `{label, box_2d: [y_min, x_min, y_max, x_max]}` with coordinates normalized to a 0-1000 range. The `convert.py` script transforms these to YOLO's center-based `[class_id, cx, cy, w, h]` format normalized to 0-1.

## Pipeline Scripts in Detail

### `config.py` — Configuration hub

Centralizes the 36-class mapping, OpenRouter API settings (model: `google/gemini-3.1-flash-lite-preview`, provider-pinned to Google, temperature 0.1), the structured detection schema, the annotation prompt, and thresholds for blur detection (Laplacian variance < 1.0), deduplication (hamming distance > 8), and YOLO confidence bands (0.70 high, 0.25 low).

### `filter.py` — Quality gate

Two-stage filter: (1) Laplacian variance blur detection rejects frames below threshold 1.0 (recalibrated for white tile surfaces which naturally have lower variance), (2) perceptual hash deduplication within each video prefix removes near-identical sequential frames (hamming distance threshold 8). Rejected frames are moved to `frames/rejected/` rather than deleted. ~3.5% rejection rate on typical footage (32 blurry + 15 duplicates from ~1340 frames).

### `annotate.py` — AI annotation

Calls Gemini Flash Lite via OpenRouter for each frame with a base64-encoded image and the annotation prompt. Returns a list of `Detection` dataclass instances (frozen, with computed `class_id` and `is_valid` properties). Rate-limited at 0.15s between calls. API errors are tracked separately from empty detections (a critical distinction — an API failure is not the same as a frame with no tiles). Failed frames get one retry with 2s backoff, then are excluded from results.

### `convert.py` — Format translation

Pure coordinate math: Gemini's `[y_min, x_min, y_max, x_max]` (0-1000 range) → YOLO's `[class_id, cx, cy, w, h]` (0-1 normalized). Writes one `.txt` file per frame in YOLO label format.

### `qa.py` — Automated validation

Runs 5 checks against the annotation results:
1. **Class distribution** — counts per class, flags empty frames
2. **Cross-frame consistency** — flags detection count jumps (±2+) and label changes between sequential frames within a video
3. **Bounding box geometry** — flags aspect ratios outside 0.3-4.0, sizes outside 5-60% of frame, coordinates touching edges
4. **Class distribution by video prefix** — flags videos where the dominant class has <70% share (suggesting misclassification)
5. **Annotation errors** — reports frames that failed the annotation API

Also draws green bounding boxes + labels on each frame and saves to `qa/` for manual visual review.

### `upload.py` — Roboflow ingestion

Uploads image + label pairs to the Roboflow project (`seans-workspace-zsmup/digital-tiles`) using `is_prediction=False`. This is critical: `True` routes annotations to review jobs that are excluded from dataset version generation.

### `split.py` — Stratified grouped splitting

The most algorithmically complex script. Replaces Roboflow's random split (which leaks near-duplicate video frames across splits) with a 3-phase grouped stratified algorithm:

1. **Mandatory coverage** — iterates classes (rarest first), ensures at least 1 video group containing that class exists in both test and valid splits
2. **Capacity fill** — assigns remaining groups toward 70/20/10 targets, prioritizing groups that improve thin class coverage (<5 instances per split)
3. **Train default** — everything unassigned goes to train

Handles Roboflow-renamed filenames by stripping the `_jpg.rf.{hash}` suffix to recover the original video prefix. Physically moves files between `train/`, `valid/`, and `test/` directories.

## Key Design Decisions

### `fliplr=0.0` is mandatory

YOLO's default horizontal flip probability is 0.5. Mirrored versions of characters like 3, 7, 9, J, and Z are not valid representations of those characters. This is the single most impactful training parameter for character detection — getting it wrong silently corrupts training data with no obvious signal in loss curves.

### Roboflow "Fit (white edges)" not "Stretch"

iPad video is 9:16 portrait. Stretching to 640×640 square distorts character proportions. "Fit (white edges)" preserves aspect ratio by padding with white, which matches the white tile background.

### Grouped splits over random splits

Sequential video frames of the same tile on the same surface are nearly identical. A random split will place frame 23 in train and frame 24 in val — the model memorizes frame 23 and "passes" on frame 24, inflating mAP without learning generalization. Grouping by video prefix forces the model to generalize across filming sessions.

### Gemini via OpenRouter (not direct API)

OpenRouter provides structured JSON output via `json_schema` response format with `strict: true`, plus provider pinning to route exclusively to Google's endpoints. This combination enables deterministic structured output that the direct Gemini API did not support at the time of development.

### API errors tracked separately from empty detections

An API timeout or 500 error is not the same as a frame with no tiles. Early versions stored both as `[]` (empty detection list), which meant silent API failures became false hard negatives in training — the model would learn that some tile-containing frames have no objects. Tracking errors separately in `annotation_errors.json` prevents this data corruption.

### `is_prediction=False` on Roboflow upload

Roboflow's upload API has two modes: predictions (which route to annotation review jobs) and ground truth (which go directly into the dataset). Annotations uploaded as predictions are excluded from dataset version generation unless manually approved through the review workflow. Using `False` bypasses this and includes annotations directly in the dataset.

### 4x Roboflow augmentation multiplier

With ~1200 raw images across 36 classes (~33 per class), the dataset is too small for effective training without augmentation. Roboflow's 4x multiplier (brightness, exposure, blur, noise) supplements YOLO's online augmentation (mosaic, HSV, scale, rotation) — both are needed, neither alone is sufficient.

### iPad as primary capture device

The model needs to learn the specific camera module, focal length, color profile, and auto-exposure behavior of the device it will be deployed on. iPhone footage is useful for domain diversity but should be <30% of total training data.

## Training

### Hardware

**Recommended: Kaggle P100 GPU** — free, ~3.3 hours for 150 epochs on ~5000 images. The ready-to-run notebook is at `kaggle/train-digit-tiles.ipynb`.

**Alternative: Apple Silicon MPS** — works but ~11 hours for the same run. Not recommended unless iterating quickly on small experiments.

### Key training parameters

```
model=yolo11n.pt   epochs=150   imgsz=640   batch=16   patience=50
cos_lr=True   fliplr=0.0   flipud=0.0   degrees=10   hsv_v=0.5
close_mosaic=15   mosaic=1.0   mixup=0.0   copy_paste=0.0
```

`close_mosaic=15` disables mosaic augmentation for the final 15 epochs to improve precision on clean single-object images. `cos_lr=True` uses cosine annealing for smoother convergence.

### Current model results (train3)

| Metric | Value |
|---|---|
| mAP50 | 0.995 |
| mAP50-95 | 0.973 |
| Precision | 0.993 |
| Recall | 0.998 |
| Training time | 3.3 hours (Kaggle P100) |
| ONNX size | 10.6 MB |
| ONNX output shape | `[1, 40, 8400]` (4 box + 36 class scores × 8400 anchors) |
| ONNX opset | 17, float32 (WASM does not support float16) |

### ONNX export

```bash
yolo export model=runs/detect/train3/weights/best.pt format=onnx imgsz=640 opset=17 half=False batch=1
```

The exported model is copied to `~/proj/superbuilders/public/models/digit-tiles.onnx` for deployment (the local directory retains the old "superbuilders" name). The app's inference worker reads class count from the ONNX output tensor dimensions automatically — no code change needed when swapping models.

## Deployment Path

```
train3/best.pt → yolo export → best.onnx (10.6 MB)
    ↓
~/proj/superbuilders/public/models/digit-tiles.onnx
    ↓
git push → Cloudflare Pages CI/CD
    ↓
PWA service worker caches model (StaleWhileRevalidate)
    ↓
iPad Safari loads ONNX → ONNX Runtime WASM → real-time inference (<120ms/frame)
```

Existing users get the cached model immediately and the updated model downloads in the background. No versioned filename needed — overwriting `digit-tiles.onnx` is sufficient.

## Environment and Config

### Required environment variables (`.env`)

```
OPENROUTER_API_KEY=sk-or-...    # For Gemini annotation
ROBOFLOW_API_KEY=...            # For dataset upload
```

Template at `.env.example`.

### External services

| Service | Purpose | Account needed |
|---|---|---|
| [OpenRouter](https://openrouter.ai/) | Routes Gemini API calls with structured output | Yes (API key) |
| [Roboflow](https://roboflow.com/) | Dataset versioning, augmentation, export | Yes (free tier works) |
| [Kaggle](https://kaggle.com/) | Free P100 GPU for training | Yes (free) |
| [Cloudflare Pages](https://pages.cloudflare.com/) | Hosts the TileSight app (not this repo) | Yes (via TileSight project) |

### Roboflow project

- Workspace: `seans-workspace-zsmup`
- Project: `digital-tiles`
- Current version: v5

## Testing

There are no automated tests (unit, integration, or otherwise) in this repository. Quality assurance is handled by:

1. **`scripts/qa.py`** — automated checks on annotation outputs (class distribution, bbox geometry, cross-frame consistency, empty frame analysis)
2. **Visual QA images** — bounding box visualizations in `auto_labels/batch/qa/` for manual review
3. **YOLO validation metrics** — mAP, precision, recall on the held-out validation set
4. **Confusion matrix** — per-class accuracy analysis (`runs/detect/train3/confusion_matrix_normalized.png`)
5. **Device testing** — manual testing on iPad Safari via Cloudflare tunnel (`cloudflared tunnel --url https://localhost:5173`)

## Gotchas

| Gotcha | Why it's dangerous | How to avoid |
|---|---|---|
| `fliplr` defaults to 0.5 in YOLO | Mirrored 3, 7, 9, J, Z silently corrupt training with no signal in loss curves | Always pass `fliplr=0.0` explicitly |
| Roboflow resize "Stretch" mode | 9:16 portrait squashed to 1:1 distorts all character proportions | Use "Fit (white edges)" |
| Roboflow random train/val split | Near-duplicate video frames leak across splits, inflating mAP by 10-20% | Use `scripts/split.py` grouped split instead |
| Augmented validation/test sets | Duplicate augmented copies inflate mAP without improving real accuracy | Augment training set only |
| `is_prediction=True` on upload | Annotations go to review jobs excluded from dataset versions — images appear uploaded but never enter training | Use `is_prediction=False` |
| Blur threshold calibrated for general images | White tile surfaces have naturally low Laplacian variance; a threshold of 3.0 (common default) rejects most frames | Threshold recalibrated to 1.0 for this domain |
| ONNX `half=True` | WASM backend does not support float16 — model fails to load in browser | Always export with `half=False` |
| `imgsz` mismatch between training and export | Different input resolution produces garbage detections | Use `imgsz=640` for both |

## Project Evolution

Development spanned March 13-15, 2026 (20 commits, linear history on `main`):

1. **Config + dependencies** — 36-class mapping, OpenRouter integration, Roboflow setup
2. **Frame filtering** — blur detection with threshold recalibrated for white surfaces, perceptual hash dedup
3. **Annotation pipeline** — Gemini structured output, YOLO format conversion, automated QA checks
4. **Error handling** — separated API failures from empty detections to prevent silent data corruption
5. **Roboflow workflow** — batch upload with `is_prediction=False`, manual review cycle
6. **Stratified splitting** — 3-phase grouped algorithm by video prefix for leak-free splits
7. **Kaggle training** — P100 notebook, 150 epochs, achieving 0.995 mAP50
8. **Documentation** — comprehensive training guide, README, Kaggle notebook

## Lessons Learned

- **Kaggle over local training.** M3 Mac MPS training works but takes 3x longer with worse developer experience. Kaggle P100 is free and significantly faster.
- **Gemini annotation over manual.** Manual Roboflow annotation: ~1 hour per 100 images. Gemini Flash Lite: ~2000 images in <30 minutes at equivalent accuracy.
- **Split strategy matters more than model architecture.** The jump from random splits to grouped stratified splits had more impact on real-world accuracy than any hyperparameter tuning.
- **Character detection is not generic object detection.** The horizontal flip gotcha alone can silently ruin a model — standard YOLO tutorials don't mention it because most objects are flip-invariant.

## Related Projects

- **[TileSight](https://github.com/FlanaganSe/superbuilders-numberGame)** — the demo iPad app that consumes the trained ONNX model for real-time tile detection
