"""Pipeline configuration: class definitions, API config, thresholds, and annotation policy."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

# Load .env from project root (if it exists)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT: Final = Path(__file__).resolve().parent.parent
FRAMES_DIR: Final = PROJECT_ROOT / "frames"
OUTPUT_DIR: Final = PROJECT_ROOT / "auto_labels"
QA_DIR: Final = OUTPUT_DIR / "qa"

# ── Class Definitions (36 classes: 0-9 digits + A-Z letters) ─────────────────

DIGIT_NAMES: Final = [str(i) for i in range(10)]
LETTER_NAMES: Final = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
CLASS_NAMES: Final = DIGIT_NAMES + LETTER_NAMES  # indices 0-9 = digits, 10-35 = letters
NC: Final = len(CLASS_NAMES)  # 36

# label string → class_id
CLASS_MAP: Final[dict[str, int]] = {name: i for i, name in enumerate(CLASS_NAMES)}

# ── OpenRouter API Config ────────────────────────────────────────────────────

OPENROUTER_BASE_URL: Final = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL: Final = "google/gemini-3.1-flash-lite-preview"

# Provider pinning: route to Google only, require structured output support.
# Slug is lowercase per OpenRouter docs.
OPENROUTER_PROVIDER: Final = {
    "order": ["google"],
    "require_parameters": True,
    "allow_fallbacks": True,
}

# Fallback models (upgrade path if lite-preview bbox quality fails pilot)
OPENROUTER_MODEL_FALLBACK_1: Final = "google/gemini-3-flash-preview"
OPENROUTER_MODEL_FALLBACK_2: Final = "google/gemini-2.5-flash"

# ── Annotation Thresholds ────────────────────────────────────────────────────

BLUR_THRESHOLD: Final = 1.0  # Laplacian variance below this = too blurry (recalibrated: white surfaces have low texture)
DEDUP_HASH_THRESHOLD: Final = 8  # perceptual hash hamming distance

YOLO_HIGH_CONFIDENCE: Final = 0.70  # YOLO detections above this = auto-accept
YOLO_LOW_CONFIDENCE: Final = 0.25  # below this = treat as no detection

# ── Gemini Structured Output Schema ──────────────────────────────────────────
# Gemini bbox format: [y_min, x_min, y_max, x_max] normalized 0-1000.
# Coordinate order is critical — Gemini was trained on this specific order.

DETECTION_SCHEMA: Final = {
    "type": "object",
    "properties": {
        "detections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "Character on the tile: digit '0'-'9' or letter 'A'-'Z'",
                    },
                    "box_2d": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "[y_min, x_min, y_max, x_max] normalized 0-1000",
                    },
                },
                "required": ["label", "box_2d"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["detections"],
    "additionalProperties": False,
}

# ── Annotation Prompt ────────────────────────────────────────────────────────

ANNOTATION_PROMPT: Final = """\
Detect all character tiles in this image. Each tile is a white rectangular \
card with a single bold black character on it, placed on a surface.

For each tile, return:
- "label": the character on the tile (single digit "0"-"9" or uppercase letter "A"-"Z")
- "box_2d": bounding box as [y_min, x_min, y_max, x_max] integers 0-1000

Annotation policy:
- Only include tiles where the character is CLEARLY READABLE and the tile is \
INDIVIDUALLY DISTINGUISHABLE
- Do NOT annotate: overlapping card piles, partially occluded tiles (>30% hidden), \
severely cropped tiles, tiles too blurry to read
- If multiple tiles are visible but some are in a pile/stack, only annotate tiles \
that are individually distinguishable
- If no clearly readable tile exists, return an empty detections array\
"""

# Temperature for Gemini calls — low for maximum consistency
GEMINI_TEMPERATURE: Final = 0.1


# ── API Key Accessors ────────────────────────────────────────────────────────


def get_openrouter_api_key() -> str:
    """Return the OpenRouter API key from environment, or raise with instructions."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. "
            "Add it to .env in the project root or export it in your shell."
        )
    return key


def get_roboflow_api_key() -> str:
    """Return the Roboflow API key from environment, or raise with instructions."""
    key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY not set. "
            "Add it to .env in the project root or export it in your shell."
        )
    return key
