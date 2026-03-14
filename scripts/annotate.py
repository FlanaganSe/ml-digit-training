"""Gemini annotation via OpenRouter: image → structured JSON detections."""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from scripts.config import (
    ANNOTATION_PROMPT,
    CLASS_MAP,
    DETECTION_SCHEMA,
    GEMINI_TEMPERATURE,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    OPENROUTER_PROVIDER,
    get_openrouter_api_key,
)


@dataclass(frozen=True)
class Detection:
    """A single detected tile with label and bounding box."""

    label: str  # e.g. "7", "A"
    y_min: int  # 0-1000 normalized
    x_min: int
    y_max: int
    x_max: int

    @property
    def class_id(self) -> int | None:
        """Return the class ID for this label, or None if unknown."""
        return CLASS_MAP.get(self.label.upper())

    @property
    def is_valid(self) -> bool:
        """Check if detection has a known class and reasonable coordinates."""
        return (
            self.class_id is not None
            and 0 <= self.y_min < self.y_max <= 1000
            and 0 <= self.x_min < self.x_max <= 1000
        )


def _encode_image(image_path: Path) -> str:
    """Read image file and return base64-encoded data URI."""
    suffix = image_path.suffix.lower()
    mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else f"image/{suffix.lstrip('.')}"
    data = image_path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _create_client() -> OpenAI:
    """Create OpenAI client configured for OpenRouter."""
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=get_openrouter_api_key(),
    )


def annotate_frame(
    image_path: Path,
    *,
    client: OpenAI | None = None,
    model: str = OPENROUTER_MODEL,
) -> list[Detection]:
    """Annotate a single frame using Gemini via OpenRouter.

    Returns a list of valid Detection objects.
    """
    if client is None:
        client = _create_client()

    image_uri = _encode_image(image_path)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ANNOTATION_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_uri}},
                ],
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "tile_detections",
                "strict": True,
                "schema": DETECTION_SCHEMA,
            },
        },
        temperature=GEMINI_TEMPERATURE,
        extra_body={"provider": OPENROUTER_PROVIDER},
    )

    raw = response.choices[0].message.content
    if not raw:
        return []

    parsed = json.loads(raw)
    detections_raw = parsed.get("detections", [])

    detections = []
    for d in detections_raw:
        box = d.get("box_2d", [])
        if len(box) != 4:
            continue
        det = Detection(
            label=d.get("label", "").upper(),
            y_min=box[0],
            x_min=box[1],
            y_max=box[2],
            x_max=box[3],
        )
        if det.is_valid:
            detections.append(det)

    return detections


def save_raw_detections(
    results: dict[Path, list[Detection]],
    output_path: Path,
) -> None:
    """Serialize annotation results to JSON for downstream pipeline steps.

    Format: {filename: [{label, box_2d: [y_min, x_min, y_max, x_max]}, ...]}
    """
    raw: dict[str, list[dict]] = {}
    for path, dets in sorted(results.items(), key=lambda x: x[0].name):
        raw[path.name] = [
            {"label": d.label, "box_2d": [d.y_min, d.x_min, d.y_max, d.x_max]}
            for d in dets
        ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(raw, indent=2))


def load_raw_detections(
    input_path: Path,
    frames_dir: Path,
) -> dict[Path, list[Detection]]:
    """Deserialize annotation results from JSON saved by save_raw_detections().

    Returns dict mapping frame paths to validated Detection lists.
    """
    raw = json.loads(input_path.read_text())
    results: dict[Path, list[Detection]] = {}
    dropped = 0
    for filename, dets_raw in raw.items():
        frame_path = frames_dir / filename
        detections = []
        for d in dets_raw:
            box = d["box_2d"]
            det = Detection(
                label=d["label"].upper(),
                y_min=box[0],
                x_min=box[1],
                y_max=box[2],
                x_max=box[3],
            )
            if det.is_valid:
                detections.append(det)
            else:
                dropped += 1
                print(f"  WARNING: dropped invalid detection in {filename}: {d}")
        results[frame_path] = detections
    if dropped:
        print(f"  WARNING: {dropped} invalid detection(s) dropped during load")
    return results


def annotate_batch(
    frames: list[Path],
    *,
    model: str = OPENROUTER_MODEL,
    delay: float = 0.15,
    on_progress: None | callable = None,
) -> dict[Path, list[Detection]]:
    """Annotate a batch of frames. Returns dict of path → detections.

    Args:
        frames: list of image paths
        model: OpenRouter model ID
        delay: seconds between API calls (rate limit courtesy)
        on_progress: optional callback(index, total, path, detections)
    """
    client = _create_client()
    results: dict[Path, list[Detection]] = {}

    for i, frame in enumerate(frames):
        try:
            detections = annotate_frame(frame, client=client, model=model)
            results[frame] = detections
        except Exception as e:
            print(f"  ERROR on {frame.name}: {e}")
            results[frame] = []

        if on_progress:
            on_progress(i, len(frames), frame, results[frame])

        if i < len(frames) - 1:
            time.sleep(delay)

    return results


if __name__ == "__main__":
    import sys

    from scripts.config import FRAMES_DIR, OUTPUT_DIR

    batch_dir = OUTPUT_DIR / "batch"
    raw_path = batch_dir / "raw_detections.json"

    # Guard: abort if batch/ already has results (non-idempotent)
    if raw_path.exists():
        print(
            f"ERROR: {raw_path} already exists from a previous run.\n"
            "Delete it before re-running, or use a different batch directory."
        )
        sys.exit(1)

    frames = sorted(FRAMES_DIR.glob("*.jpg"))
    if not frames:
        print(f"No .jpg frames found in {FRAMES_DIR}")
        sys.exit(1)

    print(f"Annotating {len(frames)} frames using {OPENROUTER_MODEL}")
    print(f"Output: {batch_dir}\n")

    def _progress(i: int, total: int, path: Path, dets: list[Detection]) -> None:
        print(f"  [{i + 1}/{total}] {path.name}: {len(dets)} detections")

    start = time.time()
    results = annotate_batch(frames, on_progress=_progress)
    elapsed = time.time() - start

    # Save results for downstream scripts
    save_raw_detections(results, raw_path)

    # Save frame list for reproducibility
    (batch_dir / "frames.txt").write_text(
        "\n".join(str(f) for f in sorted(results.keys()))
    )

    total_dets = sum(len(dets) for dets in results.values())
    empty = sum(1 for dets in results.values() if not dets)

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"  Frames:     {len(results):>5}")
    print(f"  Detections: {total_dets:>5}")
    print(f"  Empty:      {empty:>5}")

    # Warn if many frames are empty — may indicate API failures (see M4)
    if len(results) > 0 and empty / len(results) > 0.10:
        print(
            f"\n  WARNING: {empty / len(results) * 100:.0f}% of frames have no detections.\n"
            "  This may indicate API errors. Check ERROR lines above.\n"
            "  Delete the output and re-run if needed."
        )

    print(f"\nSaved to {raw_path}")
