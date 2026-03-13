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
