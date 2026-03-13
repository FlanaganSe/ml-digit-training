"""QA tools: draw bounding boxes on images, print class distribution."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2

from scripts.annotate import Detection
from scripts.config import CLASS_NAMES


def draw_detections(
    image_path: Path,
    detections: list[Detection],
    output_path: Path,
) -> None:
    """Draw bounding boxes and labels on an image, save to output_path."""
    img = cv2.imread(str(image_path))
    if img is None:
        return

    h, w = img.shape[:2]

    for det in detections:
        if not det.is_valid:
            continue

        # Convert 0-1000 normalized coords to pixel coords
        x1 = int(det.x_min / 1000 * w)
        y1 = int(det.y_min / 1000 * h)
        x2 = int(det.x_max / 1000 * w)
        y2 = int(det.y_max / 1000 * h)

        color = (0, 255, 0)  # green
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label_text = f"{det.label} (id={det.class_id})"
        cv2.putText(
            img,
            label_text,
            (x1, max(y1 - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)


def print_class_distribution(results: dict[Path, list[Detection]]) -> None:
    """Print detection counts per class across all annotated frames."""
    counts: Counter[str] = Counter()
    total_frames = len(results)
    frames_with_detections = 0
    total_detections = 0

    for _path, detections in results.items():
        valid = [d for d in detections if d.is_valid]
        if valid:
            frames_with_detections += 1
        total_detections += len(valid)
        for d in valid:
            counts[d.label] += 1

    print(f"Frames: {total_frames} total, {frames_with_detections} with detections")
    print(f"Detections: {total_detections} total")
    print()

    # Print digits
    print("Digits:")
    for i in range(10):
        name = str(i)
        print(f"  {name}: {counts.get(name, 0)}")

    # Print letters
    print("Letters:")
    for i in range(26):
        name = chr(ord("A") + i)
        print(f"  {name}: {counts.get(name, 0)}")

    # Flag any unknown labels
    known = set(CLASS_NAMES)
    unknown = {k: v for k, v in counts.items() if k not in known}
    if unknown:
        print(f"\nUnknown labels: {unknown}")
