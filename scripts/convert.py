"""Convert Gemini detections to YOLO label format."""

from __future__ import annotations

from pathlib import Path

from scripts.annotate import Detection, load_raw_detections


def detection_to_yolo(det: Detection) -> str | None:
    """Convert a single Detection to a YOLO label line.

    Returns: "class_id cx cy w h" or None if detection is invalid.
    YOLO format: all values normalized 0-1 (not 0-1000).
    """
    if not det.is_valid or det.class_id is None:
        return None

    cx = (det.x_min + det.x_max) / 2 / 1000
    cy = (det.y_min + det.y_max) / 2 / 1000
    w = (det.x_max - det.x_min) / 1000
    h = (det.y_max - det.y_min) / 1000

    return f"{det.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def detections_to_yolo(detections: list[Detection]) -> str:
    """Convert a list of Detections to YOLO label file content.

    Returns multi-line string (one line per detection), or empty string
    for frames with no valid detections (negative example).
    """
    lines = [detection_to_yolo(d) for d in detections]
    return "\n".join(line for line in lines if line is not None)


def write_yolo_labels(
    results: dict[Path, list[Detection]],
    output_dir: Path,
) -> int:
    """Write YOLO label files for all annotated frames.

    Creates one .txt file per frame. Empty detections → empty file (negative example).
    Returns count of files written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for frame_path, detections in results.items():
        label_content = detections_to_yolo(detections)
        label_path = output_dir / (frame_path.stem + ".txt")
        label_path.write_text(label_content)
        count += 1

    return count


if __name__ == "__main__":
    import sys

    from scripts.config import FRAMES_DIR, OUTPUT_DIR

    batch_dir = OUTPUT_DIR / "batch"
    raw_path = batch_dir / "raw_detections.json"

    if not raw_path.exists():
        print(
            f"ERROR: {raw_path} not found.\n"
            "Run 'python -m scripts.annotate' first."
        )
        sys.exit(1)

    results = load_raw_detections(raw_path, FRAMES_DIR)

    labels_dir = batch_dir / "labels"
    count = write_yolo_labels(results, labels_dir)

    total_dets = sum(len(dets) for dets in results.values())
    empty = sum(1 for dets in results.values() if not dets)

    print(f"Converted {count} frames → {labels_dir}")
    print(f"  Detections: {total_dets:>5}")
    print(f"  Empty:      {empty:>5}")
