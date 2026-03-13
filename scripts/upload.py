"""Upload annotated frames to Roboflow for review."""

from __future__ import annotations

from pathlib import Path

from roboflow import Roboflow

from scripts.config import CLASS_NAMES, FRAMES_DIR, OUTPUT_DIR, get_roboflow_api_key

# Roboflow project coordinates (from dataset/data.yaml)
WORKSPACE = "seans-workspace-zsmup"
PROJECT = "digital-tiles"


def create_labelmap(output_path: Path) -> Path:
    """Create a labelmap file mapping class indices to names."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(CLASS_NAMES))
    return output_path


def upload_to_roboflow(
    frames_dir: Path = FRAMES_DIR,
    labels_dir: Path | None = None,
    *,
    batch_name: str = "auto_annotated_v1",
    is_prediction: bool = True,
    on_progress: callable | None = None,
) -> int:
    """Upload images + YOLO labels to Roboflow.

    Args:
        frames_dir: directory containing .jpg frames
        labels_dir: directory containing .txt YOLO labels (defaults to auto_labels/batch/labels)
        batch_name: Roboflow batch name for grouping uploads
        is_prediction: if True, annotations appear as review suggestions
        on_progress: optional callback(index, total, frame_name, success)

    Returns count of successfully uploaded frames.
    """
    if labels_dir is None:
        labels_dir = OUTPUT_DIR / "batch" / "labels"

    # Create labelmap
    labelmap_path = OUTPUT_DIR / "labelmap.txt"
    create_labelmap(labelmap_path)

    # Connect to Roboflow
    rf = Roboflow(api_key=get_roboflow_api_key())
    project = rf.workspace(WORKSPACE).project(PROJECT)

    # Collect frames that have labels
    frames_to_upload = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        frame_path = frames_dir / (label_path.stem + ".jpg")
        if frame_path.exists():
            frames_to_upload.append((frame_path, label_path))

    uploaded = 0
    for i, (frame_path, label_path) in enumerate(frames_to_upload):
        try:
            project.single_upload(
                image_path=str(frame_path),
                annotation_path=str(label_path),
                annotation_labelmap=str(labelmap_path),
                batch_name=batch_name,
                is_prediction=is_prediction,
            )
            uploaded += 1
        except Exception as e:
            print(f"  ERROR uploading {frame_path.name}: {e}")

        if on_progress:
            on_progress(i, len(frames_to_upload), frame_path.name, True)

    return uploaded
