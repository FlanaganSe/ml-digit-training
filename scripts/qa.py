"""QA tools: draw bounding boxes, print class distribution, and automated checks."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import cv2

from scripts.annotate import Detection, load_raw_detections
from scripts.config import CLASS_NAMES
from scripts.filter import video_prefix


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


def check_cross_frame_consistency(results: dict[Path, list[Detection]]) -> list[str]:
    """Flag frames where detection count or labels change abruptly from adjacent frames.

    Groups frames by video prefix and compares sequential frames. Flags:
    - Detection count changes by more than ±2
    - Majority label changes between single-detection frames (misclassification signal)

    Returns list of warning messages.
    """
    warnings: list[str] = []

    groups: dict[str, list[tuple[Path, list[Detection]]]] = {}
    for path, dets in results.items():
        prefix = video_prefix(path)
        groups.setdefault(prefix, []).append((path, dets))

    for _prefix, frames in sorted(groups.items()):
        frames.sort(key=lambda x: x[0].name)

        for i in range(1, len(frames)):
            prev_path, prev_dets = frames[i - 1]
            curr_path, curr_dets = frames[i]

            count_diff = abs(len(curr_dets) - len(prev_dets))
            if count_diff > 2:
                warnings.append(
                    f"  {curr_path.name}: detection count {len(prev_dets)} → {len(curr_dets)} "
                    f"(±{count_diff}) from {prev_path.name}"
                )

            # Only flag label changes for single-detection frames (strongest signal)
            if len(prev_dets) == 1 and len(curr_dets) == 1:
                prev_label = prev_dets[0].label
                curr_label = curr_dets[0].label
                if prev_label != curr_label:
                    warnings.append(
                        f"  {curr_path.name}: label '{prev_label}' → '{curr_label}' "
                        f"from {prev_path.name}"
                    )

    return warnings


def check_bbox_geometry(results: dict[Path, list[Detection]]) -> list[str]:
    """Flag detections with suspicious bounding box geometry.

    Checks (on 0-1000 normalized coordinates):
    - Aspect ratio outside 0.3–4.0
    - Width or height outside 5–60% of frame
    - Coordinates touching frame edge (0 or 1000)

    Returns list of warning messages.
    """
    warnings: list[str] = []

    for path, dets in sorted(results.items(), key=lambda x: x[0].name):
        for det in dets:
            w = det.x_max - det.x_min
            h = det.y_max - det.y_min
            issues: list[str] = []

            # Guard: degenerate boxes with zero width or height
            if w <= 0 or h <= 0:
                issues.append("degenerate box (zero width/height)")
            else:
                ar = w / h
                if ar < 0.3 or ar > 4.0:
                    issues.append(f"aspect ratio {ar:.1f}")

                w_pct = w / 10  # 0-1000 → 0-100%
                h_pct = h / 10
                if w_pct < 5 or w_pct > 60:
                    issues.append(f"width {w_pct:.0f}%")
                if h_pct < 5 or h_pct > 60:
                    issues.append(f"height {h_pct:.0f}%")

            if det.x_min == 0 or det.y_min == 0 or det.x_max == 1000 or det.y_max == 1000:
                issues.append("touches frame edge")

            if issues:
                warnings.append(
                    f"  {path.name}: '{det.label}' "
                    f"[{det.y_min},{det.x_min},{det.y_max},{det.x_max}] — "
                    + ", ".join(issues)
                )

    return warnings


def check_class_distribution_by_prefix(
    results: dict[Path, list[Detection]],
) -> list[str]:
    """Flag video prefixes where the dominant class has <70% share.

    For single-letter videos, most detections should share one label.
    Mixed-letter videos will naturally fail this — warnings only.

    Returns list of warning messages.
    """
    warnings: list[str] = []

    prefix_labels: dict[str, Counter[str]] = {}
    for path, dets in results.items():
        prefix = video_prefix(path)
        if prefix not in prefix_labels:
            prefix_labels[prefix] = Counter()
        for d in dets:
            prefix_labels[prefix][d.label] += 1

    for prefix in sorted(prefix_labels):
        counts = prefix_labels[prefix]
        total = sum(counts.values())
        if total == 0:
            continue

        dominant_label, dominant_count = counts.most_common(1)[0]
        share = dominant_count / total

        if share < 0.7:
            top3 = counts.most_common(3)
            dist = ", ".join(f"'{label}':{count}" for label, count in top3)
            warnings.append(
                f"  {prefix}: dominant '{dominant_label}' = {share:.0%} "
                f"({dist}; total={total})"
            )

    return warnings


def check_empty_frames(
    results: dict[Path, list[Detection]],
) -> tuple[list[str], list[str]]:
    """Report frames with 0 detections, grouped by video prefix.

    Distinguishes:
    - Entire video with 0 detections = likely empty-surface video (informational)
    - Scattered empty frames in a video with detections = possible annotation failure

    Returns (warnings, info) where warnings are scattered empties and
    info lists empty-surface videos (expected, not counted as warnings).
    """
    warnings: list[str] = []
    info: list[str] = []

    prefix_empty: dict[str, list[str]] = {}
    prefix_total: dict[str, int] = {}
    for path, dets in results.items():
        prefix = video_prefix(path)
        prefix_total[prefix] = prefix_total.get(prefix, 0) + 1
        if not dets:
            prefix_empty.setdefault(prefix, []).append(path.name)

    for prefix in sorted(prefix_empty):
        empty_count = len(prefix_empty[prefix])
        total = prefix_total[prefix]

        if empty_count == total:
            info.append(f"{prefix} ({total} frames)")
        else:
            names = sorted(prefix_empty[prefix])[:5]
            suffix = f" (+{empty_count - 5} more)" if empty_count > 5 else ""
            warnings.append(
                f"  {prefix}: {empty_count}/{total} frames empty — "
                + ", ".join(names)
                + suffix
            )

    return warnings, info


def check_annotation_errors(errors_path: Path) -> list[str]:
    """Report annotation errors from M4 error tracking.

    Loads annotation_errors.json (if it exists) and reports failed frames.

    Returns list of warning messages.
    """
    warnings: list[str] = []

    if not errors_path.exists():
        return warnings

    try:
        errors = json.loads(errors_path.read_text())
    except json.JSONDecodeError as e:
        warnings.append(f"  {errors_path.name}: corrupt JSON — {e}")
        return warnings

    if not errors:
        return warnings

    for filename, msg in sorted(errors.items()):
        warnings.append(f"  {filename}: {msg}")

    return warnings


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
    all_warnings: list[str] = []

    # 1. Class distribution (existing — informational, no warnings)
    print("=" * 60)
    print("CLASS DISTRIBUTION")
    print("=" * 60)
    print_class_distribution(results)

    # 2. Cross-frame consistency
    print("\n" + "=" * 60)
    print("CROSS-FRAME CONSISTENCY")
    print("=" * 60)
    w = check_cross_frame_consistency(results)
    all_warnings.extend(w)
    if w:
        print(f"{len(w)} warning(s):")
        for msg in w:
            print(msg)
    else:
        print("No issues found.")

    # 3. Bbox geometry
    print("\n" + "=" * 60)
    print("BBOX GEOMETRY")
    print("=" * 60)
    w = check_bbox_geometry(results)
    all_warnings.extend(w)
    if w:
        print(f"{len(w)} warning(s):")
        for msg in w:
            print(msg)
    else:
        print("No issues found.")

    # 4. Class distribution by video prefix
    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION BY VIDEO PREFIX")
    print("=" * 60)
    w = check_class_distribution_by_prefix(results)
    all_warnings.extend(w)
    if w:
        print(f"{len(w)} warning(s) (prefixes with dominant class <70%):")
        for msg in w:
            print(msg)
    else:
        print("All prefixes have dominant class ≥70%.")

    # 5. Empty frames
    print("\n" + "=" * 60)
    print("EMPTY FRAMES")
    print("=" * 60)
    w, empty_info = check_empty_frames(results)
    all_warnings.extend(w)
    if empty_info:
        print(f"Empty-surface videos (expected): {', '.join(empty_info)}")
    if w:
        print(f"{len(w)} warning(s) (scattered empties in detected videos):")
        for msg in w:
            print(msg)
    elif not empty_info:
        print("No empty frames.")

    # 6. Annotation errors (from M4)
    errors_path = batch_dir / "annotation_errors.json"
    print("\n" + "=" * 60)
    print("ANNOTATION ERRORS")
    print("=" * 60)
    w = check_annotation_errors(errors_path)
    all_warnings.extend(w)
    if w:
        print(f"{len(w)} frame(s) failed annotation (excluded from training):")
        for msg in w:
            print(msg)
        print("Consider retrying these frames.")
    else:
        print("No annotation errors.")

    # 7. Draw QA images for frames with detections
    qa_dir = batch_dir / "qa"
    drawn = 0
    for frame_path, dets in sorted(results.items(), key=lambda x: x[0].name):
        if dets and frame_path.exists():
            draw_detections(frame_path, dets, qa_dir / frame_path.name)
            drawn += 1

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Frames analyzed: {len(results)}")
    print(f"QA images saved: {drawn} → {qa_dir}")
    print(f"Total warnings:  {len(all_warnings)}")
    if all_warnings:
        print(f"\n{len(all_warnings)} item(s) need attention — review warnings above.")
