"""Frame filtering: blur detection and perceptual hash deduplication."""

from __future__ import annotations

from pathlib import Path

import cv2
import imagehash
from PIL import Image

from scripts.config import BLUR_THRESHOLD, DEDUP_HASH_THRESHOLD, FRAMES_DIR


def blur_score(image_path: Path) -> float:
    """Return Laplacian variance — higher = sharper. Low values indicate motion blur."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def compute_blur_scores(frames: list[Path]) -> dict[Path, float]:
    """Compute blur scores for all frames."""
    return {f: blur_score(f) for f in frames}


def filter_blurry(
    scores: dict[Path, float],
    threshold: float = BLUR_THRESHOLD,
) -> tuple[list[Path], list[Path]]:
    """Split frames into (sharp, blurry) based on blur threshold."""
    sharp = [f for f, s in scores.items() if s >= threshold]
    blurry = [f for f, s in scores.items() if s < threshold]
    return sorted(sharp), sorted(blurry)


def deduplicate(
    frames: list[Path],
    threshold: int = DEDUP_HASH_THRESHOLD,
) -> tuple[list[Path], list[Path]]:
    """Remove near-duplicate frames using perceptual hashing.

    Compares each frame to the previous kept frame within the same video prefix.
    Returns (kept, removed) lists.
    """
    if not frames:
        return [], []

    # Group by video prefix to only dedup within same video
    prefix_groups: dict[str, list[Path]] = {}
    for f in sorted(frames):
        prefix = _video_prefix(f)
        prefix_groups.setdefault(prefix, []).append(f)

    kept: list[Path] = []
    removed: list[Path] = []

    for _prefix, group in sorted(prefix_groups.items()):
        prev_hash: imagehash.ImageHash | None = None
        for f in group:
            h = imagehash.phash(Image.open(f))
            if prev_hash is None or (h - prev_hash) > threshold:
                kept.append(f)
                prev_hash = h
            else:
                removed.append(f)

    return sorted(kept), sorted(removed)


def _video_prefix(path: Path) -> str:
    """Extract video prefix from frame filename.

    numbers_0001.jpg → 'numbers'
    alphaFirstHalf_0042.jpg → 'alphaFirstHalf'
    alpha2ndHalf_0100.jpg → 'alpha2ndHalf'
    """
    name = path.stem
    # Split on last underscore before the numeric suffix
    parts = name.rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else name


def filter_frames(
    frames_dir: Path = FRAMES_DIR,
    blur_threshold: float = BLUR_THRESHOLD,
    dedup_threshold: int = DEDUP_HASH_THRESHOLD,
) -> tuple[list[Path], dict[str, list[Path]]]:
    """Run full filtering pipeline: blur removal then dedup.

    Returns:
        clean: list of paths to keep
        rejected: dict with 'blurry' and 'duplicate' lists
    """
    all_frames = sorted(frames_dir.glob("*.jpg"))
    if not all_frames:
        raise FileNotFoundError(f"No .jpg frames found in {frames_dir}")

    # Step 1: blur filter
    scores = compute_blur_scores(all_frames)
    sharp, blurry = filter_blurry(scores, blur_threshold)

    # Step 2: dedup within sharp frames
    clean, duplicates = deduplicate(sharp, dedup_threshold)

    rejected = {"blurry": blurry, "duplicate": duplicates}
    return clean, rejected


def print_blur_distribution(scores: dict[Path, float]) -> None:
    """Print histogram-style distribution of blur scores for threshold calibration."""
    import statistics

    values = sorted(scores.values())
    if not values:
        print("No scores to analyze.")
        return

    print(f"Frames: {len(values)}")
    print(f"Min:    {values[0]:.1f}")
    print(f"Max:    {values[-1]:.1f}")
    print(f"Mean:   {statistics.mean(values):.1f}")
    print(f"Median: {statistics.median(values):.1f}")
    print(f"Stdev:  {statistics.stdev(values):.1f}" if len(values) > 1 else "")

    # Bucket histogram
    buckets = [0, 25, 50, 75, 100, 150, 200, 300, 500, 1000, float("inf")]
    print("\nDistribution:")
    for i in range(len(buckets) - 1):
        lo, hi = buckets[i], buckets[i + 1]
        count = sum(1 for v in values if lo <= v < hi)
        bar = "#" * count
        label = f"{lo:>6.0f}-{hi:>6.0f}" if hi != float("inf") else f"{lo:>6.0f}+     "
        print(f"  {label}: {count:>4} {bar}")


if __name__ == "__main__":
    # Quick diagnostic: print blur distribution for all frames
    all_frames = sorted(FRAMES_DIR.glob("*.jpg"))
    print(f"Found {len(all_frames)} frames in {FRAMES_DIR}\n")

    scores = compute_blur_scores(all_frames)
    print_blur_distribution(scores)

    # Show 10 lowest (blurriest) frames
    ranked = sorted(scores.items(), key=lambda x: x[1])
    print("\n10 blurriest frames:")
    for path, score in ranked[:10]:
        print(f"  {score:>8.1f}  {path.name}")
