"""Stratified grouped split: assign video clips to train/valid/test preserving class coverage."""

from __future__ import annotations

import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Final

from scripts.config import CLASS_NAMES, NC, PROJECT_ROOT

DATASET_DIR: Final = PROJECT_ROOT / "dataset"

# Target split ratios (train is the residual — not enforced directly)
VAL_RATIO: Final = 0.20
TEST_RATIO: Final = 0.10

# Regex to strip Roboflow's _jpg.rf.{hash} suffix from filenames
# Example: IMG_0118_0001_jpg.rf.403d4b692e5b4d370b39aeffea886432.jpg
#       → original stem: IMG_0118_0001
_RF_SUFFIX_RE: Final = re.compile(r"_jpg\.rf\.[a-f0-9]+$")


def roboflow_video_prefix(path: Path) -> str:
    """Extract video prefix from a Roboflow-renamed frame filename.

    IMG_0118_0001_jpg.rf.403d4b69...jpg → IMG_0118
    IMG_0202_0021_jpg.rf.34a3bdf6...jpg → IMG_0202

    Steps:
    1. Strip the _jpg.rf.{hash} suffix to recover the original stem
    2. Split on last underscore — the numeric suffix is the frame number
    3. Everything before the frame number is the video prefix
    """
    stem = path.stem
    # Step 1: strip Roboflow suffix
    original_stem = _RF_SUFFIX_RE.sub("", stem)
    # Step 2: split off frame number
    parts = original_stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return original_stem


def read_label_classes(label_path: Path) -> set[int]:
    """Read a YOLO label file and return the set of class IDs present."""
    classes: set[int] = set()
    text = label_path.read_text().strip()
    if not text:
        return classes
    for line in text.splitlines():
        parts = line.strip().split()
        if parts:
            classes.add(int(parts[0]))
    return classes


def collect_groups(
    images_dir: Path, labels_dir: Path
) -> dict[str, list[tuple[Path, Path]]]:
    """Group (image, label) pairs by video prefix.

    Returns: {prefix: [(image_path, label_path), ...]}
    """
    groups: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for img_path in sorted(images_dir.glob("*.jpg")):
        label_path = labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            prefix = roboflow_video_prefix(img_path)
            groups[prefix].append((img_path, label_path))
    return dict(groups)


def group_class_counts(
    groups: dict[str, list[tuple[Path, Path]]],
) -> dict[str, Counter[int]]:
    """For each group, count class instances across all its label files."""
    result: dict[str, Counter[int]] = {}
    for prefix, pairs in groups.items():
        counter: Counter[int] = Counter()
        for _img, label in pairs:
            for cls_id in read_label_classes(label):
                counter[cls_id] += 1
        result[prefix] = counter
    return result


def group_class_sets_from_counts(
    counts: dict[str, Counter[int]],
) -> dict[str, set[int]]:
    """Extract class ID sets from pre-computed counts."""
    return {prefix: set(counter.keys()) for prefix, counter in counts.items()}


def grouped_stratified_split(
    groups: dict[str, list[tuple[Path, Path]]],
) -> dict[str, list[str]]:
    """Assign video-prefix groups to train/valid/test splits.

    Three-phase strategy ensuring all 36 classes appear in every split:

    Phase 1 (mandatory coverage): For each class (rarest first), guarantee at
        least one group in test and one in valid. Pick the smallest unassigned
        group that provides coverage for the class.
    Phase 2 (capacity fill): Assign remaining groups toward target ratios
        (70/20/10), preferring whichever split is furthest below its target.
    Phase 3 (train): Everything not in test or valid goes to train.

    Returns: {"train": [prefix, ...], "valid": [...], "test": [...]}
    """
    total_images = sum(len(pairs) for pairs in groups.values())
    target_test = int(total_images * TEST_RATIO)
    target_valid = int(total_images * VAL_RATIO)

    per_group_counts = group_class_counts(groups)
    class_sets = group_class_sets_from_counts(per_group_counts)
    all_classes = set(range(NC))

    # Build class → groups mapping
    class_to_groups: dict[int, list[str]] = defaultdict(list)
    for prefix, classes in class_sets.items():
        for c in classes:
            class_to_groups[c].append(prefix)

    # Phase 1: Mandatory coverage — ensure test and valid each have ≥1 group per class
    # For test: prefer smallest groups (test budget is tight)
    # For valid: prefer groups with most instances of the target class (valid is used
    #   for metric computation during training — needs enough per-class examples)
    test_prefixes: set[str] = set()
    valid_prefixes: set[str] = set()
    test_covered: set[int] = set()
    valid_covered: set[int] = set()

    # Process classes from rarest (fewest groups) to most common
    classes_by_rarity = sorted(all_classes, key=lambda c: len(class_to_groups[c]))

    for cls_id in classes_by_rarity:
        candidate_groups = class_to_groups[cls_id]

        # Ensure test has coverage for this class
        if cls_id not in test_covered:
            if any(p in test_prefixes for p in candidate_groups):
                test_covered.add(cls_id)
            else:
                available = [
                    p for p in candidate_groups
                    if p not in test_prefixes and p not in valid_prefixes
                ]
                if available:
                    # For test: pick smallest group to conserve budget
                    pick = min(available, key=lambda p: len(groups[p]))
                    test_prefixes.add(pick)
                    test_covered |= class_sets[pick]
                else:
                    # All groups in valid — steal the one with fewest instances of this class
                    in_valid = [p for p in candidate_groups if p in valid_prefixes]
                    if in_valid:
                        pick = min(in_valid, key=lambda p: per_group_counts[p][cls_id])
                        valid_prefixes.discard(pick)
                        valid_covered = set()
                        for vp in valid_prefixes:
                            valid_covered |= class_sets[vp]
                        test_prefixes.add(pick)
                        test_covered |= class_sets[pick]

        # Ensure valid has coverage for this class
        if cls_id not in valid_covered:
            if any(p in valid_prefixes for p in candidate_groups):
                valid_covered.add(cls_id)
            else:
                available = [
                    p for p in candidate_groups
                    if p not in test_prefixes and p not in valid_prefixes
                ]
                if available:
                    # For valid: pick group with MOST instances of target class
                    pick = max(available, key=lambda p: per_group_counts[p][cls_id])
                    valid_prefixes.add(pick)
                    valid_covered |= class_sets[pick]

    # Phase 2: Fill toward target ratios, prioritizing class diversity
    # Compute current per-split class instance counts
    split_instance_counts: dict[str, Counter[int]] = {"test": Counter(), "valid": Counter()}
    for prefix in test_prefixes:
        split_instance_counts["test"] += per_group_counts[prefix]
    for prefix in valid_prefixes:
        split_instance_counts["valid"] += per_group_counts[prefix]

    assigned = test_prefixes | valid_prefixes
    remaining = sorted(
        set(groups.keys()) - assigned,
        key=lambda p: len(groups[p]),
    )

    test_count = sum(len(groups[p]) for p in test_prefixes)
    valid_count = sum(len(groups[p]) for p in valid_prefixes)

    for prefix in remaining:
        size = len(groups[prefix])
        test_gap = target_test - test_count
        valid_gap = target_valid - valid_count

        # Score each candidate split by how many thin classes (<5 instances) it would help
        def coverage_boost(split: str) -> int:
            return sum(
                1 for cls_id in per_group_counts[prefix]
                if split_instance_counts[split][cls_id] < 5
            )

        if test_gap > 0 or valid_gap > 0:
            # Prefer the split with more thin-class benefit; break ties by capacity gap
            candidates = []
            if test_gap > 0:
                candidates.append(("test", coverage_boost("test"), test_gap))
            if valid_gap > 0:
                candidates.append(("valid", coverage_boost("valid"), valid_gap))

            best_split, _, _ = max(candidates, key=lambda x: (x[1], x[2]))

            if best_split == "test":
                test_prefixes.add(prefix)
                test_count += size
                split_instance_counts["test"] += per_group_counts[prefix]
            else:
                valid_prefixes.add(prefix)
                valid_count += size
                split_instance_counts["valid"] += per_group_counts[prefix]
        # else: stays unassigned → goes to train

    # Phase 3: Everything else is train
    train_prefixes = set(groups.keys()) - test_prefixes - valid_prefixes

    return {
        "train": sorted(train_prefixes),
        "valid": sorted(valid_prefixes),
        "test": sorted(test_prefixes),
    }


def move_files_to_split(
    groups: dict[str, list[tuple[Path, Path]]],
    assignment: dict[str, list[str]],
    dataset_dir: Path,
) -> dict[str, int]:
    """Move image+label files into their assigned split directories.

    Files currently in dataset/train/ get moved to dataset/valid/ or dataset/test/
    as needed. Files staying in train/ are left in place.

    Returns: {split_name: count_of_files}
    """
    counts: dict[str, int] = {}

    for split_name in ("valid", "test"):
        img_dir = dataset_dir / split_name / "images"
        lbl_dir = dataset_dir / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for prefix in assignment[split_name]:
            for img_path, label_path in groups[prefix]:
                shutil.move(str(img_path), str(img_dir / img_path.name))
                shutil.move(str(label_path), str(lbl_dir / label_path.name))
                count += 1
        counts[split_name] = count

    # Train files stay in place
    counts["train"] = sum(len(groups[p]) for p in assignment["train"])
    return counts


def verify_split_classes(
    assignment: dict[str, list[str]],
    class_sets: dict[str, set[int]],
) -> dict[str, set[int]]:
    """Compute per-split class coverage from group assignments.

    Returns: {split_name: set_of_class_ids}
    """
    result: dict[str, set[int]] = {}
    for split_name, prefixes in assignment.items():
        classes: set[int] = set()
        for prefix in prefixes:
            classes |= class_sets.get(prefix, set())
        result[split_name] = classes
    return result


def print_report(
    groups: dict[str, list[tuple[Path, Path]]],
    assignment: dict[str, list[str]],
    counts: dict[str, int],
    split_classes: dict[str, set[int]],
    per_group_class_counts: dict[str, Counter[int]],
) -> tuple[list[str], list[str]]:
    """Print comprehensive split report.

    Returns (critical_warnings, info_warnings) where critical = missing classes.
    """
    critical: list[str] = []
    info: list[str] = []
    total = sum(counts.values())
    all_classes = set(range(NC))

    print("=" * 60)
    print("GROUPED STRATIFIED SPLIT REPORT")
    print("=" * 60)

    # Group overview
    print(f"\nVideo prefixes: {len(groups)}")
    print(f"Total images: {total}")
    print(f"Group sizes: {min(len(v) for v in groups.values())}–{max(len(v) for v in groups.values())}")

    # Per-split summary
    print(f"\n{'Split':<8} {'Groups':>7} {'Images':>7} {'Pct':>6} {'Classes':>8}")
    print("-" * 40)
    for split_name in ("train", "valid", "test"):
        n_groups = len(assignment[split_name])
        n_images = counts[split_name]
        pct = n_images / total * 100 if total else 0
        n_classes = len(split_classes[split_name])
        print(f"{split_name:<8} {n_groups:>7} {n_images:>7} {pct:>5.1f}% {n_classes:>7}/36")

    # Per-split prefix assignments
    for split_name in ("train", "valid", "test"):
        print(f"\n{split_name} prefixes ({len(assignment[split_name])}):")
        for prefix in sorted(assignment[split_name]):
            print(f"  {prefix} ({len(groups[prefix])} images)")

    # Class coverage details
    print("\n" + "=" * 60)
    print("CLASS COVERAGE")
    print("=" * 60)

    # Count per-class instances in each split (using pre-computed counts)
    split_class_counts: dict[str, Counter[int]] = {
        s: Counter() for s in ("train", "valid", "test")
    }
    for split_name, prefixes in assignment.items():
        for prefix in prefixes:
            split_class_counts[split_name] += per_group_class_counts[prefix]

    print(f"\n{'Class':<6} {'Train':>7} {'Valid':>7} {'Test':>7}  (images containing class)")
    print("-" * 42)
    for cls_id in range(NC):
        name = CLASS_NAMES[cls_id]
        t = split_class_counts["train"][cls_id]
        v = split_class_counts["valid"][cls_id]
        te = split_class_counts["test"][cls_id]
        flag = ""
        if v == 0:
            flag = "  *** MISSING FROM VALID ***"
            critical.append(f"Class '{name}' (id={cls_id}) MISSING from valid split")
        elif v < 5:
            flag = f"  (low: {v} in valid)"
            info.append(f"Class '{name}' has only {v} examples in valid split")
        if te == 0:
            flag += "  *** MISSING FROM TEST ***"
            critical.append(f"Class '{name}' (id={cls_id}) MISSING from test split")
        print(f"{name:<6} {t:>7} {v:>7} {te:>7}{flag}")

    # Final summary (per-class entries already in critical — just print banner)
    for split_name in ("valid", "test"):
        missing = all_classes - split_classes[split_name]
        if missing:
            missing_names = [CLASS_NAMES[c] for c in sorted(missing)]
            print(f"\n{'!' * 60}")
            print(f"  WARNING: {split_name} is missing classes: {missing_names}")
            print(f"{'!' * 60}")

    return critical, info


if __name__ == "__main__":
    import sys

    train_images = DATASET_DIR / "train" / "images"
    train_labels = DATASET_DIR / "train" / "labels"

    if not train_images.exists() or not any(train_images.glob("*.jpg")):
        print(f"ERROR: No images in {train_images}")
        sys.exit(1)

    # Guard: don't re-split if valid/test already exist
    for split_name in ("valid", "test"):
        split_dir = DATASET_DIR / split_name / "images"
        if split_dir.exists() and any(split_dir.glob("*.jpg")):
            print(
                f"ERROR: {split_dir} already contains images.\n"
                "Remove valid/ and test/ directories first to re-split."
            )
            sys.exit(1)

    # Collect and group
    groups = collect_groups(train_images, train_labels)
    print(f"Found {sum(len(v) for v in groups.values())} images in {len(groups)} video groups\n")

    # Compute class data BEFORE moving files (paths become invalid after move)
    per_group_counts = group_class_counts(groups)
    class_sets = group_class_sets_from_counts(per_group_counts)

    # Run stratified split
    assignment = grouped_stratified_split(groups)

    # Move files
    counts = move_files_to_split(groups, assignment, DATASET_DIR)

    # Verify class coverage
    per_split_classes = verify_split_classes(assignment, class_sets)

    # Report
    critical, info = print_report(
        groups, assignment, counts, per_split_classes, per_group_counts,
    )

    # Final structure
    print("\n" + "=" * 60)
    print("FINAL DIRECTORY STRUCTURE")
    print("=" * 60)
    for split_name in ("train", "valid", "test"):
        img_dir = DATASET_DIR / split_name / "images"
        lbl_dir = DATASET_DIR / split_name / "labels"
        n_img = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0
        n_lbl = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
        print(f"  {split_name}/images: {n_img}  labels: {n_lbl}")

    if critical:
        print(f"\n{len(critical)} CRITICAL warning(s) — classes missing from splits!")
        sys.exit(1)
    elif info:
        print(f"\nAll 36 classes present in all splits. {len(info)} minor note(s) above.")
    else:
        print("\nAll 36 classes present in all splits. Split complete.")
