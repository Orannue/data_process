"""
Remove sample folders under movie_raw_dataset_variants where the sum of
frame counts of all videos in that folder is below a threshold.

Layout matches generate_raw_dataset_variants.py:
  <output_root>/<1|2|3>_character_6_shot/<numeric_id>/shot*.mp4
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2

VIDEO_SUFFIXES = {".mp4", ".mkv", ".avi", ".webm", ".mov", ".m4v"}


def video_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return max(n, 0)
    finally:
        cap.release()


def total_frames_in_sample_dir(sample_dir: Path) -> int:
    total = 0
    for p in sample_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES:
            total += video_frame_count(p)
    return total


def main() -> None:
    p = argparse.ArgumentParser(
        description="Delete variant sample dirs with total video frames below min_frames.",
    )
    p.add_argument(
        "--root",
        type=Path,
        default=Path(r"H:\dataset\movie_raw_dataset_variants"),
        help="Dataset root (contains *_character_6_shot folders).",
    )
    p.add_argument(
        "--min-frames",
        type=int,
        default=417,
        help="Keep dirs with total frames >= this value; remove below.",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete; without this, only print what would be removed.",
    )
    args = p.parse_args()

    root: Path = args.root
    if not root.is_dir():
        raise FileNotFoundError(f"Root not found or not a directory: {root}")

    to_remove: list[tuple[Path, int]] = []

    for case_dir in sorted(root.iterdir()):
        if not case_dir.is_dir():
            continue
        for sample_dir in sorted(case_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            total = total_frames_in_sample_dir(sample_dir)
            if total < args.min_frames:
                to_remove.append((sample_dir, total))

    for path, total in to_remove:
        print(f"{'DELETE' if args.apply else 'WOULD DELETE'} {total} frames\t{path}")

    if args.apply and to_remove:
        for path, _ in to_remove:
            shutil.rmtree(path)
        print(f"Removed {len(to_remove)} sample folder(s).")
    elif not args.apply and to_remove:
        print(f"Dry run: {len(to_remove)} folder(s) would be removed. Pass --apply to delete.")


if __name__ == "__main__":
    main()
