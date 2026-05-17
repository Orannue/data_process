"""
Equally-spaced sample 6-shot combo folders (1_character_6_shot + 2_character_6_shot +
3_character_6_shot), optionally filtered by total video duration. Copies to output_root/6shot.

Source options:
  - Set ``--source_root`` to an existing variants tree with the three 6-shot case dirs, or
  - Set ``--raw_shot_root`` to movie/scene/character_xx/shot (e.g. movie_shot_by_character).
    The script builds only 6-shot variants into ``--source_root`` (see
    ``generate_raw_dataset_variants.py``), then samples.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2

from generate_raw_dataset_variants import process_dataset

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def sorted_combo_dirs(case_dir: Path) -> List[Path]:
    if not case_dir.is_dir():
        return []
    dirs = [p for p in case_dir.iterdir() if p.is_dir()]

    def sort_key(p: Path) -> tuple[int, str]:
        try:
            return (0, f"{int(p.name):020d}")
        except ValueError:
            return (1, p.name.lower())

    return sorted(dirs, key=sort_key)


def linspace_indices(n_total: int, k: int) -> List[int]:
    """Return k equally-spaced indices in [0, n_total-1], unique, sorted."""
    if k <= 0 or n_total <= 0:
        return []
    if n_total <= k:
        return list(range(n_total))
    if k == 1:
        return [n_total // 2]
    raw = [int(round(i * (n_total - 1) / (k - 1))) for i in range(k)]
    out: List[int] = []
    seen: set[int] = set()
    for idx in raw:
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    # If rounding collapsed duplicates, fill gaps by scanning outward (rare for large n).
    if len(out) < k:
        for i in range(n_total):
            if len(out) >= k:
                break
            if i not in seen:
                seen.add(i)
                out.append(i)
        out.sort()
    return out[:k]


def list_videos_in_combo(combo_dir: Path) -> List[Path]:
    files = [p for p in combo_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    return sorted(files, key=lambda p: p.name.lower())


def video_duration_seconds(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0.0
    try:
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            return 0.0
        return frame_count / fps
    finally:
        cap.release()


def combo_total_duration(combo_dir: Path) -> float:
    return sum(video_duration_seconds(p) for p in list_videos_in_combo(combo_dir))


def pick_sources_n_shot_duration(
    root: Path,
    subdir_names: Sequence[str],
    num_shots: int,
    min_total_sec: float,
) -> Tuple[List[Path], List[float]]:
    """Combo dirs with exactly ``num_shots`` videos and total duration >= min_total_sec."""
    merged: List[Path] = []
    for name in subdir_names:
        name = name.strip()
        if not name:
            continue
        merged.extend(sorted_combo_dirs(root / name))
    out_dirs: List[Path] = []
    out_dur: List[float] = []
    for d in merged:
        vids = list_videos_in_combo(d)
        if len(vids) != num_shots:
            continue
        total = sum(video_duration_seconds(p) for p in vids)
        if total >= min_total_sec:
            out_dirs.append(d)
            out_dur.append(total)
    return out_dirs, out_dur


def copy_renumbered(
    sources: Sequence[Path],
    dest_parent: Path,
    max_n: int,
) -> Tuple[int, List[Path]]:
    if dest_parent.exists():
        shutil.rmtree(dest_parent)
    dest_parent.mkdir(parents=True, exist_ok=True)

    n = len(sources)
    if n == 0:
        return 0, []
    take = min(max_n, n)
    idxs = linspace_indices(n, take)
    picked: List[Path] = []
    for new_i, src_idx in enumerate(idxs, start=1):
        src = sources[src_idx]
        picked.append(src)
        dst = dest_parent / str(new_i)
        shutil.copytree(src, dst, dirs_exist_ok=True)
    return take, picked


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raw_shot_root",
        type=str,
        default=None,
        help=(
            "If set, build combo folders from this raw tree (movie/scene/character/shot) "
            "into --source_root before sampling. Example: H:\\dataset\\movie_shot_by_character"
        ),
    )
    p.add_argument(
        "--rebuild_variants",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --raw_shot_root is set, rebuild variants into --source_root (default: true).",
    )
    p.add_argument(
        "--max_1_character_6_shot",
        type=int,
        default=5,
        help="Per-scene cap for 1_character_6_shot when building from --raw_shot_root.",
    )
    p.add_argument(
        "--max_2_character_6_shot",
        type=int,
        default=5,
        help="Per-scene cap for 2_character_6_shot when building from --raw_shot_root.",
    )
    p.add_argument(
        "--max_3_character_6_shot",
        type=int,
        default=5,
        help="Per-scene cap for 3_character_6_shot when building from --raw_shot_root.",
    )
    p.add_argument(
        "--variants_seed",
        type=int,
        default=42,
        help="Random seed for variant generation when using --raw_shot_root.",
    )
    p.add_argument(
        "--source_root",
        type=str,
        default=r"H:\dataset\movie_raw_dataset_variants",
        help="Variants folder: three 6-shot case dirs (see generate_raw_dataset_variants).",
    )
    p.add_argument(
        "--output_root",
        type=str,
        default=r"H:\dataset\movie_raw_dataset_variants",
        help="Parent folder where 6shot/ will be created.",
    )
    p.add_argument("--n_each", type=int, default=300, help="Max 6-shot combo folders to copy.")
    p.add_argument(
        "--six_shot_subdirs",
        type=str,
        default="1_character_6_shot,2_character_6_shot,3_character_6_shot",
        help="Comma-separated subdirs under source_root for 6-shot combos (empty = skip 6-shot).",
    )
    p.add_argument(
        "--num_shots",
        type=int,
        default=6,
        help="Exact number of video files required in each 6-shot combo folder.",
    )
    p.add_argument(
        "--min_total_duration_sec",
        type=float,
        default=30.0,
        help="Minimum sum of per-shot durations (seconds) for 6-shot sampling.",
    )
    p.add_argument(
        "--n_6shot",
        type=int,
        default=None,
        help="Override --n_each for 6-shot only (default: use --n_each).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    n_each = args.n_each
    n_6 = args.n_6shot if args.n_6shot is not None else n_each
    six_subdirs = [s.strip() for s in str(args.six_shot_subdirs).split(",") if s.strip()]

    if args.raw_shot_root:
        raw = Path(args.raw_shot_root)
        if not raw.is_dir():
            raise FileNotFoundError(f"Raw shot root not found: {raw}")
        if args.rebuild_variants:
            print(f"Building variants: {raw} -> {source_root}")
            process_dataset(
                input_root=raw,
                output_root=source_root,
                seed=args.variants_seed,
                max_1c6=args.max_1_character_6_shot,
                max_2c6=args.max_2_character_6_shot,
                max_3c6=args.max_3_character_6_shot,
            )
        elif not source_root.is_dir():
            raise FileNotFoundError(
                f"Source root not found (use --rebuild-variants or run generate_raw_dataset_variants): {source_root}"
            )
    elif not source_root.is_dir():
        raise FileNotFoundError(f"Source root not found: {source_root}")

    if six_subdirs:
        s6, durs = pick_sources_n_shot_duration(
            source_root,
            six_subdirs,
            num_shots=args.num_shots,
            min_total_sec=args.min_total_duration_sec,
        )
        print(
            f"6-shot eligible (exactly {args.num_shots} videos, "
            f"total>={args.min_total_duration_sec}s): {len(s6)} under {six_subdirs}"
        )
        if durs:
            print(
                f"  Eligible duration range: min={min(durs):.2f}s max={max(durs):.2f}s"
            )
        n6, picked6 = copy_renumbered(s6, output_root / "6shot", n_6)
        if picked6:
            picked_dur = [combo_total_duration(p) for p in picked6]
            print(f"Wrote {n6} under {output_root / '6shot'}")
            print(
                f"  Selected 6-shot total duration: min={min(picked_dur):.2f}s "
                f"max={max(picked_dur):.2f}s"
            )
        else:
            print(f"Wrote 0 under {output_root / '6shot'} (no eligible combos or n=0)")
    else:
        print("6-shot: skipped (--six_shot_subdirs empty)")

    print("Done.")


if __name__ == "__main__":
    main()
