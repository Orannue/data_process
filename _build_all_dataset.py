from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import shutil
from pathlib import Path

import cv2


LATENT_CHUNKS = [1, 3, 3, 3, 3, 3, 3, 2]
TWO_SHOT_LATENT_COMBOS = [(4, 17), (7, 14), (10, 11), (13, 8), (16, 5)]
TWO_SHOT_EXTREME_LATENT_COMBOS = [(1, 20), (19, 2)]
TOTAL_LATENT_LENGTH = sum(LATENT_CHUNKS)
TOTAL_OUTPUT_FRAMES = ((TOTAL_LATENT_LENGTH - 1) * 4) + 1
THREE_SHOT_LATENT_CANDIDATES = [
    (first, middle, TOTAL_LATENT_LENGTH - first - middle)
    for first in range(2, TOTAL_LATENT_LENGTH - 3)
    for middle in range(1, TOTAL_LATENT_LENGTH - first - 2)
    if (TOTAL_LATENT_LENGTH - first - middle) >= 3
]
THREE_SHOT_EXTREME_BY_POSITION = {
    0: [(1, 3, 17), (1, 6, 14), (1, 9, 11), (1, 12, 8), (1, 15, 5), (1, 18, 2)],
    1: [(1, 3, 17), (4, 3, 14), (7, 3, 11), (10, 3, 8), (13, 3, 5), (16, 3, 2)],
    2: [(1, 18, 2), (4, 15, 2), (7, 12, 2), (10, 9, 2), (13, 6, 2), (16, 3, 2)],
}
DEFAULT_TWO_SHOT_ROOT = Path(r"H:\dataset\dataset_500\2shot")
DEFAULT_THREE_SHOT_ROOT = Path(r"H:\dataset\dataset_500\3shot")
DEFAULT_OUTPUT_ROOT = Path(r"H:\dataset\dataset_500\all")


def numeric_sort_key(text: str):
    text = text.strip()
    if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
        return (0, int(text))

    parts = re.split(r"(\d+)", text)
    key = []
    for part in parts:
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.lower()))
    return (1, key)


def list_subdirs(root: Path) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: numeric_sort_key(p.name))


def list_mp4s(root: Path) -> list[Path]:
    return sorted(
        [p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"],
        key=lambda p: numeric_sort_key(p.name),
    )


def latent_lengths_to_frame_lengths(latent_lengths: list[int]) -> list[int]:
    if not latent_lengths:
        raise ValueError("latent_lengths cannot be empty")

    frame_lengths = [((latent_lengths[0] - 1) * 4) + 1]
    frame_lengths.extend(length * 4 for length in latent_lengths[1:])
    return frame_lengths


def generate_three_shot_latent_lengths(rng: random.Random) -> tuple[int, int, int]:
    return rng.choice(THREE_SHOT_LATENT_CANDIDATES)


def get_video_info(video_path: Path) -> tuple[int, float, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if frame_count <= 0:
        raise RuntimeError(f"Invalid frame count for video: {video_path}")
    if fps <= 1e-6:
        fps = 23.976
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid resolution for video: {video_path}")
    return frame_count, fps, width, height


def get_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if frame_count <= 0:
        raise RuntimeError(f"Invalid frame count for video: {video_path}")
    return frame_count


def list_frame_counts(videos: list[Path]) -> list[int]:
    return [get_frame_count(video_path) for video_path in videos]


def get_valid_latent_choices(videos: list[Path], latent_choices: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    frame_counts = list_frame_counts(videos)
    return get_valid_latent_choices_from_frame_counts(frame_counts, latent_choices)


def get_valid_latent_choices_from_frame_counts(
    frame_counts: list[int], latent_choices: list[tuple[int, ...]]
) -> list[tuple[int, ...]]:
    usable_frames = [count - 2 for count in frame_counts]
    valid_choices: list[tuple[int, ...]] = []
    for latent_choice in latent_choices:
        frame_lengths = latent_lengths_to_frame_lengths(list(latent_choice))
        if len(frame_lengths) != len(usable_frames):
            continue
        if all(usable >= needed for usable, needed in zip(usable_frames, frame_lengths)):
            valid_choices.append(latent_choice)
    return valid_choices


def choose_two_shot_strategy(videos: list[Path], rng: random.Random) -> tuple[list[Path], list[int], str]:
    frame_counts = list_frame_counts(videos)
    reversed_videos = list(reversed(videos))
    reversed_frame_counts = list(reversed(frame_counts))
    attempts = [
        ("normal", videos, frame_counts, TWO_SHOT_LATENT_COMBOS),
        ("extreme", videos, frame_counts, TWO_SHOT_EXTREME_LATENT_COMBOS),
        ("reversed_normal", reversed_videos, reversed_frame_counts, TWO_SHOT_LATENT_COMBOS),
        ("reversed_extreme", reversed_videos, reversed_frame_counts, TWO_SHOT_EXTREME_LATENT_COMBOS),
    ]

    for strategy_name, candidate_videos, candidate_frame_counts, latent_pool in attempts:
        valid_choices = get_valid_latent_choices_from_frame_counts(candidate_frame_counts, latent_pool)
        if valid_choices:
            return candidate_videos, list(rng.choice(valid_choices)), strategy_name

    raise RuntimeError("no_valid_latent_split_for_available_frames")


def choose_three_shot_strategy(videos: list[Path], rng: random.Random) -> tuple[list[Path], list[int], str]:
    original_order = tuple(range(len(videos)))
    permutations_to_try = [original_order]
    permutations_to_try.extend(
        perm for perm in itertools.permutations(range(len(videos))) if perm != original_order
    )

    for perm in permutations_to_try:
        candidate_videos = [videos[i] for i in perm]
        candidate_frame_counts = list_frame_counts(candidate_videos)

        valid_normal_choices = get_valid_latent_choices_from_frame_counts(
            candidate_frame_counts, THREE_SHOT_LATENT_CANDIDATES
        )
        if valid_normal_choices:
            strategy_name = "normal" if perm == original_order else f"permuted_normal:{perm}"
            return candidate_videos, list(rng.choice(valid_normal_choices)), strategy_name

        shortest_position = min(range(len(candidate_frame_counts)), key=lambda i: candidate_frame_counts[i])
        extreme_pool = THREE_SHOT_EXTREME_BY_POSITION[shortest_position]
        valid_extreme_choices = get_valid_latent_choices_from_frame_counts(candidate_frame_counts, extreme_pool)
        if valid_extreme_choices:
            position_name = ["first", "middle", "third"][shortest_position]
            strategy_name = (
                f"extreme_{position_name}"
                if perm == original_order
                else f"permuted_extreme_{position_name}:{perm}"
            )
            return candidate_videos, list(rng.choice(valid_extreme_choices)), strategy_name

    raise RuntimeError("no_valid_latent_split_for_available_frames")


def choose_contiguous_window(frame_count: int, segment_length: int, rng: random.Random) -> tuple[int, int]:
    usable_frames = frame_count - 2
    if usable_frames < segment_length:
        raise RuntimeError(
            f"Video has only {usable_frames} usable frames but needs {segment_length}: total={frame_count}"
        )

    start_min = 1
    start_max = frame_count - 1 - segment_length
    start = rng.randint(start_min, start_max)
    end = start + segment_length
    return start, end


def read_frame_segment(video_path: Path, start_frame: int, segment_length: int) -> tuple[list, float, tuple[int, int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-6:
        fps = 23.976

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid resolution for video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
    frames = []
    for _ in range(segment_length):
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Failed to read frame from {video_path} at start {start_frame}")
        frames.append(frame)
    cap.release()
    return frames, fps, (width, height)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_video(video_path: Path, frames: list, fps: float, size: tuple[int, int]) -> None:
    ensure_parent(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video: {video_path}")

    for frame in frames:
        if frame.shape[1] != size[0] or frame.shape[0] != size[1]:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        writer.write(frame)
    writer.release()


def process_folder(
    source_folder: Path,
    videos: list[Path],
    output_folder: Path,
    merged_video_path: Path,
    latent_lengths: list[int],
    rng: random.Random,
) -> dict:
    segment_lengths = latent_lengths_to_frame_lengths(latent_lengths)
    if len(videos) != len(segment_lengths):
        raise RuntimeError(
            f"Folder {source_folder} has {len(videos)} videos but expected {len(segment_lengths)}"
        )

    output_folder.mkdir(parents=True, exist_ok=True)
    all_frames = []
    merged_fps = None
    merged_size = None
    segments_meta = []

    for idx, (video_path, segment_length) in enumerate(zip(videos, segment_lengths), start=1):
        frame_count, fps, width, height = get_video_info(video_path)
        start_frame, end_frame = choose_contiguous_window(frame_count, segment_length, rng)
        frames, clip_fps, clip_size = read_frame_segment(video_path, start_frame, segment_length)

        segment_path = output_folder / f"shot{idx}.mp4"
        write_video(segment_path, frames, clip_fps, clip_size)

        if merged_fps is None:
            merged_fps = clip_fps
            merged_size = clip_size

        resized_frames = frames
        if clip_size != merged_size:
            resized_frames = [cv2.resize(frame, merged_size, interpolation=cv2.INTER_AREA) for frame in frames]
        all_frames.extend(resized_frames)

        segments_meta.append(
            {
                "source_video": str(video_path),
                "output_segment": str(segment_path),
                "frame_count": frame_count,
                "selected_latent_length": latent_lengths[idx - 1],
                "selected_length": segment_length,
                "selected_start_frame": start_frame,
                "selected_end_frame_exclusive": end_frame,
                "fps": clip_fps,
                "width": width,
                "height": height,
            }
        )

    write_video(merged_video_path, all_frames, merged_fps, merged_size)
    return {
        "source_folder": str(source_folder),
        "video_names": [video_path.name for video_path in videos],
        "output_folder": str(output_folder),
        "merged_video": str(merged_video_path),
        "latent_lengths": latent_lengths,
        "segment_lengths": segment_lengths,
        "total_output_frames": len(all_frames),
        "segments": segments_meta,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build combined 2-shot and 3-shot dataset clips.")
    parser.add_argument("--two-shot-root", type=Path, default=DEFAULT_TWO_SHOT_ROOT)
    parser.add_argument("--three-shot-root", type=Path, default=DEFAULT_THREE_SHOT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility.")
    parser.add_argument("--limit-two", type=int, default=None, help="Only process the first N 2-shot folders.")
    parser.add_argument("--limit-three", type=int, default=None, help="Only process the first N 3-shot folders.")
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete the output root before writing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    if not args.two_shot_root.is_dir():
        raise SystemExit(f"Missing 2-shot source directory: {args.two_shot_root}")
    if not args.three_shot_root.is_dir():
        raise SystemExit(f"Missing 3-shot source directory: {args.three_shot_root}")

    if args.clear_output and args.output_root.exists():
        shutil.rmtree(args.output_root)

    args.output_root.mkdir(parents=True, exist_ok=True)

    existing_entries = list(args.output_root.iterdir())
    if existing_entries:
        raise SystemExit(f"Output directory is not empty: {args.output_root}")

    manifest_path = args.output_root / "manifest.jsonl"
    skipped_path = args.output_root / "skipped.jsonl"
    selection_details_path = args.output_root / "selection_details.json"
    summary_path = args.output_root / "summary.json"

    two_shot_folders = list_subdirs(args.two_shot_root)
    three_shot_folders = list_subdirs(args.three_shot_root)
    if args.limit_two is not None:
        two_shot_folders = two_shot_folders[: args.limit_two]
    if args.limit_three is not None:
        three_shot_folders = three_shot_folders[: args.limit_three]

    output_index = 1
    skipped_count = 0
    strategy_counts: dict[str, int] = {}
    switched_order_examples: list[dict] = []
    skipped_examples: list[dict] = []
    selection_details: list[dict] = []
    with manifest_path.open("w", encoding="utf-8") as manifest, skipped_path.open("w", encoding="utf-8") as skipped:
        for source_folder in two_shot_folders:
            videos = list_mp4s(source_folder)
            try:
                chosen_videos, latent_lengths, selection_strategy = choose_two_shot_strategy(videos, rng)
            except RuntimeError:
                skipped_entry = {
                    "output_index": output_index,
                    "type": "2shot",
                    "source_folder": str(source_folder),
                    "reason": "no_valid_latent_split_for_available_frames",
                    "frame_counts": list_frame_counts(videos),
                }
                skipped.write(
                    json.dumps(skipped_entry, ensure_ascii=False) + "\n"
                )
                skipped_examples.append(skipped_entry)
                print(f"[skip 2shot] {source_folder.name} output_index={output_index} no valid latent split")
                skipped_count += 1
                output_index += 1
                continue

            output_folder = args.output_root / str(output_index)
            merged_video_path = args.output_root / f"{output_index}.mp4"
            meta = process_folder(source_folder, chosen_videos, output_folder, merged_video_path, latent_lengths, rng)
            meta["type"] = "2shot"
            meta["output_index"] = output_index
            meta["selection_strategy"] = selection_strategy
            meta["order_switched"] = selection_strategy.startswith("reversed")
            manifest.write(json.dumps(meta, ensure_ascii=False) + "\n")
            strategy_key = f"2shot:{selection_strategy}"
            strategy_counts[strategy_key] = strategy_counts.get(strategy_key, 0) + 1
            selection_details.append(
                {
                    "output_index": output_index,
                    "type": "2shot",
                    "source_folder": str(source_folder),
                    "video_name": meta["video_names"],
                    "switch_latent_frames": meta["latent_lengths"],
                    "switch_frames": meta["segment_lengths"],
                    "selection_strategy": selection_strategy,
                    "order_switched": meta["order_switched"],
                }
            )
            if meta["order_switched"]:
                switched_order_examples.append(
                    {
                        "output_index": output_index,
                        "type": "2shot",
                        "source_folder": str(source_folder),
                        "video_name": meta["video_names"],
                        "selection_strategy": selection_strategy,
                    }
                )
            print(
                f"[2shot] {source_folder.name} -> {output_index} strategy={selection_strategy} latent={latent_lengths} "
                f"frames={meta['segment_lengths']}"
            )
            output_index += 1

        for source_folder in three_shot_folders:
            videos = list_mp4s(source_folder)
            try:
                chosen_videos, latent_lengths, selection_strategy = choose_three_shot_strategy(videos, rng)
            except RuntimeError:
                skipped_entry = {
                    "output_index": output_index,
                    "type": "3shot",
                    "source_folder": str(source_folder),
                    "reason": "no_valid_latent_split_for_available_frames",
                    "frame_counts": list_frame_counts(videos),
                }
                skipped.write(
                    json.dumps(skipped_entry, ensure_ascii=False) + "\n"
                )
                skipped_examples.append(skipped_entry)
                print(f"[skip 3shot] {source_folder.name} output_index={output_index} no valid latent split")
                skipped_count += 1
                output_index += 1
                continue

            output_folder = args.output_root / str(output_index)
            merged_video_path = args.output_root / f"{output_index}.mp4"
            meta = process_folder(source_folder, chosen_videos, output_folder, merged_video_path, latent_lengths, rng)
            meta["type"] = "3shot"
            meta["output_index"] = output_index
            meta["selection_strategy"] = selection_strategy
            meta["order_switched"] = selection_strategy.startswith("permuted")
            manifest.write(json.dumps(meta, ensure_ascii=False) + "\n")
            strategy_key = f"3shot:{selection_strategy}"
            strategy_counts[strategy_key] = strategy_counts.get(strategy_key, 0) + 1
            selection_details.append(
                {
                    "output_index": output_index,
                    "type": "3shot",
                    "source_folder": str(source_folder),
                    "video_name": meta["video_names"],
                    "switch_latent_frames": meta["latent_lengths"],
                    "switch_frames": meta["segment_lengths"],
                    "selection_strategy": selection_strategy,
                    "order_switched": meta["order_switched"],
                }
            )
            if meta["order_switched"]:
                switched_order_examples.append(
                    {
                        "output_index": output_index,
                        "type": "3shot",
                        "source_folder": str(source_folder),
                        "video_name": meta["video_names"],
                        "selection_strategy": selection_strategy,
                    }
                )
            print(
                f"[3shot] {source_folder.name} -> {output_index} strategy={selection_strategy} latent={latent_lengths} "
                f"frames={meta['segment_lengths']}"
            )
            output_index += 1

    total_folders = len(list_subdirs(args.output_root))
    total_merged_videos = len([p for p in args.output_root.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"])
    summary = {
        "requested_examples": len(two_shot_folders) + len(three_shot_folders),
        "written_examples": total_folders,
        "written_merged_videos": total_merged_videos,
        "skipped_count": skipped_count,
        "skipped_output_indices": [entry["output_index"] for entry in skipped_examples],
        "skipped_examples": skipped_examples,
        "switched_order_count": len(switched_order_examples),
        "switched_order_examples": switched_order_examples,
        "strategy_counts": strategy_counts,
        "selection_details_file": str(selection_details_path),
        "manifest_file": str(manifest_path),
        "skipped_file": str(skipped_path),
        "total_output_frames": TOTAL_OUTPUT_FRAMES,
    }
    selection_details_path.write_text(json.dumps(selection_details, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Done. folders={total_folders}, merged_videos={total_merged_videos}, "
        f"manifest={manifest_path}, skipped={skipped_count}, switched={len(switched_order_examples)}, "
        f"total_output_frames={TOTAL_OUTPUT_FRAMES}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
