import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".mpeg", ".mpg"}


def natural_sort_key(text: str) -> List:
    parts = re.split(r"(\d+)", text.lower())
    return [int(p) if p.isdigit() else p for p in parts]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sort subfolders by name, read the first two videos in each subfolder, "
            "merge first N/M frames into one video, and save shot clips."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path(r"H:\dataset_small"),
        help="Root folder containing subfolders with two videos each.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(r"H:\dataset_small_output"),
        help="Folder to store merged outputs and shot clips.",
    )
    parser.add_argument(
        "--first_frames",
        type=int,
        default=49,
        help="How many frames to take from the first video.",
    )
    parser.add_argument(
        "--second_frames",
        type=int,
        default=32,
        help="How many frames to take from the second video.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override output fps. If omitted, uses first video fps (fallback 25).",
    )
    return parser.parse_args()


def get_video_files_sorted(folder: Path) -> List[Path]:
    return sorted(
        [
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        ],
        key=lambda x: natural_sort_key(x.name),
    )


def read_first_n_frames(video_path: Path, n: int) -> Tuple[Optional[List], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, 0.0

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    for _ in range(n):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return None, fps
        frames.append(frame)

    cap.release()
    return frames, fps


def write_video(video_path: Path, frames: List, fps: float) -> None:
    if not frames:
        return
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()


def load_processed_map(state_path: Path) -> Dict[str, Dict[str, str]]:
    if not state_path.exists():
        return {}
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def save_processed_map(state_path: Path, processed_map: Dict[str, Dict[str, str]]) -> None:
    state_path.write_text(
        json.dumps(processed_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def is_result_complete(output_dir: Path, seq_name: str) -> bool:
    merged = output_dir / f"{seq_name}.mp4"
    shadow = output_dir / seq_name
    return merged.exists() and (shadow / "shot1.mp4").exists() and (shadow / "shot2.mp4").exists()


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    first_frames_n = args.first_frames
    second_frames_n = args.second_frames

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "processed_subfolders.json"
    processed_map = load_processed_map(state_path)

    subfolders = sorted(
        [p for p in input_dir.iterdir() if p.is_dir()],
        key=lambda x: natural_sort_key(x.name),
    )
    skipped = []
    saved_count = 0
    already_done_count = 0

    for idx, subfolder in enumerate(subfolders, start=1):
        seq_name = f"{idx:06d}"

        if is_result_complete(output_dir, seq_name):
            existing = processed_map.get(subfolder.name, {})
            processed_map[subfolder.name] = {
                "seq_name": seq_name,
                "status": "saved",
                "order": existing.get("order", ""),
                "shot1_video": existing.get("shot1_video", ""),
                "shot2_video": existing.get("shot2_video", ""),
            }
            save_processed_map(state_path, processed_map)
            already_done_count += 1
            print(f"[EXIST] {subfolder.name} -> {seq_name}.mp4")
            continue

        videos = get_video_files_sorted(subfolder)
        if len(videos) < 2:
            skipped.append(subfolder.name)
            processed_map[subfolder.name] = {
                "seq_name": seq_name,
                "status": "skipped",
                "reason": "less_than_2_videos",
            }
            save_processed_map(state_path, processed_map)
            print(f"[SKIP] {subfolder.name}: less than 2 videos")
            continue

        video1, video2 = videos[0], videos[1]
        selected = None
        for first_video, second_video, order_tag in (
            (video1, video2, "forward"),
            (video2, video1, "reverse"),
        ):
            frames1, fps1 = read_first_n_frames(first_video, first_frames_n)
            frames2, _ = read_first_n_frames(second_video, second_frames_n)
            if frames1 is not None and frames2 is not None:
                selected = (first_video, second_video, frames1, frames2, fps1, order_tag)
                break

        if selected is None:
            skipped.append(subfolder.name)
            processed_map[subfolder.name] = {
                "seq_name": seq_name,
                "status": "skipped",
                "reason": "insufficient_frames_both_orders",
            }
            save_processed_map(state_path, processed_map)
            print(
                f"[SKIP] {subfolder.name}: "
                f"both orders failed for ({video1.name}, {video2.name}) "
                f"with requirements ({first_frames_n}, {second_frames_n})"
            )
            continue

        first_video, second_video, frames1, frames2, fps1, order_tag = selected
        saved_count += 1
        merged_video_path = output_dir / f"{seq_name}.mp4"
        shadow_subdir = output_dir / seq_name
        shadow_subdir.mkdir(parents=True, exist_ok=True)

        # Keep original shot clips.
        target_fps = args.fps if args.fps is not None else (fps1 if fps1 and fps1 > 0 else 25.0)
        write_video(shadow_subdir / "shot1.mp4", frames1, target_fps)
        write_video(shadow_subdir / "shot2.mp4", frames2, target_fps)

        # Merge shots. Resize shot2 frames to shot1 size to guarantee compatible output.
        h1, w1 = frames1[0].shape[:2]
        resized_frames2 = [cv2.resize(f, (w1, h1)) if f.shape[:2] != (h1, w1) else f for f in frames2]
        merged_frames = frames1 + resized_frames2
        write_video(merged_video_path, merged_frames, target_fps)

        processed_map[subfolder.name] = {
            "seq_name": seq_name,
            "status": "saved",
            "order": order_tag,
            "shot1_video": first_video.name,
            "shot2_video": second_video.name,
        }
        save_processed_map(state_path, processed_map)

        print(f"[OK] {subfolder.name} -> {merged_video_path.name} ({order_tag})")

    skipped_file = output_dir / "skipped_subfolders.txt"
    skipped_from_map = []
    for subfolder in subfolders:
        info = processed_map.get(subfolder.name, {})
        if info.get("status") == "skipped":
            skipped_from_map.append(subfolder.name)

    with skipped_file.open("w", encoding="utf-8") as f:
        for name in skipped_from_map:
            f.write(f"{name}\n")

    print("\nProcessing complete.")
    print(f"Total subfolders: {len(subfolders)}")
    print(f"Saved outputs: {saved_count}")
    print(f"Already existed: {already_done_count}")
    print(f"Skipped: {len(skipped_from_map)}")
    print(f"Skipped list file: {skipped_file}")
    if skipped_from_map:
        print("Skipped subfolders:")
        for name in skipped_from_map:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
