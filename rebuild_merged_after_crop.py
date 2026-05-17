from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import cv2


DEFAULT_ROOT = Path(r"H:\dataset\dataset_500\all")


def numeric_sort_key(text: str):
    text = text.strip()
    if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
        return (0, int(text))
    return (1, text.lower())


def list_subdirs(root: Path) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: numeric_sort_key(p.name))


def list_mp4s(root: Path) -> list[Path]:
    return sorted([p for p in root.glob("*.mp4") if p.is_file()], key=lambda p: numeric_sort_key(p.name))


def probe_video(video_path: Path) -> tuple[int, int, float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video size: {video_path}")
    if fps <= 1e-6:
        fps = 23.976
    if frame_count <= 0:
        raise RuntimeError(f"Invalid frame count: {video_path}")
    return width, height, fps, frame_count


def aspect_ratio_label(width: int, height: int) -> str:
    g = math.gcd(width, height)
    return f"{width // g}:{height // g}"


def rebuild_folder(folder: Path) -> dict:
    clips = list_mp4s(folder)
    if not clips:
        raise RuntimeError(f"No clips found in folder: {folder}")

    target_width, target_height, target_fps, _ = probe_video(clips[0])
    output_path = folder.parent / f"{folder.name}.mp4"
    temp_output = output_path.with_suffix(".rebuild.tmp.mp4")
    if temp_output.exists():
        temp_output.unlink()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_output), fourcc, target_fps, (target_width, target_height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create merged video: {temp_output}")

    clip_infos = []
    total_frames_written = 0
    for clip in clips:
        width, height, fps, frame_count = probe_video(clip)
        cap = cv2.VideoCapture(str(clip))
        if not cap.isOpened():
            writer.release()
            raise RuntimeError(f"Cannot read clip: {clip}")

        written_from_clip = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] != target_width or frame.shape[0] != target_height:
                frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
            written_from_clip += 1
        cap.release()

        total_frames_written += written_from_clip
        clip_infos.append(
            {
                "clip_name": clip.name,
                "clip_size": [width, height],
                "clip_ratio": aspect_ratio_label(width, height),
                "clip_fps": fps,
                "clip_frames": frame_count,
                "written_frames": written_from_clip,
                "resized_for_merge": width != target_width or height != target_height,
            }
        )

    writer.release()
    temp_output.replace(output_path)

    return {
        "output_index": int(folder.name),
        "folder": str(folder),
        "merged_video": str(output_path),
        "merged_size": [target_width, target_height],
        "merged_ratio": aspect_ratio_label(target_width, target_height),
        "merged_fps": target_fps,
        "total_frames": total_frames_written,
        "clip_count": len(clips),
        "clips": clip_infos,
    }


def summarize(entries: list[dict]) -> dict:
    merged_size_counts: Counter[str] = Counter()
    merged_ratio_counts: Counter[str] = Counter()
    clip_size_counts: Counter[str] = Counter()
    clip_ratio_counts: Counter[str] = Counter()
    resized_merge_count = 0

    for entry in entries:
        merged_size = f"{entry['merged_size'][0]}x{entry['merged_size'][1]}"
        merged_size_counts[merged_size] += 1
        merged_ratio_counts[entry["merged_ratio"]] += 1
        for clip in entry["clips"]:
            clip_size = f"{clip['clip_size'][0]}x{clip['clip_size'][1]}"
            clip_size_counts[clip_size] += 1
            clip_ratio_counts[clip["clip_ratio"]] += 1
            if clip["resized_for_merge"]:
                resized_merge_count += 1

    return {
        "merged_video_count": len(entries),
        "clip_video_count": sum(entry["clip_count"] for entry in entries),
        "merged_size_counts": dict(sorted(merged_size_counts.items())),
        "merged_ratio_counts": dict(sorted(merged_ratio_counts.items())),
        "clip_size_counts": dict(sorted(clip_size_counts.items())),
        "clip_ratio_counts": dict(sorted(clip_ratio_counts.items())),
        "resized_clip_count_for_merge": resized_merge_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild merged videos from cropped clips and report sizes.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--max-folders", type=int, default=None, help="Only rebuild the first N folders.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.root.is_dir():
        raise SystemExit(f"Missing root directory: {args.root}")

    folders = list_subdirs(args.root)
    if args.max_folders is not None:
        folders = folders[: args.max_folders]
    if not folders:
        raise SystemExit(f"No numbered folders found under: {args.root}")

    entries = []
    for idx, folder in enumerate(folders, start=1):
        entry = rebuild_folder(folder)
        entries.append(entry)
        print(f"[{idx}/{len(folders)}] rebuilt {folder.name}.mp4 size={entry['merged_size'][0]}x{entry['merged_size'][1]}")

    detail_path = args.root / "merged_after_crop_report.json"
    summary_path = args.root / "merged_after_crop_summary.json"
    detail_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = summarize(entries)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"Done. merged={summary['merged_video_count']}, "
        f"detail={detail_path}, summary={summary_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
