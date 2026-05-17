from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


DEFAULT_ROOT = Path(r"F:\dataset\out_22_21")
DEFAULT_OUTPUT = Path(r"F:\dataset\out_22_21_cropped")
CROP_RE = re.compile(r"crop=(\d+):(\d+):(\d+):(\d+)")


def list_target_videos(root: Path) -> list[Path]:
    return sorted(root.glob("*.mp4"), key=lambda p: p.name.lower())


def ffprobe_size(video_path: Path) -> tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    width_str, height_str = result.stdout.strip().split("x")
    return int(width_str), int(height_str)


def detect_crop(
    video_path: Path,
    cropdetect_limit: int,
    detect_frames: int,
    round_to: int,
    reset_count: int,
) -> tuple[int, int, int, int] | None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(video_path),
        "-vf",
        f"cropdetect={cropdetect_limit}:{round_to}:{reset_count}",
        "-frames:v",
        str(detect_frames),
        "-f",
        "null",
        os.devnull,
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    matches = CROP_RE.findall(result.stderr)
    if not matches:
        return None

    counter = Counter(tuple(int(x) for x in match) for match in matches)
    return counter.most_common(1)[0][0]


def crop_video_to_output(
    video_path: Path, output_path: Path, crop: tuple[int, int, int, int], crf: int, preset: str
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height, x, y = crop
    temp_path = output_path.with_suffix(".cropping.tmp.mp4")
    if temp_path.exists():
        temp_path.unlink()

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-i",
        str(video_path),
        "-vf",
        f"crop={width}:{height}:{x}:{y}",
        "-map",
        "0",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-c:a",
        "copy",
        str(temp_path),
    ]
    subprocess.run(cmd, check=True)
    shutil.move(str(temp_path), str(output_path))


def copy_video_to_output(video_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(video_path, output_path)


def process_video(
    video_path: Path,
    input_root: Path,
    output_root: Path,
    limit_frames: int,
    cropdetect_limit: int,
    round_to: int,
    reset_count: int,
    crop_margin: int,
    crf: int,
    preset: str,
    dry_run: bool,
) -> dict:
    rel = video_path.relative_to(input_root)
    output_path = output_root / rel

    original_width, original_height = ffprobe_size(video_path)
    detected = detect_crop(
        video_path,
        cropdetect_limit=cropdetect_limit,
        detect_frames=limit_frames,
        round_to=round_to,
        reset_count=reset_count,
    )

    if detected is None:
        if not dry_run:
            copy_video_to_output(video_path, output_path)
        return {
            "video_path": str(video_path),
            "output_path": str(output_path),
            "status": "no_crop_detected",
            "original_size": [original_width, original_height],
        }

    crop_width, crop_height, crop_x, crop_y = detected

    # Preserve full width and only remove top/bottom black bars.
    # Optional margin adds a few rows back so dark scenes / soft letterbox edges are not over-cropped.
    y0 = max(0, crop_y - crop_margin)
    y1 = min(original_height, crop_y + crop_height + crop_margin)
    final_height = y1 - y0
    final_crop = (original_width, final_height, 0, y0)
    if final_crop == (original_width, original_height, 0, 0):
        if not dry_run:
            copy_video_to_output(video_path, output_path)
        return {
            "video_path": str(video_path),
            "output_path": str(output_path),
            "status": "unchanged",
            "original_size": [original_width, original_height],
            "detected_crop": [crop_width, crop_height, crop_x, crop_y],
            "applied_crop": [original_width, original_height, 0, 0],
        }

    if dry_run:
        return {
            "video_path": str(video_path),
            "output_path": str(output_path),
            "status": "would_crop",
            "original_size": [original_width, original_height],
            "detected_crop": [crop_width, crop_height, crop_x, crop_y],
            "applied_crop": list(final_crop),
            "new_size": [final_crop[0], final_crop[1]],
        }

    crop_video_to_output(video_path, output_path, final_crop, crf=crf, preset=preset)
    new_width, new_height = ffprobe_size(output_path)
    return {
        "video_path": str(video_path),
        "output_path": str(output_path),
        "status": "cropped",
        "original_size": [original_width, original_height],
        "detected_crop": [crop_width, crop_height, crop_x, crop_y],
        "applied_crop": list(final_crop),
        "new_size": [new_width, new_height],
    }


def _process_video_mp(payload: tuple) -> dict:
    """Module-level worker for ProcessPoolExecutor (must be picklable on Windows)."""
    (
        video_path_s,
        input_root_s,
        output_root_s,
        limit_frames,
        cropdetect_limit,
        round_to,
        reset_count,
        crop_margin,
        crf,
        preset,
        dry_run,
    ) = payload
    return process_video(
        Path(video_path_s),
        Path(input_root_s),
        Path(output_root_s),
        limit_frames=limit_frames,
        cropdetect_limit=cropdetect_limit,
        round_to=round_to,
        reset_count=reset_count,
        crop_margin=crop_margin,
        crf=crf,
        preset=preset,
        dry_run=dry_run,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect and remove top/bottom black bars for clip videos.")
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Input folder; only *.mp4 directly in this folder (not in subfolders).",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output folder; same filenames as under --root.")
    parser.add_argument("--limit-frames", type=int, default=60, help="Number of frames used for crop detection.")
    parser.add_argument(
        "--cropdetect-limit",
        type=int,
        default=16,
        metavar="L",
        help="cropdetect luminance threshold (0–255). Lower = only true black counts, less aggressive crop; default 16.",
    )
    parser.add_argument("--round", type=int, default=2, help="Round crop height to this multiple.")
    parser.add_argument("--reset-count", type=int, default=0, help="cropdetect reset count.")
    parser.add_argument(
        "--crop-margin",
        type=int,
        default=8,
        metavar="PX",
        help="Extra pixels to keep above/below detected content (symmetric when possible); reduces over-crop.",
    )
    parser.add_argument("--crf", type=int, default=18, help="libx264 CRF value.")
    parser.add_argument("--preset", default="fast", help="libx264 preset.")
    parser.add_argument("--max-videos", type=int, default=None, help="Only process the first N videos.")
    parser.add_argument("--dry-run", action="store_true", help="Only report detected crop values.")
    default_jobs = max(1, min(4, os.cpu_count() or 1))
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=default_jobs,
        help=f"Parallel worker processes (ffmpeg is subprocess-heavy; default {default_jobs}). Use 1 for sequential.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.root.is_dir():
        raise SystemExit(f"Missing root directory: {args.root}")
    args.output.mkdir(parents=True, exist_ok=True)

    videos = list_target_videos(args.root)
    if args.max_videos is not None:
        videos = videos[: args.max_videos]
    if not videos:
        raise SystemExit(f"No target videos found under: {args.root}")

    crop_margin = max(0, args.crop_margin)
    jobs = max(1, args.jobs)

    if jobs == 1:
        report: list[dict] = []
        for idx, video_path in enumerate(videos, start=1):
            info = process_video(
                video_path,
                args.root,
                args.output,
                limit_frames=args.limit_frames,
                cropdetect_limit=args.cropdetect_limit,
                round_to=args.round,
                reset_count=args.reset_count,
                crop_margin=crop_margin,
                crf=args.crf,
                preset=args.preset,
                dry_run=args.dry_run,
            )
            report.append(info)
            print(f"[{idx}/{len(videos)}] {video_path.relative_to(args.root)}: {info['status']}")
    else:
        payloads = [
            (
                str(vp),
                str(args.root),
                str(args.output),
                args.limit_frames,
                args.cropdetect_limit,
                args.round,
                args.reset_count,
                crop_margin,
                args.crf,
                args.preset,
                args.dry_run,
            )
            for vp in videos
        ]
        report = []
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            for idx, info in enumerate(ex.map(_process_video_mp, payloads), start=1):
                report.append(info)
                rel = Path(info["video_path"]).relative_to(args.root)
                print(f"[{idx}/{len(videos)}] {rel}: {info['status']}")

    cropped_count = 0
    unchanged_count = 0
    no_crop_count = 0
    for info in report:
        status = info["status"]
        if status in {"cropped", "would_crop"}:
            cropped_count += 1
        elif status == "unchanged":
            unchanged_count += 1
        else:
            no_crop_count += 1

    report_path = args.output / "crop_black_bars_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"Done. videos={len(videos)}, cropped={cropped_count}, unchanged={unchanged_count}, "
        f"no_crop={no_crop_count}, report={report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
