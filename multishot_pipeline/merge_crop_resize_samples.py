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
from typing import Sequence

import cv2


DEFAULT_INPUT_ROOT = Path(r"H:\dataset\movie_multishot_output\samples_cropped")
TARGET_WIDTH = 832
TARGET_HEIGHT = 480
CROP_RE = re.compile(r"crop=(\d+):(\d+):(\d+):(\d+)")


def natural_sort_key(value: str):
    parts = re.split(r"(\d+)", value)
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


def is_candidate_clip(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() != ".mp4":
        return False
    name = path.name.lower()
    if name == "merged.mp4":
        return False
    if name.endswith(".tmp.mp4") or ".tmp." in name:
        return False
    return True


def list_sample_dirs(input_root: Path) -> list[Path]:
    dirs = {p.parent for p in input_root.rglob("*.mp4") if is_candidate_clip(p)}
    return sorted(
        dirs,
        key=lambda p: natural_sort_key(str(p.relative_to(input_root))),
    )


def list_clips(sample_dir: Path) -> list[Path]:
    return sorted(
        [p for p in sample_dir.glob("*.mp4") if is_candidate_clip(p)],
        key=lambda p: natural_sort_key(p.name),
    )


def output_dir_for_sample(
    sample_dir: Path, input_root: Path, output_root: Path | None
) -> Path:
    if output_root is None:
        return sample_dir
    rel = sample_dir.relative_to(input_root)
    if str(rel) == ".":
        return output_root / sample_dir.name
    return output_root / rel


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


def ffprobe_frame_count(video_path: Path) -> int:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    text = result.stdout.strip()
    return int(text) if text.isdigit() else 0


def probe_video_cv(video_path: Path) -> tuple[int, int, float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video size: {video_path}")
    if fps <= 1e-6:
        fps = 25.0
    return width, height, fps, frame_count


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


def top_bottom_crop(
    detected: tuple[int, int, int, int] | None,
    original_width: int,
    original_height: int,
    crop_margin: int,
) -> tuple[int, int, int, int]:
    if detected is None:
        return original_width, original_height, 0, 0

    _, crop_height, _, crop_y = detected
    y0 = max(0, crop_y - crop_margin)
    y1 = min(original_height, crop_y + crop_height + crop_margin)

    if (y1 - y0) % 2:
        if y1 < original_height:
            y1 += 1
        elif y0 > 0:
            y0 -= 1

    final_height = max(2, y1 - y0)
    return original_width, final_height, 0, y0


def merge_clips_to_temp(
    clips: Sequence[Path],
    temp_path: Path,
    codec: str,
) -> dict:
    if not clips:
        raise RuntimeError("No clips to merge.")

    target_width, target_height, target_fps, _ = probe_video_cv(clips[0])
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    if temp_path.exists():
        temp_path.unlink()

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(
        str(temp_path), fourcc, target_fps, (target_width, target_height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create temp merged video: {temp_path}")

    clip_infos = []
    total_written = 0
    try:
        for clip in clips:
            width, height, fps, frame_count = probe_video_cv(clip)
            cap = cv2.VideoCapture(str(clip))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot read clip: {clip}")

            written = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[1] != target_width or frame.shape[0] != target_height:
                    frame = cv2.resize(
                        frame,
                        (target_width, target_height),
                        interpolation=cv2.INTER_AREA,
                    )
                writer.write(frame)
                written += 1
            cap.release()

            total_written += written
            clip_infos.append(
                {
                    "clip": str(clip),
                    "size": [width, height],
                    "fps": fps,
                    "frames": frame_count,
                    "written_frames": written,
                    "resized_for_merge": width != target_width
                    or height != target_height,
                }
            )
    finally:
        writer.release()

    if total_written <= 0:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError("Merged video has no frames.")

    return {
        "temp_merged": str(temp_path),
        "merged_size": [target_width, target_height],
        "merged_fps": target_fps,
        "total_frames": total_written,
        "clips": clip_infos,
    }


def encode_crop_resize(
    input_path: Path,
    output_path: Path,
    crop: tuple[int, int, int, int],
    target_width: int,
    target_height: int,
    crf: int,
    preset: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_path.with_suffix(".final.tmp.mp4")
    if temp_output.exists():
        temp_output.unlink()

    crop_width, crop_height, crop_x, crop_y = crop
    filters = []
    input_width, input_height = ffprobe_size(input_path)
    if crop != (input_width, input_height, 0, 0):
        filters.append(f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y}")
    filters.extend(
        [
            f"scale={target_width}:{target_height}:force_original_aspect_ratio=increase",
            f"crop={target_width}:{target_height}:(iw-{target_width})/2:(ih-{target_height})/2",
        ]
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-vf",
        ",".join(filters),
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(temp_output),
    ]
    subprocess.run(cmd, check=True)
    if output_path.exists():
        output_path.unlink()
    shutil.move(str(temp_output), str(output_path))


def process_sample(
    sample_dir: Path,
    input_root: Path,
    output_root: Path | None,
    target_width: int,
    target_height: int,
    min_clips: int,
    detect_frames: int,
    cropdetect_limit: int,
    round_to: int,
    reset_count: int,
    crop_margin: int,
    crf: int,
    preset: str,
    codec: str,
    overwrite: bool,
    keep_temp: bool,
) -> dict:
    clips = list_clips(sample_dir)
    out_dir = output_dir_for_sample(sample_dir, input_root, output_root)
    output_path = out_dir / "merged.mp4"
    rel = str(sample_dir.relative_to(input_root))
    if str(rel) == ".":
        rel = sample_dir.name

    if len(clips) < min_clips:
        return {
            "sample": rel,
            "sample_dir": str(sample_dir),
            "output_path": str(output_path),
            "status": "skipped_too_few_clips",
            "clip_count": len(clips),
        }

    if output_path.exists() and not overwrite:
        return {
            "sample": rel,
            "sample_dir": str(sample_dir),
            "output_path": str(output_path),
            "status": "skipped_exists",
            "clip_count": len(clips),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    temp_merged = out_dir / "merged.merge.tmp.mp4"

    try:
        merge_info = merge_clips_to_temp(clips, temp_merged, codec=codec)
        original_width, original_height = ffprobe_size(temp_merged)
        detected = detect_crop(
            temp_merged,
            cropdetect_limit=cropdetect_limit,
            detect_frames=detect_frames,
            round_to=round_to,
            reset_count=reset_count,
        )
        applied_crop = top_bottom_crop(
            detected,
            original_width=original_width,
            original_height=original_height,
            crop_margin=crop_margin,
        )

        encode_crop_resize(
            temp_merged,
            output_path,
            crop=applied_crop,
            target_width=target_width,
            target_height=target_height,
            crf=crf,
            preset=preset,
        )
        final_width, final_height = ffprobe_size(output_path)
        final_frames = ffprobe_frame_count(output_path)

        return {
            "sample": rel,
            "sample_dir": str(sample_dir),
            "output_path": str(output_path),
            "status": "ok",
            "clip_count": len(clips),
            "target_size": [target_width, target_height],
            "detected_crop": list(detected) if detected else None,
            "applied_crop": list(applied_crop),
            "final_size": [final_width, final_height],
            "final_frames": final_frames,
            **merge_info,
        }
    finally:
        if temp_merged.exists() and not keep_temp:
            temp_merged.unlink()


def _process_sample_mp(payload: tuple) -> dict:
    try:
        return process_sample(
            sample_dir=Path(payload[0]),
            input_root=Path(payload[1]),
            output_root=Path(payload[2]) if payload[2] else None,
            target_width=payload[3],
            target_height=payload[4],
            min_clips=payload[5],
            detect_frames=payload[6],
            cropdetect_limit=payload[7],
            round_to=payload[8],
            reset_count=payload[9],
            crop_margin=payload[10],
            crf=payload[11],
            preset=payload[12],
            codec=payload[13],
            overwrite=payload[14],
            keep_temp=payload[15],
        )
    except Exception as err:
        return {
            "sample_dir": payload[0],
            "status": "failed",
            "error": str(err),
        }


def summarize(report: list[dict]) -> dict:
    status_counts = Counter(item.get("status", "unknown") for item in report)
    final_size_counts = Counter(
        f"{item['final_size'][0]}x{item['final_size'][1]}"
        for item in report
        if item.get("final_size")
    )
    return {
        "sample_count": len(report),
        "status_counts": dict(sorted(status_counts.items())),
        "final_size_counts": dict(sorted(final_size_counts.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge shot clips per sample, then run black-bar detection, crop, "
            "and resize/center-crop only on the merged video."
        )
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Optional mirrored output root. If omitted, write merged.mp4 into each "
            "source sample folder."
        ),
    )
    parser.add_argument("--target-width", type=int, default=TARGET_WIDTH)
    parser.add_argument("--target-height", type=int, default=TARGET_HEIGHT)
    parser.add_argument("--min-clips", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--limit-frames", type=int, default=60)
    parser.add_argument("--cropdetect-limit", type=int, default=16)
    parser.add_argument("--round", type=int, default=2)
    parser.add_argument("--reset-count", type=int, default=0)
    parser.add_argument("--crop-margin", type=int, default=8)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="fast")
    parser.add_argument("--codec", default="mp4v", help="OpenCV codec for temp merge.")
    default_jobs = max(1, min(4, os.cpu_count() or 1))
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=default_jobs,
        help=f"Parallel worker processes; default {default_jobs}.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root: Path = args.input_root
    output_root: Path | None = args.output_root
    if not input_root.is_dir():
        raise SystemExit(f"Missing input root: {input_root}")
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)

    sample_dirs = list_sample_dirs(input_root)
    if args.max_samples is not None:
        sample_dirs = sample_dirs[: args.max_samples]
    if not sample_dirs:
        raise SystemExit(f"No sample folders with mp4 clips found under: {input_root}")

    jobs = max(1, args.jobs)
    crop_margin = max(0, args.crop_margin)
    payloads = [
        (
            str(sample_dir),
            str(input_root),
            str(output_root) if output_root is not None else "",
            args.target_width,
            args.target_height,
            args.min_clips,
            args.limit_frames,
            args.cropdetect_limit,
            args.round,
            args.reset_count,
            crop_margin,
            args.crf,
            args.preset,
            args.codec,
            args.overwrite,
            args.keep_temp,
        )
        for sample_dir in sample_dirs
    ]

    report: list[dict] = []
    if jobs == 1:
        iterator = map(_process_sample_mp, payloads)
    else:
        executor = ProcessPoolExecutor(max_workers=jobs)
        iterator = executor.map(_process_sample_mp, payloads)

    try:
        for idx, info in enumerate(iterator, start=1):
            report.append(info)
            sample = info.get("sample") or info.get("sample_dir")
            status = info.get("status", "unknown")
            print(f"[{idx}/{len(sample_dirs)}] {status:>20} {sample}")
    finally:
        if jobs != 1:
            executor.shutdown(wait=True)

    report_root = output_root if output_root is not None else input_root / "_manifests"
    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / "merge_crop_resize_report.json"
    summary_path = report_root / "merge_crop_resize_summary.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = summarize(report)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"Done. samples={len(report)}, status={summary['status_counts']}, "
        f"report={report_path}, summary={summary_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
