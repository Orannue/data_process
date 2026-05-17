"""
For each top-level pair ``x.mp4`` + subdirectory ``x/`` under a chunked variants root
(e.g. H:\\dataset\\movie_variants_chunked), run the same hybrid shot detection as
``detect_hybrid.py`` (single-merged-video case: one clip, no inter-clip seams).

If the detected shot count is exactly **6** and each segment's length in frames
matches ``N/sample.json`` ``segments[*][--match-sample-field]`` (6 values, same
order), move ``x.mp4`` and folder ``N/`` to ``--dest``. Default field is
``frame_count_total``; if detections never match, try ``rgb_frames_extracted``
(merged-timeline lengths).

Parallelism: use ``--workers N`` (multiprocessing). This workload is CPU-heavy;
threads are limited by the GIL, so processes are used for real speedup. Each
worker loads one full video into RAM — lower ``N`` if you hit OOM.

Requires: opencv-python, scenedetect (same as detect_hybrid.py).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2

from detect_hybrid import (
    detect_scenes_via_temp_file,
    merge_short_segments,
    refine_segments_with_local_peak,
)


def _numeric_stem(p: Path) -> bool:
    return bool(re.fullmatch(r"\d+", p.stem))


def _expected_segment_lengths_from_sample_json(
    subdir: Path, field: str
) -> tuple[list[int] | None, str | None]:
    """
    Read ``subdir / sample.json`` and return (lengths, error_reason).
    lengths has one int per segment in ``segments`` order; None if unusable.
    """
    p = subdir / "sample.json"
    if not p.is_file():
        return None, "no sample.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return None, f"sample.json read/parse: {e}"
    segs = data.get("segments")
    if not isinstance(segs, list) or len(segs) != 6:
        return None, f"segments must be a list of length 6, got {type(segs).__name__} len={len(segs) if isinstance(segs, list) else 'n/a'}"
    out: list[int] = []
    for i, s in enumerate(segs):
        if not isinstance(s, dict) or field not in s:
            return None, f"segments[{i}] missing key {field!r}"
        try:
            out.append(int(s[field]))
        except (TypeError, ValueError):
            return None, f"segments[{i}][{field!r}] not an int"
    return out, None


def hybrid_shot_count_for_frames(
    frames: list,
    fps: float,
    temp_dir: Path,
    *,
    clip_content_threshold: float,
    scene_content_threshold: float,
    adaptive_threshold: float,
    min_scene_seconds: float,
    min_shot_seconds: float,
    seam_support_seconds: float,
    seam_diff_threshold: float,
    refine_search_radius_frames: int,
    refine_min_peak_gain: float,
) -> tuple[int, list[int]]:
    """
    Return (shot_count, [len0, len1, ...]) in merged frame order after the same
    pipeline as detect_hybrid (one clip).
    """
    if not frames:
        return 0, []

    temp_dir.mkdir(parents=True, exist_ok=True)
    min_scene_len_frames = max(8, int(round(fps * min_scene_seconds)))
    min_shot_len_frames = max(8, int(round(fps * min_shot_seconds)))

    clip_segments = detect_scenes_via_temp_file(
        frames=frames,
        output_dir=str(temp_dir),
        fps=fps,
        threshold=clip_content_threshold,
        adaptive_threshold=adaptive_threshold,
        min_scene_len_frames=min_scene_len_frames,
        temp_tag="clip_0",
    )
    if not clip_segments:
        clip_segments = [[0, len(frames) - 1]]
    else:
        clip_segments = refine_segments_with_local_peak(
            clip_segments,
            frames,
            search_radius_frames=refine_search_radius_frames,
            min_peak_gain=refine_min_peak_gain,
        )

    # Single merged-video case has no inter-clip seams.
    # merge_segments_across_similar_seams() would be a no-op, and the extra
    # scene-level pass only adds heavy I/O/compute.
    shot_segments = merge_short_segments(clip_segments, min_shot_len_frames)
    lengths = [e - s + 1 for s, e in shot_segments]
    return len(shot_segments), lengths


def _read_video_frames_and_fps(
    video_path: Path, *, fallback_fps: float = 25.0
) -> tuple[list, float, str | None]:
    """
    Decode frames and read fps in a single VideoCapture pass.
    Returns (frames, fps, error).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], fallback_fps, "cannot open video"
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 1e-6:
            fps = fallback_fps
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    except Exception as e:
        return [], fallback_fps, str(e)
    finally:
        cap.release()
    if not frames:
        return [], fps, "no frames decoded"
    return frames, fps, None


def _detect_kwargs_dict(args: argparse.Namespace) -> dict:
    return {
        "clip_content_threshold": args.clip_content_threshold,
        "scene_content_threshold": args.scene_content_threshold,
        "adaptive_threshold": args.adaptive_threshold,
        "min_scene_seconds": args.min_scene_seconds,
        "min_shot_seconds": args.min_shot_seconds,
        "seam_support_seconds": args.seam_support_seconds,
        "seam_diff_threshold": args.seam_diff_threshold,
        "refine_search_radius_frames": args.refine_search_radius_frames,
        "refine_min_peak_gain": args.refine_min_peak_gain,
    }


def _detect_job(item: dict) -> dict:
    """
    Run detection for one stem under source. Picklable for ProcessPoolExecutor.
    item: {"source": str, "stem": str, "kwargs": dict}
    """
    source = Path(item["source"])
    stem = item["stem"]
    kw = item["kwargs"]
    mp4 = source / f"{stem}.mp4"
    if not mp4.is_file():
        return {"stem": stem, "n_shots": -1, "shot_lengths": [], "error": "missing mp4"}

    frames, fps, read_err = _read_video_frames_and_fps(mp4)
    if read_err:
        return {"stem": stem, "n_shots": -1, "shot_lengths": [], "error": read_err}

    try:
        with tempfile.TemporaryDirectory(prefix=f"hybrid_{stem}_") as tmp:
            n_shots, lengths = hybrid_shot_count_for_frames(frames, fps, Path(tmp), **kw)
    except Exception as e:
        return {"stem": stem, "n_shots": -1, "shot_lengths": [], "error": str(e)}

    return {"stem": stem, "n_shots": n_shots, "shot_lengths": lengths, "error": None}


def _default_workers() -> int:
    try:
        n = os.cpu_count() or 1
    except NotImplementedError:
        n = 1
    return max(1, min(n//2, 4))


def _apply_detection_result(
    src: Path,
    dest: Path,
    args: argparse.Namespace,
    stats: dict,
    stem: str,
    res: dict,
    *,
    log_detect_line: bool,
) -> None:
    """
    Update stats, optionally log; move only if n_shots == 6 and detected segment
    frame lengths match sample.json (see --match-sample-field).
    """
    mp4 = src / f"{stem}.mp4"
    subdir = src / stem
    err = res.get("error")
    n_shots = res.get("n_shots", -1)
    shot_lengths: list[int] = list(res.get("shot_lengths") or [])

    if err:
        if log_detect_line:
            print(f"[error] {mp4.name}: {err}", flush=True)
        stats["errors"] += 1
        return
    if n_shots < 0:
        stats["errors"] += 1
        return

    if log_detect_line:
        lens_s = shot_lengths if shot_lengths else []
        print(f"[detect] {mp4.name}: {n_shots} shot(s) lengths={lens_s}", flush=True)

    if n_shots != 6:
        return

    stats["six_detected"] += 1

    expected, exp_err = _expected_segment_lengths_from_sample_json(
        subdir, args.match_sample_field
    )
    if expected is None:
        stats["reject_sample_json"] += 1
        print(
            f"[reject] {mp4.name}: 6 shots but sample.json: {exp_err}",
            flush=True,
        )
        return

    if shot_lengths != expected:
        stats["frame_mismatch"] += 1
        print(
            f"[reject] {mp4.name}: frame lengths mismatch "
            f"(detected={shot_lengths} vs sample.{args.match_sample_field}={expected})",
            flush=True,
        )
        return

    stats["six_matched"] += 1
    dst_mp4 = dest / mp4.name
    dst_dir = dest / mp4.stem

    if dst_mp4.exists() or dst_dir.exists():
        if not args.overwrite:
            stats["skip_dest_busy"] += 1
            print(
                f"[skip-move] {mp4.name}: destination exists (use --overwrite): "
                f"{dst_mp4} or {dst_dir}",
                flush=True,
            )
            return
        if dst_mp4.is_file():
            dst_mp4.unlink()
        if dst_dir.is_dir():
            shutil.rmtree(dst_dir)

    if args.dry_run:
        print(
            f"[dry-run] would move -> {dst_mp4} and {dst_dir}/",
            flush=True,
        )
        stats["moved"] += 1
        return

    shutil.move(str(mp4), str(dst_mp4))
    shutil.move(str(subdir), str(dst_dir))
    stats["moved"] += 1
    print(f"[moved] {mp4.name} + {mp4.stem}/ -> {dest}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Move x.mp4 + x/ to --dest when hybrid detection yields exactly 6 shots."
    )
    p.add_argument(
        "--source",
        type=Path,
        default=Path(r"H:\dataset\movie_variants_chunked"),
        help="Root with N.mp4 and N/ subfolders.",
    )
    p.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="Target directory (created if missing).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions only; do not move files.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing paths under --dest.",
    )
    p.add_argument(
        "--match-sample-field",
        type=str,
        default="rgb_frames_extracted",
        choices=("rgb_frames_extracted", "frame_count_total"),
        help=(
            "sample.json segments[i] key compared to each detected segment length (merged order). "
            "frame_count_total = source clip total frames (per your metadata). "
            "rgb_frames_extracted = length of that shot on the merged mp4 (often what scene cuts match)."
        ),
    )
    p.add_argument("--clip-content-threshold", type=float, default=25.5)
    p.add_argument("--scene-content-threshold", type=float, default=27.5)
    p.add_argument("--adaptive-threshold", type=float, default=2.5)
    p.add_argument("--min-scene-seconds", type=float, default=0.1)
    p.add_argument("--min-shot-seconds", type=float, default=0.1)
    p.add_argument("--seam-support-seconds", type=float, default=0.3)  
    p.add_argument("--seam-diff-threshold", type=float, default=5.0)
    p.add_argument("--refine-search-radius-frames", type=int, default=3)
    p.add_argument("--refine-min-peak-gain", type=float, default=1.01)
    p.add_argument(
        "--workers",
        type=int,
        default=_default_workers(),
        help=(
            "Parallel worker processes for detection (default capped at 4). "
            "Each worker loads one full video — use 1 if RAM is tight; increase on CPU-only boxes."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    src: Path = args.source
    dest: Path = args.dest

    if not src.is_dir():
        raise SystemExit(f"Source not found or not a directory: {src}")

    dest.mkdir(parents=True, exist_ok=True)

    print(
        f"[config] match sample.json segments[*][{args.match_sample_field!r}] "
        f"to detected per-shot frame lengths (merged order)",
        flush=True,
    )

    mp4s = sorted(
        [p for p in src.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"],
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
    )

    stats = {
        "candidates": 0,
        "six_detected": 0,
        "six_matched": 0,
        "moved": 0,
        "skipped_no_dir": 0,
        "errors": 0,
        "reject_sample_json": 0,
        "frame_mismatch": 0,
        "skip_dest_busy": 0,
    }

    jobs: list[dict] = []
    for mp4 in mp4s:
        if not _numeric_stem(mp4):
            continue
        subdir = src / mp4.stem
        if not subdir.is_dir():
            stats["skipped_no_dir"] += 1
            print(f"[skip] {mp4.name}: no matching folder {subdir.name}/", flush=True)
            continue

        stats["candidates"] += 1
        jobs.append(
            {
                "source": str(src.resolve()),
                "stem": mp4.stem,
                "kwargs": _detect_kwargs_dict(args),
            }
        )

    kw_note = f"workers={args.workers}"
    if args.workers > 1:
        print(
            f"[parallel] {len(jobs)} candidate(s), {kw_note} (multiprocessing; "
            "lower --workers if RAM is tight)",
            flush=True,
        )
        print(
            "[parallel] Each job prints when it finishes (completion order may differ from 1, 2, 3…). "
            "The first line can take minutes on long videos — not frozen.",
            flush=True,
        )

    if jobs:
        if args.workers <= 1:
            for j in jobs:
                res = _detect_job(j)
                _apply_detection_result(
                    src,
                    dest,
                    args,
                    stats,
                    j["stem"],
                    res,
                    log_detect_line=True,
                )
        else:
            # ex.map() yields results in *input* order, so a slow first job blocks all
            # progress logs until it returns — looks “stuck”. as_completed fixes that.
            total = len(jobs)
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                future_to_job = {ex.submit(_detect_job, j): j for j in jobs}
                done = 0
                for fut in as_completed(future_to_job):
                    job = future_to_job[fut]
                    stem = job["stem"]
                    done += 1
                    try:
                        res = fut.result()
                    except Exception as e:
                        res = {
                            "stem": stem,
                            "n_shots": -1,
                            "shot_lengths": [],
                            "error": str(e),
                        }
                    err = res.get("error")
                    n_shots = res.get("n_shots", -1)
                    if err:
                        print(
                            f"[progress] {done}/{total} {stem}.mp4 error: {err}",
                            flush=True,
                        )
                    else:
                        sl = res.get("shot_lengths") or []
                        print(
                            f"[progress] {done}/{total} {stem}.mp4 -> {n_shots} shot(s) "
                            f"lengths={list(sl)}",
                            flush=True,
                        )
                    _apply_detection_result(
                        src,
                        dest,
                        args,
                        stats,
                        stem,
                        res,
                        log_detect_line=False,
                    )

    print(
        f"\nDone.\n"
        f"  candidates (N.mp4 + N/): {stats['candidates']}\n"
        f"  detected exactly 6 shots: {stats['six_detected']}\n"
        f"  6 shots + sample.json field match ({args.match_sample_field}): {stats['six_matched']}\n"
        f"  moved (or dry-run): {stats['moved']}\n"
        f"  rejected (sample.json bad/missing): {stats['reject_sample_json']}\n"
        f"  rejected (frame length mismatch): {stats['frame_mismatch']}\n"
        f"  skipped (dest exists, matched otherwise): {stats['skip_dest_busy']}\n"
        f"  skipped (no subdir): {stats['skipped_no_dir']}\n"
        f"  errors: {stats['errors']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
