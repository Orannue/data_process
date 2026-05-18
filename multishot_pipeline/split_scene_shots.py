import argparse
import gc
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2

from utils import (
    append_jsonl,
    clean_name,
    format_seconds,
    is_video_valid,
    load_video_frames,
    mean_abs_frame_diff,
    parse_clip_start_seconds,
    read_json,
    reset_file,
    sort_clip_names,
    stable_id,
    write_json,
)

try:
    from scenedetect import AdaptiveDetector, ContentDetector, SceneManager, open_video

    HAVE_SCENEDETECT = True
except Exception:  # pragma: no cover
    HAVE_SCENEDETECT = False


def detect_segments_via_temp_file(
    frames,
    output_dir: Path,
    fps: float,
    threshold: float,
    adaptive_threshold: float,
    min_len_frames: int,
    temp_tag: str,
) -> List[List[int]]:
    if not frames:
        return []

    if not HAVE_SCENEDETECT:
        return detect_segments_by_frame_diff(
            frames=frames,
            threshold=threshold,
            min_len_frames=min_len_frames,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_path = output_dir / f"_tmp_detect_{temp_tag}.mp4"
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(temp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    for frame in frames:
        writer.write(frame)
    writer.release()

    scene_list = []
    video = None
    try:
        video = open_video(str(temp_path))
        manager = SceneManager()
        manager.add_detector(
            AdaptiveDetector(
                adaptive_threshold=adaptive_threshold,
                min_scene_len=min_len_frames,
            )
        )
        manager.detect_scenes(video, show_progress=False)
        scene_list = manager.get_scene_list()

        if not scene_list:
            if hasattr(video, "reset"):
                video.reset()
            manager = SceneManager()
            manager.add_detector(
                ContentDetector(threshold=threshold, min_scene_len=min_len_frames)
            )
            manager.detect_scenes(video, show_progress=False)
            scene_list = manager.get_scene_list()
    except Exception as exc:
        print(f"    [warn] detection failed for {temp_tag}: {exc}")
    finally:
        if video is not None and hasattr(video, "release"):
            video.release()
        video = None
        gc.collect()
        if temp_path.exists():
            unlink_with_retries(temp_path)

    if not scene_list:
        return [[0, len(frames) - 1]]
    segments = []
    for start, end in scene_list:
        s = int(start.get_frames())
        e = int(end.get_frames()) - 1
        if e >= s:
            segments.append([s, e])
    return segments or [[0, len(frames) - 1]]


def unlink_with_retries(path: Path, attempts: int = 5, delay: float = 0.2) -> bool:
    for attempt in range(attempts):
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return True
        except OSError:
            if attempt == attempts - 1:
                return False
            gc.collect()
            time.sleep(delay)
    return False


def cleanup_temp_videos(scene_dir: Path) -> int:
    if not scene_dir.exists():
        return 0
    removed = 0
    for path in scene_dir.iterdir():
        if (
            path.is_file()
            and path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
            and path.name.lower().startswith(("_tmp", "temp"))
        ):
            if unlink_with_retries(path):
                removed += 1
    return removed


def remove_scene_output_dir(scene_dir: Path, output_root: Path) -> bool:
    if not scene_dir.exists():
        return True
    resolved_scene = scene_dir.resolve()
    resolved_root = output_root.resolve()
    try:
        resolved_scene.relative_to(resolved_root)
    except ValueError:
        print(f"  [warn] refused to remove scene outside output root: {resolved_scene}")
        return False
    if resolved_scene == resolved_root:
        print(f"  [warn] refused to remove output root: {resolved_scene}")
        return False
    shutil.rmtree(resolved_scene)
    return True


def detect_segments_by_frame_diff(
    frames,
    threshold: float,
    min_len_frames: int,
) -> List[List[int]]:
    """OpenCV-only fallback when PySceneDetect is unavailable."""
    if len(frames) <= 1:
        return [[0, max(0, len(frames) - 1)]]
    diffs = [mean_abs_frame_diff(frames[i], frames[i + 1]) for i in range(len(frames) - 1)]
    if not diffs:
        return [[0, len(frames) - 1]]
    mean = float(sum(diffs) / len(diffs))
    var = float(sum((x - mean) ** 2 for x in diffs) / max(1, len(diffs)))
    dynamic_threshold = max(float(threshold), mean + 2.5 * (var ** 0.5))
    boundaries = []
    last_boundary = 0
    for idx, diff in enumerate(diffs, start=1):
        if diff >= dynamic_threshold and idx - last_boundary >= min_len_frames:
            boundaries.append(idx)
            last_boundary = idx
    return segments_from_boundaries(len(frames), boundaries)


def boundaries_from_segments(segments: Sequence[Sequence[int]]) -> List[int]:
    return [int(seg[1]) + 1 for seg in segments[:-1]]


def segments_from_boundaries(total_frames: int, boundaries: Sequence[int]) -> List[List[int]]:
    clean = []
    for b in sorted(int(x) for x in boundaries):
        if 0 < b < total_frames and (not clean or clean[-1] != b):
            clean.append(b)
    segments = []
    start = 0
    for b in clean:
        if b - 1 >= start:
            segments.append([start, b - 1])
        start = b
    if start <= total_frames - 1:
        segments.append([start, total_frames - 1])
    return segments


def refine_segments_with_local_peak(
    segments: Sequence[Sequence[int]],
    frames,
    search_radius_frames: int,
    min_peak_gain: float,
) -> List[List[int]]:
    if len(segments) <= 1 or len(frames) < 3:
        return [list(seg) for seg in segments]
    diffs = [mean_abs_frame_diff(frames[i], frames[i + 1]) for i in range(len(frames) - 1)]
    refined = []
    for boundary in boundaries_from_segments(segments):
        left = max(1, boundary - search_radius_frames)
        right = min(len(frames) - 1, boundary + search_radius_frames)
        current = diffs[boundary - 1]
        best_b = boundary
        best_score = current
        for candidate in range(left, right + 1):
            score = diffs[candidate - 1]
            if score > best_score:
                best_b = candidate
                best_score = score
        refined.append(best_b if best_score >= current * min_peak_gain else boundary)
    return segments_from_boundaries(len(frames), refined)


def merge_short_segments(segments: Sequence[Sequence[int]], min_len: int) -> List[List[int]]:
    if not segments:
        return []
    merged = [list(segments[0])]
    for start, end in segments[1:]:
        if end < start:
            continue
        cur_len = merged[-1][1] - merged[-1][0] + 1
        nxt_len = end - start + 1
        if cur_len < min_len or nxt_len < min_len:
            merged[-1][1] = int(end)
        else:
            merged.append([int(start), int(end)])
    if len(merged) > 1 and merged[-1][1] - merged[-1][0] + 1 < min_len:
        merged[-2][1] = merged[-1][1]
        merged.pop()
    return merged


def has_boundary_near(boundaries: Sequence[int], target: int, tolerance: int) -> bool:
    return any(abs(int(boundary) - target) <= tolerance for boundary in boundaries)


def merge_segments_across_soft_seams(
    segments: Sequence[Sequence[int]],
    seam_starts: Sequence[int],
    frames,
    scene_level_boundaries: Sequence[int],
    fps: float,
    seam_support_seconds: float,
    seam_diff_threshold: float,
) -> List[List[int]]:
    if len(segments) <= 1 or len(frames) < 2:
        return [list(seg) for seg in segments]
    tolerance = max(2, int(round(fps * seam_support_seconds)))
    seam_set = set(int(x) for x in seam_starts)
    merged = [list(segments[0])]
    for next_start, next_end in segments[1:]:
        prev_start, prev_end = merged[-1]
        direct_neighbor = int(next_start) == prev_end + 1
        clip_seam = int(next_start) in seam_set
        if not direct_neighbor or not clip_seam:
            merged.append([int(next_start), int(next_end)])
            continue
        supported = has_boundary_near(scene_level_boundaries, int(next_start), tolerance)
        seam_diff = mean_abs_frame_diff(frames[int(next_start) - 1], frames[int(next_start)])
        if (not supported) and seam_diff <= seam_diff_threshold:
            merged[-1][1] = int(next_end)
        else:
            merged.append([int(next_start), int(next_end)])
    return merged


def save_segment_video(path: Path, frames, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


def process_scene(
    movie_id: str,
    scene_index: int,
    scene_desc: str,
    clip_names: Sequence[str],
    moviebench_root: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> List[Dict]:
    movie_source_dir = moviebench_root / movie_id
    scene_id = f"{scene_index:04d}_{clean_name(scene_desc)}"
    scene_output_dir = output_root / movie_id / scene_id
    scene_manifest_path = scene_output_dir / "scene_manifest.json"

    if scene_manifest_path.exists() and not args.overwrite:
        if args.clean_temp:
            removed = cleanup_temp_videos(scene_output_dir)
            if removed:
                print(f"  [clean] {movie_id}/{scene_id}: removed temp videos={removed}")
        return []

    sorted_names = sort_clip_names(clip_names)
    clip_paths = [movie_source_dir / f"{name}.avi" for name in sorted_names]
    missing = [str(p) for p in clip_paths if not p.exists() or not is_video_valid(p)]
    if missing:
        print(f"  [skip] {movie_id}/{scene_id}: missing or invalid clips={len(missing)}")
        return []

    scene_output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_temp_videos(scene_output_dir)
    all_frames = []
    clip_level_segments: List[List[int]] = []
    seam_starts = []
    source_clips = []
    global_offset = 0
    fps = 25.0

    for clip_idx, (clip_name, clip_path) in enumerate(zip(sorted_names, clip_paths)):
        frames, info = load_video_frames(clip_path)
        if not frames:
            continue
        if clip_idx == 0 and info["fps"] > 0:
            fps = float(info["fps"])
        min_detect_len = max(2, int(round(fps * args.min_detect_seconds)))
        clip_segments = detect_segments_via_temp_file(
            frames=frames,
            output_dir=scene_output_dir,
            fps=fps,
            threshold=args.clip_content_threshold,
            adaptive_threshold=args.adaptive_threshold,
            min_len_frames=min_detect_len,
            temp_tag=f"clip_{clip_idx:03d}",
        )
        clip_segments = refine_segments_with_local_peak(
            clip_segments,
            frames,
            search_radius_frames=args.refine_search_radius_frames,
            min_peak_gain=args.refine_min_peak_gain,
        )
        for start, end in clip_segments:
            clip_level_segments.append([start + global_offset, end + global_offset])

        start_sec = parse_clip_start_seconds(clip_name)
        source_clips.append(
            {
                "clip_name": clip_name,
                "path": str(clip_path),
                "stitched_start_frame": global_offset,
                "stitched_end_frame": global_offset + len(frames) - 1,
                "movie_start_seconds": start_sec,
                "fps": info["fps"],
                "frame_count": len(frames),
            }
        )
        all_frames.extend(frames)
        global_offset += len(frames)
        if clip_idx < len(clip_paths) - 1:
            seam_starts.append(global_offset)

    if not all_frames:
        return []

    min_detect_len = max(2, int(round(fps * args.min_detect_seconds)))
    scene_level_segments = detect_segments_via_temp_file(
        frames=all_frames,
        output_dir=scene_output_dir,
        fps=fps,
        threshold=args.scene_content_threshold,
        adaptive_threshold=args.adaptive_threshold,
        min_len_frames=min_detect_len,
        temp_tag="scene_full",
    )
    scene_level_boundaries = boundaries_from_segments(scene_level_segments)
    shot_segments = merge_segments_across_soft_seams(
        segments=clip_level_segments,
        seam_starts=seam_starts,
        frames=all_frames,
        scene_level_boundaries=scene_level_boundaries,
        fps=fps,
        seam_support_seconds=args.seam_support_seconds,
        seam_diff_threshold=args.seam_diff_threshold,
    )
    if args.merge_short_seconds > 0:
        shot_segments = merge_short_segments(
            shot_segments, max(2, int(round(fps * args.merge_short_seconds)))
        )

    rows = []
    skipped_short = 0
    skipped_trimmed_empty = 0
    for raw_shot_idx, (start, end) in enumerate(shot_segments, start=1):
        trimmed_start = int(start) + max(0, int(args.trim_head_frames))
        trimmed_end = int(end) - max(0, int(args.trim_tail_frames))
        if trimmed_end < trimmed_start:
            skipped_trimmed_empty += 1
            continue

        duration = (trimmed_end - trimmed_start + 1) / fps
        if duration < args.min_shot_seconds:
            skipped_short += 1
            continue

        shot_frames = all_frames[trimmed_start : trimmed_end + 1]
        if not shot_frames:
            skipped_trimmed_empty += 1
            continue
        shot_idx = len(rows) + 1
        shot_id = f"shot_{shot_idx:04d}"
        filename = (
            f"{shot_id}_{format_seconds(trimmed_start / fps)}-"
            f"{format_seconds(trimmed_end / fps)}.mp4"
        )
        shot_path = scene_output_dir / filename
        if args.overwrite or not shot_path.exists():
            save_segment_video(shot_path, shot_frames, fps)
        rows.append(
            {
                "movie_id": movie_id,
                "scene_id": scene_id,
                "scene_index": scene_index,
                "scene_desc": scene_desc,
                "shot_id": shot_id,
                "shot_index": shot_idx,
                "raw_shot_index": raw_shot_idx,
                "path": str(shot_path),
                "fps": fps,
                "raw_start_frame": int(start),
                "raw_end_frame": int(end),
                "start_frame": trimmed_start,
                "end_frame": trimmed_end,
                "start_seconds": trimmed_start / fps,
                "end_seconds": trimmed_end / fps,
                "duration": duration,
                "trim_head_frames": max(0, int(args.trim_head_frames)),
                "trim_tail_frames": max(0, int(args.trim_tail_frames)),
                "sample_key": stable_id(movie_id, scene_id, shot_id),
            }
        )

    if skipped_short or skipped_trimmed_empty:
        print(
            f"  [filter] {movie_id}/{scene_id}: "
            f"short<{args.min_shot_seconds:.3f}s={skipped_short}, "
            f"empty_after_trim={skipped_trimmed_empty}"
        )

    if len(rows) == 1:
        cleanup_temp_videos(scene_output_dir)
        if remove_scene_output_dir(scene_output_dir, output_root):
            print(f"  [drop] {movie_id}/{scene_id}: only one shot; removed scene")
        return []

    write_json(
        scene_manifest_path,
        {
            "movie_id": movie_id,
            "scene_id": scene_id,
            "scene_index": scene_index,
            "scene_desc": scene_desc,
            "fps": fps,
            "stitched_frame_count": len(all_frames),
            "source_clips": source_clips,
            "shot_count": len(rows),
            "raw_shot_count": len(shot_segments),
            "skipped_short_count": skipped_short,
            "skipped_trimmed_empty_count": skipped_trimmed_empty,
            "trim_head_frames": max(0, int(args.trim_head_frames)),
            "trim_tail_frames": max(0, int(args.trim_tail_frames)),
            "min_shot_seconds": args.min_shot_seconds,
            "shots": rows,
        },
    )
    cleanup_temp_videos(scene_output_dir)
    return rows


def process_movies(args: argparse.Namespace) -> None:
    scene_json = Path(args.scene_json)
    moviebench_root = Path(args.moviebench_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "_manifests" / "shots.jsonl"
    if args.overwrite and manifest_path.exists():
        reset_file(manifest_path)
    elif not manifest_path.exists():
        reset_file(manifest_path)

    data = read_json(scene_json)
    if not HAVE_SCENEDETECT:
        print("[warn] PySceneDetect is not installed; using OpenCV frame-diff fallback.")
    only = set(args.only_movie or [])
    movie_items = list(data.items())
    if args.reverse:
        movie_items.reverse()

    for movie_rank, (movie_id, scenes) in enumerate(movie_items, start=1):
        if only and movie_id not in only:
            continue
        movie_dir = moviebench_root / movie_id
        if not movie_dir.exists():
            print(f"[skip] source movie missing: {movie_id}")
            continue
        print(f"[movie {movie_rank}/{len(movie_items)}] {movie_id}")
        for scene_index, (scene_desc, clip_names) in enumerate(scenes.items(), start=1):
            rows = process_scene(
                movie_id=movie_id,
                scene_index=scene_index,
                scene_desc=scene_desc,
                clip_names=clip_names,
                moviebench_root=moviebench_root,
                output_root=output_root,
                args=args,
            )
            if rows:
                append_jsonl(manifest_path, rows)
                print(f"  [scene {scene_index:04d}] shots={len(rows)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split MovieBench scene clips into shot videos and metadata."
    )
    parser.add_argument("--scene-json", default=r"movies_scenes.json")
    parser.add_argument("--moviebench-root", default=r"moviedataset")
    parser.add_argument("--output-root", default=r"shots")
    parser.add_argument("--only-movie", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument(
        "--clean-temp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove _tmp/temp videos from scene output directories.",
    )
    parser.add_argument("--clip-content-threshold", type=float, default=25.5)
    parser.add_argument("--scene-content-threshold", type=float, default=27.5)
    parser.add_argument("--adaptive-threshold", type=float, default=2.5)
    parser.add_argument("--min-detect-seconds", type=float, default=0.12)
    parser.add_argument(
        "--min-shot-seconds",
        type=float,
            default=2.0,
            help="Drop shots with trimmed duration less than this many seconds.",
    )
    parser.add_argument(
        "--merge-short-seconds",
        type=float,
        default=0.0,
        help="Optional pre-filter merge threshold. Default 0 means do not merge short shots.",
    )
    parser.add_argument(
        "--trim-head-frames",
        type=int,
        default=3,
        help="Frames removed from the beginning of every detected shot before saving.",
    )
    parser.add_argument(
        "--trim-tail-frames",
        type=int,
        default=3,
        help="Frames removed from the end of every detected shot before saving.",
    )
    parser.add_argument("--seam-support-seconds", type=float, default=0.3)
    parser.add_argument("--seam-diff-threshold", type=float, default=5.0)
    parser.add_argument("--refine-search-radius-frames", type=int, default=3)
    parser.add_argument("--refine-min-peak-gain", type=float, default=1.01)
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")
    process_movies(parse_args())


if __name__ == "__main__":
    main()
