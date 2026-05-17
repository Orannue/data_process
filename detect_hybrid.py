import json
import os
import re
import argparse
from pathlib import Path

import cv2
from scenedetect import AdaptiveDetector, ContentDetector, SceneManager, open_video


def detect_scenes_via_temp_file(
    frames,
    output_dir,
    fps=25.0,
    threshold=24.0,
    adaptive_threshold=2.0,
    min_scene_len_frames=15,
    temp_tag="video",
):
    """
    Write in-memory frames to a temp MP4 and detect scene boundaries.
    Returns list[[start_frame, end_frame], ...] in closed interval format.
    """
    if not frames:
        return []

    temp_filename = os.path.join(output_dir, f"temp_processing_{temp_tag}.mp4")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(temp_filename, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()

    scenes_frames = []
    video = None
    try:
        video = open_video(temp_filename)

        # First pass: adaptive detector (more robust to motion/exposure drift).
        scene_manager = SceneManager()
        scene_manager.add_detector(
            AdaptiveDetector(
                adaptive_threshold=adaptive_threshold,
                min_scene_len=min_scene_len_frames,
            )
        )
        scene_manager.detect_scenes(video, show_progress=False)
        scene_list = scene_manager.get_scene_list()

        # Fallback: content detector when adaptive gives no boundary.
        if not scene_list:
            if hasattr(video, "reset"):
                video.reset()
            scene_manager = SceneManager()
            scene_manager.add_detector(
                ContentDetector(
                    threshold=threshold,
                    min_scene_len=min_scene_len_frames,
                )
            )
            scene_manager.detect_scenes(video, show_progress=False)
            scene_list = scene_manager.get_scene_list()

        if not scene_list:
            scenes_frames.append([0, len(frames) - 1])
        else:
            for start, end in scene_list:
                scenes_frames.append([start.get_frames(), end.get_frames() - 1])
    except Exception as exc:
        print(f"    Error during detection: {exc}")
        scenes_frames.append([0, len(frames) - 1])
    finally:
        if video is not None:
            if hasattr(video, "release"):
                video.release()
            del video
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except Exception as exc:
                print(f"    Warning: Could not delete temp file: {exc}")

    return scenes_frames


def merge_short_segments(segments, min_segment_len_frames):
    if not segments:
        return []

    merged = [segments[0][:]]
    for start, end in segments[1:]:
        if end < start:
            continue
        cur_start, cur_end = merged[-1]
        cur_len = cur_end - cur_start + 1
        nxt_len = end - start + 1

        if cur_len < min_segment_len_frames:
            merged[-1][1] = end
            continue
        if nxt_len < min_segment_len_frames:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    if len(merged) > 1:
        tail_len = merged[-1][1] - merged[-1][0] + 1
        if tail_len < min_segment_len_frames:
            merged[-2][1] = merged[-1][1]
            merged.pop()

    return merged


def is_video_valid(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        valid_frames = 0
        for _ in range(10):
            ok, _ = cap.read()
            if not ok:
                break
            valid_frames += 1
        cap.release()
        return valid_frames > 0
    except Exception:
        return False


def parse_timestamp_key(filename):
    try:
        parts = filename.split("_")
        time_part = parts[-1]
        start_time, _ = time_part.split("-")

        def to_seconds(time_str):
            h, m, s = time_str.split(".")
            return int(h) * 3600 + int(m) * 60 + float(f"{s[:2]}.{s[2:]}")

        return to_seconds(start_time)
    except Exception:
        return 0


SINGLE_PERSON_POSITIVE_PATTERN = re.compile(
    r"\b(close-up|close up|face|portrait|bedroom|hospital room|office|study|library|"
    r"room|window|rowboat|interior with|inside a house|cozy dining room)\b",
    re.IGNORECASE,
)
SINGLE_PERSON_NEGATIVE_PATTERN = re.compile(
    r"\b(crowd|battle|battlefield|soldiers|party|ballroom|hall|train station|street|"
    r"market|audience|church|banquet|wedding|people|group)\b",
    re.IGNORECASE,
)


def score_scene_priority(scene_desc, scene_videos):
    score = 0.0
    clip_count = len(scene_videos)
    if clip_count <= 2:
        score += 3.0
    elif clip_count <= 4:
        score += 1.5
    elif clip_count >= 10:
        score -= 1.5
    if SINGLE_PERSON_POSITIVE_PATTERN.search(scene_desc):
        score += 2.0
    if SINGLE_PERSON_NEGATIVE_PATTERN.search(scene_desc):
        score -= 2.5
    return score


def score_movie_priority(scenes):
    if not scenes:
        return float("-inf")
    total_score = sum(score_scene_priority(desc, vids) for desc, vids in scenes.items())
    return total_score / len(scenes)


def get_original_frames_for_save(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"    Error: Could not open video file: {video_path}")
            return []
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        return frames
    except Exception as exc:
        print(f"    Error extracting original frames from {video_path}: {exc}")
        return []


def _mean_abs_diff(frame_a, frame_b):
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    small_a = cv2.resize(gray_a, (160, 90), interpolation=cv2.INTER_AREA)
    small_b = cv2.resize(gray_b, (160, 90), interpolation=cv2.INTER_AREA)
    return float(cv2.absdiff(small_a, small_b).mean())


def _extract_segment_boundaries(segments):
    # boundary means the first frame index of the right segment (scene cut around this index)
    boundaries = []
    for i in range(len(segments) - 1):
        boundaries.append(segments[i][1] + 1)
    return boundaries


def _segments_from_boundaries(total_frames, boundaries):
    if total_frames <= 0:
        return []

    clean = []
    for b in sorted(boundaries):
        if 0 < b < total_frames:
            if not clean or b != clean[-1]:
                clean.append(b)

    segments = []
    start = 0
    for b in clean:
        end = b - 1
        if end >= start:
            segments.append([start, end])
        start = b
    if start <= total_frames - 1:
        segments.append([start, total_frames - 1])
    return segments


def _adjacent_frame_diffs(frames):
    if len(frames) < 2:
        return []
    diffs = []
    for i in range(len(frames) - 1):
        diffs.append(_mean_abs_diff(frames[i], frames[i + 1]))
    return diffs


def refine_segments_with_local_peak(
    segments,
    frames,
    search_radius_frames=6,
    min_peak_gain=1.03,
):
    """
    Refine each boundary to the strongest local frame-difference peak nearby.
    This helps correct several-frame offsets from detector outputs.
    """
    total_frames = len(frames)
    if total_frames < 3 or len(segments) <= 1:
        return segments

    diffs = _adjacent_frame_diffs(frames)
    if not diffs:
        return segments

    boundaries = _extract_segment_boundaries(segments)
    refined_boundaries = []

    for b in boundaries:
        if b <= 0 or b >= total_frames:
            continue

        left = max(1, b - search_radius_frames)
        right = min(total_frames - 1, b + search_radius_frames)
        current_score = diffs[b - 1]

        best_b = b
        best_score = current_score
        for cand_b in range(left, right + 1):
            cand_score = diffs[cand_b - 1]
            if cand_score > best_score:
                best_score = cand_score
                best_b = cand_b

        # Only move when local peak is meaningfully stronger.
        if best_score >= current_score * min_peak_gain:
            refined_boundaries.append(best_b)
        else:
            refined_boundaries.append(b)

    return _segments_from_boundaries(total_frames, refined_boundaries)


def _has_boundary_near(boundaries, target_idx, tolerance):
    for b in boundaries:
        if abs(b - target_idx) <= tolerance:
            return True
    return False


def merge_segments_across_similar_seams(
    shot_segments,
    seam_start_indices,
    frames,
    scene_level_boundaries,
    fps,
    seam_support_seconds=0.45,
    seam_diff_threshold=8.0,
):
    """
    Merge only seam-induced boundaries when:
    1) scene-level detector does not strongly support a cut near seam
    2) visual difference across seam is small
    """
    if len(shot_segments) <= 1:
        return shot_segments

    if len(frames) < 2:
        return shot_segments

    tolerance = max(2, int(round(fps * seam_support_seconds)))
    # Lower value = stricter merge.
    seam_set = set(seam_start_indices)

    merged = [shot_segments[0][:]]
    for next_start, next_end in shot_segments[1:]:
        prev_start, prev_end = merged[-1]
        is_direct_neighbor = next_start == prev_end + 1
        is_clip_seam = next_start in seam_set

        if not is_direct_neighbor:
            merged.append([next_start, next_end])
            continue

        if not is_clip_seam:
            merged.append([next_start, next_end])
            continue

        if next_start <= 0 or next_start >= len(frames):
            merged.append([next_start, next_end])
            continue

        supported_by_scene_level = _has_boundary_near(
            scene_level_boundaries, next_start, tolerance
        )
        seam_diff = _mean_abs_diff(frames[next_start - 1], frames[next_start])

        if (not supported_by_scene_level) and seam_diff <= seam_diff_threshold:
            merged[-1][1] = next_end
        else:
            merged.append([next_start, next_end])

    return merged


def process_movie_scenes(
    json_data,
    moviebench_path,
    output_base_path,
    use_priority=False,
    clip_content_threshold=25,
    scene_content_threshold=27,
    adaptive_threshold=2.0,
    min_scene_seconds=0.1,
    min_shot_seconds=0.1,
    seam_support_seconds=0.3,
    seam_diff_threshold=5.0,
    refine_search_radius_frames=3,
    refine_min_peak_gain=1.01,
):
    Path(output_base_path).mkdir(parents=True, exist_ok=True)
    if use_priority:
        sorted_movies = sorted(
            json_data.items(),
            key=lambda item: score_movie_priority(item[1]),
            reverse=True,
        )
    else:
        sorted_movies = list(json_data.items())
    sorted_movies.reverse()
    for movie_rank, (movie_id, scenes) in enumerate(sorted_movies, start=1):
        print(f"\nProcessing movie: {movie_id}")
        if use_priority:
            movie_priority = score_movie_priority(scenes)
            print(
                f"  Movie priority rank: {movie_rank}/{len(sorted_movies)} | score={movie_priority:.2f}"
            )
        else:
            print(f"  Movie order: {movie_rank}/{len(sorted_movies)}")
        movie_path = os.path.join(moviebench_path, movie_id)

        if not os.path.exists(movie_path):
            print(f"Movie path does not exist: {movie_path}")
            continue

        if use_priority:
            sorted_scenes = sorted(
                scenes.items(),
                key=lambda item: score_scene_priority(item[0], item[1]),
                reverse=True,
            )
        else:
            sorted_scenes = list(scenes.items())

        for scene_desc, scene_videos in sorted_scenes:
            print(f"  Processing scene: {scene_desc}")
            if use_priority:
                scene_priority = score_scene_priority(scene_desc, scene_videos)
                print(f"    Scene priority score: {scene_priority:.2f}")

            video_info_list = []
            all_exist_and_valid = True
            sorted_scene_videos = sorted(scene_videos, key=parse_timestamp_key)

            for video_name in sorted_scene_videos:
                video_full_path = os.path.join(movie_path, f"{video_name}.avi")
                if not os.path.exists(video_full_path):
                    print(f"    Warning: Video file does not exist: {video_full_path}")
                    all_exist_and_valid = False
                    break
                if not is_video_valid(video_full_path):
                    print(f"    Warning: Video file is corrupted: {video_full_path}")
                    all_exist_and_valid = False
                    break
                video_info_list.append({"path": video_full_path, "name": video_name})

            if not all_exist_and_valid or not video_info_list:
                print(
                    f"    Skipping scene due to missing/corrupted source clips: {scene_desc}"
                )
                continue

            scene_name_clean = "".join(
                c
                for c in scene_desc.replace("Sence", "Scene")
                if c.isalnum() or c in (" ", "-", "_")
            ).rstrip()
            scene_name_clean = scene_name_clean.replace(" ", "_").replace("__", "_")
            scene_output_path = os.path.join(output_base_path, movie_id, scene_name_clean)

            if os.path.exists(scene_output_path) and any(Path(scene_output_path).iterdir()):
                print(f"    Skipping already processed scene: {scene_desc}")
                continue

            Path(scene_output_path).mkdir(parents=True, exist_ok=True)

            original_video_segments = []
            fps_for_detection = 25.0
            if video_info_list:
                cap_fps = cv2.VideoCapture(video_info_list[0]["path"])
                fps_val = cap_fps.get(cv2.CAP_PROP_FPS)
                if fps_val > 0:
                    fps_for_detection = fps_val
                cap_fps.release()

            clip_level_segments = []
            seam_start_indices = []
            global_offset = 0
            min_scene_len_frames = max(8, int(round(fps_for_detection * min_scene_seconds)))

            for clip_idx, video_info in enumerate(video_info_list):
                frames = get_original_frames_for_save(video_info["path"])
                if not frames:
                    print(f"      Warning: Could not extract frames from {video_info['path']}")
                    continue

                clip_segments = detect_scenes_via_temp_file(
                    frames=frames,
                    output_dir=scene_output_path,
                    fps=fps_for_detection,
                    threshold=clip_content_threshold,
                    adaptive_threshold=adaptive_threshold,
                    min_scene_len_frames=min_scene_len_frames,
                    temp_tag=f"clip_{clip_idx}",
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

                for start_frame, end_frame in clip_segments:
                    clip_level_segments.append(
                        [start_frame + global_offset, end_frame + global_offset]
                    )

                original_video_segments.extend(frames)
                global_offset += len(frames)
                if clip_idx < len(video_info_list) - 1:
                    seam_start_indices.append(global_offset)

            if not original_video_segments:
                print(f"    No frames extracted for scene: {scene_desc}")
                Path(scene_output_path).rmdir()
                continue

            print(f"    Total stitched frames: {len(original_video_segments)}")
            print(f"    Source FPS: {fps_for_detection:.3f}")

            # Scene-level pass only for seam correction support.
            scene_level_segments = detect_scenes_via_temp_file(
                frames=original_video_segments,
                output_dir=scene_output_path,
                fps=fps_for_detection,
                threshold=scene_content_threshold,
                adaptive_threshold=adaptive_threshold,
                min_scene_len_frames=min_scene_len_frames,
                temp_tag="scene_full",
            )
            scene_level_boundaries = _extract_segment_boundaries(scene_level_segments)

            shot_segments = merge_segments_across_similar_seams(
                shot_segments=clip_level_segments,
                seam_start_indices=seam_start_indices,
                frames=original_video_segments,
                scene_level_boundaries=scene_level_boundaries,
                fps=fps_for_detection,
                seam_support_seconds=seam_support_seconds,
                seam_diff_threshold=seam_diff_threshold,
            )

            min_shot_len_frames = max(8, int(round(fps_for_detection * min_shot_seconds)))
            shot_segments = merge_short_segments(shot_segments, min_shot_len_frames)

            print(f"    Detected {len(shot_segments)} shots.")

            for shot_idx, (start_frame, end_frame) in enumerate(shot_segments):
                shot_frames = original_video_segments[start_frame : end_frame + 1]
                if not shot_frames:
                    print(f"      No frames to save for shot {shot_idx + 1}")
                    continue

                start_seconds = start_frame / fps_for_detection
                end_seconds = end_frame / fps_for_detection
                start_h = int(start_seconds // 3600)
                start_m = int((start_seconds % 3600) // 60)
                start_s = start_seconds % 60
                end_h = int(end_seconds // 3600)
                end_m = int((end_seconds % 3600) // 60)
                end_s = end_seconds % 60
                start_ts = f"{start_h:02d}.{start_m:02d}.{start_s:06.3f}"
                end_ts = f"{end_h:02d}.{end_m:02d}.{end_s:06.3f}"

                shot_filename = os.path.join(
                    scene_output_path,
                    f"{movie_id}_{scene_name_clean}_shot_{shot_idx + 1}_{start_ts}-{end_ts}.mp4",
                )
                height, width = shot_frames[0].shape[:2]
                writer = cv2.VideoWriter(
                    shot_filename,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps_for_detection,
                    (width, height),
                )
                for frame in shot_frames:
                    writer.write(frame)
                writer.release()

                if not (os.path.exists(shot_filename) and os.path.getsize(shot_filename) > 0):
                    print(f"      Error: Generated video file is invalid: {shot_filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json-path",
        type=str,
        default=r"F:\dataset\movie\movies_scenes.json",
        help="Path to movie->scene mapping JSON.",
    )
    parser.add_argument(
        "--moviebench-path",
        type=str,
        default=r"F:\dataset\movie\moviebench",
        help="Path to source clip directory.",
    )
    parser.add_argument(
        "--output-base-path",
        type=str,
        default=r"H:\dataset\movie_shot",
        help="Path to save generated shot videos.",
    )
    parser.add_argument(
        "--use-priority",
        action="store_true",
        help="Enable priority-based sorting and priority score logs.",
    )
    parser.add_argument(
        "--clip-content-threshold",
        type=float,
        default=25.5,
        help="ContentDetector threshold for per-clip detection. Lower is more sensitive.",
    )
    parser.add_argument(
        "--scene-content-threshold",
        type=float,
        default=27.5,
        help="ContentDetector threshold for scene-level support pass. Lower is more sensitive.",
    )
    parser.add_argument(
        "--adaptive-threshold",
        type=float,
        default=2.5,
        help="AdaptiveDetector threshold. Lower is more sensitive.",
    )
    parser.add_argument(
        "--min-scene-seconds",
        type=float,
        default=0.1,
        help="Minimum segment length during detector passes.",
    )
    parser.add_argument(
        "--min-shot-seconds",
        type=float,
        default=0.1,
        help="Minimum final output shot length.",
    )
    parser.add_argument(
        "--seam-support-seconds",
        type=float,
        default=0.3,
        help="Support window around clip seam for scene-level boundary confirmation.",
    )
    parser.add_argument(
        "--seam-diff-threshold",
        type=float,
        default=5.0,
        help="Merge seams only if visual difference <= threshold (lower means stricter merge).",
    )
    parser.add_argument(
        "--refine-search-radius-frames",
        type=int,
        default=3,
        help="Boundary refinement search radius in frames around each cut.",
    )
    parser.add_argument(
        "--refine-min-peak-gain",
        type=float,
        default=1.01,
        help="Move boundary only if nearby peak is this many times stronger.",
    )
    args = parser.parse_args()

    print("Loading scene data...")
    with open(args.json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    print("Starting processing with Hybrid scene+clip strategy...")
    process_movie_scenes(
        json_data,
        args.moviebench_path,
        args.output_base_path,
        use_priority=args.use_priority,
        clip_content_threshold=args.clip_content_threshold,
        scene_content_threshold=args.scene_content_threshold,
        adaptive_threshold=args.adaptive_threshold,
        min_scene_seconds=args.min_scene_seconds,
        min_shot_seconds=args.min_shot_seconds,
        seam_support_seconds=args.seam_support_seconds,
        seam_diff_threshold=args.seam_diff_threshold,
        refine_search_radius_frames=args.refine_search_radius_frames,
        refine_min_peak_gain=args.refine_min_peak_gain,
    )
    print("Processing complete!")


if __name__ == "__main__":
    main()
