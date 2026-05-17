"""
For folders under movie_raw_dataset_variants_80frames (6-shot samples):
  - Fixed latent: shot1 = 22, shots 2..6 = 21 each (total latent 127, RGB 505).
  - Try all permutations of the 6 source videos so that each slot has enough frames    for the required RGB window (85, 84, 84, 84, 84, 84).
  - Re-encode trimmed shots, merged video, and sample.json (same fields as adjust_chunked_six_only_to_4n1).

Sample folders: either .../1_character_6_shot/<id>/ with shot1_*.mp4..shot6_*.mp4,
or any subfolder with sample.json + 6 segments. Nested case folders are preserved in output layout.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2

# Fixed pattern: shot1 latent 22 (n1 % 3 == 1), shots 2..6 latent 21 each (multiple of 3).
SHOT1_LATENT = 22
SHOT2_THROUGH_6_LATENT = 21
LATENT_LENGTHS: tuple[int, ...] = (SHOT1_LATENT,) + (SHOT2_THROUGH_6_LATENT,) * 5
TOTAL_TARGET_LATENT = sum(LATENT_LENGTHS)  # 127
TOTAL_TARGET_RGB = 1 + 4 * (TOTAL_TARGET_LATENT - 1)  # 505

RGB_LENGTHS: tuple[int, ...] = (
    1 + (SHOT1_LATENT - 1) * 4,
    *[SHOT2_THROUGH_6_LATENT * 4] * 5,
)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

CASE_DIR_NAMES = frozenset(
    {"1_character_6_shot", "2_character_6_shot", "3_character_6_shot"}
)


@dataclass
class Solution:
    permutation: tuple[int, ...]
    latent_lengths: tuple[int, ...]
    rgb_lengths: tuple[int, ...]
    total_latent: int
    total_rgb: int
    trimmed_frames: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Permute 6-shot raw variant folders so shot1 uses 22 latent (85 RGB) "
            "and shots 2..6 use 21 latent (84 RGB each). Writes trimmed mp4s + sample.json."
        )
    )
    p.add_argument(
        "--source-root",
        type=Path,
        default=Path(r"F:\dataset\movie_raw_dataset_variants_80frames"),
        help="Root containing case folders or sample subfolders.",
    )
    p.add_argument(
        "--dest-root",
        type=Path,
        default=None,
        help="Output root. Default: sibling folder with timestamp.",
    )
    p.add_argument("--overwrite", action="store_true", help="Allow non-empty destination.")
    p.add_argument("--codec", type=str, default="mp4v", help="FourCC for output mp4.")
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel processes (0 = auto ~min(cpu//2, 4); 1 = serial).",
    )
    return p.parse_args()


def _default_workers() -> int:
    try:
        n = os.cpu_count() or 1
    except NotImplementedError:
        n = 1
    return max(1, min(n // 2, 4))


def _iter_subdirs(root: Path) -> list[Path]:
    out = [p for p in root.iterdir() if p.is_dir()]
    out.sort(key=lambda x: (not x.name.isdigit(), int(x.name) if x.name.isdigit() else x.name))
    return out


def _safe_int(x, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _video_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()


def _pick_start(old_start: int, frame_total: int, need_len: int) -> int:
    max_start = max(0, frame_total - need_len)
    return min(max(old_start, 0), max_start)


def _center_start(frame_total: int, need_len: int) -> int:
    return max(0, (frame_total - need_len) // 2)


def _iter_sample_folders(source_root: Path) -> list[tuple[str, Path]]:
    """(group_name, sample_dir). group_name is case folder name or source_root.name."""
    out: list[tuple[str, Path]] = []
    subs = [p for p in source_root.iterdir() if p.is_dir()]
    has_cases = any(p.name in CASE_DIR_NAMES for p in subs)
    if has_cases:
        for case in sorted(subs, key=lambda p: p.name):
            if case.name not in CASE_DIR_NAMES:
                continue
            for s in _iter_subdirs(case):
                out.append((case.name, s))
    else:
        for s in _iter_subdirs(source_root):
            out.append((source_root.name, s))
    return out


def _segments_from_shot_files(folder: Path) -> list[dict] | None:
    items: list[tuple[int, Path]] = []
    for p in folder.iterdir():
        if not p.is_file() or p.suffix.lower() not in VIDEO_EXTS:
            continue
        m = re.match(r"shot(\d+)_", p.name, re.I)
        if not m:
            continue
        items.append((int(m.group(1)), p))
    if len(items) != 6:
        return None
    items.sort(key=lambda t: t[0])
    if [i for i, _ in items] != list(range(1, 7)):
        return None
    segments: list[dict] = []
    for idx, path in items:
        fc = _video_frame_count(path)
        if fc <= 0:
            return None
        segments.append(
            {
                "source_video": str(path.resolve()),
                "frame_count_total": fc,
                "selected_start_frame": -1,
            }
        )
    return segments


def load_sample_bundle(folder: Path) -> dict | None:
    """Return sample_data dict with key 'segments' (len 6), or None."""
    js = folder / "sample.json"
    if js.is_file():
        try:
            data = json.loads(js.read_text(encoding="utf-8"))
        except Exception:
            return None
        segs = data.get("segments")
        if not isinstance(segs, list) or len(segs) != 6:
            return None
        fixed: list[dict] = []
        for s in segs:
            if not isinstance(s, dict):
                return None
            sv = s.get("source_video")
            if not sv:
                return None
            p = Path(sv)
            if not p.is_file():
                return None
            fc = _safe_int(s.get("frame_count_total"), 0)
            if fc <= 0:
                fc = _video_frame_count(p)
            if fc <= 0:
                return None
            d = dict(s)
            d["source_video"] = str(p.resolve())
            d["frame_count_total"] = fc
            fixed.append(d)
        data = dict(data)
        data["segments"] = fixed
        return data
    segs = _segments_from_shot_files(folder)
    if segs is None:
        return None
    return {
        "segments": segs,
        "video_names": [Path(s["source_video"]).name for s in segs],
        "source_folder": str(folder.resolve()),
    }


def _solution_rank_key(sol: Solution) -> tuple[int, tuple[int, ...]]:
    """Lower is better: less trim, then lexicographic permutation."""
    return (sol.trimmed_frames, sol.permutation)


def solve_fixed_latent_permutation(segments: list[dict]) -> Solution | None:
    if len(segments) != 6:
        return None
    frame_caps = [_safe_int(s.get("frame_count_total"), 0) for s in segments]
    if any(x <= 0 for x in frame_caps):
        return None
    need_rgb = list(RGB_LENGTHS)
    cap_sum = sum(frame_caps)
    if cap_sum < TOTAL_TARGET_RGB:
        return None

    best: Solution | None = None
    best_key: tuple[int, tuple[int, ...]] | None = None

    for perm in itertools.permutations(range(6)):
        caps_order = [frame_caps[perm[i]] for i in range(6)]
        ok = True
        trim = 0
        for pos in range(6):
            if caps_order[pos] < need_rgb[pos]:
                ok = False
                break
            trim += caps_order[pos] - need_rgb[pos]
        if not ok:
            continue
        cand = Solution(
            permutation=perm,
            latent_lengths=LATENT_LENGTHS,
            rgb_lengths=tuple(need_rgb),
            total_latent=TOTAL_TARGET_LATENT,
            total_rgb=TOTAL_TARGET_RGB,
            trimmed_frames=trim,
        )
        key = _solution_rank_key(cand)
        if best is None or (best_key is not None and key < best_key):
            best = cand
            best_key = key

    return best


def _chunk_splits(latent_lengths: tuple[int, ...]) -> list[list[int]]:
    n1 = latent_lengths[0]
    first = [1] + [3] * ((n1 - 1) // 3)
    out = [first]
    for li in latent_lengths[1:]:
        out.append([3] * (li // 3))
    return out


def _segment_cumulative_switches(
    segment_latent_lengths: tuple[int, ...] | list[int],
    segment_rgb_lengths: tuple[int, ...] | list[int],
) -> tuple[list[int], list[int]]:
    sw_lat: list[int] = []
    sw_rgb: list[int] = []
    c_lat = 0
    c_rgb = 0
    for li, ri in zip(segment_latent_lengths, segment_rgb_lengths, strict=True):
        c_lat += int(li)
        c_rgb += int(ri)
        sw_lat.append(c_lat)
        sw_rgb.append(c_rgb)
    return sw_lat, sw_rgb


def _read_video_window(
    video_path: Path, start_frame: int, length: int
) -> tuple[list, float, tuple[int, int], int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 1e-6:
            fps = 23.976
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            raise RuntimeError(f"invalid resolution: {video_path}")
        if frame_count <= 0:
            raise RuntimeError(f"empty video: {video_path}")
        start_frame = _pick_start(start_frame, frame_count, length)
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
        frames = []
        for _ in range(length):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        if len(frames) != length:
            raise RuntimeError(
                f"read short from {video_path.name}: expected {length}, got {len(frames)}"
            )
        return frames, fps, (width, height), frame_count
    finally:
        cap.release()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_video(
    video_path: Path, frames: list, fps: float, size: tuple[int, int], fourcc: int
) -> None:
    _ensure_parent(video_path)
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"cannot create video: {video_path}")
    try:
        for f in frames:
            if (f.shape[1], f.shape[0]) != size:
                f = cv2.resize(f, size, interpolation=cv2.INTER_AREA)
            writer.write(f)
    finally:
        writer.release()


def _resolve_start(seg: dict, guessed_total: int, need_rgb: int) -> int:
    raw = seg.get("selected_start_frame")
    if raw is None or _safe_int(raw, -1) < 0:
        return _center_start(guessed_total, need_rgb)
    return _pick_start(_safe_int(raw, 0), guessed_total, need_rgb)


def process_and_write_sample(
    sample_id: str,
    sample_data: dict,
    solution: Solution,
    dest_sub_root: Path,
    fourcc: int,
) -> dict:
    out_sub = dest_sub_root / sample_id
    out_sub.mkdir(parents=True, exist_ok=True)

    old_segments = list(sample_data["segments"])
    reordered = [old_segments[i] for i in solution.permutation]
    chunks = _chunk_splits(solution.latent_lengths)
    switch_lat, switch_rgb = _segment_cumulative_switches(
        solution.latent_lengths, solution.rgb_lengths
    )

    merged_path = dest_sub_root / f"{sample_id}.mp4"
    merged_writer = None
    merged_size = None
    merged_fps = None

    new_segments: list[dict] = []
    try:
        for idx, seg in enumerate(reordered, start=1):
            src_video = Path(seg["source_video"])
            need_rgb = int(solution.rgb_lengths[idx - 1])
            guessed_total = _safe_int(seg.get("frame_count_total"), 0)
            start = _resolve_start(seg, guessed_total, need_rgb)

            frames, fps, size, actual_total = _read_video_window(src_video, start, need_rgb)
            corrected_start = _pick_start(start, actual_total, need_rgb)
            if corrected_start != start:
                frames, fps, size, actual_total = _read_video_window(
                    src_video, corrected_start, need_rgb
                )
                start = corrected_start
            end_exc = start + need_rgb

            shot_path = out_sub / f"shot{idx}.mp4"
            _write_video(shot_path, frames, fps, size, fourcc)

            if merged_writer is None:
                merged_size = size
                merged_fps = fps
                _ensure_parent(merged_path)
                merged_writer = cv2.VideoWriter(
                    str(merged_path), fourcc, merged_fps, merged_size
                )
                if not merged_writer.isOpened():
                    raise RuntimeError(f"cannot create merged video: {merged_path}")
            out_frames = frames
            if size != merged_size:
                out_frames = [
                    cv2.resize(f, merged_size, interpolation=cv2.INTER_AREA) for f in frames
                ]
            for f in out_frames:
                merged_writer.write(f)

            latent = int(solution.latent_lengths[idx - 1])
            new_seg = dict(seg)
            new_seg["output_segment"] = str(shot_path)
            new_seg["frame_count_total"] = actual_total
            new_seg["latent_length"] = latent
            new_seg["rgb_frames_extracted"] = need_rgb
            new_seg["latent_from_rgb"] = latent
            new_seg["selected_start_frame"] = start
            new_seg["selected_end_frame_exclusive"] = end_exc
            new_seg["chunk_latent_lengths"] = chunks[idx - 1]
            new_seg["chunk_latent_patterns"] = [[x] for x in chunks[idx - 1]]
            new_segments.append(new_seg)
    finally:
        if merged_writer is not None:
            merged_writer.release()

    payload = dict(sample_data)
    payload["video_names"] = [Path(x["source_video"]).name for x in new_segments]
    payload["output_folder"] = str(out_sub)
    payload["merged_video"] = str(merged_path)
    payload["segment_latent_lengths"] = [int(x) for x in solution.latent_lengths]
    payload["chunk_splits_per_shot"] = chunks
    payload["total_frame_length"] = int(solution.total_rgb)
    payload["total_latent_frame_length"] = int(solution.total_latent)
    payload["switch_frames"] = switch_rgb
    payload["switch_latent_frames"] = switch_lat
    payload["segments"] = new_segments
    payload["shot_source_permutation_1based"] = [i + 1 for i in solution.permutation]
    payload["new_constraints"] = {
        "shot1_latent": SHOT1_LATENT,
        "shot2_to_shot6_latent_each": SHOT2_THROUGH_6_LATENT,
        "shot1_rgb": RGB_LENGTHS[0],
        "shot2_to_shot6_rgb_each": RGB_LENGTHS[1],
        "target_total_latent_frames": TOTAL_TARGET_LATENT,
        "target_total_rgb_frames": TOTAL_TARGET_RGB,
        "per_shot_latent": [int(x) for x in solution.latent_lengths],
        "per_shot_rgb": [int(x) for x in solution.rgb_lengths],
        "note": "Order of source videos permuted so each slot meets RGB length; see shot_source_permutation_1based.",
    }
    (out_sub / "sample.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def _run_one_job(item: dict) -> dict:
    group = item["group"]
    folder: Path = Path(item["folder"])
    dest_sub_root: Path = Path(item["dest_sub_root"])
    fourcc = cv2.VideoWriter_fourcc(*item["codec"])
    sample_id = item["sample_id"]

    sample_data = load_sample_bundle(folder)
    if sample_data is None:
        return {
            "id": sample_id,
            "group": group,
            "status": "skip_no_6_shots_or_bad_json",
        }
    solution = solve_fixed_latent_permutation(list(sample_data["segments"]))
    if solution is None:
        return {
            "id": sample_id,
            "group": group,
            "status": "skip_no_feasible_permutation",
        }
    try:
        process_and_write_sample(sample_id, sample_data, solution, dest_sub_root, fourcc)
    except Exception as e:
        return {
            "id": sample_id,
            "group": group,
            "status": "error_process_video",
            "reason": str(e),
        }
    return {
        "id": sample_id,
        "group": group,
        "status": "ok",
        "perm_1based": [i + 1 for i in solution.permutation],
        "trimmed_frames": solution.trimmed_frames,
    }


def main() -> int:
    args = parse_args()
    source_root: Path = args.source_root
    if not source_root.is_dir():
        raise SystemExit(f"source root missing: {source_root}")

    if args.dest_root is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_root = source_root.parent / f"{source_root.name}_22_21_latent_{ts}"
    else:
        dest_root = args.dest_root

    if dest_root.exists() and any(dest_root.iterdir()) and not args.overwrite:
        raise SystemExit(f"destination not empty: {dest_root} (use --overwrite)")
    dest_root.mkdir(parents=True, exist_ok=True)

    workers = args.workers if args.workers > 0 else _default_workers()
    fourcc = cv2.VideoWriter_fourcc(*args.codec)

    pairs = _iter_sample_folders(source_root)
    jobs: list[dict] = []
    for group, folder in pairs:
        sample_id = f"{group}_{folder.name}"
        dest_sub_root = dest_root / group
        dest_sub_root.mkdir(parents=True, exist_ok=True)
        jobs.append(
            {
                "group": group,
                "folder": str(folder.resolve()),
                "dest_sub_root": str(dest_sub_root.resolve()),
                "sample_id": sample_id,
                "codec": args.codec,
            }
        )

    ok = 0
    fail = 0
    summary: list[dict] = []

    def _apply(res: dict) -> None:
        nonlocal ok, fail
        if res.get("status") == "ok":
            ok += 1
            print(
                f"[ok] {res.get('group')}/{res['id']}: perm={res.get('perm_1based')} "
                f"trimmed={res.get('trimmed_frames')}",
                flush=True,
            )
        else:
            fail += 1
            print(
                f"[skip/fail] {res.get('group')}/{res.get('id')}: {res.get('status')} "
                f"{res.get('reason', '')}",
                flush=True,
            )
        summary.append(res)

    if workers <= 1:
        for j in jobs:
            res = _run_one_job(j)
            _apply(res)
    else:
        total = len(jobs)
        print(
            f"[parallel] {total} sample(s), workers={workers}",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_run_one_job, j): j for j in jobs}
            done = 0
            for fut in as_completed(futs):
                done += 1
                sid = futs[fut]["sample_id"]
                try:
                    res = fut.result()
                except Exception as e:
                    res = {"id": sid, "status": "error_worker", "reason": str(e)}
                print(f"[progress] {done}/{total} {sid}", flush=True)
                _apply(res)

    report = {
        "source_root": str(source_root),
        "dest_root": str(dest_root),
        "workers": workers,
        "fixed_latent_lengths": list(LATENT_LENGTHS),
        "fixed_rgb_lengths": list(RGB_LENGTHS),
        "total_latent": TOTAL_TARGET_LATENT,
        "total_rgb": TOTAL_TARGET_RGB,
        "processed_ok": ok,
        "failed_or_skipped": fail,
        "items": summary,
    }
    (dest_root / "_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[done] ok={ok} failed_or_skipped={fail}")
    print(f"[dest] {dest_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
