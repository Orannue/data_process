from __future__ import annotations

import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2

# Fixed merged length: sum(latent) == TOTAL_TARGET_LATENT => sum(rgb) == TOTAL_TARGET_RGB
# (shot1 rgb = 1 + 4*(n1-1); shots 2..6 rgb = 4*ni).
TOTAL_TARGET_LATENT = 121
TOTAL_TARGET_RGB = 1 + 4 * (TOTAL_TARGET_LATENT - 1)  # 481


@dataclass
class Solution:
    permutation: tuple[int, ...]
    latent_lengths: tuple[int, ...]  # (n1, n2, n3, n4, n5, n6)
    rgb_lengths: tuple[int, ...]
    total_latent: int
    total_rgb: int
    trimmed_frames: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Reorder/crop each 6-shot sample so that: "
            "shot1 latent n1 follows [1,3,3,...] (n1 % 3 == 1), rgb1=1+(n1-1)*4; "
            "shots 2..6 each have latent 3*m_i (independent m_i), rgb_i=latent_i*4; "
            f"merged totals are fixed: latent={TOTAL_TARGET_LATENT}, rgb={TOTAL_TARGET_RGB}. "
            "Then write shot mp4s, merged mp4, and updated sample.json."
        )
    )
    p.add_argument(
        "--total-latent",
        type=int,
        default=TOTAL_TARGET_LATENT,
        help="Target sum of latent frames over6 shots (default 121 => 481 rgb frames).",
    )
    p.add_argument(
        "--source-root",
        type=Path,
        default=Path(r"H:\dataset\movie_variants_chunked_six_only"),
        help="Root containing N/sample.json folders.",
    )
    p.add_argument(
        "--dest-root",
        type=Path,
        default=None,
        help="Output root. If omitted, auto-creates a timestamped latest folder.",
    )
    p.add_argument("--overwrite", action="store_true", help="Allow reusing non-empty destination.")
    p.add_argument("--codec", type=str, default="mp4v", help="FourCC codec for output mp4.")
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Parallel processes (0 = auto: min(cpu_count//2, 4); 1 = serial). "
            "Each process decodes/encodes full shots — lower if RAM or disk is tight."
        ),
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


def _pick_start(old_start: int, frame_total: int, need_len: int) -> int:
    max_start = max(0, frame_total - need_len)
    return min(max(old_start, 0), max_start)


def _max_n1(frame_total: int) -> int:
    # Need rgb1 = 1 + (n1 - 1) * 4 <= frame_total
    return (frame_total - 1) // 4 + 1


def _max_n2(frame_total: int) -> int:
    # Need rgb = n2 * 4 <= frame_total
    return frame_total // 4


def _largest_mod_leq(limit: int, mod: int, rem: int) -> int:
    if limit < rem:
        return -1
    return limit - ((limit - rem) % mod)


def _max_latent_shot2to6_from_cap_rgb(cap_rgb: int) -> int:
    """Largest n with n % 3 == 0 and 4 * n <= cap_rgb."""
    return _largest_mod_leq(cap_rgb // 4, 3, 0)


def _find_rest_latents_fixed_sum(
    need: int, caps_lat_max: list[int]
) -> tuple[int, ...] | None:
    """
    caps_lat_max:5 entries, max latent per shot (positions 2..6), each multiple of 3, >= 3.
    need = sum of 5 latents. First valid tuple (greedy descending on first slots) or None.
    """
    if len(caps_lat_max) != 5:
        return None

    def dfs(remain: int, i: int, acc: list[int]) -> bool:
        if i == 4:
            if remain < 3 or remain % 3 != 0 or remain > caps_lat_max[4]:
                return False
            acc.append(remain)
            return True
        slots_after = 4 - i
        min_tail = 3 * slots_after
        hi = min(caps_lat_max[i], remain - min_tail)
        hi = hi // 3 * 3
        for v in range(hi, 2, -3):
            acc.append(v)
            if dfs(remain - v, i + 1, acc):
                return True
            acc.pop()
        return False

    acc: list[int] = []
    if dfs(need, 0, acc):
        return tuple(acc)
    return None


def _solution_rank_key(sol: Solution) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    r = sol.latent_lengths[1:]
    spread = max(r) - min(r)
    return (spread, sol.latent_lengths, sol.permutation)


def solve_one_sample(segments: list[dict], *, total_latent: int) -> Solution | None:
    if len(segments) != 6:
        return None

    frame_caps = [_safe_int(s.get("frame_count_total"), 0) for s in segments]
    if any(x <= 0 for x in frame_caps):
        return None

    total_rgb = 1 + 4 * (total_latent - 1)
    cap_sum = sum(frame_caps)
    if cap_sum < total_rgb:
        return None

    min_rest_sum = 5 * 3
    if total_latent - min_rest_sum < 1:
        return None

    best: Solution | None = None
    best_key: tuple[int, tuple[int, ...], tuple[int, ...]] | None = None

    for perm in itertools.permutations(range(6)):
        caps_order = [frame_caps[perm[i]] for i in range(6)]
        max_n1 = _largest_mod_leq(_max_n1(caps_order[0]), 3, 1)
        if max_n1 < 1:
            continue
        lat_max_rest = [_max_latent_shot2to6_from_cap_rgb(caps_order[j]) for j in range(1, 6)]
        if any(m < 3 for m in lat_max_rest):
            continue

        n1_hi = min(max_n1, total_latent - min_rest_sum)
        n1_lo = 1
        if n1_lo > n1_hi:
            continue
        n1_hi = n1_hi - ((n1_hi - 1) % 3)
        if n1_hi < n1_lo:
            continue

        for n1 in range(n1_hi, n1_lo - 1, -3):
            rem = total_latent - n1
            if rem < min_rest_sum or rem % 3 != 0:
                continue
            rest = _find_rest_latents_fixed_sum(rem, lat_max_rest)
            if rest is None:
                continue
            latents = (n1,) + rest
            rgbs = (1 + (n1 - 1) * 4,) + tuple(4 * x for x in rest)
            cand = Solution(
                permutation=perm,
                latent_lengths=latents,
                rgb_lengths=rgbs,
                total_latent=total_latent,
                total_rgb=total_rgb,
                trimmed_frames=cap_sum - total_rgb,
            )
            key = _solution_rank_key(cand)
            if best is None:
                best = cand
                best_key = key
            elif best_key is not None and key < best_key:
                best = cand
                best_key = key

    return best


def _chunk_splits(solution: Solution) -> list[list[int]]:
    n1 = solution.latent_lengths[0]
    first = [1] + [3] * ((n1 - 1) // 3)
    out = [first]
    for li in solution.latent_lengths[1:]:
        out.append([3] * (li // 3))
    return out


def _segment_cumulative_switches(
    segment_latent_lengths: tuple[int, ...] | list[int],
    segment_rgb_lengths: tuple[int, ...] | list[int],
) -> tuple[list[int], list[int]]:
    """
    Cumulative boundaries after each segment (inclusive of that segment), same length as segments.
    E.g. segment_latent_lengths [4,6,6] -> switch_latent_frames [4,10,16].
    """
    if len(segment_latent_lengths) != len(segment_rgb_lengths):
        raise ValueError("latent and rgb segment length lists must match")
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


def _read_video_window(video_path: Path, start_frame: int, length: int) -> tuple[list, float, tuple[int, int], int]:
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


def _write_video(video_path: Path, frames: list, fps: float, size: tuple[int, int], fourcc: int) -> None:
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


def process_and_write_sample(
    sample_id: str, sample_data: dict, solution: Solution, dest_root: Path, fourcc: int
) -> dict:
    out_sub = dest_root / sample_id
    out_sub.mkdir(parents=True, exist_ok=True)

    old_segments = list(sample_data["segments"])
    reordered = [old_segments[i] for i in solution.permutation]
    chunks = _chunk_splits(solution)
    switch_lat, switch_rgb = _segment_cumulative_switches(
        solution.latent_lengths, solution.rgb_lengths
    )

    merged_path = dest_root / f"{sample_id}.mp4"
    merged_writer = None
    merged_size = None
    merged_fps = None

    new_segments: list[dict] = []
    try:
        for idx, seg in enumerate(reordered, start=1):
            src_video = Path(seg["source_video"])
            need_rgb = int(solution.rgb_lengths[idx - 1])
            guessed_total = _safe_int(seg.get("frame_count_total"), 0)
            start = _pick_start(_safe_int(seg.get("selected_start_frame"), 0), guessed_total, need_rgb)

            frames, fps, size, actual_total = _read_video_window(src_video, start, need_rgb)
            # Re-clamp with actual frame count and re-read only when metadata was stale.
            corrected_start = _pick_start(start, actual_total, need_rgb)
            if corrected_start != start:
                frames, fps, size, actual_total = _read_video_window(src_video, corrected_start, need_rgb)
                start = corrected_start
            end_exc = start + need_rgb

            shot_path = out_sub / f"shot{idx}.mp4"
            _write_video(shot_path, frames, fps, size, fourcc)

            if merged_writer is None:
                merged_size = size
                merged_fps = fps
                _ensure_parent(merged_path)
                merged_writer = cv2.VideoWriter(str(merged_path), fourcc, merged_fps, merged_size)
                if not merged_writer.isOpened():
                    raise RuntimeError(f"cannot create merged video: {merged_path}")
            out_frames = frames
            if size != merged_size:
                out_frames = [cv2.resize(f, merged_size, interpolation=cv2.INTER_AREA) for f in frames]
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
        "shot1_latent_pattern": "[1,3,3,...] (n1 % 3 == 1)",
        "shot2_to_shot6_latent": "each is 3 * m_i independently (m_i >= 1)",
        "shot1_rgb": "1 + (n1 - 1) * 4",
        "shot2_to_shot6_rgb": "latent_i * 4",
        "target_total_latent_frames": int(solution.total_latent),
        "target_total_rgb_frames": int(solution.total_rgb),
        "merged_rgb_identity": "sum(per_shot_rgb) == 1 + 4 * (total_latent - 1)",
        "per_shot_latent": [int(x) for x in solution.latent_lengths],
    }
    (out_sub / "sample.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _run_one_sample(
    sample_id: str,
    sample_data: dict,
    dest_root: Path,
    total_latent: int,
    fourcc: int,
) -> dict:
    """Solve + encode one sample; used serially and from worker processes."""
    segments = sample_data.get("segments")
    if not isinstance(segments, list) or len(segments) != 6:
        return {"id": sample_id, "status": "skip_not_6_segments"}
    solution = solve_one_sample(segments, total_latent=total_latent)
    if solution is None:
        return {"id": sample_id, "status": "skip_no_feasible_solution"}
    try:
        process_and_write_sample(sample_id, sample_data, solution, dest_root, fourcc)
    except Exception as e:
        return {"id": sample_id, "status": "error_process_video", "reason": str(e)}
    return {
        "id": sample_id,
        "status": "ok",
        "perm_1based": [i + 1 for i in solution.permutation],
        "latent_lengths": list(solution.latent_lengths),
        "rgb_lengths": list(solution.rgb_lengths),
        "total_latent": solution.total_latent,
        "total_rgb": solution.total_rgb,
        "trimmed_frames": solution.trimmed_frames,
    }


def _mp_run_one_sample_packed(item: dict) -> dict:
    """Picklable entry for ProcessPoolExecutor (Windows spawn)."""
    return _run_one_sample(
        item["sample_id"],
        item["sample_data"],
        Path(item["dest_root"]),  
        int(item["total_latent"]),
        cv2.VideoWriter_fourcc(*item["codec"]),
    )


def main() -> int:
    args = parse_args()
    if args.total_latent < 16:
        raise SystemExit("--total-latent must be >= 16 (need n1>=1 and five shots >=3 each).")
    src_root: Path = args.source_root
    if not src_root.is_dir():
        raise SystemExit(f"source root missing: {src_root}")

    if args.dest_root is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_root = (
            src_root.parent / f"{src_root.name}_latest_lat{args.total_latent}_rgb{1 + 4 * (args.total_latent - 1)}_{ts}"
        )
    else:
        dest_root = args.dest_root

    if dest_root.exists() and any(dest_root.iterdir()) and not args.overwrite:
        raise SystemExit(f"destination not empty: {dest_root} (use --overwrite)")
    dest_root.mkdir(parents=True, exist_ok=True)

    workers = args.workers if args.workers > 0 else _default_workers()
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    ok = 0
    fail = 0
    summary: list[dict] = []

    jobs: list[dict] = []
    for subdir in _iter_subdirs(src_root):
        sample_path = subdir / "sample.json"
        if not sample_path.is_file():
            fail += 1
            summary.append({"id": subdir.name, "status": "skip_no_sample_json"})
            continue
        try:
            sample_data = json.loads(sample_path.read_text(encoding="utf-8"))
        except Exception as e:
            fail += 1
            summary.append({"id": subdir.name, "status": "skip_bad_json", "reason": str(e)})
            continue

        jobs.append(
            {
                "sample_id": subdir.name,
                "sample_data": sample_data,
                "dest_root": str(dest_root.resolve()),
                "total_latent": args.total_latent,
                "codec": args.codec,
            }
        )

    def _apply_result(res: dict) -> None:
        nonlocal ok, fail
        if res.get("status") == "ok":
            ok += 1
            print(
                f"[ok] {res['id']}: perm={res.get('perm_1based')} "
                f"latents={res.get('latent_lengths')} total_latent={res.get('total_latent')} "
                f"total_rgb={res.get('total_rgb')}",
                flush=True,
            )
        else:
            fail += 1
        summary.append(res)

    if workers <= 1:
        for job in jobs:
            res = _run_one_sample(
                job["sample_id"],
                job["sample_data"],
                dest_root,
                job["total_latent"],
                fourcc,
            )
            _apply_result(res)
    else:
        total = len(jobs)
        print(
            f"[parallel] {total} sample(s), workers={workers} (each loads/decodes video — reduce if OOM)",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=workers) as ex:
            future_to_id = {ex.submit(_mp_run_one_sample_packed, j): j["sample_id"] for j in jobs}
            done = 0
            for fut in as_completed(future_to_id):
                sid = future_to_id[fut]
                done += 1
                try:
                    res = fut.result()
                except Exception as e:
                    res = {"id": sid, "status": "error_worker", "reason": str(e)}
                print(f"[progress] {done}/{total} {sid}", flush=True)
                _apply_result(res)

    report = {
        "source_root": str(src_root),
        "dest_root": str(dest_root),
        "workers": workers,
        "target_total_latent": args.total_latent,
        "target_total_rgb": 1 + 4 * (args.total_latent - 1),
        "rules": {
            "shot1_latent": "n1 where n1 % 3 == 1 and n1 >= 1",
            "shot2_to_shot6_latent": "each n_i where n_i % 3 == 0 and n_i >= 3 (independent per shot)",
            "shot1_rgb": "1 + (n1 - 1) * 4",
            "shot2_to_shot6_rgb": "latent_i * 4",
            "fixed_merged_totals": "sum(latent_i) == --total-latent; sum(rgb_i) == 1 + 4*(that-1)",
        },
        "processed_ok": ok,
        "failed_or_skipped": fail,
        "items": summary,
    }
    (dest_root / "_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] ok={ok} failed_or_skipped={fail}")
    print(f"[dest] {dest_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
