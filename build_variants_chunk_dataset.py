"""
From H:\\dataset\\movie_raw_dataset_variants (1/2/3_character_6_shot), extract
6-shot clips whose segment latent lengths (L1..L6) sum to TOTAL_LATENT (126).
Frame counts follow _build_all_dataset: first shot ((L1-1)*4+1), others Li*4.

Each Li must be a cumulative sum along the cyclic template
[1,3,3,3,3,3,3,2] (looped), e.g. 1,4,7,...,21,22,25,.... Among all feasible
tuples, pick the most balanced one toward (21,21,21,21,21,21).

Optional `--max-latent-tuple-iterations N` (N>0): enumerate at most N feasible
(L1..L6) tuples; if more exist, the sample is invalid. N=0 uses exact DP (default).

Invalid samples are deleted by default (pass --keep-invalid to keep them).

Parallelism: validation stays serial; video decode/encode uses `--workers` (default
capped at 4). Merged output is streamed to limit RAM. With `--workers 1` and `--seed`,
RNG matches the historical single-process behavior; multiple workers use a per-index
derived RNG. Use `--stall-timeout` to abort if no worker finishes in N seconds.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import zlib
import statistics
import re
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import cv2

# Six segments must sum to this latent count.
TOTAL_LATENT = 126
# Ideal per-shot latent count for a balanced split (126 / 6).
TARGET_LATENT_PER_SHOT = TOTAL_LATENT // 6
TEMPLATE_CHUNKS = [1, 3, 3, 3, 3, 3, 3, 2]


def _prefix_sums(values: list[int]) -> list[int]:
    out: list[int] = []
    s = 0
    for v in values:
        s += v
        out.append(s)
    return out


ALLOWED_CHUNK_LATENT_LENGTHS = tuple(_prefix_sums(TEMPLATE_CHUNKS))
# Only these part sizes appear in ordered chunkings (template-prefix lengths).
ALLOWED_PARTS: tuple[int, ...] = tuple(sorted(ALLOWED_CHUNK_LATENT_LENGTHS))

DEFAULT_SOURCE_ROOT = Path(r"F:\dataset\movie_raw_dataset_variants")
DEFAULT_OUTPUT_ROOT = Path(r"H:\dataset\movie_variants_chunked")

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
        key=lambda p: numeric_sort_key(p.stem),
    )


def latent_lengths_to_frame_lengths(latent_lengths: list[int]) -> list[int]:
    if not latent_lengths:
        raise ValueError("latent_lengths cannot be empty")
    frame_lengths = [((latent_lengths[0] - 1) * 4) + 1]
    frame_lengths.extend(length * 4 for length in latent_lengths[1:])
    return frame_lengths


def max_latent_for_shot(usable: int, shot_index: int) -> int:
    """Max Li such that required RGB fits in usable (excluding first/last frame)."""
    if usable < 1:
        return 0
    if shot_index == 0:
        return (usable - 1) // 4 + 1
    return usable // 4


def is_valid_chunk_latent_length(n: int) -> bool:
    return n in ALLOWED_CHUNK_LATENT_LENGTHS


def chunk_latent_length_to_pattern(length: int) -> list[int]:
    if not is_valid_chunk_latent_length(length):
        raise ValueError(f"Invalid chunk latent length: {length}")
    s = 0
    for i, v in enumerate(TEMPLATE_CHUNKS):
        s += v
        if s == length:
            return TEMPLATE_CHUNKS[: i + 1]
    raise ValueError(f"Length {length} is not a template-prefix sum.")


def cyclic_prefix_latents_upto(max_lat: int) -> list[int]:
    """
    All cumulative sums from repeatedly looping TEMPLATE_CHUNKS, clipped by max_lat.
    Example: 1,4,7,10,13,16,19,21,22,25,...
    """
    if max_lat < 1:
        return []
    out: list[int] = []
    s = 0
    i = 0
    n = len(TEMPLATE_CHUNKS)
    while True:
        s += TEMPLATE_CHUNKS[i % n]
        if s > max_lat:
            break
        out.append(s)
        i += 1
    return out


def allowed_latents_for_shot(usable: int, shot_index: int) -> list[int]:
    """Allowed Li values for this shot: cyclic-template cumulative sums within RGB capacity."""
    max_lat = max_latent_for_shot(usable, shot_index)
    return cyclic_prefix_latents_upto(max_lat)


def can_achieve_total_latent(usables: list[int], total: int = TOTAL_LATENT) -> bool:
    """Whether some (L1..L6) with Li in cyclic-template cumulative set sums to `total`."""
    choices = [allowed_latents_for_shot(usables[i], i) for i in range(6)]
    if any(not c for c in choices):
        return False
    if sum(max(c) for c in choices) < total:
        return False
    dp = {0}
    for i in range(6):
        nxt: set[int] = set()
        for s in dp:
            for L in choices[i]:
                ns = s + L
                if ns <= total:
                    nxt.add(ns)
        dp = nxt
    return total in dp


def iter_latent_tuples(usables: list[int], total: int = TOTAL_LATENT):
    """All feasible (L1..L6) with Li in cyclic-template cumulative set and sum = total."""
    choices = [allowed_latents_for_shot(usables[i], i) for i in range(6)]
    if any(not c for c in choices):
        return

    def dfs(idx: int, rem: int, cur: list[int]):
        if idx == 5:
            if rem in choices[5]:
                yield tuple(cur + [rem])
            return
        # Prune by reachable min/max remainder of later shots.
        min_rest = [0] * 7
        max_rest = [0] * 7
        for j in range(5, -1, -1):
            min_rest[j] = min_rest[j + 1] + min(choices[j])
            max_rest[j] = max_rest[j + 1] + max(choices[j])
        for L in choices[idx]:
            rem2 = rem - L
            if rem2 < min_rest[idx + 1] or rem2 > max_rest[idx + 1]:
                continue
            yield from dfs(idx + 1, rem2, cur + [L])

    yield from dfs(0, total, [])


def _latent_tuple_balance_key(lt: tuple[int, ...]) -> tuple[int, int, tuple[int, ...]]:
    """Lower is better: min sum (Li-21)^2, then min range, then lex tie-break."""
    dev_sq = sum((x - TARGET_LATENT_PER_SHOT) ** 2 for x in lt)
    span = max(lt) - min(lt)
    return (dev_sq, span, lt)


def _pick_most_balanced_latent_tuple_dp(usables: list[int]) -> tuple[int, ...] | None:
    """Exact minimum sum_i (Li-21)^2 via DP (no iteration cap)."""
    m = TARGET_LATENT_PER_SHOT
    choices = [allowed_latents_for_shot(usables[i], i) for i in range(6)]
    if any(not c for c in choices):
        return None
    costs: dict[int, int] = {0: 0}
    back_layers: list[dict[int, tuple[int, int]]] = []

    for i in range(6):
        next_costs: dict[int, int] = {}
        next_back: dict[int, tuple[int, int]] = {}
        for s, c in costs.items():
            for L in choices[i]:
                ns = s + L
                if ns > TOTAL_LATENT:
                    continue
                nc = c + (L - m) ** 2
                if ns not in next_costs or nc < next_costs[ns] or (
                    nc == next_costs[ns] and L < next_back[ns][1]
                ):
                    next_costs[ns] = nc
                    next_back[ns] = (s, L)
        costs = next_costs
        back_layers.append(next_back)

    if TOTAL_LATENT not in costs:
        return None

    s = TOTAL_LATENT
    lt_rev: list[int] = []
    for i in range(5, -1, -1):
        ps, L = back_layers[i][s]
        lt_rev.append(L)
        s = ps
    lt_rev.reverse()
    return tuple(lt_rev)


def pick_most_balanced_latent_tuple(
    usables: list[int], *, max_latent_tuple_iterations: int = 0
) -> tuple[tuple[int, ...] | None, str | None]:
    """
    Returns (latent_tuple, discard_reason). discard_reason is set when the sample
    should be abandoned (cap exceeded or infeasible).
    """
    choices = [allowed_latents_for_shot(usables[i], i) for i in range(6)]
    no_choice = [i + 1 for i, c in enumerate(choices) if not c]
    if no_choice:
        return (
            None,
            f"shot(s) {no_choice} cannot fit even the minimum allowed latent length "
            f"{ALLOWED_CHUNK_LATENT_LENGTHS[0]}",
        )

    m = TARGET_LATENT_PER_SHOT
    if all(m in c for c in choices):
        return ((m,) * 6, None)

    if max_latent_tuple_iterations <= 0:
        lt = _pick_most_balanced_latent_tuple_dp(usables)
        if lt is None:
            return (None, "no feasible latent tuple under cyclic-template constraints (DP infeasible)")
        return (lt, None)

    it = iter(iter_latent_tuples(usables))
    best: tuple[int, ...] | None = None
    best_key: tuple[int, int, tuple[int, ...]] | None = None
    n = 0
    while True:
        lt = next(it, None)
        if lt is None:
            if best is None:
                return (None, "no feasible latent tuple under cyclic-template constraints (enumeration empty)")
            return (best, None)
        if n >= max_latent_tuple_iterations:
            return (
                None,
                f"exceeded --max-latent-tuple-iterations ({max_latent_tuple_iterations}); "
                "more feasible (L1..L6) tuples exist",
            )
        n += 1
        k = _latent_tuple_balance_key(lt)
        if best_key is None or k < best_key:
            best_key = k
            best = lt
            if k[0] == 0:
                return (lt, None)


def _composition_better(a: tuple[int, tuple[int, ...]], b: tuple[int, tuple[int, ...]]) -> bool:
    na, ta = a
    nb, tb = b
    if na != nb:
        return na < nb
    va = statistics.pvariance(ta) if len(ta) > 1 else 0.0
    vb = statistics.pvariance(tb) if len(tb) > 1 else 0.0
    if va != vb:
        return va < vb
    return ta < tb


def pick_balanced_chunk_composition(L: int) -> list[int]:
    """
    Ordered composition of L using ALLOWED_PARTS: minimize chunk count, then
    variance of chunk sizes, then lexicographic order. DP in O(L), avoids
    enumerating exponentially many compositions for large L.
    """
    if L < 1:
        raise ValueError(L)
    best: dict[int, tuple[int, tuple[int, ...]]] = {0: (0, ())}
    for s in range(1, L + 1):
        chosen: tuple[int, tuple[int, ...]] | None = None
        for p in ALLOWED_PARTS:
            if p > s:
                continue
            n0, tail = best[s - p]
            cand = (n0 + 1, (p,) + tail)
            if chosen is None or _composition_better(cand, chosen):
                chosen = cand
        if chosen is None:
            raise RuntimeError(f"no valid chunk composition for segment latent length {L}")
        best[s] = chosen
    return list(best[L][1])


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


def latent_chunks_to_rgb_chunk_lengths(shot_index: int, chunk_latents: list[int]) -> list[int]:
    if shot_index == 0:
        out = [((chunk_latents[0] - 1) * 4) + 1]
        out.extend(c * 4 for c in chunk_latents[1:])
        return out
    return [c * 4 for c in chunk_latents]


def build_switch_lists(
    segment_latents: list[int],
    chunk_splits_per_shot: list[list[int]],
) -> tuple[list[int], list[int], list[list[int]], list[list[list[int]]]]:
    segment_frame_lengths = latent_lengths_to_frame_lengths(segment_latents)
    switch_latent: list[int] = []
    switch_rgb: list[int] = []
    latent_cursor = 0
    rgb_cursor = 0
    chunk_lat_lens: list[list[int]] = []
    chunk_patterns: list[list[list[int]]] = []

    for shot_i, seg_rgb_len in enumerate(segment_frame_lengths):
        chunks = list(chunk_splits_per_shot[shot_i])
        if sum(chunks) != segment_latents[shot_i]:
            raise RuntimeError(
                f"shot {shot_i}: chunk latent sum {sum(chunks)} != segment latent {segment_latents[shot_i]}"
            )
        chunk_lat_lens.append(chunks)
        chunk_patterns.append([chunk_latent_length_to_pattern(c) for c in chunks])
        rgb_lens = latent_chunks_to_rgb_chunk_lengths(shot_i, chunks)
        if sum(rgb_lens) != seg_rgb_len:
            raise RuntimeError(
                f"RGB/chunk sum mismatch shot={shot_i}: rgb={sum(rgb_lens)} expected={seg_rgb_len}"
            )
        cum_lat = 0
        cum_rgb = 0
        for ci, cl in enumerate(chunks):
            cum_lat += cl
            cum_rgb += rgb_lens[ci]
            # Record every global chunk boundary except the very end of sample.
            is_last_global_chunk = (
                shot_i == len(segment_frame_lengths) - 1 and ci == len(chunks) - 1
            )
            if not is_last_global_chunk:
                switch_latent.append(latent_cursor + cum_lat)
                switch_rgb.append(rgb_cursor + cum_rgb)
        latent_cursor += segment_latents[shot_i]
        rgb_cursor += seg_rgb_len

    return switch_latent, switch_rgb, chunk_lat_lens, chunk_patterns


def list_usables(videos: list[Path]) -> tuple[list[int] | None, str]:
    if len(videos) != 6:
        return None, f"expected 6 mp4 files, got {len(videos)}"
    usables: list[int] = []
    for vp in videos:
        try:
            fc = get_frame_count(vp)
        except RuntimeError as e:
            return None, str(e)
        usables.append(fc - 2)
    return usables, ""


def process_one_split(
    source_folder: Path,
    videos: list[Path],
    output_folder: Path,
    merged_video_path: Path,
    json_path: Path,
    latent_lengths: list[int],
    chunk_splits_per_shot: list[list[int]],
    rng: random.Random,
    *,
    output_index: int | None = None,
    verbose: bool = False,
) -> dict:
    segment_lengths = latent_lengths_to_frame_lengths(latent_lengths)
    switch_latent, switch_rgb, chunk_lat_lens, chunk_patterns = build_switch_lists(
        latent_lengths, chunk_splits_per_shot
    )

    output_folder.mkdir(parents=True, exist_ok=True)
    merged_writer = None
    merged_fps = None
    merged_size = None
    total_rgb = 0
    segments_meta = []

    try:
        for idx, (video_path, segment_length, li) in enumerate(
            zip(videos, segment_lengths, latent_lengths), start=1
        ):
            if verbose:
                tag = f"index={output_index} " if output_index is not None else ""
                print(
                    f"  [{tag}shot {idx}/6] read/encode {segment_length} RGB frames from {video_path.name}…",
                    flush=True,
                )
            frame_count, fps, width, height = get_video_info(video_path)
            start_frame, end_frame = choose_contiguous_window(frame_count, segment_length, rng)
            frames, clip_fps, clip_size = read_frame_segment(video_path, start_frame, segment_length)
            segment_path = output_folder / f"shot{idx}.mp4"
            write_video(segment_path, frames, clip_fps, clip_size)
            if merged_fps is None:
                merged_fps = clip_fps
                merged_size = clip_size
                ensure_parent(merged_video_path)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                merged_writer = cv2.VideoWriter(
                    str(merged_video_path), fourcc, merged_fps, merged_size
                )
                if not merged_writer.isOpened():
                    raise RuntimeError(f"Cannot create merged video: {merged_video_path}")
            resized = frames
            if clip_size != merged_size:
                resized = [cv2.resize(f, merged_size, interpolation=cv2.INTER_AREA) for f in frames]
            for frame in resized:
                merged_writer.write(frame)
            total_rgb += len(resized)
            segments_meta.append(
                {
                    "source_video": str(video_path),
                    "output_segment": str(segment_path),
                    "frame_count_total": frame_count,
                    "latent_length": li,
                    "rgb_frames_extracted": segment_length,
                    "latent_from_rgb": (segment_length - 1) // 4 + 1,
                    "selected_start_frame": start_frame,
                    "selected_end_frame_exclusive": end_frame,
                    "chunk_latent_lengths": chunk_lat_lens[idx - 1],
                    "chunk_latent_patterns": chunk_patterns[idx - 1],
                }
            )
            del frames
            del resized

        if verbose:
            tag = f"index={output_index} " if output_index is not None else ""
            print(
                f"  [{tag}merge] merged video done ({total_rgb} frames, streamed)…",
                flush=True,
            )
    finally:
        if merged_writer is not None:
            merged_writer.release()
    payload = {
        "source_folder": str(source_folder),
        "video_names": [v.name for v in videos],
        "output_folder": str(output_folder),
        "merged_video": str(merged_video_path),
        "segment_latent_lengths": list(latent_lengths),
        "chunk_splits_per_shot": [list(x) for x in chunk_splits_per_shot],
        "total_frame_length": total_rgb,
        "total_latent_frame_length": TOTAL_LATENT,
        "switch_frames": switch_rgb,
        "switch_latent_frames": switch_latent,
        "segments": segments_meta,
        "reference_template": [1, 3, 3, 3, 3, 3, 3, 2],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _job_rng(base_seed: int | None, output_index: int, sample_dir_str: str) -> random.Random:
    """
    Per-sample RNG for multiprocessing (--workers > 1). With a single worker, main()
    uses one advancing Random(args.seed) so runs match the historical behavior.
    """
    if base_seed is None:
        h = zlib.adler32(sample_dir_str.encode("utf-8", errors="replace")) & 0xFFFFFFFF
        return random.Random(int(h) ^ (output_index * 2654435761))
    x = (base_seed ^ (output_index * 0x9E3779B9)) & 0xFFFFFFFF
    return random.Random(x if x != 0 else 1)


def _mp_encode_sample(
    output_root_str: str,
    char_name: str,
    sample_dir_str: str,
    video_path_strs: list[str],
    latent_lengths: list[int],
    chunk_splits: list[list[int]],
    output_index: int,
    base_seed: int | None,
    verbose: bool,
) -> dict:
    rng = _job_rng(base_seed, output_index, sample_dir_str)
    sample_dir = Path(sample_dir_str)
    videos = [Path(p) for p in video_path_strs]
    out_root = Path(output_root_str)
    out_dir = out_root / str(output_index)
    merged = out_root / f"{output_index}.mp4"
    meta_json = out_dir / "sample.json"
    return process_one_split(
        sample_dir,
        videos,
        out_dir,
        merged,
        meta_json,
        latent_lengths,
        chunk_splits,
        rng,
        output_index=output_index,
        verbose=verbose,
    )


def _default_worker_count() -> int:
    n = os.cpu_count() or 1
    return max(1, min(n//2, 4))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build 6-shot chunked dataset from movie_raw_dataset_variants.")
    p.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--keep-invalid",
        action="store_true",
        help="Do not delete invalid source sample folders (default is delete).",
    )
    p.add_argument("--clear-output", action="store_true", help="Remove output root before writing.")
    p.add_argument("--limit-samples", type=int, default=None, help="Max samples per character folder (debug).")
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print per-sample details (usables, paths, timing).",
    )
    p.add_argument(
        "--max-latent-tuple-iterations",
        type=int,
        default=0,
        help=(
            "0 (default): exact DP for balanced (L1..L6), fastest. "
            "Positive N: enumerate at most N feasible tuples; if any further tuple exists, "
            "treat sample as invalid (abandon folder). Example: 2000000."
        ),
    )
    p.add_argument(
        "--workers",
        type=int,
        default=_default_worker_count(),
        help=(
            "Parallel processes for the video decode/encode phase only. "
            "Lower this if RAM is tight (each task buffers one shot at a time; merged file is streamed). "
            "Use 1 for single-process (same RNG stream as before when --seed is set)."
        ),
    )
    p.add_argument(
        "--stall-timeout",
        type=float,
        default=0.0,
        help=(
            "When using multiple workers: if no encode task finishes in this many seconds, "
            "abort (0 = disabled). Helps detect stuck decoder/encoder I/O."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.source_root.is_dir():
        raise SystemExit(f"Source root not found: {args.source_root}")

    if args.clear_output and args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = args.output_root / "manifest.jsonl"
    removed_path = args.output_root / "removed_sources.jsonl"
    global_index = 1

    char_folders = [
        args.source_root / name
        for name in ("1_character_6_shot", "2_character_6_shot", "3_character_6_shot")
    ]

    stats = {"samples_seen": 0, "invalid": 0, "deleted": 0, "outputs_written": 0}
    t0_all = time.perf_counter()

    print(
        "build_variants_chunk_dataset — start\n"
        f"  source-root: {args.source_root}\n"
        f"  output-root: {args.output_root}\n"
        f"  seed: {args.seed!r}  keep-invalid: {args.keep_invalid}\n"
        f"  latent split: each Li must be a cyclic-template cumulative sum of {TEMPLATE_CHUNKS}; "
        f"total=126, objective=most balanced toward ({TARGET_LATENT_PER_SHOT},)*6\n"
        f"  chunk split: deterministic DP (fewest parts, then lowest variance)\n"
        f"  (if stuck, I/O is usually decoding/encoding — try -v)\n"
        f"  max-latent-tuple-iterations: {args.max_latent_tuple_iterations} "
        f"(0 = exact DP; >0 = cap, excess feasible tuples => invalid sample)\n"
        f"  encode workers: {args.workers} (video I/O only; --workers 1 keeps one RNG stream if --seed is set)\n"
        f"  stall-timeout: {args.stall_timeout}s "
        f"({'off' if args.stall_timeout <= 0 else 'on — abort if no task completes in window'})\n"
        f"  two phases: (1) scan all source folders — [plan] lines have NO files on disk yet; "
        f"(2) encode — then {args.output_root.name}/N/ and N.mp4 appear.\n"
        f"  manifest: {manifest_path}\n"
        f"  removed log: {removed_path}",
        flush=True,
    )

    with manifest_path.open("w", encoding="utf-8") as manifest, removed_path.open(
        "w", encoding="utf-8"
    ) as removed_log:
        encode_jobs: list[dict] = []
        global_index = 1
        for char_root in char_folders:
            if not char_root.is_dir():
                print(f"[warn] skip missing character folder: {char_root}", flush=True)
                continue
            samples = list_subdirs(char_root)
            if args.limit_samples is not None:
                samples = samples[: args.limit_samples]
            print(
                f"[folder] {char_root.name}: {len(samples)} sample dir(s)",
                flush=True,
            )
            for sample_dir in samples:
                stats["samples_seen"] += 1
                videos = list_mp4s(sample_dir)
                usables, err = list_usables(videos)
                if usables is None:
                    _log_invalid(sample_dir, err, removed_log, delete_invalid=not args.keep_invalid, stats=stats)
                    continue
                if not can_achieve_total_latent(usables):
                    _log_invalid(
                        sample_dir,
                        "no (L1..L6) with sum 126 fits in per-video usable frames",
                        removed_log,
                        delete_invalid=not args.keep_invalid,
                        stats=stats,
                    )
                    continue

                if args.verbose:
                    print(
                        f"[run] {char_root.name}/{sample_dir.name}: usables={usables} "
                        f"(frame_count-2 per video)",
                        flush=True,
                    )
                else:
                    print(
                        f"[run] {char_root.name}/{sample_dir.name}: constrained latent + chunk split…",
                        flush=True,
                    )
                t_enum = time.perf_counter()
                latent_tuple, latent_discard_reason = pick_most_balanced_latent_tuple(
                    usables,
                    max_latent_tuple_iterations=args.max_latent_tuple_iterations,
                )
                dt_enum = time.perf_counter() - t_enum
                if latent_tuple is None:
                    _log_invalid(
                        sample_dir,
                        latent_discard_reason or "no feasible latent tuple",
                        removed_log,
                        delete_invalid=not args.keep_invalid,
                        stats=stats,
                    )
                    continue
                try:
                    chunk_splits = [pick_balanced_chunk_composition(L) for L in latent_tuple]
                except RuntimeError as e:
                    _log_invalid(sample_dir, str(e), removed_log, delete_invalid=not args.keep_invalid, stats=stats)
                    continue

                bk = _latent_tuple_balance_key(latent_tuple)
                print(
                    f"[run] {char_root.name}/{sample_dir.name}: "
                    f"latents={list(latent_tuple)} balance(sum_sq={bk[0]}, range={bk[1]}) "
                    f"in {dt_enum:.2f}s",
                    flush=True,
                )

                out_dir = args.output_root / str(global_index)
                merged = args.output_root / f"{global_index}.mp4"
                print(
                    f"[plan] index={global_index} -> {out_dir.name}/ + {merged.name} "
                    f"(latents={list(latent_tuple)}) — not written until encode phase after full scan",
                    flush=True,
                )
                encode_jobs.append(
                    {
                        "output_index": global_index,
                        "char_name": char_root.name,
                        "sample_dir": str(sample_dir),
                        "sample_basename": sample_dir.name,
                        "videos": [str(v) for v in videos],
                        "latent_tuple": list(latent_tuple),
                        "chunk_splits": [list(x) for x in chunk_splits],
                        "balance_key": bk,
                    }
                )
                global_index += 1

        if not encode_jobs:
            print("[info] No valid samples to encode.", flush=True)
        else:
            out_abs = str(args.output_root.resolve())
            print(
                f"\n{'=' * 72}\n"
                f"[encode-phase] {len(encode_jobs)} sample(s) -> {out_abs}\n"
                f"  Writing indices 1..{encode_jobs[-1]['output_index']} "
                f"({'1 worker' if args.workers <= 1 else f'{args.workers} workers'}).\n"
                f"{'=' * 72}\n",
                flush=True,
            )
            if args.workers <= 1:
                rng = random.Random(args.seed) if args.seed is not None else random.Random()
                for job in encode_jobs:
                    idx = job["output_index"]
                    bk = job["balance_key"]
                    sample_dir = Path(job["sample_dir"])
                    videos = [Path(p) for p in job["videos"]]
                    out_dir = args.output_root / str(idx)
                    merged = args.output_root / f"{idx}.mp4"
                    meta_json = out_dir / "sample.json"
                    t_write = time.perf_counter()
                    meta = process_one_split(
                        sample_dir,
                        videos,
                        out_dir,
                        merged,
                        meta_json,
                        job["latent_tuple"],
                        job["chunk_splits"],
                        rng,
                        output_index=idx,
                        verbose=args.verbose,
                    )
                    dt_write = time.perf_counter() - t_write
                    meta["output_index"] = idx
                    meta["character_folder"] = job["char_name"]
                    meta["source_sample_dir"] = job["sample_dir"]
                    meta["target_latent_per_shot"] = TARGET_LATENT_PER_SHOT
                    meta["latent_balance"] = {
                        "sum_sq_deviation_from_target": bk[0],
                        "range_max_minus_min": bk[1],
                    }
                    manifest.write(json.dumps(meta, ensure_ascii=False) + "\n")
                    lt = meta["segment_latent_lengths"]
                    stats["outputs_written"] += 1
                    print(
                        f"[ok] index={idx} <- {job['sample_basename']} "
                        f"latents={lt} sum={sum(lt)} (wrote in {dt_write:.1f}s)",
                        flush=True,
                    )
            else:
                output_root_str = str(args.output_root)
                future_to_idx = {}
                with ProcessPoolExecutor(max_workers=args.workers) as ex:
                    for job in encode_jobs:
                        idx = job["output_index"]
                        fut = ex.submit(
                            _mp_encode_sample,
                            output_root_str,
                            job["char_name"],
                            job["sample_dir"],
                            job["videos"],
                            job["latent_tuple"],
                            job["chunk_splits"],
                            idx,
                            args.seed,
                            args.verbose,
                        )
                        future_to_idx[fut] = idx
                    pending = set(future_to_idx.keys())
                    results: dict[int, dict] = {}
                    stall_s = args.stall_timeout
                    try:
                        while pending:
                            timeout = stall_s if stall_s and stall_s > 0 else None
                            done, pending = wait(pending, return_when=FIRST_COMPLETED, timeout=timeout)
                            if not done:
                                print(
                                    f"[fatal] Stall: no encode finished in {stall_s}s; "
                                    "increase --stall-timeout or use --workers 1 to debug.",
                                    flush=True,
                                )
                                ex.shutdown(wait=False, cancel_futures=True)
                                print(
                                    "\nAborted (encode stall). Partial files may exist under output-root.",
                                    flush=True,
                                )
                                return 2
                            for fut in done:
                                idx = future_to_idx[fut]
                                results[idx] = fut.result()
                                print(
                                    f"[encoded] index={idx} -> "
                                    f"{Path(output_root_str) / str(idx)} + "
                                    f"{Path(output_root_str) / f'{idx}.mp4'}",
                                    flush=True,
                                )
                    except Exception as e:
                        print(f"[fatal] encode failed: {e}", flush=True)
                        ex.shutdown(wait=False, cancel_futures=True)
                        raise

                for job in encode_jobs:
                    idx = job["output_index"]
                    bk = job["balance_key"]
                    meta = results[idx]
                    meta["output_index"] = idx
                    meta["character_folder"] = job["char_name"]
                    meta["source_sample_dir"] = job["sample_dir"]
                    meta["target_latent_per_shot"] = TARGET_LATENT_PER_SHOT
                    meta["latent_balance"] = {
                        "sum_sq_deviation_from_target": bk[0],
                        "range_max_minus_min": bk[1],
                    }
                    manifest.write(json.dumps(meta, ensure_ascii=False) + "\n")
                    lt = meta["segment_latent_lengths"]
                    stats["outputs_written"] += 1
                    print(
                        f"[ok] index={idx} <- {job['sample_basename']} latents={lt} sum={sum(lt)}",
                        flush=True,
                    )

    elapsed = time.perf_counter() - t0_all
    print(
        f"\nDone in {elapsed:.1f}s.\n"
        f"  samples_seen: {stats['samples_seen']}\n"
        f"  invalid: {stats['invalid']}\n"
        f"  deleted: {stats['deleted']}\n"
        f"  outputs_written: {stats['outputs_written']}\n"
        f"  last output index: {global_index - 1}\n"
        f"  manifest: {manifest_path}",
        flush=True,
    )
    return 0


def _log_invalid(
    sample_dir: Path,
    reason: str,
    removed_log,
    delete_invalid: bool,
    stats: dict | None = None,
) -> None:
    entry = {"path": str(sample_dir), "reason": reason}
    removed_log.write(json.dumps(entry, ensure_ascii=False) + "\n")
    if stats is not None:
        stats["invalid"] = stats.get("invalid", 0) + 1
    print(f"[invalid] {sample_dir} -> {reason}", flush=True)
    if delete_invalid:
        shutil.rmtree(sample_dir)
        if stats is not None:
            stats["deleted"] = stats.get("deleted", 0) + 1
        print(f"  deleted {sample_dir}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
