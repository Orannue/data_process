"""
Build 6-shot raw dataset variants only:
  - 1_character_6_shot   (six shots, one character)
  - 2_character_6_shot   (char1, char2, char1, char2, char1, char2; triple per char)
  - 3_character_6_shot   (char1, char2, char3, char1, char2, char3; pair per character)

Only shots with frame count >= min_frames (default 80) are used.

Reuses helpers from generate_raw_dataset.py. Dedupes identical shot-sets and,
within each scene, prefers combos that do not reuse shots already picked
(non-overlap greedy, then fill if needed).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from generate_raw_dataset import (
    collect_characters,
    ensure_dir,
    list_subdirs,
    save_combo,
)

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "generate_raw_dataset_variants requires opencv-python (cv2) for frame-count filtering."
    ) from e


def _combo_key(combo: Tuple[Path, ...]) -> tuple[str, ...]:
    return tuple(sorted(p.as_posix().lower() for p in combo))


def _video_frame_count(video_path: Path) -> int | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return n if n > 0 else None
    finally:
        cap.release()


def filter_character_shots_min_frames(
    character_to_shots: Dict[str, List[Path]],
    min_frames: int,
) -> Dict[str, List[Path]]:
    """Keep only shots whose reported frame count is >= min_frames."""
    if min_frames <= 0:
        return dict(character_to_shots)
    out: Dict[str, List[Path]] = {}
    for name, shots in character_to_shots.items():
        kept: List[Path] = []
        for p in shots:
            fc = _video_frame_count(p)
            if fc is not None and fc >= min_frames:
                kept.append(p)
        if kept:
            out[name] = kept
    return out


def _default_parallel_workers() -> int:
    """Conservative default: ~half of logical CPUs, capped (avoids freezing the machine)."""
    n = os.cpu_count() or 4
    return max(1, min(8, max(1, n // 2)))


def select_diverse_combos(
    candidates: List[Tuple[Path, ...]],
    max_n: int,
    rng: random.Random,
) -> List[Tuple[Path, ...]]:
    """Prefer combos with no shot overlap; then fill up to max_n allowing overlap."""
    if max_n <= 0 or not candidates:
        return []
    order = list(candidates)
    rng.shuffle(order)
    selected: List[Tuple[Path, ...]] = []
    used_paths: set[str] = set()

    for combo in order:
        if len(selected) >= max_n:
            break
        paths = {p.as_posix().lower() for p in combo}
        if paths & used_paths:
            continue
        selected.append(combo)
        used_paths |= paths

    if len(selected) < max_n:
        remaining = [c for c in order if c not in selected]
        rng.shuffle(remaining)
        for combo in remaining:
            if len(selected) >= max_n:
                break
            selected.append(combo)

    return selected


def build_1char_6shot(
    character_to_shots: Dict[str, List[Path]],
    max_per_scene: int,
    rng: random.Random,
    max_candidate_pool: int,
) -> List[Tuple[Path, Path, Path, Path, Path, Path]]:
    all_hex: List[Tuple[Path, Path, Path, Path, Path, Path]] = []
    seen: set[tuple[str, ...]] = set()
    for _, shots in sorted(character_to_shots.items()):
        n = len(shots)
        for i, j, k, l, m, o in combinations(range(n), 6):
            combo = (shots[i], shots[j], shots[k], shots[l], shots[m], shots[o])
            key = _combo_key(combo)
            if key in seen:
                continue
            seen.add(key)
            all_hex.append(combo)
            if len(all_hex) >= max_candidate_pool:
                return select_diverse_combos(all_hex, max_per_scene, rng)
    return select_diverse_combos(all_hex, max_per_scene, rng)


def build_2char_6shot(
    character_to_shots: Dict[str, List[Path]],
    max_per_scene: int,
    rng: random.Random,
    max_candidate_pool: int,
) -> List[Tuple[Path, Path, Path, Path, Path, Path]]:
    """
    Order: char1 shot_i, char2 shot_p, char1 shot_j, char2 shot_q, char1 shot_k, char2 shot_r
    Each character uses three shots i<j<k / p<q<r.
    """
    names = list(character_to_shots.keys())
    candidates: List[Tuple[Path, Path, Path, Path, Path, Path]] = []
    seen: set[tuple[str, ...]] = set()
    stop = False
    for char1 in names:
        if stop:
            break
        s1 = character_to_shots[char1]
        if len(s1) < 3:
            continue
        for char2 in names:
            if stop:
                break
            if char1 == char2:
                continue
            s2 = character_to_shots[char2]
            if len(s2) < 3:
                continue

            for i, j, k in combinations(range(len(s1)), 3):
                if stop:
                    break
                for p, q, r in combinations(range(len(s2)), 3):
                    if stop:
                        break
                    a, b, c = s1[i], s1[j], s1[k]
                    u, v, w = s2[p], s2[q], s2[r]
                    combo = (a, u, b, v, c, w)
                    key = _combo_key(combo)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(combo)
                    if len(candidates) >= max_candidate_pool:
                        stop = True
                        break
                if stop:
                    break
            if stop:
                break
        if stop:
            break

    return select_diverse_combos(candidates, max_per_scene, rng)


def build_3char_6shot(
    character_to_shots: Dict[str, List[Path]],
    max_per_scene: int,
    rng: random.Random,
    max_candidate_pool: int,
) -> List[Tuple[Path, Path, Path, Path, Path, Path]]:
    """
    Order: c1_a, c2_b, c3_u, c1_c, c2_d, c3_v
    c1, c2, c3 each contribute two shots (pairs).
    """
    names = list(character_to_shots.keys())
    candidates: List[Tuple[Path, Path, Path, Path, Path, Path]] = []
    seen: set[tuple[str, ...]] = set()
    stop = False
    for c1 in names:
        if stop:
            break
        s1 = character_to_shots[c1]
        if len(s1) < 2:
            continue
        for c2 in names:
            if stop:
                break
            if c2 == c1:
                continue
            s2 = character_to_shots[c2]
            if len(s2) < 2:
                continue
            for c3 in names:
                if stop:
                    break
                if c3 in (c1, c2):
                    continue
                s3 = character_to_shots[c3]
                if len(s3) < 2:
                    continue

                for i, j in combinations(range(len(s1)), 2):
                    if stop:
                        break
                    for p, q in combinations(range(len(s2)), 2):
                        if stop:
                            break
                        shot_c1a, shot_c1c = s1[i], s1[j]
                        shot_c2b, shot_c2d = s2[p], s2[q]
                        for u, v in combinations(range(len(s3)), 2):
                            if stop:
                                break
                            shot_c3u, shot_c3v = s3[u], s3[v]
                            combo = (shot_c1a, shot_c2b, shot_c3u, shot_c1c, shot_c2d, shot_c3v)
                            key = _combo_key(combo)
                            if key in seen:
                                continue
                            seen.add(key)
                            candidates.append(combo)
                            if len(candidates) >= max_candidate_pool:
                                stop = True
                                break
                        if stop:
                            break
                    if stop:
                        break
                if stop:
                    break
            if stop:
                break
        if stop:
            break

    return select_diverse_combos(candidates, max_per_scene, rng)


def process_scene(
    scene_dir: Path,
    max_1c6: int,
    max_2c6: int,
    max_3c6: int,
    rng: random.Random,
    max_candidate_pool: int,
    min_frames: int,
) -> Tuple[
    List[Tuple[Path, Path, Path, Path, Path, Path]],
    List[Tuple[Path, Path, Path, Path, Path, Path]],
    List[Tuple[Path, Path, Path, Path, Path, Path]],
]:
    character_to_shots = filter_character_shots_min_frames(
        collect_characters(scene_dir), min_frames
    )
    if not character_to_shots:
        return [], [], []

    h1 = build_1char_6shot(character_to_shots, max_1c6, rng, max_candidate_pool)
    h2 = build_2char_6shot(character_to_shots, max_2c6, rng, max_candidate_pool)
    h3 = build_3char_6shot(character_to_shots, max_3c6, rng, max_candidate_pool)
    return h1, h2, h3


def _scene_rng_seed(base_seed: int, scene_dir: Path) -> int:
    msg = f"{base_seed}|{scene_dir.resolve().as_posix()}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(msg).digest()[:8], "little")


def _process_scene_worker(
    args: Tuple[Path, int, int, int, int, int, int],
) -> Tuple[
    Path,
    List[Tuple[Path, Path, Path, Path, Path, Path]],
    List[Tuple[Path, Path, Path, Path, Path, Path]],
    List[Tuple[Path, Path, Path, Path, Path, Path]],
]:
    scene_dir, max_1c6, max_2c6, max_3c6, scene_seed, max_candidate_pool, min_frames = args
    rng = random.Random(scene_seed)
    h1, h2, h3 = process_scene(
        scene_dir, max_1c6, max_2c6, max_3c6, rng, max_candidate_pool, min_frames
    )
    return scene_dir, h1, h2, h3


def _iter_scenes(input_root: Path) -> List[Path]:
    scenes: List[Path] = []
    for movie_dir in list_subdirs(input_root):
        for scene_dir in list_subdirs(movie_dir):
            scenes.append(scene_dir)
    return scenes


def process_dataset(
    input_root: Path,
    output_root: Path,
    seed: int,
    max_1c6: int = 5,
    max_2c6: int = 5,
    max_3c6: int = 5,
    jobs: int = 1,
    max_candidate_pool: int = 50_000,
    min_frames: int = 80,
) -> None:
    import shutil

    h1 = output_root / "1_character_6_shot"
    h2 = output_root / "2_character_6_shot"
    h3 = output_root / "3_character_6_shot"

    print(
        f"Preparing output folders under {output_root.resolve()} "
        f"(deleting previous 1/2/3_character_6_shot if they exist; can take a while if huge) ...",
        flush=True,
    )
    for case_dir in (h1, h2, h3):
        if case_dir.exists():
            shutil.rmtree(case_dir)
        ensure_dir(case_dir)

    m1, m2, m3 = 1, 1, 1
    scenes = _iter_scenes(input_root)
    print(
        f"Found {len(scenes)} scene(s). min_frames={min_frames}. "
        f"Writing under {output_root.resolve()}",
        flush=True,
    )
    if not scenes:
        print("Nothing to do (no movie/scene folders under input root).", flush=True)
        return

    if jobs <= 1:
        rng = random.Random(seed)
        for i, scene_dir in enumerate(scenes, start=1):
            print(
                f"Computing {i}/{len(scenes)} {scene_dir.parent.name}/{scene_dir.name} ...",
                flush=True,
            )
            s1, s2, s3 = process_scene(
                scene_dir,
                max_1c6,
                max_2c6,
                max_3c6,
                rng,
                max_candidate_pool,
                min_frames,
            )
            print(
                f"[{scene_dir.parent.name}/{scene_dir.name}] "
                f"1_character_6_shot={len(s1)}, "
                f"2_character_6_shot={len(s2)}, "
                f"3_character_6_shot={len(s3)}",
                flush=True,
            )
            for combo in s1:
                save_combo(h1, m1, combo)
                m1 += 1
            for combo in s2:
                save_combo(h2, m2, combo)
                m2 += 1
            for combo in s3:
                save_combo(h3, m3, combo)
                m3 += 1
        return

    worker_args: List[Tuple[Path, int, int, int, int, int, int]] = [
        (
            scene_dir,
            max_1c6,
            max_2c6,
            max_3c6,
            _scene_rng_seed(seed, scene_dir),
            max_candidate_pool,
            min_frames,
        )
        for scene_dir in scenes
    ]
    max_workers = min(jobs, len(worker_args))
    print(
        f"Parallel: {max_workers} worker(s); each scene is copied to disk as soon as it finishes "
        f"(no giant in-memory batch). max_candidate_pool={max_candidate_pool}.",
        flush=True,
    )
    print(
        "Starting worker pool (on Windows the first results may take a few seconds) ...",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_scene_worker, arg) for arg in worker_args]
        done = 0
        for fut in as_completed(futures):
            scene_dir, s1, s2, s3 = fut.result()
            done += 1
            print(
                f"[{done}/{len(worker_args)}] "
                f"[{scene_dir.parent.name}/{scene_dir.name}] "
                f"1_character_6_shot={len(s1)} "
                f"2_character_6_shot={len(s2)} "
                f"3_character_6_shot={len(s3)} — saving",
                flush=True,
            )
            for combo in s1:
                save_combo(h1, m1, combo)
                m1 += 1
            for combo in s2:
                save_combo(h2, m2, combo)
                m2 += 1
            for combo in s3:
                save_combo(h3, m3, combo)
                m3 += 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build 1_character_6_shot, 2_character_6_shot, 3_character_6_shot raw datasets.",
    )
    p.add_argument(
        "--input_root",
        type=str,
        default=r"H:\dataset\movie_shot_by_character",
        help="Input root: movie/scene/character_xx/shot",
    )
    p.add_argument(
        "--output_root",
        type=str,
        default=r"F:\dataset\movie_raw_dataset_variants_80frames",
        help="Output root with three 6-shot case folders.",
    )
    p.add_argument(
        "--max_1_character_6_shot",
        type=int,
        default=5,
        help="Max combos per scene for 1_character_6_shot.",
    )
    p.add_argument(
        "--max_2_character_6_shot",
        type=int,
        default=5,
        help="Max combos per scene for 2_character_6_shot.",
    )
    p.add_argument(
        "--max_3_character_6_shot",
        type=int,
        default=5,
        help="Max combos per scene for 3_character_6_shot.",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--jobs",
        type=int,
        default=_default_parallel_workers(),
        help=(
            "Parallel scene workers (separate processes; best for this CPU-heavy work). "
            "Use 1 for the original single-threaded RNG across scenes. "
            "Values >1 use a deterministic per-scene RNG (seed + scene path). "
            "Default is conservative (~half of CPUs, max 8) to limit load; raise for more speed if you have RAM/thermals headroom."
        ),
    )
    p.add_argument(
        "--max_candidate_pool",
        type=int,
        default=50_000,
        help=(
            "Stop collecting unique 6-shot candidates per scene after this many (each of the 1/2/3-char builders). "
            "Prevents huge lists and OOM on busy scenes; lower if memory is tight."
        ),
    )
    p.add_argument(
        "--min_frames",
        type=int,
        default=80,
        help="Only use shot videos with at least this many frames (OpenCV CAP_PROP_FRAME_COUNT).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("generate_raw_dataset_variants: starting ...", flush=True)
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    process_dataset(
        input_root=input_root,
        output_root=output_root,
        seed=args.seed,
        max_1c6=args.max_1_character_6_shot,
        max_2c6=args.max_2_character_6_shot,
        max_3c6=args.max_3_character_6_shot,
        jobs=args.jobs,
        max_candidate_pool=args.max_candidate_pool,
        min_frames=args.min_frames,
    )
    print("Done.")


if __name__ == "__main__":
    main()
