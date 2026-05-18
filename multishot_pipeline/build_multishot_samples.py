import argparse
import itertools
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from utils import append_jsonl, iter_jsonl, reset_file, stable_id, write_json


def load_character_shots(path: Path) -> List[Dict]:
    return list(iter_jsonl(path))


def merge_character_lists(existing: Sequence[Dict], incoming: Sequence[Dict]) -> List[Dict]:
    by_id: Dict[str, Dict] = {}
    for item in list(existing) + list(incoming):
        char_id = item.get("character_id")
        if not char_id:
            continue
        old = by_id.get(char_id)
        if old is None or float(item.get("confidence", 0.0)) > float(
            old.get("confidence", 0.0)
        ):
            by_id[char_id] = dict(item)
    return sorted(
        by_id.values(),
        key=lambda item: float(item.get("confidence", 0.0)),
        reverse=True,
    )


def dedupe_scene_shots(scene_shots: Sequence[Dict]) -> List[Dict]:
    by_key: Dict[Tuple[str, str], Dict] = {}
    for shot in scene_shots:
        key = (str(shot.get("shot_id", "")), str(shot.get("path", "")))
        if key not in by_key:
            by_key[key] = dict(shot)
            continue
        merged = by_key[key]
        merged["characters"] = merge_character_lists(
            merged.get("characters", []), shot.get("characters", [])
        )
        merged["character_count"] = len(merged["characters"])
        merged["dominant_character"] = (
            merged["characters"][0]["character_id"] if merged["characters"] else None
        )
        merged["is_empty_shot"] = bool(
            merged.get("is_empty_shot", False) or shot.get("is_empty_shot", False)
        )
        if not merged.get("empty_shot_video_path") and shot.get("empty_shot_video_path"):
            merged["empty_shot_video_path"] = shot["empty_shot_video_path"]
        if not merged.get("stats") and shot.get("stats"):
            merged["stats"] = shot["stats"]
    return sorted(by_key.values(), key=lambda s: int(s.get("shot_index", 0)))


def char_set(shot: Dict, min_conf: float) -> set:
    return {
        c["character_id"]
        for c in shot.get("characters", [])
        if float(c.get("confidence", 0.0)) >= min_conf
    }


def is_empty_shot(shot: Dict) -> bool:
    if bool(shot.get("is_empty_shot", False)):
        return True
    stats = shot.get("stats", {})
    return (
        int(shot.get("character_count", 0) or 0) == 0
        and int(stats.get("face_detection_count", 1) or 0) == 0
    )


def mean_confidence_for_chars(shots: Sequence[Dict], chars: Sequence[str]) -> float:
    vals = []
    target = set(chars)
    for shot in shots:
        for c in shot.get("characters", []):
            if c.get("character_id") in target:
                vals.append(float(c.get("confidence", 0.0)))
    return float(np.mean(vals)) if vals else 0.0


def shot_frame_count(shot: Dict) -> int:
    stats = shot.get("stats", {})
    return int(stats.get("frame_count", shot.get("frame_count", 0)) or 0)


def shot_duration(shot: Dict) -> float:
    stats = shot.get("stats", {})
    return float(stats.get("duration", shot.get("duration", 0.0)) or 0.0)


def score_candidate(
    shots: Sequence[Dict],
    mode: str,
    shared_chars: Sequence[str],
) -> float:
    identity = mean_confidence_for_chars(shots, shared_chars)
    length_bonus = min(1.0, len(shots) / 6.0)
    mode_bonus = 0.08 if mode == "dialogue_pair" else 0.04
    return (
        0.70 * identity
        + 0.30 * length_bonus
        + mode_bonus
    )


def candidate_mode_and_chars(
    shots: Sequence[Dict], min_conf: float
) -> Tuple[Optional[str], List[str]]:
    sets = [char_set(shot, min_conf) for shot in shots]
    if any(not s for s in sets):
        return None, []
    shared = sorted(set.intersection(*sets))
    if shared:
        return "same_character", shared

    union = sorted(set.union(*sets))
    if len(union) == 2:
        coverage = {c: 0 for c in union}
        for s in sets:
            for c in s:
                if c in coverage:
                    coverage[c] += 1
        if min(coverage.values()) >= 1:
            return "dialogue_pair", union
    return None, []


def generate_candidates_for_scene(
    scene_shots: Sequence[Dict],
    min_shots: int,
    max_shots: int,
    max_gap_shots: int,
    min_conf: float,
    min_score: float,
    strategy: str = "window",
    max_candidates: Optional[int] = None,
    max_candidate_combinations: Optional[int] = None,
) -> List[Dict]:
    shots = sorted(scene_shots, key=lambda s: int(s.get("shot_index", 0)))
    candidates = []
    seen_keys = set()

    def maybe_add_candidate(combo: Sequence[Dict]) -> None:
        indices = [int(s.get("shot_index", 0)) for s in combo]
        if any((b - a) <= 0 or (b - a) > max_gap_shots for a, b in zip(indices, indices[1:])):
            return
        key = tuple(str(s.get("path", s.get("shot_id", ""))) for s in combo)
        if key in seen_keys:
            return
        mode, chars = candidate_mode_and_chars(combo, min_conf)
        if mode is None:
            return
        score = score_candidate(combo, mode, chars)
        if score < min_score:
            return
        seen_keys.add(key)
        candidates.append(
            {
                "mode": mode,
                "characters": chars,
                "score": round(score, 6),
                "shot_ids": [s["shot_id"] for s in combo],
                "shot_indices": indices,
                "shot_paths": [s["path"] for s in combo],
                "shot_frame_counts": [shot_frame_count(s) for s in combo],
                "shot_durations": [
                    round(shot_duration(s), 3)
                    for s in combo
                ],
                "shots": [
                    {
                        "shot_id": s["shot_id"],
                        "shot_index": int(s.get("shot_index", 0)),
                        "path": s["path"],
                        "frame_count": shot_frame_count(s),
                        "duration": round(shot_duration(s), 3),
                        "role": "character",
                    }
                    for s in combo
                ],
                "duration": round(
                    sum(shot_duration(s) for s in combo),
                    3,
                ),
            }
        )

    checked = 0
    for length in range(min_shots, max_shots + 1):
        if strategy == "combinations":
            iterator = itertools.combinations(shots, length)
        else:
            iterator = (
                shots[start : start + length]
                for start in range(0, max(0, len(shots) - length + 1))
            )
        for combo in iterator:
            if max_candidate_combinations is not None and checked >= max_candidate_combinations:
                break
            checked += 1
            maybe_add_candidate(combo)
        if max_candidate_combinations is not None and checked >= max_candidate_combinations:
            break
    candidates.sort(key=lambda c: (-c["score"], c["shot_indices"]))
    if max_candidates is not None and max_candidates > 0:
        candidates = candidates[:max_candidates]
    return candidates


def insert_empty_shot_by_probability(
    candidate: Dict,
    empty_shots: Sequence[Dict],
    probability: float,
    max_shots: int,
    rng: random.Random,
) -> Dict:
    result = {
        **candidate,
        "shot_ids": list(candidate["shot_ids"]),
        "shot_indices": list(candidate["shot_indices"]),
        "shot_paths": list(candidate["shot_paths"]),
        "shot_frame_counts": list(candidate.get("shot_frame_counts", [])),
        "shot_durations": list(candidate.get("shot_durations", [])),
        "shots": [dict(shot) for shot in candidate.get("shots", [])],
        "shot_roles": ["character"] * len(candidate["shot_paths"]),
        "contains_empty_shot": False,
    }
    probability = max(0.0, min(1.0, float(probability)))
    if (
        probability <= 0.0
        or not empty_shots
        or rng.random() >= probability
    ):
        return result

    empty = rng.choice(list(empty_shots))
    empty_duration = round(shot_duration(empty), 3)
    empty_frame_count = shot_frame_count(empty)
    empty_shot_info = {
        "shot_id": empty["shot_id"],
        "shot_index": int(empty.get("shot_index", 0)),
        "path": empty["path"],
        "frame_count": empty_frame_count,
        "duration": empty_duration,
        "role": "empty",
    }
    if len(result["shot_paths"]) < max_shots:
        position = rng.randint(0, len(result["shot_paths"]))
        result["shot_ids"].insert(position, empty["shot_id"])
        result["shot_indices"].insert(position, int(empty.get("shot_index", 0)))
        result["shot_paths"].insert(position, empty["path"])
        result["shot_frame_counts"].insert(position, empty_frame_count)
        result["shot_durations"].insert(position, empty_duration)
        result["shots"].insert(position, empty_shot_info)
        result["shot_roles"].insert(position, "empty")
        result["empty_shot_action"] = "insert"
    else:
        position = rng.randrange(len(result["shot_paths"]))
        result["replaced_shot_id"] = result["shot_ids"][position]
        result["replaced_shot_path"] = result["shot_paths"][position]
        result["shot_ids"][position] = empty["shot_id"]
        result["shot_indices"][position] = int(empty.get("shot_index", 0))
        result["shot_paths"][position] = empty["path"]
        if result["shot_frame_counts"]:
            result["shot_frame_counts"][position] = empty_frame_count
        if result["shot_durations"]:
            result["shot_durations"][position] = empty_duration
        if result["shots"]:
            result["shots"][position] = empty_shot_info
        result["shot_roles"][position] = "empty"
        result["empty_shot_action"] = "replace"
    result["contains_empty_shot"] = True
    result["empty_shot_id"] = empty["shot_id"]
    result["empty_shot_path"] = empty["path"]
    result["empty_shot_position"] = position
    if result["shot_durations"]:
        result["duration"] = round(sum(result["shot_durations"]), 3)
    return result


def select_samples_for_scene(
    candidates: Sequence[Dict],
    empty_shots: Sequence[Dict],
    max_samples: int,
    random_pool_size: int,
    empty_shot_probability: float,
    max_shots: int,
    rng: random.Random,
) -> List[Dict]:
    selected = []
    seen_keys = set()

    pool = list(candidates[: max(1, random_pool_size)])
    rng.shuffle(pool)

    for candidate in pool:
        candidate = insert_empty_shot_by_probability(
            candidate=candidate,
            empty_shots=empty_shots,
            probability=empty_shot_probability,
            max_shots=max_shots,
            rng=rng,
        )

        key = tuple(candidate["shot_paths"])
        if key in seen_keys:
            continue

        seen_keys.add(key)
        selected.append(candidate)

        if len(selected) >= max_samples:
            break

    return selected


def read_video_frames(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return [], 25.0, (0, 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, fps, (width, height)


def write_merged_video(paths: Sequence[str], out_path: Path) -> bool:
    all_frames = []
    target_size = None
    target_fps = None
    for path_str in paths:
        frames, fps, size = read_video_frames(Path(path_str))
        if not frames:
            return False
        if target_size is None:
            target_size = size
            target_fps = fps
        width, height = target_size
        for frame in frames:
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            all_frames.append(frame)
    if not all_frames or target_size is None:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(target_fps or 25.0),
        target_size,
    )
    for frame in all_frames:
        writer.write(frame)
    writer.release()
    return out_path.exists() and out_path.stat().st_size > 0


def process(args: argparse.Namespace) -> None:
    character_manifest = Path(args.character_manifest)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "samples.jsonl"
    summary_path = output_root / "summary.json"
    if args.overwrite or not manifest_path.exists():
        reset_file(manifest_path)

    rows = load_character_shots(character_manifest)
    by_scene: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for row in rows:
        by_scene[(row["movie_id"], row["scene_id"])].append(row)

    rng = random.Random(args.seed)
    written = 0
    scene_count = 0
    for (movie_id, scene_id), scene_shots in sorted(by_scene.items()):
        scene_shots = dedupe_scene_shots(scene_shots)
        if len(scene_shots) <= 1:
            print(f"[drop] {movie_id}/{scene_id}: only one final shot; skipped samples")
            continue
        character_shots = [row for row in scene_shots if not is_empty_shot(row)]
        empty_shots = [row for row in scene_shots if is_empty_shot(row)]
        candidates = generate_candidates_for_scene(
            scene_shots=character_shots,
            min_shots=args.min_shots,
            max_shots=args.max_shots,
            max_gap_shots=args.max_gap_shots,
            min_conf=args.min_character_confidence,
            min_score=args.min_score,
            strategy=args.candidate_strategy,
            max_candidates=args.max_candidate_pool,
            max_candidate_combinations=args.max_candidate_combinations,
        )
        if not candidates:
            continue
        scene_count += 1
        selected = select_samples_for_scene(
            candidates=candidates,
            empty_shots=empty_shots,
            max_samples=args.max_samples_per_scene,
            random_pool_size=args.random_selection_pool_size,
            empty_shot_probability=args.empty_shot_probability,
            max_shots=args.max_shots,
            rng=rng,
        )
        out_rows = []
        for rank, candidate in enumerate(selected, start=1):
            sample_id = f"{movie_id}_{scene_id}_{rank:04d}_{stable_id(*candidate['shot_paths'], length=8)}"
            out_video = None
            if args.write_videos:
                out_path = output_root / "videos" / movie_id / scene_id / f"{sample_id}.mp4"
                ok = write_merged_video(candidate["shot_paths"], out_path)
                if ok:
                    out_video = str(out_path)
            out_rows.append(
                {
                    "sample_id": sample_id,
                    "movie_id": movie_id,
                    "scene_id": scene_id,
                    **candidate,
                    "merged_video_path": out_video,
                }
            )
        append_jsonl(manifest_path, out_rows)
        written += len(out_rows)
        print(f"[scene] {movie_id}/{scene_id} candidates={len(candidates)} kept={len(out_rows)}")

    write_json(
        summary_path,
        {
            "scene_count_with_samples": scene_count,
            "sample_count": written,
            "character_manifest": str(character_manifest),
            "output_root": str(output_root),
            "write_videos": args.write_videos,
            "empty_shot_probability": args.empty_shot_probability,
            "seed": args.seed,
            "candidate_strategy": args.candidate_strategy,
            "max_candidate_pool": args.max_candidate_pool,
            "max_candidate_combinations": args.max_candidate_combinations,
            "random_selection_pool_size": args.random_selection_pool_size,
        },
    )
    print(f"[done] samples={written} scenes={scene_count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build multishot samples from scene-level character metadata."
    )
    parser.add_argument(
        "--character-manifest",
        default=r"H:\dataset\movie_multishot_output\characters\_manifests\character_shots.jsonl",
    )
    parser.add_argument("--output-root", default=r"H:\dataset\movie_multishot_output\samples")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--write-videos", action="store_true")
    parser.add_argument("--min-shots", type=int, default=2)
    parser.add_argument("--max-shots", type=int, default=6)
    parser.add_argument("--max-gap-shots", type=int, default=30)
    parser.add_argument("--min-character-confidence", type=float, default=0.1)
    parser.add_argument("--min-score", type=float, default=0.1)
    parser.add_argument("--max-samples-per-scene", type=int, default=5)
    parser.add_argument(
        "--random-selection-pool-size",
        type=int,
        default=10,
        help="Randomly select final samples from the top N ranked candidates per scene.",
    )
    parser.add_argument(
        "--candidate-strategy",
        choices=["window", "combinations"],
        default="window",
        help=(
            "window only scores contiguous shot windows; combinations scores arbitrary "
            "subsets and can be much slower."
        ),
    )
    parser.add_argument("--max-candidate-pool", type=int, default=2000)
    parser.add_argument(
        "--max-candidate-combinations",
        type=int,
        default=50000,
        help="Safety cap used by --candidate-strategy combinations.",
    )
    parser.add_argument(
        "--empty-shot-probability",
        type=float,
        default=0.3,
        help=(
            "Probability of inserting one zero-face empty shot from the same scene "
            "at a random position in each selected sample."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    process(parse_args())


if __name__ == "__main__":
    main()
