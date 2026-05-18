import argparse
import itertools
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from utils import append_jsonl, cosine_similarity, iter_jsonl, reset_file, stable_id, write_json


def load_character_shots(path: Path) -> List[Dict]:
    return list(iter_jsonl(path))


def char_set(shot: Dict, min_conf: float) -> set:
    return {
        c["character_id"]
        for c in shot.get("characters", [])
        if float(c.get("confidence", 0.0)) >= min_conf
    }


def mean_confidence_for_chars(shots: Sequence[Dict], chars: Sequence[str]) -> float:
    vals = []
    target = set(chars)
    for shot in shots:
        for c in shot.get("characters", []):
            if c.get("character_id") in target:
                vals.append(float(c.get("confidence", 0.0)))
    return float(np.mean(vals)) if vals else 0.0


def background_similarity(a: Dict, b: Dict) -> float:
    hist_a = a.get("stats", {}).get("background_hist", [])
    hist_b = b.get("stats", {}).get("background_hist", [])
    if not hist_a or not hist_b:
        return 0.0
    return max(0.0, min(1.0, cosine_similarity(hist_a, hist_b)))


def quality_score(shot: Dict) -> float:
    stats = shot.get("stats", {})
    quality = stats.get("quality", {})
    brightness = float(quality.get("brightness", 0.0))
    blur = float(quality.get("blur", 0.0))
    duration = float(stats.get("duration", shot.get("duration", 0.0)) or 0.0)
    bright_score = 1.0 - min(1.0, abs(brightness - 118.0) / 118.0)
    blur_score = min(1.0, blur / 120.0)
    duration_score = min(1.0, duration / 2.0)
    return float(0.35 * bright_score + 0.35 * blur_score + 0.30 * duration_score)


def temporal_score(shots: Sequence[Dict], max_gap_shots: int) -> float:
    indices = [int(s.get("shot_index", 0)) for s in shots]
    if len(indices) <= 1:
        return 1.0
    gaps = [b - a for a, b in zip(indices, indices[1:])]
    if any(g <= 0 or g > max_gap_shots for g in gaps):
        return 0.0
    return float(np.mean([1.0 / g for g in gaps]))


def score_candidate(
    shots: Sequence[Dict],
    mode: str,
    shared_chars: Sequence[str],
    min_conf: float,
    max_gap_shots: int,
) -> float:
    temporal = temporal_score(shots, max_gap_shots)
    if temporal <= 0:
        return 0.0
    adjacent_bg = [
        background_similarity(a, b) for a, b in zip(shots, shots[1:])
    ]
    bg = float(np.mean(adjacent_bg)) if adjacent_bg else 0.0
    identity = mean_confidence_for_chars(shots, shared_chars)
    quality = float(np.mean([quality_score(s) for s in shots]))
    length_bonus = min(1.0, len(shots) / 4.0)
    mode_bonus = 0.08 if mode == "dialogue_pair" else 0.04
    return (
        0.34 * identity
        + 0.26 * bg
        + 0.18 * temporal
        + 0.16 * quality
        + 0.06 * length_bonus
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
) -> List[Dict]:
    shots = sorted(scene_shots, key=lambda s: int(s.get("shot_index", 0)))
    candidates = []
    for length in range(min_shots, max_shots + 1):
        for combo in itertools.combinations(shots, length):
            indices = [int(s.get("shot_index", 0)) for s in combo]
            if any((b - a) <= 0 or (b - a) > max_gap_shots for a, b in zip(indices, indices[1:])):
                continue
            mode, chars = candidate_mode_and_chars(combo, min_conf)
            if mode is None:
                continue
            score = score_candidate(combo, mode, chars, min_conf, max_gap_shots)
            if score < min_score:
                continue
            candidates.append(
                {
                    "mode": mode,
                    "characters": chars,
                    "score": round(score, 6),
                    "shot_ids": [s["shot_id"] for s in combo],
                    "shot_indices": indices,
                    "shot_paths": [s["path"] for s in combo],
                    "duration": round(
                        sum(float(s.get("stats", {}).get("duration", s.get("duration", 0.0)) or 0.0) for s in combo),
                        3,
                    ),
                }
            )
    candidates.sort(key=lambda c: (-c["score"], c["shot_indices"]))
    return candidates


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

    written = 0
    scene_count = 0
    for (movie_id, scene_id), scene_shots in sorted(by_scene.items()):
        candidates = generate_candidates_for_scene(
            scene_shots=scene_shots,
            min_shots=args.min_shots,
            max_shots=args.max_shots,
            max_gap_shots=args.max_gap_shots,
            min_conf=args.min_character_confidence,
            min_score=args.min_score,
        )
        if not candidates:
            continue
        scene_count += 1
        selected = candidates[: args.max_samples_per_scene]
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
    parser.add_argument("--max-shots", type=int, default=5)
    parser.add_argument("--max-gap-shots", type=int, default=3)
    parser.add_argument("--min-character-confidence", type=float, default=0.35)
    parser.add_argument("--min-score", type=float, default=0.48)
    parser.add_argument("--max-samples-per-scene", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    process(parse_args())


if __name__ == "__main__":
    main()

