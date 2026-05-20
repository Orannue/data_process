from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import itertools

from build_multishot_samples import (
    candidate_mode_and_chars,
    is_empty_shot,
    score_candidate,
    shot_duration,
    shot_frame_count,
)


def max_blocks_for_shot_frame_count(frame_count: int, is_first: bool) -> int:
    if is_first:
        return max(0, (int(frame_count) - 1) // 12)
    return max(0, int(frame_count) // 12)


def fixed_latent_block_budget(latent_frames: int) -> int:
    latent_frames = int(latent_frames)
    if latent_frames < 4 or (latent_frames - 1) % 3 != 0:
        raise ValueError(
            "fixed latent sample requires latent_frames = 1 + 3 * N, "
            f"got {latent_frames}"
        )
    return (latent_frames - 1) // 3


def fixed_latent_frame_count(latent_frames: int) -> int:
    return 4 * int(latent_frames) - 3


def combo_supports_fixed_latent(
    shots: Sequence[Dict],
    latent_frames: int,
) -> Tuple[bool, List[int], int]:
    block_budget = fixed_latent_block_budget(latent_frames)
    max_blocks = [
        max_blocks_for_shot_frame_count(shot_frame_count(shot), idx == 0)
        for idx, shot in enumerate(shots)
    ]
    if any(block < 1 for block in max_blocks):
        return False, max_blocks, block_budget
    return sum(max_blocks) >= block_budget, max_blocks, block_budget


def generate_fixed_latent_candidates_for_scene(
    scene_shots: Sequence[Dict],
    min_shots: int,
    max_shots: int,
    latent_frames: int,
    max_gap_shots: int,
    min_conf: float,
    min_score: float,
    strategy: str = "window",
    max_candidates: Optional[int] = None,
    max_candidate_combinations: Optional[int] = None,
) -> List[Dict]:
    shots = [
        shot
        for shot in sorted(scene_shots, key=lambda s: int(s.get("shot_index", 0)))
        if not is_empty_shot(shot)
    ]
    frame_target = fixed_latent_frame_count(latent_frames)
    candidates = []
    seen_keys = set()
    checked = 0

    for length in range(int(min_shots), int(max_shots) + 1):
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

            indices = [int(s.get("shot_index", 0)) for s in combo]
            if any((b - a) <= 0 or (b - a) > max_gap_shots for a, b in zip(indices, indices[1:])):
                continue

            supports, max_blocks, block_budget = combo_supports_fixed_latent(
                combo,
                latent_frames=latent_frames,
            )
            if not supports:
                continue

            key = tuple(str(s.get("path", s.get("shot_id", ""))) for s in combo)
            if key in seen_keys:
                continue

            mode, chars = candidate_mode_and_chars(combo, min_conf)
            if mode is None:
                continue

            score = score_candidate(combo, mode, chars)
            if score < min_score:
                continue

            seen_keys.add(key)
            candidates.append(
                {
                    "mode": f"fixed_latent_{mode}",
                    "characters": chars,
                    "score": round(score, 6),
                    "shot_ids": [s["shot_id"] for s in combo],
                    "shot_indices": indices,
                    "shot_paths": [s["path"] for s in combo],
                    "shot_frame_counts": [shot_frame_count(s) for s in combo],
                    "shot_durations": [round(shot_duration(s), 3) for s in combo],
                    "fixed_latent_frames": int(latent_frames),
                    "fixed_frame_count": int(frame_target),
                    "fixed_block_budget": int(block_budget),
                    "fixed_max_blocks": [int(v) for v in max_blocks],
                    "latent_rule": {
                        "first_shot": "1+3*k",
                        "other_shots": "3*k",
                        "total_latent_frames": int(latent_frames),
                        "total_frames": int(frame_target),
                    },
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
                    "duration": round(sum(shot_duration(s) for s in combo), 3),
                }
            )

        if max_candidate_combinations is not None and checked >= max_candidate_combinations:
            break

    candidates.sort(key=lambda c: (-c["score"], c["shot_indices"]))
    if max_candidates is not None and max_candidates > 0:
        candidates = candidates[:max_candidates]
    return candidates

