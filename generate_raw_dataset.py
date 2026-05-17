import argparse
import random
import shutil
from itertools import combinations
from pathlib import Path
import re
from typing import Dict, List, Sequence, Tuple


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def list_subdirs(path: Path) -> List[Path]:
    return sorted([p for p in path.iterdir() if p.is_dir()])


def list_shots(character_dir: Path) -> List[Path]:
    return sorted(
        [p for p in character_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    )


def is_non_contiguous(idx_a: int, idx_b: int) -> bool:
    return abs(idx_a - idx_b) > 1


def extract_shot_index(shot: Path) -> int | None:
    matches = re.findall(r"\d+", shot.stem)
    if not matches:
        return None
    return int(matches[-1])


def shots_non_contiguous_by_name(shot1: Path, shot2: Path) -> bool:
    idx1 = extract_shot_index(shot1)
    idx2 = extract_shot_index(shot2)
    if idx1 is None or idx2 is None:
        # If we cannot parse indexes from names, keep candidate.
        return True
    return is_non_contiguous(idx1, idx2)


def sample_same_character_pairs(
    shots: Sequence[Path],
) -> List[Tuple[Path, Path]]:
    candidates: List[Tuple[Path, Path]] = []
    n = len(shots)
    for i, j in combinations(range(n), 2):
        # Need both checks:
        # 1) non-contiguous in this character folder order
        # 2) non-contiguous by shot numbering in names (if available)
        if is_non_contiguous(i, j) and shots_non_contiguous_by_name(shots[i], shots[j]):
            candidates.append((shots[i], shots[j]))
    return candidates


def sample_same_character_triples(
    shots: Sequence[Path],
) -> List[Tuple[Path, Path, Path]]:
    candidates: List[Tuple[Path, Path, Path]] = []
    n = len(shots)
    for i, j, k in combinations(range(n), 3):
        # Keep shot1 and shot2 non-contiguous as requested.
        if is_non_contiguous(i, j) and shots_non_contiguous_by_name(shots[i], shots[j]):
            candidates.append((shots[i], shots[j], shots[k]))
    return candidates


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dedup_unordered_pairs(
    combos: Sequence[Tuple[Path, Path]],
) -> List[Tuple[Path, Path]]:
    unique: List[Tuple[Path, Path]] = []
    seen = set()
    for a, b in combos:
        key = tuple(sorted([a.as_posix().lower(), b.as_posix().lower()]))
        if key in seen:
            continue
        seen.add(key)
        unique.append((a, b))
    return unique


def dedup_unordered_triples(
    combos: Sequence[Tuple[Path, Path, Path]],
) -> List[Tuple[Path, Path, Path]]:
    unique: List[Tuple[Path, Path, Path]] = []
    seen = set()
    for a, b, c in combos:
        key = tuple(
            sorted([a.as_posix().lower(), b.as_posix().lower(), c.as_posix().lower()])
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append((a, b, c))
    return unique


def save_combo(output_case_dir: Path, combo_idx: int, shots: Sequence[Path]) -> None:
    combo_dir = output_case_dir / str(combo_idx)
    ensure_dir(combo_dir)
    for i, shot in enumerate(shots, start=1):
        target_name = f"shot{i}_{shot.name}"
        shutil.copy2(shot, combo_dir / target_name)


def collect_characters(scene_dir: Path) -> Dict[str, List[Path]]:
    character_to_shots: Dict[str, List[Path]] = {}
    for character_dir in list_subdirs(scene_dir):
        shots = list_shots(character_dir)
        if shots:
            character_to_shots[character_dir.name] = shots
    return character_to_shots


def build_case1_same_character(
    character_to_shots: Dict[str, List[Path]],
    max_per_scene_pairs: int,
    max_per_scene_triples: int,
    rng: random.Random,
) -> Tuple[int, int, List[Tuple[Path, Path]], List[Tuple[Path, Path, Path]]]:
    all_pairs: List[Tuple[Path, Path]] = []
    all_triples: List[Tuple[Path, Path, Path]] = []

    for _, shots in sorted(character_to_shots.items()):
        all_pairs.extend(sample_same_character_pairs(shots))
        all_triples.extend(sample_same_character_triples(shots))

    all_pairs = dedup_unordered_pairs(all_pairs)
    all_triples = dedup_unordered_triples(all_triples)

    if len(all_pairs) > max_per_scene_pairs:
        all_pairs = rng.sample(all_pairs, max_per_scene_pairs)
    if len(all_triples) > max_per_scene_triples:
        all_triples = rng.sample(all_triples, max_per_scene_triples)

    return len(all_pairs), len(all_triples), all_pairs, all_triples


def build_case2_two_characters(
    character_to_shots: Dict[str, List[Path]],
    max_per_scene: int,
    rng: random.Random,
) -> List[Tuple[Path, Path, Path]]:

    valid_char1 = [name for name, shots in character_to_shots.items() if len(shots) >= 2]
    all_chars = list(character_to_shots.keys())
    candidates: List[Tuple[Path, Path, Path]] = []

    for char1 in valid_char1:
        char1_shots = character_to_shots[char1]
        for char2 in all_chars:
            if char1 == char2:
                continue
            char2_shots = character_to_shots[char2]

            for idx1, idx3 in combinations(range(len(char1_shots)), 2):
                for shot2 in char2_shots:
                    shot1 = char1_shots[idx1]
                    shot3 = char1_shots[idx3]
                    # Enforce shot1 and shot2 non-contiguous as requested.
                    if not shots_non_contiguous_by_name(shot1, shot2):
                        continue
                    candidates.append((shot1, shot2, shot3))

    candidates = dedup_unordered_triples(candidates)
    if len(candidates) > max_per_scene:
        candidates = rng.sample(candidates, max_per_scene)
    return candidates


def process_scene(
    scene_dir: Path,
    max_same_character_pairs_scene: int,
    max_same_character_triples_scene: int,
    max_two_character_scene: int,
    rng: random.Random,
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path, Path]], List[Tuple[Path, Path, Path]]]:
    character_to_shots = collect_characters(scene_dir)
    if not character_to_shots:
        return [], [], []

    pair_count, triple_count, pair_combos, triple_combos = build_case1_same_character(
        character_to_shots=character_to_shots,
        max_per_scene_pairs=max_same_character_pairs_scene,
        max_per_scene_triples=max_same_character_triples_scene,
        rng=rng,
    )
    case2_combos = build_case2_two_characters(
        character_to_shots=character_to_shots,
        max_per_scene=max_two_character_scene,
        rng=rng,
    )
    print(
        f"[{scene_dir.parent.name}/{scene_dir.name}] "
        f"1_character_2_shot={pair_count}, "
        f"1_character_3shot={triple_count}, "
        f"2_character_3shot={len(case2_combos)}"
    )
    return pair_combos, triple_combos, case2_combos


def process_dataset(
    input_root: Path,
    output_root: Path,
    max_same_character_pairs_scene: int,
    max_same_character_triples_scene: int,
    max_two_character_scene: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    case1_2_dir = output_root / "1_character_2_shot"
    case1_3_dir = output_root / "1_character_3shot"
    case2_3_dir = output_root / "2_character_3shot"

    for case_dir in (case1_2_dir, case1_3_dir, case2_3_dir):
        if case_dir.exists():
            shutil.rmtree(case_dir)
        ensure_dir(case_dir)

    next_idx_case1_2 = 1
    next_idx_case1_3 = 1
    next_idx_case2_3 = 1

    for movie_dir in list_subdirs(input_root):
        for scene_dir in list_subdirs(movie_dir):
            pairs, triples, two_chars = process_scene(
                scene_dir=scene_dir,
                max_same_character_pairs_scene=max_same_character_pairs_scene,
                max_same_character_triples_scene=max_same_character_triples_scene,
                max_two_character_scene=max_two_character_scene,
                rng=rng,
            )

            for combo in pairs:
                save_combo(case1_2_dir, next_idx_case1_2, combo)
                next_idx_case1_2 += 1
            for combo in triples:
                save_combo(case1_3_dir, next_idx_case1_3, combo)
                next_idx_case1_3 += 1
            for combo in two_chars:
                save_combo(case2_3_dir, next_idx_case2_3, combo)
                next_idx_case2_3 += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build raw dataset combos from grouped character shots."
    )
    parser.add_argument(
        "--input_root",
        type=str,
        default=r"H:\dataset\movie_shot_by_character",
        help="Input root: movie/scene/character_xx/shot",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=r"H:\dataset\movie_raw_dataset",
        help="Output root: {1_character_2_shot,1_character_3shot,2_character_3shot}/idx",
    )
    parser.add_argument(
        "--max_same_character_pairs_scene",
        type=int,
        default=5,
        help="For each scene, max number of 1_character_2_shot combos.",
    )
    parser.add_argument(
        "--max_same_character_triples_scene",
        type=int,
        default=5,
        help="For each scene, max number of 1_character_3shot combos.",
    )
    parser.add_argument(
        "--max_two_character_scene",
        type=int,
        default=5,
        help="For each scene, max number of (character1, character2, character1) 3-shot combos.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    process_dataset(
        input_root=input_root,
        output_root=output_root,
        max_same_character_pairs_scene=args.max_same_character_pairs_scene,
        max_same_character_triples_scene=args.max_same_character_triples_scene,
        max_two_character_scene=args.max_two_character_scene,
        seed=args.seed,
    )
    print("Done.")


if __name__ == "__main__":
    main()
