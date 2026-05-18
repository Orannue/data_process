import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def run(cmd):
    print(" ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full multishot pipeline.")
    parser.add_argument("--scene-json", default=r"F:\dataset\movie\movies_scenes.json")
    parser.add_argument("--moviebench-root", default=r"F:\dataset\movie\moviebench")
    parser.add_argument("--work-root", default=r"H:\dataset\movie_multishot_output")
    parser.add_argument("--only-movie", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-split", action="store_true")
    parser.add_argument("--skip-characters", action="store_true")
    parser.add_argument("--skip-samples", action="store_true")
    parser.add_argument("--write-videos", action="store_true")
    parser.add_argument("--write-character-videos", action="store_true")
    parser.add_argument(
        "--empty-shot-probability",
        type=float,
        default=0.3,
        help="Probability of inserting one empty shot into each selected sample.",
    )
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument(
        "--candidate-strategy",
        choices=["window", "combinations"],
        default="window",
        help="Use contiguous shot windows by default to avoid combination explosion.",
    )
    parser.add_argument("--max-candidate-pool", type=int, default=2000)
    parser.add_argument("--max-candidate-combinations", type=int, default=50000)
    parser.add_argument(
        "--random-selection-pool-size",
        type=int,
        default=10,
        help="Randomly select final samples from the top N ranked candidates per scene.",
    )
    parser.add_argument(
        "--scene-workers",
        type=int,
        default=1,
        help="Use parallel scene splitting when greater than 1.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    work_root = Path(args.work_root)
    shots_root = work_root / "shots"
    chars_root = work_root / "characters"
    samples_root = work_root / "samples"

    only_args = []
    for movie in args.only_movie:
        only_args.extend(["--only-movie", movie])
    overwrite = ["--overwrite"] if args.overwrite else []

    if not args.skip_split:
        split_script = (
            "split_scene_shots_parallel.py"
            if args.scene_workers and args.scene_workers > 1
            else "split_scene_shots.py"
        )
        scene_worker_args = (
            ["--scene-workers", str(args.scene_workers)]
            if args.scene_workers and args.scene_workers > 1
            else []
        )
        run(
            [
                sys.executable,
                str(HERE / split_script),
                "--scene-json",
                args.scene_json,
                "--moviebench-root",
                args.moviebench_root,
                "--output-root",
                str(shots_root),
                *scene_worker_args,
                *only_args,
                *overwrite,
            ]
        )

    if not args.skip_characters:
        character_video_args = (
            ["--write-character-videos"] if args.write_character_videos else []
        )
        run(
            [
                sys.executable,
                str(HERE / "character_cluster.py"),
                "--shots-root",
                str(shots_root),
                "--shot-manifest",
                str(shots_root / "_manifests" / "shots.jsonl"),
                "--output-root",
                str(chars_root),
                *character_video_args,
                *only_args,
                *overwrite,
            ]
        )

    if not args.skip_samples:
        write_videos = ["--write-videos"] if args.write_videos else []
        run(
            [
                sys.executable,
                str(HERE / "build_multishot_samples.py"),
                "--character-manifest",
                str(chars_root / "_manifests" / "character_shots.jsonl"),
                "--output-root",
                str(samples_root),
                "--empty-shot-probability",
                str(args.empty_shot_probability),
                "--seed",
                str(args.sample_seed),
                "--candidate-strategy",
                args.candidate_strategy,
                "--max-candidate-pool",
                str(args.max_candidate_pool),
                "--max-candidate-combinations",
                str(args.max_candidate_combinations),
                "--random-selection-pool-size",
                str(args.random_selection_pool_size),
                *write_videos,
                *overwrite,
            ]
        )


if __name__ == "__main__":
    main()
