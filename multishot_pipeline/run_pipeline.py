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
                *write_videos,
                *overwrite,
            ]
        )


if __name__ == "__main__":
    main()
