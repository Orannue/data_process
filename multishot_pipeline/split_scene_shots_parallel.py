import argparse
import concurrent.futures
import os
from pathlib import Path
from typing import Dict, List

import cv2

from split_scene_shots import HAVE_SCENEDETECT, process_scene
from utils import append_jsonl, read_json, reset_file


def build_scene_jobs(args: argparse.Namespace) -> List[Dict]:
    scene_json = Path(args.scene_json)
    moviebench_root = Path(args.moviebench_root)
    output_root = Path(args.output_root)
    data = read_json(scene_json)
    only = set(args.only_movie or [])
    movie_items = list(data.items())
    if args.reverse:
        movie_items.reverse()

    jobs: List[Dict] = []
    for movie_rank, (movie_id, scenes) in enumerate(movie_items, start=1):
        if only and movie_id not in only:
            continue
        movie_dir = moviebench_root / movie_id
        if not movie_dir.exists():
            print(f"[skip] source movie missing: {movie_id}")
            continue
        for scene_index, (scene_desc, clip_names) in enumerate(scenes.items(), start=1):
            jobs.append(
                {
                    "job_index": len(jobs),
                    "movie_rank": movie_rank,
                    "movie_total": len(movie_items),
                    "movie_id": movie_id,
                    "scene_index": scene_index,
                    "scene_desc": scene_desc,
                    "clip_names": list(clip_names),
                    "moviebench_root": str(moviebench_root),
                    "output_root": str(output_root),
                    "args": args,
                }
            )
    return jobs


def process_scene_job(job: Dict) -> Dict:
    os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    rows = process_scene(
        movie_id=job["movie_id"],
        scene_index=job["scene_index"],
        scene_desc=job["scene_desc"],
        clip_names=job["clip_names"],
        moviebench_root=Path(job["moviebench_root"]),
        output_root=Path(job["output_root"]),
        args=job["args"],
    )
    return {
        "job_index": job["job_index"],
        "movie_id": job["movie_id"],
        "scene_index": job["scene_index"],
        "rows": rows,
        "error": None,
    }


def process_movies_parallel_by_scene(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "_manifests" / "shots.jsonl"
    if args.overwrite and manifest_path.exists():
        reset_file(manifest_path)
    elif not manifest_path.exists():
        reset_file(manifest_path)

    if not HAVE_SCENEDETECT:
        print("[warn] PySceneDetect is not installed; using OpenCV frame-diff fallback.")

    jobs = build_scene_jobs(args)
    if not jobs:
        print("[done] no scene jobs to process.")
        return

    worker_count = min(max(1, int(args.scene_workers)), len(jobs))
    print(f"[parallel] scene_workers={worker_count}, scenes={len(jobs)}")

    results: Dict[int, Dict] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_job = {executor.submit(process_scene_job, job): job for job in jobs}
        for done_count, future in enumerate(
            concurrent.futures.as_completed(future_to_job), start=1
        ):
            job = future_to_job[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "job_index": job["job_index"],
                    "movie_id": job["movie_id"],
                    "scene_index": job["scene_index"],
                    "rows": [],
                    "error": repr(exc),
                }

            results[result["job_index"]] = result
            if result["error"]:
                print(
                    f"[{done_count}/{len(jobs)}] [error] "
                    f"{result['movie_id']} scene={result['scene_index']:04d}: "
                    f"{result['error']}"
                )
            else:
                print(
                    f"[{done_count}/{len(jobs)}] "
                    f"{result['movie_id']} scene={result['scene_index']:04d} "
                    f"shots={len(result['rows'])}"
                )

    for job_index in sorted(results):
        rows = results[job_index]["rows"]
        if rows:
            append_jsonl(manifest_path, rows)

    errors = [result for result in results.values() if result["error"]]
    print(
        f"[done] scenes={len(jobs)}, failed={len(errors)}, "
        f"manifest={manifest_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split MovieBench scene clips into shot videos in parallel by scene."
    )
    parser.add_argument("--scene-json", default=r"movies_scenes.json")
    parser.add_argument("--moviebench-root", default=r"moviedataset")
    parser.add_argument("--output-root", default=r"shots")
    parser.add_argument("--only-movie", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument(
        "--scene-workers",
        type=int,
        default=4,
        help="Number of scene worker processes.",
    )
    parser.add_argument(
        "--clean-temp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove _tmp/temp videos from scene output directories.",
    )
    parser.add_argument("--clip-content-threshold", type=float, default=25.5)
    parser.add_argument("--scene-content-threshold", type=float, default=27.5)
    parser.add_argument("--adaptive-threshold", type=float, default=2.5)
    parser.add_argument("--min-detect-seconds", type=float, default=0.12)
    parser.add_argument(
        "--min-shot-seconds",
        type=float,
        default=1.0,
        help="Drop shots with trimmed duration <= this many seconds.",
    )
    parser.add_argument(
        "--merge-short-seconds",
        type=float,
        default=0.0,
        help="Optional pre-filter merge threshold. Default 0 means do not merge short shots.",
    )
    parser.add_argument(
        "--trim-head-frames",
        type=int,
        default=3,
        help="Frames removed from the beginning of every detected shot before saving.",
    )
    parser.add_argument(
        "--trim-tail-frames",
        type=int,
        default=3,
        help="Frames removed from the end of every detected shot before saving.",
    )
    parser.add_argument("--seam-support-seconds", type=float, default=0.3)
    parser.add_argument("--seam-diff-threshold", type=float, default=5.0)
    parser.add_argument("--refine-search-radius-frames", type=int, default=3)
    parser.add_argument("--refine-min-peak-gain", type=float, default=1.01)
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")
    process_movies_parallel_by_scene(parse_args())


if __name__ == "__main__":
    main()
