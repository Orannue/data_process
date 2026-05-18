import argparse
import concurrent.futures
import multiprocessing
import os
import random
from pathlib import Path
from typing import Dict, List

import cv2

from build_multishot_samples import (
    generate_candidates_for_scene,
    insert_empty_shot_by_probability,
    is_empty_shot,
    select_samples_for_scene,
    write_merged_video,
)
from character_cluster import (
    FACENET_IMPORT_ERROR,
    InceptionResnetV1,
    MTCNN,
    cluster_scene_detections,
    configure_torch_model_cache,
    extract_detections_from_shot,
    get_shot_duration_seconds,
    existing_scene_has_single_shot,
    remove_output_scene_dir,
    summarize_clusters,
    torch,
    write_empty_shot_videos,
)
from split_scene_shots import HAVE_SCENEDETECT, process_scene
from utils import append_jsonl, clean_name, read_json, read_json as load_json, reset_file, stable_id, write_json


WORKER = {
    "device": None,
    "mtcnn": None,
    "resnet": None,
}


def choose_worker_device(device_arg: str, devices_arg: str) -> str:
    if devices_arg:
        devices = [d.strip() for d in devices_arg.split(",") if d.strip()]
        if devices:
            identity = multiprocessing.current_process()._identity
            worker_number = identity[0] if identity else 1
            return devices[(worker_number - 1) % len(devices)]
    return device_arg or ("cuda" if torch.cuda.is_available() else "cpu")


def init_worker(
    device_arg: str,
    devices_arg: str,
    face_model: str,
    model_cache_dir: str,
) -> None:
    os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    if FACENET_IMPORT_ERROR is not None:
        raise RuntimeError(
            "The full scene pipeline requires torch, facenet_pytorch, and scikit-learn."
        ) from FACENET_IMPORT_ERROR

    configure_torch_model_cache(model_cache_dir)
    device_name = choose_worker_device(device_arg, devices_arg)
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    WORKER["device"] = device
    WORKER["mtcnn"] = MTCNN(keep_all=True, device=device)
    WORKER["resnet"] = InceptionResnetV1(pretrained=face_model).eval().to(device)
    print(f"[worker] pid={os.getpid()} device={device}", flush=True)


def build_scene_jobs(args: argparse.Namespace) -> List[Dict]:
    scene_json = Path(args.scene_json)
    moviebench_root = Path(args.moviebench_root)
    work_root = Path(args.work_root)
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
                    "work_root": str(work_root),
                    "args": args,
                }
            )
    return jobs


def load_existing_scene_rows(job: Dict, shots_root: Path) -> List[Dict]:
    scene_id = f"{job['scene_index']:04d}_{clean_name(job['scene_desc'])}"
    manifest_path = shots_root / job["movie_id"] / scene_id / "scene_manifest.json"
    if not manifest_path.exists():
        return []
    data = load_json(manifest_path)
    return data.get("shots", [])


def cluster_characters_for_scene(
    scene_rows: List[Dict],
    args: argparse.Namespace,
    chars_root: Path,
) -> Dict:
    if not scene_rows:
        return {"characters": [], "shots": [], "detections": []}

    movie_id = scene_rows[0]["movie_id"]
    scene_id = scene_rows[0]["scene_id"]
    out_scene_dir = chars_root / movie_id / scene_id
    scene_json = out_scene_dir / "characters.json"
    if scene_json.exists() and not args.overwrite:
        if existing_scene_has_single_shot(scene_json):
            remove_output_scene_dir(out_scene_dir, chars_root)
            return {"characters": [], "shots": [], "detections": []}
        data = load_json(scene_json)
        return {
            "characters": data.get("characters", []),
            "shots": data.get("shots", []),
            "detections": [],
        }

    valid_scene_rows = [
        row
        for row in sorted(scene_rows, key=lambda r: int(r.get("shot_index", 0)))
        if get_shot_duration_seconds(Path(row["path"])) >= args.min_shot_duration
    ]
    if len(valid_scene_rows) <= 1:
        remove_output_scene_dir(out_scene_dir, chars_root)
        return {"characters": [], "shots": [], "detections": []}

    detections: List[Dict] = []
    shot_stats: Dict[str, Dict] = {}
    for row in valid_scene_rows:
        dets, stats = extract_detections_from_shot(
            shot_row=row,
            mtcnn=WORKER["mtcnn"],
            resnet=WORKER["resnet"],
            device=WORKER["device"],
            sample_frames=args.sample_frames,
            min_face=args.min_face,
            max_center_distance=args.max_center_distance,
            min_core_face_area_ratio=args.min_core_face_area_ratio,
        )
        detections.extend(dets)
        shot_stats[row["shot_id"]] = stats

    clusters = cluster_scene_detections(
        detections,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        cluster_max_distance=args.cluster_max_distance,
    )
    characters, shot_to_occurrences = summarize_clusters(
        movie_id=movie_id,
        scene_id=scene_id,
        clusters=clusters,
        min_shots_per_character=args.min_shots_per_character,
        scene_shot_count=len(valid_scene_rows),
    )

    shot_records = []
    for row in valid_scene_rows:
        shot_id = row["shot_id"]
        occ = sorted(
            shot_to_occurrences.get(shot_id, []),
            key=lambda x: x["confidence"],
            reverse=True,
        )
        shot_records.append(
            {
                **row,
                "characters": occ,
                "dominant_character": occ[0]["character_id"] if occ else None,
                "character_count": len(occ),
                "is_empty_shot": int(shot_stats.get(shot_id, {}).get("face_detection_count", 0) or 0) == 0,
                "stats": shot_stats.get(shot_id, {}),
            }
        )

    empty_video_paths = write_empty_shot_videos(
        shot_records=shot_records,
        out_scene_dir=out_scene_dir,
        overwrite=args.overwrite,
    )
    for row in shot_records:
        copied_path = empty_video_paths.get(str(row["shot_id"]))
        if copied_path:
            row["empty_shot_video_path"] = copied_path

    write_json(
        scene_json,
        {
            "movie_id": movie_id,
            "scene_id": scene_id,
            "shot_count": len(valid_scene_rows),
            "detection_count": len(detections),
            "character_count": len(characters),
            "characters": characters,
            "shots": shot_records,
        },
    )
    return {"characters": characters, "shots": shot_records, "detections": detections}


def build_samples_for_scene(
    shot_records: List[Dict],
    args: argparse.Namespace,
    samples_root: Path,
) -> List[Dict]:
    if not shot_records:
        return []
    movie_id = shot_records[0]["movie_id"]
    scene_id = shot_records[0]["scene_id"]
    character_shots = [row for row in shot_records if not is_empty_shot(row)]
    empty_shots = [row for row in shot_records if is_empty_shot(row)]
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
    rng = random.Random(args.seed + int(stable_id(movie_id, scene_id, length=8), 16))
    selected = select_samples_for_scene(
        candidates=candidates,
        empty_shots=empty_shots,
        max_samples=args.max_samples_per_scene,
        random_pool_size=args.random_selection_pool_size,
        empty_shot_probability=args.empty_shot_probability,
        max_shots=args.max_shots,
        rng=rng,
    )
    rows = []
    for rank, candidate in enumerate(selected, start=1):
        sample_id = (
            f"{movie_id}_{scene_id}_{rank:04d}_"
            f"{stable_id(*candidate['shot_paths'], length=8)}"
        )
        out_video = None
        if args.write_videos:
            out_path = samples_root / "videos" / movie_id / scene_id / f"{sample_id}.mp4"
            if write_merged_video(candidate["shot_paths"], out_path):
                out_video = str(out_path)
        rows.append(
            {
                "sample_id": sample_id,
                "movie_id": movie_id,
                "scene_id": scene_id,
                **candidate,
                "merged_video_path": out_video,
            }
        )

    if rows:
        write_json(
            samples_root / "scenes" / movie_id / scene_id / "samples.json",
            {
                "movie_id": movie_id,
                "scene_id": scene_id,
                "candidate_count": len(candidates),
                "sample_count": len(rows),
                "samples": rows,
            },
        )
    return rows


def process_scene_pipeline_job(job: Dict) -> Dict:
    args = job["args"]
    work_root = Path(job["work_root"])
    shots_root = work_root / "shots"
    chars_root = work_root / "characters"
    samples_root = work_root / "samples"

    rows = process_scene(
        movie_id=job["movie_id"],
        scene_index=job["scene_index"],
        scene_desc=job["scene_desc"],
        clip_names=job["clip_names"],
        moviebench_root=Path(job["moviebench_root"]),
        output_root=shots_root,
        args=args,
    )
    if not rows:
        rows = load_existing_scene_rows(job, shots_root)

    character_result = cluster_characters_for_scene(rows, args, chars_root)
    sample_rows = build_samples_for_scene(character_result["shots"], args, samples_root)

    return {
        "job_index": job["job_index"],
        "movie_id": job["movie_id"],
        "scene_index": job["scene_index"],
        "shot_rows": rows,
        "character_rows": character_result["shots"],
        "characters": character_result["characters"],
        "sample_rows": sample_rows,
        "empty_rows": [
            row for row in character_result["shots"] if row.get("is_empty_shot")
        ],
        "error": None,
    }


def prepare_manifests(work_root: Path, overwrite: bool) -> Dict[str, Path]:
    paths = {
        "shots": work_root / "shots" / "_manifests" / "shots.jsonl",
        "character_shots": work_root / "characters" / "_manifests" / "character_shots.jsonl",
        "characters": work_root / "characters" / "_manifests" / "characters.jsonl",
        "empty_shots": work_root / "characters" / "_manifests" / "empty_shots.jsonl",
        "samples": work_root / "samples" / "samples.jsonl",
    }
    for path in paths.values():
        if overwrite or not path.exists():
            reset_file(path)
    return paths


def process(args: argparse.Namespace) -> None:
    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    manifests = prepare_manifests(work_root, args.overwrite)

    if not HAVE_SCENEDETECT:
        print("[warn] PySceneDetect is not installed; using OpenCV frame-diff fallback.")
    if args.devices:
        gpu_count = len([d for d in args.devices.split(",") if d.strip()])
        if args.scene_workers > gpu_count:
            print(
                "[warn] scene_workers is larger than the number of listed GPUs; "
                "multiple workers will share some GPUs."
            )
    elif (args.device or "").startswith("cuda") and args.scene_workers > 1:
        print(
            "[warn] Multiple workers are using the same CUDA device. "
            "Use --devices cuda:0,cuda:1,... to distribute work."
        )

    jobs = build_scene_jobs(args)
    if not jobs:
        print("[done] no scene jobs to process.")
        return

    worker_count = min(max(1, int(args.scene_workers)), len(jobs))
    print(f"[parallel-full] scene_workers={worker_count}, scenes={len(jobs)}")

    results: Dict[int, Dict] = {}
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=init_worker,
        initargs=(args.device, args.devices, args.face_model, args.model_cache_dir),
    ) as executor:
        future_to_job = {executor.submit(process_scene_pipeline_job, job): job for job in jobs}
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
                    "shot_rows": [],
                    "character_rows": [],
                    "characters": [],
                    "empty_rows": [],
                    "sample_rows": [],
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
                    f"[{done_count}/{len(jobs)}] {result['movie_id']} "
                    f"scene={result['scene_index']:04d} "
                    f"shots={len(result['shot_rows'])} "
                    f"char_shots={len(result['character_rows'])} "
                    f"samples={len(result['sample_rows'])}"
                )

    for job_index in sorted(results):
        result = results[job_index]
        if result["shot_rows"]:
            append_jsonl(manifests["shots"], result["shot_rows"])
        if result["character_rows"]:
            append_jsonl(manifests["character_shots"], result["character_rows"])
        if result["characters"]:
            append_jsonl(manifests["characters"], result["characters"])
        if result["empty_rows"]:
            append_jsonl(manifests["empty_shots"], result["empty_rows"])
        if result["sample_rows"]:
            append_jsonl(manifests["samples"], result["sample_rows"])

    failed = [r for r in results.values() if r["error"]]
    sample_count = sum(len(r["sample_rows"]) for r in results.values())
    write_json(
        work_root / "samples" / "summary.json",
        {
            "scene_count": len(jobs),
            "failed_scene_count": len(failed),
            "sample_count": sample_count,
            "write_videos": args.write_videos,
            "empty_shot_probability": args.empty_shot_probability,
            "seed": args.seed,
        },
    )
    print(f"[done] scenes={len(jobs)}, failed={len(failed)}, samples={sample_count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run split, character clustering, and sample building in parallel by scene."
    )
    parser.add_argument("--scene-json", default=r"F:\dataset\movie\movies_scenes.json")
    parser.add_argument("--moviebench-root", default=r"F:\dataset\movie\moviebench")
    parser.add_argument("--work-root", default=r"H:\dataset\movie_multishot_output")
    parser.add_argument("--only-movie", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--scene-workers", type=int, default=1)
    parser.add_argument("--write-videos", action="store_true")

    parser.add_argument("--clean-temp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clip-content-threshold", type=float, default=25.5)
    parser.add_argument("--scene-content-threshold", type=float, default=27.5)
    parser.add_argument("--adaptive-threshold", type=float, default=2.5)
    parser.add_argument("--min-detect-seconds", type=float, default=0.12)
    parser.add_argument("--min-shot-seconds", type=float, default=2.0)
    parser.add_argument("--merge-short-seconds", type=float, default=0.0)
    parser.add_argument("--trim-head-frames", type=int, default=3)
    parser.add_argument("--trim-tail-frames", type=int, default=3)
    parser.add_argument("--seam-support-seconds", type=float, default=0.3)
    parser.add_argument("--seam-diff-threshold", type=float, default=5.0)
    parser.add_argument("--refine-search-radius-frames", type=int, default=3)
    parser.add_argument("--refine-min-peak-gain", type=float, default=1.01)

    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--devices",
        default=None,
        help="Comma-separated worker device list, e.g. cuda:0,cuda:1,cuda:2,cuda:3.",
    )
    parser.add_argument("--face-model", default="vggface2")
    parser.add_argument(
        "--model-cache-dir",
        default=".",
        help="Torch/facenet model cache directory; default is the current directory.",
    )
    parser.add_argument("--sample-frames", type=int, default=8)
    parser.add_argument("--min-face", type=int, default=30)
    parser.add_argument("--min-shot-duration", type=float, default=2.0)
    parser.add_argument("--max-center-distance", type=float, default=0.6)
    parser.add_argument("--min-core-face-area-ratio", type=float, default=0.02)
    parser.add_argument("--dbscan-eps", type=float, default=0.40)
    parser.add_argument("--dbscan-min-samples", type=int, default=2)
    parser.add_argument("--cluster-max-distance", type=float, default=0.40)
    parser.add_argument(
        "--min-shots-per-character",
        type=int,
        default=2,
        help=(
            "Kept for compatibility. Single-shot characters are retained unless "
            "the whole scene has only one single-shot character."
        ),
    )

    parser.add_argument("--min-shots", type=int, default=3)
    parser.add_argument("--max-shots", type=int, default=6)
    parser.add_argument("--max-gap-shots", type=int, default=30)
    parser.add_argument("--min-character-confidence", type=float, default=0.35)
    parser.add_argument("--min-score", type=float, default=0.48)
    parser.add_argument("--max-samples-per-scene", type=int, default=5)
    parser.add_argument("--random-selection-pool-size",    type=int,    default=10,   help="Randomly select final samples from the top N ranked candidates per scene.")
    parser.add_argument("--empty-shot-probability", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--candidate-strategy",
        choices=["window", "combinations"],
        default="window",
    )
    parser.add_argument("--max-candidate-pool", type=int, default=2000)
    parser.add_argument("--max-candidate-combinations", type=int, default=50000)
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")
    process(parse_args())


if __name__ == "__main__":
    main()
