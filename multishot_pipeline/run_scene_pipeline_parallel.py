import argparse
import concurrent.futures
import multiprocessing
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

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


PIPELINE_STAGE_ORDER = ("split", "character", "sample")


WORKER = {
    "device": None,
    "mtcnn": None,
    "resnet": None,
}


def init_cpu_worker() -> None:
    os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass


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
    init_cpu_worker()

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
            print(f"[skip] source movie missing: {movie_id}", flush=True)
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


def parse_stage_names(stages_arg: str) -> List[str]:
    if not stages_arg or stages_arg.strip().lower() == "all":
        return list(PIPELINE_STAGE_ORDER)
    names = [item.strip().lower() for item in stages_arg.split(",") if item.strip()]
    invalid = [name for name in names if name not in PIPELINE_STAGE_ORDER]
    if invalid:
        raise ValueError(
            f"Invalid --stages value(s): {invalid}. "
            f"Allowed stages: {', '.join(PIPELINE_STAGE_ORDER)}"
        )
    ordered = [stage for stage in PIPELINE_STAGE_ORDER if stage in set(names)]
    if not ordered:
        raise ValueError("--stages did not contain any runnable stage.")
    return ordered


def scene_id_for_job(job: Dict) -> str:
    return f"{job['scene_index']:04d}_{clean_name(job['scene_desc'])}"


def read_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return load_json(path)
    except Exception:
        return None


def stage_status_path(work_root: Path, stage: str, job: Dict) -> Path:
    return (
        work_root
        / "_stage_status"
        / stage
        / job["movie_id"]
        / f"{scene_id_for_job(job)}.json"
    )


def load_stage_status(work_root: Path, stage: str, job: Dict) -> Dict:
    data = read_json_if_exists(stage_status_path(work_root, stage, job))
    return data if isinstance(data, dict) else {}


def write_stage_status(
    work_root: Path,
    stage: str,
    job: Dict,
    status: str,
    **extra: Any,
) -> None:
    write_json(
        stage_status_path(work_root, stage, job),
        {
            "stage": stage,
            "status": status,
            "movie_id": job["movie_id"],
            "scene_index": job["scene_index"],
            "scene_id": scene_id_for_job(job),
            **extra,
        },
    )


def load_existing_scene_rows(job: Dict, shots_root: Path) -> List[Dict]:
    scene_id = scene_id_for_job(job)
    manifest_path = shots_root / job["movie_id"] / scene_id / "scene_manifest.json"
    if not manifest_path.exists():
        return []
    data = load_json(manifest_path)
    return data.get("shots", [])


def load_split_result_from_disk(job: Dict, work_root: Path, require: bool) -> Dict:
    rows = load_existing_scene_rows(job, work_root / "shots")
    status = load_stage_status(work_root, "split", job)
    error = None
    if require and not rows and status.get("status") != "done":
        error = "Missing split output. Run --stages split first."
    return {
        "job_index": job["job_index"],
        "movie_id": job["movie_id"],
        "scene_index": job["scene_index"],
        "shot_rows": rows,
        "error": error,
    }


def load_existing_character_result(job: Dict, chars_root: Path) -> Dict:
    scene_json = chars_root / job["movie_id"] / scene_id_for_job(job) / "characters.json"
    data = read_json_if_exists(scene_json)
    if not isinstance(data, dict):
        return {"characters": [], "shots": [], "empty_rows": []}
    shots = data.get("shots", [])
    return {
        "characters": data.get("characters", []),
        "shots": shots,
        "empty_rows": [row for row in shots if row.get("is_empty_shot")],
    }


def load_character_result_from_disk(job: Dict, work_root: Path, require: bool) -> Dict:
    data = load_existing_character_result(job, work_root / "characters")
    status = load_stage_status(work_root, "character", job)
    error = None
    if require and not data["shots"] and status.get("status") != "done":
        error = "Missing character output. Run --stages character first."
    return {
        "job_index": job["job_index"],
        "movie_id": job["movie_id"],
        "scene_index": job["scene_index"],
        "character_rows": data["shots"],
        "characters": data["characters"],
        "empty_rows": data["empty_rows"],
        "error": error,
    }


def load_existing_sample_rows(job: Dict, samples_root: Path) -> List[Dict]:
    samples_json = (
        samples_root
        / "scenes"
        / job["movie_id"]
        / scene_id_for_job(job)
        / "samples.json"
    )
    data = read_json_if_exists(samples_json)
    if not isinstance(data, dict):
        return []
    rows = data.get("samples", [])
    return rows if isinstance(rows, list) else []


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
    samples_json = samples_root / "scenes" / movie_id / scene_id / "samples.json"
    if samples_json.exists() and not args.overwrite:
        data = load_json(samples_json)
        rows = data.get("samples", [])
        return rows if isinstance(rows, list) else []

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
            samples_json,
            {
                "movie_id": movie_id,
                "scene_id": scene_id,
                "candidate_count": len(candidates),
                "sample_count": len(rows),
                "samples": rows,
            },
        )
    return rows


def process_split_scene_job(job: Dict) -> Dict:
    args = job["args"]
    work_root = Path(job["work_root"])
    shots_root = work_root / "shots"

    status = load_stage_status(work_root, "split", job)
    if status.get("status") == "done" and not args.overwrite:
        rows = load_existing_scene_rows(job, shots_root)
    else:
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

    write_stage_status(
        work_root,
        "split",
        job,
        "done",
        shot_count=len(rows),
        has_output=bool(rows),
    )

    return {
        "job_index": job["job_index"],
        "movie_id": job["movie_id"],
        "scene_index": job["scene_index"],
        "shot_rows": rows,
        "error": None,
    }


def process_character_scene_job(job: Dict, shot_rows: List[Dict]) -> Dict:
    args = job["args"]
    work_root = Path(job["work_root"])
    chars_root = work_root / "characters"
    status = load_stage_status(work_root, "character", job)
    if status.get("status") == "done" and not args.overwrite:
        character_result = load_existing_character_result(job, chars_root)
        character_result = {
            "characters": character_result["characters"],
            "shots": character_result["shots"],
        }
    else:
        character_result = cluster_characters_for_scene(shot_rows, args, chars_root)

    write_stage_status(
        work_root,
        "character",
        job,
        "done",
        character_shot_count=len(character_result["shots"]),
        character_count=len(character_result["characters"]),
        has_output=bool(character_result["shots"]),
    )

    return {
        "job_index": job["job_index"],
        "movie_id": job["movie_id"],
        "scene_index": job["scene_index"],
        "character_rows": character_result["shots"],
        "characters": character_result["characters"],
        "empty_rows": [
            row for row in character_result["shots"] if row.get("is_empty_shot")
        ],
        "error": None,
    }


def process_sample_scene_job(job: Dict, character_rows: List[Dict]) -> Dict:
    args = job["args"]
    work_root = Path(job["work_root"])
    samples_root = work_root / "samples"
    status = load_stage_status(work_root, "sample", job)
    if status.get("status") == "done" and not args.overwrite:
        sample_rows = load_existing_sample_rows(job, samples_root)
    else:
        sample_rows = build_samples_for_scene(character_rows, args, samples_root)

    write_stage_status(
        work_root,
        "sample",
        job,
        "done",
        sample_count=len(sample_rows),
        has_output=bool(sample_rows),
    )

    return {
        "job_index": job["job_index"],
        "movie_id": job["movie_id"],
        "scene_index": job["scene_index"],
        "sample_rows": sample_rows,
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
        # Per-scene JSON files are the resume source of truth; rebuild aggregates.
        reset_file(path)
    return paths


def resolve_worker_count(requested: int, job_count: int) -> int:
    return min(max(1, int(requested)), max(1, int(job_count)))


def stage_error_result(job: Dict, stage: str, exc: Exception) -> Dict:
    return {
        "job_index": job["job_index"],
        "movie_id": job["movie_id"],
        "scene_index": job["scene_index"],
        f"{stage}_error": repr(exc),
        "error": repr(exc),
    }


def process(args: argparse.Namespace) -> None:
    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    manifests = prepare_manifests(work_root, args.overwrite)

    if not HAVE_SCENEDETECT:
        print("[warn] PySceneDetect is not installed; using OpenCV frame-diff fallback.", flush=True)

    jobs = build_scene_jobs(args)
    if not jobs:
        print("[done] no scene jobs to process.", flush=True)
        return

    selected_stages = parse_stage_names(args.stages)
    selected_stage_set = set(selected_stages)
    split_workers = resolve_worker_count(args.split_workers or args.scene_workers, len(jobs))
    character_workers = resolve_worker_count(
        args.character_workers or args.scene_workers,
        len(jobs),
    )
    sample_workers = resolve_worker_count(args.sample_workers or args.scene_workers, len(jobs))

    if args.devices:
        gpu_count = len([d for d in args.devices.split(",") if d.strip()])
        if character_workers > gpu_count:
            print(
                "[warn] character_workers is larger than the number of listed GPUs; "
                "multiple workers will share some GPUs.",
                flush=True,
            )
    elif (args.device or "").startswith("cuda") and character_workers > 1:
        print(
            "[warn] Multiple character workers are using the same CUDA device. "
            "Use --devices cuda:0,cuda:1,... to distribute work.",
            flush=True,
        )

    print(
        "[stage-batched] "
        f"stages={','.join(selected_stages)}, scenes={len(jobs)}, "
        f"split_workers={split_workers}, "
        f"character_workers={character_workers}, sample_workers={sample_workers}",
        flush=True,
    )

    split_results: Dict[int, Dict] = {}
    if "split" in selected_stage_set:
        print("[stage 1/3] split shots", flush=True)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=split_workers,
            initializer=init_cpu_worker,
        ) as executor:
            future_to_job = {executor.submit(process_split_scene_job, job): job for job in jobs}
            for done_count, future in enumerate(
                concurrent.futures.as_completed(future_to_job), start=1
            ):
                job = future_to_job[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = stage_error_result(job, "split", exc)
                    result["shot_rows"] = []
                    write_stage_status(
                        work_root,
                        "split",
                        job,
                        "error",
                        error=repr(exc),
                    )
                split_results[result["job_index"]] = result
                if result["error"]:
                    print(
                        f"[split {done_count}/{len(jobs)}] [error] "
                        f"{result['movie_id']} scene={result['scene_index']:04d}: "
                        f"{result['error']}",
                        flush=True,
                    )
                else:
                    print(
                        f"[split {done_count}/{len(jobs)}] {result['movie_id']} "
                        f"scene={result['scene_index']:04d} "
                        f"shots={len(result['shot_rows'])}",
                        flush=True,
                    )
    else:
        require_split = "character" in selected_stage_set
        print(
            f"[stage 1/3] split skipped; loading existing split rows "
            f"require={int(require_split)}",
            flush=True,
        )
        split_results = {
            job["job_index"]: load_split_result_from_disk(job, work_root, require_split)
            for job in jobs
        }

    character_results: Dict[int, Dict] = {}
    if "character" in selected_stage_set:
        character_jobs = [
            job
            for job in jobs
            if not split_results.get(job["job_index"], {}).get("error")
            and split_results.get(job["job_index"], {}).get("shot_rows")
        ]
        print(
            f"[stage 2/3] character clustering scenes={len(character_jobs)}",
            flush=True,
        )
        for job in jobs:
            split_result = split_results.get(job["job_index"], {})
            if split_result.get("error"):
                continue
            if not split_result.get("shot_rows"):
                character_results[job["job_index"]] = {
                    "job_index": job["job_index"],
                    "movie_id": job["movie_id"],
                    "scene_index": job["scene_index"],
                    "character_rows": [],
                    "characters": [],
                    "empty_rows": [],
                    "error": None,
                }
                write_stage_status(
                    work_root,
                    "character",
                    job,
                    "done",
                    character_shot_count=0,
                    character_count=0,
                    has_output=False,
                    skipped_reason="no_split_rows",
                )

        if character_jobs:
            character_workers = resolve_worker_count(character_workers, len(character_jobs))
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=character_workers,
                initializer=init_worker,
                initargs=(args.device, args.devices, args.face_model, args.model_cache_dir),
            ) as executor:
                future_to_job = {
                    executor.submit(
                        process_character_scene_job,
                        job,
                        split_results[job["job_index"]]["shot_rows"],
                    ): job
                    for job in character_jobs
                }
                for done_count, future in enumerate(
                    concurrent.futures.as_completed(future_to_job), start=1
                ):
                    job = future_to_job[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = stage_error_result(job, "character", exc)
                        result["character_rows"] = []
                        result["characters"] = []
                        result["empty_rows"] = []
                        write_stage_status(
                            work_root,
                            "character",
                            job,
                            "error",
                            error=repr(exc),
                        )
                    character_results[result["job_index"]] = result
                    if result["error"]:
                        print(
                            f"[character {done_count}/{len(character_jobs)}] [error] "
                            f"{result['movie_id']} scene={result['scene_index']:04d}: "
                            f"{result['error']}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[character {done_count}/{len(character_jobs)}] "
                            f"{result['movie_id']} scene={result['scene_index']:04d} "
                            f"char_shots={len(result['character_rows'])} "
                            f"characters={len(result['characters'])} "
                            f"empty={len(result['empty_rows'])}",
                            flush=True,
                        )
    else:
        require_character = "sample" in selected_stage_set
        print(
            f"[stage 2/3] character skipped; loading existing character rows "
            f"require={int(require_character)}",
            flush=True,
        )
        character_results = {
            job["job_index"]: load_character_result_from_disk(
                job,
                work_root,
                require_character,
            )
            for job in jobs
        }

    sample_results: Dict[int, Dict] = {}
    if "sample" in selected_stage_set:
        sample_jobs = [
            job
            for job in jobs
            if not character_results.get(job["job_index"], {}).get("error")
            and character_results.get(job["job_index"], {}).get("character_rows")
        ]
        print(f"[stage 3/3] build samples scenes={len(sample_jobs)}", flush=True)
        for job in jobs:
            character_result = character_results.get(job["job_index"], {})
            if character_result.get("error"):
                continue
            if not character_result.get("character_rows"):
                sample_results[job["job_index"]] = {
                    "job_index": job["job_index"],
                    "movie_id": job["movie_id"],
                    "scene_index": job["scene_index"],
                    "sample_rows": [],
                    "error": None,
                }
                write_stage_status(
                    work_root,
                    "sample",
                    job,
                    "done",
                    sample_count=0,
                    has_output=False,
                    skipped_reason="no_character_rows",
                )

        if sample_jobs:
            sample_workers = resolve_worker_count(sample_workers, len(sample_jobs))
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=sample_workers,
                initializer=init_cpu_worker,
            ) as executor:
                future_to_job = {
                    executor.submit(
                        process_sample_scene_job,
                        job,
                        character_results[job["job_index"]]["character_rows"],
                    ): job
                    for job in sample_jobs
                }
                for done_count, future in enumerate(
                    concurrent.futures.as_completed(future_to_job), start=1
                ):
                    job = future_to_job[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = stage_error_result(job, "sample", exc)
                        result["sample_rows"] = []
                        write_stage_status(
                            work_root,
                            "sample",
                            job,
                            "error",
                            error=repr(exc),
                        )
                    sample_results[result["job_index"]] = result
                    if result["error"]:
                        print(
                            f"[sample {done_count}/{len(sample_jobs)}] [error] "
                            f"{result['movie_id']} scene={result['scene_index']:04d}: "
                            f"{result['error']}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[sample {done_count}/{len(sample_jobs)}] "
                            f"{result['movie_id']} scene={result['scene_index']:04d} "
                            f"samples={len(result['sample_rows'])}",
                            flush=True,
                        )
    else:
        print("[stage 3/3] sample skipped", flush=True)

    failed_job_indexes = {
        job["job_index"]
        for job in jobs
        if split_results.get(job["job_index"], {}).get("error")
        or character_results.get(job["job_index"], {}).get("error")
        or sample_results.get(job["job_index"], {}).get("error")
    }
    for job_index in sorted(split_results):
        result = split_results[job_index]
        if result.get("shot_rows"):
            append_jsonl(manifests["shots"], result["shot_rows"])
    for job_index in sorted(character_results):
        result = character_results[job_index]
        if result.get("character_rows"):
            append_jsonl(manifests["character_shots"], result["character_rows"])
        if result.get("characters"):
            append_jsonl(manifests["characters"], result["characters"])
        if result.get("empty_rows"):
            append_jsonl(manifests["empty_shots"], result["empty_rows"])
    for job_index in sorted(sample_results):
        result = sample_results[job_index]
        if result.get("sample_rows"):
            append_jsonl(manifests["samples"], result["sample_rows"])

    sample_count = sum(len(r.get("sample_rows", [])) for r in sample_results.values())
    stage_counts = {
        "split": {
            "scene_count": len(split_results),
            "failed_scene_count": sum(1 for row in split_results.values() if row.get("error")),
        },
        "character": {
            "scene_count": len(character_results),
            "failed_scene_count": sum(1 for row in character_results.values() if row.get("error")),
        },
        "sample": {
            "scene_count": len(sample_results),
            "failed_scene_count": sum(1 for row in sample_results.values() if row.get("error")),
        },
    }
    summary = {
        "scene_count": len(jobs),
        "selected_stages": selected_stages,
        "failed_scene_count": len(failed_job_indexes),
        "stage_counts": stage_counts,
        "split_worker_count": split_workers,
        "character_worker_count": character_workers,
        "sample_worker_count": sample_workers,
        "sample_count": sample_count,
        "write_videos": args.write_videos,
        "empty_shot_probability": args.empty_shot_probability,
        "seed": args.seed,
    }
    write_json(work_root / "_stage_pipeline_summary.json", summary)
    write_json(work_root / "samples" / "summary.json", summary)
    print(
        f"[done] scenes={len(jobs)}, failed={len(failed_job_indexes)}, "
        f"samples={sample_count}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run split, character clustering, and sample building in staged "
            "parallel batches."
        )
    )
    parser.add_argument("--scene-json", default=r"movies_scenes.json")
    parser.add_argument("--moviebench-root", default=r"moviedataset")
    parser.add_argument("--work-root", default=r"H:\dataset\movie_multishot_output")
    parser.add_argument("--only-movie", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument(
        "--stages",
        default="split,character,sample",
        help=(
            "Comma-separated stages to run: split,character,sample. "
            "Use one stage for a resumable partial run, e.g. --stages split."
        ),
    )
    parser.add_argument("--scene-workers", type=int, default=1)
    parser.add_argument(
        "--split-workers",
        type=int,
        default=None,
        help="Workers for the CPU shot-splitting stage. Default: --scene-workers.",
    )
    parser.add_argument(
        "--character-workers",
        type=int,
        default=None,
        help="Workers for the GPU character-clustering stage. Default: --scene-workers.",
    )
    parser.add_argument(
        "--sample-workers",
        type=int,
        default=None,
        help="Workers for the CPU sample-building stage. Default: --scene-workers.",
    )
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
        default=1,
        help=(
            "Deprecated compatibility option; it no longer filters characters. "
            "Single-shot characters are retained in multi-shot scenes."
        ),
    )

    parser.add_argument("--min-shots", type=int, default=2)
    parser.add_argument("--max-shots", type=int, default=6)
    parser.add_argument("--max-gap-shots", type=int, default=30)
    parser.add_argument("--min-character-confidence", type=float, default=0.1)
    parser.add_argument("--min-score", type=float, default=0.1)
    parser.add_argument("--max-samples-per-scene", type=int, default=8)
    parser.add_argument(
        "--random-selection-pool-size",
        type=int,
        default=12,
        help="Randomly select final samples from the top N ranked candidates per scene.",
    )
    parser.add_argument("--empty-shot-probability", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
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
