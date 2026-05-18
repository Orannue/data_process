from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


HERE = Path(__file__).resolve().parent
DONE_MOVIE_STATUSES = {"done", "done_no_samples"}
TRANSIENT_MOVIE_STATUSES = {"queued", "running"}
TRANSIENT_ARCHIVE_STATUSES = {"downloading", "extracting"}
PRINT_LOCK = threading.Lock()


class CommandFailed(RuntimeError):
    def __init__(self, step: str, returncode: int, log_path: Path):
        super().__init__(
            f"{step} failed with exit code {returncode}. See log: {log_path}"
        )
        self.step = step
        self.returncode = returncode
        self.log_path = log_path


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def log(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_name(value: str, max_len: int = 160) -> str:
    value = value.replace("\\", "/")
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = value.strip("._-") or "item"
    if len(value) <= max_len:
        return value
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()[:10]
    return f"{value[: max_len - 11].rstrip('._-')}_{digest}"


def quote_cmd(cmd: Sequence[object]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def split_extra_args(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return shlex.split(value, posix=os.name != "nt")


def matches_any(path_text: str, patterns: Sequence[str]) -> bool:
    normalized = path_text.replace("\\", "/")
    return any(fnmatch.fnmatch(normalized, pattern) for pattern in patterns)


def count_devices(devices: Optional[str]) -> int:
    if not devices:
        return 0
    return len([item.strip() for item in devices.split(",") if item.strip()])


def parse_device_groups(args: argparse.Namespace) -> List[Optional[str]]:
    if args.device_groups:
        groups = []
        for group in re.split(r"[;|]", args.device_groups):
            group = group.strip()
            if group:
                groups.append(group)
        return groups or [args.devices]

    if not args.devices:
        return [None]

    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if args.movie_workers <= 1 or len(devices) <= 1:
        return [",".join(devices)]

    group_count = min(args.movie_workers, len(devices))
    groups: List[str] = []
    start = 0
    for idx in range(group_count):
        group_size = len(devices) // group_count
        if idx < len(devices) % group_count:
            group_size += 1
        group_devices = devices[start : start + group_size]
        start += group_size
        groups.append(",".join(group_devices))
    return groups


def scene_workers_for_group(args: argparse.Namespace, device_group: Optional[str]) -> int:
    if args.scene_workers is not None:
        return max(1, int(args.scene_workers))
    device_count = count_devices(device_group)
    if device_count:
        return device_count
    return 1


class StateStore:
    def __init__(self, path: Path):
        self.path = path
        self.data: Dict[str, Any] = {
            "version": 1,
            "created_at": now_iso(),
            "archives": {},
            "movies": {},
        }
        if path.exists():
            loaded = read_json(path)
            if isinstance(loaded, dict):
                self.data.update(loaded)
                self.data.setdefault("archives", {})
                self.data.setdefault("movies", {})

    @property
    def archives(self) -> Dict[str, Dict[str, Any]]:
        return self.data.setdefault("archives", {})

    @property
    def movies(self) -> Dict[str, Dict[str, Any]]:
        return self.data.setdefault("movies", {})

    def reset_stale_running(self) -> None:
        changed = False
        for movie_id, row in self.movies.items():
            if row.get("status") in TRANSIENT_MOVIE_STATUSES:
                row["previous_status"] = row.get("status")
                row["status"] = "pending_after_interrupted_run"
                row["updated_at"] = now_iso()
                changed = True
        for archive_key, row in self.archives.items():
            if row.get("status") in TRANSIENT_ARCHIVE_STATUSES:
                row["previous_status"] = row.get("status")
                row["status"] = "pending_after_interrupted_run"
                row["updated_at"] = now_iso()
                changed = True
        if changed:
            self.save()

    def update_archive(self, archive_key: str, **fields: Any) -> None:
        row = self.archives.setdefault(archive_key, {})
        row.update(fields)
        row["updated_at"] = now_iso()

    def update_movie(self, movie_id: str, **fields: Any) -> None:
        row = self.movies.setdefault(movie_id, {})
        row.update(fields)
        row["updated_at"] = now_iso()

    def save(self) -> None:
        self.data["updated_at"] = now_iso()
        write_json(self.path, self.data)


def run_logged(
    cmd: Sequence[object],
    log_path: Path,
    step: str,
    stream: bool = False,
    cwd: Optional[Path] = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd_text = quote_cmd(cmd)
    with log_path.open("a", encoding="utf-8", errors="replace") as log_file:
        log_file.write(f"\n[{now_iso()}] $ {cmd_text}\n")
        log_file.flush()
        process = subprocess.Popen(
            [str(part) for part in cmd],
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            if stream:
                log(f"[{step}] {line.rstrip()}")
        returncode = process.wait()
        log_file.write(f"[{now_iso()}] exit={returncode}\n")
        log_file.flush()
    if returncode != 0:
        raise CommandFailed(step=step, returncode=returncode, log_path=log_path)


def load_expected_movies(scene_json: Path) -> List[str]:
    data = read_json(scene_json)
    if not isinstance(data, dict):
        raise ValueError(f"Scene json must be a dict keyed by movie id: {scene_json}")
    return sorted(str(key) for key in data.keys())


def discover_movies(extract_root: Path, expected_movies: Iterable[str]) -> Dict[str, Path]:
    expected = set(expected_movies)
    found: Dict[str, Path] = {}
    if not extract_root.is_dir():
        return found

    for root, dirnames, _filenames in os.walk(extract_root):
        root_path = Path(root)
        kept = []
        for dirname in dirnames:
            if dirname.startswith("_") or dirname == "__MACOSX":
                continue
            if dirname in expected:
                previous = found.get(dirname)
                if previous is None or len(root_path.parts) < len(previous.parts):
                    found[dirname] = root_path
                continue
            kept.append(dirname)
        dirnames[:] = kept
    return found


def local_archive_path(download_root: Path, archive_key: str) -> Path:
    return download_root / Path(archive_key)


def archive_marker_path(extract_root: Path, archive_key: str) -> Path:
    return extract_root / "_archive_markers" / f"{safe_name(archive_key)}.done.json"


def archive_destination(args: argparse.Namespace, archive_key: str) -> Path:
    if not args.extract_per_archive:
        return args.extract_root
    archive_stem = (
        archive_key[:-3] if archive_key.lower().endswith(".7z") else archive_key
    )
    return args.extract_root / safe_name(archive_stem)


def list_local_archives(download_root: Path, patterns: Sequence[str]) -> List[Tuple[str, Path]]:
    if not download_root.is_dir():
        return []
    rows = []
    for path in download_root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(download_root).as_posix()
        if matches_any(rel, patterns):
            rows.append((rel, path))
    return sorted(rows, key=lambda item: item[0])


def list_hf_archives(args: argparse.Namespace) -> List[str]:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: huggingface_hub. Install it with "
            "`pip install huggingface_hub` or `conda install -c conda-forge "
            "huggingface_hub`."
        ) from exc

    api = HfApi(token=args.hf_token)
    files = api.list_repo_files(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
    )
    archives = [
        filename
        for filename in files
        if matches_any(filename, args.archive_patterns)
    ]
    archives = sorted(archives)
    if args.max_archives is not None:
        archives = archives[: args.max_archives]
    return archives


def download_archive(args: argparse.Namespace, archive_key: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: huggingface_hub. Install it before downloading."
        ) from exc

    args.download_root.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=args.repo_id,
        filename=archive_key,
        repo_type="dataset",
        revision=args.revision,
        token=args.hf_token,
        local_dir=str(args.download_root),
        force_download=args.force_download,
    )
    return Path(path)


def extract_archive(args: argparse.Namespace, archive_key: str, archive_path: Path) -> Dict[str, Any]:
    marker = archive_marker_path(args.extract_root, archive_key)
    destination = archive_destination(args, archive_key)
    if marker.exists() and not args.overwrite_extract:
        return {
            "archive_key": archive_key,
            "archive_path": str(archive_path),
            "destination": str(destination),
            "status": "skipped_extracted",
            "marker": str(marker),
        }

    destination.mkdir(parents=True, exist_ok=True)
    log_path = args.work_root / "_logs" / "extract" / f"{safe_name(archive_key)}.log"
    cmd = [
        args.sevenzip,
        "x",
        str(archive_path),
        f"-o{destination}",
        "-y",
    ]
    run_logged(
        cmd,
        log_path=log_path,
        step=f"extract:{archive_key}",
        stream=args.stream_subprocess_output,
    )
    marker_data = {
        "archive_key": archive_key,
        "archive_path": str(archive_path),
        "destination": str(destination),
        "extracted_at": now_iso(),
        "archive_size": archive_path.stat().st_size if archive_path.exists() else None,
    }
    write_json(marker, marker_data)
    archive_deleted = False
    if args.delete_archive_after_extract and archive_path.exists():
        archive_path.unlink()
        archive_deleted = True
    return {
        **marker_data,
        "status": "extracted",
        "marker": str(marker),
        "archive_deleted": archive_deleted,
    }


def has_mp4(root: Path) -> bool:
    return root.is_dir() and any(root.rglob("*.mp4"))


def write_movie_marker(movie_work: Path, data: Dict[str, Any]) -> None:
    write_json(movie_work / "_movie_pipeline_status.json", data)


def build_pipeline_cmd(
    args: argparse.Namespace,
    movie_id: str,
    moviebench_root: Path,
    movie_work: Path,
    device_group: Optional[str],
) -> List[str]:
    workers = scene_workers_for_group(args, device_group)
    cmd = [
        args.python,
        str(args.pipeline_dir / "run_scene_pipeline_parallel.py"),
        "--scene-json",
        str(args.scene_json),
        "--moviebench-root",
        str(moviebench_root),
        "--work-root",
        str(movie_work),
        "--only-movie",
        movie_id,
        "--scene-workers",
        str(workers),
    ]
    if args.write_videos:
        cmd.append("--write-videos")
    if args.overwrite:
        cmd.append("--overwrite")
    if device_group:
        cmd.extend(["--devices", device_group])
    elif args.device:
        cmd.extend(["--device", args.device])
    if args.model_cache_dir:
        cmd.extend(["--model-cache-dir", str(args.model_cache_dir)])
    cmd.extend(split_extra_args(args.pipeline_extra_args))
    return cmd


def build_crop_cmd(args: argparse.Namespace, movie_work: Path) -> List[str]:
    cmd = [
        args.python,
        str(args.pipeline_dir / "crop_sample_shots.py"),
        "--samples-manifest",
        str(movie_work / "samples" / "samples.jsonl"),
        "--output-root",
        str(movie_work / "cropped_samples"),
        "--max-latent-frames",
        str(args.max_latent_frames),
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    cmd.extend(split_extra_args(args.crop_extra_args))
    return cmd


def build_merge_cmd(args: argparse.Namespace, movie_work: Path, final_movie_root: Path) -> List[str]:
    cmd = [
        args.python,
        str(args.pipeline_dir / "merge_crop_resize_samples.py"),
        "--input-root",
        str(movie_work / "cropped_samples"),
        "--output-root",
        str(final_movie_root),
        "--target-width",
        str(args.target_width),
        "--target-height",
        str(args.target_height),
        "--min-clips",
        str(args.min_clips),
        "-j",
        str(args.merge_jobs),
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    cmd.extend(split_extra_args(args.merge_extra_args))
    return cmd


def process_movie(
    args: argparse.Namespace,
    movie_id: str,
    moviebench_root: Path,
    device_group: Optional[str],
) -> Dict[str, Any]:
    movie_work = args.work_root / "movies" / movie_id
    final_movie_root = args.final_root / movie_id
    logs_dir = movie_work / "_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    started_at = now_iso()
    marker_base = {
        "movie_id": movie_id,
        "moviebench_root": str(moviebench_root),
        "movie_work": str(movie_work),
        "final_root": str(final_movie_root),
        "device_group": device_group,
        "scene_workers": scene_workers_for_group(args, device_group),
        "started_at": started_at,
    }
    write_movie_marker(movie_work, {**marker_base, "status": "running"})

    try:
        run_logged(
            build_pipeline_cmd(args, movie_id, moviebench_root, movie_work, device_group),
            log_path=logs_dir / "01_scene_character_samples.log",
            step=f"{movie_id}:scene_pipeline",
            stream=args.stream_subprocess_output,
        )

        samples_manifest = movie_work / "samples" / "samples.jsonl"
        if not samples_manifest.exists():
            raise FileNotFoundError(f"Missing samples manifest: {samples_manifest}")

        run_logged(
            build_crop_cmd(args, movie_work),
            log_path=logs_dir / "02_crop_sample_shots.log",
            step=f"{movie_id}:crop",
            stream=args.stream_subprocess_output,
        )

        cropped_root = movie_work / "cropped_samples"
        if not has_mp4(cropped_root):
            result = {
                **marker_base,
                "status": "done_no_samples",
                "finished_at": now_iso(),
                "reason": "No cropped sample mp4 files were produced.",
            }
            write_movie_marker(movie_work, result)
            return result

        run_logged(
            build_merge_cmd(args, movie_work, final_movie_root),
            log_path=logs_dir / "03_merge_crop_resize.log",
            step=f"{movie_id}:merge_resize",
            stream=args.stream_subprocess_output,
        )

        result = {
            **marker_base,
            "status": "done",
            "finished_at": now_iso(),
        }
        write_movie_marker(movie_work, result)
        return result

    except Exception as exc:
        result = {
            **marker_base,
            "status": "failed",
            "finished_at": now_iso(),
            "error": repr(exc),
        }
        write_movie_marker(movie_work, result)
        return result


def should_submit_movie(
    args: argparse.Namespace,
    state: StateStore,
    movie_id: str,
    submitted: set[str],
    submitted_count: int,
) -> bool:
    if movie_id in submitted:
        return False
    if args.only_movie and movie_id not in args.only_movie:
        return False
    if args.max_movies is not None and submitted_count >= args.max_movies:
        return False

    status = state.movies.get(movie_id, {}).get("status")
    if args.overwrite:
        return True
    if status in DONE_MOVIE_STATUSES:
        return False
    if status == "failed" and not args.retry_failed:
        return False
    return True


def main() -> int:
    args = parse_args()
    args.scene_json = args.scene_json.resolve()
    args.download_root = args.download_root.resolve()
    args.extract_root = args.extract_root.resolve()
    args.work_root = args.work_root.resolve()
    args.final_root = args.final_root.resolve()
    args.pipeline_dir = args.pipeline_dir.resolve()
    args.model_cache_dir = args.model_cache_dir.resolve() if args.model_cache_dir else None

    args.work_root.mkdir(parents=True, exist_ok=True)
    args.final_root.mkdir(parents=True, exist_ok=True)
    state = StateStore(args.state_file or (args.work_root / "_batch_pipeline_state.json"))
    state.reset_stale_running()

    expected_movies = load_expected_movies(args.scene_json)
    only = set(args.only_movie or [])
    if only:
        missing = sorted(only - set(expected_movies))
        if missing:
            log(f"[warn] --only-movie ids not present in scene json: {missing}")

    device_groups = parse_device_groups(args)
    if args.scene_workers is None:
        default_workers = scene_workers_for_group(args, device_groups[0])
        log(
            "[mode] scene-dominant mode: "
            f"movie_workers={args.movie_workers}, scene_workers={default_workers}, "
            f"device_groups={device_groups}"
        )
    else:
        log(
            "[mode] configured mode: "
            f"movie_workers={args.movie_workers}, scene_workers={args.scene_workers}, "
            f"device_groups={device_groups}"
        )

    submitted_movies: set[str] = set()
    submitted_movie_count = 0
    download_futures: Dict[Future, str] = {}
    extract_futures: Dict[Future, Tuple[str, Path]] = {}
    process_futures: Dict[Future, str] = {}
    scheduled_extracts: set[str] = set()
    available_movies_cache: Dict[str, Path] = {}
    last_discover_at = 0.0

    download_executor = ThreadPoolExecutor(max_workers=max(1, args.download_workers))
    extract_executor = ThreadPoolExecutor(max_workers=max(1, args.extract_workers))
    process_executor = ThreadPoolExecutor(max_workers=max(1, args.movie_workers))

    def submit_extract(archive_key: str, archive_path: Path) -> None:
        if args.skip_extract or archive_key in scheduled_extracts:
            return
        marker = archive_marker_path(args.extract_root, archive_key)
        if marker.exists() and not args.overwrite_extract:
            archive_deleted = False
            if args.delete_archive_after_extract and archive_path.exists():
                archive_path.unlink()
                archive_deleted = True
            state.update_archive(
                archive_key,
                status="extracted",
                archive_path=str(archive_path),
                marker=str(marker),
                archive_deleted=archive_deleted,
            )
            return
        scheduled_extracts.add(archive_key)
        state.update_archive(
            archive_key,
            status="extracting",
            archive_path=str(archive_path),
            destination=str(archive_destination(args, archive_key)),
        )
        state.save()
        future = extract_executor.submit(extract_archive, args, archive_key, archive_path)
        extract_futures[future] = (archive_key, archive_path)
        log(f"[extract queued] {archive_key}")

    def schedule_available_movies(force: bool = False) -> None:
        nonlocal submitted_movie_count
        nonlocal available_movies_cache, last_discover_at
        if args.skip_process:
            return
        current_time = time.monotonic()
        if force or current_time - last_discover_at >= args.discover_interval_seconds:
            available_movies_cache = discover_movies(args.extract_root, expected_movies)
            last_discover_at = current_time
        if not available_movies_cache:
            return
        for movie_id in sorted(available_movies_cache):
            if not should_submit_movie(
                args, state, movie_id, submitted_movies, submitted_movie_count
            ):
                continue
            group = device_groups[submitted_movie_count % len(device_groups)]
            parent_root = available_movies_cache[movie_id]
            submitted_movies.add(movie_id)
            submitted_movie_count += 1
            state.update_movie(
                movie_id,
                status="queued",
                moviebench_root=str(parent_root),
                device_group=group,
                queued_at=now_iso(),
            )
            state.save()
            future = process_executor.submit(process_movie, args, movie_id, parent_root, group)
            process_futures[future] = movie_id
            log(
                f"[movie queued] {movie_id} "
                f"root={parent_root} devices={group} "
                f"scene_workers={scene_workers_for_group(args, group)}"
            )

    try:
        schedule_available_movies(force=True)

        for archive_key, archive_path in list_local_archives(
            args.download_root, args.archive_patterns
        ):
            submit_extract(archive_key, archive_path)

        if not args.skip_download:
            archives = list_hf_archives(args)
            log(f"[download] found {len(archives)} archives from {args.repo_id}")
            for archive_key in archives:
                archive_state = state.archives.get(archive_key, {})
                marker = archive_marker_path(args.extract_root, archive_key)
                if marker.exists() and not args.overwrite_extract:
                    state.update_archive(
                        archive_key,
                        status="extracted",
                        marker=str(marker),
                    )
                    log(f"[download skipped: already extracted] {archive_key}")
                    continue
                local_path = local_archive_path(args.download_root, archive_key)
                state_archive_path = archive_state.get("archive_path")
                if state_archive_path:
                    candidate_path = Path(state_archive_path)
                    if candidate_path.exists() and candidate_path.stat().st_size > 0:
                        local_path = candidate_path
                if (
                    args.skip_existing_downloads
                    and local_path.exists()
                    and local_path.stat().st_size > 0
                    and not args.force_download
                ):
                    state.update_archive(
                        archive_key,
                        status="downloaded",
                        archive_path=str(local_path),
                    )
                    log(f"[download skipped: local exists] {archive_key}")
                    submit_extract(archive_key, local_path)
                    continue
                state.update_archive(archive_key, status="downloading")
                state.save()
                future = download_executor.submit(download_archive, args, archive_key)
                download_futures[future] = archive_key
                log(f"[download queued] {archive_key}")

        state.save()

        while download_futures or extract_futures or process_futures:
            futures = list(download_futures) + list(extract_futures) + list(process_futures)
            done, _pending = wait(
                futures,
                timeout=args.poll_seconds,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                schedule_available_movies()
                state.save()
                continue

            for future in done:
                if future in download_futures:
                    archive_key = download_futures.pop(future)
                    try:
                        archive_path = future.result()
                        state.update_archive(
                            archive_key,
                            status="downloaded",
                            archive_path=str(archive_path),
                            downloaded_at=now_iso(),
                        )
                        log(f"[download done] {archive_key}")
                        submit_extract(archive_key, archive_path)
                    except Exception as exc:
                        state.update_archive(
                            archive_key,
                            status="failed_download",
                            error=repr(exc),
                        )
                        log(f"[download failed] {archive_key}: {exc}")

                elif future in extract_futures:
                    archive_key, archive_path = extract_futures.pop(future)
                    try:
                        result = future.result()
                        state.update_archive(
                            archive_key,
                            status=result.get("status", "extracted"),
                            archive_path=str(archive_path),
                            destination=result.get("destination"),
                            marker=result.get("marker"),
                            archive_deleted=bool(result.get("archive_deleted")),
                            extracted_at=now_iso(),
                        )
                        suffix = " deleted_archive=1" if result.get("archive_deleted") else ""
                        log(f"[extract done] {archive_key}{suffix}")
                    except Exception as exc:
                        state.update_archive(
                            archive_key,
                            status="failed_extract",
                            archive_path=str(archive_path),
                            error=repr(exc),
                        )
                        log(f"[extract failed] {archive_key}: {exc}")
                    schedule_available_movies(force=True)

                elif future in process_futures:
                    movie_id = process_futures.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {
                            "movie_id": movie_id,
                            "status": "failed",
                            "finished_at": now_iso(),
                            "error": repr(exc),
                        }
                    state.update_movie(movie_id, **result)
                    status = result.get("status")
                    if status in DONE_MOVIE_STATUSES:
                        log(f"[movie done] {movie_id}: {status}")
                    else:
                        log(f"[movie failed] {movie_id}: {result.get('error')}")

            schedule_available_movies()
            state.save()

    except KeyboardInterrupt:
        log("[interrupt] stopping after current subprocesses finish...")
        raise
    finally:
        state.save()
        download_executor.shutdown(wait=True)
        extract_executor.shutdown(wait=True)
        process_executor.shutdown(wait=True)

    total_done = sum(
        1 for row in state.movies.values() if row.get("status") in DONE_MOVIE_STATUSES
    )
    total_failed = sum(1 for row in state.movies.values() if row.get("status") == "failed")
    log(
        f"[done] movies_done={total_done}, movies_failed={total_failed}, "
        f"state={state.path}"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download Hugging Face 7z archives, extract them, discover new "
            "movies, and run the multishot pipeline while downloads continue."
        )
    )

    parser.add_argument("--repo-id", default="Orannue/moviedataset")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument(
        "--archive-pattern",
        dest="archive_patterns",
        action="append",
        default=None,
        help="Archive glob inside the HF repo or download root. Repeatable.",
    )
    parser.add_argument("--max-archives", type=int, default=None)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--skip-existing-downloads",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Skip HF download when the local archive already exists. "
            "Use --force-download or --no-skip-existing-downloads to download again."
        ),
    )

    parser.add_argument("--scene-json", type=Path, default=Path("movies_scenes.json"))
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--extract-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--state-file", type=Path, default=None)
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=Path.cwd() / "model_cache",
        help="Torch/facenet model cache directory passed to character clustering.",
    )
    parser.add_argument("--pipeline-dir", type=Path, default=HERE)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--sevenzip", default="7z")

    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-process", action="store_true")
    parser.add_argument("--extract-per-archive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--delete-archive-after-extract",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete local .7z after successful extraction to save disk space.",
    )
    parser.add_argument("--overwrite-extract", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--only-movie", action="append", default=[])
    parser.add_argument("--max-movies", type=int, default=None)

    parser.add_argument("--download-workers", type=int, default=1)
    parser.add_argument("--extract-workers", type=int, default=1)
    parser.add_argument(
        "--movie-workers",
        type=int,
        default=1,
        help="Default 1: one movie uses all listed GPUs and parallelizes by scene.",
    )
    parser.add_argument(
        "--scene-workers",
        type=int,
        default=None,
        help="Scene workers per movie. Default: number of devices in that movie's group.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--devices",
        default=None,
        help="Comma-separated devices, e.g. cuda:0,cuda:1,cuda:2,cuda:3.",
    )
    parser.add_argument(
        "--device-groups",
        default=None,
        help=(
            "Optional movie worker device groups. Example: "
            "'cuda:0,cuda:1;cuda:2,cuda:3'."
        ),
    )

    parser.add_argument("--write-videos", action="store_true")
    parser.add_argument("--max-latent-frames", type=int, default=127)
    parser.add_argument("--target-width", type=int, default=832)
    parser.add_argument("--target-height", type=int, default=480)
    parser.add_argument("--min-clips", type=int, default=1)
    parser.add_argument("--merge-jobs", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--pipeline-extra-args", default=None)
    parser.add_argument("--crop-extra-args", default=None)
    parser.add_argument("--merge-extra-args", default=None)

    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument(
        "--discover-interval-seconds",
        type=float,
        default=60.0,
        help="Minimum interval for rescanning extracted movie folders.",
    )
    parser.add_argument("--stream-subprocess-output", action="store_true")

    args = parser.parse_args()
    if args.archive_patterns is None:
        args.archive_patterns = ["*.7z"]
    return args


if __name__ == "__main__":
    raise SystemExit(main())
