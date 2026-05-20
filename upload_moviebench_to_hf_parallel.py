import csv
import os
import shutil
import subprocess
import threading   
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi, get_token


REPO_ID = "Orannue/moviedataset"
REPO_TYPE = "dataset"

SOURCE_ROOT = Path(r"F:\dataset\movie\moviebench")
STAGE_ROOT = Path(r"F:\moviebench_hf_stage")
LOG_ROOT = Path(r"F:\moviebench_hf_upload_logs")
XET_CACHE_ROOT = Path(r"H:\hf_xet_cache")
SEVEN_ZIP = Path(r"D:\7-Zip\7z.exe")

REPO_ARCHIVE_DIR = "moviebench_7z"
ARCHIVE_SUFFIX = ".7z"
SPLIT_THRESHOLD_BYTES = int(
    os.environ.get("MOVIEBENCH_SPLIT_THRESHOLD_BYTES", str(10_000_000_000))
)
SPLIT_TARGET_BYTES = int(
    os.environ.get("MOVIEBENCH_SPLIT_TARGET_BYTES", str(9_500_000_000))
)

MAX_WORKERS = int(os.environ.get("MOVIEBENCH_WORKERS", "1"))
MAX_PACK_WORKERS = int(os.environ.get("MOVIEBENCH_PACK_WORKERS", "1"))
SEVEN_ZIP_THREADS = os.environ.get("MOVIEBENCH_7Z_THREADS", "on")
HF_UPLOAD_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").strip()

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)
os.environ["HF_XET_CACHE"] = str(XET_CACHE_ROOT)
os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)

uploaded_lock = threading.Lock()
print_lock = threading.Lock()
pack_semaphore = threading.Semaphore(MAX_PACK_WORKERS)


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with print_lock:
        print(f"[{stamp}] {message}", flush=True)


def get_user_env(name: str) -> str | None:
    if os.name != "nt":
        return None
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, name)
            return value
    except FileNotFoundError:
        return None
    except OSError:
        return None


def get_hf_token() -> str:
    token = get_user_env("HF_TOKEN") or os.environ.get("HF_TOKEN")

    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. In PowerShell, run: $env:HF_TOKEN = 'hf_xxx'"
        )
    return token.strip()


def load_uploaded(uploaded_log: Path) -> dict[str, int]:
    if not uploaded_log.exists():
        return {}

    names: dict[str, int] = {}
    with uploaded_log.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            name = row.get("name")
            if name:
                try:
                    size_bytes = int(row.get("size_bytes") or "0")
                except ValueError:
                    size_bytes = 0
                names[name] = size_bytes
    return names


def ensure_uploaded_log(uploaded_log: Path) -> None:
    if uploaded_log.exists():
        return

    uploaded_log.parent.mkdir(parents=True, exist_ok=True)
    with uploaded_log.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["name", "size_bytes", "source_dir", "uploaded_at"])


def append_uploaded(uploaded_log: Path, row: list[str]) -> None:
    with uploaded_lock:
        with uploaded_log.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(row)


def write_manifest(uploaded_log: Path, manifest_file: Path) -> None:
    shutil.copyfile(uploaded_log, manifest_file)


def remove_stale_paths(paths: list[Path]) -> None:
    existing_paths = [path for path in paths if path.exists()]
    if not existing_paths:
        return

    log(f"Removing stale staged files: {len(existing_paths)} files")
    for path in existing_paths:
        path.unlink(missing_ok=True)


def archive_part_name(movie_name: str, part_index: int) -> str:
    return f"{movie_name}_{part_index:02d}{ARCHIVE_SUFFIX}"


def iter_movie_files(directory: Path) -> list[Path]:
    return sorted(
        [path for path in directory.rglob("*") if path.is_file()],
        key=lambda path: str(path.relative_to(directory)).lower(),
    )


def plan_archive_parts(directory: Path) -> list[tuple[str, list[Path]]]:
    files = iter_movie_files(directory)
    total_size = sum(path.stat().st_size for path in files)

    if total_size <= SPLIT_THRESHOLD_BYTES:
        return [(f"{directory.name}{ARCHIVE_SUFFIX}", files)]

    chunks: list[list[Path]] = []
    current_chunk: list[Path] = []
    current_size = 0

    for path in files:
        file_size = path.stat().st_size
        if current_chunk and current_size + file_size > SPLIT_TARGET_BYTES:
            chunks.append(current_chunk)
            current_chunk = []
            current_size = 0

        current_chunk.append(path)
        current_size += file_size

    if current_chunk:
        chunks.append(current_chunk)

    return [
        (archive_part_name(directory.name, index), chunk)
        for index, chunk in enumerate(chunks, start=1)
    ]


def write_7z_listfile(listfile_path: Path, paths: list[Path]) -> None:
    with listfile_path.open("w", encoding="utf-8", newline="\n") as handle:
        for path in paths:
            handle.write(str(path.relative_to(SOURCE_ROOT)) + "\n")


def pack_movie_files(directory: Path, archive_path: Path, files: list[Path]) -> None:
    if archive_path.exists():
        if archive_path.stat().st_size == 0:
            log(f"Removing zero-byte staged archive: {archive_path}")
            archive_path.unlink(missing_ok=True)
        else:
            log(f"Using existing staged archive: {archive_path}")
            return

    if not files:
        raise RuntimeError(f"No files found to archive for {directory.name}")

    tmp_path = archive_path.with_name(
        f"{archive_path.name}.packing.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    tmp_path.unlink(missing_ok=True)
    seven_zip_log = LOG_ROOT / f"{directory.name}.7z.log"
    listfile_path = LOG_ROOT / f"{archive_path.name}.list"
    write_7z_listfile(listfile_path, files)

    with seven_zip_log.open("a", encoding="utf-8", errors="replace") as log_handle:
        log(f"Creating staged archive: {archive_path}")
        subprocess.run(
            [
                str(SEVEN_ZIP),
                "a",
                "-y",
                "-t7z",
                "-mx=0",
                f"-mmt={SEVEN_ZIP_THREADS}",
                "-scsUTF-8",
                "-bd",
                "-bb0",
                str(tmp_path),
                f"@{listfile_path}",
            ],
            cwd=str(SOURCE_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=True,
        )

    if not tmp_path.exists() or tmp_path.stat().st_size == 0:
        raise RuntimeError(f"7z did not create a non-empty archive for {directory.name}")

    tmp_path.replace(archive_path)


def upload_with_retry(
    local_path: Path,
    path_in_repo: str,
    commit_message: str,
    hf_token: str,
) -> None:
    last_error: BaseException | None = None
    api = HfApi(endpoint=HF_UPLOAD_ENDPOINT)

    for attempt in range(1, 6):
        try:
            log(f"UPLOAD attempt {attempt}: {path_in_repo}")
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                token=hf_token,
                commit_message=commit_message,
            )
            log(f"UPLOAD done: {path_in_repo}")
            return
        except BaseException as exc:
            last_error = exc
            text = str(exc)

            if "401 Client Error" in text or "Invalid username or password" in text:
                raise RuntimeError(
                    f"Upload authentication failed for {path_in_repo}. "
                    f"Use https://huggingface.co for writes and check HF_TOKEN."
                ) from exc

            if "403 Forbidden" in text or "create_pr=1" in text:
                raise RuntimeError(
                    f"Upload is forbidden for {path_in_repo}; check write access."
                ) from exc

            wait_seconds = min(300, 30 * attempt)
            log(f"UPLOAD failed attempt {attempt}: {exc!r}; retrying in {wait_seconds}s")
            time.sleep(wait_seconds)

    raise RuntimeError(f"Upload failed after retries: {path_in_repo}") from last_error


def process_movie(
    index: int,
    total: int,
    directory: Path,
    uploaded_log: Path,
    uploaded: dict[str, int],
    hf_token: str,
) -> str:
    archive_parts = plan_archive_parts(directory)
    archive_names = [name for name, _ in archive_parts]
    single_archive_name = f"{directory.name}{ARCHIVE_SUFFIX}"
    single_archive_path = STAGE_ROOT / single_archive_name

    with uploaded_lock:
        if single_archive_name in uploaded:
            log(f"SKIP [{index}/{total}] already uploaded: {single_archive_name}")
            return directory.name

        if all(name in uploaded for name in archive_names):
            log(f"SKIP [{index}/{total}] already uploaded: {', '.join(archive_names)}")
            return directory.name

    log(f"PACK [{index}/{total}] {directory.name}")
    if len(archive_parts) > 1:
        remove_stale_paths([single_archive_path])

    with pack_semaphore:
        for archive_name, files in archive_parts:
            with uploaded_lock:
                if archive_name in uploaded:
                    log(f"SKIP part already uploaded: {archive_name}")
                    continue

            archive_path = STAGE_ROOT / archive_name
            repo_path = f"{REPO_ARCHIVE_DIR}/{archive_name}"
            pack_movie_files(directory, archive_path, files)

            size_bytes = archive_path.stat().st_size
            if size_bytes > SPLIT_THRESHOLD_BYTES:
                log(f"WARNING {archive_name} is still over 10G ({size_bytes} bytes)")
            log(f"READY [{index}/{total}] {archive_name} ({size_bytes} bytes)")

            upload_with_retry(
                local_path=archive_path,
                path_in_repo=repo_path,
                commit_message=f"Upload {archive_name}",
                hf_token=hf_token,
            )

            uploaded_at = datetime.now().isoformat()
            append_uploaded(
                uploaded_log,
                [archive_name, str(size_bytes), str(directory), uploaded_at],
            )

            with uploaded_lock:
                uploaded[archive_name] = size_bytes

            log(f"CLEAN {archive_name}")
            archive_path.unlink(missing_ok=True)

    return directory.name


def main() -> int:
    hf_token = get_hf_token()

    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"Source folder not found: {SOURCE_ROOT}")
    if not SEVEN_ZIP.exists():
        raise FileNotFoundError(f"7-Zip executable not found: {SEVEN_ZIP}")

    STAGE_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    XET_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    uploaded_log = LOG_ROOT / "uploaded.tsv"
    manifest_file = LOG_ROOT / "MANIFEST.tsv"
    ensure_uploaded_log(uploaded_log)
    uploaded = load_uploaded(uploaded_log)

    log(f"HF upload endpoint: {HF_UPLOAD_ENDPOINT}")
    log(f"Max concurrent movie jobs: {MAX_WORKERS}")
    log(f"Max concurrent pack jobs: {MAX_PACK_WORKERS}")
    log(f"7z threads per job: {SEVEN_ZIP_THREADS}")
    log(f"Split threshold bytes: {SPLIT_THRESHOLD_BYTES}")
    log(f"Split target bytes: {SPLIT_TARGET_BYTES}")

    api = HfApi(endpoint=HF_UPLOAD_ENDPOINT)
    info = api.repo_info(REPO_ID, repo_type=REPO_TYPE, token=hf_token)
    log(f"Repo OK: {info.id}")

    for metadata_name in ["README.md", "DOWNLOAD_ISSUE.md"]:
        metadata_path = SOURCE_ROOT / metadata_name
        if metadata_path.exists():
            upload_with_retry(
                local_path=metadata_path,
                path_in_repo=metadata_name,
                commit_message=f"Upload {metadata_name}",
                hf_token=hf_token,
            )

    dirs = sorted(
        [path for path in SOURCE_ROOT.iterdir() if path.is_dir()],
        key=lambda p: p.name,
    )
    total = len(dirs)

    futures = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for index, directory in enumerate(dirs, start=1):
            futures.append(
                executor.submit(
                    process_movie,
                    index,
                    total,
                    directory,
                    uploaded_log,
                    uploaded,
                    hf_token,
                )
            )

        for future in as_completed(futures):
            archive_name = future.result()
            log(f"DONE movie: {archive_name}")

    log("Writing manifest file.")
    write_manifest(uploaded_log, manifest_file)
    upload_with_retry(
        local_path=manifest_file,
        path_in_repo="MANIFEST.tsv",
        commit_message="Upload manifest",
        hf_token=hf_token,
    )

    log(f"DONE. Uploaded one 7z archive per movie to {REPO_ID}/{REPO_ARCHIVE_DIR}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        log(f"FATAL: {exc!r}")
        raise
