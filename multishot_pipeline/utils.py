import hashlib
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
TEMP_VIDEO_PREFIXES = ("_tmp", "temp")


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_jsonl(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def reset_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def clean_name(text: str, max_len: int = 120) -> str:
    text = text.replace("Sence", "Scene")
    text = re.sub(r"^\s*Scene\s+(\d+)\s*:\s*", r"Scene_\1_", text, flags=re.I)
    text = re.sub(r"[^A-Za-z0-9._ -]+", "", text)
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"_+", "_", text)
    text = text.strip("._-")
    if not text:
        text = "scene"
    if len(text) > max_len:
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
        text = f"{text[: max_len - 9].rstrip('_')}_{digest}"
    return text


def stable_id(*parts: str, length: int = 12) -> str:
    raw = "::".join(str(p) for p in parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:length]


def timecode_to_seconds(value: str) -> float:
    """Parse HH.MM.SS.mmm, HH.MM.SSmmm, HH:MM:SS.mmm, or seconds."""
    value = value.strip()
    if not value:
        return 0.0
    if ":" in value:
        chunks = value.split(":")
        if len(chunks) == 3:
            return int(chunks[0]) * 3600 + int(chunks[1]) * 60 + float(chunks[2])
    dot_parts = value.split(".")
    if len(dot_parts) >= 4:
        h, m, s = dot_parts[:3]
        frac = "".join(dot_parts[3:])
        return int(h) * 3600 + int(m) * 60 + float(f"{int(s)}.{frac}")
    if len(dot_parts) == 3:
        h, m, sec = dot_parts
        return int(h) * 3600 + int(m) * 60 + float(sec)
    try:
        return float(value)
    except ValueError:
        return 0.0


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}.{m:02d}.{s:06.3f}"


def parse_clip_start_seconds(name: str) -> float:
    stem = Path(name).stem
    try:
        time_part = stem.rsplit("_", 1)[-1]
        start, _ = time_part.split("-", 1)
        return timecode_to_seconds(start)
    except Exception:
        return 0.0


def sort_clip_names(names: Sequence[str]) -> List[str]:
    return sorted(names, key=lambda n: (parse_clip_start_seconds(n), n))


def video_info(path: Path) -> Dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {
            "ok": False,
            "fps": 0.0,
            "frame_count": 0,
            "width": 0,
            "height": 0,
            "duration": 0.0,
        }
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {
        "ok": fps > 0 and frames > 0 and width > 0 and height > 0,
        "fps": fps,
        "frame_count": frames,
        "width": width,
        "height": height,
        "duration": frames / fps if fps > 0 else 0.0,
    }


def is_video_valid(path: Path, probe_frames: int = 10) -> bool:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return False
    valid = 0
    for _ in range(probe_frames):
        ok, _ = cap.read()
        if not ok:
            break
        valid += 1
    cap.release()
    return valid > 0


def load_video_frames(path: Path) -> Tuple[List[np.ndarray], Dict]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return [], video_info(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or (frames[0].shape[1] if frames else 0))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or (frames[0].shape[0] if frames else 0))
    cap.release()
    fps = fps if fps > 0 else 25.0
    return frames, {
        "ok": bool(frames),
        "fps": fps,
        "frame_count": len(frames),
        "width": width,
        "height": height,
        "duration": len(frames) / fps if fps > 0 else 0.0,
    }


def sample_indices(frame_count: int, sample_count: int) -> List[int]:
    if frame_count <= 0 or sample_count <= 0:
        return []
    if frame_count <= sample_count:
        return list(range(frame_count))
    return np.linspace(0, frame_count - 1, sample_count, dtype=int).tolist()


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12 or not math.isfinite(norm):
        return vec
    return vec / norm


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-12
    return float(np.dot(va, vb) / denom)


def mean_abs_frame_diff(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    small_a = cv2.resize(gray_a, (160, 90), interpolation=cv2.INTER_AREA)
    small_b = cv2.resize(gray_b, (160, 90), interpolation=cv2.INTER_AREA)
    return float(cv2.absdiff(small_a, small_b).mean())


def frame_quality_stats(frames: Sequence[np.ndarray]) -> Dict:
    if not frames:
        return {"brightness": 0.0, "blur": 0.0, "sample_count": 0}
    brightness_values = []
    blur_values = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_values.append(float(gray.mean()))
        blur_values.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    return {
        "brightness": float(np.mean(brightness_values)),
        "blur": float(np.mean(blur_values)),
        "sample_count": len(frames),
    }


def hsv_histogram(frames: Sequence[np.ndarray], bins: Tuple[int, int] = (24, 16)) -> List[float]:
    if not frames:
        return []
    hist_acc = None
    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, list(bins), [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        hist = hist.flatten().astype(np.float32)
        hist_acc = hist if hist_acc is None else hist_acc + hist
    hist_acc = hist_acc / max(1, len(frames))
    return l2_normalize(hist_acc).astype(float).tolist()


def read_sampled_frames(path: Path, sample_count: int) -> Tuple[List[Tuple[int, np.ndarray]], Dict]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return [], video_info(path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    wanted = set(sample_indices(frame_count, sample_count))
    frames: List[Tuple[int, np.ndarray]] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            frames.append((idx, frame))
        idx += 1
    cap.release()
    return frames, {
        "ok": bool(frames),
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": frame_count / fps if fps > 0 else 0.0,
    }


def iter_scene_dirs(root: Path) -> Iterator[Tuple[Path, Path]]:
    for movie_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for scene_dir in sorted(p for p in movie_dir.iterdir() if p.is_dir()):
            yield movie_dir, scene_dir


def list_videos(path: Path) -> List[Path]:
    return sorted(
        p
        for p in path.iterdir()
        if p.is_file()
        and p.suffix.lower() in VIDEO_EXTS
        and not p.name.lower().startswith(TEMP_VIDEO_PREFIXES)
    )
