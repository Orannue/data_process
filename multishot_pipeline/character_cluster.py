import argparse
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from utils import (
    append_jsonl,
    frame_quality_stats,
    hsv_histogram,
    iter_jsonl,
    iter_scene_dirs,
    list_videos,
    read_sampled_frames,
    read_json,
    reset_file,
    write_json,
)


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec) + 1e-12
    return vec / norm


try:
    import torch
    from facenet_pytorch import InceptionResnetV1, MTCNN
    from sklearn.cluster import DBSCAN

    FACENET_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    torch = None
    MTCNN = None
    InceptionResnetV1 = None
    DBSCAN = None
    FACENET_IMPORT_ERROR = exc


def configure_torch_model_cache(model_cache_dir: Optional[str]) -> None:
    if not model_cache_dir:
        return
    cache_dir = Path(model_cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(cache_dir)
    if torch is not None and hasattr(torch, "hub"):
        torch.hub.set_dir(str(cache_dir))
    print(f"[info] torch model cache={cache_dir}")


def clip_box(box, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width - 1, x2)
    y2 = min(height - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def pick_core_face_box(
    boxes: Sequence[Sequence[float]],
    width: int,
    height: int,
    min_face: int,
    max_center_distance: float,
    min_core_face_area_ratio: float,
) -> Optional[Tuple[int, int, int, int]]:
    frame_area = float(width * height)
    max_center_dist_den = float(np.hypot(width / 2.0, height / 2.0)) + 1e-12
    frame_center_x = width / 2.0
    frame_center_y = height / 2.0

    best_box: Optional[Tuple[int, int, int, int]] = None
    best_score = float("-inf")

    for box in boxes:
        clipped = clip_box(box, width, height)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped

        face_w = x2 - x1
        face_h = y2 - y1
        if face_w < min_face or face_h < min_face:
            continue

        face_area_ratio = (face_w * face_h) / frame_area
        if face_area_ratio < min_core_face_area_ratio:
            continue

        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        center_dist_norm = (
            np.hypot(center_x - frame_center_x, center_y - frame_center_y)
            / max_center_dist_den
        )
        if center_dist_norm > max_center_distance:
            continue

        score = (2.0 * face_area_ratio) - center_dist_norm
        if score > best_score:
            best_score = score
            best_box = clipped

    if best_box is not None:
        return best_box

    fallback_best: Optional[Tuple[int, int, int, int]] = None
    fallback_best_area = 0
    for box in boxes:
        clipped = clip_box(box, width, height)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        face_w = x2 - x1
        face_h = y2 - y1
        if face_w < min_face or face_h < min_face:
            continue
        area = face_w * face_h
        if area > fallback_best_area:
            fallback_best_area = area
            fallback_best = clipped
    return fallback_best


def face_embedding(face_rgb: np.ndarray, resnet, device) -> np.ndarray:
    face = cv2.resize(face_rgb, (160, 160))
    tensor = torch.from_numpy(face).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - 0.5) / 0.5
    tensor = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        emb = resnet(tensor).cpu().numpy()[0]
    return l2_normalize(emb)


def get_shot_duration_seconds(shot_path: Path) -> float:
    cap = cv2.VideoCapture(str(shot_path))
    if not cap.isOpened():
        return 0.0
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if fps <= 0:
        return 0.0
    return frame_count / fps


def extract_detections_from_shot(
    shot_row: Dict,
    mtcnn,
    resnet,
    device,
    sample_frames: int,
    min_face: int,
    max_center_distance: float = 0.6,
    min_core_face_area_ratio: float = 0.02,
) -> Tuple[List[Dict], Dict]:
    path = Path(shot_row["path"])
    sampled, info = read_sampled_frames(path, sample_frames)
    frames_only = [frame for _, frame in sampled]
    quality = frame_quality_stats(frames_only)
    hist = hsv_histogram(frames_only)
    detections: List[Dict] = []

    for frame_idx, frame_bgr in sampled:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        boxes, probs = mtcnn.detect(frame_rgb)
        if boxes is None:
            continue
        height, width = frame_rgb.shape[:2]
        frame_area = float(width * height)
        core_face = pick_core_face_box(
            boxes=boxes,
            width=width,
            height=height,
            min_face=min_face,
            max_center_distance=max_center_distance,
            min_core_face_area_ratio=min_core_face_area_ratio,
        )
        if core_face is None:
            continue
        x1, y1, x2, y2 = core_face
        prob = 1.0
        if probs is not None:
            for box, box_prob in zip(boxes, probs):
                if clip_box(box, width, height) == core_face:
                    prob = float(box_prob)
                    break
        face_w = x2 - x1
        face_h = y2 - y1
        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        emb = face_embedding(crop, resnet, device)
        center_x = ((x1 + x2) / 2.0) / max(1.0, width)
        center_y = ((y1 + y2) / 2.0) / max(1.0, height)
        detections.append(
            {
                "movie_id": shot_row["movie_id"],
                "scene_id": shot_row["scene_id"],
                "shot_id": shot_row["shot_id"],
                "shot_index": shot_row.get("shot_index", 0),
                "frame_index": frame_idx,
                "box": [x1, y1, x2, y2],
                "prob": float(prob),
                "area_ratio": float((face_w * face_h) / frame_area),
                "center": [float(center_x), float(center_y)],
                "embedding": emb.astype(float).tolist(),
            }
        )

    shot_stats = {
        "path": str(path),
        "fps": info.get("fps", 0.0),
        "frame_count": info.get("frame_count", 0),
        "duration": info.get("duration", shot_row.get("duration", 0.0)),
        "quality": quality,
        "background_hist": hist,
        "sampled_frames": len(sampled),
        "face_detection_count": len(detections),
    }
    return detections, shot_stats


def cluster_scene_detections(
    detections: List[Dict],
    dbscan_eps: float,
    dbscan_min_samples: int,
    cluster_max_distance: Optional[float] = None,
) -> Dict[int, List[Dict]]:
    if not detections:
        return {}

    by_shot: Dict[str, List[Dict]] = defaultdict(list)
    for det in detections:
        by_shot[det["shot_id"]].append(det)

    shot_reps = []
    shot_ids = []
    for shot_id, shot_dets in sorted(
        by_shot.items(), key=lambda item: min(d.get("shot_index", 0) for d in item[1])
    ):
        embeddings = np.asarray([d["embedding"] for d in shot_dets], dtype=np.float32)
        shot_reps.append(l2_normalize(np.mean(embeddings, axis=0)))
        shot_ids.append(shot_id)

    if not shot_reps:
        return {}

    X = np.stack(shot_reps, axis=0).astype(np.float32)
    labels = DBSCAN(
        eps=dbscan_eps, min_samples=dbscan_min_samples, metric="cosine"
    ).fit_predict(X)

    shot_rep_by_id = dict(zip(shot_ids, shot_reps))
    shot_groups: Dict[int, List[str]] = defaultdict(list)
    noise_shot_ids = []
    for label, shot_id in zip(labels, shot_ids):
        if int(label) == -1:
            noise_shot_ids.append(shot_id)
            continue
        shot_groups[int(label)].append(shot_id)

    max_distance = dbscan_eps if cluster_max_distance is None else float(cluster_max_distance)
    refined_groups: List[List[str]] = []
    for label in sorted(shot_groups):
        for shot_id in shot_groups[label]:
            rep = shot_rep_by_id[shot_id]
            best_group_idx = None
            best_group_distance = float("inf")
            for group_idx, group in enumerate(refined_groups):
                distances = [
                    1.0 - float(np.dot(rep, shot_rep_by_id[other_id]))
                    for other_id in group
                ]
                group_distance = max(distances) if distances else 0.0
                if group_distance <= max_distance and group_distance < best_group_distance:
                    best_group_idx = group_idx
                    best_group_distance = group_distance
            if best_group_idx is None:
                refined_groups.append([shot_id])
            else:
                refined_groups[best_group_idx].append(shot_id)
    for shot_id in noise_shot_ids:
        refined_groups.append([shot_id])

    clusters: Dict[int, List[Dict]] = defaultdict(list)
    for cluster_id, group in enumerate(refined_groups):
        for shot_id in group:
            clusters[cluster_id].extend(by_shot[shot_id])
    return dict(clusters)


def summarize_clusters(
    movie_id: str,
    scene_id: str,
    clusters: Dict[int, List[Dict]],
    min_shots_per_character: int,
    scene_shot_count: int = 0,
) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    character_items = []

    for cluster_id in sorted(clusters.keys()):
        dets = clusters[cluster_id]
        shot_ids = sorted({d["shot_id"] for d in dets})
        by_shot: Dict[str, List[Dict]] = defaultdict(list)
        for det in dets:
            by_shot[det["shot_id"]].append(det)

        character_items.append((cluster_id, dets, by_shot))

    if scene_shot_count <= 1 and len(character_items) == 1 and len(character_items[0][2]) == 1:
        return [], defaultdict(list)

    characters = []
    shot_to_occurrences: Dict[str, List[Dict]] = defaultdict(list)
    for local_idx, (cluster_id, dets, by_shot) in enumerate(character_items, start=1):
        char_id = f"char_{local_idx:03d}"
        for shot_id, shot_dets in by_shot.items():
            mean_prob = float(np.mean([d["prob"] for d in shot_dets]))
            mean_area = float(np.mean([d["area_ratio"] for d in shot_dets]))
            confidence = min(1.0, 0.60 * mean_prob + 8.0 * mean_area)
            shot_to_occurrences[shot_id].append(
                {
                    "character_id": char_id,
                    "confidence": confidence,
                    "detections": len(shot_dets),
                    "mean_area_ratio": mean_area,
                }
            )

        characters.append(
            {
                "movie_id": movie_id,
                "scene_id": scene_id,
                "character_id": char_id,
                "cluster_id": cluster_id,
                "detection_count": len(dets),
                "shot_count": len(by_shot),
            }
        )

    return characters, shot_to_occurrences


def compact_shot_record_for_scene_json(row: Dict) -> Dict:
    stats = row.get("stats", {})
    return {
        "shot_id": row.get("shot_id"),
        "shot_index": row.get("shot_index", 0),
        "path": row.get("path"),
        "is_empty_shot": bool(row.get("is_empty_shot", False)),
        "empty_shot_video_path": row.get("empty_shot_video_path"),
        "duration": stats.get("duration", row.get("duration", 0.0)),
        "characters": row.get("characters", []),
        "dominant_character": row.get("dominant_character"),
        "character_count": row.get("character_count", 0),
    }


def compact_scene_json(
    movie_id: str,
    scene_id: str,
    scene_rows: Sequence[Dict],
    detections: Sequence[Dict],
    characters: Sequence[Dict],
    shot_records: Sequence[Dict],
) -> Dict:
    return {
        "movie_id": movie_id,
        "scene_id": scene_id,
        "shot_count": len(scene_rows),
        "detection_count": len(detections),
        "character_count": len(characters),
        "characters": list(characters),
        "shots": [compact_shot_record_for_scene_json(row) for row in shot_records],
    }


def write_character_videos(
    scene_rows: Sequence[Dict],
    clusters: Dict[int, List[Dict]],
    characters: Sequence[Dict],
    out_scene_dir: Path,
    overwrite: bool,
) -> Dict[str, Dict]:
    shot_path_by_id = {row["shot_id"]: Path(row["path"]) for row in scene_rows}
    video_dir = out_scene_dir / "character_videos"
    if overwrite and video_dir.exists():
        shutil.rmtree(video_dir)
    results = {}
    for character in characters:
        cluster_id = int(character["cluster_id"])
        detections = clusters.get(cluster_id, [])
        shot_ids = sorted(
            {det["shot_id"] for det in detections},
            key=lambda sid: min(
                det.get("shot_index", 0) for det in detections if det["shot_id"] == sid
            ),
        )
        char_dir = video_dir / character["character_id"]
        copied = 0
        for shot_id in shot_ids:
            source = shot_path_by_id.get(shot_id)
            if source is None or not source.exists():
                continue
            char_dir.mkdir(parents=True, exist_ok=True)
            target = char_dir / source.name
            if overwrite or not target.exists():
                shutil.copy2(source, target)
            if target.exists() and target.stat().st_size > 0:
                copied += 1
        if copied:
            results[character["character_id"]] = {
                "video_dir": str(char_dir),
                "video_count": copied,
            }
    return results


def attach_character_video_dirs(
    characters: Sequence[Dict], video_info: Dict[str, Dict]
) -> List[Dict]:
    result = []
    for character in characters:
        row = dict(character)
        info = video_info.get(row["character_id"])
        if info:
            row.update(info)
        result.append(row)
    return result


def write_empty_shot_videos(
    shot_records: Sequence[Dict],
    out_scene_dir: Path,
    overwrite: bool,
) -> Dict[str, str]:
    video_dir = out_scene_dir / "empty_shots"
    if overwrite and video_dir.exists():
        shutil.rmtree(video_dir)

    results: Dict[str, str] = {}
    for row in shot_records:
        if not row.get("is_empty_shot"):
            continue
        source = Path(row["path"])
        if not source.exists():
            continue
        video_dir.mkdir(parents=True, exist_ok=True)
        target = video_dir / source.name
        if overwrite or not target.exists():
            shutil.copy2(source, target)
        if target.exists() and target.stat().st_size > 0:
            results[str(row["shot_id"])] = str(target)
    return results


def remove_output_scene_dir(scene_dir: Path, output_root: Path) -> bool:
    if not scene_dir.exists():
        return False
    resolved_scene = scene_dir.resolve()
    resolved_root = output_root.resolve()
    if resolved_scene == resolved_root or resolved_root not in resolved_scene.parents:
        raise RuntimeError(f"Refusing to remove unsafe scene path: {resolved_scene}")
    shutil.rmtree(resolved_scene)
    return True


def existing_scene_has_single_shot(scene_json: Path) -> bool:
    if not scene_json.exists():
        return False
    try:
        data = read_json(scene_json)
    except Exception:
        return False
    shots = data.get("shots", [])
    return len(shots) == 1 or int(data.get("shot_count", 0) or 0) == 1


def load_shot_rows(shots_root: Path, shot_manifest: Optional[Path]) -> List[Dict]:
    if shot_manifest and shot_manifest.exists():
        return list(iter_jsonl(shot_manifest))

    rows = []
    for movie_dir, scene_dir in iter_scene_dirs(shots_root):
        for shot_idx, path in enumerate(list_videos(scene_dir), start=1):
            rows.append(
                {
                    "movie_id": movie_dir.name,
                    "scene_id": scene_dir.name,
                    "scene_index": 0,
                    "scene_desc": scene_dir.name,
                    "shot_id": f"shot_{shot_idx:04d}",
                    "shot_index": shot_idx,
                    "path": str(path),
                    "duration": 0.0,
                }
            )
    return rows


def process_dataset(args: argparse.Namespace) -> None:
    if FACENET_IMPORT_ERROR is not None:
        raise RuntimeError(
            "character_cluster.py requires torch, facenet_pytorch, and scikit-learn. "
            "Install the packages in requirements.txt before running this stage."
        ) from FACENET_IMPORT_ERROR

    shots_root = Path(args.shots_root)
    output_root = Path(args.output_root)
    shot_manifest = Path(args.shot_manifest) if args.shot_manifest else None
    output_root.mkdir(parents=True, exist_ok=True)
    out_manifest = output_root / "_manifests" / "character_shots.jsonl"
    out_chars_manifest = output_root / "_manifests" / "characters.jsonl"
    out_empty_manifest = output_root / "_manifests" / "empty_shots.jsonl"
    if args.overwrite:
        reset_file(out_manifest)
        reset_file(out_chars_manifest)
        reset_file(out_empty_manifest)
    elif not out_manifest.exists():
        reset_file(out_manifest)
        reset_file(out_chars_manifest)
        reset_file(out_empty_manifest)
    elif not out_empty_manifest.exists():
        reset_file(out_empty_manifest)

    rows = load_shot_rows(shots_root, shot_manifest)
    only = set(args.only_movie or [])
    if only:
        rows = [row for row in rows if row.get("movie_id") in only]
    by_scene: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for row in rows:
        by_scene[(row["movie_id"], row["scene_id"])].append(row)

    configure_torch_model_cache(args.model_cache_dir)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[info] device={device}")
    mtcnn = MTCNN(keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained=args.face_model).eval().to(device)

    for idx, ((movie_id, scene_id), scene_rows) in enumerate(sorted(by_scene.items()), start=1):
        scene_rows = sorted(scene_rows, key=lambda r: int(r.get("shot_index", 0)))
        out_scene_dir = output_root / movie_id / scene_id
        scene_json = out_scene_dir / "characters.json"
        if scene_json.exists() and not args.overwrite:
            if existing_scene_has_single_shot(scene_json):
                if remove_output_scene_dir(out_scene_dir, output_root):
                    print(f"[drop] {movie_id}/{scene_id}: only one final shot; removed scene")
                continue
            print(f"[skip] {movie_id}/{scene_id}")
            continue
        valid_scene_rows = [
            row
            for row in scene_rows
            if get_shot_duration_seconds(Path(row["path"])) >= args.min_shot_duration
        ]
        if len(valid_scene_rows) <= 1:
            if remove_output_scene_dir(out_scene_dir, output_root):
                print(f"[drop] {movie_id}/{scene_id}: only one final shot; removed scene")
            print(
                f"[skip] {movie_id}/{scene_id} "
                f"valid_shots_after_duration_filter={len(valid_scene_rows)}"
            )
            continue
        print(
            f"[scene {idx}/{len(by_scene)}] {movie_id}/{scene_id} "
            f"shots={len(valid_scene_rows)} (duration>={args.min_shot_duration}s)"
        )

        detections: List[Dict] = []
        shot_stats: Dict[str, Dict] = {}
        for row in valid_scene_rows:
            dets, stats = extract_detections_from_shot(
                shot_row=row,
                mtcnn=mtcnn,
                resnet=resnet,
                device=device,
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
        if args.write_character_videos:
            video_info = write_character_videos(
                scene_rows=valid_scene_rows,
                clusters=clusters,
                characters=characters,
                out_scene_dir=out_scene_dir,
                overwrite=args.overwrite,
            )
            characters = attach_character_video_dirs(characters, video_info)

        shot_records = []
        for row in valid_scene_rows:
            shot_id = row["shot_id"]
            occ = sorted(
                shot_to_occurrences.get(shot_id, []),
                key=lambda x: x["confidence"],
                reverse=True,
            )
            stats = shot_stats.get(shot_id, {})
            is_empty_shot = int(stats.get("face_detection_count", 0) or 0) == 0
            shot_records.append(
                {
                    **row,
                    "characters": occ,
                    "dominant_character": occ[0]["character_id"] if occ else None,
                    "character_count": len(occ),
                    "is_empty_shot": is_empty_shot,
                    "stats": stats,
                }
            )

        if args.write_empty_shot_videos:
            empty_video_paths = write_empty_shot_videos(
                shot_records=shot_records,
                out_scene_dir=out_scene_dir,
                overwrite=args.overwrite,
            )
            for row in shot_records:
                copied_path = empty_video_paths.get(str(row["shot_id"]))
                if copied_path:
                    row["empty_shot_video_path"] = copied_path

        empty_records = [row for row in shot_records if row.get("is_empty_shot")]
        write_json(
            scene_json,
            compact_scene_json(
                movie_id=movie_id,
                scene_id=scene_id,
                scene_rows=valid_scene_rows,
                detections=detections,
                characters=characters,
                shot_records=shot_records,
            ),
        )
        append_jsonl(out_manifest, shot_records)
        append_jsonl(out_chars_manifest, characters)
        append_jsonl(out_empty_manifest, empty_records)
        print(
            f"  characters={len(characters)} detections={len(detections)} "
            f"empty_shots={len(empty_records)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect faces and cluster scene-level character identities."
    )
    parser.add_argument("--shots-root", default=r"H:\dataset\movie_multishot_output\shots")
    parser.add_argument(
        "--shot-manifest",
        default=r"H:\dataset\movie_multishot_output\shots\_manifests\shots.jsonl",
    )
    parser.add_argument(
        "--output-root", default=r"H:\dataset\movie_multishot_output\characters"
    )
    parser.add_argument("--only-movie", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default=None)
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
    parser.add_argument(
        "--write-character-videos",
        action="store_true",
        help="Copy each character's source shot videos into per-character folders.",
    )
    parser.add_argument(
        "--write-empty-shot-videos",
        action="store_true",
        help="Copy shots with zero detected faces into each scene's empty_shots folder.",
    )
    return parser.parse_args()


def main() -> None:
    process_dataset(parse_args())


if __name__ == "__main__":
    main()
