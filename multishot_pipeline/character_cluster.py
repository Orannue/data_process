import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from utils import (
    append_jsonl,
    cosine_similarity,
    frame_quality_stats,
    hsv_histogram,
    iter_jsonl,
    iter_scene_dirs,
    l2_normalize,
    list_videos,
    read_sampled_frames,
    reset_file,
    stable_id,
    write_json,
)

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


def clip_box(box, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def face_embedding(face_rgb: np.ndarray, resnet, device) -> np.ndarray:
    face = cv2.resize(face_rgb, (160, 160), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(face).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - 0.5) / 0.5
    tensor = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        emb = resnet(tensor).cpu().numpy()[0]
    return l2_normalize(emb)


def extract_detections_from_shot(
    shot_row: Dict,
    mtcnn,
    resnet,
    device,
    sample_frames: int,
    min_face: int,
    min_prob: float,
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
        probs = probs if probs is not None else [1.0] * len(boxes)
        height, width = frame_rgb.shape[:2]
        frame_area = float(width * height)

        for box, prob in zip(boxes, probs):
            if float(prob) < min_prob:
                continue
            clipped = clip_box(box, width, height)
            if clipped is None:
                continue
            x1, y1, x2, y2 = clipped
            face_w = x2 - x1
            face_h = y2 - y1
            if face_w < min_face or face_h < min_face:
                continue
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
) -> Dict[int, List[Dict]]:
    if not detections:
        return {}
    X = np.asarray([det["embedding"] for det in detections], dtype=np.float32)
    labels = DBSCAN(
        eps=dbscan_eps, min_samples=dbscan_min_samples, metric="cosine"
    ).fit_predict(X)
    clusters: Dict[int, List[Dict]] = defaultdict(list)
    for label, det in zip(labels, detections):
        if int(label) == -1:
            continue
        clusters[int(label)].append(det)
    return dict(clusters)


def summarize_clusters(
    movie_id: str,
    scene_id: str,
    clusters: Dict[int, List[Dict]],
    min_shots_per_character: int,
) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    characters = []
    shot_to_occurrences: Dict[str, List[Dict]] = defaultdict(list)

    for local_idx, cluster_id in enumerate(sorted(clusters.keys()), start=1):
        dets = clusters[cluster_id]
        shot_ids = sorted({d["shot_id"] for d in dets})
        if len(shot_ids) < min_shots_per_character:
            continue
        char_id = f"char_{local_idx:03d}"
        embeddings = np.asarray([d["embedding"] for d in dets], dtype=np.float32)
        rep = l2_normalize(np.mean(embeddings, axis=0)).astype(float).tolist()
        by_shot: Dict[str, List[Dict]] = defaultdict(list)
        for det in dets:
            by_shot[det["shot_id"]].append(det)

        shot_support = {}
        for shot_id, shot_dets in by_shot.items():
            mean_prob = float(np.mean([d["prob"] for d in shot_dets]))
            mean_area = float(np.mean([d["area_ratio"] for d in shot_dets]))
            confidence = min(1.0, 0.60 * mean_prob + 8.0 * mean_area)
            shot_support[shot_id] = {
                "detections": len(shot_dets),
                "mean_prob": mean_prob,
                "mean_area_ratio": mean_area,
                "confidence": confidence,
            }
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
                "shot_count": len(shot_support),
                "shot_support": shot_support,
                "representative_embedding": rep,
            }
        )

    return characters, shot_to_occurrences


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
    if args.overwrite:
        reset_file(out_manifest)
        reset_file(out_chars_manifest)
    elif not out_manifest.exists():
        reset_file(out_manifest)
        reset_file(out_chars_manifest)

    rows = load_shot_rows(shots_root, shot_manifest)
    by_scene: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for row in rows:
        by_scene[(row["movie_id"], row["scene_id"])].append(row)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[info] device={device}")
    mtcnn = MTCNN(keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained=args.face_model).eval().to(device)

    for idx, ((movie_id, scene_id), scene_rows) in enumerate(sorted(by_scene.items()), start=1):
        scene_rows = sorted(scene_rows, key=lambda r: int(r.get("shot_index", 0)))
        out_scene_dir = output_root / movie_id / scene_id
        scene_json = out_scene_dir / "characters.json"
        if scene_json.exists() and not args.overwrite:
            print(f"[skip] {movie_id}/{scene_id}")
            continue
        print(f"[scene {idx}/{len(by_scene)}] {movie_id}/{scene_id} shots={len(scene_rows)}")

        detections: List[Dict] = []
        shot_stats: Dict[str, Dict] = {}
        for row in scene_rows:
            dets, stats = extract_detections_from_shot(
                shot_row=row,
                mtcnn=mtcnn,
                resnet=resnet,
                device=device,
                sample_frames=args.sample_frames,
                min_face=args.min_face,
                min_prob=args.min_face_prob,
            )
            detections.extend(dets)
            shot_stats[row["shot_id"]] = stats

        clusters = cluster_scene_detections(
            detections,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_samples=args.dbscan_min_samples,
        )
        characters, shot_to_occurrences = summarize_clusters(
            movie_id=movie_id,
            scene_id=scene_id,
            clusters=clusters,
            min_shots_per_character=args.min_shots_per_character,
        )

        shot_records = []
        for row in scene_rows:
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
                    "stats": shot_stats.get(shot_id, {}),
                }
            )

        write_json(
            scene_json,
            {
                "movie_id": movie_id,
                "scene_id": scene_id,
                "shot_count": len(scene_rows),
                "detection_count": len(detections),
                "character_count": len(characters),
                "characters": characters,
                "shots": shot_records,
            },
        )
        append_jsonl(out_manifest, shot_records)
        append_jsonl(out_chars_manifest, characters)
        print(f"  characters={len(characters)} detections={len(detections)}")


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
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--face-model", default="vggface2")
    parser.add_argument("--sample-frames", type=int, default=8)
    parser.add_argument("--min-face", type=int, default=28)
    parser.add_argument("--min-face-prob", type=float, default=0.90)
    parser.add_argument("--dbscan-eps", type=float, default=0.38)
    parser.add_argument("--dbscan-min-samples", type=int, default=2)
    parser.add_argument("--min-shots-per-character", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    process_dataset(parse_args())


if __name__ == "__main__":
    main()
