import argparse
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from scenedetect import ContentDetector, SceneManager, open_video
from sklearn.cluster import DBSCAN


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def list_shot_files(scene_dir: Path) -> List[Path]:
    return sorted(
        [p for p in scene_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    )


def sample_frame_indices(frame_count: int, sample_count: int) -> List[int]:
    if frame_count <= 0:
        return []
    if frame_count <= sample_count:
        return list(range(frame_count))
    return np.linspace(0, frame_count - 1, sample_count, dtype=int).tolist()


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec) + 1e-12
    return vec / norm


def clip_box_to_frame(
    box: np.ndarray, frame_width: int, frame_height: int
) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_width - 1, x2)
    y2 = min(frame_height - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def pick_core_face_box(
    boxes: np.ndarray,
    frame_width: int,
    frame_height: int,
    min_face: int,
    max_center_distance: float,
    min_core_face_area_ratio: float,
) -> Optional[Tuple[int, int, int, int]]:
    frame_area = float(frame_width * frame_height)
    max_center_dist_den = float(
        np.hypot(frame_width / 2.0, frame_height / 2.0)
    ) + 1e-12
    frame_center_x = frame_width / 2.0
    frame_center_y = frame_height / 2.0

    best_box: Optional[Tuple[int, int, int, int]] = None
    best_score = float("-inf")

    for box in boxes:
        clipped = clip_box_to_frame(box, frame_width, frame_height)
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

        # Larger and closer-to-center faces are treated as the core character.
        score = (2.0 * face_area_ratio) - center_dist_norm
        if score > best_score:
            best_score = score
            best_box = clipped

    if best_box is not None:
        return best_box

    # Fallback: if strict "core face" rules reject all faces, choose the largest
    # valid face in frame to avoid losing same-person shots at frame edges.
    fallback_best: Optional[Tuple[int, int, int, int]] = None
    fallback_best_area = 0
    for box in boxes:
        clipped = clip_box_to_frame(box, frame_width, frame_height)
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


def extract_face_embeddings_from_shot(
    shot_path: Path,
    mtcnn: MTCNN,
    resnet: InceptionResnetV1,
    device: torch.device,
    sample_frames: int,
    min_face: int,
    max_center_distance: float,
    min_core_face_area_ratio: float,
) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(shot_path))
    if not cap.isOpened():
        return []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = set(sample_frame_indices(frame_count, sample_frames))

    embeddings: List[np.ndarray] = []
    frame_id = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if frame_id not in indices:
            frame_id += 1
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(frame_rgb)
        if boxes is None:
            frame_id += 1
            continue

        core_box = pick_core_face_box(
            boxes=boxes,
            frame_width=frame_rgb.shape[1],
            frame_height=frame_rgb.shape[0],
            min_face=min_face,
            max_center_distance=max_center_distance,
            min_core_face_area_ratio=min_core_face_area_ratio,
        )
        if core_box is None:
            frame_id += 1
            continue

        x1, y1, x2, y2 = core_box
        face = frame_rgb[y1:y2, x1:x2]
        face_resized = cv2.resize(face, (160, 160))
        face_tensor = torch.from_numpy(face_resized).permute(2, 0, 1).float() / 255.0
        face_tensor = (face_tensor - 0.5) / 0.5
        face_tensor = face_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            emb = resnet(face_tensor).cpu().numpy()[0]
        embeddings.append(l2_normalize(emb))

        frame_id += 1

    cap.release()
    return embeddings


def cluster_scene_faces(
    shot_files: List[Path],
    mtcnn: MTCNN,
    resnet: InceptionResnetV1,
    device: torch.device,
    sample_frames: int,
    min_face: int,
    max_center_distance: float,
    min_core_face_area_ratio: float,
    dbscan_eps: float,
    dbscan_min_samples: int,
) -> Dict[int, Set[Path]]:
    shot_embeddings_for_cluster: List[np.ndarray] = []
    shot_owners: List[Path] = []

    for shot in shot_files:
        shot_embeddings = extract_face_embeddings_from_shot(
            shot,
            mtcnn,
            resnet,
            device,
            sample_frames,
            min_face,
            max_center_distance,
            min_core_face_area_ratio,
        )
        if not shot_embeddings:
            continue
        # Use one representative embedding per shot to reduce frame-level noise.
        shot_rep = l2_normalize(np.mean(np.stack(shot_embeddings, axis=0), axis=0))
        shot_embeddings_for_cluster.append(shot_rep)
        shot_owners.append(shot)

    if not shot_embeddings_for_cluster:
        return {}

    X = np.stack(shot_embeddings_for_cluster, axis=0)
    labels = DBSCAN(
        eps=dbscan_eps, min_samples=dbscan_min_samples, metric="cosine"
    ).fit_predict(X)

    cluster_to_shots: Dict[int, Set[Path]] = defaultdict(set)
    for label, shot in zip(labels, shot_owners):
        if label == -1:
            continue
        cluster_to_shots[int(label)].add(shot)
    return cluster_to_shots


def save_grouped_shots(
    cluster_to_shots: Dict[int, Set[Path]],
    out_scene_dir: Path,
    resplit_selected_videos: bool,
    subshot_threshold: float,
    min_subshot_duration: float,
) -> int:
    saved_character_dirs = 0
    for idx, cluster_id in enumerate(sorted(cluster_to_shots.keys()), start=1):
        character_dir = out_scene_dir / f"character_{idx:02d}"
        character_dir.mkdir(parents=True, exist_ok=True)
        saved_any = False
        for shot in sorted(cluster_to_shots[cluster_id]):
            if resplit_selected_videos:
                saved_count = split_video_to_subshots(
                    video_path=shot,
                    output_dir=character_dir,
                    threshold=subshot_threshold,
                    min_subshot_duration=min_subshot_duration,
                )
                if saved_count > 0:
                    saved_any = True
            else:
                target = character_dir / shot.name
                shutil.copy2(shot, target)
                saved_any = True

        if not saved_any and not any(character_dir.iterdir()):
            character_dir.rmdir()
        elif saved_any:
            saved_character_dirs += 1
    return saved_character_dirs


def detect_subshot_segments(video_path: Path, threshold: float) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    video = None
    try:
        video = open_video(str(video_path))
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))
        scene_manager.detect_scenes(video, show_progress=False)
        scene_list = scene_manager.get_scene_list()

        if not scene_list:
            return []

        for start, end in scene_list:
            start_idx = start.get_frames()
            end_idx = end.get_frames() - 1
            if end_idx >= start_idx:
                segments.append((start_idx, end_idx))
        return segments
    except Exception:
        return []
    finally:
        if video is not None and hasattr(video, "release"):
            video.release()


def split_video_to_subshots(
    video_path: Path,
    output_dir: Path,
    threshold: float,
    min_subshot_duration: float,
) -> int:
    segments = detect_subshot_segments(video_path, threshold)
    if not segments:
        return 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    saved_count = 0

    for sub_idx, (start_frame, end_frame) in enumerate(segments, start=1):
        duration = (end_frame - start_frame + 1) / fps
        if duration <= min_subshot_duration:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        out_name = f"{video_path.stem}_subshot_{sub_idx:03d}.mp4"
        out_path = output_dir / out_name
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

        frame_idx = start_frame
        while frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            frame_idx += 1
        writer.release()

        if out_path.exists() and out_path.stat().st_size > 0:
            saved_count += 1
        else:
            if out_path.exists():
                out_path.unlink()

    cap.release()
    return saved_count


def filter_clusters(cluster_to_shots: Dict[int, Set[Path]]) -> Dict[int, Set[Path]]:
    return {
        cluster_id: shots
        for cluster_id, shots in cluster_to_shots.items()
        if len(shots) >= 2
    }


def process_dataset(
    input_root: Path,
    output_root: Path,
    sample_frames: int,
    min_face: int,
    min_shot_duration: float,
    max_center_distance: float,
    min_core_face_area_ratio: float,
    resplit_selected_videos: bool,
    subshot_threshold: float,
    min_subshot_duration: float,
    dbscan_eps: float,
    dbscan_min_samples: int,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    mtcnn = MTCNN(keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    movie_dirs = sorted([p for p in input_root.iterdir() if p.is_dir()])
    for movie_dir in movie_dirs:
        print(f"\n[Movie] {movie_dir.name}")
        out_movie_dir = output_root / movie_dir.name
        if out_movie_dir.exists():
            print("  -> Skipped: movie already exists in output.")
            continue

        scene_dirs = sorted([p for p in movie_dir.iterdir() if p.is_dir()])

        for scene_dir in scene_dirs:
            all_shot_files = list_shot_files(scene_dir)
            shot_files = [
                shot
                for shot in all_shot_files
                if get_shot_duration_seconds(shot) > min_shot_duration
            ]
            if len(shot_files) <= 1:
                if len(all_shot_files) > 0:
                    print(
                        f"  [Scene] {scene_dir.name} | valid_shots_after_duration_filter={len(shot_files)} -> skipped"
                    )
                continue

            print(
                f"  [Scene] {scene_dir.name} | shots={len(shot_files)} (duration>{min_shot_duration}s)"
            )
            cluster_to_shots = cluster_scene_faces(
                shot_files=shot_files,
                mtcnn=mtcnn,
                resnet=resnet,
                device=device,
                sample_frames=sample_frames,
                min_face=min_face,
                max_center_distance=max_center_distance,
                min_core_face_area_ratio=min_core_face_area_ratio,
                dbscan_eps=dbscan_eps,
                dbscan_min_samples=dbscan_min_samples,
            )

            if not cluster_to_shots:
                print("    -> No valid face clusters found.")
                continue

            cluster_to_shots = filter_clusters(cluster_to_shots)
            if not cluster_to_shots:
                print("    -> No character groups with at least 2 shots, skipped.")
                continue

            out_scene_dir = out_movie_dir / scene_dir.name
            out_scene_dir.mkdir(parents=True, exist_ok=True)
            saved_characters = save_grouped_shots(
                cluster_to_shots=cluster_to_shots,
                out_scene_dir=out_scene_dir,
                resplit_selected_videos=resplit_selected_videos,
                subshot_threshold=subshot_threshold,
                min_subshot_duration=min_subshot_duration,
            )
            if saved_characters == 0:
                print("    -> All subshots <= min_subshot_duration, scene skipped.")
                if not any(out_scene_dir.iterdir()):
                    out_scene_dir.rmdir()
                continue

            print(f"    -> Saved {saved_characters} character folders.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group shots by recurring characters per scene."
    )
    parser.add_argument(
        "--input_root",
        type=str,
        default=r"H:\dataset\movie_shot",
        help="Input root: movie/scene/shot",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=r"H:\dataset\movie_shot_by_character",
        help="Output root: movie/scene/character_x/shot",
    )
    parser.add_argument(
        "--sample_frames",
        type=int,
        default=8,
        help="How many frames to sample from each shot.",
    )
    parser.add_argument(
        "--min_face",
        type=int,
        default=30,
        help="Minimum face width/height in pixels.",
    )
    parser.add_argument(
        "--min_shot_duration",
        type=float,
        default=1.0,
        help="Minimum shot duration in seconds (strictly greater than this value).",
    )
    parser.add_argument(
        "--max_center_distance",
        type=float,
        default=0.6,
        help="Max normalized distance to frame center for core face [0,1].",
    )
    parser.add_argument(
        "--min_core_face_area_ratio",
        type=float,
        default=0.02,
        help="Minimum core face area ratio relative to frame area.",
    )
    parser.add_argument(
        "--resplit_selected_videos",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Re-split each selected video into single continuous subshots before saving.",
    )
    parser.add_argument(
        "--subshot_threshold",
        type=float,
        default=30.0,
        help="PySceneDetect ContentDetector threshold for final subshot splitting.",
    )
    parser.add_argument(
        "--min_subshot_duration",
        type=float,
        default=1.0,
        help="Minimum subshot duration in seconds after final split (strictly greater).",
    )
    parser.add_argument(
        "--dbscan_eps",
        type=float,
        default=0.40,
        help="DBSCAN eps for cosine distance.",
    )
    parser.add_argument(
        "--dbscan_min_samples",
        type=int,
        default=2,
        help="DBSCAN min_samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    process_dataset(
        input_root=input_root,
        output_root=output_root,
        sample_frames=args.sample_frames,
        min_face=args.min_face,
        min_shot_duration=args.min_shot_duration,
        max_center_distance=args.max_center_distance,
        min_core_face_area_ratio=args.min_core_face_area_ratio,
        resplit_selected_videos=args.resplit_selected_videos,
        subshot_threshold=args.subshot_threshold,
        min_subshot_duration=args.min_subshot_duration,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
