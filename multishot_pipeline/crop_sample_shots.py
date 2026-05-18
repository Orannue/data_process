import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2

from utils import append_jsonl, iter_jsonl, reset_file, write_json


def read_video_info(path: Path) -> Dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"ok": False, "fps": 0.0, "frame_count": 0, "width": 0, "height": 0}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {
        "ok": fps > 0 and frame_count > 0 and width > 0 and height > 0,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
    }


def shot_rows_from_sample(sample: Dict) -> List[Dict]:
    rows = sample.get("shots") or []
    if rows:
        return [dict(row) for row in rows]

    shot_ids = sample.get("shot_ids", [])
    shot_indices = sample.get("shot_indices", [])
    shot_paths = sample.get("shot_paths", [])
    shot_roles = sample.get("shot_roles", [])
    result = []
    for idx, path in enumerate(shot_paths):
        result.append(
            {
                "shot_id": shot_ids[idx] if idx < len(shot_ids) else f"shot_{idx + 1:04d}",
                "shot_index": shot_indices[idx] if idx < len(shot_indices) else idx + 1,
                "path": path,
                "role": shot_roles[idx] if idx < len(shot_roles) else "character",
            }
        )
    return result


def max_blocks_for_shot(frame_count: int, is_first: bool) -> int:
    if is_first:
        return max(0, (frame_count - 1) // 12)
    return max(0, frame_count // 12)


def distribute_blocks(max_blocks: Sequence[int], block_budget: int) -> Optional[List[int]]:
    if not max_blocks:
        return None
    if any(block < 1 for block in max_blocks):
        return None
    if len(max_blocks) > block_budget:
        return None

    total_max = sum(max_blocks)
    if total_max <= block_budget:
        return list(max_blocks)

    blocks = [1 for _ in max_blocks]
    remaining = block_budget - len(blocks)
    capacities = [block - 1 for block in max_blocks]
    total_capacity = sum(capacities)
    if remaining < 0 or total_capacity <= 0:
        return blocks if sum(blocks) <= block_budget else None

    assigned = [0 for _ in max_blocks]
    remainders: List[Tuple[float, int]] = []
    for idx, capacity in enumerate(capacities):
        exact = remaining * (capacity / total_capacity)
        whole = min(capacity, int(exact))
        assigned[idx] = whole
        remainders.append((exact - whole, idx))

    leftover = remaining - sum(assigned)
    for _, idx in sorted(remainders, reverse=True):
        if leftover <= 0:
            break
        room = capacities[idx] - assigned[idx]
        if room <= 0:
            continue
        assigned[idx] += 1
        leftover -= 1

    return [base + extra for base, extra in zip(blocks, assigned)]


def target_frames_for_blocks(blocks: int, is_first: bool) -> int:
    return 12 * blocks + 1 if is_first else 12 * blocks


def latent_count_for_blocks(blocks: int, is_first: bool) -> int:
    return 3 * blocks + 1 if is_first else 3 * blocks


def center_crop_range(frame_count: int, target_frames: int) -> Tuple[int, int]:
    start = max(0, (frame_count - target_frames) // 2)
    end = start + target_frames - 1
    return start, end


def write_video_crop(
    source_path: Path,
    out_path: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
    size: Tuple[int, int],
) -> bool:
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps or 25.0),
        size,
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    expected = end_frame - start_frame + 1
    written = 0
    for _ in range(expected):
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] != size[0] or frame.shape[0] != size[1]:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        writer.write(frame)
        written += 1

    cap.release()
    writer.release()
    return written == expected and out_path.exists() and out_path.stat().st_size > 0


def build_crop_plan(sample: Dict, max_latent_frames: int) -> Optional[Dict]:
    shots = shot_rows_from_sample(sample)
    if not shots:
        return None

    infos = []
    for idx, shot in enumerate(shots):
        path = Path(shot["path"])
        info = read_video_info(path)
        if not info["ok"]:
            return None
        info["path"] = path
        info["shot"] = shot
        info["max_blocks"] = max_blocks_for_shot(info["frame_count"], idx == 0)
        infos.append(info)

    block_budget = (int(max_latent_frames) - 1) // 3
    blocks = distribute_blocks([info["max_blocks"] for info in infos], block_budget)
    if blocks is None:
        return None

    crop_ranges = []
    switch_frames = []
    switch_latent_frames = []
    total_frames = 0
    total_latent_frames = 0
    for idx, (info, block_count) in enumerate(zip(infos, blocks)):
        is_first = idx == 0
        target_frames = target_frames_for_blocks(block_count, is_first)
        latent_frames = latent_count_for_blocks(block_count, is_first)
        start_frame, end_frame = center_crop_range(info["frame_count"], target_frames)
        total_frames += target_frames
        total_latent_frames += latent_frames
        if idx < len(infos) - 1:
            switch_frames.append(total_frames)
            switch_latent_frames.append(total_latent_frames)
        crop_ranges.append(
            {
                "shot_id": info["shot"].get("shot_id", f"shot_{idx + 1:04d}"),
                "shot_index": int(info["shot"].get("shot_index", idx + 1)),
                "role": info["shot"].get("role", "character"),
                "path": str(info["path"]),
                "frame_count": int(info["frame_count"]),
                "target_frames": int(target_frames),
                "latent_frames": int(latent_frames),
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "fps": float(info["fps"]),
                "width": int(info["width"]),
                "height": int(info["height"]),
            }
        )

    if total_latent_frames > max_latent_frames:
        return None

    return {
        "crop_ranges": crop_ranges,
        "switch_frames": switch_frames,
        "switch_latent_frames": switch_latent_frames,
        "total_frames": total_frames,
        "total_latent_frames": total_latent_frames,
    }


def output_sample_dir(output_root: Path, sample: Dict) -> Path:
    return output_root / sample["movie_id"] / sample["scene_id"] / sample["sample_id"]


def process_sample(sample: Dict, args: argparse.Namespace, output_root: Path) -> Optional[Dict]:
    plan = build_crop_plan(sample, args.max_latent_frames)
    if plan is None:
        return None

    sample_dir = output_sample_dir(output_root, sample)
    if sample_dir.exists() and not args.overwrite:
        return {
            "sample_id": sample["sample_id"],
            "movie_id": sample["movie_id"],
            "scene_id": sample["scene_id"],
            "output_dir": str(sample_dir),
            "info_path": str(sample_dir / "info.json"),
            "total_frames": plan["total_frames"],
            "total_latent_frames": plan["total_latent_frames"],
            "skipped_existing": True,
        }

    written_paths = []
    for idx, crop in enumerate(plan["crop_ranges"], start=1):
        out_path = sample_dir / f"shot_{idx:04d}.mp4"
        ok = write_video_crop(
            source_path=Path(crop["path"]),
            out_path=out_path,
            start_frame=crop["start_frame"],
            end_frame=crop["end_frame"],
            fps=crop["fps"],
            size=(crop["width"], crop["height"]),
        )
        if not ok:
            return None
        written_paths.append(str(out_path))

    write_json(
        sample_dir / "info.json",
        {
            "total_frames": plan["total_frames"],
            "total_latent_frames": plan["total_latent_frames"],
            "switch_frames": plan["switch_frames"],
            "switch_latent_frames": plan["switch_latent_frames"],
        },
    )

    return {
        "sample_id": sample["sample_id"],
        "movie_id": sample["movie_id"],
        "scene_id": sample["scene_id"],
        "output_dir": str(sample_dir),
        "info_path": str(sample_dir / "info.json"),
        "shot_paths": written_paths,
        "total_frames": plan["total_frames"],
        "total_latent_frames": plan["total_latent_frames"],
    }


def process(args: argparse.Namespace) -> None:
    samples_manifest = Path(args.samples_manifest)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "_manifests" / "cropped_samples.jsonl"
    if args.overwrite or not manifest_path.exists():
        reset_file(manifest_path)

    written = 0
    skipped = 0
    for sample in iter_jsonl(samples_manifest):
        result = process_sample(sample, args, output_root)
        if result is None:
            skipped += 1
            print(f"[skip] {sample.get('sample_id', '<unknown>')}: cannot satisfy crop plan")
            continue
        append_jsonl(manifest_path, [result])
        written += 1
        print(
            f"[sample] {result['sample_id']} frames={result['total_frames']} "
            f"latent={result['total_latent_frames']}"
        )

    write_json(
        output_root / "summary.json",
        {
            "samples_manifest": str(samples_manifest),
            "output_root": str(output_root),
            "max_latent_frames": args.max_latent_frames,
            "written": written,
            "skipped": skipped,
        },
    )
    print(f"[done] cropped={written} skipped={skipped}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop selected multishot samples into latent-valid shot segments."
    )
    parser.add_argument(
        "--samples-manifest",
        default=r"H:\dataset\movie_multishot_output\samples\samples.jsonl",
    )
    parser.add_argument(
        "--output-root",
        default=r"H:\dataset\movie_multishot_output\cropped_samples",
    )
    parser.add_argument("--max-latent-frames", type=int, default=127)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    process(parse_args())


if __name__ == "__main__":
    main()
