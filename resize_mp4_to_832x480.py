import argparse
from pathlib import Path
from typing import List, Tuple

import cv2


TARGET_WIDTH = 832
TARGET_HEIGHT = 480


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively convert all MP4 files to 832x480 with minimal-content-loss crop "
            "and save them into a mirrored output folder."
        )
    )
    parser.add_argument(
        "--input_root",
        type=Path,
        default=Path(r"H:\dataset_small_output"),
        help="Input root folder to scan for mp4 files.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path(r"H:\dataset_small_output_832x480"),
        help="Output root folder with mirrored subfolder structure.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output video if it already exists.",
    )
    return parser.parse_args()


def list_mp4_files(input_root: Path) -> List[Path]:
    return sorted(
        [
            p
            for p in input_root.rglob("*")
            if p.is_file() and p.suffix.lower() == ".mp4"
        ]
    )


def resize_and_crop_frame_cover(frame, target_w: int, target_h: int):
    src_h, src_w = frame.shape[:2]
    if src_h <= 0 or src_w <= 0:
        raise RuntimeError("Invalid frame shape.")

    # Cover strategy: scale up until both dimensions cover target,
    # then center-crop. This minimizes crop amount while avoiding black borders.
    scale = max(target_w / float(src_w), target_h / float(src_h))
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    x0 = max(0, (resized_w - target_w) // 2)
    y0 = max(0, (resized_h - target_h) // 2)
    x1 = x0 + target_w
    y1 = y0 + target_h
    cropped = resized[y0:y1, x0:x1]

    if cropped.shape[1] != target_w or cropped.shape[0] != target_h:
        # Safety fallback if boundary rounding causes mismatch.
        cropped = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return cropped


def get_video_meta(cap: cv2.VideoCapture) -> Tuple[float, int, int]:
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return fps, width, height


def convert_video(in_path: Path, out_path: Path, target_w: int, target_h: int) -> None:
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {in_path}")

    fps, _, _ = get_video_meta(cap)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (target_w, target_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output writer: {out_path}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_frame = resize_and_crop_frame_cover(frame, target_w, target_h)
        writer.write(out_frame)
        frame_count += 1

    writer.release()
    cap.release()

    if frame_count == 0:
        if out_path.exists():
            out_path.unlink()
        raise RuntimeError(f"No valid frames in input video: {in_path}")


def main() -> None:
    args = parse_args()
    input_root: Path = args.input_root
    output_root: Path = args.output_root

    if not input_root.exists() or not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    mp4_files = list_mp4_files(input_root)
    print(f"Found {len(mp4_files)} mp4 files under {input_root}")

    ok_count = 0
    skip_count = 0
    fail_count = 0

    for idx, in_file in enumerate(mp4_files, start=1):
        rel = in_file.relative_to(input_root)
        out_file = output_root / rel

        if out_file.exists() and not args.overwrite:
            skip_count += 1
            if idx % 100 == 0:
                print(f"[{idx}/{len(mp4_files)}] SKIP {rel}")
            continue

        try:
            convert_video(
                in_path=in_file,
                out_path=out_file,
                target_w=TARGET_WIDTH,
                target_h=TARGET_HEIGHT,
            )
            ok_count += 1
            print(f"[{idx}/{len(mp4_files)}] OK   {rel}")
        except Exception as err:
            fail_count += 1
            print(f"[{idx}/{len(mp4_files)}] FAIL {rel} | {err}")

    print("\nDone.")
    print(f"Converted: {ok_count}")
    print(f"Skipped:   {skip_count}")
    print(f"Failed:    {fail_count}")
    print(f"Output:    {output_root}")


if __name__ == "__main__":
    main()
