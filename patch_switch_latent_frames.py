"""
Patch movie_variants_chunked_cropped_832x480/*.json:
- Add switch_latent_frames from new_chunked/<id>/sample.json, dropping trailing 121.
- Trim switch_frames in x.json by dropping trailing 481.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DEFAULT_CROPPED_ROOT = Path(r"H:\dataset\movie_variants_chunked_cropped_832x480")
NEW_CHUNKED_ROOT = Path(r"H:\dataset\new_chunked")


def trim_switch_latent(frames: list) -> list:
    if not isinstance(frames, list):
        raise TypeError("switch_latent_frames must be a list")
    out = list(frames)
    if out and out[-1] == 121:
        out = out[:-1]
    return out


def trim_switch_rgb(frames: list) -> list:
    if not isinstance(frames, list):
        raise TypeError("switch_frames must be a list")
    out = list(frames)
    if out and out[-1] == 481:
        out = out[:-1]
    return out


def main() -> int:
    cropped_root = (
        Path(sys.argv[1]).resolve()
        if len(sys.argv) > 1
        else DEFAULT_CROPPED_ROOT
    )
    if not cropped_root.is_dir():
        print(f"Missing cropped root: {cropped_root}", file=sys.stderr)
        return 1
    if not NEW_CHUNKED_ROOT.is_dir():
        print(f"Missing new_chunked root: {NEW_CHUNKED_ROOT}", file=sys.stderr)
        return 1

    json_files = sorted(cropped_root.glob("*.json"))
    json_files = [p for p in json_files if p.stem.isdigit()]
    updated = 0
    skipped_no_sample = 0
    errors: list[str] = []

    for path in json_files:
        vid = path.stem
        sample_path = NEW_CHUNKED_ROOT / vid / "sample.json"
        if not sample_path.is_file():
            local_sample = cropped_root / vid / "sample.json"
            if local_sample.is_file():
                sample_path = local_sample
            else:
                skipped_no_sample += 1
                errors.append(f"No sample: {NEW_CHUNKED_ROOT / vid / 'sample.json'} or {local_sample}")
                continue

        try:
            with sample_path.open(encoding="utf-8") as f:
                sample = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            errors.append(f"{sample_path}: {e}")
            continue

        if "switch_latent_frames" not in sample:
            errors.append(f"No switch_latent_frames in {sample_path}")
            continue

        try:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            errors.append(f"{path}: {e}")
            continue

        if "switch_frames" not in data:
            errors.append(f"No switch_frames in {path}")
            continue

        try:
            latent = trim_switch_latent(sample["switch_latent_frames"])
            rgb = trim_switch_rgb(data["switch_frames"])
        except TypeError as e:
            errors.append(f"{path}: {e}")
            continue

        data["switch_frames"] = rgb
        data["switch_latent_frames"] = latent

        try:
            with path.open("w", encoding="utf-8", newline="\n") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.write("\n")
        except OSError as e:
            errors.append(f"{path}: write {e}")
            continue
        updated += 1

    print(f"Updated {updated} files.")
    if skipped_no_sample:
        print(f"Skipped (no sample.json): {skipped_no_sample}")
    if errors:
        print("Issues:", file=sys.stderr)
        for line in errors[:50]:
            print(f"  {line}", file=sys.stderr)
        if len(errors) > 50:
            print(f"  ... and {len(errors) - 50} more", file=sys.stderr)
        return 2

    return 0



if __name__ == "__main__":
    # Usage: python patch_switch_latent_frames.py [cropped_json_dir]
    raise SystemExit(main())
