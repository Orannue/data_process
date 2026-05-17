"""Add switch_latent_frames to out_22_21_cropped_832x480/*.json from out_22_21/<id>/sample.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

DEFAULT_CROPPED = Path(r"F:\dataset\out_22_21_cropped_832x480")
DEFAULT_SOURCE = Path(r"F:\dataset\out_22_21")


def main() -> int:
    cropped_root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_CROPPED
    source_root = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else DEFAULT_SOURCE

    if not cropped_root.is_dir():
        print(f"Missing cropped root: {cropped_root}", file=sys.stderr)
        return 1
    if not source_root.is_dir():
        print(f"Missing source root: {source_root}", file=sys.stderr)
        return 1

    json_files = sorted(cropped_root.glob("*.json"))
    updated = 0
    errors: list[str] = []

    for path in json_files:
        vid = path.stem
        sample_path = source_root / vid / "sample.json"
        if not sample_path.is_file():
            errors.append(f"No sample: {sample_path}")
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

        latent = sample["switch_latent_frames"]
        if not isinstance(latent, list):
            errors.append(f"switch_latent_frames not a list in {sample_path}")
            continue

        try:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            errors.append(f"{path}: {e}")
            continue

        data["switch_latent_frames"] = list(latent)

        try:
            with path.open("w", encoding="utf-8", newline="\n") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.write("\n")
        except OSError as e:
            errors.append(f"{path}: write {e}")
            continue
        updated += 1

    print(f"Updated {updated} / {len(json_files)} files.")
    if errors:
        print("Issues:", file=sys.stderr)
        for line in errors[:80]:
            print(f"  {line}", file=sys.stderr)
        if len(errors) > 80:
            print(f"  ... and {len(errors) - 80} more", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
