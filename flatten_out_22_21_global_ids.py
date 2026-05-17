"""
Validate out_22_21 samples (shot1=22 latent, shots 2-6=21 latent), then flatten
  <root>/<case>/<case>_<n>/  ->  <root>/<global_id>/
  <root>/<case>/<case>_<n>.mp4 -> <root>/<global_id>.mp4
and refresh paths inside each sample.json.

Order: all 1_character_6_shot (by trailing id), then 2_, then 3_.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

CASE_ORDER = ("1_character_6_shot", "2_character_6_shot", "3_character_6_shot")
EXPECTED_LATENTS = [22, 21, 21, 21, 21, 21]
EXPECTED_RGB = [85, 84, 84, 84, 84, 84]
EXPECTED_TOTAL_LATENT = 127
EXPECTED_TOTAL_RGB = 505


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=Path,
        default=Path(r"F:\dataset\out_22_21"),
        help="Dataset root (contains three case folders).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate and print plan; do not move files.",
    )
    p.add_argument(
        "--delete-case-dirs",
        action="store_true",
        help="After flattening, remove now-empty case directories.",
    )
    return p.parse_args()


def _local_id(folder_name: str, case: str) -> int:
    prefix = f"{case}_"
    if not folder_name.startswith(prefix):
        raise ValueError(f"folder {folder_name!r} does not start with {prefix!r}")
    return int(folder_name[len(prefix) :])


def _iter_samples(root: Path) -> list[tuple[str, Path, int]]:
    rows: list[tuple[str, Path, int]] = []
    for case in CASE_ORDER:
        case_dir = root / case
        if not case_dir.is_dir():
            continue
        for sub in case_dir.iterdir():
            if not sub.is_dir():
                continue
            try:
                lid = _local_id(sub.name, case)
            except ValueError:
                continue
            rows.append((case, sub, lid))
    rows.sort(key=lambda t: (CASE_ORDER.index(t[0]), t[2]))
    return rows


def _validate_sample(sample_dir: Path) -> list[str]:
    err: list[str] = []
    js_path = sample_dir / "sample.json"
    if not js_path.is_file():
        return ["missing sample.json"]
    try:
        data = json.loads(js_path.read_text(encoding="utf-8"))
    except Exception as e:
        return [f"bad json: {e}"]
    sl = data.get("segment_latent_lengths")
    if sl != EXPECTED_LATENTS:
        err.append(f"segment_latent_lengths {sl!r} != {EXPECTED_LATENTS}")
    if data.get("total_latent_frame_length") != EXPECTED_TOTAL_LATENT:
        err.append(
            f"total_latent_frame_length {data.get('total_latent_frame_length')} != {EXPECTED_TOTAL_LATENT}"
        )
    if data.get("total_frame_length") != EXPECTED_TOTAL_RGB:
        err.append(
            f"total_frame_length {data.get('total_frame_length')} != {EXPECTED_TOTAL_RGB}"
        )
    segs = data.get("segments")
    if not isinstance(segs, list) or len(segs) != 6:
        err.append(f"segments: expected 6, got {type(segs)}")
        return err
    for i, seg in enumerate(segs):
        if not isinstance(seg, dict):
            err.append(f"segment[{i}] not dict")
            continue
        if seg.get("latent_length") != EXPECTED_LATENTS[i]:
            err.append(
                f"shot{i+1} latent_length {seg.get('latent_length')} != {EXPECTED_LATENTS[i]}"
            )
        if seg.get("rgb_frames_extracted") != EXPECTED_RGB[i]:
            err.append(
                f"shot{i+1} rgb_frames_extracted {seg.get('rgb_frames_extracted')} != {EXPECTED_RGB[i]}"
            )
    return err


def _update_json_paths(
    data: dict, new_folder: Path, new_merged: Path, case: str, old_folder_name: str
) -> dict:
    data = dict(data)
    data["output_folder"] = str(new_folder.resolve())
    data["merged_video"] = str(new_merged.resolve())
    data["global_sample_id"] = int(new_folder.name)
    data["flatten_source_case"] = case
    data["flatten_source_folder"] = old_folder_name
    segs = data.get("segments")
    if isinstance(segs, list):
        new_segs = []
        for i, seg in enumerate(segs):
            if isinstance(seg, dict):
                s = dict(seg)
                s["output_segment"] = str((new_folder / f"shot{i + 1}.mp4").resolve())
                new_segs.append(s)
            else:
                new_segs.append(seg)
        data["segments"] = new_segs
    return data


def main() -> int:
    args = parse_args()
    root: Path = args.root
    if not root.is_dir():
        raise SystemExit(f"root not found: {root}")

    rows = _iter_samples(root)
    if not rows:
        raise SystemExit("no sample folders found under case dirs")

    bad: list[tuple[str, list[str]]] = []
    for case, sub, lid in rows:
        e = _validate_sample(sub)
        if e:
            bad.append((str(sub.relative_to(root)), e))

    print(f"[validate] samples={len(rows)} ok={len(rows) - len(bad)} bad={len(bad)}")
    if bad:
        for rel, e in bad[:40]:
            print(f"  FAIL {rel}: {'; '.join(e)}")
        if len(bad) > 40:
            print(f"  ... and {len(bad) - 40} more")
        return 1

    plan: list[tuple[str, Path, Path, int]] = []
    for gid, (case, sub, lid) in enumerate(rows, start=1):
        merged = root / case / f"{sub.name}.mp4"
        if not merged.is_file():
            print(f"[error] missing merged mp4: {merged}")
            return 1
        plan.append((case, sub, merged, gid))

    for case, sub, merged, gid in plan[:8]:
        print(f"  gid {gid}: {case}/{sub.name} -> {gid}/ + {gid}.mp4")
    if len(plan) > 8:
        print(f"  ... {len(plan) - 8} more")

    if args.dry_run:
        print("[dry-run] no files moved.")
        return 0

    staging = root / "_flatten_staging"
    if staging.exists():
        raise SystemExit(f"remove staging dir first: {staging}")
    staging.mkdir(parents=True)

    try:
        for case, sub, merged, gid in plan:
            td = staging / f"d_{gid}"
            tm = staging / f"m_{gid}.mp4"
            shutil.move(str(sub), str(td))
            shutil.move(str(merged), str(tm))

        for case, sub, _merged, gid in plan:
            td = staging / f"d_{gid}"
            tm = staging / f"m_{gid}.mp4"
            dest_dir = root / str(gid)
            dest_merged = root / f"{gid}.mp4"
            if dest_dir.exists() or dest_merged.exists():
                raise RuntimeError(f"collision at {dest_dir} or {dest_merged}")
            shutil.move(str(td), str(dest_dir))
            shutil.move(str(tm), str(dest_merged))
            js = dest_dir / "sample.json"
            data = json.loads(js.read_text(encoding="utf-8"))
            new_data = _update_json_paths(
                data, dest_dir, dest_merged, case=case, old_folder_name=sub.name
            )
            js.write_text(json.dumps(new_data, ensure_ascii=False, indent=2), encoding="utf-8")

        staging.rmdir()
    except Exception:
        print(f"[fatal] stopped; inspect {staging} and partial moves under {root}")
        raise

    if args.delete_case_dirs:
        for case in CASE_ORDER:
            cd = root / case
            if cd.is_dir():
                try:
                    cd.rmdir()
                except OSError:
                    print(f"[warn] could not rmdir (not empty?): {cd}")

    map_path = root / "_global_id_mapping.json"
    map_path.write_text(
        json.dumps(
            {
                "root": str(root.resolve()),
                "count": len(plan),
                "order": "1_character_6_shot then 2_ then 3_ by local id",
                "items": [
                    {
                        "global_id": gid,
                        "source_case": case,
                        "source_folder": sub.name,
                    }
                    for case, sub, _m, gid in plan
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[done] flattened {len(plan)} samples under {root}")
    print(f"[done] wrote {map_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
