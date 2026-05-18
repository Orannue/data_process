from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from caption_shots_with_qwen3_vl import (  # noqa: E402
    VIDEO_EXTS,
    build_joint_messages,
    distribute_frame_samples,
    generate_joint_caption_with_retry,
    get_video_frame_count,
    natural_sort_key,
    sample_shot_frames,
)


DEFAULT_INPUT_ROOT = Path(r"H:\dataset\movie_multishot_output\cropped_samples")
INFO_JSON_NAME = "info.json"


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def list_sample_dirs(input_root: Path) -> List[Path]:
    info_paths = sorted(
        input_root.rglob(INFO_JSON_NAME),
        key=lambda p: natural_sort_key(str(p.parent.relative_to(input_root))),
    )
    return [p.parent for p in info_paths if "_manifests" not in p.parts]


def is_shot_video(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
        return False
    name = path.name.lower()
    if name == "merged.mp4":
        return False
    if name.endswith(".tmp.mp4") or ".tmp." in name:
        return False
    return name.startswith("shot")


def resolve_shot_paths(sample_dir: Path) -> List[Path]:
    shots = [p for p in sample_dir.iterdir() if is_shot_video(p)]
    return sorted(shots, key=lambda p: natural_sort_key(p.name))


def prompt_for_shot(
    shot_path: Path,
    shot_prompts: Dict[str, str],
    fallback_values: List[str],
    index: int,
) -> str:
    for key in (shot_path.name, shot_path.stem, str(index + 1), f"shot_{index + 1:04d}.mp4"):
        value = shot_prompts.get(key)
        if value:
            return value
    return fallback_values[index] if index < len(fallback_values) else ""


def build_ordered_prompts(
    shot_paths: List[Path],
    shot_prompts: Dict[str, str],
) -> Tuple[Dict[str, str], List[str]]:
    fallback_values = list(shot_prompts.values())
    ordered_by_file: Dict[str, str] = {}
    ordered_list: List[str] = []
    for idx, shot_path in enumerate(shot_paths):
        prompt = prompt_for_shot(shot_path, shot_prompts, fallback_values, idx)
        ordered_by_file[shot_path.name] = prompt
        ordered_list.append(prompt)
    return ordered_by_file, ordered_list


def has_existing_caption(info: Dict[str, Any], shot_count: int) -> bool:
    prompts = info.get("prompts")
    return isinstance(prompts, list) and len(prompts) == shot_count and all(prompts)


def caption_one_sample(
    args: argparse.Namespace,
    api_key: str,
    sample_dir: Path,
    idx: int,
    total: int,
) -> Dict[str, Any]:
    info_path = sample_dir / INFO_JSON_NAME
    info = load_json(info_path)
    shot_paths = resolve_shot_paths(sample_dir)
    rel = str(sample_dir.relative_to(args.input_root))

    if not shot_paths:
        return {
            "sample": rel,
            "sample_dir": str(sample_dir),
            "status": "failed",
            "reason": "shot_files_not_found",
        }
    if has_existing_caption(info, len(shot_paths)) and not args.overwrite:
        return {
            "sample": rel,
            "sample_dir": str(sample_dir),
            "status": "skipped_existing",
            "shot_count": len(shot_paths),
        }

    frame_counts = [get_video_frame_count(path) for path in shot_paths]
    allocations = distribute_frame_samples(frame_counts, args.total_frame_samples)
    frame_urls_by_shot: Dict[str, List[str]] = {}
    for shot_path, alloc in zip(shot_paths, allocations):
        urls = sample_shot_frames(
            shot_path,
            alloc,
            args.target_frame_height,
            args.jpeg_quality,
        )
        if urls:
            frame_urls_by_shot[shot_path.name] = urls

    if not frame_urls_by_shot:
        return {
            "sample": rel,
            "sample_dir": str(sample_dir),
            "status": "failed",
            "reason": "cannot_read_sample_frames",
            "shot_count": len(shot_paths),
        }

    global_prompt, character_bindings, raw_shot_prompts = generate_joint_caption_with_retry(
        api_base=args.api_base,
        api_key=api_key,
        model=args.model,
        messages=build_joint_messages(frame_urls_by_shot),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout_seconds=args.timeout_seconds,
        api_retry=args.retry,
        response_retry=args.response_retry,
        json_fix_retry=args.json_fix_retry,
        json_fix_max_tokens=args.json_fix_max_tokens,
        response_format_json=not args.no_response_format_json,
        sleep_between_calls=args.sleep_between_calls,
    )
    time.sleep(max(0.0, args.sleep_between_calls))

    _, prompts = build_ordered_prompts(shot_paths, raw_shot_prompts)
    if any(not prompt for prompt in prompts):
        return {
            "sample": rel,
            "sample_dir": str(sample_dir),
            "status": "failed",
            "reason": "missing_ordered_prompt",
            "raw_shot_prompts": raw_shot_prompts,
        }

    info.pop("shot_prompts", None)
    info.update(
        {
            "global_prompt": global_prompt,
            "character_bindings": character_bindings,
            "prompts": prompts,
            "caption_model": args.model,
            "caption_source_shots": [path.name for path in shot_paths],
            "caption_updated_at": int(time.time()),
        }
    )
    save_json(info_path, info)
    print(f"[{idx}/{total}] OK {rel}")
    return {
        "sample": rel,
        "sample_dir": str(sample_dir),
        "info_path": str(info_path),
        "status": "ok",
        "shot_count": len(shot_paths),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Caption split shot videos in cropped sample folders with Qwen3-VL and "
            "write prompts back into each folder's info.json."
        )
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument(
        "--state-json",
        type=Path,
        default=None,
        help="Optional state/report JSON. Default: <input-root>/_manifests/qwen3_vl_info_caption_state.json",
    )
    parser.add_argument("--api-base", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--model", default="qwen3-vl-plus")
    parser.add_argument(
        "--api-key",
        default="",
        help="API key. If omitted, read DASHSCOPE_API_KEY or OPENAI_API_KEY.",
    )
    parser.add_argument("--total-frame-samples", type=int, default=20)
    parser.add_argument("--target-frame-height", type=int, default=480)
    parser.add_argument("--jpeg-quality", type=int, default=82)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--retry", type=int, default=10)
    parser.add_argument("--response-retry", type=int, default=4)
    parser.add_argument("--json-fix-retry", type=int, default=4)
    parser.add_argument("--json-fix-max-tokens", type=int, default=4096)
    parser.add_argument("--no-response-format-json", action="store_true")
    parser.add_argument("--sleep-between-calls", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input_root.is_dir():
        raise SystemExit(f"Missing input root: {args.input_root}")

    api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing API key. Use --api-key or set DASHSCOPE_API_KEY/OPENAI_API_KEY.")

    state_path = args.state_json
    if state_path is None:
        state_path = args.input_root / "_manifests" / "qwen3_vl_info_caption_state.json"

    sample_dirs = list_sample_dirs(args.input_root)
    if args.max_samples is not None:
        sample_dirs = sample_dirs[: args.max_samples]
    if not sample_dirs:
        raise SystemExit(f"No sample info.json files found under: {args.input_root}")

    results: List[Dict[str, Any]] = []
    lock = threading.Lock()
    workers = max(1, int(args.workers))

    def run_one(i: int, folder: Path) -> Dict[str, Any]:
        try:
            result = caption_one_sample(args, api_key, folder, i, len(sample_dirs))
        except Exception as err:
            rel = str(folder.relative_to(args.input_root))
            result = {
                "sample": rel,
                "sample_dir": str(folder),
                "status": "failed",
                "reason": str(err),
            }
            print(f"[{i}/{len(sample_dirs)}] FAIL {rel}: {err}")
        with lock:
            results.append(result)
            save_json(
                state_path,
                {
                    "input_root": str(args.input_root),
                    "model": args.model,
                    "updated_at": int(time.time()),
                    "results": results,
                },
            )
        return result

    if workers == 1:
        for i, folder in enumerate(sample_dirs, start=1):
            run_one(i, folder)
    else:
        print(f"Using {workers} workers for {len(sample_dirs)} sample(s).")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(run_one, i, folder): (i, folder)
                for i, folder in enumerate(sample_dirs, start=1)
            }
            for future in as_completed(futures):
                future.result()

    counts: Dict[str, int] = {}
    for result in results:
        status = str(result.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    print(f"Done. status={counts}, state={state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
