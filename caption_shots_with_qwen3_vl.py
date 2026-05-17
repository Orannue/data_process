import argparse
import base64
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import floor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
SAMPLE_JSON_NAME = "sample.json"


def natural_sort_key(text: str) -> List[object]:
    parts = re.split(r"(\d+)", text.lower())
    return [int(p) if p.isdigit() else p for p in parts]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Caption all shot videos in each subfolder under an input root."
    )
    parser.add_argument(
        "--input_root",
        type=Path,
        default=Path(r"F:\dataset\out_22_21"),
        help="Root folder containing per-video subfolders. If a subfolder has sample.json "
        "(e.g. from build_variants_chunk_dataset), total_frames, total_latent_frames, "
        "and switch_frames are copied into caption outputs.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path(r"F:\dataset\out_22_21\qwen3_vl_shot_captions.json"),
        help="Output JSON containing all generated captions.",
    )
    parser.add_argument(
        "--state_json",
        type=Path,
        default=Path(r"F:\dataset\out_22_21\qwen3_vl_caption_state.json"),
        help="State file for resumable processing.",
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-vl-plus",
        help="Qwen3-VL model name.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="sk-1a98edd3390f477d97b8bf8cf666812f",
        help="API key. If omitted, read from DASHSCOPE_API_KEY or OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--total_frame_samples",
        type=int,
        default=20,
        help="Total sampled frame count across all shots in one source folder.",
    )
    parser.add_argument(
        "--target_frame_height",
        type=int,
        default=480,
        help="Resize sampled frames to approximately this height while preserving aspect ratio.",
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=82,
        help="JPEG quality 1-100 for encoded frames (lower = smaller request bodies, fewer400s from oversized payloads).",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Max tokens for each caption generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for generation.",
    )
    parser.add_argument(
        "--timeout_seconds",
        type=int,
        default=120,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--retry",
        type=int,
        default=10,
        help="Retry times for each API call.",
    )
    parser.add_argument(
        "--response_retry",
        type=int,
        default=4,
        help="Retry times for invalid/unparseable model response JSON.",
    )
    parser.add_argument(
        "--json_fix_retry",
        type=int,
        default=4,
        help="When JSON parsing fails, call API to repair JSON up to this count.",
    )
    parser.add_argument(
        "--json_fix_max_tokens",
        type=int,
        default=4096,
        help="Max tokens for the JSON repair API call (needs room for long prompts).",
    )
    parser.add_argument(
        "--no_response_format_json",
        action="store_true",
        help="Do not send response_format json_object (use if the API rejects it).",
    )
    parser.add_argument(
        "--sleep_between_calls",
        type=float,
        default=0.3,
        help="Sleep seconds between API calls.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel folder workers (HTTP-bound; use 1 for strict serial + lowest rate-limit risk).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate captions even if already present in state/output.",
    )
    return parser.parse_args()


def sample_indices(frame_count: int, sample_count: int) -> List[int]:
    if frame_count <= 0:
        return []
    if sample_count <= 1:
        return [max(0, frame_count // 2)]
    if frame_count <= sample_count:
        return list(range(frame_count))
    return [round(i * (frame_count - 1) / (sample_count - 1)) for i in range(sample_count)]


def frame_to_data_url(frame_bgr, jpeg_quality: int = 82) -> str:
    q = max(1, min(100, int(jpeg_quality)))
    ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        raise RuntimeError("Failed to encode frame to jpg.")
    b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def resize_frame_keep_ratio(frame_bgr, target_height: int):
    if target_height <= 0:
        return frame_bgr
    height, width = frame_bgr.shape[:2]
    if height <= 0 or width <= 0:
        return frame_bgr
    if height == target_height:
        return frame_bgr
    scale = target_height / float(height)
    new_width = max(1, int(round(width * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(frame_bgr, (new_width, target_height), interpolation=interpolation)


def sample_shot_frames(
    shot_path: Path, frame_samples: int, target_height: int, jpeg_quality: int
) -> List[str]:
    if frame_samples <= 0:
        return []
    cap = cv2.VideoCapture(str(shot_path))
    if not cap.isOpened():
        return []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = set(sample_indices(frame_count, frame_samples))

    frame_data_urls: List[str] = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in indices:
            resized = resize_frame_keep_ratio(frame, target_height)
            frame_data_urls.append(frame_to_data_url(resized, jpeg_quality=jpeg_quality))
        frame_idx += 1
    cap.release()
    return frame_data_urls


def load_json_or_default(path: Path, default_value):
    if not path.exists():
        return default_value
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default_value


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def list_input_subdirs(input_root: Path) -> List[Path]:
    if not input_root.exists() or not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    return sorted([p for p in input_root.iterdir() if p.is_dir()], key=lambda p: natural_sort_key(p.name))


def load_sample_json_timing_fields(source_folder: Path) -> Dict[str, Any]:
    """
    Read build_variants_chunk_dataset (or compatible) sample.json next to shot videos.
    Maps total_frame_length -> total_frames, total_latent_frame_length -> total_latent_frames;
    copies switch_frames unchanged.
    """
    sample_path = source_folder / SAMPLE_JSON_NAME
    if not sample_path.is_file():
        return {}
    try:
        data = json.loads(sample_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    out: Dict[str, Any] = {}
    if "total_frame_length" in data:
        out["total_frames"] = data["total_frame_length"]
    if "total_latent_frame_length" in data:
        out["total_latent_frames"] = data["total_latent_frame_length"]
    if "switch_frames" in data:
        out["switch_frames"] = data["switch_frames"]
    return out


def resolve_shot_paths(source_folder: Path, video_names: Optional[List[str]]) -> List[Path]:
    resolved: List[Path] = []
    if video_names:
        for name in video_names:
            candidate = source_folder / str(name)
            if candidate.exists() and candidate.is_file() and candidate.suffix.lower() in VIDEO_EXTS:
                resolved.append(candidate)
    if resolved:
        return resolved

    fallback = sorted(
        [
            p
            for p in source_folder.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS and p.name.lower().startswith("shot")
        ],
        key=lambda p: natural_sort_key(p.name),
    )
    return fallback


def get_video_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(0, frame_count)


def distribute_frame_samples(frame_counts: List[int], total_samples: int) -> List[int]:
    shot_count = len(frame_counts)
    if shot_count == 0 or total_samples <= 0:
        return [0] * shot_count

    weights = [max(1, c) for c in frame_counts]
    if shot_count <= total_samples:
        allocations = [1] * shot_count
        remaining = total_samples - shot_count
    else:
        allocations = [0] * shot_count
        remaining = total_samples

    if remaining <= 0:
        return allocations

    weight_sum = float(sum(weights))
    raw_shares = [remaining * (w / weight_sum) for w in weights]
    int_shares = [floor(v) for v in raw_shares]
    for idx, v in enumerate(int_shares):
        allocations[idx] += v

    left = remaining - sum(int_shares)
    if left > 0:
        remainders = sorted(
            [(raw_shares[idx] - int_shares[idx], idx) for idx in range(shot_count)],
            key=lambda x: (-x[0], x[1]),
        )
        for _, idx in remainders[:left]:
            allocations[idx] += 1

    return allocations


def build_joint_messages(frame_urls_by_shot: Dict[str, List[str]]) -> List[Dict]:
    shot_key_order = ", ".join(f'"{name}"' for name in frame_urls_by_shot.keys())
    system_prompt = (
        "You are a movie shot captioning assistant. You will receive sampled frames from multiple shots"
        " in the same source folder."
        " Build one consistent character mapping across all shots: the same person must keep the same"
        " label and the same appearance description whenever appearance is unchanged."
        " Use at most three major characters, labeled as [character1], [character2], and [character3]."
        " Output must be one valid JSON object only, with no Markdown fences and no extra explanation."
        " Every string value must be valid JSON: escape any internal double quotes as backslash-double-quote,"
        " or avoid double quotes entirely inside strings (use single quotes for quoted speech if needed)."
    )

    user_content: List[Dict] = [
        {
            "type": "text",
            "text": (
                "You will receive several groups of sampled frames, one group per shot, in this order: "
                f"{shot_key_order}. "
                "Return strict JSON exactly in this shape: "
                '{"global_prompt":"...",'
                '"character_bindings":{"character1":"...","character2":"...","character3":"..."},'
                '"shots":[{"file":"exact_filename.mp4","prompt":"..."}, ...]} '
                "The shots array must have one object per group above, in the same order, with file equal"
                " to that shot's filename (as shown in the frame section headers). "
                "Requirements:"
                " 1) Write global_prompt as a concise overall description for this folder."
                " 2) Each per-shot prompt: character appearance and action, then scene/environment, then camera language."
                " 3) Do NOT use labels like scene: or camera: inside prompt text."
                " 4) Camera description: plain language only, no focal length, aperture, ISO, or shutter numbers."
                " 5) Mention characters like '[character1] with blond hair and in a red T-shirt, ...'."
                " 6) If appearance is unchanged across shots, keep the same appearance wording."
                " 7) English only."
                " 8) Each string value must be a single line (no raw newline characters inside JSON strings)."
                " 9) Do not include trailing commas. Do not wrap the JSON in code fences."
            ),
        }
    ]
    for shot_name, urls in frame_urls_by_shot.items():
        user_content.append({"type": "text", "text": f"[Start of frames for {shot_name}]"})
        for data_url in urls:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                }
            )
        user_content.append({"type": "text", "text": f"[End of frames for {shot_name}]"})

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def post_chat_completion(
    api_base: str,
    api_key: str,
    model: str,
    messages: List[Dict],
    max_tokens: int,
    temperature: float,
    timeout_seconds: int,
    retry: int,
    response_format_json: bool = False,
) -> str:
    api_base = api_base.rstrip("/")
    url = f"{api_base}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if response_format_json:
        payload["response_format"] = {"type": "json_object"}
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    last_err: Optional[str] = None
    for attempt in range(1, retry + 1):
        try:
            request = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(request, timeout=timeout_seconds) as resp:
                resp_text = resp.read().decode("utf-8")
            data = json.loads(resp_text)
            text = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if not text:
                raise RuntimeError("Empty caption returned.")
            return text
        except urllib.error.HTTPError as err:
            detail = ""
            try:
                detail = err.read().decode("utf-8", errors="replace").strip()
            except Exception:
                pass
            last_err = f"HTTP Error {err.code}: {err.reason}"
            if detail:
                # DashScope / OpenAI-style errors are JSON; keep enough to debug 400s.
                last_err += f" | {detail[:4000]}"
            if attempt < retry:
                time.sleep(min(2.0 * attempt, 6.0))
            else:
                break
        except (urllib.error.URLError, TimeoutError, RuntimeError) as err:
            last_err = str(err)
            if attempt < retry:
                time.sleep(min(2.0 * attempt, 6.0))
            else:
                break
    raise RuntimeError(f"Chat completion failed after {retry} attempts: {last_err}")


def normalize_caption(text: str) -> str:
    return text.replace("\n", " ").strip()


def strip_markdown_code_fences(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def extract_first_json_object(text: str) -> str:
    text = strip_markdown_code_fences(text)
    start = text.find("{")
    if start == -1:
        raise RuntimeError("Model output does not contain a JSON object.")
    decoder = json.JSONDecoder()
    try:
        _, end = decoder.raw_decode(text[start:])
        return text[start : start + end]
    except json.JSONDecodeError:
        pass
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise RuntimeError("Unbalanced braces in model JSON output.")


def repair_json_light(blob: str) -> str:
    blob = blob.strip()
    blob = re.sub(r",\s*}", "}", blob)
    blob = re.sub(r",\s*]", "]", blob)
    return blob


def parse_joint_data(data: dict) -> Tuple[str, Dict[str, str], Dict[str, str]]:
    if not isinstance(data, dict):
        raise RuntimeError("Parsed JSON is not an object.")

    global_prompt = normalize_caption(str(data.get("global_prompt", "")).strip())
    if not global_prompt:
        raise RuntimeError("Missing global_prompt in JSON response.")

    bindings_raw = data.get("character_bindings", {})
    if isinstance(bindings_raw, dict):
        bindings = {
            str(k): str(v).replace("\n", " ").strip()
            for k, v in bindings_raw.items()
            if str(k).startswith("character") and str(v).strip()
        }
    else:
        bindings = {}

    shots_raw = data.get("shots")
    shot_prompts: Dict[str, str] = {}
    if isinstance(shots_raw, list):
        for item in shots_raw:
            if not isinstance(item, dict):
                continue
            key = str(
                item.get("file")
                or item.get("filename")
                or item.get("name")
                or item.get("shot")
                or ""
            ).strip()
            value = item.get("prompt")
            if value is None:
                value = item.get("caption", "")
            value = normalize_caption(str(value).strip())
            if key and value:
                shot_prompts[key] = value
    elif isinstance(shots_raw, dict):
        for k, v in shots_raw.items():
            key = str(k).strip()
            value = normalize_caption(str(v).strip())
            if key and value:
                shot_prompts[key] = value
    else:
        raise RuntimeError("Missing shots array/object in JSON response.")
    if not shot_prompts:
        raise RuntimeError("No per-shot prompts found in JSON response.")
    return global_prompt, bindings, shot_prompts


def parse_joint_response(text: str) -> Tuple[str, Dict[str, str], Dict[str, str]]:
    blob = extract_first_json_object(text)
    last_err: Optional[Exception] = None
    for candidate in (blob, repair_json_light(blob)):
        try:
            data = json.loads(candidate)
            return parse_joint_data(data)
        except json.JSONDecodeError as err:
            last_err = err
            continue
    raise RuntimeError(str(last_err) if last_err else "Failed to parse model JSON.")


def build_json_fix_messages(raw_text: str) -> List[Dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are a strict JSON repair assistant. "
                "Repair malformed JSON to valid JSON only. "
                "Do not add Markdown fences or any explanation. "
                "Escape double quotes inside string values with backslash. "
                'Use shape: {"global_prompt":"...","character_bindings":{...},"shots":[{"file":"...","prompt":"..."}]}'
            ),
        },
        {
            "role": "user",
            "content": (
                "Fix the following output into valid JSON while preserving meaning. "
                'Required keys: "global_prompt" (string), "character_bindings" (object), '
                '"shots" (array of objects with string keys file and prompt).\n'
                "Malformed output:\n"
                f"{raw_text}"
            ),
        },
    ]


def try_fix_and_parse_response(
    raw_response: str,
    api_base: str,
    api_key: str,
    model: str,
    timeout_seconds: int,
    api_retry: int,
    json_fix_retry: int,
    json_fix_max_tokens: int,
    response_format_json: bool,
) -> Tuple[str, Dict[str, str], Dict[str, str]]:
    last_err = "unknown json fix error"
    for fix_attempt in range(1, json_fix_retry + 1):
        try:
            fixed_text = post_chat_completion(
                api_base=api_base,
                api_key=api_key,
                model=model,
                messages=build_json_fix_messages(raw_response),
                max_tokens=json_fix_max_tokens,
                temperature=0.0,
                timeout_seconds=timeout_seconds,
                retry=api_retry,
                response_format_json=response_format_json,
            )
            return parse_joint_response(fixed_text)
        except Exception as err:
            last_err = str(err)
            if fix_attempt < json_fix_retry:
                time.sleep(0.3)
            else:
                break
    raise RuntimeError(f"JSON repair failed after {json_fix_retry} attempts: {last_err}")


def generate_joint_caption_with_retry(
    api_base: str,
    api_key: str,
    model: str,
    messages: List[Dict],
    max_tokens: int,
    temperature: float,
    timeout_seconds: int,
    api_retry: int,
    response_retry: int,
    json_fix_retry: int,
    json_fix_max_tokens: int,
    response_format_json: bool,
    sleep_between_calls: float,
) -> Tuple[str, Dict[str, str], Dict[str, str]]:
    last_err = "unknown error"
    use_json_format = response_format_json
    for attempt in range(1, response_retry + 1):
        try:
            raw_response = post_chat_completion(
                api_base=api_base,
                api_key=api_key,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                retry=api_retry,
                response_format_json=use_json_format,
            )
            try:
                return parse_joint_response(raw_response)
            except Exception as parse_err:
                if json_fix_retry > 0:
                    print(f"    JSON parse failed, trying repair API: {parse_err}")
                    return try_fix_and_parse_response(
                        raw_response=raw_response,
                        api_base=api_base,
                        api_key=api_key,
                        model=model,
                        timeout_seconds=timeout_seconds,
                        api_retry=api_retry,
                        json_fix_retry=json_fix_retry,
                        json_fix_max_tokens=json_fix_max_tokens,
                        response_format_json=use_json_format,
                    )
                raise
        except RuntimeError as err:
            last_err = str(err)
            # Many multimodal gateways return 400 if response_format json_object is unsupported;
            # retry once without it before burning outer response_retry slots.
            if (
                use_json_format
                and response_format_json
                and "HTTP Error 400" in last_err
                and attempt == 1
            ):
                print(
                    "    API returned 400 with response_format json_object; retrying without it "
                    "(use --no_response_format_json to skip next time)."
                )
                use_json_format = False
                time.sleep(max(0.0, sleep_between_calls))
                continue
            if attempt < response_retry:
                print(
                    f"    API/request failed, retrying ({attempt}/{response_retry}): {last_err}"
                )
                time.sleep(max(0.0, sleep_between_calls))
            else:
                break
        except Exception as err:
            last_err = str(err)
            if attempt < response_retry:
                print(
                    f"    Unexpected error, retrying ({attempt}/{response_retry}): {last_err}"
                )
                time.sleep(max(0.0, sleep_between_calls))
            else:
                break
    raise RuntimeError(
        f"Failed to get valid JSON response after {response_retry} attempts: {last_err}"
    )


def process_one_folder(
    args: argparse.Namespace,
    api_key: str,
    source_folder: Path,
    idx: int,
    total: int,
    output_data: Dict[str, Dict[str, Any]],
    state_data: Dict[str, Dict[str, str]],
    io_lock: threading.Lock,
) -> str:
    """
    Caption one source folder. Mutates output_data/state_data and saves JSON under io_lock.
    Returns 'success' or 'failed'.
    """
    source_key = str(source_folder)
    video_name = source_folder.name
    output_parent_dir = args.output_json.parent
    folder_caption_path = output_parent_dir / f"{video_name}.json"

    shot_paths = resolve_shot_paths(source_folder, video_names=None)
    if not shot_paths:
        with io_lock:
            state_data[source_key] = {"status": "failed", "reason": "shot_files_not_found"}
            save_json(args.state_json, state_data)
        print(f"[{idx}/{total}] FAIL {source_key}: no shot videos found")
        return "failed"

    try:
        frame_counts = [get_video_frame_count(p) for p in shot_paths]
        allocations = distribute_frame_samples(frame_counts, args.total_frame_samples)
        frame_urls_by_shot: Dict[str, List[str]] = {}
        for shot_path, alloc in zip(shot_paths, allocations):
            urls = sample_shot_frames(
                shot_path, alloc, args.target_frame_height, args.jpeg_quality
            )
            if urls:
                frame_urls_by_shot[shot_path.name] = urls
        if not frame_urls_by_shot:
            raise RuntimeError("Cannot read sample frames from shot video(s).")

        global_prompt, character_bindings, shot_prompts = generate_joint_caption_with_retry(
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
        time.sleep(args.sleep_between_calls)

        timing_fields = load_sample_json_timing_fields(source_folder)
        out_entry = {
            "source_folder": source_key,
            "video_name": video_name,
            "global_prompt": global_prompt,
            "shot_prompts": shot_prompts,
            "character_bindings": character_bindings,
            **timing_fields,
        }
        folder_payload = {
            "source_folder": source_key,
            "video_name": video_name,
            "model": args.model,
            "global_prompt": global_prompt,
            "character_bindings": character_bindings,
            "shot_prompts": shot_prompts,
            **timing_fields,
        }

        with io_lock:
            output_data[source_key] = out_entry
            save_json(args.output_json, output_data)
            save_json(folder_caption_path, folder_payload)
            state_data[source_key] = {"status": "done", "updated_at": int(time.time())}
            save_json(args.state_json, state_data)
        print(f"[{idx}/{total}] OK {source_key}")
        return "success"
    except Exception as err:
        with io_lock:
            state_data[source_key] = {"status": "failed", "reason": str(err)}
            save_json(args.state_json, state_data)
        print(f"[{idx}/{total}] FAIL {source_key}: {err}")
        return "failed"


def main() -> None:
    args = parse_args()
    input_root: Path = args.input_root

    api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing API key. Use --api_key or set DASHSCOPE_API_KEY/OPENAI_API_KEY.")

    output_data: Dict[str, Dict[str, Any]] = load_json_or_default(args.output_json, {})
    state_data: Dict[str, Dict[str, str]] = load_json_or_default(args.state_json, {})

    subdirs = list_input_subdirs(input_root)
    print(f"Found {len(subdirs)} subfolders in {input_root}")

    success_count = 0
    skip_count = 0
    fail_count = 0

    tasks: List[Tuple[int, Path]] = []
    for i, source_folder in enumerate(subdirs, start=1):
        source_key = str(source_folder)
        video_name = source_folder.name
        output_parent_dir = args.output_json.parent
        folder_caption_path = output_parent_dir / f"{video_name}.json"

        if folder_caption_path.exists() and not args.overwrite:
            skip_count += 1
            if i % 20 == 0:
                print(f"[{i}/{len(subdirs)}] Skip existing folder caption: {source_key}")
            continue

        already_done = (
            source_key in output_data
            and "global_prompt" in output_data[source_key]
            and "shot_prompts" in output_data[source_key]
            and state_data.get(source_key, {}).get("status") == "done"
        )
        if already_done and not args.overwrite:
            skip_count += 1
            if i % 20 == 0:
                print(f"[{i}/{len(subdirs)}] Skip done: {source_key}")
            continue

        tasks.append((i, source_folder))

    workers = max(1, int(args.workers))
    io_lock = threading.Lock()

    if workers == 1:
        for idx, source_folder in tasks:
            outcome = process_one_folder(
                args,
                api_key,
                source_folder,
                idx,
                len(subdirs),
                output_data,
                state_data,
                io_lock,
            )
            if outcome == "success":
                success_count += 1
            else:
                fail_count += 1
    else:
        print(f"Using {workers} parallel workers for {len(tasks)} folder(s).")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(
                    process_one_folder,
                    args,
                    api_key,
                    source_folder,
                    idx,
                    len(subdirs),
                    output_data,
                    state_data,
                    io_lock,
                ): (idx, source_folder)
                for idx, source_folder in tasks
            }
            for future in as_completed(future_to_idx):
                try:
                    outcome = future.result()
                except Exception as err:
                    idx, source_folder = future_to_idx[future]
                    print(f"[{idx}/{len(subdirs)}] FAIL {source_folder}: worker error: {err}")
                    fail_count += 1
                    continue
                if outcome == "success":
                    success_count += 1
                else:
                    fail_count += 1

    print("\nCaptioning complete.")
    print(f"Success: {success_count}")
    print(f"Skipped(done): {skip_count}")
    print(f"Failed: {fail_count}")
    print(f"Output JSON: {args.output_json}")
    print(f"State JSON: {args.state_json}")


if __name__ == "__main__":
    main()
