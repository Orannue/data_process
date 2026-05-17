import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI


SYSTEM_PROMPT_TEMPLATE = """Act as an expert AI video generation prompt engineer specialized in multi-shot and camera-controllable video generation. Your task is to generate a completely original, creative, and action-oriented video concept based on the parameters provided below. Then, you will render this sequence into two distinct JSON formats based on strict syntax rules.

### Sequence Parameters:
* Number of Shots: [NUMBER_OF_SHOTS]
* Total Characters in the Story: [TOTAL_CHARACTERS]
* Max Characters Per Shot: [MAX_CHARACTERS_PER_SHOT]

### CRITICAL CONSTRAINT: Character Orientation
In almost every shot description (both action and camera settings), you MUST prioritize keeping the primary subject(s) front-facing with clear, unobstructed facial visibility. The composition must focus on capturing their frontal view during the action.

### Task Workflow:
1.  **Generate Random Concept:** Imagine a dynamic, visually engaging sequence matching the `Sequence Parameters` above. You must invent exactly [TOTAL_CHARACTERS] distinct characters (labeled [character1], [character2], etc.). However, ensure that NO MORE THAN [MAX_CHARACTERS_PER_SHOT] characters appear together in any single shot.
2.  **Format 1 Generation (Global + Per-Shot):** Break the context into global variables and local actions.
3.  **Format 2 Generation (Standalone Per-Shot):** Re-integrate the context into self-contained strings using specified formulas.

---

### Syntax and Structural Rules:

#### Format 1: Global + Per-Shot (Separated Context)
* **`global_caption`**: A single string containing: `[bracketed_character_tags] identity and detailed appearance descriptions for ALL [TOTAL_CHARACTERS] characters + Full Scene description (environment, lighting, detailed atmosphere, artistic style, color temperature)`.
    * **Strict Negative Rule:** Do NOT include any specific actions or camera movements here.
* **`prompt`**: An array of [NUMBER_OF_SHOTS] strings representing the shots in chronological order.
    * Each string MUST start with `"[shot cut]"`.
    * Content: `Specific character action(s) for the characters present in this shot + detailed framing/Camera settings (movement, angle, lens specs, DOF)`.
    * **Strict Negative Rule:** Do NOT repeat the scene description or full character appearance here. Refer to them only by bracketed tags (e.g., "[character1]").

#### Format 2: Standalone Per-Shot (Integrated Context)
* **`prompts`**: An array of [NUMBER_OF_SHOTS] strings, one per shot, fully self-contained in chronological order.
* **Strict Composition Order:** Every string in this array MUST follow this exact semantic order:
    `[character_name], [detailed appearance description seamlessly integrated with the current action], [frame positioning], [environment, lighting, atmosphere], [camera movement, angle, detailed lens specs, depth of field, MANDATORY instruction to keep faces front-facing and visible].`
    * You do NOT need to explicitly write section labels like `Scene:` or `Camera:`.
    * Keep the content naturally written, but keep the order strictly consistent.
    *(Note: If multiple characters are in the shot, adapt the start naturally: "[character1], [appearance], and [character2], [appearance], [joint action], positioned...")*
* **Strict Negative Rule:** Do NOT make the character's general appearance a standalone sentence (e.g., Avoid "[character1] is a doctor. He runs."). It must be part of the continuous flow (e.g., "[character1], a doctor in a white coat running, positioned...").

#### General Strict Rules for Both Formats:
1.  **Tags:** Always use bracketed tags for characters (e.g., `[character1]`, `[character2]`).
2.  **Continuity:** Maintain perfect logical continuity between shots (e.g., if shot 1 ends on the ground, shot 2 must start from the ground).
3.  **Scene Purity:** In Format 2, the scene-related segment must ONLY contain environmental/atmospheric details. Do not put characters here.
4.  **Camera Purity:** In both formats, the camera-related segment must ONLY contain cinematographic instructions and subject orientation instructions (front-facing). Do not put actions here.

---

### Expected Output Structure (JSON ONLY):

{
  "random_concept_summary": "A brief summary of the action sequence you invented, mentioning how the characters interact across the shots.",
  "format_1": {
    "global_caption": "[character1] is... [character2] is... [Scene description]...",
    "prompt": [
      "[shot cut] [character1] sprints... [Camera details focusing on front-facing view]...",
      // ... continue for [NUMBER_OF_SHOTS] shots
    ]
  },
  "format_2": {
    "prompts": [
      "[character1], [appearance + action], positioned..., [scene description], [camera description focusing on front-facing view]...",
      // ... continue for [NUMBER_OF_SHOTS] shots
    ]
  }
}

### Execution Command:
Generate the random concept and output the JSON now following all rules and parameters above. Do not ask for concept input or output any conversational text outside the JSON.
"""


def fill_system_prompt(number_of_shots: int, total_characters: int, max_characters_per_shot: int) -> str:
    prompt = SYSTEM_PROMPT_TEMPLATE
    prompt = prompt.replace("[NUMBER_OF_SHOTS]", str(number_of_shots))
    prompt = prompt.replace("[TOTAL_CHARACTERS]", str(total_characters))
    prompt = prompt.replace("[MAX_CHARACTERS_PER_SHOT]", str(max_characters_per_shot))
    return prompt


def extract_json(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("Empty model response")

    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, flags=re.DOTALL)
    if fence_match:
        return json.loads(fence_match.group(1))

    object_match = re.search(r"(\{.*\})", stripped, flags=re.DOTALL)
    if object_match:
        return json.loads(object_match.group(1))

    raise ValueError("Could not parse JSON from model response")


def validate_shape(result: Dict[str, Any], number_of_shots: int) -> None:
    if "format_1" not in result or "format_2" not in result:
        raise ValueError("Missing format_1 or format_2")

    fmt1 = result["format_1"]
    fmt2 = result["format_2"]
    if not isinstance(fmt1, dict) or not isinstance(fmt2, dict):
        raise ValueError("format_1/format_2 must be objects")

    p1 = fmt1.get("prompt")
    p2 = fmt2.get("prompts")
    if not isinstance(p1, list) or not isinstance(p2, list):
        raise ValueError("format_1.prompt and format_2.prompts must be lists")
    if len(p1) != number_of_shots or len(p2) != number_of_shots:
        raise ValueError(
            f"Shot count mismatch: format_1={len(p1)}, format_2={len(p2)}, expected={number_of_shots}"
        )


def single_generation(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    client = OpenAI(base_url=base_url, api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "Generate the random concept and output the JSON now following all rules and parameters above. Do not output any text outside JSON.",
            },
        ],
    )
    content = completion.choices[0].message.content or ""
    if isinstance(content, list):
        content = "\n".join([str(x) for x in content])
    return extract_json(str(content))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multi-shot prompts via API.")
    parser.add_argument("--base_url", type=str, default="https://api.xi-ai.cn/v1")
    parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--number_of_shots", type=int, required=True)
    parser.add_argument("--total_characters", type=int, required=True)
    parser.add_argument("--max_characters_per_shot", type=int, required=True)
    parser.add_argument("--num_generations", type=int, default=1, help="How many independent generations to run.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--sleep_between", type=float, default=0.0)
    parser.add_argument("--output", type=str, default="generated_prompts.json")
    parser.add_argument("--quiet", action="store_true", help="Disable intermediate progress logs.")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Missing API key. Pass --api_key or set OPENAI_API_KEY.")
    if args.num_generations < 1:
        raise ValueError("--num_generations must be >= 1")

    system_prompt = fill_system_prompt(
        args.number_of_shots,
        args.total_characters,
        args.max_characters_per_shot,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print(
            f"[start] model={args.model}, generations={args.num_generations}, shots={args.number_of_shots}, "
            f"characters={args.total_characters}, output={output_path}",
            flush=True,
        )

    all_results: List[Dict[str, Any]] = []
    for i in range(args.num_generations):
        last_error = None
        gen_start = time.time()
        if not args.quiet:
            print(f"[generation {i + 1}/{args.num_generations}] started", flush=True)
        for attempt in range(args.max_retries + 1):
            try:
                if not args.quiet:
                    print(
                        f"[generation {i + 1}/{args.num_generations}] attempt {attempt + 1}/{args.max_retries + 1}",
                        flush=True,
                    )
                result = single_generation(
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=args.model,
                    system_prompt=system_prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                validate_shape(result, args.number_of_shots)
                result["_meta"] = {
                    "generation_index": i,
                    "timestamp": int(time.time()),
                    "model": args.model,
                    "number_of_shots": args.number_of_shots,
                    "total_characters": args.total_characters,
                    "max_characters_per_shot": args.max_characters_per_shot,
                }
                all_results.append(result)
                if not args.quiet:
                    elapsed = time.time() - gen_start
                    summary = str(result.get("random_concept_summary", "")).strip().replace("\n", " ")
                    if len(summary) > 120:
                        summary = summary[:117] + "..."
                    print(
                        f"[generation {i + 1}/{args.num_generations}] success in {elapsed:.1f}s"
                        + (f" | summary: {summary}" if summary else ""),
                        flush=True,
                    )
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if not args.quiet:
                    print(
                        f"[generation {i + 1}/{args.num_generations}] attempt {attempt + 1} failed: {exc}",
                        flush=True,
                    )
                if attempt >= args.max_retries:
                    raise RuntimeError(
                        f"Generation {i} failed after {args.max_retries + 1} attempts: {exc}"
                    ) from exc
                time.sleep(1.0)

        if last_error is not None and len(all_results) <= i:
            raise RuntimeError(f"Generation {i} failed: {last_error}")

        if args.sleep_between > 0 and i < args.num_generations - 1:
            if not args.quiet:
                print(f"[wait] sleeping {args.sleep_between}s before next generation", flush=True)
            time.sleep(args.sleep_between)

    # Write standard JSON when output ends with .json; keep JSONL compatibility for .jsonl
    if not args.quiet:
        print(f"[write] saving {len(all_results)} item(s) to {output_path}", flush=True)
    if output_path.suffix.lower() == ".jsonl":
        with output_path.open("w", encoding="utf-8") as f:
            for item in all_results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"Done. Generated {len(all_results)} items -> {output_path}")


if __name__ == "__main__":
    main()
