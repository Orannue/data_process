import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# 输入视频根目录
INPUT_ROOT = r"F:\dataset\out_22_21_cropped"
# 输出视频根目录
OUTPUT_ROOT = r"F:\dataset\out_22_21_cropped_832x480"

# ffmpeg 可执行文件名（如果不在 PATH 中，就写绝对路径）
FFMPEG_BIN = "ffmpeg"

# 并行任务数：每个任务会起一个 ffmpeg 进程，过大易占满 CPU/磁盘；设为 1 即串行
MAX_WORKERS = max(1, min(4, (os.cpu_count()//2 or 4)))

_print_lock = threading.Lock()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def process_one(input_path: str, output_path: str) -> bool:
    ensure_dir(os.path.dirname(output_path))

    # 使用表达式自动中心裁剪到 16:9，再缩放到 832x480
    vf = (
        "crop='if(gte(iw/ih,16/9),ih*16/9,iw)':'if(gte(iw/ih,16/9),ih,iw*9/16)',"
        "scale=832:480"
    )

    cmd = [
        FFMPEG_BIN,
        "-y",              # 覆盖输出
        "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264", # 或者 copy/其他编码，建议重编码为 h264
        "-preset", "fast",
        "-crf", "18",      # 质量控制，可自行调整
        "-c:a", "aac",     # 音频编码；原来如果是 aac 也可以用 -c:a copy
        "-b:a", "128k",
        output_path,
    ]

    with _print_lock:
        print("Processing:")
        print("  IN :", input_path)
        print("  OUT:", output_path)
    proc = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        err_tail = (proc.stderr or "").strip()
        if len(err_tail) > 1200:
            err_tail = err_tail[-1200:]
        with _print_lock:
            print(f"  失败: ffmpeg exit {proc.returncode}")
            if err_tail:
                print(err_tail)
        return False
    return True


def collect_jobs():
    jobs = []
    for root, _, files in os.walk(INPUT_ROOT):
        for name in files:
            if not name.lower().endswith(".mp4"):
                continue
            in_full = os.path.join(root, name)
            rel_path = os.path.relpath(in_full, INPUT_ROOT)
            out_full = os.path.join(OUTPUT_ROOT, rel_path)
            jobs.append((in_full, out_full))
    return jobs


def main():
    jobs = collect_jobs()
    print(f"共 {len(jobs)} 个 mp4，并行数 MAX_WORKERS={MAX_WORKERS}")

    ok = fail = 0
    if MAX_WORKERS == 1:
        for in_full, out_full in jobs:
            if process_one(in_full, out_full):
                ok += 1
            else:
                fail += 1
    else:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_paths = {
                executor.submit(process_one, inf, outf): (inf, outf) for inf, outf in jobs
            }
            for future in as_completed(future_to_paths):
                try:
                    if future.result():
                        ok += 1
                    else:
                        fail += 1
                except Exception as e:
                    inf, outf = future_to_paths[future]
                    with _print_lock:
                        print(f" 异常 {inf} -> {outf}: {e}")
                    fail += 1

    print("全部处理完成。")
    print(f"成功: {ok}，失败: {fail}")

if __name__ == "__main__":
    main()