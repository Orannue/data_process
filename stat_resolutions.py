import os
import json
import subprocess
from math import fabs
from collections import defaultdict

ROOT = r"H:\dataset\dataset_500\all_new"
TARGET_W, TARGET_H = 832, 480
TARGET_AR = TARGET_W / TARGET_H
TARGET_AREA = TARGET_W * TARGET_H

def get_resolution_ffprobe(path: str):
    # 只取第一个 video stream 的宽高
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        path,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        info = json.loads(out)
        stream = info["streams"][0]
        return int(stream["width"]), int(stream["height"])
    except Exception as e:
        print(f"读取分辨率失败: {path} ({e})")
        return None

def score_resolution(w: int, h: int):
    # 先比纵横比差异，再用面积差作为次级排序
    ar = w / h
    area = w * h
    ar_diff = fabs(ar - TARGET_AR)
    area_diff = fabs(area - TARGET_AREA)
    return (ar_diff, area_diff)

def main():
    results = []
    # 所有文件的原始统计
    for root, _, files in os.walk(ROOT):
        for name in files:
            if not name.lower().endswith(".mp4"):
                continue
            full = os.path.join(root, name)
            res = get_resolution_ffprobe(full)
            if not res:
                continue
            w, h = res
            s = score_resolution(w, h)
            results.append({
                "path": full,
                "width": w,
                "height": h,
                "aspect_ratio": round(w / h, 6),
                "area": w * h,
                "score": s,
            })
            print(f"{full} -> {w}x{h}, score={s}")

    if not results:
        print("没有找到任何 .mp4 或无法读取分辨率")
        return

    # 统计每种分辨率出现次数
    groups = defaultdict(lambda: {"count": 0, "ar": None, "area": None, "score": None})
    for r in results:
        key = (r["width"], r["height"])
        g = groups[key]
        g["count"] += 1
        g["ar"] = r["aspect_ratio"]
        g["area"] = r["area"]
        g["score"] = r["score"]

    total_files = len(results)
    print(f"\n总视频数: {total_files}")
    print(f"不同分辨率种类: {len(groups)}")

    # 把每种分辨率整理出来，按出现次数排序
    grouped_list = []
    for (w, h), info in groups.items():
        ar_diff, area_diff = info["score"]
        grouped_list.append({
            "width": w,
            "height": h,
            "count": info["count"],
            "aspect_ratio": info["ar"],
            "area": info["area"],
            "ar_diff": ar_diff,
            "area_diff": area_diff,
        })

    # 先按出现次数降序，再按离 832x480 的接近程度排序，方便人工观察
    grouped_list.sort(key=lambda x: (-x["count"], x["ar_diff"], x["area_diff"]))

    print("\n========== 所有分辨率分布（按数量排序，前 20 个） ==========")
    print("宽x高\t数量\t纵横比\t与832x480的AR差\t面积差")
    for g in grouped_list[:20]:
        print(f"{g['width']}x{g['height']}\t{g['count']}\t{g['aspect_ratio']:.6f}\t{g['ar_diff']:.6f}\t{int(g['area_diff'])}")

    # 自动选择一个“主流且接近 832x480” 的目标分辨率：
    #  - 优先数量多
    #  - 同数量下优先纵横比接近，其次面积接近
    grouped_list_for_choice = sorted(
        grouped_list,
        key=lambda x: (x["ar_diff"], x["area_diff"], -x["count"])
    )

    # 可以限制一下“不要和 832x480 差太远”，比如 AR 差 < 0.2
    AR_DIFF_MAX = 0.2
    candidate = None
    for g in grouped_list_for_choice:
        if g["ar_diff"] <= AR_DIFF_MAX:
            candidate = g
            break
    if candidate is None:
        # 如果所有分辨率纵横比都差得比较远，就退而求其次选整体最接近的
        candidate = grouped_list_for_choice[0]

    print("\n========== 推荐的统一目标分辨率（按主流且接近 832x480 选择） ==========")
    print(f"建议目标分辨率: {candidate['width']}x{candidate['height']}")
    print(f"该分辨率视频数量: {candidate['count']} / {total_files}")
    print(f"纵横比: {candidate['aspect_ratio']:.6f}")
    print(f"纵横比与 832x480 差值: {candidate['ar_diff']:.6f}")
    print(f"面积与 832x480 像素差值: {int(candidate['area_diff'])}")
    print("\n接下来你可以：")
    print("1）用上面列出的分辨率分布，人工确认是否接受这个推荐值；")
    print("2）然后写一个统一裁剪/缩放脚本，把所有视频转换成该分辨率（必要时先按比例裁剪，再缩放）。")

    # 如果你想把完整统计结果保存出来：
    # import csv
    # with open("video_resolutions.csv", "w", newline="", encoding="utf-8") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["path", "width", "height", "aspect_ratio", "area"])
    #     for r in results:
    #         writer.writerow([r["path"], r["width"], r["height"], r["aspect_ratio"], r["area"]])

if __name__ == "__main__":
    main()