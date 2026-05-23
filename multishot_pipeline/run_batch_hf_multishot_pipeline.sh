#!/usr/bin/env bash
set -euo pipefail

# =========================
# 服务器运行参数
# 约定：1 表示开启，0 表示关闭；空字符串 "" 表示使用脚本默认逻辑。
# =========================

# [必填] Hugging Face 数据集仓库 ID。当前就是你的 raw data 仓库。
REPO_ID="Orannue/moviedataset"

# [可选] Hugging Face 仓库版本、分支或 commit。通常保持 main。 
REVISION="main"

# [可选] Hugging Face token。不要写死在脚本里；需要权限时先 export HF_TOKEN="你的token"。
HF_TOKEN="${HF_TOKEN:-}"

# [可选] 需要下载和解压的压缩包匹配规则。默认处理所有 .7z。
ARCHIVE_PATTERN="*.7z"

# [必填] 场景标注 JSON。相对路径会按项目根目录解析，也可以填绝对路径。
SCENE_JSON="movies_scenes.json"

# [必填] Hugging Face 下载下来的 .7z 存放目录。解压成功后默认会删除这里的 .7z。
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-data_raw/moviedataset_archives}"

# [必填] .7z 解压后的 raw movie 数据目录。脚本会从这里发现 movie_id。
EXTRACT_ROOT="${EXTRACT_ROOT:-data_raw/moviedataset_extracted}"

# [必填] 中间结果目录。shots、characters、samples、cropped_samples、日志都会写这里。
WORK_ROOT="${WORK_ROOT:-movie_multishot_work}"

# [可选] 额外最终输出目录。MERGE_IN_PLACE=1 时不会用它保存 merged.mp4。
FINAL_ROOT="${FINAL_ROOT:-movie_multishot_final}"

# [可选] batch 状态文件路径。留空时默认写到 WORK_ROOT/_batch_pipeline_state.json。
STATE_FILE="${STATE_FILE:-}"

# [可选] 复用旧 work root 里的 split/character 结果，同时把新 sample/crop/merge 写到 WORK_ROOT。
REUSE_WORK_ROOT="${REUSE_WORK_ROOT:-}"

# [可选] Inception/FaceNet 权重缓存目录。留空时默认写到当前运行目录的 model_cache。
MODEL_CACHE_DIR=""

# [可选] Python 命令。conda 环境里一般保持 python；也可以填 /path/to/python。
PYTHON_BIN="${PYTHON_BIN:-python}"

# [必填] 7z 命令。确保服务器能直接运行 7z；否则填绝对路径。
SEVENZIP_BIN="${SEVENZIP_BIN:-7z}"

# =========================
# 并行与吞吐参数
# =========================

# [可选] 同时下载几个 .7z。网络或磁盘压力大时调小；下载慢时可调到 2-4。
DOWNLOAD_WORKERS=5

# [可选] 同时解压几个 .7z。解压吃 CPU/IO，磁盘压力大时建议 1-2。
EXTRACT_WORKERS=2

# [必填] 同时处理几部电影。8 卡建议先用 2；CPU/IO 很强时可尝试 4。
MOVIE_WORKERS="${MOVIE_WORKERS:-4}"

# [可选] 每部电影的默认 scene worker 数。CHARACTER_WORKERS 留空时会用这个值。
SCENE_WORKERS="${SCENE_WORKERS:-4}"

# [可选] 每部电影 split shot 阶段的 CPU worker 数。scene 很多时可以大于 GPU 数。
SPLIT_WORKERS="${SPLIT_WORKERS:-8}"

# [可选] 每部电影 character_cluster 阶段的 GPU worker 数。留空表示使用 SCENE_WORKERS。
CHARACTER_WORKERS="${CHARACTER_WORKERS:-8}"

# [可选] 每部电影 build_multishot_samples 阶段的 CPU worker 数。
SAMPLE_WORKERS="${SAMPLE_WORKERS:-8}"

# [必填] 可用 GPU 列表。脚本会按 MOVIE_WORKERS 自动连续分组。
DEVICES="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7"

# [可选] 手动指定每个 movie worker 的 GPU 组，例如 "cuda:0,cuda:1;cuda:2,cuda:3"。留空自动分组。
DEVICE_GROUPS=""

# [可选] 单设备模式。使用 DEVICES 时这里留空。
DEVICE=""

# =========================
# 样本处理参数
# =========================

# [可选] 是否在 build_multishot_samples 阶段写原始 merged sample video。通常设 0。
WRITE_VIDEOS=0

# [可选] sample 生成策略。default 为旧逻辑；fixed-latent 为 2-3 shots / 22 latent / 85 frames 逻辑。
SAMPLE_BUILDER="${SAMPLE_BUILDER:-default}"

# [可选] fixed-latent sample 的总 latent frame 数。22 latent 对应 85 帧。
FIXED_SAMPLE_LATENT_FRAMES="${FIXED_SAMPLE_LATENT_FRAMES:-22}"

# [可选] fixed-latent sample 最少 shot 数。
FIXED_SAMPLE_MIN_SHOTS="${FIXED_SAMPLE_MIN_SHOTS:-2}"

# [可选] fixed-latent sample 最多 shot 数。
FIXED_SAMPLE_MAX_SHOTS="${FIXED_SAMPLE_MAX_SHOTS:-3}"

# [建议保持默认] merged.mp4 是否写回 sample 的 shot 文件夹。1 表示和 shot_0001.mp4 放一起。
MERGE_IN_PLACE="${MERGE_IN_PLACE:-1}"

# [可选] crop_sample_shots 的 latent 帧上限。
MAX_LATENT_FRAMES="${MAX_LATENT_FRAMES:-127}"

# [可选] 最终视频宽度。
TARGET_WIDTH="${TARGET_WIDTH:-832}"

# [可选] 最终视频高度。
TARGET_HEIGHT="${TARGET_HEIGHT:-480}"

# [可选] merge 阶段每个 sample 至少需要几个 clip。multishot 建议 2。
MIN_CLIPS="${MIN_CLIPS:-2}"

# [可选] merge/crop/resize 的 CPU 并行数。每个 movie worker 都会用这个值，别设太夸张。
MERGE_JOBS="${MERGE_JOBS:-8}"

# =========================
# 断点续跑、覆盖、清理参数
# =========================

# [建议保持默认] 本地已有 .7z 时跳过下载，避免重复下载。
SKIP_EXISTING_DOWNLOADS="${SKIP_EXISTING_DOWNLOADS:-1}"

# [可选] 强制重新下载 .7z。通常保持 0。
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"

# [可选] 跳过下载阶段。已有本地压缩包或已解压时可设 1。
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"

# [可选] 跳过解压阶段。EXTRACT_ROOT 已经准备好时可设 1。
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"

# [可选] 跳过处理阶段。只想下载和解压时设 1。
SKIP_PROCESS="${SKIP_PROCESS:-0}"

# [建议保持默认] 每个 archive 解压到独立子目录，避免不同压缩包文件名互相覆盖。
EXTRACT_PER_ARCHIVE="${EXTRACT_PER_ARCHIVE:-1}"

# [建议保持默认] 解压成功后自动删除本地 .7z，节省磁盘空间。
DELETE_ARCHIVE_AFTER_EXTRACT="${DELETE_ARCHIVE_AFTER_EXTRACT:-1}"

# [可选] 重新解压并覆盖已有解压 marker。通常保持 0。
OVERWRITE_EXTRACT="${OVERWRITE_EXTRACT:-0}"

# [可选] 覆盖已处理电影的输出。通常保持 0；需要全量重跑时设 1。
OVERWRITE_OUTPUTS="${OVERWRITE_OUTPUTS:-0}"

# [可选] 重新跑之前失败的电影。修复依赖或参数后建议保持 1。
RETRY_FAILED="${RETRY_FAILED:-1}"

# =========================
# 小规模测试参数
# =========================

# [可选] 只跑某一部电影，例如 "0001_American_Beauty"。留空表示不限制。
ONLY_MOVIE=""

# [可选] 最多处理几部电影。smoke test 可设 1 或 3；留空表示不限制。
MAX_MOVIES=""

# [可选] 最多下载几个 archive。smoke test 可设 1；留空表示不限制。
MAX_ARCHIVES=""

# =========================
# 高级透传参数
# =========================

# [可选] 透传给 run_scene_pipeline_parallel.py 的额外参数，例如 "--min-shot-seconds 2.0"。
PIPELINE_EXTRA_ARGS="${PIPELINE_EXTRA_ARGS:-}"

# [可选] 只跑部分 scene pipeline 阶段。全流程用 split,character,sample；只跑一个阶段可填 split / character / sample。
PIPELINE_STAGES="${PIPELINE_STAGES:-split,character,sample}"

# [可选] 跑 crop/merge 后处理阶段。auto 表示完整 scene pipeline 后自动跑 crop,merge；已有 sample 时可设 crop,merge。
POST_STAGES="${POST_STAGES:-auto}"

# [可选] 透传给 crop_sample_shots.py 的额外参数。
CROP_EXTRA_ARGS="${CROP_EXTRA_ARGS:-}"

# [可选] 透传给 merge_crop_resize_samples.py 的额外参数，例如 "--crf 20 --preset medium"。
MERGE_EXTRA_ARGS="${MERGE_EXTRA_ARGS:-}"

# [可选] 主循环轮询间隔，单位秒。
POLL_SECONDS=10

# [可选] 扫描 EXTRACT_ROOT 发现新电影的最小间隔，单位秒。目录很大时不要太小。
DISCOVER_INTERVAL_SECONDS=200

# [可选] 是否实时打印子进程输出。0 表示只写日志文件；1 表示同时打印到终端。
STREAM_SUBPROCESS_OUTPUT=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BATCH_SCRIPT="${SCRIPT_DIR}/batch_hf_multishot_pipeline.py"

if [[ "${SCENE_JSON}" = /* ]]; then
  SCENE_JSON_PATH="${SCENE_JSON}"
else
  SCENE_JSON_PATH="${REPO_ROOT}/${SCENE_JSON}"
fi

if [[ -z "${MODEL_CACHE_DIR}" ]]; then
  MODEL_CACHE_DIR="${PWD}/model_cache"
fi

cmd=(
  "${PYTHON_BIN}" "${BATCH_SCRIPT}"
  --repo-id "${REPO_ID}"
  --revision "${REVISION}"
  --archive-pattern "${ARCHIVE_PATTERN}"
  --scene-json "${SCENE_JSON_PATH}"
  --download-root "${DOWNLOAD_ROOT}"
  --extract-root "${EXTRACT_ROOT}"
  --work-root "${WORK_ROOT}"
  --final-root "${FINAL_ROOT}"
  --model-cache-dir "${MODEL_CACHE_DIR}"
  --sample-builder "${SAMPLE_BUILDER}"
  --fixed-sample-latent-frames "${FIXED_SAMPLE_LATENT_FRAMES}"
  --fixed-sample-min-shots "${FIXED_SAMPLE_MIN_SHOTS}"
  --fixed-sample-max-shots "${FIXED_SAMPLE_MAX_SHOTS}"
  --pipeline-dir "${SCRIPT_DIR}"
  --python "${PYTHON_BIN}"
  --sevenzip "${SEVENZIP_BIN}"
  --download-workers "${DOWNLOAD_WORKERS}"
  --extract-workers "${EXTRACT_WORKERS}"
  --movie-workers "${MOVIE_WORKERS}"
  --max-latent-frames "${MAX_LATENT_FRAMES}"
  --target-width "${TARGET_WIDTH}"
  --target-height "${TARGET_HEIGHT}"
  --min-clips "${MIN_CLIPS}"
  --merge-jobs "${MERGE_JOBS}"
  --pipeline-stages "${PIPELINE_STAGES}"
  --post-stages "${POST_STAGES}"
  --poll-seconds "${POLL_SECONDS}"
  --discover-interval-seconds "${DISCOVER_INTERVAL_SECONDS}"
)

if [[ -n "${HF_TOKEN:-}" ]]; then
  cmd+=(--hf-token "${HF_TOKEN}")
fi

if [[ -n "${REUSE_WORK_ROOT:-}" ]]; then
  cmd+=(--source-work-root "${REUSE_WORK_ROOT}")
fi

if [[ -n "${SCENE_WORKERS:-}" ]]; then
  cmd+=(--scene-workers "${SCENE_WORKERS}")
fi

if [[ -n "${SPLIT_WORKERS:-}" ]]; then
  cmd+=(--split-workers "${SPLIT_WORKERS}")
fi

if [[ -n "${CHARACTER_WORKERS:-}" ]]; then
  cmd+=(--character-workers "${CHARACTER_WORKERS}")
fi

if [[ -n "${SAMPLE_WORKERS:-}" ]]; then
  cmd+=(--sample-workers "${SAMPLE_WORKERS}")
fi

if [[ -n "${DEVICES:-}" ]]; then
  cmd+=(--devices "${DEVICES}")
fi

if [[ -n "${DEVICE_GROUPS:-}" ]]; then
  cmd+=(--device-groups "${DEVICE_GROUPS}")
fi

if [[ -n "${DEVICE:-}" ]]; then
  cmd+=(--device "${DEVICE}")
fi

if [[ "${WRITE_VIDEOS}" == "1" ]]; then
  cmd+=(--write-videos)
fi

if [[ "${MERGE_IN_PLACE}" == "1" ]]; then
  cmd+=(--merge-in-place)
else
  cmd+=(--no-merge-in-place)
fi

if [[ "${SKIP_EXISTING_DOWNLOADS}" == "1" ]]; then
  cmd+=(--skip-existing-downloads)
else
  cmd+=(--no-skip-existing-downloads)
fi

if [[ "${FORCE_DOWNLOAD}" == "1" ]]; then
  cmd+=(--force-download)
fi

if [[ "${SKIP_DOWNLOAD}" == "1" ]]; then
  cmd+=(--skip-download)
fi

if [[ "${SKIP_EXTRACT}" == "1" ]]; then
  cmd+=(--skip-extract)
fi

if [[ "${SKIP_PROCESS}" == "1" ]]; then
  cmd+=(--skip-process)
fi

if [[ "${EXTRACT_PER_ARCHIVE}" == "1" ]]; then
  cmd+=(--extract-per-archive)
else
  cmd+=(--no-extract-per-archive)
fi

if [[ "${DELETE_ARCHIVE_AFTER_EXTRACT}" == "1" ]]; then
  cmd+=(--delete-archive-after-extract)
else
  cmd+=(--no-delete-archive-after-extract)
fi

if [[ "${OVERWRITE_EXTRACT}" == "1" ]]; then
  cmd+=(--overwrite-extract)
fi

if [[ "${OVERWRITE_OUTPUTS}" == "1" ]]; then
  cmd+=(--overwrite)
fi

if [[ "${RETRY_FAILED}" == "1" ]]; then
  cmd+=(--retry-failed)
fi

if [[ -n "${ONLY_MOVIE}" ]]; then
  cmd+=(--only-movie "${ONLY_MOVIE}")
fi

if [[ -n "${MAX_MOVIES}" ]]; then
  cmd+=(--max-movies "${MAX_MOVIES}")
fi

if [[ -n "${MAX_ARCHIVES}" ]]; then
  cmd+=(--max-archives "${MAX_ARCHIVES}")
fi

if [[ -n "${STATE_FILE}" ]]; then
  cmd+=(--state-file "${STATE_FILE}")
fi

if [[ -n "${PIPELINE_EXTRA_ARGS}" ]]; then
  cmd+=(--pipeline-extra-args "${PIPELINE_EXTRA_ARGS}")
fi

if [[ -n "${CROP_EXTRA_ARGS}" ]]; then
  cmd+=(--crop-extra-args "${CROP_EXTRA_ARGS}")
fi

if [[ -n "${MERGE_EXTRA_ARGS}" ]]; then
  cmd+=(--merge-extra-args "${MERGE_EXTRA_ARGS}")
fi

if [[ "${STREAM_SUBPROCESS_OUTPUT}" == "1" ]]; then
  cmd+=(--stream-subprocess-output)
fi

printf 'Running command:\n'
printf '  %q' "${cmd[@]}"
printf '\n\n'

exec "${cmd[@]}"
