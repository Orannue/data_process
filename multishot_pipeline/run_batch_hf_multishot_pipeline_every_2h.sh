#!/usr/bin/env bash
set -euo pipefail

# Run run_batch_hf_multishot_pipeline.sh once every 2 hours.
# The batch script is resumable:
# - existing .7z files are not downloaded again
# - extracted archives are skipped
# - movies with valid merged.mp4 outputs are skipped
# - failed movies are retried only if RETRY_FAILED=1 in the batch script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BATCH_SCRIPT="${SCRIPT_DIR}/run_batch_hf_multishot_pipeline.sh"

# Optional: seconds between runs. 7200 seconds = 2 hours.
INTERVAL_SECONDS="${INTERVAL_SECONDS:-7200}"

# Optional: number of rounds. Empty means forever. Use MAX_ROUNDS=1 for a test.
MAX_ROUNDS="${MAX_ROUNDS:-}"

# Optional: loop log directory.
LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/batch_loop_logs}"

mkdir -p "${LOG_ROOT}"

round=0
while true; do
  round=$((round + 1))
  timestamp="$(date '+%Y%m%d_%H%M%S')"
  log_file="${LOG_ROOT}/batch_${timestamp}.log"

  echo "[$(date '+%F %T')] round=${round} start; log=${log_file}"

  set +e
  bash "${BATCH_SCRIPT}" 2>&1 | tee "${log_file}"
  exit_code="${PIPESTATUS[0]}"
  set -e

  echo "[$(date '+%F %T')] round=${round} exit_code=${exit_code}"

  if [[ -n "${MAX_ROUNDS}" && "${round}" -ge "${MAX_ROUNDS}" ]]; then
    echo "MAX_ROUNDS=${MAX_ROUNDS}; exiting."
    exit "${exit_code}"
  fi

  echo "Sleeping ${INTERVAL_SECONDS}s before next Hugging Face check..."
  sleep "${INTERVAL_SECONDS}"
done
