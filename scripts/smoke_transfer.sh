#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/smoke_transfer.sh

Downloads RoboMemArena Multi-Transferring if needed, then runs smoke.sh.

Options:
  --root PATH        Local RoboMemArena root. Default: datasets/robomemarena if datasets exists, else data/robomemarena
  --full            Download the full Multi-Transferring folder. Default.
  --tiny            Download only a tiny task subset for quick network tests.
  --task ID          Tiny-mode transfer task: 18, 19, 25, or 26. Default: 18
  --episodes N       Episodes to use for smoke. Default: 8 in full mode, 2 in tiny mode
  --output-dir PATH  Smoke output dir. Default: outputs/smoke_transfer
  --no-download      Do not download; only run using existing local files.
  --videos           Also export videos and manifest.json.
  --overwrite        Overwrite videos in --videos mode.
  --python PATH      Python executable. Default: $PYTHON, python, then python3
  -h, --help         Show this help.
EOF
}

die() {
  echo "smoke_transfer.sh: $*" >&2
  exit 1
}

has_hdf5() {
  local path="$1"
  [[ -d "${path}" ]] || return 1
  find "${path}" -name '*.hdf5' -print -quit 2>/dev/null | grep -q .
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
if [[ -n "${ROBOMEMARENA_ROOT:-}" ]]; then
  root="${ROBOMEMARENA_ROOT}"
elif [[ -e "${repo_root}/datasets" ]]; then
  root="datasets/robomemarena"
else
  root="data/robomemarena"
fi
task="18"
episodes=""
output_dir="outputs/smoke_transfer"
download=1
full=1
videos=0
overwrite=0
python_bin="${PYTHON:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      [[ $# -ge 2 ]] || die "--root requires a path"
      root="$2"
      shift 2
      ;;
    --full)
      full=1
      shift
      ;;
    --tiny)
      full=0
      shift
      ;;
    --task)
      [[ $# -ge 2 ]] || die "--task requires an id"
      task="$2"
      shift 2
      ;;
    --episodes)
      [[ $# -ge 2 ]] || die "--episodes requires a number"
      episodes="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || die "--python requires a path"
      python_bin="$2"
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || die "--output-dir requires a path"
      output_dir="$2"
      shift 2
      ;;
    --no-download)
      download=0
      shift
      ;;
    --videos)
      videos=1
      shift
      ;;
    --overwrite)
      overwrite=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

case "${task}" in
  18) task_dir="Multi-Transferring/18_chocolate_butter_cabinet_dataset" ;;
  19) task_dir="Multi-Transferring/19_tomato_milk_orange_cabinet_dataset" ;;
  25) task_dir="Multi-Transferring/25_butter_cream_dataset" ;;
  26) task_dir="Multi-Transferring/26_chocolate_pudding_cream_dataset" ;;
  *) die "--task must be one of: 18, 19, 25, 26" ;;
esac

if [[ -z "${episodes}" ]]; then
  if [[ "${full}" -eq 1 ]]; then
    episodes=8
  else
    episodes=2
  fi
fi

cd "${repo_root}"

if [[ -z "${python_bin}" ]]; then
  if command -v python >/dev/null 2>&1; then
    python_bin="python"
  elif command -v python3 >/dev/null 2>&1; then
    python_bin="python3"
  else
    die "could not find python. Pass --python PATH."
  fi
fi

if [[ "${full}" -eq 1 ]]; then
  data_check_path="${root}/Multi-Transferring"
else
  data_check_path="${root}/${task_dir}/subtask_data"
fi

if ! has_hdf5 "${data_check_path}"; then
  if [[ "${download}" -eq 0 ]]; then
    die "no local transfer HDF5 files found under ${data_check_path}"
  fi
  if [[ "${full}" -eq 1 ]]; then
    echo "[transfer] local full transfer data missing; downloading Multi-Transferring (~145GB)"
    "${python_bin}" "${script_dir}/download_robomemarena_transfer.py" \
      --root "${root}" \
      --full
  else
    echo "[transfer] local transfer data missing; downloading a tiny task ${task} subset"
    "${python_bin}" "${script_dir}/download_robomemarena_transfer.py" \
      --root "${root}" \
      --task "${task}" \
      --episodes "${episodes}"
  fi
fi

cmd=(
  "${script_dir}/smoke.sh"
  --root "${root}"
  --max-episodes "${episodes}"
  --families "source_target_memory,subtask_progress_memory"
  --python "${python_bin}"
  --episode-strategy "round-robin-tasks"
  --output-dir "${output_dir}"
  --prompt-items "8"
)
if [[ "${videos}" -eq 1 ]]; then
  cmd+=(--videos)
fi
if [[ "${overwrite}" -eq 1 ]]; then
  cmd+=(--overwrite)
fi

"${cmd[@]}"
