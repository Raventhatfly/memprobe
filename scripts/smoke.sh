#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/smoke.sh
  scripts/smoke.sh /path/to/robomemarena
  scripts/smoke.sh --root /path/to/robomemarena
  ROBOMEMARENA_ROOT=/path/to/robomemarena scripts/smoke.sh

Runs a small RoboMemArena MemProbe smoke test:
  1. Generate QA JSONL.
  2. Format a few Qwen-VL multiple-choice prompts with --dry-run.
  3. Optionally export matching videos with --videos.

Options:
  --root PATH           RoboMemArena root directory.
                        Optional; if omitted, common local/lab paths are searched.
  --output-dir PATH     Output directory. Default: outputs/smoke
  --max-episodes N      Number of episodes to scan. Default: 5
  --max-qa N            Number of QA items to keep in --videos mode. Default: 10
  --families LIST       Comma-separated QA families. Default: subtask_progress_memory
  --seed N              Choice shuffle seed. Default: 0
  --episode-strategy S  Episode selection: sorted or round-robin-tasks. Default: sorted
  --python PATH         Python executable. Default: $PYTHON, python, then python3
  --prompt-items N      Number of prompts to print. Default: 5
  --videos              Also write videos and manifest.json.
  --dataset PATH        RGB dataset path for --videos.
                         Default: /data/demo_0/obs/agentview_rgb
  --fps N               Video FPS for --videos. Default: 20
  --overwrite           Overwrite existing videos in --videos mode.
  -h, --help            Show this help.
EOF
}

die() {
  echo "smoke.sh: $*" >&2
  exit 1
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
default_root="/n/netscratch/hankyang_lab/Lab/felix/dataset/robomemarena"

looks_like_robomemarena_root() {
  local candidate="$1"
  [[ -d "${candidate}" ]] || return 1
  find "${candidate}" -path '*/subtask_data/*.hdf5' -print -quit 2>/dev/null | grep -q .
}

discover_root() {
  local candidate
  local candidates=(
    "${default_root}"
    "/n/netscratch/hankyang_lab/Lab/feiyang/dataset/robomemarena"
    "/n/netscratch/hankyang_lab/Lab/feiyang/datasets/robomemarena"
    "/n/netscratch/hankyang_lab/Lab/felix/datasets/robomemarena"
    "/n/holylabs/LABS/hankyang_lab/Lab/felix/dataset/robomemarena"
    "/n/holylabs/LABS/hankyang_lab/Lab/feiyang/dataset/robomemarena"
    "${repo_root}/data/robomemarena"
    "${repo_root}/datasets/robomemarena"
    "${HOME:-}/data/robomemarena"
    "${HOME:-}/dataset/robomemarena"
    "${HOME:-}/datasets/robomemarena"
    "/data/robomemarena"
    "/dataset/robomemarena"
    "/datasets/robomemarena"
  )

  for candidate in "${candidates[@]}"; do
    [[ -n "${candidate}" ]] || continue
    if looks_like_robomemarena_root "${candidate}"; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  local search_roots=(
    "${repo_root}"
    "${HOME:-}/data"
    "${HOME:-}/dataset"
    "${HOME:-}/datasets"
    "/data"
    "/dataset"
    "/datasets"
    "/n/netscratch/hankyang_lab/Lab"
    "/n/holylabs/LABS/hankyang_lab/Lab"
  )

  for candidate in "${search_roots[@]}"; do
    [[ -d "${candidate}" ]] || continue
    while IFS= read -r found; do
      if looks_like_robomemarena_root "${found}"; then
        printf '%s\n' "${found}"
        return 0
      fi
    done < <(find "${candidate}" -maxdepth 6 -type d -iname 'robomemarena' 2>/dev/null)
  done

  return 1
}

root="${ROBOMEMARENA_ROOT:-}"
output_dir="outputs/smoke"
max_episodes=5
max_qa=10
families="subtask_progress_memory"
seed=0
episode_strategy="sorted"
python_bin="${PYTHON:-}"
prompt_items=5
with_videos=0
dataset="/data/demo_0/obs/agentview_rgb"
fps=20
overwrite=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      [[ $# -ge 2 ]] || die "--root requires a path"
      root="$2"
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || die "--output-dir requires a path"
      output_dir="$2"
      shift 2
      ;;
    --max-episodes)
      [[ $# -ge 2 ]] || die "--max-episodes requires a number"
      max_episodes="$2"
      shift 2
      ;;
    --max-qa)
      [[ $# -ge 2 ]] || die "--max-qa requires a number"
      max_qa="$2"
      shift 2
      ;;
    --families)
      [[ $# -ge 2 ]] || die "--families requires a value"
      families="$2"
      shift 2
      ;;
    --seed)
      [[ $# -ge 2 ]] || die "--seed requires a number"
      seed="$2"
      shift 2
      ;;
    --episode-strategy)
      [[ $# -ge 2 ]] || die "--episode-strategy requires a value"
      episode_strategy="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || die "--python requires a path"
      python_bin="$2"
      shift 2
      ;;
    --prompt-items)
      [[ $# -ge 2 ]] || die "--prompt-items requires a number"
      prompt_items="$2"
      shift 2
      ;;
    --videos)
      with_videos=1
      shift
      ;;
    --dataset)
      [[ $# -ge 2 ]] || die "--dataset requires a path"
      dataset="$2"
      shift 2
      ;;
    --fps)
      [[ $# -ge 2 ]] || die "--fps requires a number"
      fps="$2"
      shift 2
      ;;
    --overwrite)
      overwrite=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      die "unknown option: $1"
      ;;
    *)
      [[ -z "${root}" ]] || die "unexpected positional argument: $1"
      root="$1"
      shift
      ;;
  esac
done

if [[ -z "${root}" ]]; then
  echo "[smoke] no --root passed; searching for RoboMemArena data..."
  root="$(discover_root || true)"
fi

[[ -n "${root}" ]] || {
  die "could not auto-find RoboMemArena. Put it at data/robomemarena, ~/data/robomemarena, or pass --root once."
}
looks_like_robomemarena_root "${root}" || die "does not look like a RoboMemArena root with */subtask_data/*.hdf5: ${root}"

if [[ -z "${python_bin}" ]]; then
  if command -v python >/dev/null 2>&1; then
    python_bin="python"
  elif command -v python3 >/dev/null 2>&1; then
    python_bin="python3"
  else
    die "could not find python. Pass --python PATH."
  fi
fi

if ! "${python_bin}" -c 'import h5py' >/dev/null 2>&1 && ! command -v h5dump >/dev/null 2>&1; then
  die "need either Python package h5py or the h5dump command. Try: ${python_bin} -m pip install -e ."
fi

if [[ "${with_videos}" -eq 1 ]]; then
  for module in h5py imageio numpy; do
    if ! "${python_bin}" -c "import ${module}" >/dev/null 2>&1; then
      die "video smoke needs Python package '${module}'. Try: ${python_bin} -m pip install -e . imageio imageio-ffmpeg"
    fi
  done
fi

mkdir -p "${repo_root}/${output_dir}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "[smoke] repo=${repo_root}"
echo "[smoke] python=${python_bin}"
echo "[smoke] root=${root}"
echo "[smoke] output=${repo_root}/${output_dir}"

qa_path="${repo_root}/${output_dir}/qa.jsonl"
prompt_path="${repo_root}/${output_dir}/prompts.txt"

if [[ "${with_videos}" -eq 1 ]]; then
  cmd=(
    "${python_bin}" "${script_dir}/make_qa_video_smoke.py"
    --root "${root}"
    --max-episodes "${max_episodes}"
    --max-qa "${max_qa}"
    --families "${families}"
    --episode-strategy "${episode_strategy}"
    --output-dir "${repo_root}/${output_dir}"
    --dataset "${dataset}"
    --fps "${fps}"
  )
  if [[ "${overwrite}" -eq 1 ]]; then
    cmd+=(--overwrite)
  fi
  echo "[smoke] generating QA plus videos"
  "${cmd[@]}"
else
  echo "[smoke] generating QA"
  "${python_bin}" "${script_dir}/generate_robomemarena_qa.py" \
    --root "${root}" \
    --max-episodes "${max_episodes}" \
    --families "${families}" \
    --episode-strategy "${episode_strategy}" \
    --seed "${seed}" \
    --output "${qa_path}"
fi

echo "[smoke] formatting dry-run prompts"
"${python_bin}" "${script_dir}/eval_qwen_vl_mcq.py" \
  --input "${qa_path}" \
  --max-items "${prompt_items}" \
  --dry-run | tee "${prompt_path}"

echo "[smoke] done"
echo "[smoke] qa=${qa_path}"
echo "[smoke] prompts=${prompt_path}"
if [[ "${with_videos}" -eq 1 ]]; then
  echo "[smoke] manifest=${repo_root}/${output_dir}/manifest.json"
  echo "[smoke] videos=${repo_root}/${output_dir}/videos"
fi
