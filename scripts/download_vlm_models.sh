#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/download_vlm_models.sh [options]

Download the open-weight VLMs used by the MemProbe annotation pipeline.
Downloads are resumable and use the standard Hugging Face cache by default.

Options:
  --profile NAME       Model set: core, all, or extended. Default: all
  --cache-dir PATH     Override the Hugging Face Hub cache directory.
                       Default: HF_HUB_CACHE, then HF_HOME/hub, then
                       ~/.cache/huggingface/hub
  --model REPO_ID      Download only this model. Repeat for multiple models.
                       When present, --profile is ignored.
  --max-workers N      Parallel files per model. Default: 8
  --list               Print the selected model IDs and exit.
  --dry-run            Print commands without downloading.
  -h, --help           Show this help.

Profiles:
  core
    Qwen/Qwen3-VL-8B-Instruct
    allenai/Molmo2-8B
    DAMO-NLP-SG/VideoLLaMA3-7B

  all (default)
    Everything in core, plus:
    Qwen/Qwen2.5-VL-7B-Instruct
    OpenGVLab/InternVL3_5-8B-HF
    Efficient-Large-Model/LongVILA-R1-7B

  extended
    Everything in all, plus:
    Qwen/Qwen3-VL-32B-Instruct
    lmms-lab/LongVA-7B-DPO

Authentication:
  Public models normally need no login. If Hugging Face requests credentials,
  run `hf auth login` first or export HF_TOKEN.

License note:
  LongVILA-R1 and LongVA weights have non-commercial restrictions. Review each
  model card before redistribution or commercial use.
EOF
}

die() {
  echo "download_vlm_models.sh: $*" >&2
  exit 1
}

print_command() {
  printf ' %q' "$@"
  printf '\n'
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

core_models=(
  "Qwen/Qwen3-VL-8B-Instruct"
  "allenai/Molmo2-8B"
  "DAMO-NLP-SG/VideoLLaMA3-7B"
)

all_extra_models=(
  "Qwen/Qwen2.5-VL-7B-Instruct"
  "OpenGVLab/InternVL3_5-8B-HF"
  "Efficient-Large-Model/LongVILA-R1-7B"
)

extended_extra_models=(
  "Qwen/Qwen3-VL-32B-Instruct"
  "lmms-lab/LongVA-7B-DPO"
)

profile="all"
cache_dir=""
max_workers="8"
list_only=0
dry_run=0
custom_models=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      [[ $# -ge 2 ]] || die "--profile requires a value"
      profile="$2"
      shift 2
      ;;
    --cache-dir)
      [[ $# -ge 2 ]] || die "--cache-dir requires a path"
      cache_dir="$2"
      shift 2
      ;;
    --model)
      [[ $# -ge 2 ]] || die "--model requires a Hugging Face repository ID"
      custom_models+=("$2")
      shift 2
      ;;
    --max-workers)
      [[ $# -ge 2 ]] || die "--max-workers requires a number"
      max_workers="$2"
      shift 2
      ;;
    --list)
      list_only=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ "${max_workers}" =~ ^[1-9][0-9]*$ ]] || die "--max-workers must be a positive integer"

if [[ ${#custom_models[@]} -gt 0 ]]; then
  models=("${custom_models[@]}")
else
  case "${profile}" in
    core)
      models=("${core_models[@]}")
      ;;
    all)
      models=("${core_models[@]}" "${all_extra_models[@]}")
      ;;
    extended)
      models=(
        "${core_models[@]}"
        "${all_extra_models[@]}"
        "${extended_extra_models[@]}"
      )
      ;;
    *)
      die "--profile must be one of: core, all, extended"
      ;;
  esac
fi

# Preserve input order while avoiding duplicate downloads from repeated --model.
declare -A seen_models=()
unique_models=()
for model in "${models[@]}"; do
  [[ "${model}" == */* ]] || die "invalid Hugging Face repository ID: ${model}"
  if [[ -z "${seen_models[${model}]+x}" ]]; then
    unique_models+=("${model}")
    seen_models["${model}"]=1
  fi
done
models=("${unique_models[@]}")

if [[ -n "${cache_dir}" && "${cache_dir}" != /* ]]; then
  cache_dir="${repo_root}/${cache_dir}"
fi

echo "[vlm-download] profile=${profile}"
if [[ -n "${cache_dir}" ]]; then
  display_cache_dir="${cache_dir}"
elif [[ -n "${HF_HUB_CACHE:-}" ]]; then
  display_cache_dir="${HF_HUB_CACHE}"
elif [[ -n "${HF_HOME:-}" ]]; then
  display_cache_dir="${HF_HOME}/hub"
else
  display_cache_dir="${HOME}/.cache/huggingface/hub"
fi
echo "[vlm-download] cache=${display_cache_dir}"
echo "[vlm-download] models=${#models[@]}"
printf '  %s\n' "${models[@]}"

if [[ "${list_only}" -eq 1 ]]; then
  exit 0
fi

if command -v hf >/dev/null 2>&1; then
  download_command=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  echo "[vlm-download] using legacy 'huggingface-cli download'"
  download_command=(huggingface-cli download)
elif [[ "${dry_run}" -eq 1 ]]; then
  echo "[vlm-download] Hugging Face CLI not found; showing modern CLI commands only"
  download_command=(hf download)
else
  die "Hugging Face CLI not found. Install it with: pip install -U huggingface_hub"
fi

if [[ -n "${cache_dir}" ]]; then
  mkdir -p "${cache_dir}"
fi

failures=()
for index in "${!models[@]}"; do
  model="${models[${index}]}"
  cmd=(
    "${download_command[@]}"
    "${model}"
    --max-workers "${max_workers}"
  )
  if [[ -n "${cache_dir}" ]]; then
    cmd+=(--cache-dir "${cache_dir}")
  fi

  echo
  echo "[vlm-download] [$((index + 1))/${#models[@]}] ${model}"

  if [[ "${dry_run}" -eq 1 ]]; then
    printf '[vlm-download] command:'
    print_command "${cmd[@]}"
    continue
  fi

  if "${cmd[@]}"; then
    echo "[vlm-download] complete=${model}"
  else
    echo "[vlm-download] FAILED=${model}" >&2
    failures+=("${model}")
  fi
done

if [[ ${#failures[@]} -gt 0 ]]; then
  echo >&2
  echo "[vlm-download] failed models:" >&2
  printf '  %s\n' "${failures[@]}" >&2
  echo "[vlm-download] rerun the same command to resume partial downloads" >&2
  exit 1
fi

echo
if [[ "${dry_run}" -eq 1 ]]; then
  echo "[vlm-download] dry run complete"
else
  echo "[vlm-download] all downloads complete"
fi
