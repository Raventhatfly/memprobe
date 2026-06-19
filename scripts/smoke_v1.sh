#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
python_bin="${PYTHON:-python}"
output_dir="${1:-${repo_root}/outputs/smoke_v1}"

cd "${repo_root}"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${python_bin}" -m memprobe inspect \
  --manifest examples/mvp/episode.json

"${python_bin}" -m memprobe generate \
  --manifest examples/mvp/episode.json \
  --proposals examples/mvp/events.json \
  --verified-only \
  --output-dir "${output_dir}"

echo "[smoke-v1] output=${output_dir}"
