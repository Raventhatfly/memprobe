#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/download_droid.sh [mode] [options]

Modes:
  --sample-rlds     Download DROID 100-episode RLDS sample (~2GB). Default.
  --full-rlds       Download full DROID RLDS (~1.7TB).
  --raw-lite        Download raw DROID MP4 data without SVO or stereo MP4 (~5.6TB).
  --raw-full        Download raw DROID data with stereo HD videos/SVO (~8.7TB).

Options:
  --target PATH     Output directory. Default depends on mode:
                    datasets/droid_100, datasets/droid, datasets/droid_raw_nostereo,
                    or datasets/droid_raw if ./datasets exists; otherwise under ./data.
  --dry-run         Print the gsutil command without running it.
  -h, --help        Show this help.

Notes:
  - Requires gsutil. Install with Google Cloud SDK or: pip install gsutil
  - Official DROID bucket: gs://gresearch/robotics
EOF
}

die() {
  echo "download_droid.sh: $*" >&2
  exit 1
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

mode="sample-rlds"
target=""
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sample-rlds)
      mode="sample-rlds"
      shift
      ;;
    --full-rlds)
      mode="full-rlds"
      shift
      ;;
    --raw-lite)
      mode="raw-lite"
      shift
      ;;
    --raw-full)
      mode="raw-full"
      shift
      ;;
    --target)
      [[ $# -ge 2 ]] || die "--target requires a path"
      target="$2"
      shift 2
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

if [[ -e "${repo_root}/datasets" ]]; then
  base_dir="${repo_root}/datasets"
else
  base_dir="${repo_root}/data"
fi

case "${mode}" in
  sample-rlds)
    source_uri="gs://gresearch/robotics/droid_100"
    default_target="${base_dir}/droid_100"
    exclude_pattern=""
    size_hint="~2GB"
    ;;
  full-rlds)
    source_uri="gs://gresearch/robotics/droid"
    default_target="${base_dir}/droid"
    exclude_pattern=""
    size_hint="~1.7TB"
    ;;
  raw-lite)
    source_uri="gs://gresearch/robotics/droid_raw"
    default_target="${base_dir}/droid_raw_nostereo"
    exclude_pattern=".*SVO.*|.*stereo.*\\.mp4$"
    size_hint="~5.6TB"
    ;;
  raw-full)
    source_uri="gs://gresearch/robotics/droid_raw"
    default_target="${base_dir}/droid_raw"
    exclude_pattern=""
    size_hint="~8.7TB"
    ;;
  *)
    die "unknown mode: ${mode}"
    ;;
esac

target="${target:-${default_target}}"
mkdir -p "${target}"

cmd=(gsutil -m rsync -r)
if [[ -n "${exclude_pattern}" ]]; then
  cmd+=(-x "${exclude_pattern}")
fi
cmd+=("${source_uri}" "${target}")

echo "[droid] mode=${mode}"
echo "[droid] size=${size_hint}"
echo "[droid] source=${source_uri}"
echo "[droid] target=${target}"

if [[ "${dry_run}" -eq 1 ]]; then
  printf '[droid] command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

command -v gsutil >/dev/null 2>&1 || die "gsutil not found. Install Google Cloud SDK or run: pip install gsutil"

"${cmd[@]}"
echo "[droid] done"
