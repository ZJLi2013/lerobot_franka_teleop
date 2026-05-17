#!/usr/bin/env bash
# Batch-process all .mp4 files under a LeRobot camera videos directory using
# video_roi_reframe.py. The output preserves the same chunk-XXX/file-XXX.mp4
# relative structure under OUTPUT_DIR.
#
# Usage:
#   batch_video_roi_reframe.sh <CAMERA_DIR> <OUTPUT_DIR> <ROI_JSON> [extra args...]
#
# Arguments:
#   CAMERA_DIR   Source camera dir, e.g.
#                .../milk_power_2_20260515_v01/videos/observation.images.exterior_image
#   OUTPUT_DIR   Mirror dir to write processed videos into. Must NOT equal
#                CAMERA_DIR (we don't overwrite in place). Same relative paths
#                are reproduced under OUTPUT_DIR.
#   ROI_JSON     Path to ROI JSON produced by `pick-roi` (or hand-edited).
#   extra args   Forwarded verbatim to `video_roi_reframe.py process`. Use this
#                to switch codec/quality, e.g. `--codec opencv --fourcc mp4v`
#                or `--codec libsvtav1 --crf 30`.
#
# Examples:
#   # Default codec (libsvtav1, LeRobot-compatible):
#   batch_video_roi_reframe.sh \
#       babycare/milk_power_2_20260515_v01/videos/observation.images.exterior_image \
#       dataset_babycare_20260515/babycare_postprocessed_crop/milk_power_2_20260515_v01/videos/observation.images.exterior_image \
#       scripts/tools/roi.json
#
#   # Same as the tested mp4v output:
#   batch_video_roi_reframe.sh CAMERA_DIR OUTPUT_DIR roi.json --codec opencv --fourcc mp4v
#
# Env:
#   PYTHON       Python interpreter to use (default: `python`).
#   SKIP_EXISTING=1  Skip files whose output already exists.
#   DRY_RUN=1    Print the commands without executing them.

set -euo pipefail

if [ $# -lt 3 ]; then
    sed -n '2,40p' "$0"
    exit 1
fi

CAMERA_DIR=$(realpath -m -- "$1")
OUTPUT_DIR=$(realpath -m -- "$2")
ROI_JSON=$(realpath -m -- "$3")
shift 3
EXTRA_ARGS=("$@")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/video_roi_reframe.py"
PYTHON_BIN="${PYTHON:-python}"

if [ ! -d "$CAMERA_DIR" ]; then
    echo "[ERROR] CAMERA_DIR is not a directory: $CAMERA_DIR" >&2
    exit 1
fi
if [ ! -f "$ROI_JSON" ]; then
    echo "[ERROR] ROI_JSON not found: $ROI_JSON" >&2
    exit 1
fi
if [ ! -f "$PY_SCRIPT" ]; then
    echo "[ERROR] Python helper not found: $PY_SCRIPT" >&2
    exit 1
fi
if [ "$CAMERA_DIR" = "$OUTPUT_DIR" ]; then
    echo "[ERROR] OUTPUT_DIR must differ from CAMERA_DIR (no in-place overwrite)." >&2
    exit 1
fi

mapfile -d '' -t videos < <(find "$CAMERA_DIR" -type f -name '*.mp4' -print0 | sort -z)

if [ "${#videos[@]}" -eq 0 ]; then
    echo "[ERROR] No .mp4 files found under: $CAMERA_DIR" >&2
    exit 1
fi

echo "[INFO] Found ${#videos[@]} videos under $CAMERA_DIR"
echo "[INFO] Output dir : $OUTPUT_DIR"
echo "[INFO] ROI JSON   : $ROI_JSON"
echo "[INFO] Python     : $PYTHON_BIN $PY_SCRIPT"
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    echo "[INFO] Extra args : ${EXTRA_ARGS[*]}"
fi

skip_existing="${SKIP_EXISTING:-0}"
dry_run="${DRY_RUN:-0}"

ok=0
skipped=0
failed=0
failed_files=()

for src in "${videos[@]}"; do
    rel="${src#"$CAMERA_DIR/"}"
    dst="$OUTPUT_DIR/$rel"

    if [ "$skip_existing" = "1" ] && [ -f "$dst" ]; then
        echo "[SKIP] $rel (output already exists)"
        skipped=$((skipped + 1))
        continue
    fi

    mkdir -p -- "$(dirname -- "$dst")"
    echo "[PROCESS] $rel"
    echo "          -> $dst"

    cmd=("$PYTHON_BIN" "$PY_SCRIPT" process "$src" -r "$ROI_JSON" -o "$dst")
    if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
        cmd+=("${EXTRA_ARGS[@]}")
    fi

    if [ "$dry_run" = "1" ]; then
        printf '          $'
        printf ' %q' "${cmd[@]}"
        printf '\n'
        continue
    fi

    if "${cmd[@]}"; then
        ok=$((ok + 1))
    else
        echo "[FAIL] $rel" >&2
        failed=$((failed + 1))
        failed_files+=("$rel")
    fi
done

echo
echo "[SUMMARY] processed=$ok  skipped=$skipped  failed=$failed  total=${#videos[@]}"
if [ "$failed" -gt 0 ]; then
    echo "[SUMMARY] failed files:"
    for f in "${failed_files[@]}"; do
        echo "  - $f"
    done
    exit 1
fi
