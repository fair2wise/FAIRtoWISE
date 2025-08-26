#!/usr/bin/env bash
# Startup script: startup_extractor.sh
# -----------------------------------------------------------------------------
# Purpose: Launch extract_terms_linkml.py under Nsight profiling on NERSC Perlmutter.
# Usage:   ./startup_extractor.sh [-o output_prefix]
# Options:
#   -o OUTPUT_PREFIX  Prefix for Nsight output files (default: "extract_<timestamp>")
#   -h                Show this help message and exit
# -----------------------------------------------------------------------------

set -Eeuo pipefail
IFS=$'\n\t'

module load python  # Load Python environment on NERSC Perlmutter

OUTPUT_PREFIX=""
TIMESTAMP=""

usage() {
  sed -n '2,8p' "$0"
  exit 1
}

while getopts "o:h" opt; do
  case "${opt}" in
    o) OUTPUT_PREFIX="$OPTARG" ;;
    h) usage ;;
    *) usage ;;
  esac
done

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
if [[ -z "$OUTPUT_PREFIX" ]]; then
  OUTPUT_PREFIX="pdfs2concepts_${TIMESTAMP}"
fi

/global/homes/d/dabramov/nsight-systems-2025.3.1/bin/nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=cpu \
  --cpuctxsw=process-tree \
  --output "$OUTPUT_PREFIX" \
  --force-overwrite=true \
  python pdfs2concepts.py