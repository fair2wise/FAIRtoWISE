#!/usr/bin/env bash
# Startup script: startup_extractor.sh
# -----------------------------------------------------------------------------
# Purpose: Launch json2kg.py under Nsight profiling on NERSC Perlmutter.
# Usage:   ./startup_concepts2kg.sh [-o output_prefix]
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
  OUTPUT_PREFIX="concepts2kg_${TIMESTAMP}"
fi

/global/homes/d/dabramov/nsight-systems-2025.3.1/bin/nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=cpu \
  --cpuctxsw=process-tree \
  --output "$OUTPUT_PREFIX" \
  --force-overwrite=true \
  python concepts2kg.py storage/terminology/extracted_terms-test.json storage/kg/mat_kg.json --validate-relations --max-workers 32 --very-verbose