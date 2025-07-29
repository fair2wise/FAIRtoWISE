#!/usr/bin/env bash
# -----------------------------------------------------------------------------
#  startup_ollama_mistral.sh  — single Ollama server, 80 %-of-GPU auto-tuning
# -----------------------------------------------------------------------------
set -Eeuo pipefail
IFS=$'\n\t'

module load python        # nvidia-smi etc. on NERSC Perlmutter

# ---------------- defaults ----------------------------------------------------
GPU_LIST="0,1,2,3"
NUM_PARALLEL=""                   # auto if blank
MODEL_TAG="mistral-small3.1:latest"
OUTPUT_PREFIX=""
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"

usage() { sed -n '15,50p' "$0"; exit 1; }

while getopts "g:n:o:m:h" opt; do
  case "$opt" in
    g) GPU_LIST="$OPTARG"  ;;
    n) NUM_PARALLEL="$OPTARG" ;;
    o) OUTPUT_PREFIX="$OPTARG" ;;
    m) MODEL_TAG="$OPTARG" ;;
    h|*) usage ;;
  esac
done

[[ -z "$OUTPUT_PREFIX" ]] && OUTPUT_PREFIX="ollama_${TIMESTAMP}"

# ---------------- runtime knobs -----------------------------------------------
export CUDA_VISIBLE_DEVICES="$GPU_LIST"

# this should be higher than the number of threads in the extraction script
export OLLAMA_NUM_PARALLEL=32

# spill excess KV pages to RAM instead of falling back to CPU weights
export OLLAMA_KV_CACHE_TYPE=pmem

# shard KV across GPUs; weights already sharded automatically
export OLLAMA_SCHED_SPREAD=1

# Flash-attention not supported by q4_K_M build on A100 → disable to avoid warn
export OLLAMA_FLASH_ATTENTION=0

# keep just this one model in GPU memory
export OLLAMA_MAX_LOADED_MODELS=1

# GGML CPU helpers (EPYC 7763) — helps the tokenizer / small ops
export OMP_NUM_THREADS=64

echo "[startup] GPUs             : $GPU_LIST"
echo "[startup] Parallel requests: $OLLAMA_NUM_PARALLEL"
echo "[startup] Model            : $MODEL_TAG"
echo "[startup] Nsight prefix    : $OUTPUT_PREFIX"
echo

# ---------------- (optional) pre-pull to avoid network delay ------------------
# ollama pull "$MODEL_TAG" || { echo "pull failed"; exit 1; }

# ---------------- Nsight Systems wrapper -------------------------------------
NSYS_BIN="/global/homes/d/dabramov/nsight-systems-2025.3.1/bin/nsys"

exec "$NSYS_BIN" profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --output "$OUTPUT_PREFIX" \
  --force-overwrite=true \
  ollama serve
