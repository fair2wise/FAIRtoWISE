#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# startup_ollama_cluster.sh — start per-node Ollama scaled + nginx proxy
# -----------------------------------------------------------------------------
set -Eeuo pipefail
IFS=$'\n\t'

module load python  # ensures nvidia-smi and Python env are available

# -------------- defaults (customizable via flags) -----------------------------
GPU_LIST="0,1,2,3"
NUM_PARALLEL=32
MODEL_TAG="mistral-small3.1:latest"
OUTPUT_PREFIX=""
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
START_PORT=11434
NUM_INSTANCES=4  # one per GPU under CUDA_VISIBLE_DEVICES

usage() {
  sed -n '16,50p' "$0"
  exit 1
}

while getopts "g:n:o:m:p:i:h" opt; do
  case "$opt" in
    g) GPU_LIST="$OPTARG" ;;
    n) NUM_PARALLEL="$OPTARG" ;;
    o) OUTPUT_PREFIX="$OPTARG" ;;
    m) MODEL_TAG="$OPTARG" ;;
    p) START_PORT="$OPTARG" ;;
    i) NUM_INSTANCES="$OPTARG" ;;
    h|*) usage ;;
  esac
done

: "${OUTPUT_PREFIX:=ollama_${TIMESTAMP}}"

# --------------------- export Ollama GPU & performance variables -------------
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export OLLAMA_NUM_PARALLEL="$NUM_PARALLEL"
export OLLAMA_KV_CACHE_TYPE=pmem
export OLLAMA_SCHED_SPREAD=1
export OLLAMA_FLASH_ATTENTION=0
export OLLAMA_MAX_LOADED_MODELS=1
export OMP_NUM_THREADS=64

echo "[startup] GPUs            : $GPU_LIST"
echo "[startup] Parallel reqs   : $OLLAMA_NUM_PARALLEL"
echo "[startup] Model           : $MODEL_TAG"
echo

# --------------- optionally pull model to avoid runtime delay -----------------
# ollama pull "$MODEL_TAG" || { echo "pull failed"; exit 1; }

# ------------------- launch Ollama scaled instances --------------------------
# using theodufort's scaler
curl -fsSL -o ollama_server_scaler.sh \
  https://raw.githubusercontent.com/theodufort/ollama-server-scaler/main/ollama_server_scaler.sh
chmod +x ollama_server_scaler.sh

./ollama_server_scaler.sh \
  --num-instances "$NUM_INSTANCES" \
  --start_port "$START_PORT" &

# ------------------------ prepare nginx proxy config --------------------------
cat > ollama_proxy.conf <<EOF
events {}
http {
  upstream ollama_backend {
    least_conn;
EOF

for ((i=0; i<NUM_INSTANCES; i++)); do
  port=$((START_PORT + i))
  echo "    server 127.0.0.1:$port;" >> ollama_proxy.conf
done

cat >> ollama_proxy.conf <<'EOF'
  }

  server {
    listen START_PORT;

    location / {
      proxy_pass http://ollama_backend;
      proxy_set_header Host $host;
      proxy_set_header X-Real-IP $remote_addr;
      proxy_buffering off;
      proxy_http_version 1.1;
      proxy_set_header Connection "";
    }
  }
}
EOF

# placeholder hack: replace START_PORT
sed -i "s/START_PORT/$START_PORT/" ollama_proxy.conf

# -------------------------- start nginx using this config ---------------------
nginx -c "$(pwd)/ollama_proxy.conf" -g "daemon off;" &
echo "[startup] Nginx proxy active on port $START_PORT → backends $START_PORT–$((START_PORT+NUM_INSTANCES-1))"
echo "[startup] Done. Connect to http://$(hostname):$START_PORT"

# ----------------------------- end of script -----------------------------------
