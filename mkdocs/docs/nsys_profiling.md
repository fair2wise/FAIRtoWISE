# NVTX Python Module
Install `nvtx` in your environment:
```python
pip install nvtx
```

Push/pop ranges around your critical regions:

```python
import nvtx

def heavy_compute(...):
    nvtx.range_push("heavy_compute")
    try:
        # ← your GPU/CPU work here
        ...
    finally:
        nvtx.range_pop()

def preprocess(...):
    with nvtx.annotate("preprocess", color="blue"):
        # automatically does push/pop
        ...
```

Decorator form (for entire functions):

```
from nvtx import annotate

@annotate("encode_embeddings", color="green")
def encode_embeddings(texts):
    return model.encode(texts, ...)
```

# Ollama serve (1 Node)


```bash
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
```

Ollama Serve (4 nodes)

Download `ollama-server-scaler.sh`
```bash
curl -fsSL -o ollama_server_scaler.sh https://raw.githubusercontent.com/theodufort/ollama-server-scaler/main/ollama_server_scaler.sh
chmod +x ollama_server_scaler.sh
```
From the jupyter terminal (perlmutter) run:

```
srun -N4 --ntasks-per-node=1 --gpus-per-node=4 \ 
    bash -lc './ollama_server_scaler.sh --num-instances 4 --start_port 11434'

BAM. Now 
```

Ollama Optimization
Good morning, team. Yesterday, I optimized our Ollama configuration after nsys report revealed highly variable LLM response times  (4s to 200s). By creating a new model with the batch size increased from the default 512 to 4096, I achieved a 10% performance gain. ollama create mistralSmall:h100 -f ./mistral_small.Modelfile
But as we don't have much concurrent request - NUM_PARALLEL=8 (environment variable for ollama serve) does not scale well. probably key is to use different models for different requests, that may streamline the request (hosting 2 models seems ideal as we have many VRAM) (edited) 
mistral_small.Modelfile 
FROM mistral-small3.1:latest
# --- Parameters for H100 Performance Tuning ---
# 1. Offload all layers to the GPU. This is essential.
PARAMETER num_gpu 999
# 3. Set a very large batch size to saturate the H100's cores.
PARAMETER num_batch 4096
# 5. Set a large context window.
PARAMETER num_ctx 4096

Profile: Extraction Script



(nersc-python) dabramov@nid001008:/pscratch/sd/d/dabramov/fair2wise> nsys profile   --trace=cuda,nvtx,osrt   --sample=cpu   --cpuctxsw=process-tree   --show-output=true   --output=extract_terms_trace --force-overwrite true   python /pscratch/sd/d/dabramov/fair2wise/extract_terms_linkml.py

Notes:
This is mainly CPU bound



Profile: ollama serve
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OLLAMA_SCHED_SPREAD=true        
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_FLASH_ATTENTION=true




(nersc-python) dabramov@nid001008:/pscratch/sd/d/dabramov/fair2wise> nsys profile   --trace=cuda,nvtx,osrt   --sample=none   --cpuctxsw=none   --show-output=true   --output=ollama_trace --force-overwrite true  ollama serve

Notes:
This is the GPU heavy-component (inference)
There was an error at the end generating the report:



[1/1] [7%                          ] ollama_trace.nsys-rep
Importer error status: Importation failed.
Import Failed with unexpected exception: /dvs/p4/build/sw/devtools/Agora/Rel/QuadD_Main/QuadD/Host/QdstrmImporter/main.cpp(34): Throw in function {anonymous}::Importer::Importer(const boost::filesystem::path&, const boost::filesystem::path&)
Dynamic exception type: boost::wrapexcept<QuadDCommon::RuntimeException>
std::exception::what: RuntimeException
[QuadDCommon::tag_message*] = Status: AnalysisFailed
Error {
  Type: RuntimeError
  SubError {
    Type: InvalidArgument
    Props {
      Items {
        Type: OriginalExceptionClass
        Value: "N5boost10wrapexceptIN11QuadDCommon24InvalidArgumentExceptionEEE"
      }
      Items {
        Type: OriginalFile
        Value: "/dvs/p4/build/sw/devtools/Agora/Rel/QuadD_Main/QuadD/Host/Analysis/Modules/EventCollection.cpp"
      }
      Items {
        Type: OriginalLine
        Value: "1055"
      }
      Items {
        Type: OriginalFunction
        Value: "void QuadDAnalysis::EventCollection::CheckOrder(QuadDAnalysis::EventCollectionHelper::EventContainer&, const QuadDAnalysis::ConstEvent&) const"
      }
      Items {
        Type: ErrorText
        Value: "Wrong event order has been detected when adding events to the collection:\nnew event ={ StartNs=156667046066 StopNs=156667059202 GlobalId=312618860172123 Event={ TraceProcessEvent=[{ Correlation=4510028 EventClass=0 TextId=2093 ReturnValue=0 },] } Type=48 }\nlast event ={ StartNs=157198969174 StopNs=157198979764 GlobalId=312618860172123 Event={ TraceProcessEvent=[{ Correlation=4579607 EventClass=0 TextId=2466 ReturnValue=0 },] } Type=48 }"
      }
    }
  }
}
Generated:
    /pscratch/sd/d/dabramov/fair2wise/ollama_trace.qdstrm

Attempt 2:
Using a full node (4xA100GPUs) I can get ollama serve running across all GPUs



(nersc-python) dabramov@nid001361:/pscratch/sd/d/dabramov/fair2wise> export CUDA_VISIBLE_DEVICES=0,1,2,3
(nersc-python) dabramov@nid001361:/pscratch/sd/d/dabramov/fair2wise> nsys profile   --trace=cuda,nvtx,osrt   --sample=none   --show-output=true   --output=ollama_trace_allgpus   --env OLLAMA_NUM_PARALLEL=4,OLLAMA_MAX_LOADED_MODELS=4   ollama serve


Profile: KG-RAG Responses

(nersc-python) dabramov@nid001008:/pscratch/sd/d/dabramov/fair2wise> nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=cpu \
  --cpuctxsw=process-tree \
  --show-output=true \
  --force-overwrite=true \
  --output=kg_rag_trace \
  python /pscratch/sd/d/dabramov/fair2wise/kg_rag_ollama.py \
    --question "tell me about organic photovoltaics"



Notes:
https://developer.nvidia.com/blog/insights-techniques-and-evaluation-for-llm-driven-knowledge-graphs/


Nvidia-smi
run this during the when your job is running nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 1 -f gpu_log.csv &
and when job is done pkill -f nvidia-smi
sample log:
(watch -n0.5 nvidia-smi still works and fun to watch if you have a lot of time) (edited) 


Installing latest version of nsys:



# choose a target dir (change if you like)
export NSYS_DIR="$PSCRATCH/nsight-systems-2025.3.1"

# download (resume‐capable)
wget -c \
  https://developer.download.nvidia.com/devtools/nsight-systems/NsightSystems-linux-public-2025.3.1.90-3582212.run \
  -O "$PSCRATCH/NsightSystems-2025.3.1.run"

# install silently
chmod +x "$PSCRATCH/NsightSystems-2025.3.1.run"
"$PSCRATCH/NsightSystems-2025.3.1.run" --silent --target "$NSYS_DIR"

# add to PATH permanently
echo "export PATH=\"$NSYS_DIR/bin:\$PATH\"" >> ~/.bashrc
source ~/.bashrc

# sanity check
nsys --version   # → NVIDIA Nsight Systems 2025.3.1

# Or

/global/homes/d/dabramov/nsight-systems-2025.3.1/bin/nsys –version

Nsight output (copy paste a thread)
Run this to get a CSV file with a summary of the NVTX annotated functions:


nsys stats --format csv,column --output .,- extract_20250715_152436.nsys-rep


Name	Start	Duration	TID	Category
fuzzy_merge	177.01s	321.270 s	257643	
fuzzy_merge	530.952s	24.995 s	257643	
fuzzy_merge	510.564s	20.388 s	257643	
fuzzy_merge	555.948s	20.249 s	257643	
fuzzy_merge	498.281s	12.283 s	257643	
fuzzy_merge	167.909s	9.101 s	257643	
_save_terms_threadsafe	576.197s	5.439 ms	257643	
_extract_and_attach_properties	576.196s	632.775 μs	257643	
_register_new_term	498.281s	96.828 μs	257643	
_register_new_term	576.196s	92.650 μs	257643	
_register_new_term	177.01s	92.279 μs	257643	
_register_new_term	510.564s	81.648 μs	257643	
get_context_snippet	555.947s	61.489 μs	257643	
get_context_snippet	498.281s	47.081 μs	257643	
get_schema_context_for_llm	52.6009s	42.312 μs	257643	
get_context_snippet	167.909s	41.771 μs	257643	
get_context_snippet	510.564s	41.621 μs	257643	
get_context_snippet	530.952s	40.889 μs	257643	
get_context_snippet	177.01s	38.805 μs	257643	
_prepare_prompt	52.601s	14.067 μs	257643	
validate_and_fix_term	167.909s	10.811 μs	257643	
validate_and_fix_term	498.281s	10.219 μs	257643	
validate_and_fix_term	555.947s	9.819 μs	257643	
check_relation_validity	498.281s	8.767 μs	257643	
validate_and_fix_term	510.564s	8.747 μs	257643	
check_relation_validity	510.564s	8.276 μs	257643	
check_relation_validity	177.01s	8.075 μs	257643	
check_relation_validity	530.952s	7.695 μs	257643	
check_relation_validity	576.196s	7.194 μs	257643	
validate_and_fix_term	530.952s	5.741 μs	257643	
validate_and_fix_term	177.01s	5.561 μs	257643	
check_relation_validity	498.281s	4.578 μs	257643	
check_relation_validity	510.564s	2.996 μs	257643	
check_relation_validity	177.01s	2.695 μs	257643	
check_relation_validity	498.281s	2.565 μs	257643	
check_relation_validity	530.952s	2.515 μs	257643	
check_relation_validity	530.952s	2.465 μs	257643	
check_relation_validity	576.196s	2.395 μs	257643	
_postprocess_term	167.909s	2.294 μs	257643	
check_relation_validity	498.281s	2.224 μs	257643	
check_relation_validity	177.01s	2.184 μs	257643	
check_relation_validity	177.01s	2.174 μs	257643	
check_relation_validity	576.196s	1.944 μs	257643	
_postprocess_term	177.01s	1.894 μs	257643	
_postprocess_term	498.281s	1.884 μs	257643	
_postprocess_term	510.564s	1.783 μs	257643	
check_relation_validity	510.564s	1.754 μs	257643	
check_relation_validity	530.952s	1.733 μs	257643	
_postprocess_term	555.948s	1.713 μs	257643	
normalize_term	530.952s	1.643 μs	257643	
_postprocess_term	530.952s	1.633 μs	257643	
check_relation_validity	510.564s	1.613 μs	257643	
check_relation_validity	576.196s	1.513 μs	257643	
normalize_term	498.281s	1.232 μs	257643	
normalize_term	510.564s	1.113 μs	257643	
normalize_term	498.281s	1.043 μs	257643	
normalize_term	167.909s	741 ns	257643	
normalize_term	555.948s	732 ns	257643	
normalize_term	177.01s	721 ns	257643	
normalize_term	576.196s	712 ns	257643	
normalize_term	510.564s	641 ns	257643	
normalize_term	177.01s	621 ns	257643	
normalize_term	498.281s	601 ns	257643	
normalize_term	510.564s	541 ns	257643	
normalize_term	498.281s	521 ns	257643	
normalize_term	576.196s	521 ns	257643	
normalize_term	510.564s	431 ns	257643	
normalize_term	498.281s	421 ns	257643	
normalize_term	530.952s	421 ns	257643	
normalize_term	177.01s	411 ns	257643	
normalize_term	530.952s	381 ns	257643	
normalize_term	530.952s	381 ns	257643	
normalize_term	530.952s	381 ns	257643	
normalize_term	177.01s	361 ns	257643	
normalize_term	576.196s	340 ns	257643	
normalize_term	576.196s	321 ns	257643	
normalize_term	177.01s	301 ns	257643	
normalize_term	510.564s	290 ns	257643	


Try to reduce the duplicated nodes at the beginning


