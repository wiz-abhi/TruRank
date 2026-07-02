# TruRank — sealed reproduction environment for the Redrob "India Runs" challenge.
#
# Two-phase design (matches submission_spec §10.3):
#   1. OFFLINE precompute — builds data/processed/candidates_cache.pkl and
#      downloads the embedding + cross-encoder weights. Network is allowed here.
#   2. ONLINE rank — produces submission.csv. rank.py hard-locks the network off
#      (HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE) and must finish ≤5 min on CPU.
#
# Build:
#   docker build -t trurank .
#
# Run (mount the directory holding candidates.jsonl into /app/data/raw):
#   docker run --rm -v "$(pwd)/data/raw:/app/data/raw" -v "$(pwd)/out:/app/out" trurank
#
# The output lands at out/submission.csv. The full 100K run does the precompute
# once (~a few minutes, network on) then ranks (~30s, network off).

FROM python:3.11-slim

WORKDIR /app

# System build deps for torch/sentence-transformers wheels are already in slim;
# keep the image lean.
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

# Dependencies first for layer caching. Install the CPU-only torch wheel
# explicitly — the default PyPI torch on Linux is the multi-GB CUDA build,
# which wastes build time and disk for a CPU-only ranking step.
COPY requirements-rank.txt .
RUN pip install --no-cache-dir torch==2.12.1 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-rank.txt

# Project source.
COPY src/ src/
COPY config.yaml precompute.py rank.py ./

# Reproduce pipeline: precompute (builds cache + downloads models, network ON)
# then rank (network hard-locked OFF inside rank.py). candidates.jsonl is mounted
# at run time under data/raw/.
CMD ["sh", "-c", \
     "mkdir -p out && \
      python precompute.py --candidates data/raw/candidates.jsonl --out data/processed/candidates_cache.pkl && \
      python rank.py --candidates data/raw/candidates.jsonl --cache data/processed/candidates_cache.pkl --out out/submission.csv"]
