#!/usr/bin/env bash
# Builds the DFlash draft with its own Q4_K lm_head (base Qwen3.6 head, taken from the MTP sidecar).
# Without its own head the draft reuses Clover's 1 GB BF16 head; the 286 MB Q4_K copy makes drafting cheaper.
# Output only changes speed: the target model verifies every drafted token.
#   clover/make-draft.sh [models_dir]   (reads ../clover by default, writes drafts/ in the repo)
set -euo pipefail
cd "$(dirname "$0")/.."
M=${1:-../clover}
B=${B:-build}
[ -f "$B/bin/libggml-base.so" ] || { echo "build first: clover/build.sh"; exit 1; }
mkdir -p drafts
g++ -O2 -std=c++17 clover/addhead.cpp -Iggml/include -L"$B/bin" -lggml-base -Wl,-rpath,"$PWD/$B/bin" -o "$B/bin/addhead"
"$B/bin/addhead" "$M/dflash-Qwen3.6-35B-A3B-Q8_0.gguf" "$M/mtp-Qwen3.6-35B-A3B-Q8_0.gguf" \
    "drafts/dflash-Qwen3.6-35B-A3B-Q8_0-q4k-head.gguf" q4_k
ls -l "drafts/dflash-Qwen3.6-35B-A3B-Q8_0-q4k-head.gguf"
