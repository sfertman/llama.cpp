#!/usr/bin/env bash
# Build llama-server / llama-bench with Vulkan. Run from anywhere; output goes to <repo>/build.
#   clover/build.sh            build into build/
#   clover/build.sh build-sbx  build into another directory
set -euo pipefail
cd "$(dirname "$0")/.."
B=${1:-build}
# HF_UI_VERSION pins the bundled web UI release so configuring does not depend on a fresh download
HF_UI_VERSION=b11951 cmake -S . -B "$B" -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
cmake --build "$B" -j"$(nproc)" --target llama-server llama-bench llama-cli
echo "built: $B/bin/llama-server"
