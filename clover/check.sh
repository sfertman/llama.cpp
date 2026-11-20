#!/usr/bin/env bash
# Regression check on the GPU host, after a build or rebase. Stop any server on the port first.
#   clover/check.sh [models_dir]
# Prints AR and prompt speeds, DFlash speed and acceptance, and diffs greedy output against
# clover/check-ref.txt (saved from a known-good build; the first run creates it).
set -uo pipefail
cd "$(dirname "$0")/.."
M=${1:-../clover}
B=${B:-build}
PORT=${PORT:-9399}
T="$M/Clover-35B-A3B-Preview-Q4_K_M.gguf"
D="drafts/dflash-Qwen3.6-35B-A3B-Q8_0-q4k-head.gguf"
OUT=$(mktemp -d)

echo "== llama-bench (AR decode, prompt 4096)"
"$B/bin/llama-bench" -m "$T" -ngl 99 -fa 1 -ctk bf16 -ctv bf16 -ub 1024 -r 2 -p 4096 -n 128 2>/dev/null | grep -E "pp4096|tg128"

echo "== DFlash n-max 3, greedy"
"$B/bin/llama-server" -m "$T" -md "$D" -ngl 99 -ngld 99 --spec-type draft-dflash --spec-draft-n-max 3 \
    -c 8192 -ub 1024 -np 1 -fa on -ctk bf16 -ctv bf16 --temp 0 --host 127.0.0.1 --port "$PORT" > "$OUT/server.log" 2>&1 &
PID=$!
for _ in $(seq 120); do curl -s -m 2 "http://127.0.0.1:$PORT/health" | grep -q ok && break; sleep 1; done
curl -s -m 300 "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
    -d '{"prompt":"Write a Python function that parses an ISO 8601 date string and returns a datetime. Explain each step.\n\n","n_predict":400,"temperature":0,"cache_prompt":false}' \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); t=d["timings"]; print("tg %.1f t/s, draft accepted %s/%s" % (t["predicted_per_second"], t.get("draft_n_accepted"), t.get("draft_n"))); open(sys.argv[1],"w").write(d["content"])' "$OUT/text.txt"
kill "$PID"; wait "$PID" 2>/dev/null

echo "== greedy output"
if [ -f clover/check-ref.txt ]; then
    if cmp -s clover/check-ref.txt "$OUT/text.txt"; then echo "identical to clover/check-ref.txt"
    else echo "DIFFERS from clover/check-ref.txt (expected after upstream kernel changes; read it):"; diff <(fold -w 100 clover/check-ref.txt) <(fold -w 100 "$OUT/text.txt") | head -20; fi
else
    cp "$OUT/text.txt" clover/check-ref.txt; echo "saved clover/check-ref.txt"
fi
echo "(server log: $OUT/server.log)"
