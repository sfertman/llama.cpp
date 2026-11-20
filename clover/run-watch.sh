#!/usr/bin/env bash
# Dev helper (host): (re)starts llama-server whenever .watch/restart is touched.
#   .watch/bin   - "build" or "build-sbx" (which build dir's llama-server to run)
#   .watch/args  - llama-server arguments (one line)
#   .watch/env   - optional LLAMA_*=value / GGML_VK_PERF_LOGGER*=value lines
# Server output goes to .watch/server.log. Ctrl+C stops the watcher and the server.
set -u
cd "$(dirname "$0")/.."
W=.watch
mkdir -p "$W"
pid=""
last=""

stop() { [ -n "$pid" ] && kill "$pid" 2>/dev/null && wait "$pid" 2>/dev/null; pid=""; }
trap 'stop; exit 0' INT TERM

while true; do
    cur=$(stat -c %Y.%s "$W/restart" 2>/dev/null || true)
    if [ -n "$cur" ] && [ "$cur" != "$last" ]; then
        last=$cur
        stop
        dir=$(cat "$W/bin" 2>/dev/null || echo build)
        case "$dir" in build|build-sbx) ;; *) echo "bad bin dir: $dir" > "$W/server.log"; continue ;; esac
        envs=()
        while IFS= read -r line; do
            [[ "$line" =~ ^(LLAMA_|GGML_VK_PERF_LOGGER)[A-Z0-9_]*=[A-Za-z0-9._-]*$ ]] && envs+=("$line")
        done < <(cat "$W/env" 2>/dev/null)
        read -r -a args < "$W/args"
        echo "[watch] $(date +%T) starting $dir/bin/llama-server ${envs[*]-} ${args[*]}" > "$W/server.log"
        env "${envs[@]}" "./$dir/bin/llama-server" "${args[@]}" >> "$W/server.log" 2>&1 &
        pid=$!
    fi
    sleep 1
done
