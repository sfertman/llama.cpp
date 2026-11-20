# Clover-35B-A3B on llama.cpp master (Vulkan, DFlash)

This branch is upstream llama.cpp master plus a few Clover commits. It runs the official
Clover-35B-A3B-Preview GGUF with its reference numerics, Vulkan fusions, and DFlash
speculative decoding. Drafting changes speed only: the target model verifies every token.

| Commit | What it does |
|---|---|
| clover: load the pre-merge official GGUF | skips the diffusion-only tensors, converts raw `A_log` |
| clover: restore the reference bf16 rounding | makes the graph match the original implementation op for op |
| vulkan: Clover chain fusions and indirect GDN state | speed (AR 46.9 → 54.8 tok/s on the 8060S) |
| clover: tooling | this folder |

## 1. Get the files (~26 GB)

Put the model files in one directory next to the repo, e.g. `~/llama.cpp/clover/` beside `~/llama.cpp/<repo>/`.
Paths below assume that layout (`../clover` from the repo root). Nothing is ever written there; it can be read-only.

| File | Source | SHA-256 (start) |
|---|---|---|
| `Clover-35B-A3B-Preview-Q4_K_M.gguf` (23.0 GB) | https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview-GGUF | `4d439bb83248` |
| `dflash-Qwen3.6-35B-A3B-Q8_0.gguf` (421 MB) | https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF | `fdc58dc581a1` |
| `mtp-Qwen3.6-35B-A3B-Q8_0.gguf` (1.99 GB) | https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF | `8c59eac56ba6` |

```sh
mkdir -p ~/llama.cpp/clover && cd ~/llama.cpp/clover
curl -LO https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview-GGUF/resolve/main/Clover-35B-A3B-Preview-Q4_K_M.gguf
curl -LO https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF/resolve/main/dflash-Qwen3.6-35B-A3B-Q8_0.gguf
curl -LO https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF/resolve/main/mtp-Qwen3.6-35B-A3B-Q8_0.gguf
sha256sum *.gguf
```

The MTP file is only used once, as the source of the draft's lm_head in step 3.

## 2. Build

Needs cmake, g++, the Vulkan headers/loader and `glslc` (shaderc).

```sh
cd ~/llama.cpp/<repo>
clover/build.sh            # -> build/bin/llama-server, llama-bench
```

## 3. Make the draft model (once)

```sh
clover/make-draft.sh       # reads ../clover, writes drafts/dflash-Qwen3.6-35B-A3B-Q8_0-q4k-head.gguf (707 MB)
```

This gives the DFlash draft its own Q4_K copy of the base Qwen3.6 lm_head. Otherwise the draft
reuses Clover's 1 GB BF16 head on every draft step. The Q4_K copy is 286 MB.

## 4. Run

One profile, by hand (from the repo root):

```sh
build/bin/llama-server -m ../clover/Clover-35B-A3B-Preview-Q4_K_M.gguf \
  -md drafts/dflash-Qwen3.6-35B-A3B-Q8_0-q4k-head.gguf --spec-type draft-dflash -ngld 99 \
  --spec-draft-n-max 3 -ngl 99 -c 32768 -ub 1024 -np 1 -fa on -ctk bf16 -ctv bf16 --temp 0 \
  --host 0.0.0.0 --port 9080
```

Two profiles, chosen by the request's `"model"` field (`clover-chat` or `clover-code`):

```sh
build/bin/llama-server --models-preset clover/presets.ini --models-max 1 --host 0.0.0.0 --port 9080
```

- `clover-chat`: `n-max 3`, for chat, prose and thinking. It loads at startup.
- `clover-code`: `n-max 7`, `p-min 0.35`, for code and summaries.
- `--models-max 1` keeps one copy in memory (~26 GB) and swaps on demand (a few seconds per switch). Without it both stay loaded (~52 GB).

Measured speeds (Radeon 8060S, on the previous base; re-measure with `clover/check.sh`):

| Profile / mode | Speed |
|---|---|
| AR | 54.8 tok/s |
| `clover-chat` | 58–70 tok/s |
| `clover-code` | ~80 tok/s |
| Summaries | ~85 tok/s |
| Prompt processing | 1350 t/s at 4K, 1100 t/s at 16K |

## 5. Check after a build or update

```sh
clover/check.sh            # stop the server on 9080 first if the GPU is busy
```

This prints prompt and decode speeds plus DFlash speed and acceptance. It also diffs the greedy
output against `clover/check-ref.txt`. The first run creates that file; commit it from a known-good
build. Upstream kernel changes can alter the text after some tokens. Read it, don't just diff it.

## 6. Update from upstream

```sh
git fetch origin
git rebase origin/master   # conflicts, if any, are almost always in ggml-vulkan.cpp
clover/build.sh && clover/check.sh
```

If something regresses: `git reset --hard <last good tag>` and rebuild.
Tag each good state, e.g. `git tag clover-YYYYMMDD`.

## Switches (environment)

| Variable | Effect |
|---|---|
| `LLAMA_CLOVER_NO_BF16_ROUND=1` | skip the reference bf16 rounding (faster, not reference-exact) |
| `LLAMA_CLOVER_NO_VK_FUSION=1` | disable the Clover Vulkan chain fusions |
| `LLAMA_CLOVER_NO_GDN_STATE_FUSION=1` | disable the in-cache GDN state read/write |
| `LLAMA_CLOVER_VK_FUSION_MODES=<mask>` | enable fusion modes by bit (round, l2, gdn gate, shexp, attn gate, round+add, gated norm) |

`clover/run-watch.sh` is a dev helper: it restarts the server whenever `.watch/restart` is touched,
using `.watch/args` and `.watch/env`.
