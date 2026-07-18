# Chord-managed DiffusionGemma (CHRD-DIFF-01)

`llama-diffusion-daemon` (the DiffusionGemma inference binary, HTTP on `:8877`)
is now **owned and lifecycle-managed by Chord** (`src/diffusion.rs`), not a
perpetual standalone systemd service. Operator hard requirement: diffusion must
never sit resident in VRAM indefinitely — Chord lazy-starts it on demand and
idle-evicts it after inactivity, the same "no perpetual holds" posture Chord
already applies to on-demand `llama-server` backends (`models::routing`).

## What changed

- `src/diffusion.rs` — a new `DiffusionManager` (process-global, like
  `gpu_exclusive::GPU_EXCLUSIVE`) that spawns/kills the bare
  `llama-diffusion-daemon` process via `tokio::process`, polls `GET /health`
  before declaring it up (mirrors `snap::vllm::VLLMAdapter`'s pattern for a
  docker-managed engine, adapted for a bare process), and tracks last-activity
  for idle eviction.
- `src/main.rs` — spawns `diffusion::spawn_idle_reaper` alongside the existing
  `models::routing::idle_stop_sweep` on-demand-backend reaper.
- `src/routes.rs` (`chat_completions`) — a request for the managed model
  (`DIFFUSION_MODEL_ID`, default `diffusion-gemma`) is routed to
  `DiffusionManager::ensure_running()` and forwarded to its `:8877`
  `/v1/chat/completions`, ahead of (and independent from) the existing P5
  tag-aware `ModelRegistry`/`Backend` routing — this daemon was never
  registered as a `Backend` there.
- `src/routes.rs` (`gpu_exclusive_acquire`) and `src/admin/idle.rs` (idle-mode
  release) both now also call `diffusion::global().stop()` on a fresh
  GPU-exclusive grant / idle-mode release, so this daemon is evicted alongside
  resident Ollama models and on-demand `llama-server` backends — never left
  contending for VRAM with a foreign GPU-exclusive holder or during idle-mode.

## Config (env)

| Var | Default | Meaning |
|---|---|---|
| `DIFFUSION_MODEL_ID` | `diffusion-gemma` | Chat-completions model id that routes to this daemon. |
| `DIFFUSION_DAEMON_BIN` | `/opt/nvme-scratch/dgem/llama-diffusion/build-vulkan/bin/llama-diffusion-daemon` | Binary path. |
| `DIFFUSION_MODEL_PATH` | `/opt/nvme-scratch/dgem/diffusiongemma-eval/models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf` | GGUF weights path. |
| `DIFFUSION_LD_LIBRARY_PATH` | `/opt/nvme-scratch/dgem/llama-diffusion/build-vulkan/bin` | `LD_LIBRARY_PATH` for the child process. |
| `DIFFUSION_BIND` | `127.0.0.1` | Bind host. |
| `DIFFUSION_PORT` | `8877` | Bind port. |
| `DIFFUSION_IDLE_SECS` | `300` | Idle window before Chord evicts (kills) the daemon. `0` disables eviction. |
| `DIFFUSION_START_TIMEOUT_SECS` | `120` | Health-poll budget on lazy start. |
| `DIFFUSION_EXTRA_ARGS` | `-ngl 99 -t 4 --diffusion-eb auto -c 8192 -ub 8192 -b 8192` | Space-separated extra CLI args after `-m <model>`. |

No secrets are involved — the daemon takes no credentials.

## Ops action required at deploy time (NOT performed by this change)

Once this is deployed to the GPU host, the standalone `dgem.service` systemd
unit **must be stopped and disabled**:

```
systemctl stop dgem.service
systemctl disable dgem.service
```

Leaving it enabled would race Chord's own spawn/kill of the same binary/port
and reintroduce the "sits in VRAM perpetually" problem this change exists to
fix. This is a deploy/ops step for the orchestrator — a build agent does not
reach into the managing host to run it.
