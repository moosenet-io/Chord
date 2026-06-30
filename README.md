<p align="center"><img src="assets/banner.svg" alt="Chord" width="640"></p>

<p align="center"><img src="assets/badges.svg" alt="badges"></p>

# Chord

Inference manager for local LLM fleets on unified-memory hardware.

## Overview

Chord is the inference manager for the Lumina / Harmony stack. It fronts one or
more local LLM backends and decides, per request, **which model runs on which
backend, with which launch flags, and whether it can fit in memory at all**.

Lumina runs interactive assistant workloads; Harmony runs agentic coder
workloads. Both share the same finite pool of accelerator memory on
unified-memory APUs, where a single misjudged context size or a leaked process
can starve the next model. Chord exists to make that sharing safe and
deterministic: it measures each model's serving profile, routes accordingly,
admits work only when memory genuinely allows, swaps models cleanly, and tracks
the operating mode of the whole fleet.

Chord is part of the **Lumina constellation**. It is currently being separated
out of the `lumina-constellation` monorepo (the `chord-proxy` crate) into this
standalone repository.

## Architecture

<p align="center"><img src="assets/architecture.svg" alt="Chord architecture" width="100%"></p>

Per-model serving profiles, measured by the intake harness, drive routing,
admission, and launch flags. Two front doors feed Chord Core: **Lumina**
(assistant, control plane) and **Harmony** (coders, agentic plane). Chord Core
is built from four cooperating components:

- **Routing** — serving-profile lookup selects the best runtime and the
  backend per model, with a chat-role pin so interactive traffic stays on the
  intended backend.
- **Memory Coordinator** — substrate-aware admission control. It models the
  hardware as either separate ceilings or a unified memory pool, and performs
  tier-aware eviction to make room without overcommitting.
- **Clean-Swap Launcher** — a verified teardown-then-launch cycle: tear down
  the outgoing model, verify the memory release, then launch the next model
  with an explicit `-c` context size. Includes orphan force-kill and a
  false-OOM guard so transient failures don't masquerade as real
  out-of-memory conditions.
- **Mode Controller** — tracks the operating mode with persisted state:
  *assistant-live* (pin a model and swap the rest around it) versus
  *batch-coder* (give the GPU fully to one model at a time).

Two cross-cutting subsystems wrap the core:

- **SNAP — observability & inventory** ([`src/snap`](src/snap)): a real GPU VRAM
  reader (sysfs / rocm-smi / Ollama) and engine-health poller, an on-disk GGUF +
  Ollama manifest scanner with quant detection, passive per-model activity
  observation, and a request-analytics logger with imputed cloud-cost / savings.
  It exposes a read-only observability surface on distinct paths behind Chord's
  own JWT auth, and includes a vLLM engine adapter. SNAP can optionally persist
  its streams (analytics, inventory, activity, VRAM samples) to Postgres; this is
  **default-off**, gated behind `CHORD_SNAP_PERSIST`, best-effort, and never
  fails or slows a proxied request.
- **Runtime isolation (ISO)** ([`src/supervisor`](src/supervisor)): every runtime
  Chord launches is spawned with a scrubbed environment and an explicit egress
  posture. ISO-01 scrubs telemetry/online env vars and declares the policy
  (advisory); ISO-02 enforces it in the kernel with a per-runtime network
  namespace — a `Serve` runtime gets **no route** (every external `connect()`
  fails at the kernel, so it cannot phone home even if it ignores the opt-outs),
  while a `Pull` runtime gets a constrained, nftables-filtered path to the
  configured model-source allow-list only. It is **fail-closed**: without
  `CAP_NET_ADMIN` the launch is refused rather than run with full host egress
  (an explicit, loud `CHORD_ALLOW_UNISOLATED=1` override exists, off by default).

These route across a **three-tier backend stack**:

1. **llama.cpp-rocm** — the GPU tier; broadest model support and most
   VRAM-efficient. Uses `--no-mmap` and keep-warm for large models.
2. **ollama-rocm** — the GPU tier for architectures llama.cpp can't load
   (e.g. gemma, gpt-oss, glm, qwen families).
3. **CPU tier** — a genuine system-RAM fallback for small models, used as a
   last resort.

Routing prefers `llama.cpp-rocm`, falls back to `ollama-rocm`, and finally to
the CPU tier.

## Status

Chord ships as the standalone Rust crate `chord-proxy`, version **1.4**. It
depends on `terminus-rs` 1.1 (the serving-profile types + intake DB config) from
the Gitea crate registry. The core inference manager — routing, the substrate-aware
Memory Coordinator, the verified Clean-Swap Launcher, the Mode Controller, the
SNAP observability subsystem, and per-runtime kernel egress isolation — is in
place; the coder benchmark charts are the remaining pending item.

## Documentation

Component explainers (written from the source in [`src/`](src)) live in
[`docs/`](docs/):

- **[docs/architecture.md](docs/architecture.md)** — component deep-dive: request
  flow, Routing, backend tiers, model registry & storage tiering,
  memory/residency management, the control API, the agentic loop, and the search
  harness, each mapped to its real module/types.
- **[docs/serving.md](docs/serving.md)** — the serving / coordinator subsystem
  (Memory Coordinator, Clean-Swap Launcher, Mode Controller) mapped to the code
  that actually ships, with a present/partial/absent table.
- **[docs/egress.md](docs/egress.md)** — the runtime isolation model: ISO-01
  env-scrub + egress policy and ISO-02 per-runtime network-namespace enforcement,
  the fail-closed posture, and the honest scope boundaries.
- **[docs/snap-persistence.md](docs/snap-persistence.md)** — the optional,
  default-off SNAP → Postgres persistence layer (`CHORD_SNAP_PERSIST`).
- **[docs/README.md](docs/README.md)** — the docs index.

## Test Results

<!-- CHART: per-model BLITZ vs MULTI-FILE pass rates — themed SVG, generated at coder-sweep completion -->

Benchmark charts are generated from the **MINT v2 coder harness** run and will be
committed under [`docs/charts/`](docs/charts/) (themed SVG, per-model BLITZ vs
MULTI-FILE pass rates) once the coder sweep completes. _Charts pending._

## License

MIT
