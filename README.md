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

Early — separation from the monorepo is in progress. Chord ships as the Rust
crate `chord-proxy`. Version **1.0**.

## License

MIT
