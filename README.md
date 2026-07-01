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

### Model fleet manager (autonomous model curation)

Chord doesn't just route requests to a fixed set of models — it manages the
fleet's models as a living inventory across three **storage tiers**. The
file-backed registry ([`src/models/registry.rs`](src/models/registry.rs),
`ModelRegistry` / `ModelRecord` / `StorageTier`) tracks every known model as
*hot* (resident in VRAM), *warm* (on local disk, not loaded), or *cold* (only in
the archive, e.g. NFS). At startup `reconcile()` *finds* what actually exists by
walking the local and archive Ollama manifest trees — including Hugging-Face
style names (`hf.co/org/model:tag`) — and rewrites the records to match reality,
so the registry never drifts from disk. SNAP's inventory scanner
([`src/snap/inventory.rs`](src/snap/inventory.rs)) complements this with a
quant-aware sweep of GGUF and Ollama files across every configured storage
location.

Movement between tiers is automatic and safe. On a request for a cold model,
the `PullCoordinator` ([`src/models/transfer.rs`](src/models/transfer.rs))
transparently promotes it cold → warm — copying the manifest and its referenced
blobs from the archive to the local root, with a disk precheck, per-model
concurrent-pull dedup, timeout, and mid-copy cleanup so a failed pull never
leaves corrupt state. Under disk pressure the eviction sweep
([`src/models/eviction.rs`](src/models/eviction.rs)) does the reverse, archiving
the least-recently-requested warm models warm → cold, archive-first /
delete-after, never touching hot or protected models. Acquisition from outside
the host is treated as a privileged operation: a runtime that needs to *fetch* a
model runs in the `Pull` network namespace
([`src/supervisor/egress_filter.rs`](src/supervisor/egress_filter.rs)), which
gets a default-drop, nftables-filtered egress path to **only** the configured
model-source allow-list — never a baked-in host — while serving runtimes get no
route at all. The result is a self-curating model fleet: it knows what it has,
pulls what a request needs, reclaims disk when it's tight, and reaches the
network for new weights only through a locked-down, allow-listed door.

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
- **[docs/test-results.md](docs/test-results.md)** — the S86 coder-fleet sweep
  results: themed BLITZ vs MULTI-FILE pass-rate charts, leaderboard, table, and
  takeaways.
- **[docs/README.md](docs/README.md)** — the docs index.

## Test Results

Results from the **S86 coder-fleet sweep** on `gfx1151` (MINT v2 harness,
`qwen3:8b` judge) — full charts, table, and takeaways in
[`docs/test-results.md`](docs/test-results.md).

[![BLITZ vs MULTI-FILE pass rate by model](docs/charts/coder-sweep-blitz-vs-multi.svg)](docs/test-results.md)

`qwen3-coder:30b` tops the fleet at **81% overall** with a perfect BLITZ score and
is now the served model on `gfx1151`; multi-file coordination is where the fleet
separates, and base/general models score near zero.

## License

MIT
