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

Alongside the ROCm GPU tiers, Chord also registers a **`vulkan`** backend — a
`llama.cpp` `llama-server` built with the Vulkan/RADV (Mesa) driver
(`-DGGML_VULKAN=ON`, Mesa 25.0.7 on `gfx1151`). It is a *driver-stable*
alternative to the ROCm-only lemonade build for **dense large models**, used when
ROCm is unavailable or unstable. It is memory-bound like HIP/ROCm (~5 tok/s at
70B), so it is intended for batch/async serving rather than interactive traffic.
Dense-large models (70B/32B-dense class; `llama3.3:70b` validated) are tagged to
it via the model registry. See
[docs/serving.md](docs/serving.md#serving-backends) for details and the validated
`llama3.3:70b` numbers.

### SLM router (DOCGEN-03)

Beyond serving/proxying chat traffic, Chord owns a standalone **SLM router**
([`src/router`](src/router)) — a small-model routing capability for in-process
callers that need a generation without picking a model themselves. This is the
mechanism behind "all documentation-engine inference routes through Chord, and
Chord decides the destination": the `moosenet/Terminus` documentation engine
(a separate spec item) only *asks* the router to generate; the router *owns*
the destination decision.

- **Policy** ([`src/router/policy.rs`](src/router/policy.rs)) is pure decision
  logic, no network I/O: an explicit, env-configured `RoutingPolicy` maps a
  request's estimated token count to one of three destinations — a local
  cheap/fast model for simple requests, a local high-context model once a
  request exceeds `SLM_ROUTER_CONTEXT_THRESHOLD_TOKENS`, or OpenRouter's
  frontier-free tier once a request exceeds even the local high-context
  ceiling (`SLM_ROUTER_LOCAL_HIGH_CTX_MAX_TOKENS`) — so a request is never
  silently truncated, only escalated.
- **Router** ([`src/router/slm_router.rs`](src/router/slm_router.rs)) resolves
  a decision to a real backend and executes the generation, reusing the
  existing backend catalogue (`models::backends::seed_from_env`) rather than
  a second one — the local destinations are just different model names sent
  to the primary Ollama-compatible backend (the same shape
  `AgenticModelRouter`'s fast/deep models already use), and the cloud
  destination reuses the existing `"openrouter"` backend's bearer-key
  indirection (`Backend::api_key_env`) unchanged.
- **Egress isolation**: every hop that would reach the cloud destination is
  checked against `SLM_ROUTER_CLOUD_EGRESS_ALLOWLIST` (comma-separated
  hostnames) *before* any network call — fail-closed, same posture as
  `supervisor::egress_policy`. `SLM_ROUTER_CLOUD_ALLOWED=false` disables cloud
  routing outright, independent of the allow-list.
- **Graceful fallback, never a silent failure**: a destination that is
  egress-denied, unconfigured, or fails at execution time falls back per
  policy (cloud → local high-context → local cheap). If every destination in
  the chain fails, `route_and_execute` returns a hard `SlmRouterError` — it
  never fabricates or silently drops a generation.
- **Logged for evaluation**: every routing decision the router acts on
  (including failed/fallback hops) is logged via `tracing` and retained
  in-memory (`SlmRouter::decisions()`) — the feed a future SLM-router
  evaluation sweep (DOCGEN-04) consumes to judge routing quality.
- The router assumes its input is already PII-swept by the caller (the doc
  engine's own PII gate) — it is not itself a PII gate, only a destination
  decision + execution layer.

Env vars: `SLM_ROUTER_CONTEXT_THRESHOLD_TOKENS` (default 6000),
`SLM_ROUTER_LOCAL_HIGH_CTX_MAX_TOKENS` (default 32000),
`SLM_ROUTER_LOCAL_HIGH_CTX_MODEL` / `SLM_ROUTER_LOCAL_CHEAP_MODEL` /
`SLM_ROUTER_CLOUD_MODEL` (model names/tags), `SLM_ROUTER_CLOUD_ALLOWED`
(default true), `SLM_ROUTER_CLOUD_EGRESS_ALLOWLIST` (default
`openrouter.ai`).

