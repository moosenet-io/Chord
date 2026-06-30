# Chord — Architecture

A component deep-dive, written from the source in [`../src/`](../src). Every
component below names the real module, type, or function that backs it. Where the
[architecture diagram](../assets/architecture.svg) shows a box whose logic is
spread across several modules (or whose spec'd shape differs from what shipped in
this extracted crate), that is called out explicitly rather than papered over.

<p align="center"><img src="../assets/architecture.svg" alt="Chord architecture" width="100%"></p>

## What Chord is

Chord (`chord-proxy`) is the inference manager that fronts a fleet of local LLM
backends. A single process exposes **two axum listeners** built in
[`main.rs`](../src/main.rs):

- the **proxy port** (`CHORD_PROXY_PORT`, default `9099`) — the request front
  door, router built by [`routes::build_router`](../src/routes.rs);
- the **control port** (`CHORD_CONTROL_PORT`, default `8090`) — the operator /
  dashboard API, router built by
  [`control::build_control_router`](../src/control.rs). A bind failure on the
  control port is logged but never takes the proxy down (`main.rs` spawns it in a
  task that only warns on error).

Shared state is the single `AppState` struct ([`routes.rs`](../src/routes.rs)),
which carries the MCP proxy, the agentic executor, the rate limiter, the model
registry, the pull coordinator, the local evictor, and the disk probe/lock — so
the proxy handlers and the control handlers operate over the same live registry.

## Request flow (the proxy front door)

The OpenAI-compatible entry point is
[`routes::chat_completions`](../src/routes.rs) (`POST /v1/chat/completions`). In
order, a request goes through:

1. **Auth** — `auth_check` validates the JWT (`Authorization: Bearer …`) against
   `CHORD_JWT_SECRET`. An empty secret disables auth cluster-wide (used in tests
   and trusted single-tenant deploys). Auth failures are recorded by the
   `AuditLogger` (token hashed, never stored).
2. **Backend-configured check** — if `CHORD_LLM_URL` is unset the endpoint returns
   `503` immediately.
3. **Rate limit** — `ProxyRateLimiter::check_and_record` applies the per-user
   daily LLM budget; over-budget returns `429` with `Retry-After`.
4. **Alias resolution** — `config::resolve_model_alias` rewrites the request's
   `model` (e.g. a `lumina-fast` alias → the real `gpt-oss:20b`) so the upstream
   never sees a name it doesn't know. The model name is normalized to a tagged
   `name:tag` registry key (untagged ⇒ `:latest`).
5. **Pull-on-miss (storage tier)** — the resolved model's tier is looked up in the
   registry; **only** a `Cold` model triggers a transparent archive pull
   (`PullCoordinator::ensure_local`) before inference. Hot / Warm / registry-unknown
   models pass straight through. Any known model has its `last_requested` bumped.
6. **Backend routing** — `models::routing::resolve_and_ensure` picks the model's
   tagged backend, starts it on demand if needed, and returns the upstream URL.
   On any failure it falls back to `CHORD_LLM_URL` ("availability over strictness").
7. **Forward** — hop-by-hop headers are stripped, the (possibly model-rewritten)
   body is forwarded, and the upstream response — JSON or `text/event-stream` — is
   streamed straight back to the caller.

The other proxy routes are `/v1/tools/list`, `/v1/tools/call`,
`/v1/tools/discover` (the MCP tool surface), `/v1/agent/execute` (the agentic
loop, below), `/v1/infer` (one prompt → normalized per-backend metrics), and
`/health` / `/v1/audit/summary` (no auth).

## Components

### Routing

**What it is.** Two distinct routing decisions, in two modules:

- **Backend-per-model (the "how to serve it" decision)** —
  [`models::routing`](../src/models/routing.rs).
  `resolve_and_ensure` maps a registry model to its tagged
  [`Backend`](../src/models/backends.rs), converts it to the
  `terminus_rs` lifecycle shape (`to_resolved`), and calls
  `lifecycle::ensure_up` to start an on-demand backend before forwarding.
  A companion `idle_stop_sweep` (spawned from `main.rs`, 60 s interval) stops any
  on-demand GPU backend whose `idle_stop_secs` has elapsed — "no perpetual holds".
  Always-on, Ollama, and daemon backends are never stopped.

- **Chat-role pin (the "which model is the assistant" decision)** —
  [`routing::assistant_profile`](../src/routing/assistant_profile.rs).
  `decide_chat_role` / `fetch_chat_role_decision` consume the S84 assistant-intake
  measurement (via `terminus_rs::intake::assistant::reporting`) and return a
  `ChatRoleDecision`:
  - `Route { model_id, backend_tag, behavioral_mean }` — point the Lumina chat
    alias at a measured-fit model **that already cleared the intake latency /
    degradation guard**, and
  - `KeepDefault { reason }` — when no candidate cleared the guard, *or* the
    measured pick isn't a registry-known model, keep the operator's current alias.

  This is the "chat-role pin" box on the diagram: it pins interactive traffic to
  an intended, vetted backend and refuses to route the chat alias to a model the
  registry can't actually serve.

### Backend tiers

**What it is.** First-class, hardware-tagged inference backends —
[`models::backends`](../src/models/backends.rs).
The data model is the `Backend` struct (`name`, `url`, `hardware: Hardware`,
`kind: BackendKind`, `always_on`, `idle_stop_secs`, optional `LaunchSpec`).
`Backend::on_demand()` is true for non-`always_on`, non-`Daemon` backends.

`seed_from_env` builds the default catalogue from env, mirroring the three-tier
stack in the README/diagram:

| Backend | `hardware` / `kind` | Tier | Notes |
|---------|---------------------|------|-------|
| `ollama` | `Cpu` / `Ollama` | CPU | primary, `always_on` (ROCm doesn't engage on this APU, so it's the CPU tier) |
| `ollama-cpu` | `Cpu` / `Ollama` | CPU | resident embeddings / micro-jobs |
| `lemonade-coder` | `Gpu` / `LlamaServer` | GPU (llama.cpp) | one fixed model, unit-managed, idle-stops at 900 s |
| `llama-gpu` | `Gpu` / `LlamaServer` | GPU (llama.cpp) | generic on-demand server that loads *any* requested model's blob |

The generic `llama-gpu` `LaunchSpec` is where the diagram's
"**launch w/ explicit `-c`**" and "**`--no-mmap`**" annotations physically live —
its `args` are `-c 32768 -ngl 999 -fa 1 --no-mmap --host 0.0.0.0 --port …` with
the model passed via `-m`. The CPU tier is the genuine system-RAM fallback "last
resort" for small models.

> Note on the diagram's `ollama-rocm` label: in this crate the Ollama backends are
> tagged `Hardware::Cpu` because ROCm does not engage on the target APU. The
> "ollama as a GPU fallback for architectures llama.cpp can't load" story is a
> deployment/topology concern, not a separate code path here — both Ollama
> backends share the same `BackendKind::Ollama` routing.

### Model registry & storage tiering

**What it is.** The persistent record of every known model and which storage tier
it lives at — [`models::registry`](../src/models/registry.rs), type
`ModelRegistry` over `ModelRecord`. The tiers are the `StorageTier` enum:

- `Hot` — loaded in VRAM (or marked loaded);
- `Warm` — present on local disk, not loaded;
- `Cold` — only in the archive (e.g. NFS), must be pulled before use.

The registry is a JSON file (atomic temp-file-then-rename `save()`; corrupt JSON
rebuilds empty rather than panicking). At startup `reconcile()` walks the local
and archive Ollama manifest trees and re-tiers records to match on-disk reality
(including demoting a model whose local copy vanished out-of-band). `register_external`
/ `register_diffusiongemma_from_env` track non-Ollama models (a `llama-diffusion`
daemon model) that `reconcile()` deliberately leaves alone.

`warm_eviction_candidates()` is the LRU candidate set the eviction logic consumes
(warm, non-protected, Ollama-managed only). Protected models — by per-record flag
or the configured `MODEL_PROTECTED` set — are never demoted to `Cold`
(`set_tier` refuses it).

### Memory / residency management

> **Spec mapping.** The diagram labels a "**Memory Coordinator**" box
> ("substrate-aware", "SeparateCeilings", "UnifiedPool", "admission",
> "tier-aware eviction"), a "**Clean-Swap Launcher**" box ("teardown →",
> "verify release →", "orphan force-kill", "false-OOM guard",
> "launch w/ explicit `-c`"), and a "**Mode Controller**" box. As of
> chord-proxy 1.1.0 all three ship in [`src/serving/`](../src/serving) —
> `SeparateCeilings`, `UnifiedPool`, the `VramResidencyManager` coordinator, the
> `clean_swap` barrier with `verify_release` (orphan kill + false-OOM guard), and
> the persisted `ModeController` are all real types. The serving subsystem doc
> ([serving.md](serving.md)) walks through them in detail.

Residency / memory behaviour spans the VRAM serving layer and the storage layer:

- **VRAM Memory Coordinator** — [`serving::residency::VramResidencyManager`](../src/serving/residency.rs)
  owns the resident set, in-flight reservations, the pinned chat model, and the
  operating mode behind one lock. `register_resident` is the admission entry: it
  sizes admissible free VRAM through the active substrate accounting model
  ([`serving::memory_model`](../src/serving/memory_model.rs): `SeparateCeilings`
  for a fixed carveout, `UnifiedPool` for dynamic-GTT), asks
  [`serving::eviction::plan_admission`](../src/serving/eviction.rs) for a
  tier-aware plan (transient → keep-warm LRU; the `Tier::Chat` pin is never
  evicted), claims victims under the lock to avoid a double-eviction race, and
  reclaims their VRAM outside it. Any unreadable counter is fail-safe "won't fit".
- **Clean-Swap Launcher** — [`serving::swap::clean_swap`](../src/serving/swap.rs)
  enforces teardown → verify-release → launch. [`serving::release_verify::verify_release`](../src/serving/release_verify.rs)
  confirms the device returned to `baseline + tolerance`, force-kills an orphaned
  backend, and refuses to launch onto a polluted device (the false-OOM guard).
  Every swap launches with an explicit `-c <n_ctx>`
  ([`serving::launcher`](../src/serving/launcher.rs) builds the command; a missing
  profile ctx is filled by `default_ctx_for_footprint`).
- **Mode Controller** — [`serving::mode::ModeController`](../src/serving/mode.rs)
  with `OperatingMode::{AssistantLive, BatchCoder}`; switching off assistant-live
  requires explicit confirm, and the mode is persisted via
  `residency::read_persisted_mode` so it survives a restart.
- **Tier-aware eviction (storage)** — [`models::eviction`](../src/models/eviction.rs).
  `evict_to_archive` performs an archive-first, verify-then-delete, GC-aware
  warm → cold eviction. `run_eviction_sweep` runs a cooldown pass then a
  disk-pressure pass (LRU above `MODEL_DISK_PRESSURE_PERCENT`); `evict_for_space`
  is the targeted pre-pull variant. A shared `DiskOpLock` serialises destructive
  disk ops. This is the **disk** tier (warm↔cold), distinct from the VRAM ceiling.
- **Archive pull / admission-by-space** — [`models::transfer`](../src/models/transfer.rs).
  `PullCoordinator::ensure_local` is the cold → warm copy with a disk precheck that
  fails fast, per-model dedup locking, timeout, and partial-file cleanup.

See [serving.md](serving.md) for the full module-by-module walkthrough.

### Control API (operator / dashboard surface)

**What it is.** The second listener — [`control`](../src/control.rs),
`build_control_router`. All endpoints require the same JWT as the proxy. It
exposes the registry and tiering controls:

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/models` | list every registry record |
| GET | `/api/models/:name` | single model detail (404 unknown) |
| POST | `/api/models/:name/archive` | warm → cold (`evict_to_archive`); Hot → 409, protected → 403 |
| POST | `/api/models/:name/pull` | cold → warm (`ensure_local`); insufficient space → 507 |
| POST | `/api/models/:name/protect` | toggle/set the protected flag |
| GET | `/api/storage` | local + archive disk usage |
| POST | `/api/models/sweep` | trigger an eviction sweep (202 Accepted, runs async) |
| GET | `/health` | version metadata (no auth) |

### Agentic loop

**What it is.** A guarded LLM↔tool execution loop —
[`agentic`](../src/agentic), entry type `AgenticExecutor`
([`loop_runner.rs`](../src/agentic/loop_runner.rs)), reached via
`POST /v1/agent/execute`. It runs the model↔tool loop up to `max_tool_calls`
iterations and returns an `AgenticResponse` whose execution log is **metadata
only** — tool arguments and raw results never cross the wire.

Five security guards run at every step (each emits a `SecurityEvent`):

- `PermissionEnforcer` ([permissions.rs](../src/agentic/permissions.rs)) — per-user
  allowed-tool sets;
- `ArgumentGuard` ([argument_guard.rs](../src/agentic/argument_guard.rs)) — blocks
  shell / SQL injection and credential patterns in tool arguments;
- `ResultGuard` ([result_guard.rs](../src/agentic/result_guard.rs)) — sanitizes
  suspicious tool results;
- `ResponseGuard` ([response_guard.rs](../src/agentic/response_guard.rs)) — detects
  cross-step injection chains;
- `BehavioralMonitor` ([behavioral_monitor.rs](../src/agentic/behavioral_monitor.rs))
  — flags internal-data → external-tool exfiltration patterns.

Within the loop, `AgenticModelRouter`
([model_router.rs](../src/agentic/model_router.rs)) escalates **once** from the
fast model (`CHORD_FAST_MODEL`) to the deep model (`CHORD_DEEP_MODEL`) when a
complexity heuristic fires (tool-result count, total chars, or reasoning
keywords) — capped at one escalation per execution so VRAM isn't thrashed.
Progress is streamed as SSE `ProgressEvent`s
([streaming.rs](../src/agentic/streaming.rs)) when the caller sets `stream: true`.

### Search harness (Harness-1)

**What it is.** A stateful research state machine inside the agentic executor —
[`harness`](../src/harness), type `SearchHarness` over a `WorkingMemory`
(candidate pool, curated set, evidence graph, verification records). The harness
holds the bookkeeping; the model emits one `HarnessAction` per turn and the
harness renders a compact observation back. A `SearchBudget` caps turns.

The `ResearchDetector` ([detector.rs](../src/harness/detector.rs)) decides whether
a query warrants the full harness (explicit `/research` command, intent keywords,
or a complexity score above `HARNESS_TRIGGER_THRESHOLD`) versus a plain search.
When it fires, `harness_integration` rotates VRAM via `HarnessVramManager` through
the sequence `personality → search-model → synthesis → personality`, then builds a
citation-style `SynthesisPrompt` from the curated documents. Every VRAM-rotation
failure degrades gracefully (`SwapOutcome::Fallback` / `Degraded`) — never a crash.

## Configuration surface

All operational knobs come from env (parsed in [`config.rs`](../src/config.rs)) —
nothing infrastructure-specific is hardcoded. Key variables: `CHORD_PROXY_PORT`,
`CHORD_CONTROL_PORT`, `CHORD_JWT_SECRET`, `CHORD_LLM_URL`, `CHORD_MODEL_ALIASES`,
`MODEL_LOCAL_PATH` / `MODEL_ARCHIVE_PATH` / `MODEL_REGISTRY_PATH`,
`MODEL_PROTECTED`, `MODEL_DISK_PRESSURE_PERCENT`, `MODEL_WARM_COOLDOWN_HOURS`,
`CHORD_FAST_MODEL` / `CHORD_DEEP_MODEL`, and the `HARNESS_*` harness knobs.
