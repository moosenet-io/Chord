# models

Model storage tiering and backend lifecycle (374 KG nodes, `src/models/`).
Tracks every known model across three storage tiers — `Hot` (loaded), `Warm`
(on local disk), `Cold` (archive only) — in a file-backed JSON registry that
survives restarts and is reconciled against the on-disk Ollama manifest trees.
Also owns the backend catalogue and the on-demand start/idle-stop lifecycle
glue, plus fleet-data-driven coding-model selection.

## Key types and functions

| Symbol | Kind | File | What it does |
|---|---|---|---|
| `models::registry::ModelRegistry` | struct | `src/models/registry.rs` | The persistent registry over `ModelRecord`s; atomic temp-file-then-rename `save()`; corrupt JSON rebuilds empty |
| `models::registry::ModelRegistry::new` | function | `src/models/registry.rs` | Constructor (see `load_or_new` for the never-fails startup path) |
| `ModelRegistry::reconcile` / `scan_disk` / `apply_reconcile` | functions | `src/models/registry.rs` | Re-tier records to match on-disk reality; the slow manifest scan runs off-lock (`spawn_blocking`), the fast apply under it |
| `models::eviction::FsLocalEvictor::new` | function | `src/models/eviction.rs` | The local-disk evictor (top-5 call-graph hotspot) |
| `models::eviction::run_eviction_sweep` | function | `src/models/eviction.rs` | Background sweep: cooldown pass (warm→cold after inactivity) then disk-pressure LRU pass |
| `models::eviction::new_disk_op_lock` | function | `src/models/eviction.rs` | The shared `DiskOpLock` serialising all destructive disk ops (canonical order: disk_op_lock → registry) |
| `models::gc::run_gc` | function | `src/models/gc.rs` | Orphan-blob GC (MSM-03): deletes local blobs referenced by no manifest, only when provably safe |
| `models::transfer::PullCoordinator::ensure_local` | function | `src/models/transfer.rs` | Cold→warm archive pull: disk precheck, per-model dedup lock, timeout, partial-file cleanup |
| `models::transfer::StatvfsProbe` | struct | `src/models/transfer.rs` | The real free-disk probe behind admission-by-space |
| `models::backends::seed_from_env` | function | `src/models/backends.rs` | Builds the default backend catalogue (`Backend` structs: Ollama CPU tiers, unit-managed and generic on-demand llama.cpp GPU servers) |
| `models::routing::resolve_and_ensure` | function | `src/models/routing.rs` | Chat-path glue: pick the model's tagged backend, start it on demand (`terminus_rs` lifecycle), return the upstream URL; falls back to `CHORD_LLM_URL` |
| `models::routing::idle_stop_sweep` | function | `src/models/routing.rs` | Stops on-demand GPU backends past their `idle_stop_secs` — no perpetual holds |
| `models::coding_selector::DbCodeProfileSource` | struct | `src/models/coding_selector.rs` | Ranks real coder-sweep fleet data for `POST /v1/coding/select` (CPROX-02) |
| `models::work_type::WorkTypeCode` | enum | `src/models/work_type.rs` | The "shape of coding work" a caller requests instead of a hardcoded model alias |
| `models::rope_ingest::read_exact_buf` | function | `src/models/rope_ingest.rs` | GGUF metadata reading for RoPE/context ingestion |
| `models::batch_suitability` | module | `src/models/batch_suitability.rs` | Suitability scoring for batch workloads |

## How it connects

**routes** calls into this subsystem on every chat completion: registry tier
lookup, `PullCoordinator::ensure_local` on a `Cold` hit, then
`resolve_and_ensure` for the backend. **control** drives the operator surface
(`/api/models*`, `/api/storage*`): archive/pull/protect/sweep/reconcile/GC map
directly onto `evict_to_archive`, `ensure_local`, registry flags, and `run_gc`.
**router** (the SLM router) resolves its destinations from the same
`seed_from_env` catalogue. `main.rs` spawns the eviction sweep and idle-stop
sweep loops. The out-of-process `deploy/model-storage-manager/` service drives
reconcile → sweep → gc on a timer through the control API.

## Configuration

`MODEL_LOCAL_PATH`, `MODEL_ARCHIVE_PATH`, `MODEL_REGISTRY_PATH`,
`MODEL_PROTECTED`, `MODEL_DISK_PRESSURE_PERCENT`, `MODEL_WARM_COOLDOWN_HOURS`,
`MODEL_SWEEP_INTERVAL_SECS`, `MODEL_PULL_TIMEOUT_SECS`,
`MODEL_ARCHIVE_COPY_TIMEOUT_SECS`, `MODEL_GC_MIN_AGE_SECS`,
`MODEL_SOURCE_ALLOWLIST`, `OLLAMA_URL`, `OLLAMA_CPU_URL`,
`OPENROUTER_API_KEY_CHORDHARMONY` (Owl Alpha registration is additionally gated
behind `OPENROUTER_OWL_ALPHA_ENABLED=1`).

## Notes and gaps

- Every background pass is best-effort: reconcile/persist/eviction/GC failures
  are logged and the loop continues; nothing here can abort startup.
- `MODEL_WARM_COOLDOWN_HOURS=0` disables cooldown eviction entirely (warned at
  startup).
- Protected models (per-record flag or `MODEL_PROTECTED`) can never be demoted
  to `Cold`; `MODEL_PROTECTED` is re-applied authoritatively on every reconcile
  (MSM-05).
- Non-Ollama registrations (DiffusionGemma, the opt-in OpenRouter Owl Alpha
  entry) are deliberately left alone by `reconcile()`.
- This page does not cover the VRAM side — residency, admission, and clean
  swaps live in [serving.md](serving.md).
