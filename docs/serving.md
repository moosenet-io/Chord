# Chord — Serving & Coordinator Subsystem

The [architecture diagram](../assets/architecture.svg) names three serving boxes —
**Memory Coordinator**, **Clean-Swap Launcher**, and **Mode Controller** — under
the headings SRV-11/12/13. This document maps each to the **actual code in this
extracted crate** and is explicit about which pieces ship here and which do not.

> **Scope honesty.** A repo-wide search of [`../src/`](../src) finds **no**
> `MemoryCoordinator`, `SeparateCeilings`, `UnifiedPool`, `admission`,
> `substrate`, `CleanSwap`, `ModeController`, `orphan`, or `false-OOM` symbols.
> The serving *behaviour* the diagram describes is realised by the residency,
> storage-tier, and lifecycle modules below. Where a spec'd capability (a VRAM
> admission ceiling, substrate switch, false-OOM guard, persisted operating mode)
> is **not present** in this crate, that is stated plainly rather than invented.

## The residency picture that does ship

Chord separates two notions that the diagram collapses into "memory":

1. **Storage residency** — is a model `Hot` (VRAM), `Warm` (local disk), or
   `Cold` (archive)? Owned by
   [`models::registry::ModelRegistry`](../src/models/registry.rs) via the
   `StorageTier` enum.
2. **VRAM residency** — which single model is actually loaded on the GPU right
   now? Driven by [`harness::vram_lifecycle`](../src/harness/vram_lifecycle.rs)
   through Chord's lifecycle control API.

Promotion runs `cold → warm → hot`: the **pull** (cold→warm) is
[`models::transfer`](../src/models/transfer.rs); the **swap/load** (warm→hot) is
`vram_lifecycle`; the registry tracks where each model currently sits. Demotion
(`warm → cold`) is [`models::eviction`](../src/models/eviction.rs).

---

## Box 1 — "Memory Coordinator" (SRV-11)

**Diagram annotations:** substrate-aware · SeparateCeilings · UnifiedPool ·
admission · tier-aware eviction.

**What actually backs it:**

### Admission (by disk space, not VRAM ceiling)
[`models::transfer`](../src/models/transfer.rs). Before a cold model is pulled,
`PullCoordinator::ensure_local` sums the model's on-disk size from its manifest
blobs (`parse_manifest_blobs` / `ManifestBlobs`) and checks it against free space
reported by a `DiskSpaceProbe` (`StatvfsProbe` in production). Insufficient space
→ `PullError::InsufficientDiskSpace` and **nothing is written** (fail-fast). Two
requests for the same cold model are de-duplicated by a per-model async lock, the
copy is timeout-wrapped, and any partial files are cleaned up on failure or
timeout.

This is the only "admission" gate that physically exists: a model is admitted to
the **local disk** only if there is room. There is **no VRAM-headroom admission
check** in this crate — making a model VRAM-resident is a swap (below), and the
swap delegates the actual fit/OOM behaviour to the underlying backend.

### Tier-aware eviction
[`models::eviction`](../src/models/eviction.rs). This is the real, fully-tested
"tier-aware eviction" — operating on the warm↔cold **disk** tier:

- `evict_to_archive` — the safe single-model warm → cold path:
  validate Warm + non-protected → copy manifest + blobs local→archive (skipping
  blobs already in archive with a matching size) → **verify** every blob +
  manifest exist in archive with matching sizes → only then remove the local copy
  via a GC-aware `LocalEvictor` (a blob shared by another local manifest is kept)
  → update the registry to `Cold` and `save()`. A failed/partial copy leaves the
  model warm (archive-first, delete-after).
- `run_eviction_sweep` — a **cooldown pass** (archive every warm, non-protected
  model idle longer than `MODEL_WARM_COOLDOWN_HOURS`, independent of disk
  pressure; `0` disables it) followed by a **disk-pressure pass** (LRU-evict warm
  models while local usage exceeds `MODEL_DISK_PRESSURE_PERCENT`, re-checking after
  each, skipping persistently-failing candidates). Spawned on an interval from
  [`main.rs`](../src/main.rs) and also invokable via `POST /api/models/sweep`.
- `evict_for_space` — targeted pre-pull eviction: free at least *N* bytes by
  evicting LRU warm models until the probe reports enough space.
- A shared `DiskOpLock` (`new_disk_op_lock`) serialises every destructive disk op
  (sweeps, pulls, pre-pull eviction) so they never interleave.

Safety invariants enforced in code: never evict Hot / protected / non-Warm
models; no archive mounted ⇒ skip the sweep entirely (don't evict with nowhere to
put the data); GC-aware blob removal for content-addressed shared blobs.

### "substrate-aware / SeparateCeilings / UnifiedPool" — NOT in this crate
There is no substrate abstraction, no separate-ceilings-vs-unified-pool VRAM model,
and no `MemoryCoordinator` type in the extracted `src/`. The hardware distinction
that *does* exist is the coarse `Hardware::{Gpu, Cpu}` tag on each
[`Backend`](../src/models/backends.rs), used for routing — not a memory-pool
accounting model. If/when SRV-11's VRAM admission lands, the natural seams are the
`DiskSpaceProbe`-style probe abstraction in `transfer.rs` and the backend
`Hardware` tag.

---

## Box 2 — "Clean-Swap Launcher" (SRV-12)

**Diagram annotations:** teardown → · verify release → · orphan force-kill ·
false-OOM guard · launch w/ explicit `-c`.

**What actually backs it:**

### The swap mechanism
[`harness::vram_lifecycle`](../src/harness/vram_lifecycle.rs). `HarnessVramManager`
performs a model swap by calling Chord's lifecycle control API:
`SwapClient::swap(model, engine)` → `POST {control_url}/api/lifecycle/swap`, and
`restore()` → `…/api/lifecycle/restore`. A swap to a new model **evicts whatever
was previously loaded** — that is the "teardown → launch" cycle as expressed in
this crate. Each call is timeout-bounded (`HARNESS_SWAP_TIMEOUT_SECS`, default
10 s); a `tokio::sync::Mutex` serialises rotations so two swaps never race the
GPU. `current_model` tracking means a back-to-back request for an already-resident
model is an `AlreadyWarm` no-op (no redundant teardown). Every failure mode is a
graceful `SwapOutcome` (`Fallback` / `Degraded`), never a crash.

### "launch w/ explicit -c"
This annotation is real and lives in
[`models::backends::seed_from_env`](../src/models/backends.rs): the generic
on-demand `llama-gpu` backend's `LaunchSpec.args` are
`-c 32768 -ngl 999 -fa 1 --no-mmap --host 0.0.0.0 --port …`. The explicit context
size (`-c`) and `--no-mmap` (keep-warm for large models) are baked into the launch
spec, and `models::routing::to_resolved` hands that `LaunchSpec` to the
`terminus_rs` lifecycle helper that actually spawns the server.

### On-demand teardown of idle backends
[`models::routing::idle_stop_sweep`](../src/models/routing.rs) stops any
on-demand backend whose `idle_stop_secs` has elapsed since its last use (last-use
is tracked by `lifecycle::ensure_up` touching a shared file on every request).
Always-on / Ollama / daemon backends are exempt. This is the "no perpetual holds"
teardown for backends (as opposed to per-model swaps).

### "verify release → / orphan force-kill / false-OOM guard" — NOT in this crate
The *model-swap* verify-release, orphan-process force-kill, and false-OOM guard
described on the diagram are **not implemented in this extracted `src/`** — the
swap simply trusts the lifecycle control API's success/failure and maps it to a
`SwapOutcome`. The crate's verified, archive-first **storage** eviction
(`eviction::verify_archive_copy`, the verify-before-delete guard) is the only
"verify release" that ships, and it concerns disk, not VRAM. The actual
teardown/launch/verify of a backend process lives behind
`terminus_rs::intake::lifecycle` (an external dependency), not in this repo; a
false-OOM guard would belong there or in a future SRV-12 module.

---

## Box 3 — "Mode Controller" (SRV-13)

**Diagram annotations:** assistant-live (pin + swap around) · batch-coder
(full GPU, 1-at-a-time) · persisted state.

**What actually backs it:**

There is **no `ModeController` type and no persisted operating-mode state file** in
this extracted `src/`. The two operating regimes the diagram names are *expressed*
by existing mechanisms rather than centralised in one controller:

- **assistant-live ("pin + swap around")** — the chat-role pin
  ([`routing::assistant_profile`](../src/routing/assistant_profile.rs)) decides the
  resident assistant model, and `HarnessVramManager`'s
  `personality → search → synthesis → personality` rotation
  ([vram_lifecycle.rs](../src/harness/vram_lifecycle.rs)) swaps other models in and
  out *around* that pinned personality model, restoring it at the end.
- **batch-coder ("full GPU, one-at-a-time")** — the dedicated `lemonade-coder`
  GPU backend in [`backends::seed_from_env`](../src/models/backends.rs) is a
  single fixed-model `llama-server`; the on-demand `idle_stop_secs` lifecycle
  ([routing.rs](../src/models/routing.rs)) gives the GPU to one backend at a time
  and releases it when idle.

What is **missing** versus the SRV-13 spec: an explicit mode enum, an API to switch
modes, and any persisted mode state. The behaviours exist; the *controller* that
names, switches, and persists the mode does not (in this crate). The registry's
`save()` (atomic JSON persistence) is the obvious place such state would live.

---

## Summary: spec vs. shipped (this crate)

| Diagram box / annotation | Present here? | Backing code |
|--------------------------|---------------|--------------|
| Routing: backend-per-model | ✅ | `models::routing::resolve_and_ensure` |
| Routing: chat-role pin | ✅ | `routing::assistant_profile::decide_chat_role` |
| Three backend tiers (llama.cpp / ollama / CPU) | ✅ (tags) | `models::backends` (`Hardware`/`BackendKind`) |
| Tier-aware eviction | ✅ (disk warm↔cold) | `models::eviction` |
| Admission | ⚠️ disk-space only | `models::transfer::PullCoordinator` |
| VRAM swap / teardown→launch | ✅ (via control API) | `harness::vram_lifecycle::HarnessVramManager` |
| launch w/ explicit `-c` / `--no-mmap` | ✅ | `models::backends::seed_from_env` `LaunchSpec` |
| substrate-aware / SeparateCeilings / UnifiedPool | ❌ not in this crate | — |
| verify release / orphan force-kill / false-OOM guard (VRAM) | ❌ not in this crate | (storage verify-before-delete exists in `eviction`) |
| Mode Controller / persisted mode state | ❌ not as a type | behaviours via pin + lifecycle |

Legend: ✅ implemented · ⚠️ partial / different shape than spec · ❌ absent in the
extracted `src/`.
