# Chord — Serving & Coordinator Subsystem

The [architecture diagram](../assets/architecture.svg) names three serving boxes —
**Memory Coordinator**, **Clean-Swap Launcher**, and **Mode Controller** — under
the headings SRV-11/12/13. As of chord-proxy **1.1.0** all three ship as real
code in [`src/serving/`](../src/serving). This document maps each box to its
modules, structs, and functions.

The serving module ([`serving::mod`](../src/serving/mod.rs)) is the *consume* side
of the S85 serving dimension: the shared profile/runtime types live in
`terminus_rs::intake::serving`, the harness writes the `serving_profile` rows, and
Chord reads them into a [`profile::RoutingMap`](../src/serving/profile.rs) and
launches the correct runtime per model. Layered above the profile reader are the
three coordinator subsystems below.

## Storage residency vs. VRAM residency

Chord still distinguishes two notions of "memory":

1. **Storage residency** — is a model `Hot` (VRAM), `Warm` (local disk), or
   `Cold` (archive)? Owned by
   [`models::registry::ModelRegistry`](../src/models/registry.rs) via the
   `StorageTier` enum; promotion (`models::transfer`) and warm↔cold demotion
   (`models::eviction`) are unchanged.
2. **VRAM residency** — which models are loaded on the GPU *right now*, and which
   one may be admitted next? This is what the new
   [`serving::residency`](../src/serving/residency.rs) Memory Coordinator owns.

---

## Box 1 — Memory Coordinator (SRV-11)

Modules: [`memory_model`](../src/serving/memory_model.rs),
[`residency`](../src/serving/residency.rs), [`eviction`](../src/serving/eviction.rs).

### Substrate-aware accounting
[`memory_model.rs`](../src/serving/memory_model.rs) is the substrate model. It
classifies the host once at boot (`classify_substrate` → `Substrate::{FixedCarveout,
DynamicGtt}`) from a `SubstrateInfo` (BIOS VRAM carveout, GTT total, system RAM),
and picks the accounting policy with `select_memory_model`. The `MemoryModel` trait
answers one question — *how much free memory may a candidate in a given `Pool`
(`Vram`/`SystemRam`) draw from?* — via two implementations:

- **`SeparateCeilings`** — ACTIVE under a fixed BIOS carveout: a GPU candidate
  checks GPU free, a CPU candidate checks CPU free, and the pools never cross.
- **`UnifiedPool`** — for dynamic-GTT substrates: every resident (GPU *or* CPU)
  draws from one physical pool, so the admissible free for any candidate is the
  physical total minus the combined draw of all residents — the cross-pool effect
  `SeparateCeilings` cannot see.

A `MemorySnapshot` carries live free counters; **any unreadable counter is `None`,
which the model turns into a fail-safe "won't fit"** rather than guessing — it never
risks an OOM launch on a counter it could not read.

### Admission ceiling (the coordinator)
[`residency::VramResidencyManager`](../src/serving/residency.rs) is the
`MemoryCoordinator`. It holds the resident set, in-flight **reservations** (VRAM an
admission has committed to but not yet launched, netted out of free VRAM so two
concurrent admissions cannot double-spend the same GB), the pinned chat model, and
the current `OperatingMode` — all behind one mutex, so admission is decide-and-claim
with no read-decide-write race. Free VRAM and warm-launch are injected through the
`WarmLauncher` trait (production wires the sysfs counter `config::read_free_vram_gb`
+ the SRV-04 launcher; tests drive it deterministically with no GPU).

`register_resident` is the admission entry: it computes the admissible free GB for
the candidate's pool via the active `MemoryModel` (`admissible_free_gpu`, reservations
netted out), asks the eviction policy for a plan, **claims** the eviction targets
under the lock (removing them from the resident map so a concurrent admission cannot
re-plan the same victim — the double-eviction race), then reclaims their VRAM outside
the lock. Sanitized `ResidencyEvent`s (`admit`/`reuse`/`queue`/`evict`/
`admission-denied`) carry only a model id, tier, and decision word — never an endpoint,
host, or path (S6/S77). State is persisted atomically to the residency state file
(`config::residency_state_path`).

### Tier-aware eviction
[`eviction.rs`](../src/serving/eviction.rs) is the pure, deterministic eviction
policy. `plan_admission(need_gb, free_gb, residents)` returns an `EvictionPlan`:

- **`Fits { transient_first }`** — the candidate fits now, possibly after evicting
  the listed `Tier::Transient` residents (cheap build/validation models, evicted
  first); an empty list means it already fit.
- **`RequiresKeepWarmEviction { transient_first, keep_warm_lru }`** — transients
  alone are not enough; admitting requires evicting `Tier::KeepWarm` residents. The
  caller must QUEUE first (bounded wait) and only then evict these LRU-ordered
  keep-warm targets.
- **`CannotAdmit`** — even evicting every non-pinned resident cannot make room (or
  the footprint is unknown). The caller denies and **never evicts the pinned chat
  model**.

`Tier::Chat` is the load-bearing safety invariant: `Tier::is_evictable()` is `false`
for the chat tier, so the live chat-role model is never evicted while serving. The
policy is clock-free — the caller stamps each `ResidentView` with a monotonic last-use
tick, lower = LRU target — so it stays pure and deterministic.

---

## Box 2 — Clean-Swap Launcher (SRV-12)

Modules: [`swap`](../src/serving/swap.rs),
[`release_verify`](../src/serving/release_verify.rs),
[`launcher`](../src/serving/launcher.rs).

### The verified clean-swap barrier
[`swap::clean_swap`](../src/serving/swap.rs) is the only path that brings a model
resident, and it enforces **teardown → verify-release → launch** in order. A
`SwapRequest` names the outgoing model (`from`, or `None`), the incoming model
(`to`), and the incoming model's explicit context. The barrier:

1. Tears the outgoing backend down via the `Teardown` trait (signal stop + wait for
   process exit). A failure here is `SwapError::TeardownFailed` and it **never
   proceeds to launch**.
2. Verifies the device actually released (below) — including force-killing an orphan.
3. Only on a verified-clean device launches the incoming model via the
   `CleanLauncher` trait with an **explicit `-c <n_ctx>`** (the
   `launched_with_explicit_ctx` flag on every `SwapEvent` is always true on a
   completed swap). When the profile carries no context,
   `default_ctx_for_footprint` computes a safe explicit one (reduced for large
   models), so a swap never relies on a runtime's implicit default.

### Verify-release + orphan kill + false-OOM guard
[`release_verify::verify_release`](../src/serving/release_verify.rs) is the guard.
The `DeviceProbe` trait reports memory currently *in use* on the device; at/below
`baseline + tolerance` (from `ReleaseConfig`, sourced from config — no literals) the
device counts as released. If memory is still held, `is_orphan_present` detects an
orphaned backend process (e.g. a crashed `llama-server` still holding the device) and
`force_kill_orphan` is the escalation step. **An unreadable counter is treated as
"not released" (fail-safe — never assume a clean device we cannot measure).** If the
device will not return to baseline even after escalation, the barrier refuses to
launch (`SwapError::DeviceNotReleased`) — this is the **false-OOM guard**: it would
rather refuse than launch onto a polluted device and trigger a spurious OOM. All
events (`device_released`, `escalation_required`, the explicit ctx) are sanitized.

### The runtime launcher
[`launcher.rs`](../src/serving/launcher.rs) builds the actual launch command.
`build_launch_command` produces a pure-data `LaunchCommand` (binary, argv, env pairs,
health endpoint) from a model's serving profile — the binary and endpoint come from
config helpers, never literals, and the only env key it sets is the ROCm gfx override
(`gfx_override_version`). `LaunchError` is genericized (S77): `UnknownModel` (no
profile row — Chord will not guess a runtime), `RuntimeNotConfigured`,
`AllRuntimesFailed` (best + fallback exhausted), `KeepWarmMustUseResidency` (a
keep-warm model must be admitted through the coordinator, never inline cold-launched),
and `CpuModelTooLarge`. A `ServeHandle` records which runtime ultimately served (so a
fallback is observable) and whether the serve came from a warm residency slot.

---

## Box 3 — Mode Controller (SRV-13)

Module: [`mode`](../src/serving/mode.rs) (+ persistence in `residency`).

[`mode::ModeController`](../src/serving/mode.rs) is the real, persisted mode
controller. `OperatingMode` is an explicit enum with two regimes:

- **`AssistantLive`** ("assistant-live") — the default. The chat model is pinned +
  resident and coders swap into the leftover VRAM around it.
- **`BatchCoder`** ("batch-coder") — no pinned assistant; the full GPU is given to
  one-at-a-time coder swaps.

The controller reasons over per-pool ceilings: `coder_headroom_gb` returns the GPU a
coder model may use under the current mode (full ceiling in batch-coder; the
assistant's leftover in assistant-live), `coder_fits` tests a footprint against it,
and `oversize_reason` gives a clear rejection when a coder is too big for
assistant-live but would fit batch-coder. `request_switch` returns a `ModeAction`
(`NoChange`, `UnpinAssistant` when leaving assistant-live, `PinAssistant` when
entering it); **switching OFF assistant-live without an explicit `confirm` is refused**
(`ModeError::ConfirmRequired`) so live Lumina is never silently dropped.

Mode is persisted: the `VramResidencyManager` holds the active `OperatingMode`,
`switch_mode` / `restore_mode` mutate it and write the state file, and the free
function `read_persisted_mode(path)` restores the mode across a restart so the chosen
operating regime survives a reboot.

---

## Summary: spec vs. shipped (chord-proxy 1.1.0)

| Diagram box / annotation | Present? | Backing code |
|--------------------------|----------|--------------|
| Routing: backend-per-model | ✅ | `models::routing::resolve_and_ensure` |
| Routing: chat-role pin | ✅ | `routing::assistant_profile::decide_chat_role` |
| Serving-profile reader + per-model launch | ✅ | `serving::profile`, `serving::launcher` |
| Substrate-aware accounting | ✅ | `serving::memory_model` (`SeparateCeilings` / `UnifiedPool`) |
| Memory Coordinator / VRAM admission ceiling | ✅ | `serving::residency::VramResidencyManager` |
| Tier-aware eviction (transient → keep-warm LRU, chat pinned) | ✅ | `serving::eviction::plan_admission` |
| Clean swap: teardown → verify-release → launch | ✅ | `serving::swap::clean_swap` |
| Verify release / orphan force-kill / false-OOM guard | ✅ | `serving::release_verify::verify_release` |
| Launch w/ explicit `-c` | ✅ | `serving::swap` + `serving::launcher` |
| Mode Controller (assistant-live / batch-coder) + persisted state | ✅ | `serving::mode::ModeController`, `residency::read_persisted_mode` |
| Storage tiers + warm↔cold eviction | ✅ | `models::registry`, `models::eviction` |

Legend: ✅ implemented in the extracted `src/`.

## Launch-env scrub & egress policy (S88 ISO-01)

The runtime supervisor's launch-environment scrub and egress-policy *config
surface* are documented separately in [egress.md](./egress.md). In short: every
cold-launch's env is scrubbed (minimal base + telemetry-off/offline opt-outs +
proxy-strip via `supervisor::launch_env::build_runtime_env`), and a per-launch
egress posture is declared (`Serve` → Denied, `Pull` → allow-list-or-Denied via
`supervisor::egress_policy::posture_for`). **ISO-01 is ADVISORY** (relies on tools
honouring opt-outs); the kernel netns guarantee is ISO-02 and is not yet built.

## Serving backends

The backend catalogue is defined in
[`models::backends`](../src/models/backends.rs) and seeded by `seed_from_env()`.
Each `Backend` is hardware-tagged (`gpu`/`cpu`), speaks a `BackendKind`
(`ollama` / `llama-server` / `daemon`), and is either `always_on` or **on-demand**
(started when a model tagged to it is requested, idle-stopped otherwise). A model
is mapped to a backend via `ModelRecord::backend` in the model registry
(`<path>/model-registry.json`); untagged models resolve to the default
(`ollama`) backend.

| Backend | Hardware | Kind | Managed | Notes |
|---------|----------|------|---------|-------|
| `ollama` | cpu | ollama | always-on (`ollama.service`) | primary/general; default backend |
| `ollama-cpu` | cpu | ollama | always-on (`ollama-cpu.service`) | embeddings / micro jobs |
| `lemonade-coder` | gpu | llama-server | on-demand (`lemonade-coder.service`) | dedicated GPU coder, one fixed model (ROCm b1258) |
| `llama-gpu` | gpu | llama-server | on-demand (spawned) | generic: serves any model's Ollama blob on GPU (ROCm b1258) |
| `vulkan` | gpu | llama-server | on-demand (spawned) | generic Vulkan/RADV (Mesa) build; driver-stable ROCm alternative |

### The `vulkan` (RADV / Mesa) backend

`vulkan` is a `llama.cpp` `llama-server` built with the Vulkan backend
(`-DGGML_VULKAN=ON`) against **Mesa 25.0.7 RADV** on `gfx1151`. Binary on the
Inference host: `/root/llama-vk/build/bin/llama-server` (override with
`VULKAN_LLAMA_BIN`; port defaults to `8083`, override `VULKAN_LLAMA_PORT`). It is
seeded with the validated launch flags:

```
-c 32768 --no-mmap --no-warmup -ngl 99 --host 127.0.0.1 --port <port>
```

It is a generic on-demand backend (same shape as `llama-gpu`): it loads ANY
requested model's Ollama GGUF blob on the GPU and idle-stops after 600s.

**When to use it.** Vulkan/RADV is a *driver-stable* alternative to the ROCm-only
lemonade build (b1258) — reach for it when ROCm is unavailable or unstable. It is
memory-bound like HIP/ROCm (both ~5 tok/s at 70B), so it is intended for **dense
large models in batch/async mode**, not latency-sensitive interactive traffic.

**Validation (llama3.3:70b, Q4_K_M, 42.5 GB, on `gfx1151`):**

| Metric | Result |
|--------|--------|
| Cold-load | ~13 s (`--no-warmup`) |
| Peak VRAM @ 32k ctx | 50.6 GB / 96 GB |
| Generation | 5.3 tok/s |
| Prompt | 22–24 tok/s |
| `dmesg` | clean (no GPU faults) |

Vulkan generation throughput ≈ HIP/ROCm at 70B (both memory-bound), confirming it
as a viable driver-stable fallback for dense large models.

### Tagging models to Vulkan

Vulkan is offered for the **dense-large** class (70B/32B-dense; per the
dense-retest shortlist). `backends::is_vulkan_candidate()` classifies a model name
(MoE tags such as `a3b`/`a22b`/`moe` are excluded even at large sizes), and
`ModelRegistry::tag_vulkan_candidates()` tags every present dense-large model to
the `vulkan` backend **additively** — it never overrides an existing operator tag
and is a no-op if the `vulkan` backend is absent. An operator can also tag any
model explicitly with `set_model_backend(model, Some("vulkan"))` (or by editing the
registry JSON's `backend` field).

## Test Results

Serving behaviour is covered by the `tests/serving_*.rs` integration suites
(`serving_memory_model`, `serving_residency`, `serving_launch`, `serving_swap`,
`serving_mode`, `serving_chat_pin`). Result charts are published below.

![Test results](charts/.gitkeep)
