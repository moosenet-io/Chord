# serving

Serving-profile reader + runtime launcher + VRAM residency (361 KG nodes,
`src/serving/`). The consume side of the S85 serving dimension: reads per-model
`serving_profile` rows (written by the Terminus intake harness) into an
in-memory routing map, launches the correct runtime with the right env/flags,
and manages VRAM as a first-class resource — substrate-aware accounting,
tier-aware admission/eviction planning, clean swaps with release verification,
and a persisted operating mode. The pre-existing deep dive
[../serving.md](../serving.md) remains the module-by-module walkthrough; this
page is the symbol-level summary.

## Key types and functions

| Symbol | Kind | File | What it does |
|---|---|---|---|
| `serving::profile::RoutingMap` | struct | `src/serving/profile.rs` | model_id → `RouteEntry` map loaded from the intake DB (`DbProfileSource`); empty on unconfigured DB (fail-open) |
| `serving::profile::RouteEntry::from_profile` | function | `src/serving/profile.rs` | Converts a serving-profile row into the launchable route shape |
| `serving::profile::EnvSpec::parse` | function | `src/serving/profile.rs` | Parses the profile's `env_json` (gfx override, mmap, flash-attn, rope/yarn, ctx, thinking block) |
| `serving::profile::RopeScalingMethod::as_str` | function | `src/serving/profile.rs` | RoPE scaling method names for llama.cpp `--rope-scaling` args |
| `serving::launcher::entry` | function | `src/serving/launcher.rs` | The launch entry: build command + env for the profiled runtime, spawn, health-check, fall back to `fallback_runtime` |
| `serving::launcher::build_launch_command` / `scrub_launch_env` | functions | `src/serving/launcher.rs` | Command/arg assembly from the profile row; env scrubbing via the supervisor |
| `serving::launcher::ResidencyManager` (trait) + `PassThroughResidency` | trait / struct | `src/serving/launcher.rs` | The admission seam SRV-04 defines and SRV-05 implements |
| `serving::residency::VramResidencyManager` | struct | `src/serving/residency.rs` | Owns the resident set, in-flight reservations, the pinned chat model, and the operating mode behind one lock |
| `serving::residency::ResidencyEvent::new` | function | `src/serving/residency.rs` | Residency event records (observability + persisted state) |
| `serving::memory_model::select_memory_model` / `SeparateCeilings` / `UnifiedPool` | function / structs | `src/serving/memory_model.rs` | Substrate-aware VRAM accounting: fixed carveout vs dynamic-GTT unified pool |
| `serving::eviction::plan_admission` | function | `src/serving/eviction.rs` | Tier-aware admission plan: transient → keep-warm LRU; the `Tier::Chat` pin is never evicted |
| `serving::swap::clean_swap` | function | `src/serving/swap.rs` | The swap barrier: teardown → verify-release → launch, with explicit `-c` ctx |
| `serving::release_verify::verify_release` | function | `src/serving/release_verify.rs` | Confirms VRAM returned to baseline + tolerance; orphan force-kill; false-OOM guard |
| `serving::mode::ModeController` / `OperatingMode` | struct / enum | `src/serving/mode.rs` | `AssistantLive` / `BatchCoder`; leaving assistant-live requires explicit confirm; persisted across restarts |

## How it connects

**routes** reads the `RoutingMap` on every chat completion to resolve the
per-request `thinking` hint (`resolve_thinking_request`), and **control**'s
`GET /api/models` computes `supports_thinking` from the same live map. The
launcher consumes **supervisor** for both the scrubbed runtime env
(`build_runtime_env`) and the network-namespace isolation
(`launch_isolation`/`netns`); the clean swap tears namespaces down
(`NetnsReapingTeardown`). The map itself is loaded from the intake DB
(`terminus_rs::config::intake_database_url`) in a startup task — unreachable DB
⇒ empty map, feature reports unavailable, proxy unaffected.

## Configuration

`CHORD_VRAM_FREE_SYSFS_PATH`, `CHORD_VRAM_TOTAL_SYSFS_PATH`,
`CHORD_GTT_TOTAL_SYSFS_PATH`, `CHORD_RESIDENCY_STATE_PATH`,
`CHORD_RESIDENCY_WAIT_THRESHOLD_MS`, `CHORD_RELEASE_BASELINE_GB`,
`CHORD_RELEASE_TOLERANCE_GB`, `CHORD_RELEASE_TIMEOUT_MS`, `CHORD_RELEASE_POLL_MS`,
`CHORD_SWAP_ENGINE`, `CHORD_SWAP_BASE_CTX`, `CHORD_SWAP_MIN_CTX`,
`CHORD_SWAP_LARGE_MODEL_GB`, `CHORD_CHAT_PIN_MAX_COLD_LOAD_S`.

## Notes and gaps

- The routing map is loaded **once at startup** and not hot-reloaded: a model
  reprofiled or newly validated after Chord starts is not reflected until the
  next restart.
- Every runtime binary/endpoint comes from the SRV-01 config helpers, never a
  literal (the S77 PII gate); the launcher only assembles *arguments* from the
  profile row.
- Any unreadable VRAM counter is treated fail-safe as "won't fit" — admission
  never optimistically launches on unknown headroom.
- Integration coverage lives in `tests/serving_*.rs` (launch, memory model,
  mode, residency, swap, chat pin).
