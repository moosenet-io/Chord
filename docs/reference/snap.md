# snap

SNAP — the observability and inventory subsystem (185 KG nodes, `src/snap/`),
ported from the earlier harmony-chord codebase as a purely additive layer: a
read-only telemetry surface on the control port, gated by the same JWT auth as
`/api/models`, with no change to the request path. It contributes the device
truth (real VRAM reads), engine health polling, on-disk model inventory,
passive activity observation, request analytics, and a vLLM engine adapter.

## Key types and functions

| Symbol | Kind | File | What it does |
|---|---|---|---|
| `snap::config::SnapConfig` / `parse_storage_locations` | struct / function | `src/snap/config.rs` | Env-sourced config; `SNAP_STORAGE_LOCATIONS` parses `name:tier:/path` entries |
| `snap::spawn_health_monitor` | function | `src/snap/mod.rs` | Starts the background poller that fills the process-global shared state |
| `snap::state::InferenceState` | struct | `src/snap/state.rs` | The shared snapshot (engines + per-model load) the API routes read |
| `snap::health` | module | `src/snap/health.rs` | Polls only env-configured engine endpoints every `SNAP_POLL_INTERVAL_SECS` |
| `snap::vram` | module | `src/snap/vram.rs` | The *actual* GPU read: sysfs `mem_info_vram_total` with `rocm-smi --json` fallback and an Ollama-allocation roll-up |
| `snap::inventory::ModelInventory::filter` | function | `src/snap/inventory.rs` | Inventory filtering — the second-highest-ranked function in the whole repo |
| `snap::inventory::ModelRecord` | struct | `src/snap/inventory.rs` | A discovered GGUF file (quant detection) or Ollama manifest entry, with size/tier/cleanup-candidate flags |
| `snap::activity::ActivityTracker::new` | function | `src/snap/activity.rs` | Passive per-engine/per-model in-use observation derived from live state |
| `snap::analytics::RequestLogger` | struct | `src/snap/analytics.rs` | Append-only request log + imputed cloud-cost and savings summaries |
| `snap::analytics::RequestLogger::query` | function | `src/snap/analytics.rs` | The analytics read path behind `/api/analytics/*` |
| `snap::vllm::VLLMAdapter` / `VLLMConfig` | struct | `src/snap/vllm.rs` | vLLM `EngineAdapter` backend option (container lifecycle: lazy start, `/health` poll, stop) |
| `snap::api::snap_routes` | function | `src/snap/api.rs` | The six GET endpoints merged into the control router |
| `snap::storage` | module | `src/snap/storage.rs` | Persistence for SNAP state (see [../snap-persistence.md](../snap-persistence.md)) |

## Endpoints

All `GET`, control port, JWT-gated: `/api/vram`, `/api/activity`,
`/api/inventory`, `/api/analytics/requests`, `/api/analytics/cost`,
`/api/analytics/savings`.

## How it connects

`main.rs` spawns the health monitor at startup; **control** merges
`snap_routes` into the control router. SNAP deliberately does not touch the
proxy request path — harmony-chord's streaming reverse proxy and mutating
lifecycle/config endpoints were *not* ported, because chord already owns those
concerns (`routes.rs`, `mcp_proxy.rs`, `routing/`, `serving/`). `snap::vram` is
the device-truth complement to `serving::memory_model` (which is accounting);
the diffusion daemon manager reuses `snap::vllm`'s lazy-start/poll/stop pattern.

## Configuration

`SNAP_POLL_INTERVAL_SECS`, `SNAP_VRAM_SAMPLE_SECS`, `SNAP_DATA_DIR` (falls back
to `CHORD_DATA_DIR`, then the system temp dir), `SNAP_STORAGE_LOCATIONS`,
`CHORD_SNAP_PERSIST`, plus the engine endpoints it may poll: `OLLAMA_URL`,
`OLLAMA_CPU_URL`, `CHORD_VLLM_URL` (and a llama-server URL when configured).

## Notes and gaps

- Best-effort by design: with no engine URLs configured the poller simply
  records empty snapshots.
- The analytics "savings" figures are *imputed* against representative cloud
  pricing — they are estimates, not billing data.
- Wiring `RequestLogger` into the live proxy path was deferred at port time;
  check `routes.rs` for current call sites before assuming every request is
  logged to analytics.
