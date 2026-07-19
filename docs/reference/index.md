# Reference

One page per subsystem, derived from the code knowledge graph (17 subsystems,
3,182 nodes) and verified against source. Node counts are real KG counts at the
analyzed ref.

| Subsystem | KG nodes | Source | Page |
|---|---|---|---|
| agentic | 539 | `src/agentic/` | [agentic.md](agentic.md) |
| models | 374 | `src/models/` | [models.md](models.md) |
| serving | 361 | `src/serving/` | [serving.md](serving.md) |
| crates (chord-tui, chord-secrets) | 339 | `crates/` | [chord-tui.md](chord-tui.md) |
| harness | 229 | `src/harness/` | [harness.md](harness.md) |
| snap | 185 | `src/snap/` | [snap.md](snap.md) |
| sweep_status | 132 | `src/sweep_status/` | [sweep_status.md](sweep_status.md) |
| router | 111 | `src/router/` | [router.md](router.md) |
| routes | 107 | `src/routes.rs`, `src/control.rs` | [routes.md](routes.md) |
| supervisor | 70 | `src/supervisor/` | [supervisor.md](supervisor.md) |
| mcp_proxy + catalog | 37 + 34 | `src/mcp_proxy.rs`, `src/catalog.rs` | [mcp_proxy.md](mcp_proxy.md) |
| gpu_exclusive | 40 | `src/gpu_exclusive.rs` | [gpu_exclusive.md](gpu_exclusive.md) |

Not given standalone pages (covered inline where noted):

- **audit** (59 nodes, `src/audit.rs`) — covered in [routes.md](routes.md#audit-logging)
- **config** (50 nodes, `src/config.rs`) — covered in the per-page Configuration
  sections and [getting-started](../getting-started.md#minimal-configuration)
- **tests** (185 nodes, `tests/`) — integration-test code, not a runtime subsystem
- **misc** (330 nodes: `auth.rs`, `session.rs`, `coding_proxy.rs`,
  `embeddings.rs`, `diffusion.rs`, `routing/`, `validation/`, `admin/`,
  `secrets_bootstrap.rs`, …) — covered across [routes.md](routes.md),
  [models.md](models.md), and the [architecture page](../architecture.md)

## Configuration surface

44 env keys across `src/config.rs` and the feature-colocated config modules.
Names only (values are vault-materialized at runtime); each page lists its own
keys. The families:

- Core: `CHORD_PROXY_PORT`, `CHORD_CONTROL_PORT`, `CHORD_JWT_SECRET`,
  `CHORD_API_KEY`, `CHORD_LLM_URL`, `CHORD_MODEL_ALIASES`,
  `CHORD_TOOL_TIMEOUT_SECS`, `CHORD_CATALOG_CACHE_SECS`,
  `MCP_BACKEND_URL`, `MCP_BACKEND_TOKEN`
- Rate limits: `CHORD_RATE_LLM_USER`, `CHORD_RATE_TOOL_USER`,
  `CHORD_RATE_DEEP_USER`, `CHORD_RATE_LLM_GUEST`, `CHORD_RATE_TOOL_GUEST`,
  `CHORD_RATE_DEEP_GUEST`
- Model tiering: `MODEL_LOCAL_PATH`, `MODEL_ARCHIVE_PATH`,
  `MODEL_REGISTRY_PATH`, `MODEL_PROTECTED`, `MODEL_DISK_PRESSURE_PERCENT`,
  `MODEL_WARM_COOLDOWN_HOURS`, `MODEL_SWEEP_INTERVAL_SECS`,
  `MODEL_PULL_TIMEOUT_SECS`, `MODEL_ARCHIVE_COPY_TIMEOUT_SECS`,
  `MODEL_GC_MIN_AGE_SECS`, `MODEL_SOURCE_ALLOWLIST`
- Serving/VRAM: `CHORD_VRAM_FREE_SYSFS_PATH`, `CHORD_VRAM_TOTAL_SYSFS_PATH`,
  `CHORD_GTT_TOTAL_SYSFS_PATH`, `CHORD_RESIDENCY_STATE_PATH`,
  `CHORD_RESIDENCY_WAIT_THRESHOLD_MS`, `CHORD_RELEASE_BASELINE_GB`,
  `CHORD_RELEASE_TOLERANCE_GB`, `CHORD_RELEASE_TIMEOUT_MS`,
  `CHORD_RELEASE_POLL_MS`, `CHORD_SWAP_ENGINE`, `CHORD_SWAP_BASE_CTX`,
  `CHORD_SWAP_MIN_CTX`, `CHORD_SWAP_LARGE_MODEL_GB`,
  `CHORD_CHAT_PIN_MAX_COLD_LOAD_S`
- Agentic/harness: `CHORD_FAST_MODEL`, `CHORD_DEEP_MODEL`,
  `HARNESS_SEARCH_MODEL`, `HARNESS_SYNTHESIS_MODEL`,
  `HARNESS_VRAM_INTEGRATION`, `HARNESS_SWAP_TIMEOUT_SECS`, `CHORD_CONTROL_URL`
- Observability: `SNAP_POLL_INTERVAL_SECS`, `SNAP_VRAM_SAMPLE_SECS`,
  `SNAP_DATA_DIR`, `SNAP_STORAGE_LOCATIONS`, `CHORD_SNAP_PERSIST`,
  `CHORD_DATA_DIR`, `CHORD_STATE_DIR`, and the `CHORD_SWEEP_*` monitor family
- Backends: `OLLAMA_URL`, `OLLAMA_CPU_URL`, `OLLAMA_BASE_URL`, `CHORD_VLLM_URL`,
  `OPENROUTER_URL`, `OPENROUTER_API_KEY`, `OPENROUTER_API_KEY_ENV_NAME`
- Features: `EMBED_*` (embeddings), `DIFFUSION_*` (managed diffusion daemon),
  `PERSONAL_BACKEND_URL` / `PERSONAL_BACKEND_TOKEN` (federation)
- Isolation: `CHORD_OUTBOUND_PROXY`, `CHORD_RUNTIME_TELEMETRY_OFF`,
  `CHORD_IP_BIN`, `CHORD_NFT_BIN`, `CHORD_ALLOW_UNISOLATED`
- Secrets: `INFISICAL_URL`, `INFISICAL_CLIENT_ID`, `INFISICAL_CLIENT_SECRET`,
  `CHORD_INFISICAL_PROJECT_ID`, `CHORD_INFISICAL_ENVIRONMENT`,
  `CHORD_INFISICAL_SECRET_PATH`

## Legacy pages

Earlier generated shards kept for continuity, superseded by the pages above:
[overview.md](overview.md), [architecture.md](architecture.md),
[mcp-tool-dispatch.md](mcp-tool-dispatch.md),
[idle-mode-free-the-host-for-the-compiler-bld-09.md](idle-mode-free-the-host-for-the-compiler-bld-09.md),
[secrets-own-<secret-manager>-client.md](secrets-own-<secret-manager>-client.md),
[status.md](status.md), [test-results.md](test-results.md),
[documentation.md](documentation.md), [license.md](license.md).
