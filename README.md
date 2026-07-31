<h1 align="center">Chord</h1>

<p align="center"><em>The Lumina Constellation's inference proxy and orchestrator: one Rust process that routes LLM traffic, manages model storage and VRAM lifecycles, and dispatches MCP tools for the whole fleet.</em></p>

<p align="center">Rust · 133 modules · 3,182 KG nodes · 2,587 functions · 44 config keys · analyzed <code>f44e483</code></p>

<p align="center"><a href="docs/index.md">Docs</a> · <a href="docs/getting-started.md">Getting started</a> · <a href="docs/reference/index.md">Reference</a> · <a href="docs/architecture.md">Architecture</a> · <a href="docs/guides/index.md">Guides</a></p>

---

## What is Chord

Chord (`chord-proxy`) is the always-on backbone that sits between the fleet's agents (Lumina, Harmony, the Terminus build pipeline) and its local inference hardware. It exposes an OpenAI-compatible `POST /v1/chat/completions` front door plus an MCP tool surface (`/v1/tools/list|call|discover`), and behind those endpoints it owns every decision the callers shouldn't have to make: which backend serves a model, whether the model must first be pulled from cold archive storage, whether the request may run at all while another job holds the GPU, and whether a per-request `thinking` hint can be honored for the target model.

It is an *orchestrator*, not just a passthrough. The `models` subsystem tracks every known model across three storage tiers (hot / warm / cold) in a persistent registry, with background disk-pressure eviction, cooldown demotion, orphan-blob GC, transparent cold→warm pulls on request, and — since TIER-05 — a **cold-archive disk-quota with score-based pruning**: when the NFS archive (grown monotonically by the Ask-4 auto-promotion loop) exceeds its quota, the least-qualified cold models (lowest measured `assistant_avg_value`, falling back to lowest practical `fit_score` past a grace window) are pruned back under quota, GC-aware and dry-run-by-default. The `serving` subsystem reads per-model serving profiles and performs clean VRAM swaps (teardown → verify-release → launch) with substrate-aware memory accounting, while `supervisor` gives each launched runtime a scrubbed environment and a fail-closed network namespace. On-demand backends are started when routed to and stopped when idle — no perpetual GPU holds, including the Chord-managed DiffusionGemma daemon.

Around the core proxy sit the agent-facing capabilities: a guarded agentic tool-calling loop (`agentic`, five security guards, one-shot model escalation), a stateful research search harness (`harness`), an SLM router that picks inference destinations for the documentation engine (`router`), fleet-data-driven coding-model selection (`POST /v1/coding/select`), local-first embeddings with fallback (`/v1/embeddings`), and an operator control API (second listener) for model tiering, idle mode, and the SNAP observability surface. A ratatui control TUI (`chord-tui`) and a shared <secret-manager> client (`chord-secrets`) live in the same workspace.

## Architecture

Derived from the code knowledge graph (17 subsystems, 114 cross-subsystem call edges). Node labels carry real KG symbol counts.

```mermaid
flowchart LR
    routes["routes (107)<br/>proxy port"]
    control["control API<br/>+ admin idle"]
    mcp["mcp_proxy (37)<br/>+ catalog (34)"]
    agentic["agentic (539)"]
    harness["harness (229)"]
    models["models (374)"]
    serving["serving (361)"]
    supervisor["supervisor (70)"]
    snap["snap (185)"]
    gpu["gpu_exclusive (40)"]
    router["router (111)"]
    sweep["sweep_status (132)"]

    routes -->|tools list/call/discover| mcp
    routes -->|agent/execute| agentic
    routes -->|tier lookup + pull + backend routing| models
    routes -->|thinking-mode resolution| serving
    routes -->|inference gate| gpu
    routes -->|merged /v1/sweep/*| sweep
    agentic -->|guarded tool calls| mcp
    agentic -->|research episodes| harness
    serving -->|scrubbed env + netns| supervisor
    router -->|backend catalogue| models
    control -->|archive / pull / protect / sweep / gc| models
    control -->|vram / inventory / analytics| snap
```

`mcp_proxy` falls back to the in-process `terminus-rs` Rust tool registry when the MCP backend is unreachable; an optional second, unfiltered proxy federates a personal tool backend under `/v1/personal/tools/*` when `PERSONAL_BACKEND_URL` is set.

## Subsystems

| Subsystem | KG nodes | What it does | Reference |
|---|---|---|---|
| `agentic` | 539 | Guarded LLM↔tool loop: five security guards, one-shot fast→deep model escalation, SSE progress | [reference/agentic](docs/reference/agentic.md) |
| `models` | 374 | Storage tiering: persistent registry, eviction/GC, archive pulls, backend catalogue + on-demand lifecycle | [reference/models](docs/reference/models.md) |
| `serving` | 361 | Serving profiles, runtime launcher, VRAM residency/admission, clean swap, mode controller | [reference/serving](docs/reference/serving.md) |
| `crates/` | 339 | `chord-tui` control TUI client + `chord-secrets` <secret-manager> Universal Auth client | [reference/chord-tui](docs/reference/chord-tui.md) |
| `harness` | 229 | Harness-1 research state machine: working memory, curated evidence, VRAM rotation | [reference/harness](docs/reference/harness.md) |
| `snap` | 185 | Observability: real VRAM reader, engine health poller, model inventory, request analytics, vLLM adapter | [reference/snap](docs/reference/snap.md) |
| `sweep_status` | 132 | "Is the benchmarking sweep healthy" monitor: working/stuck/idle verdicts from GPU + DB + systemd signals | [reference/sweep_status](docs/reference/sweep_status.md) |
| `router` | 111 | SLM router: destination decisions (local high-context / local cheap / frontier-free) + routing-quality eval | [reference/router](docs/reference/router.md) |
| `routes` | 107 | Proxy-port HTTP surface: chat completions, tools, agent, embeddings, infer, coding select, GPU-exclusive | [reference/routes](docs/reference/routes.md) |
| `supervisor` | 70 | Launch posture: env scrubbing + fail-closed per-runtime network namespaces with nftables egress filtering | [reference/supervisor](docs/reference/supervisor.md) |
| `gpu_exclusive` | 40 | GPU handoff lock: external sweeps take the GPU without stopping Chord; inference paths gate with 503 | [reference/gpu_exclusive](docs/reference/gpu_exclusive.md) |
| `mcp_proxy` + `catalog` | 71 | MCP backend proxy with Rust-tool fallback registry and merged, allowlisted tool catalog | [reference/mcp_proxy](docs/reference/mcp_proxy.md) |

Smaller units (`audit`, `config`, `misc`: auth/session/validation/diffusion/embeddings/coding_proxy) are covered inside the pages above — see the [reference index](docs/reference/index.md).

## Quick start

```sh
# Build the workspace (root crate + chord-tui + chord-secrets)
cargo build --release

# The two root binaries
./target/release/chord-proxy --version
./target/release/batch-report --help
```

`chord-proxy` binds two listeners: the proxy port (`CHORD_PROXY_PORT`, default 9099) and the control port (`CHORD_CONTROL_PORT`, default 8090). Minimum useful configuration, by key name (values come from the vault at runtime, never inlined):

- `CHORD_JWT_SECRET` — Bearer-JWT auth for both listeners (empty disables auth for trusted single-tenant use)
- `CHORD_LLM_URL` — upstream LLM backend; unset means `/v1/chat/completions` returns 503
- `MCP_BACKEND_URL` / `MCP_BACKEND_TOKEN` — MCP tool backend (Rust fallback tools still serve without it)
- `MODEL_LOCAL_PATH` / `MODEL_ARCHIVE_PATH` / `MODEL_REGISTRY_PATH` — the storage-tiering roots
- Cold-archive quota (TIER-05, score-based pruning of the cold NFS archive):
  - `MODEL_ARCHIVE_QUOTA_DRY_RUN` — default **`1` (ON)**: log a `[cold-quota]` would-prune plan and delete nothing; set `0` to arm deletion
  - `MODEL_ARCHIVE_QUOTA_PERCENT` — quota as a percent of the archive mount (default **80**); `MODEL_ARCHIVE_QUOTA_GB` — absolute GiB quota, takes precedence when set
  - `MODEL_ARCHIVE_QUOTA_GRACE_DAYS` (default 14), `MODEL_ARCHIVE_QUOTA_MIN_KEEP` (default 20), `MODEL_ARCHIVE_QUOTA_MAX_PASS_FRACTION` (default 0.25), `MODEL_ARCHIVE_QUOTA_MIN_SANE_GB` (default 10), `MODEL_ARCHIVE_QUOTA_FALLBACK_FIT` (default true) — the safety gates
- Assistant-mode resident set (TRTR-07 — three models held resident so a tool turn does not cold-load several GB, released on a mode swap). **This is the single owner of VRAM residency** — CHRD-PIN-01 retired the older `MODEL_KEEP_RESIDENT` / `keep_alive:-1` pinning path, and Chord unloads any leftover indefinite pin at startup (`CHORD_RESIDENT_PIN_HORIZON_DAYS`, default 365):
  - `CHORD_RESIDENT_SET_ENABLED` — default **on**; `0` disables residency entirely (every role reports `disabled`)
  - `CHORD_RESIDENT_ROLE_PERSONALITY` / `CHORD_RESIDENT_ROLE_ROUTER` / `CHORD_RESIDENT_ROLE_EMBEDDING` — the **alias key** each role resolves through (defaults `lumina-fast` / `lumina-fast` / `lumina-embed`). Personality resolves through the **interactive** tier: `lumina-deep` weights responsiveness at 0.05 and so selects for depth over turn latency, which is the wrong target for the chat turn a human waits on. Personality and router sharing an alias is not a degradation — a shared model is held once. **A role whose alias does not resolve degrades LOUDLY and pins nothing — uniformly, with no per-role fallback**: it warns at `WARN` naming the role, the alias that failed, and the fix, then holds no model and exempts nothing. There is deliberately no code-level substitute (an earlier revision fell back to `EMBED_LOCAL_MODEL` for the embedding role; that quietly substituted a different model than the operator configured, i.e. a second implicit source of residency truth, and was removed). If the embedding role should hold a model, configure `lumina-embed` in `CHORD_MODEL_ALIASES` — normally to the same model `EMBED_LOCAL_MODEL` names, so `/v1/embeddings` and the resident set cannot disagree. These name Chord aliases, never model names: Chord owns model selection, and the dynamic alias updater may repoint a role mid-residency (the new target is warmed, then the old one released)
  - `CHORD_RESIDENT_KEEP_ALIVE` (default `24h`), `CHORD_RESIDENT_REWARM_DEBOUNCE_SECS` (default 30 — a rapid idle-lease acquire/release cycle must not thrash-warm), `CHORD_RESIDENT_WARM_TIMEOUT_SECS` (default 300), `CHORD_RESIDENT_REFRESH_SECS` (default 300 — background reconcile tick)
  - `CHORD_RESIDENT_REASSERT_SECS` (default 3600) — how stale a role's last successful warm must be before an ordinary reconcile tick re-issues its keep-alive call. Deliberately much longer than the reconcile tick: a steady-state tick RETAINS what is already held and issues **no** warm requests, and the VRAM plan charges already-resident models nothing (they are already inside the free-VRAM reading), so residency neither oscillates nor storms
  - **Release always wins — at BOTH layers (lifecycle guard).** A warm pass is slow network I/O and cannot hold a lock across it, so every release bumps a generation counter; a pass that completes against a stale generation DISCARDS itself and installs nothing. That is the **bookkeeping** guarantee: once `release` has returned, no in-flight warm can re-install the eviction exemption or mark the set active. It is *not* by itself a GPU guarantee — a warm request already on the wire could still complete and load the model again with `keep_alive=24h`, leaving Chord reporting "released" while the shared GPU stayed occupied. So release is also authoritative at the **Ollama** layer: it cancels the pass's `CancelToken` (which aborts the in-flight HTTP request rather than merely ignoring it), it **waits** for the in-flight pass to finish and undo itself before returning — and therefore before the idle path unloads VRAM — and every model a cancelled pass had put a warm request on the wire for gets an explicit role-shaped `keep_alive: 0` **compensating unload**. Concurrent warms coalesce into a single pass rather than duplicating warm requests (and a coalesced caller's wait is bounded, so a cancelled leader can never strand its subscribers), and a refresh that fails preserves a slot that is still warm and valid instead of downgrading it
  - `CHORD_RESIDENT_RELEASE_DRAIN_SECS` (default 30) / `CHORD_RESIDENT_RELEASE_CANCEL_GRACE_SECS` (default 5) — the release wait is **hard-bounded at 35s**: graceful drain, then cancellation, then a short grace. A release that hangs is its own outage on a shared GPU, so on expiry release reports `drain_timed_out`, issues the compensating unloads itself from its pending snapshot, and returns anyway (the pass compensates too when it eventually completes; the unload is idempotent). **Limit, stated plainly:** dropping the HTTP request cannot abort a load *inside* Ollama — that is what the compensating unload exists for; the one residual is the timeout path, where a load completing after both that compensation and the idle path's own `evict_resident_models` sweep survives only until the pass itself finishes and compensates
  - **The warm requests themselves are test-covered at the production seam.** The role lifecycle is testable through a `ResidentEnv` trait, so manager-level tests alone would pass even if the real warm call became a no-op. `AppStateEnv` is therefore also exercised against a stub Ollama, asserting the request that actually goes out: the chat roles load through `/api/generate`, the embedding role through `/api/embed` (it cannot serve `/api/generate`), each naming its resolved model and carrying the configured **`keep_alive`** — without which Ollama would load the model and then drop it on its own default timer, leaving residency as bookkeeping only. Non-2xx and connection errors surface as a warm failure for that role, and a release/re-warm cycle genuinely re-issues the requests
  - Observability: **`GET /admin/resident-set`** (control port, JWT-auth) reports per role the alias, resolved model, state (`warm` / `released` / `unresolved` / `missing` / `warm-failed` / `dropped-vram` / `disabled`), `warm`, `last_used`, and size, plus the residency exemption in force
  - Mode swap: entering idle (`POST /admin/idle` — Harmony's BLD-11 idle lease, MINT's sweep) releases the set **after** the in-flight drain, making the models immediately reclaimable; `POST /admin/activate` re-warms. Fail-soft throughout — an unresolvable alias, a never-pulled model, an unreachable Ollama, or a VRAM shortfall degrades that role (personality > router > embedding) and is logged, never blocking startup or refusing a turn
- `INFISICAL_URL` + `INFISICAL_CLIENT_ID` / `INFISICAL_CLIENT_SECRET` — optional secrets bootstrap at startup

Note: the build depends on `terminus-rs` from a private Cargo registry (`registry = "gitea"` in `Cargo.toml`), so it requires that registry to be configured. See [Getting started](docs/getting-started.md) for the full walkthrough and [the configuration notes](docs/reference/index.md#configuration-surface) for all 44 env keys.

## Documentation

| Page | What's in it |
|---|---|
| [docs/index.md](docs/index.md) | Documentation hub and full page inventory |
| [docs/getting-started.md](docs/getting-started.md) | Build, configure, run, verify — clone to first request |
| [docs/architecture.md](docs/architecture.md) | Full derived diagram, per-subsystem narrative, request-flow walkthrough |
| [docs/reference/index.md](docs/reference/index.md) | Per-subsystem reference pages (12) + configuration surface |
| [docs/guides/index.md](docs/guides/index.md) | Operator guides: model tiering ops, GPU-exclusive handoff, idle mode |
| [docs/serving.md](docs/serving.md) | Deep dive: the serving/VRAM subsystem (pre-existing, still authoritative) |
| [docs/egress.md](docs/egress.md) | Deep dive: runtime egress isolation scope and guarantees |
| [docs/chord-tui.md](docs/chord-tui.md) | The control TUI client |

## At a glance

- 2,587 functions · 337 structs · 92 enums · 33 traits across 133 modules (3,182 KG nodes, 7,892 edges)
- 2 root binaries (`chord-proxy`, `batch-report`) + the `chord-tui` binary; 3 workspace members
- Top call-graph hotspots: `mcp_proxy::FallbackRegistry::contains`, `snap::inventory::ModelInventory::filter`, `models::gc::lock`, `models::eviction::FsLocalEvictor::new`, `models::transfer::PullCoordinator::new`
- Failure discipline throughout: background loops are best-effort (log and continue), optional integrations fail open to "feature disabled", and launch isolation fails closed

## Contributing

Changes go through the constellation's spec/build pipeline (ingest → worktree → test gate → dual review → merge → verify). Integration tests live in [`tests/`](tests/); `cargo test --workspace` runs the default suite (the live-federation test is feature-gated behind `personal-live-test`).

## License

MIT — see [LICENSE](LICENSE).
