# routes

The HTTP surface (107 KG nodes for `src/routes.rs`, plus `src/control.rs` and
the merged feature routers). Chord binds two listeners from `main.rs`: the
**proxy port** (`CHORD_PROXY_PORT`, default 9099, `routes::build_router`) and
the **control port** (`CHORD_CONTROL_PORT`, default 8090,
`control::build_control_router`). Both share the same `AppState`; a control-port
bind failure warns but never takes the proxy down.

## Proxy port (`routes::build_router`)

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | no | Liveness |
| GET | `/v1/audit/summary` | no | Audit aggregates |
| POST | `/v1/tools/list` | JWT | Merged, allowlisted tool catalog |
| POST | `/v1/tools/call` | JWT | Execute a tool by name |
| POST | `/v1/tools/discover` | JWT | Search the catalog by query |
| POST | `/v1/personal/tools/list` / `.../call` | JWT | Federated personal-tool catalog (only when `PERSONAL_BACKEND_URL` is set; never merged into the default catalog) |
| POST | `/v1/agent/execute` | JWT | The guarded agentic loop |
| POST | `/v1/chat/completions` | JWT | OpenAI-compatible LLM proxy (alias rewrite, tier pull, backend routing, thinking honoring, streaming passthrough) |
| POST | `/v1/embeddings` | JWT | Local-first embeddings with OpenRouter fallback |
| POST | `/v1/infer` | JWT | One prompt → normalized per-backend metrics |
| POST | `/v1/coding/select` | JWT | Fleet-driven coding-model resolution (returns a resolution, not a completion) |
| POST | `/v1/gpu-exclusive/acquire` / `.../release` | JWT | External GPU handoff lock |
| GET | `/v1/gpu-exclusive/status` | JWT | Current lock holder |
| GET | `/v1/sweep/status` / `.../history` | no | Benchmarking-sweep health (aggregate only) |

## Control port (`control::build_control_router`)

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | no | Version metadata |
| GET | `/metrics` | no | Prometheus text-exposition application metrics (PROMEX-02, `src/metrics.rs`) |
| GET | `/api/models` / `/api/models/:name` | JWT | Registry records (incl. `supports_thinking`) |
| POST | `/api/models/:name/archive` / `pull` / `protect` | JWT | Tier operations (warm→cold, cold→warm, protection flag) |
| GET | `/api/storage` | JWT | Local + archive disk usage |
| POST | `/api/models/sweep` / `reconcile`, `/api/storage/gc` | JWT | Eviction sweep (202 async), reconcile+persist, orphan-blob GC |
| POST/GET | `/api/sweep/session*` | JWT | RESIL-02 sweep action-queue cache (durable resume) |
| POST/GET | `/admin/idle`, POST `/admin/activate` | JWT | BLD-09 idle mode: drain/release for the compiler; restore; status |
| GET | `/admin/activity` | JWT | CHORD-ACT-01: is inference actually in flight, and how long quiet |
| GET | `/api/vram`, `/api/activity`, `/api/inventory`, `/api/analytics/*` | JWT | SNAP observability (merged `snap::api::snap_routes`) |

## Key symbols

| Symbol | Kind | File | What it does |
|---|---|---|---|
| `routes::build_router` | function | `src/routes.rs` | Assembles the proxy-port router over `AppState` |
| `routes::AppState` | struct | `src/routes.rs` | The shared state: proxy, executor, rate limiter, registry, pull coordinator, evictor, routing map, personal proxy, embeddings config |
| `routes::auth_check` / `auth_error_response` | functions | `src/routes.rs` | JWT validation (`auth::validate_jwt` / `auth::extract_bearer`) and the uniform 401 |
| `routes::chat_completions` | function | `src/routes.rs` | The chat front door (see the [architecture page](../architecture.md#request-flow-the-proxy-front-door) for the nine-step flow) |
| `control::build_control_router` | function | `src/control.rs` | Assembles the control-port router |
| `rate_limiter::ProxyRateLimiter` | struct | `src/rate_limiter.rs` | Per-user daily budgets (LLM / tool / deep), 429 + `Retry-After` |
| `session::McpSession` / `SessionState` | structs | `src/session.rs` | MCP backend session handling |
| `audit::AuditLogger` | struct | `src/audit.rs` | See below |

## Audit logging

`audit::AuditLogger` (`src/audit.rs`) writes one JSONL metadata record per
request to `${CHORD_AUDIT_PATH}/…` — request type (`Llm`, `ToolList`,
`ToolCall`, `ToolDiscover`, …), outcome, duration, hashed identity. Sensitive
content (tool arguments, LLM messages, memory content) is **never** logged.
Rotation: `.1`…`.10` at 100 MiB per file. Every reachable outcome of the three
tool handlers — auth failure, rate-limit rejection, proxy success/error —
produces exactly one entry. Known gap (documented in `routes.rs`): a malformed
request body fails Axum's `Json<T>` extractor before the handler runs, so it is
not audited; closing it needs a `tower` layer ahead of the extractor (the
`AuditLayer` sketch in `src/middleware.rs` is not wired in).

## Configuration

`CHORD_PROXY_PORT`, `CHORD_CONTROL_PORT`, `CHORD_JWT_SECRET`, `CHORD_LLM_URL`,
`CHORD_MODEL_ALIASES`, `CHORD_TOOL_TIMEOUT_SECS`, the `CHORD_RATE_*` family,
`CHORD_AUDIT_PATH`, `EMBED_*` (embeddings), `PERSONAL_BACKEND_URL` /
`PERSONAL_BACKEND_TOKEN` (federation).

## Notes and gaps

- An empty `CHORD_JWT_SECRET` disables auth on both listeners — for tests and
  trusted single-tenant deploys only.
- Unset optional features consistently return 503 with a clear reason
  (`CHORD_LLM_URL` unset, personal federation unset, coding intake DB unset)
  rather than failing startup.
- The GPU-exclusive gate applies to the inference paths only; tools, health,
  and read-only endpoints keep serving while the GPU is lent out.
