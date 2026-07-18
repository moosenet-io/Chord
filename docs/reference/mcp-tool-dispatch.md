## MCP tool dispatch

Beyond model serving, Chord is the constellation's authenticated MCP tool
gateway. Every tool endpoint requires JWT auth when `CHORD_JWT_SECRET` is
configured (the same secret as the LLM calls; an empty secret disables auth for
local/dev use) and is rate-limited per role. Tool-call outcomes are recorded to
a sanitized audit log that never captures tool arguments or raw query text.

### Core registry — `/v1/tools/*`

The core catalog ([`src/catalog.rs`](src/catalog.rs)) is the merge of two
sources: MCP tools from the upstream MCP backend (`MCP_BACKEND_URL`) and the
embedded `terminus-rs` Rust registry, which acts as the fallback position
(an MCP tool wins when a name collides). The catalog is cached
(`CHORD_CATALOG_CACHE_SECS`, default 5 min).

What Chord *serves* is deliberately narrower than what it can *reach*: a static
**core allowlist** ([`src/tool_allowlist.rs`](src/tool_allowlist.rs)) scopes
`/v1/tools/list`, `/v1/tools/discover`, and `/v1/tools/call` to Chord's actual
job — the build/spec-execution pipeline (`gitea_*`, `github_*`, `plane_*`),
DiffusionGemma review (`dgem_*`), and Chord's own model-serving domain
(`model_advisor_*`, `serving_profile*`, `serving_residency*`, `model_intake*`),
plus a few built-ins. Anything off the list is excluded from `list`/`discover`
**and** rejected outright by `call`, even if a caller already knows the name.
General personal-utility / secrets / fleet-ops tools are intentionally *not*
served here — that surface belongs to Lumina core talking to Terminus directly.

- `POST /v1/tools/list` — merged, allowlisted catalog.
- `POST /v1/tools/call` — execute an allowlisted tool by name.
- `POST /v1/tools/discover` — model-free keyword search over the catalog, so a
  caller assembles a small per-turn toolset instead of the whole hub.
- `POST /v1/agent/execute` — the guarded agentic tool-calling loop.

### Personal federation — `/v1/personal/tools/*`

A second, **independent and unfiltered** `McpProxy` is federated in when
`PERSONAL_BACKEND_URL` is set, pointed at the standalone `terminus_personal`
Rust MCP binary (which self-fetches its own secrets from the vault). It is
reachable *only* at `/v1/personal/tools/list` and `/v1/personal/tools/call` and
is **never** merged into the core `/v1/tools/*` catalog — the two share auth,
rate-limiting, and audit machinery, but are separate catalogs, not one relaxed
security posture. When `PERSONAL_BACKEND_URL` is unset the personal routes
return a clean `503` (never a panic or hang), and the core catalog is provably
unchanged.

### Agentic model routing (S92 hybrid)

Chord's agentic loop ([`src/agentic/model_router.rs`](src/agentic/model_router.rs))
runs on a **fast** model by default (`CHORD_FAST_MODEL`) and escalates **once**
per execution to a **deep** model (`CHORD_DEEP_MODEL`) when a turn gets complex.
The escalation decision is *hybrid*: a cheap, deterministic keyword-and-size
heuristic (a small conservative reasoning-word list, plus tool-result-count and
total-character thresholds) is complemented by a local **Supra-Router-51M**
daemon (`SUPRA_ROUTER_URL`, deployed with the loopback-bound `dgem.service`
pattern) that classifies the same turn. The router is gated by `ROUTER_MODE`,
which **defaults to `Shadow`**: today the Supra decision is computed and logged
alongside the heuristic (shadow-vs-actual agreement reporting) but the
**heuristic still drives the actual routing**, because the 51M model's license
is unresolved — only an explicit `ROUTER_MODE=active` lets the Supra decision
lead. A user `/deep` prefix forces the deep model outright, and escalation is
capped at one per execution so a single request never thrashes VRAM.

### Embeddings — `/v1/embeddings` (EMBED-01)

`POST /v1/embeddings` is an OpenAI-compatible embeddings endpoint
([`src/embeddings.rs`](src/embeddings.rs)): `input` may be a single string or
an array of strings, and the response is the standard
`{"object":"list","data":[{"object":"embedding","embedding":[...],"index":n}],"model":...,"usage":{...}}`
shape, order-preserved. Same JWT auth as every other endpoint, and counted
against the caller's LLM rate-limit budget.

Embeddings are served **local-first** from the fleet Ollama
(`EMBED_LOCAL_URL` / `EMBED_LOCAL_MODEL`, e.g. Qwen3-Embedding) and **fall
back to OpenRouter** (`EMBED_FALLBACK_MODEL`, same Qwen3-Embedding family so
vectors from either path are compatible) whenever the local backend is
unreachable, errors, times out, or returns a vector of the wrong
dimensionality. Chord never hands back a vector whose length doesn't match
`EMBED_DIM` (default 1024) — a dimension mismatch is treated exactly like a
backend failure, and a mismatch (or any other failure) on *both* paths is a
structured `502`, never a partial or garbage response. `OPENROUTER_API_KEY`
is fetched from <secret-manager> at startup (see below) and read fresh at dispatch
time — never a literal, never logged.

| Env var | Purpose | Default |
|---|---|---|
| `EMBED_LOCAL_URL` | Full URL of the local embeddings endpoint. Unset → local disabled, every request goes straight to the fallback. | *(unset)* |
| `EMBED_LOCAL_MODEL` | Model name requested from the local backend. | `qwen3-embedding` |
| `EMBED_FALLBACK_MODEL` | Model name requested from OpenRouter. Must stay the same model family as `EMBED_LOCAL_MODEL`. | `qwen/qwen3-embedding` |
| `EMBED_FALLBACK_BASE_URL` | OpenRouter API base (no `/embeddings` suffix). | `https://openrouter.ai/api/v1` |
| `EMBED_DIM` | Expected embedding dimensionality; enforced on every vector from either backend. | `1024` |
| `EMBED_MAX_BATCH_SIZE` | Maximum number of inputs accepted per request (a `400` above this). | `256` |
| `EMBED_TIMEOUT_SECS` | Per-backend request timeout. | `30` |
| `OPENROUTER_API_KEY` | Bearer key for the OpenRouter fallback. <secret-manager>-sourced (see below), never a literal. | *(unset → fallback disabled)* |

