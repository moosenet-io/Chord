# Getting started

From clone to a first authenticated request. Everything here uses only the real
binaries in `Cargo.toml` (`chord-proxy`, `batch-report`, and the workspace's
`chord-tui`) and real env keys from the source — key *names* only; values are
materialized from the vault (<secret-manager>) at runtime, never authored in files.

## Prerequisites

- Rust (the toolchain is pinned by [`rust-toolchain.toml`](../rust-toolchain.toml))
- Access to the private Cargo registry: the root crate depends on
  `terminus-rs` with `registry = "gitea"` ([`Cargo.toml`](../Cargo.toml)), so
  your `~/.cargo/config.toml` must define that registry before the build resolves
  (operator-specific: the registry index URL and token for your deployment)
- Linux for the full feature set — the network-namespace launch isolation
  (`supervisor::netns`) is `#[cfg(target_os = "linux")]` and needs
  `CAP_NET_ADMIN`; on other platforms it reports `Unsupported`

## Build and test

```sh
cargo build --release --workspace
cargo test --workspace
```

The default test suite is self-contained (mock backends, `httpmock`). One live
integration test is feature-gated: `cargo test --features personal-live-test`
calls a real federated personal backend and needs `PERSONAL_BACKEND_URL` /
`PERSONAL_BACKEND_TOKEN` set.

## Minimal configuration

`chord-proxy` starts with everything optional degraded-but-up: unset features
log a clear "disabled" line and their endpoints return 503, rather than
blocking startup. The keys that matter first:

| Key | Controls |
|---|---|
| `CHORD_PROXY_PORT` | Proxy listener (default 9099) |
| `CHORD_CONTROL_PORT` | Control/operator listener (default 8090) |
| `CHORD_JWT_SECRET` | Bearer-JWT auth for both listeners; empty disables auth (trusted single-tenant / tests only) |
| `CHORD_LLM_URL` | Upstream LLM backend for `/v1/chat/completions` |
| `CHORD_MODEL_ALIASES` | Alias → real model-name rewrites (JSON object). A malformed value fails LOUD — see `--check-config` below |
| `MCP_BACKEND_URL`, `MCP_BACKEND_TOKEN` | MCP tool backend; without it, the in-process Rust fallback tools still serve |
| `MODEL_LOCAL_PATH`, `MODEL_ARCHIVE_PATH`, `MODEL_REGISTRY_PATH` | Storage-tiering roots + the persistent registry JSON |
| `MODEL_PROTECTED` | Models that must never be evicted to cold |
| `INFISICAL_URL`, `INFISICAL_CLIENT_ID`, `INFISICAL_CLIENT_SECRET` | Optional startup secrets bootstrap (fetches `CHORD_JWT_SECRET`/`CHORD_API_KEY` fresh; falls back to the static env) |

The rest of the 44-key surface (SNAP observability, sweep-status monitor,
embeddings, diffusion daemon, egress isolation, harness models, rate limits) is
inventoried per subsystem in the [reference pages](reference/index.md).

## Run

```sh
./target/release/chord-proxy
```

Startup logs announce each subsystem's state: model-alias count, LLM proxy
target (or "disabled"), the number of registered Rust fallback tools, model
registry tier counts, the eviction sweep interval, the SNAP monitor, the
sweep-status monitor, and both listener ports.

`./target/release/chord-proxy --version` prints the version line and exits.

`--check-config` validates the alias configuration and exits — `0` when
`CHORD_MODEL_ALIASES` parses cleanly, `1` when it is unparseable or when any
individual entry had to be dropped. Run it with the service's environment sourced
*before* restarting, since one env value backs every alias on the host.

**On a deployed host, the binary is `<path>/harmony-chord`, not `chord-proxy`.**
The cargo bin target is `chord-proxy`, but the module ships via
`OCI_INSTALL=( "chord-proxy:<path>/harmony-chord:chord.service" )`, so that is
the path the installed service actually runs — and it is the path the runtime error
message tells the operator to use:

```sh
# On the deployed host (what an operator wants, mid-incident):
set -a; . /path/to/the/chord/env/file; set +a
<path>/harmony-chord --check-config; echo "rc=$?"

# From a build tree:
./target/release/chord-proxy --check-config; echo "rc=$?"
```

Whitespace in the JSON object is handled asymmetrically, on purpose: a
whitespace-padded *value* is trimmed, but a whitespace-padded *key*
(`{" lumina-fast ": "…"}`) is **rejected** and reported as a dropped entry rather
than silently trimmed. Alias lookups match the requested model name literally, so a
padded key could never have resolved anyway; trimming it would invent a mapping that
was not written, and would let `{"foo":"a", " foo ":"b"}` collapse to one
iteration-order-dependent `foo` while still reporting a clean parse.

Both flags are handled before logging, the secrets bootstrap, and any listener bind,
so they are safe to run on a live host. If aliasing is already degraded on a running
process, `GET /health` reports it under `model_aliases` and lists `model_aliases` in
`degraded`.

## Verify

```sh
# Liveness (no auth) on both listeners
curl -s http://localhost:9099/health
curl -s http://localhost:8090/health

# Application metrics (Prometheus text format, no auth, control port)
curl -s http://localhost:8090/metrics | head

# Authenticated: list the merged tool catalog
curl -s -X POST http://localhost:9099/v1/tools/list \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' -d '{}'

# Authenticated: an OpenAI-compatible chat completion (requires CHORD_LLM_URL)
curl -s -X POST http://localhost:9099/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"model": "lumina-fast", "messages": [{"role": "user", "content": "hello"}]}'
```

(`$TOKEN` is a JWT signed with `CHORD_JWT_SECRET` — operator-specific: your
deployment's token-issuing procedure.)

## The other binaries

- **`batch-report`** ([`src/bin/batch-report.rs`](../src/bin/batch-report.rs)) —
  read-only performance-curve report generator over `run_score_points` sweep
  data.
- **`chord-tui`** ([`crates/chord-tui`](../crates/chord-tui)) — the ratatui
  control TUI. A pure *client* of Chord/Terminus control endpoints: it never
  restarts or reconfigures the live proxy. See [chord-tui.md](chord-tui.md).

## Where next

- [Architecture](architecture.md) — how a request flows and how the subsystems fit
- [Guides](guides/index.md) — model tiering operations, GPU-exclusive handoff, idle mode
- [Reference](reference/index.md) — per-subsystem symbol tables and config keys
