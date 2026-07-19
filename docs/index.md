# Chord — Documentation

Chord (`chord-proxy`) is the Lumina Constellation's inference proxy and orchestrator: one Rust process that routes LLM traffic, manages model storage and VRAM lifecycles, and dispatches MCP tools for the fleet. This page is the hub for all of its documentation.

## Start here

| Page | What's in it |
|---|---|
| [Getting started](getting-started.md) | Build, configure, run, verify — clone to first request |
| [Architecture](architecture.md) | Derived subsystem diagram, request-flow walkthrough, per-component narrative |
| [Reference index](reference/index.md) | One page per subsystem (12), plus the configuration surface |
| [Guides index](guides/index.md) | Task-oriented operator guides |

## Reference (per subsystem)

- [agentic](reference/agentic.md) — the guarded agentic tool-calling loop and its five security guards
- [models](reference/models.md) — storage tiering, eviction/GC, archive pulls, backend catalogue and lifecycle
- [serving](reference/serving.md) — serving profiles, runtime launcher, VRAM residency, clean swap
- [harness](reference/harness.md) — the Harness-1 research state machine
- [snap](reference/snap.md) — observability: VRAM reader, health poller, inventory, analytics
- [sweep_status](reference/sweep_status.md) — benchmarking-sweep health monitor
- [router](reference/router.md) — the SLM router and its routing-quality evaluation
- [routes](reference/routes.md) — the proxy-port HTTP surface (and audit logging)
- [supervisor](reference/supervisor.md) — launch-env scrubbing and network-namespace isolation
- [gpu_exclusive](reference/gpu_exclusive.md) — the GPU handoff lock
- [mcp_proxy](reference/mcp_proxy.md) — MCP backend proxy, fallback registry, tool catalog and allowlist
- [chord-tui](reference/chord-tui.md) — the workspace sub-crates: control TUI + <secret-manager> client

## Guides

- [Model tiering operations](guides/model-tiering-operations.md) — archive, pull, protect, sweep, reconcile, GC via the control API
- [GPU-exclusive handoff](guides/gpu-exclusive-handoff.md) — lending the GPU to a benchmarking sweep without stopping Chord
- [Idle mode and the activity signal](guides/idle-mode.md) — freeing the host for heavy builds, and detecting genuine idle windows

## Deep dives (pre-existing, still authoritative)

- [serving.md](serving.md) — the serving/VRAM subsystem, module by module
- [egress.md](egress.md) — runtime egress isolation: what is and is not guaranteed
- [chord-tui.md](chord-tui.md) — the control TUI client
- [coding-proxy.md](coding-proxy.md) — `POST /v1/coding/select` fleet-driven coding-model resolution
- [diffusion-gemma-managed.md](diffusion-gemma-managed.md) — Chord-managed DiffusionGemma daemon lifecycle
- [snap-persistence.md](snap-persistence.md) — SNAP state persistence
- [rout-01-routing-map.md](rout-01-routing-map.md) — the serving-profile routing map
- [model-testing-methodology.md](model-testing-methodology.md) — how fleet model results are measured
- [test-results.md](test-results.md) / [assistant-results.md](assistant-results.md) / [contributing-results.md](contributing-results.md) — measured model results and how to contribute them

## Legacy pages

Earlier generated shards under [reference/](reference/index.md#legacy-pages) (overview, status, mcp-tool-dispatch, idle-mode, secrets, test-results, documentation, license) are kept for continuity; the pages above supersede them.
