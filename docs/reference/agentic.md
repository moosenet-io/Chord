# agentic

The guarded agentic execution loop (539 KG nodes, `src/agentic/`). Entry type
`AgenticExecutor` accepts an `AgenticRequest`, runs the internal LLM↔tool loop
up to `max_tool_calls` iterations with five security guards applied at every
step, and returns an `AgenticResponse` whose execution log is metadata-only —
tool arguments and raw results never appear in any returned struct. Reached via
`POST /v1/agent/execute` on the proxy port.

## Key types and functions

| Symbol | Kind | File | What it does |
|---|---|---|---|
| `agentic::AgenticExecutor` | struct | `src/agentic/loop_runner.rs` | The loop: model call → guard checks → tool execution → repeat, with a final synthesized answer |
| `agentic::AgenticRequest` / `AgenticResponse` / `ExecutionStep` | structs | `src/agentic/context.rs` | The request/response contract; `ExecutionStep` carries tool name, duration, status — metadata only |
| `agentic::SecurityEvent` / `SecurityAction` | struct / enum | `src/agentic/mod.rs` | Shared guard-outcome record: `Blocked`, `Sanitized`, or `Warned`, with guard name and reason |
| `agentic::permissions::PermissionEnforcer` | struct | `src/agentic/permissions.rs` | Per-user allowed-tool sets — the first gate before any tool runs |
| `agentic::argument_guard::ArgumentGuard::scan` | function | `src/agentic/argument_guard.rs` | Blocks shell/SQL-injection and credential patterns in tool arguments (one of the repo's top call-graph hotspots) |
| `agentic::result_guard::ResultGuard::scan` | function | `src/agentic/result_guard.rs` | Sanitizes suspicious tool results before they re-enter the model context |
| `agentic::response_guard::ResponseGuard` | struct | `src/agentic/response_guard.rs` | Detects cross-step injection chains in model responses |
| `agentic::behavioral_monitor::BehavioralMonitor::with_config` | function | `src/agentic/behavioral_monitor.rs` | Flags internal-data → external-tool exfiltration patterns across the whole execution |
| `agentic::model_router::AgenticModelRouter` | struct | `src/agentic/model_router.rs` | Selects the model per step; escalates once fast → deep when `ComplexityHeuristic` fires |
| `agentic::model_router::ComplexityHeuristic::default` | function | `src/agentic/model_router.rs` | The escalation trigger: tool-result count, total content size, reasoning-keyword match |
| `agentic::streaming` (SSE `ProgressEvent`) | module | `src/agentic/streaming.rs` | Streams per-step progress when the caller sets `stream: true` |
| `agentic::synthesis` | module | `src/agentic/synthesis.rs` | Builds the final answer from accumulated tool evidence |
| `agentic::harness_integration` | module | `src/agentic/harness_integration.rs` | Bridges research-shaped queries into the Harness-1 search state machine |
| `agentic::router_classifier` | module | `src/agentic/router_classifier.rs` | Query classification feeding the model-routing decision |

## How it connects

`routes::agent_execute` constructs the request and hands it to the
`AgenticExecutor` held in `AppState` (built once in `main.rs` around a shared
`McpProxy`). Tool execution flows through **mcp_proxy** — so every agentic tool
call gets the same backend/fallback behavior and allowlist scoping as a direct
`/v1/tools/call`. Research-shaped queries route through **harness** via
`harness_integration`, which also drives the VRAM rotation
(`harness::vram_lifecycle`) around a search episode. Model escalation targets
are plain model names resolved by the normal chat-completions path.

## Configuration

Key names only: `CHORD_FAST_MODEL` (lightweight default model),
`CHORD_DEEP_MODEL` (the one-shot escalation target).

## Notes and gaps

- The escalation is deliberately capped at **one per execution** so VRAM is not
  thrashed by repeated model swaps mid-loop.
- Guard verdicts are surfaced as `SecurityEvent`s in the metadata log; this page
  does not document each guard's exact pattern set — read the guard sources for
  the current rules.
- Per-user permission sets (`PermissionEnforcer`) are code-configured; there is
  no runtime admin API for editing them.
