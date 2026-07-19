# harness

The Harness-1 search harness (229 KG nodes, `src/harness/`) — a stateful
research state machine inside the agentic executor. The harness maintains
per-search working memory (candidate pool, curated set, evidence graph,
verification records) and presents a compact rendered observation to the model
each turn; the model emits one action, the harness executes it and re-renders.
This is the stateful cognitive-offloading principle: the harness holds the
bookkeeping, the model makes the decisions.

## Key types and functions

| Symbol | Kind | File | What it does |
|---|---|---|---|
| `harness::SearchHarness` | struct | `src/harness/mod.rs` | The episode driver: working memory + injectable backend + turn budget + recent-action history |
| `harness::Observation` | struct | `src/harness/mod.rs` | What the model sees after each turn: compact rendered state + completion flag |
| `harness::actions::HarnessAction` | enum | `src/harness/actions.rs` | The action vocabulary the model emits, one per turn |
| `harness::state::WorkingMemory::new` | function | `src/harness/state.rs` | The per-search memory: candidates, curated set, evidence graph, verification records |
| `harness::state::Importance::rank` | function | `src/harness/state.rs` | Ranking used to keep the rendered observation compact |
| `harness::state::SearchBudget` | struct | `src/harness/state.rs` | Turn cap (default from `HARNESS_MAX_TURNS`; callers may override) |
| `harness::executor::Executor` | struct | `src/harness/executor.rs` | Executes actions against a `SearchBackend` |
| `harness::executor::SearchBackend` (trait) + `MockBackend` | trait / struct | `src/harness/executor.rs` | The injectable search seam; `MockBackend::with_search` powers the unit tests |
| `harness::detector::ResearchDetector::detect` | function | `src/harness/detector.rs` | Decides plain search vs full harness: explicit `/research`, intent keywords, or a complexity score over threshold |
| `harness::vram_lifecycle::HarnessVramManager` | struct | `src/harness/vram_lifecycle.rs` | HRNS-03: the four-step VRAM rotation — personality → search model → synthesis → personality |
| `harness::vram_lifecycle::HttpSwapClient` | struct | `src/harness/vram_lifecycle.rs` | Performs each swap through Chord's lifecycle control API (never talks to Ollama directly); a top call-graph hotspot |
| `harness::tool_definition` | module | `src/harness/tool_definition.rs` | The harness's tool-facing definition surface |

## How it connects

**agentic** is the sole caller: `agentic::harness_integration` routes
research-shaped queries here after `ResearchDetector` fires, drives the episode
loop, and hands the curated documents to `agentic::synthesis` for the
citation-style final answer. VRAM rotation goes through the HTTP control API at
`CHORD_CONTROL_URL` (the `HttpSwapClient`), so the harness reuses Chord's own
lifecycle machinery rather than owning model processes. Every rotation failure
degrades (`SwapOutcome::Fallback` / `Degraded`) — a research query never crashes
the loop, it just runs on the currently resident model.

## Configuration

`HARNESS_SEARCH_MODEL`, `HARNESS_SYNTHESIS_MODEL`, `HARNESS_VRAM_INTEGRATION`,
`HARNESS_SWAP_TIMEOUT_SECS`, `CHORD_CONTROL_URL` (plus `HARNESS_MAX_TURNS` /
`HARNESS_TRIGGER_THRESHOLD` read by the budget and detector).

## Notes and gaps

- The model names for each rotation step come entirely from config/env —
  nothing is hardcoded outside default-fallback strings.
- The search backend is a trait seam; this page does not document which
  concrete backend a given deployment wires in.
- The rendered observation format is versioned informally by the rendering code
  in `mod.rs` — treat it as internal, not a stable API.
