# ROUT-01: Model-tier / local-vs-cloud routing decision map (S92)

This document is the "written map" deliverable for ROUT-01: every place a
local-vs-cloud or small-vs-big model decision is made today across
`moosenet/Chord` and `moosenet/Harmony`, found by grepping both repos during
the S92 hybrid-routing build.

## Chord (`moosenet/Chord`)

| Location | What it decides | Mechanism |
|---|---|---|
| `src/agentic/model_router.rs` — `ComplexityHeuristic::assess_complexity` | Whether to escalate from the fast model to the deep model, both at turn-zero (called with 0 tool results / 0 chars / the initial query) and mid-session | 6-keyword substring match (`analyze`, `compare`, `synthesize`, `evaluate`, `explain why`, `reason about`) OR `tool_result_count > 2` OR `total_chars > 5,000` |
| `src/agentic/model_router.rs` — `AgenticModelRouter::escalate` | Enforces max-one-escalation per execution; once escalated, `current_model()` never reverts | Simple boolean flag (`escalated`) |
| `src/agentic/model_router.rs` — `is_deep_request` / `force_deep` | User-forced `/deep` prefix bypasses the heuristic entirely | Prefix string match |
| `src/models/coding_selector.rs`, `src/models/batch_suitability.rs`, `src/models/work_type.rs` (uncommitted local work on a sibling branch, not part of this spec) | Model *selection within a tier* for coding-proxy requests (S94 scope) — a related but distinct concern from small-vs-big routing | Separate from ROUT-04's scope |

Chord's routing is the confirmed, single, real turn-zero + mid-session
decision point this spec (ROUT-02/03/04) integrates with. No other file in
Chord makes an independent local-vs-cloud call — `routing.rs` (top-level
module) handles MCP tool/request routing, not model-tier selection, and is
unrelated.

## Harmony (`moosenet/Harmony`, `harmony-core/`)

Harmony does **not** purely consume Chord's routing decision — it has its own,
separate provider/model-tier selection layer sitting *above* Chord for a
subset of its traffic:

| Location | What it decides | Mechanism |
|---|---|---|
| `src/conductor/task_classifier.rs` (PROV-03) | Whether a given provider should be *skipped* for a task, based on `TaskCharacteristics` (file count, description length, acceptance-criteria count, task type) | Advisory filter over an already-selected provider pool; never allowed to empty the pool |
| `src/analysis/routing.rs` — `route_task` / `route_task_for_phase` | Routes a task to `"gemini"`, `"codex"`, or the default `"executor"` pool | Keyword-in-title match (`GEMINI_KEYWORDS`, `CODEX_KEYWORDS`) plus a phase-sequence fallback (`seq % 7`) |
| `src/providers/chord.rs` (`ChordProvider`) | For `local_gpu`/`local_cpu` tiers specifically, dispatches through Chord (`CHORD_PROXY_URL`), which then applies **Chord's own** `model_router.rs` decision underneath | This is the one path where Harmony *does* purely consume Chord's decision — but only for the tiers it routes through Chord in the first place |
| `src/providers/thinking_class.rs` (referenced by `chord.rs`, THINK-01/THINK-02) | Per-request thinking-mode recommendation (`ThinkingRecommendation`) from `StepDescriptor` + `StepComplexitySignals` | Separate classifier, orthogonal to small-vs-big model choice |
| `src/conductor/gpu_scheduler.rs`, `src/conductor/budget.rs`, `src/conductor/modes.rs` | GPU/cost budget-aware scheduling across the worker pool | Not a per-request model-tier decision; operates at the fleet level |

**Conclusion:** Harmony has its own, independent, keyword/heuristic-based
provider-tier decision (`analysis/routing.rs`, `conductor/task_classifier.rs`)
for its own dispatch/conductor traffic, separate from and upstream of
anything that eventually flows through `ChordProvider` into Chord's
`model_router.rs`. This spec's scope (ROUT-01..06) is Chord's routing layer
only — Harmony's own routing layer is out of scope for S92 and is flagged
here as a candidate for a future, dedicated hybrid-routing spec if the
operator wants the same Supra-Router treatment applied there. No code in
Harmony was modified by this spec.

## Daemon deployment precedent reused (ROUT-01)

The Supra-Router-51M daemon (see `deploy/supra-router.service` in this
branch) mirrors the `dgem.service` pattern already proven on `<host>` for
DiffusionGemma this session:

- `Type=simple`, loopback-bound (`Environment=...BIND=127.0.0.1`), no external
  exposure
- `Restart=on-failure` / `RestartSec=5` for resilience
- Model path and port supplied via `Environment=` lines, not hardcoded into
  `ExecStart`
- `WantedBy=multi-user.target`

**Status of the live daemon stand-up**: the systemd unit and directory layout
are written and match the proven pattern, but the actual HF→GGUF model
conversion and live process start were **not** executed as part of this
build — see the top-level build report for why (disk/time scope, and this
was flagged rather than silently skipped or fabricated).
