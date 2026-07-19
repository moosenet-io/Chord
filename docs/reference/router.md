# router

The Chord SLM router (111 KG nodes, `src/router/`, DOCGEN-03/04). Given a
generation request — from the Terminus documentation engine, or any future
in-process caller — `SlmRouter` decides the inference destination (local
high-context / local cheap / OpenRouter frontier-free) per an explicit
`RoutingPolicy`, and executes the generation on the chosen destination, falling
back gracefully and never silently when a destination is unavailable or
egress-denied. This is the mechanism behind "all doc-engine inference routes
through Chord, and Chord owns the destination decision": the caller only ever
asks the router to generate; it never picks a model itself.

## Key types and functions

| Symbol | Kind | File | What it does |
|---|---|---|---|
| `router::slm_router::SlmRouter::route_and_execute` | function | `src/router/slm_router.rs` | The entry: decide destination, execute, fall back with an explicit reason |
| `router::slm_router::RoutingDecision` | struct | `src/router/slm_router.rs` | The recorded decision (destination + rationale) returned alongside the output |
| `router::slm_router::Executor` (trait) / `HttpExecutor` / `FakeExecutor` | trait / structs | `src/router/slm_router.rs` | The execution seam: real HTTP dispatch vs the test double |
| `router::policy::RoutingPolicy::from_env` | function | `src/router/policy.rs` | The explicit, env-driven policy: which request shapes go to which destination |
| `router::policy::RoutingDestination` / `RoutingRequest` | enum / struct | `src/router/policy.rs` | The destination vocabulary and the request shape the policy examines |
| `router::eval::score_decision` | function | `src/router/eval.rs` | Scores a routing decision's appropriateness (routing quality, not raw model quality) |
| `router::eval::HeuristicCompletenessGrader::grade` | function | `src/router/eval.rs` | Heuristic doc-quality grader used in the eval sweep |
| `router::eval::panel_request` / `fixed_panel` | functions | `src/router/eval.rs` | Builds the fixed evaluation panel a candidate router is scored against |
| `router::eval::destination_latency_ms` | function | `src/router/eval.rs` | Latency measurement per destination for the cost/latency axes |
| `router::eval::summarize_candidate` / `CandidateEvalSummary` | function / struct | `src/router/eval.rs` | Cross-run comparison summary for a candidate router |
| `router::eval_storage::normalize_model_id` | function | `src/router/eval_storage.rs` | Canonical model ids for stable persistence |
| `router::eval_storage` (`RouterEvalDbError`) | module | `src/router/eval_storage.rs` | Persists eval results to Postgres for cross-run comparison |

## How it connects

Destinations resolve to real `models::backends::Backend`s from
`models::backends::seed_from_env` — the same backend catalogue the chat path's
`models::routing::resolve_and_ensure` uses — so the router reuses the existing
routing substrate (including bearer-key handling for the OpenRouter tier)
rather than reimplementing backend resolution. The eval sweep
(`eval`/`eval_storage`) persists to Postgres. The router is an in-process
capability: it is not currently mounted as its own public HTTP endpoint in
`routes::build_router`.

## Configuration

`RoutingPolicy::from_env` reads the policy surface; destination backends come
from the `models::backends` env family (`OLLAMA_URL`, `OPENROUTER_URL`,
`OPENROUTER_API_KEY` / `OPENROUTER_API_KEY_ENV_NAME`, and related keys — names
only, values vault-materialized).

## Notes and gaps

- The eval axes are decision appropriateness, resulting doc quality, cost, and
  latency — deliberately routing-quality metrics, not a model benchmark.
- Fallback is explicit: a downgraded destination is recorded in the decision,
  never silently substituted.
- This page does not cover the *chat-path* routing decisions — backend-per-model
  and the chat-role pin live in [models.md](models.md) and
  `src/routing/assistant_profile.rs` (see the [architecture page](../architecture.md#routing)).
