# Chord — Coding Proxy (S94 CPROX-01..04)

Today Harmony picks a coding model by a hardcoded name. This feature lets it
instead send a **work-type code** — "I need CODE work of this shape" — and have
Chord pick the best REAL model from the coder-sweep's measured fleet data
(`code_profile_runs` / `model_profiles` / `assistant_dimension_score` in the
read-only intake Postgres DB), with health-gated fallback if the top pick isn't
currently servable.

Four pieces, one endpoint:

| Item | What | Where |
|------|------|-------|
| CPROX-01 | The `WorkTypeCode` request shape | [`src/models/work_type.rs`](../src/models/work_type.rs) |
| CPROX-02 | Fleet-driven scoring/ranking engine | [`src/models/coding_selector.rs`](../src/models/coding_selector.rs) |
| CPROX-03 | The HTTP endpoint | [`src/coding_proxy.rs`](../src/coding_proxy.rs) |
| CPROX-04 | Ranked-list fallback on unavailability | same file — `select_with_fallback` |

## `WorkTypeCode` (CPROX-01)

```json
{
  "language": "rust",             // Bash | Python | Rust | TypeScript
  "task_shape": "multi_file_build", // QuickEdit | MultiFileBuild
  "reasoning_need": "enrich",      // Plan | Enrich | Review | Execute
  "context_depth_need": "long"     // Short | Long
}
```

Every field is a closed serde enum — a typo'd or unknown value fails
deserialization at the edge (Axum's `Json` extractor rejects it, yielding a
clean 4xx) rather than being guessed deep in the matching engine. See
`WorkTypeCode::to_query_key` for the normalized cache/log key the matching
engine uses.

**Note on scope:** the coder sweep's `code_profile_runs` table only carries a
`language` column (and one `task_type`, `"build_modify"`, so far) — there is no
per-`task_shape` or per-`reasoning_need` breakdown in the sweep data yet.
`language` is the only field CPROX-02 currently filters the SQL query on;
`task_shape` / `reasoning_need` are accepted now so Harmony's call sites don't
need to change again once the sweep grows a finer-grained breakdown.
`context_depth_need == Long` DOES already affect ranking today, via the YaRN
long-context bonus described below.

## Scoring engine (CPROX-02)

### Why not reuse `model_dual_profile` directly
The DB exposes one existing cross-join view, `model_dual_profile` — it groups
`code_profile_runs` by `(model_id, backend_tag, mem_config)` but **across every
language at once**. Reusing it here would blend a model's Python and Rust
scores together, which is wrong for a per-language pick. CPROX-02 queries
`code_profile_runs` directly, joined to `model_profiles` for the model name,
adding `language` to the same `(model_id, backend_tag, mem_config)` grouping
`model_dual_profile` already established.

### `mem_config` is never blended
`code_profile_runs.mem_config` distinguishes the S85 `dynamic_gtt` memory
configuration from legacy/untagged runs. These are NOT comparable — e.g. on
live data `qwen3-coder:30b` averages an effective score of ~4.19 untagged vs.
~1.75 under `dynamic_gtt`. Every aggregate and every ranked candidate keeps
`mem_config` as part of its identity; `candidates_never_blend_mem_config` in
`coding_selector.rs` is the regression test.

### The scoring formula

```text
combined_score = 0.60 * (avg_effective_score / 5.0)
               + 0.25 * compile_pass_rate
               + 0.15 * test_pass_rate
```

`avg_effective_score` is the sweep's own graduated 0-5 score (already blends
compiles + tests + independent-change-correctness + LLM idiom judging — see
`terminus_rs::intake::code_v2::graduated_score`), so it carries the most
weight. `compile_pass_rate`/`test_pass_rate` are added directly (not just
implied by the average) so a handful of high scores can't hide a low overall
reliability; compiling matters slightly more than tests passing (0.25 vs 0.15)
because a change that doesn't compile is useless regardless of what its tests
would have said. All three terms are pre-normalized to `[0, 1]`.

### YaRN long-context preference (graceful degrade)
For `context_depth_need == Long`, a candidate with a populated
`dim7_yarn_depth` / `usable_ceiling_tokens` metric in `assistant_dimension_score`
gets a fixed `+0.10` ranking bonus. As of this item the sweep has recorded
**zero** such rows yet — confirmed against the live intake DB. This is expected
(the YaRN validation harness, `src/validation/yarn_validate.rs`, is a separate,
still-in-progress sweep), not a bug: no candidate gets the bonus, nothing
errors, nothing is fabricated.

### MoE / backend-safety gating — exclusion, not a flag
Per spec, a candidate that fails the backend-safety check is **excluded from
the ranked list entirely** — never returned with a warning attached, never
visible to the caller as "the pick" (or as any pick at all). `rank_candidates`
drops such candidates before scoring/sorting; there is no safety flag on
`CodingCandidate` because an unsafe candidate simply never becomes one.

**Which signal — a documented deviation.** The first version of this item
reused [`models::backends::is_vulkan_candidate`](../src/models/backends.rs)
whole for this gate, and only flagged (didn't exclude) failing candidates — a
review caught both problems. Fixing the "flag, don't exclude" bug is
straightforward; the harder finding was that `is_vulkan_candidate` is the
WRONG signal to filter on at all: it answers "is this tag BOTH non-MoE AND one
of the large 32B/34B/70B/72B dense size classes" (a vulkan-tier ELIGIBILITY
gate), not a safety verdict. Filtering the ranked list by it was verified
against the live Rust-language aggregates to wrongly exclude ~13 of ~14 real
fleet models — `codestral:latest`, `devstral:24b`, `gemma3:12b`,
`qwen2.5-coder:14b-instruct`, etc. — none of which are MoE, all of which
would vanish from every ranking. That is far more destructive than the spec's
exclusion requirement intends, so this module instead calls
[`models::backends::is_moe_tagged`](../src/models/backends.rs) — the exact
MoE-substring check (`moe`/`a3b`/`a22b`) `is_vulkan_candidate` has always used
internally, factored out to its own function so both callers share it (reuse,
not reimplementation) without inheriting the unrelated size gate.

**Known residual gap**, not introduced by this fix: `qwen3-coder:30b` is a
genuine MoE model (30B total / 3B active) — the registry module's own tests
already call it "a MoE coder" — but its stored tag in the DB doesn't literally
contain `moe`/`a3b`/`a22b`, so `is_moe_tagged` does not catch it and it is
NOT excluded today. Closing that needs either a curated model-family list or a
real per-model architecture signal ingested by the sweep (`model_profiles` has
no such column yet) — flagged as a follow-up, not fixed here, since inventing
a bespoke per-model exception list inside `coding_selector.rs` would be exactly
the kind of un-sourced "magic" this module's scoring rules otherwise avoid.

### Data source abstraction
`CodeProfileSource` (async trait) mirrors the established
`serving::profile::ProfileSource` pattern in this codebase: production uses
`DbCodeProfileSource` (a `sqlx::PgPool`, URL from
`terminus_rs::config::intake_database_url()` — no literal DSN anywhere), tests
use `StaticCodeProfileSource` (fixed fixtures, no DB). A `#[ignore]`d
integration test (`live_db_rust_aggregates_are_never_blended_across_mem_config`)
hits the real read-only intake DB when `INTAKE_DATABASE_URL`/`DATABASE_URL` is
set and `cargo test -- --ignored` is used — skipped otherwise.

## `POST /v1/coding/select` (CPROX-03)

Request body: a `WorkTypeCode` (see above). Same JWT auth as every other proxy
endpoint (`Authorization: Bearer ...`, or auth disabled when `jwt_secret` is
empty). **Not** rate-limited against the per-user LLM budget — this endpoint
resolves a model, it does not itself perform inference (see the design
decision below), so the caller's actual inference call is what should consume
budget, exactly once.

Success (`200`):

```json
{
  "selected": {
    "model_id": "qwen3-coder:30b",
    "backend": "llama-gpu",
    "confidence": 0.83,
    "mem_config": null
  },
  "fallback_tier": 0,
  "candidates_considered": 4
}
```

Failure (`503`, always JSON `{"error": "..."}`, never a 500 or a hang):

- `coding-model selection is not configured (no intake DB)` — data source never connected.
- `coding-model selection store is temporarily unavailable` — DB query failed.
- `no measured coding-model candidates exist yet for this language`
- `all N ranked coding-model candidate(s) failed their health check — refusing
  to return an unverified selection` — CPROX-04's terminal case.

Malformed/unknown-variant/empty body → `4xx` from Axum's `Json` extractor
before the handler runs — never a 500, never a hang. Concretely: invalid JSON
*syntax* (e.g. an empty body) is `400 Bad Request`; valid JSON that doesn't
match the schema (e.g. an unknown enum variant) is `422 Unprocessable Entity`.
See `test_coding_select_malformed_body_returns_4xx_not_500` (asserts 422) /
`test_coding_select_empty_body_returns_4xx_not_500` (asserts 400) in
`tests/e2e.rs`.

### Design decision: a resolution, not a transparent proxy
This endpoint returns `model_id` / `backend` / `confidence` for the caller to
dispatch itself against Chord's EXISTING `/v1/chat/completions` (or
`/v1/infer`) — it does not forward a chat-completion body itself.
`chat_completions` already owns the full inference-proxy surface (JWT auth,
per-user LLM rate-limit budget, model-alias rewriting, streaming passthrough,
audit logging); duplicating all of that here to also proxy the completion
would either re-implement ~300 lines of established logic or need a deeper
refactor than this item's scope. The two concerns are genuinely separable:
"which model should serve this coding work" (a ranking decision over sweep
data) is orthogonal to "how do I safely forward a chat completion" (already
solved). Harmony calls this endpoint once per work item, then reuses its
existing chat-completions client code unchanged, pointed at the resolved
model name.

### Backend resolution (a documented judgment call)
`code_profile_runs.backend_tag` only distinguishes `"gpu"` from
absent/legacy — not fine-grained enough to name a specific serving backend
(`llama-gpu` vs `lemonade-coder` vs `vulkan`). This module maps `Some("gpu")` →
the generic on-demand `llama-gpu` backend (serves ANY requested model's blob
on GPU) and anything else → the always-on `ollama` backend. Neither mapping
ever names `vulkan`, so nothing reaching this module can be resolved onto it.
MoE-tagged candidates never even reach this module — CPROX-02 excludes them
from the ranked list entirely (see above) — so there is no per-candidate
safety flag to check here at all.

## Ranked-list fallback (CPROX-04)

CPROX-02 returns a full ranked `Vec<CodingCandidate>`, not just the winner. The
route handler (`select_with_fallback`) walks it best-first:

1. Resolve the candidate's backend (skip — not a health failure — if that
   backend isn't configured on this host at all).
2. Probe `GET {backend_url}/health` via `HttpHealthChecker`, a production
   implementation of the EXISTING `serving::launcher::HealthChecker` trait
   (reused, not reinvented) that mirrors the exact `GET {base_url}/health`
   convention `snap::health::poll_vllm` already established elsewhere in this
   codebase.
3. First healthy candidate wins; `fallback_tier` in the response records how
   many ranked candidates were skipped before it (0 = the top pick was
   healthy).
4. If every candidate fails, return the distinguishable
   `AllCandidatesUnavailable` 503 — never silently fall through to something
   unverified.

## What's NOT done / open follow-ups

- `task_shape` / `reasoning_need` don't yet affect ranking (see the scope note
  above) — the sweep data doesn't support that breakdown yet.
- Backend resolution is coarse (`gpu` → `llama-gpu`, else → `ollama`); no
  attempt to route dense-large candidates onto `vulkan` specifically.
- No caching of `rank_for_work_type`'s DB query — every `/v1/coding/select`
  call re-queries `code_profile_runs`. Fine at today's sweep-query volume; a
  future item could add a short TTL cache if this becomes hot-path traffic.
- **`is_moe_tagged`'s residual gap** (see "MoE / backend-safety gating"
  above): `qwen3-coder:30b` is a genuine MoE model not caught by the
  name-substring check, so it is NOT excluded today. A real per-model
  architecture signal (ideally ingested by the sweep itself, not name-sniffed)
  would close this properly.
