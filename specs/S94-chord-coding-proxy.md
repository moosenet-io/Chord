# Fleet-driven coding-model selection in Chord
plane_project: ORC
module: Refractor
prefix: CPROX
spec_id: S94-chord-coding-proxy

## Metadata
- **Project:** ORC
- **Author:** <operator> (Moose)
- **Session:** S94
- **Date:** 2026-07-03
- **Context:** A multi-day model-benchmarking sweep (coder + assistant test suites) is producing real, measured per-model quality/reliability data in Postgres (`model_full_profile` / `model_dual_profile` views joining `code_profile_runs`, `assistant_dimension_score`, `model_operational_profiles`). Today, Harmony picks a specific coding model by hardcoded name and manages its own backend/process lifecycle. This spec moves that authority into Chord: Harmony will send a work-type-coded request instead of a literal model name, and Chord resolves it to the best-fit currently-available model using the measured fleet data, while also enforcing backend safety (e.g. never routing MoE-class models through the known-unsafe Vulkan/llama-server path).

## Pre-flight
- Repository: `moosenet/Chord` on Gitea
- Dependencies: `rustup`, `cargo` (workspace already builds; some transitive deps require a `--features ssh2/vendored-openssl` build flag or a working `OPENSSL_DIR`/pkg-config on the build host — check the existing CI/build docs for the current working method rather than assuming pkg-config is available)
- Vault secrets required: none new — this reads only Postgres and existing Chord config
- Infrastructure: Postgres reachable via `INTAKE_DATABASE_URL`/`DATABASE_URL` (READ-ONLY access to `model_full_profile`, `model_dual_profile`, `code_profile_runs`, `assistant_dimension_score`, `model_operational_profiles` — this data belongs to a live, ongoing benchmarking sweep; items in this spec MUST NOT INSERT/UPDATE/DELETE any row in these tables, ever)
- Baseline tests: run `cargo test --workspace` on current `main` and record the count before starting
- **CRITICAL CONSTRAINT**: do not touch the live production deployment of Chord (its systemd service is intentionally stopped right now to keep the concurrent benchmarking sweep GPU-contention-free) — no restart, no redeploy, no config push to any live host. All work stays in worktrees, pushed as reviewed branches. Do NOT merge to `main` — leave the final PR open/pushed after dual-review approval for the operator's own merge decision, given this is a cross-repo architecture change.

### CPROX-01: Work-type code schema for coding requests
- **Priority:** High
- **Labels:** chord, models, routing, schema
- **Agent:** claude
- **Estimate:** 4h
- **Description:** Define a `WorkTypeCode` (or similarly-named) request shape that callers (Harmony) send instead of a literal model name, describing what KIND of coding work is needed rather than which model should do it.

  ## FILES
  - `src/models/work_type.rs` — new module: the `WorkTypeCode` struct/enum and its (de)serialization
  - `src/models/mod.rs` — register the new module
  - `docs/model-testing-methodology.md` — cross-reference the new schema against the measured dimensions it's meant to match against (this doc already exists from a prior spec and documents the fleet's measurement methodology)

  ## APPROACH
  1. Define `WorkTypeCode` with fields: `language` (enum: Bash, Python, Rust, TypeScript — matching the coder harness's actual measured languages), `task_shape` (enum: QuickEdit, MultiFileBuild — matching the corpus's standard/deep/blitz tiers, collapse to two shapes if the finer tiers don't map cleanly), `reasoning_need` (enum: Plan, Enrich, Review, Execute — deliberately mirroring Harmony's existing step classification so the two systems speak the same vocabulary without a translation layer), `context_depth_need` (enum: Short, Long).
  2. Implement `Serialize`/`Deserialize` (serde) so this is a clean JSON wire shape.
  3. Add a `WorkTypeCode::to_query_key()` or similar helper that produces whatever normalized form the matching engine (CPROX-02) will look up against.
  4. Do NOT hardcode any infrastructure values — this is pure data-shape code, no URLs/hosts involved.

  ## TEST PLAN
  - Unit tests: every enum variant round-trips through JSON serialize/deserialize.
  - Unit test: an unknown/malformed JSON payload fails deserialization with a clear error, not a panic.
  - `cargo test --workspace` — all existing tests still pass.
  - Verify no hardcoded IPs or org names in new/modified files.

  ## EDGE CASES
  - A request with a language not in the enum (e.g. "go") — must fail deserialization cleanly, not silently coerce to a default.
  - Missing optional-looking fields that are actually required — reject with a specific error naming the missing field.

- **Acceptance criteria:**
  - [ ] `WorkTypeCode` struct exists with all 4 dimensions and serializes/deserializes cleanly
  - [ ] Unknown enum values fail deserialization with a clear error, never panic
  - [ ] No hardcoded infrastructure values in new/modified code
  - [ ] All existing tests still pass
  - [ ] At least one negative test (malformed payload)

---

### CPROX-02: Fleet-driven model matching/scoring engine
- **Priority:** High
- **Labels:** chord, models, routing, scoring
- **Blocked by:** CPROX-01
- **Agent:** claude
- **Estimate:** 6h
- **Description:** Given a `WorkTypeCode`, query the real fleet measurement data and rank candidate models by measured quality, filtered so results measured under different memory/hardware configurations are never blended together.

  ## FILES
  - `src/models/coding_selector.rs` — new module: the scoring/ranking engine
  - `src/models/mod.rs` — register the new module
  - `src/models/backends.rs` — reuse (do not duplicate) the existing MoE/backend-safety gating logic already present from prior serving-safety work

  ## APPROACH
  1. Query `code_profile_runs` (and the `model_full_profile`/`model_dual_profile` join views if they already expose what's needed — check first, don't duplicate a join that already exists) filtered by `WorkTypeCode.language`, aggregating `code_quality_score`, `first_pass_score`/effective-score columns, and compile/test pass rates, GROUPED so a comparison is only ever made within the same `mem_config` value (never blend `dynamic_gtt` and legacy/untagged rows as if comparable).
  2. Rank candidates by a documented, simple scoring formula (e.g. weighted average of quality score and pass rate) — document the formula's rationale in a doc comment, don't hide the weighting in magic numbers with no explanation.
  3. For `context_depth_need == Long`, prefer models with populated YaRN context-depth data (`dim7_yarn_depth`'s `usable_ceiling_tokens` metric, if present in `assistant_dimension_score`) over ones with none — but do not treat "no long-context data yet" as disqualifying; the sweep is incomplete, most models won't have this data, degrade gracefully to the language/quality ranking alone when it's absent.
  4. Before returning a top candidate, run it through the SAME backend-safety check `backends.rs` already enforces (MoE-class models never routed through the Vulkan/llama-server path) — reuse that function, do not reimplement the MoE detection logic.
  5. All Postgres access is read-only `SELECT` — no write path anywhere in this module.

  ## TEST PLAN
  - Unit tests against a seeded/mocked data source (do NOT require a live Postgres connection for unit tests — inject the data access behind a trait so tests use fixtures) covering: a work-type with clear fleet data returns a sensibly-ranked top candidate; a work-type with NO fleet data yet degrades to a documented sane default rather than erroring; two models measured under different `mem_config` values are never averaged/compared directly against each other.
  - Integration test (may hit the real read-only Postgres connection, gated behind an env var so it's skippable when DB access isn't available in CI): confirms a real query against the live schema returns without error.
  - `cargo test --workspace` — all existing tests still pass.
  - Verify no hardcoded IPs or org names in new/modified files (the Postgres connection string must come through existing config helpers, never a literal).

  ## EDGE CASES
  - Zero rows for a given language/work-type combination at all — return a clear "no data" outcome the caller (CPROX-03/04) can act on, not a panic or an empty-looking-like-valid result.
  - A model with rows in BOTH `mem_config` values — must be scored per-config, never merged into one blended score.
  - A model that scores well but fails the backend-safety check — excluded from the ranked list entirely, not returned with a warning attached (the caller should never see an unsafe candidate as "the pick").

- **Acceptance criteria:**
  - [ ] Ranking never blends rows across different `mem_config` values
  - [ ] Backend-unsafe candidates (MoE-on-Vulkan) are excluded from results, reusing existing gating logic
  - [ ] Missing/incomplete fleet data degrades to a documented default instead of erroring
  - [ ] Unit tests use injected/mocked data, not a live DB dependency
  - [ ] No hardcoded infrastructure values in new/modified code
  - [ ] All existing tests still pass

---

### CPROX-03: Coding-proxy HTTP endpoint
- **Priority:** High
- **Labels:** chord, api, routing
- **Blocked by:** CPROX-02
- **Agent:** claude
- **Estimate:** 5h
- **Description:** Expose the matching engine as an actual endpoint callers (Harmony) can send a `WorkTypeCode`-tagged request to, instead of naming a model directly.

  ## FILES
  - `src/routes.rs` — add the new route (research the existing routing/proxy pattern in this file first — mirror its shape rather than inventing a new one)
  - `src/models/coding_selector.rs` — expose whatever entry point the route needs

  ## APPROACH
  1. Read `src/routes.rs`'s existing `chat_completions` handler and routing conventions before designing the new endpoint's shape — decide whether this should be a distinct route that returns a model resolution (name/backend/confidence) for the caller to then dispatch itself, or whether it should transparently proxy the full chat-completion using the resolved model — pick whichever fits the existing architecture better and document the decision in a doc comment.
  2. Wire the new route to call CPROX-02's matching engine with a `WorkTypeCode` parsed from the request body (via CPROX-01's schema).
  3. Response must include which model/backend was actually selected (for observability/debugging — a caller or operator should be able to see what Chord picked and why, not just get an opaque response).
  4. No hardcoded URLs/ports — reuse existing config helpers for anything infra-related.

  ## TEST PLAN
  - Integration test hitting the new route with a mocked/fixture-backed matching engine, confirming: a valid `WorkTypeCode` request resolves to a model and the response identifies which one; an invalid/malformed request returns a clear 4xx, not a 500 or a hang.
  - `cargo test --workspace` — all existing tests still pass.
  - Verify no hardcoded IPs or org names in new/modified files.

  ## EDGE CASES
  - Malformed `WorkTypeCode` JSON in the request body — clear 400, not a panic.
  - Matching engine returns "no data" (per CPROX-02) — the endpoint must still respond usefully (documented default model), not fail the request outright.

- **Acceptance criteria:**
  - [ ] New route accepts a `WorkTypeCode`-shaped request and resolves it via CPROX-02
  - [ ] Response identifies the actually-selected model/backend
  - [ ] Malformed requests return a clear 4xx, never a 500/hang
  - [ ] No hardcoded infrastructure values in new/modified code
  - [ ] All existing tests still pass

---

### CPROX-04: Graceful fallback when the top candidate is unavailable
- **Priority:** Medium
- **Labels:** chord, models, reliability
- **Blocked by:** CPROX-03
- **Agent:** claude
- **Estimate:** 3h
- **Description:** If CPROX-02's top-ranked model for a work-type isn't currently loaded/healthy, the endpoint must fall back to the next candidate rather than failing the request outright.

  ## FILES
  - `src/models/coding_selector.rs` — extend ranking to return an ordered candidate list, not just a single top pick
  - `src/routes.rs` — the new route from CPROX-03 tries candidates in order until one is healthy

  ## APPROACH
  1. Change CPROX-02's engine to return a ranked LIST of candidates (already-safety-filtered), not just the single best.
  2. In the route handler, try the top candidate's health/availability (reuse whatever health-check mechanism Chord already has for models, don't invent a new one); on failure, try the next; log which fallback tier was used.
  3. If every candidate is unavailable, return a clear error naming that condition — never silently succeed with something unverified.

  ## TEST PLAN
  - Unit/integration test: top candidate reported unhealthy → second candidate is tried and used, with the fallback logged/observable in the response.
  - Unit/integration test: all candidates unhealthy → clear error, not a silent wrong answer.
  - `cargo test --workspace` — all existing tests still pass.
  - Verify no hardcoded IPs or org names in new/modified files.

  ## EDGE CASES
  - Exactly one candidate exists and it's unhealthy — must fail clearly, not loop or hang.
  - Health check itself errors (network blip) — treat as unhealthy for that candidate and move to the next, don't propagate the health-check's own transient error as the whole request's failure.

- **Acceptance criteria:**
  - [ ] Unhealthy top candidate triggers fallback to the next-ranked candidate
  - [ ] Fallback usage is observable in the response/logs
  - [ ] All-candidates-unavailable produces a clear error, never a silent bad result
  - [ ] No hardcoded infrastructure values in new/modified code
  - [ ] All existing tests still pass
