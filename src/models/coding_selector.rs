//! CPROX-02/04: fleet-driven coding-model matching/scoring engine.
//!
//! Given a [`WorkTypeCode`](crate::models::work_type::WorkTypeCode), rank the
//! REAL coder-sweep fleet data (`code_profile_runs`, joined to `model_profiles`
//! for the model name) instead of a hardcoded model alias.
//!
//! ## Why not `model_dual_profile` / a "`model_full_profile`" view
//! The spec for this item asks to check first whether an existing view already
//! covers this join before writing a new one. As of this item, Postgres exposes
//! exactly one such view, `model_dual_profile` (there is no `model_full_profile`
//! view in this database). It aggregates `code_profile_runs` grouped by
//! `(model_id, backend_tag, mem_config)` — but *across every language at once*
//! (no `language` column in its `GROUP BY`), because it exists to answer "does
//! this model have ANY builder/assistant profile", not "which Rust coder is
//! best". Reusing it here would silently blend a model's Python and Rust scores
//! together, which is wrong for a per-language pick. So this module queries
//! `code_profile_runs` directly, adding `language` to the existing
//! `(model_id, backend_tag, mem_config)` grouping `model_dual_profile` already
//! established — same shape, one more dimension, no duplicated join logic
//! reinvented from scratch.
//!
//! ## The `mem_config` hard requirement
//! `code_profile_runs.mem_config` distinguishes memory configurations —
//! `carveout` (the dominant value on the live data today), the S85
//! `dynamic_gtt`, and legacy/untagged runs (`mem_config IS NULL`). These are
//! NOT comparable: on the live data, `qwen3-coder:30b` averages an effective
//! score of ~4.19 under `carveout` vs. ~1.89 untagged for the same model —
//! blending them would produce a meaningless average. Every aggregate query and
//! every ranking step in this module keeps `mem_config` as part of the grouping
//! key; [`candidates_never_blend_mem_config`] is the regression test for this.
//!
//! ## Task-category ranking and the coverage gate (CPROX-04)
//! Ranking prefers runs matching the requested [`TaskShape`]'s own sweep
//! category (`blitz` / `multi_file` / `deep`) over a language-wide average.
//! An earlier version of this module stated the sweep had no per-task-shape
//! breakdown; that is no longer true — `code_profile_runs.task_category` is
//! populated — but the data is very uneven, and the unevenness is a trap:
//!
//! - `blitz` is genuinely measured (live, language=rust: 220 runs across 38
//!   models, 151 with a recorded `compiles` value, a real score spread).
//! - `multi_file` and `deep` are NOT measured for the languages this
//!   constellation actually writes. Live counts of runs with a recorded
//!   `compiles` value, by language and category:
//!
//!   | language   | blitz | multi_file | deep |
//!   |------------|-------|------------|------|
//!   | rust       | 151   | **0**      | **0** |
//!   | python     | 64    | **0**      | **0** |
//!   | typescript | 52    | **0**      | **0** |
//!
//!   Fleet-wide, `multi_file` does have 47 measured runs — but every one of
//!   them is `go` (24) or `java` (23), and `deep` has zero in every language.
//!   An earlier revision of this comment claimed a flat "ZERO for every model";
//!   that was true for the languages checked and wrong as a global statement,
//!   and it is corrected here rather than left to be discovered later.
//!
//!   The unmeasured rows still carry `first_pass_score = 0`, a real zero rather
//!   than a NULL, so a naive `GROUP BY task_category` returns a complete-looking
//!   ranking in which everything ties at zero: picking its top entry means
//!   picking arbitrarily while presenting an absence as a measurement.
//!
//! - A fourth bucket matters and is easy to miss: **4804 of the 6100 rows have
//!   `task_category IS NULL`**, and they hold 1220 of the 1662 measured compiles
//!   fleet-wide — roughly 73% of ALL measurement in the table is uncategorized.
//!   Those rows are correctly excluded from a per-category load (they cannot be
//!   attributed to a category that was never recorded) but ARE included in the
//!   language-wide fallback. This is a large part of why the fallback is so much
//!   better-evidenced than any per-category load, and therefore why the coverage
//!   gate below usually chooses it.
//!
//! So [`category_coverage_is_usable`] gates the per-category path, and every
//! result carries a [`SelectionBasis`] saying which evidence it actually rests
//! on. The fallback is never silent.
//!
//! ## Scoring formula (documented, no unexplained magic numbers)
//! Each `(model_id, backend_tag, mem_config)` group's `combined_score` is:
//!
//! ```text
//! combined_score = 0.60 * (avg_effective_score / 5.0)
//!                + 0.25 * shrunk(compile_pass_rate, measured_compile_count)
//!                + 0.15 * shrunk(test_pass_rate,    measured_test_count)
//! ```
//!
//! where `shrunk` pulls a rate toward a 0.5 prior in proportion to how little
//! evidence backs it (see [`shrunk_rate`]). CPROX-04 added that: Postgres
//! `avg()` skips NULLs, so an unmeasured run silently leaves the denominator
//! and a rate computed from 4 of 16 runs was previously trusted exactly as much
//! as one computed from 30 of 30. Live, `OlympicCoder-32B` reads a perfect
//! 1.00/1.00 on rust from 4 measured runs — the ranking was rewarding models
//! for not being measured.
//!
//! - `avg_effective_score` is the sweep's own graduated 0-5 score (the harness
//!   already blends compiles + tests + independent-change-correctness + LLM
//!   idiom judging into one number — see `terminus_rs::intake::code_v2::
//!   graduated_score`), so it carries the most weight (0.60) as the single best
//!   existing signal.
//! - `compile_pass_rate` and `test_pass_rate` are added directly (not just
//!   implied by the average) so a model whose few high scores hide a low
//!   overall reliability doesn't get over-ranked; compiling matters slightly
//!   more than tests passing because a change that doesn't compile is useless
//!   regardless of what its tests would have said (0.25 vs 0.15).
//! - All three terms are pre-normalized to `[0, 1]`, so `combined_score` is
//!   itself in `[0, 1]` — no separate rescale needed downstream.
//!
//! ## Context-depth preference (YaRN)
//! For `context_depth_need == Long`, candidates with a populated
//! `dim7_yarn_depth` / `usable_ceiling_tokens` metric (in
//! `assistant_dimension_score`) are preferred via a fixed ranking bonus — see
//! [`YARN_LONG_CONTEXT_BONUS`]. As of this item, the sweep has recorded ZERO
//! `dim7_yarn_depth` rows yet (confirmed against the live intake DB) — this is
//! expected, not a bug: the YaRN validation harness (`src/validation/
//! yarn_validate.rs`) is a separate, still-in-progress sweep. Absent data simply
//! means no candidate gets the bonus; nothing errors and nothing is fabricated.
//!
//! ## MoE / backend-safety gating — EXCLUSION, not a flag
//! Per spec, a candidate that fails the backend-safety check is **excluded
//! from the ranked list entirely** — never returned with a warning attached,
//! never visible to the caller as "the pick" (or as any pick at all). This
//! module's [`rank_candidates`] drops such candidates before they are ever
//! scored/sorted/returned; there is no `vulkan_safe`-style flag surfaced on
//! [`CodingCandidate`] because an unsafe candidate simply never becomes one.
//!
//! **Which signal decides "backend-unsafe" — a documented deviation.** The
//! original version of this item reused
//! [`crate::models::backends::is_vulkan_candidate`] whole for this gate. That
//! was wrong: `is_vulkan_candidate` answers "is this tag BOTH non-MoE AND one
//! of the large 32B/34B/70B/72B dense size classes" — it is a vulkan-tier
//! ELIGIBILITY gate, not a safety verdict, and its `false` case fires for
//! almost every dense model that simply isn't one of those four sizes. Using
//! it as an exclusion filter was verified (against the live Rust-language
//! aggregates) to wrongly exclude ~13 of ~14 real fleet models — e.g.
//! `codestral:latest`, `devstral:24b`, `gemma3:12b`,
//! `qwen2.5-coder:14b-instruct` — none of which are MoE, all of which would
//! vanish from every ranking. That is a far more destructive outcome than the
//! spec's exclusion requirement intends, so this module instead calls
//! [`crate::models::backends::is_moe_tagged`] — the exact MoE-substring check
//! `is_vulkan_candidate` has always used internally, factored out to its own
//! function so both callers share it (reuse, not reimplementation) without
//! also inheriting the unrelated size gate. See `is_moe_tagged`'s doc comment
//! for a known residual gap this narrower signal still has (`qwen3-coder:30b`,
//! a genuine MoE model per the registry's own test comments, isn't tag-flagged
//! as MoE and is therefore NOT excluded by this check) — closing that gap
//! needs a curated model-family list or a real per-model architecture signal
//! from the sweep, out of scope for this fix. **This is a deliberate deviation
//! from directly reusing `is_vulkan_candidate`, flagged here rather than made
//! silently** — the exclusion behavior itself (spec's actual requirement) is
//! implemented as written; only the choice of *which existing function*
//! constitutes "the MoE/backend-safety gate" changed.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::models::backends::is_moe_tagged;
use crate::models::work_type::{ContextDepthNeed, WorkTypeCode};

/// Weight of the sweep's own graduated 0-5 effective score in the combined
/// ranking score. See the module-level "Scoring formula" doc comment.
const WEIGHT_EFFECTIVE_SCORE: f64 = 0.60;
/// Weight of the compile pass rate.
const WEIGHT_COMPILE_RATE: f64 = 0.25;
/// Weight of the test pass rate.
const WEIGHT_TEST_RATE: f64 = 0.15;

/// Fixed ranking bonus applied to a candidate's `combined_score` when
/// `context_depth_need == Long` AND the candidate has a populated YaRN
/// usable-ceiling-tokens metric. Additive on the already-`[0,1]`-normalized
/// score so a long-context-capable model can out-rank a marginally
/// higher-quality model that has no validated long-context data at all — but
/// cannot alone beat a MUCH better short-context model (bonus is small
/// relative to the 0-1 score range).
const YARN_LONG_CONTEXT_BONUS: f64 = 0.10;

/// One aggregated row from `code_profile_runs` (grouped by
/// `model_id, backend_tag, mem_config, language` — see module docs). This is
/// the pre-ranking data shape; [`CodeProfileSource`] implementations produce
/// these, [`rank_candidates`] turns them into scored, safety-gated
/// [`CodingCandidate`]s.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CodeAggregate {
    pub model_id: String,
    /// `code_profile_runs.backend_tag` (observed values: `"gpu"` or absent/NULL).
    pub backend_tag: Option<String>,
    /// `code_profile_runs.mem_config` (observed values: `"dynamic_gtt"` or
    /// absent/NULL for legacy/untagged runs). NEVER blended across values.
    pub mem_config: Option<String>,
    pub run_count: i64,
    /// Average of `coalesce(retry_score, first_pass_score)` — the sweep's
    /// graduated 0-5 quality signal.
    pub avg_effective_score: Option<f64>,
    /// Fraction of runs where `compiles = true`, **over the runs where
    /// `compiles` was actually recorded** — see [`measured_compile_count`].
    pub compile_pass_rate: Option<f64>,
    /// Fraction of runs where `tests_pass = true`, over the runs where
    /// `tests_pass` was actually recorded — see [`measured_test_count`].
    pub test_pass_rate: Option<f64>,
    /// How many of `run_count` runs actually recorded a `compiles` value.
    ///
    /// This is NOT a formality. Postgres `avg()` skips NULLs, so an unmeasured
    /// run silently leaves the denominator: on the live intake DB,
    /// `OlympicCoder-32B` (rust) reports `compile_pass_rate = 1.00` from **4
    /// measured of 16 runs** — the other 12 produced no signal at all and it
    /// still reads perfect. Without this count there is no way to tell that
    /// apart from a model measured 30 times that passed every one.
    pub measured_compile_count: i64,
    /// How many of `run_count` runs actually recorded a `tests_pass` value.
    /// Same hazard as [`measured_compile_count`] (live: `deepcoder:14b` rust
    /// reports `test_pass_rate = 1.00` from 5 measured of 16 runs).
    pub measured_test_count: i64,
    /// How many runs backed [`avg_effective_score`](Self::avg_effective_score)
    /// — i.e. how many were FINALIZED (judged).
    ///
    /// The same coverage discipline as the two counts above, applied to the
    /// term that carries the largest weight (0.60) in [`combined_score`]. The
    /// score aggregate is `FILTER (WHERE finalized)`ed because an unjudged row
    /// holds a pre-judge score; that filter makes the score's `n` differ from
    /// `run_count` AND from `measured_compile_count`, so it needs its own count
    /// rather than borrowing either. Zero here with a non-zero `run_count`
    /// means "runs exist, none judged" — an absence, not a low score.
    pub measured_score_count: i64,
}

/// Source of `code_profile_runs` aggregates. Abstracted (mirrors
/// [`crate::serving::profile::ProfileSource`]'s established pattern in this
/// codebase) so unit tests use fixtures and only a gated integration test hits
/// the real read-only intake DB.
#[async_trait]
pub trait CodeProfileSource: Send + Sync {
    /// Load per-`(model_id, backend_tag, mem_config)` aggregates for one
    /// `language`. Every implementation MUST group by `mem_config` (never
    /// blend it away) — see the module docs' hard requirement.
    ///
    /// `task_category` filters to one sweep category (`blitz`/`multi_file`/
    /// `deep`); `None` means language-wide across all categories, which is the
    /// pre-CPROX-04 behaviour and the fallback path.
    async fn load_aggregates(
        &self,
        language: &str,
        task_category: Option<&str>,
    ) -> Result<Vec<CodeAggregate>, SelectorError>;

    /// Best-effort YaRN long-context signal for `model_id` (within the SAME
    /// `mem_config` as the candidate being scored — the same non-blending rule
    /// applies here). `None` when no `dim7_yarn_depth` / `usable_ceiling_tokens`
    /// row exists yet — the expected, common case today (see module docs).
    async fn yarn_usable_ceiling_tokens(
        &self,
        model_id: &str,
        mem_config: Option<&str>,
    ) -> Option<f64>;
}

/// A selector data-source failure. Carries no infra detail (host/DSN) — same
/// discipline as [`crate::serving::profile::ProfileLoadError`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectorError {
    NotConfigured,
    StoreUnavailable,
}

impl std::fmt::Display for SelectorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SelectorError::NotConfigured => f.write_str("coding-profile store is not configured"),
            SelectorError::StoreUnavailable => {
                f.write_str("coding-profile store is temporarily unavailable")
            }
        }
    }
}

impl std::error::Error for SelectorError {}

/// What evidence a ranking actually rests on. Returned to the caller so a
/// fallback can never be mistaken for a per-shape pick.
///
/// This exists because of a specific trap in the live data. `multi_file` and
/// `deep` rows carry `first_pass_score = 0` and `compiles = NULL` — the score
/// is a real zero, not an absent value. So a naive `GROUP BY task_category`
/// returns a complete-looking ranking in which every model scores identically
/// zero, and picking "the best" of it is picking arbitrarily while presenting
/// an absence as a measurement. The gate in [`category_coverage_is_usable`]
/// refuses that ranking; this enum makes the refusal visible instead of silent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionBasis {
    /// Ranked on runs matching the requested `task_shape`'s category, which
    /// passed the coverage gate. The strongest basis available.
    PerCategory,
    /// The requested category had too little real measurement, so the ranking
    /// is language-wide across all categories. The pick is defensible for the
    /// language but says nothing specific about this task shape.
    LanguageFallback,
    /// Neither the category nor the language had usable measured data. The
    /// candidate list may be empty or entirely unmeasured; a caller must not
    /// present any entry as an evidence-based recommendation.
    InsufficientData,
}

/// Minimum runs of a single candidate that must carry a real `compiles` value
/// before that candidate counts as measured for the coverage gate.
const MIN_MEASURED_RUNS_PER_CANDIDATE: i64 = 4;
/// Minimum number of such measured candidates a category needs before it is
/// ranked on its own. Below this there is nothing to compare — one measured
/// model is not a ranking.
const MIN_QUALIFYING_CANDIDATES: usize = 3;

/// Whether a category's aggregates carry enough real measurement to rank on.
///
/// Verified against the live intake DB (language=rust): `blitz` passes — 220
/// runs across 38 models, 151 of them with a recorded `compiles` value.
/// `multi_file` (196 runs) and `deep` (120 runs) both fail: `compiles` is NULL
/// on EVERY one of their runs, so no candidate reaches
/// `MIN_MEASURED_RUNS_PER_CANDIDATE` and nothing qualifies. Same result for
/// python and typescript. Fleet-wide `multi_file` does have 47 measured runs,
/// but all of them are go or java.
pub fn category_coverage_is_usable(aggregates: &[CodeAggregate]) -> bool {
    aggregates
        .iter()
        .filter(|a| a.measured_compile_count >= MIN_MEASURED_RUNS_PER_CANDIDATE)
        .count()
        >= MIN_QUALIFYING_CANDIDATES
}

/// A ranked candidate list together with the evidence it rests on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankedSelection {
    pub candidates: Vec<CodingCandidate>,
    pub basis: SelectionBasis,
    /// The sweep category that was requested (`blitz`/`multi_file`/`deep`) —
    /// present even when `basis` is a fallback, so the caller can see WHICH
    /// shape lacked data rather than just that something did.
    pub requested_category: String,
}

/// A ranked, backend-safety-gated coding-model candidate — CPROX-04's fallback
/// list is built directly from a `Vec<CodingCandidate>` sorted best-first.
/// There is NO safety/unsafe flag on this type: a candidate that failed the
/// MoE/backend-safety gate never becomes one of these in the first place (see
/// the module-level "MoE / backend-safety gating" doc comment) — the caller
/// can never see an unsafe candidate as "the pick" or as any pick at all.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CodingCandidate {
    pub model_id: String,
    pub backend_tag: Option<String>,
    pub mem_config: Option<String>,
    pub run_count: i64,
    pub avg_effective_score: Option<f64>,
    pub compile_pass_rate: Option<f64>,
    pub test_pass_rate: Option<f64>,
    /// Carried through from [`CodeAggregate::measured_compile_count`] so a
    /// caller can see how much evidence backs `compile_pass_rate` rather than
    /// reading a bare `1.00` as a strong result.
    pub measured_compile_count: i64,
    /// Carried through from [`CodeAggregate::measured_test_count`].
    pub measured_test_count: i64,
    /// Carried through from [`CodeAggregate::measured_score_count`] — how many
    /// judged runs back `avg_effective_score`, the heaviest scoring term.
    pub measured_score_count: i64,
    /// `[0, 1]`-ish combined ranking score (see module docs). Higher is better.
    pub combined_score: f64,
    /// Whether the YaRN long-context bonus was applied to this candidate.
    pub yarn_bonus_applied: bool,
}

/// Compute the documented combined score for one aggregate. Pure — no I/O, no
/// randomness, fully unit-testable. Missing rate/score fields degrade to `0.0`
/// for that term (never `NaN`, never a panic) — an aggregate with no compile
/// data at all just doesn't earn the compile-rate term's credit.
pub fn combined_score(agg: &CodeAggregate) -> f64 {
    let effective = agg.avg_effective_score.unwrap_or(0.0) / 5.0;
    let compile = shrunk_rate(agg.compile_pass_rate, agg.measured_compile_count);
    let test = shrunk_rate(agg.test_pass_rate, agg.measured_test_count);
    WEIGHT_EFFECTIVE_SCORE * effective + WEIGHT_COMPILE_RATE * compile + WEIGHT_TEST_RATE * test
}

/// Prior mean that a thinly-measured pass rate is pulled toward. 0.5 = "no
/// information": a rate backed by no evidence contributes neither credit nor
/// penalty relative to an even coin.
const RATE_PRIOR_MEAN: f64 = 0.5;
/// Strength of that prior, in pseudo-observations. Chosen so a 4-sample rate —
/// the real `OlympicCoder-32B` case that motivated this — is discounted
/// substantially (1.00 → 0.72) while a 30-sample rate is nearly untouched
/// (1.00 → 0.93). Larger values would over-flatten genuine differences between
/// the well-measured models that do most of the fleet's work.
const RATE_PRIOR_STRENGTH: f64 = 5.0;

/// Shrink an observed pass rate toward [`RATE_PRIOR_MEAN`] in proportion to how
/// little evidence backs it:
///
/// ```text
/// shrunk = (rate * n + PRIOR_MEAN * PRIOR_STRENGTH) / (n + PRIOR_STRENGTH)
/// ```
///
/// This is the fix for the coverage-blindness described on
/// [`CodeAggregate::measured_compile_count`]. Before it, a 1.00 from 4 measured
/// runs outranked a 0.95 from 30 — the ranking rewarded *not being measured*.
///
/// `None` (no measured runs at all) returns the prior itself rather than `0.0`.
/// That is a deliberate correction: scoring an unmeasured model as 0.0 asserts
/// it failed, which is a claim the data does not make. Returning the prior says
/// "no evidence either way", and the caller learns the difference from
/// [`SelectionBasis`] and the measured counts, not from a silently deflated score.
fn shrunk_rate(rate: Option<f64>, measured_n: i64) -> f64 {
    let n = measured_n.max(0) as f64;
    match rate {
        Some(r) if n > 0.0 => {
            (r * n + RATE_PRIOR_MEAN * RATE_PRIOR_STRENGTH) / (n + RATE_PRIOR_STRENGTH)
        }
        _ => RATE_PRIOR_MEAN,
    }
}

/// Turn a set of same-language aggregates into ranked, backend-safety-gated
/// candidates. Pure (given the yarn-lookup results already resolved) — the
/// async DB lookups happen in [`rank_for_work_type`], this is the testable
/// core.
///
/// `yarn_tokens` maps `(model_id, mem_config)` → usable ceiling tokens, for
/// candidates that have one; a missing entry means "no YaRN data" (no bonus,
/// no error — see module docs).
///
/// MoE-tagged aggregates (per [`crate::models::backends::is_moe_tagged`] — see
/// the module-level doc comment for why this signal, not
/// `is_vulkan_candidate`, is used) are EXCLUDED entirely here, before scoring
/// or sorting — they never appear anywhere in the returned `Vec`, per spec.
pub fn rank_candidates(
    aggregates: &[CodeAggregate],
    context_depth_need: ContextDepthNeed,
    yarn_tokens: &std::collections::HashMap<(String, Option<String>), f64>,
) -> Vec<CodingCandidate> {
    let mut out: Vec<CodingCandidate> = aggregates
        .iter()
        .filter(|agg| !is_moe_tagged(&agg.model_id))
        .map(|agg| {
            let base_score = combined_score(agg);
            let key = (agg.model_id.clone(), agg.mem_config.clone());
            let has_yarn_data = yarn_tokens.contains_key(&key);
            let apply_bonus = context_depth_need == ContextDepthNeed::Long && has_yarn_data;
            let combined = if apply_bonus {
                base_score + YARN_LONG_CONTEXT_BONUS
            } else {
                base_score
            };
            CodingCandidate {
                model_id: agg.model_id.clone(),
                backend_tag: agg.backend_tag.clone(),
                mem_config: agg.mem_config.clone(),
                run_count: agg.run_count,
                avg_effective_score: agg.avg_effective_score,
                compile_pass_rate: agg.compile_pass_rate,
                test_pass_rate: agg.test_pass_rate,
                measured_compile_count: agg.measured_compile_count,
                measured_test_count: agg.measured_test_count,
                measured_score_count: agg.measured_score_count,
                combined_score: combined,
                yarn_bonus_applied: apply_bonus,
            }
        })
        .collect();

    // Best-first, stable on ties (by model_id) so the ranking is deterministic
    // for tests and for the fallback ordering in CPROX-04.
    out.sort_by(|a, b| {
        b.combined_score
            .partial_cmp(&a.combined_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.model_id.cmp(&b.model_id))
    });
    out
}

/// The full CPROX-02/04 entry point: given a [`WorkTypeCode`], rank the fleet
/// for it and report what evidence the ranking rests on.
///
/// Order of preference, and why it is a preference rather than a hard rule:
/// 1. Rank on runs matching the requested shape's own sweep category — the
///    most specific evidence — but ONLY if that category passes
///    [`category_coverage_is_usable`].
/// 2. Otherwise re-load language-wide and rank on that, reporting
///    [`SelectionBasis::LanguageFallback`]. A language-wide pick is a real,
///    defensible answer; it just is not a per-shape one, and the caller is told.
/// 3. If neither has usable coverage, still return the language-wide ranking
///    but mark it [`SelectionBasis::InsufficientData`] so nothing downstream
///    presents it as evidence-based.
///
/// The fallback is never silent. That is the whole point: as of CPROX-04 the
/// `multi_file` and `deep` categories have zero measured runs for every model,
/// so a request for those shapes takes path 2 every time — and OpenHands, whose
/// workload is exactly multi-file, needs to know that its model was chosen on
/// language-wide evidence rather than on anything about multi-file work.
pub async fn rank_for_work_type(
    source: &dyn CodeProfileSource,
    work_type: &WorkTypeCode,
) -> Result<RankedSelection, SelectorError> {
    let language = work_type.language.as_str();
    let category = work_type.task_shape.to_task_category();

    let per_category = source.load_aggregates(language, Some(category)).await?;

    let (aggregates, basis) = if category_coverage_is_usable(&per_category) {
        (per_category, SelectionBasis::PerCategory)
    } else {
        let language_wide = source.load_aggregates(language, None).await?;
        let basis = if category_coverage_is_usable(&language_wide) {
            SelectionBasis::LanguageFallback
        } else {
            SelectionBasis::InsufficientData
        };
        (language_wide, basis)
    };

    let mut yarn_tokens = std::collections::HashMap::new();
    if work_type.context_depth_need == ContextDepthNeed::Long {
        for agg in &aggregates {
            if let Some(tokens) = source
                .yarn_usable_ceiling_tokens(&agg.model_id, agg.mem_config.as_deref())
                .await
            {
                yarn_tokens.insert((agg.model_id.clone(), agg.mem_config.clone()), tokens);
            }
        }
    }

    Ok(RankedSelection {
        candidates: rank_candidates(&aggregates, work_type.context_depth_need, &yarn_tokens),
        basis,
        requested_category: category.to_string(),
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Production data source (Postgres, read-only)
// ─────────────────────────────────────────────────────────────────────────────

/// Production [`CodeProfileSource`]: reads `code_profile_runs` / `model_profiles`
/// / `assistant_dimension_score` over a `sqlx::PgPool`. NO literal DSN/host —
/// matches the established pattern in `crate::serving::profile::DbProfileSource`.
/// The pool itself is built by the caller (e.g. from `terminus_rs::config::
/// intake_database_url()`); this struct only wraps it, same division of
/// responsibility as `DbProfileSource::new`/`connect`.
pub struct DbCodeProfileSource {
    pool: sqlx::PgPool,
}

impl DbCodeProfileSource {
    pub fn new(pool: sqlx::PgPool) -> Self {
        DbCodeProfileSource { pool }
    }
}

#[async_trait]
impl CodeProfileSource for DbCodeProfileSource {
    async fn load_aggregates(
        &self,
        language: &str,
        task_category: Option<&str>,
    ) -> Result<Vec<CodeAggregate>, SelectorError> {
        use sqlx::Row;

        // Mirrors `model_dual_profile`'s `(model_id, backend_tag, mem_config)`
        // grouping, adding `language` to the filter/group (see module docs on
        // why the view itself isn't reused directly). Read-only SELECT.
        //
        // `count(cpr.compiles)` / `count(cpr.tests_pass)` count NON-NULL values
        // only — that is precisely the coverage signal `avg()` throws away, and
        // the reason `measured_compile_count` exists. Do not "simplify" these
        // to `count(*)`: that would restore the bug they were added to fix.
        //
        // `$2 IS NULL OR cpr.task_category = $2` keeps one statement for both
        // the per-category and language-wide loads rather than duplicating the
        // aggregate expressions in two near-identical queries that could drift.
        //
        // `finalized` is applied PER METRIC, not as a row filter. An earlier
        // revision of this query had a blanket `AND cpr.finalized`, described as
        // "excludes in-flight runs". That description was wrong and the filter
        // was harmful — corrected here after reading the writer.
        //
        // `finalized` does not mean "this run finished". The harness writes each
        // row in two phases (`intake::storage`): Phase 1 inserts the run with
        // `finalized = false` and ALREADY carries the real `compiles` /
        // `tests_pass` results; Phase 2 is the idiom-judge pass, which patches
        // `code_quality_score`, bumps `first_pass_score`/`retry_score`, and only
        // then sets `finalized = true`. So `finalized = false` means "not yet
        // judged", NOT "not yet measured".
        //
        // The blanket filter therefore discarded valid compile evidence. Live,
        // language=rust category=blitz — the single best-measured category in
        // the constellation's own language — the split is 87 measured compiles
        // on finalized rows and 64 on unfinalized ones. The filter was throwing
        // away 64 of 151, i.e. 42% of the evidence, from the one category that
        // has any.
        //
        // But the two metric families genuinely differ, so they are treated
        // differently rather than picking one blanket answer:
        //   - `compiles` / `tests_pass` are Phase-1 facts. Every row counts.
        //   - the effective score is Phase-2-adjusted, so an unfinalized row
        //     holds a PRE-judge value that is systematically lower. Averaging
        //     those in would bias a model down in proportion to how many of its
        //     runs happen to be unjudged. Hence `FILTER (WHERE cpr.finalized)`
        //     on the score aggregate only.
        //
        // `count(cpr.compiles)` / `count(cpr.tests_pass)` count NON-NULL values
        // only — that is precisely the coverage signal `avg()` throws away, and
        // the reason `measured_compile_count` exists. Do not "simplify" these
        // to `count(*)`: that would restore the bug they were added to fix.
        // `measured_score_count` applies the same discipline to the score term,
        // which carries the largest weight (0.60) in `combined_score` and was
        // the last place a rate could still be trusted without knowing its `n`.
        //
        // `$2 IS NULL OR cpr.task_category = $2` keeps one statement for both
        // the per-category and language-wide loads rather than duplicating the
        // aggregate expressions in two near-identical queries that could drift.
        let rows = sqlx::query(
            "SELECT mp.model_name AS model_id, \
                    cpr.backend_tag, \
                    cpr.mem_config, \
                    count(*) AS run_count, \
                    count(cpr.compiles) AS measured_compile_count, \
                    count(cpr.tests_pass) AS measured_test_count, \
                    count(coalesce(cpr.retry_score, cpr.first_pass_score)) \
                        FILTER (WHERE cpr.finalized) AS measured_score_count, \
                    avg(coalesce(cpr.retry_score, cpr.first_pass_score)::float8) \
                        FILTER (WHERE cpr.finalized) AS avg_effective_score, \
                    avg(cpr.compiles::int::float8) AS compile_pass_rate, \
                    avg(cpr.tests_pass::int::float8) AS test_pass_rate \
             FROM code_profile_runs cpr \
             JOIN model_profiles mp ON mp.id = cpr.profile_id \
             WHERE cpr.language = $1 \
               AND ($2::text IS NULL OR cpr.task_category = $2) \
             GROUP BY mp.model_name, cpr.backend_tag, cpr.mem_config",
        )
        .bind(language)
        .bind(task_category)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| {
            tracing::error!(error = %e, "coding selector aggregate query failed");
            SelectorError::StoreUnavailable
        })?;

        Ok(rows
            .into_iter()
            .map(|r| CodeAggregate {
                model_id: r.get("model_id"),
                backend_tag: r.get("backend_tag"),
                mem_config: r.get("mem_config"),
                run_count: r.get("run_count"),
                avg_effective_score: r.get("avg_effective_score"),
                compile_pass_rate: r.get("compile_pass_rate"),
                test_pass_rate: r.get("test_pass_rate"),
                measured_compile_count: r.get("measured_compile_count"),
                measured_test_count: r.get("measured_test_count"),
                measured_score_count: r.get("measured_score_count"),
            })
            .collect())
    }

    async fn yarn_usable_ceiling_tokens(
        &self,
        model_id: &str,
        mem_config: Option<&str>,
    ) -> Option<f64> {
        use sqlx::Row;

        // `mem_config` is nullable and part of the non-blending key: match it
        // with `IS NOT DISTINCT FROM` (NULL = NULL) exactly like
        // `model_dual_profile`'s own join does.
        let row = sqlx::query(
            "SELECT value FROM assistant_dimension_score \
             WHERE model_id = $1 AND dimension = 'dim7_yarn_depth' \
               AND metric = 'usable_ceiling_tokens' \
               AND mem_config IS NOT DISTINCT FROM $2 \
             ORDER BY created_at DESC LIMIT 1",
        )
        .bind(model_id)
        .bind(mem_config)
        .fetch_optional(&self.pool)
        .await
        .ok()
        .flatten()?;

        row.try_get::<f64, _>("value").ok()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test/fixture data source
// ─────────────────────────────────────────────────────────────────────────────

/// Fixed-fixture [`CodeProfileSource`] for unit tests — no Postgres needed.
#[derive(Debug, Clone, Default)]
pub struct StaticCodeProfileSource {
    /// The language-wide set, returned for a `task_category = None` load.
    pub aggregates: Vec<CodeAggregate>,
    /// Per-category sets, keyed by sweep category (`blitz`/`multi_file`/`deep`).
    /// A category absent from this map returns EMPTY, exactly as production does
    /// for a category with no rows.
    pub category_aggregates: std::collections::HashMap<String, Vec<CodeAggregate>>,
    /// `(model_id, mem_config) -> usable_ceiling_tokens`.
    pub yarn: std::collections::HashMap<(String, Option<String>), f64>,
}

impl StaticCodeProfileSource {
    pub fn new(aggregates: Vec<CodeAggregate>) -> Self {
        StaticCodeProfileSource {
            aggregates,
            category_aggregates: std::collections::HashMap::new(),
            yarn: std::collections::HashMap::new(),
        }
    }

    /// Give one sweep category its own fixture set, so a test can model the
    /// real situation where `blitz` is measured and `multi_file` is not.
    pub fn with_category(mut self, category: &str, aggregates: Vec<CodeAggregate>) -> Self {
        self.category_aggregates
            .insert(category.to_string(), aggregates);
        self
    }

    pub fn with_yarn(mut self, model_id: &str, mem_config: Option<&str>, tokens: f64) -> Self {
        self.yarn
            .insert((model_id.to_string(), mem_config.map(str::to_string)), tokens);
        self
    }
}

#[async_trait]
impl CodeProfileSource for StaticCodeProfileSource {
    async fn load_aggregates(
        &self,
        language: &str,
        task_category: Option<&str>,
    ) -> Result<Vec<CodeAggregate>, SelectorError> {
        // The fixture stores all languages together; filter here to mimic the
        // production query's `WHERE language = $1`. Fixtures in tests key their
        // language via the `model_id` naming convention or store one language
        // per fixture — this filter is a no-op unless the caller sets an
        // `agg.model_id` marker; callers instead just build per-language fixture
        // sets directly, so in practice this returns everything given. Kept
        // simple: real language filtering already happened when constructing
        // the fixture set the test needs.
        let _ = language;
        match task_category {
            // `aggregates` is the language-wide set (the `task_category IS NULL`
            // load). A category with no fixture entry returns EMPTY, not the
            // language-wide set — mirroring production, where a category with no
            // rows yields no rows. Falling back to the language-wide data here
            // would make the coverage gate untestable, because every category
            // would silently look well-measured.
            Some(cat) => Ok(self.category_aggregates.get(cat).cloned().unwrap_or_default()),
            None => Ok(self.aggregates.clone()),
        }
    }

    async fn yarn_usable_ceiling_tokens(
        &self,
        model_id: &str,
        mem_config: Option<&str>,
    ) -> Option<f64> {
        self.yarn
            .get(&(model_id.to_string(), mem_config.map(str::to_string)))
            .copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A WELL-MEASURED aggregate: every one of its 16 runs recorded both a
    /// compile and a test result. Tests that care about thin coverage use
    /// [`agg_measured`] instead and pass explicit counts.
    fn agg(model: &str, mem_config: Option<&str>, eff: f64, compile: f64, test: f64) -> CodeAggregate {
        agg_measured(model, mem_config, eff, compile, test, 16, 16)
    }

    #[allow(clippy::too_many_arguments)]
    fn agg_measured(
        model: &str,
        mem_config: Option<&str>,
        eff: f64,
        compile: f64,
        test: f64,
        measured_compile: i64,
        measured_test: i64,
    ) -> CodeAggregate {
        CodeAggregate {
            model_id: model.to_string(),
            backend_tag: Some("gpu".to_string()),
            mem_config: mem_config.map(str::to_string),
            run_count: 16,
            avg_effective_score: Some(eff),
            compile_pass_rate: Some(compile),
            test_pass_rate: Some(test),
            measured_compile_count: measured_compile,
            measured_test_count: measured_test,
            measured_score_count: measured_compile,
        }
    }

    /// CPROX-04 CHANGED this formula: the rate terms are now shrunk toward a
    /// 0.5 prior by their measured sample count. The pre-CPROX-04 version of
    /// this test asserted a perfect aggregate scored exactly 1.0; that is no
    /// longer true and SHOULD no longer be true, because a rate is now credited
    /// in proportion to the evidence behind it. Updated deliberately, not
    /// relaxed to make a failure go away.
    #[test]
    fn combined_score_matches_documented_formula() {
        // 16 measured runs at rate 1.0 → (1.0*16 + 0.5*5)/(16+5) = 18.5/21
        let shrunk_perfect = 18.5f64 / 21.0;

        let a = agg("m", None, 5.0, 1.0, 1.0);
        let expected_a = 0.60 * 1.0 + 0.25 * shrunk_perfect + 0.15 * shrunk_perfect;
        assert!((combined_score(&a) - expected_a).abs() < 1e-9);

        // A rate of exactly the prior is unmoved by shrinkage at any n, so this
        // case pins the formula's midpoint independently of the sample count.
        let c = agg("m", None, 2.5, 0.5, 0.5);
        assert!((combined_score(&c) - 0.5).abs() < 1e-9);
    }

    /// The OlympicCoder-32B case from the live intake DB: `compile_pass_rate`
    /// and `test_pass_rate` both read 1.00, from only 4 measured runs of 16.
    /// Before CPROX-04 that outranked a better-evidenced rival; the ranking
    /// rewarded NOT being measured. This is the regression test for that.
    #[test]
    fn thinly_measured_perfect_rate_loses_to_a_well_measured_slightly_worse_one() {
        let thin = agg_measured("thin-4-of-16", None, 2.0, 1.0, 1.0, 4, 4);
        let thick = agg_measured("thick-30", None, 2.0, 0.95, 0.95, 30, 30);

        assert!(
            combined_score(&thick) > combined_score(&thin),
            "30-sample 0.95 must beat 4-sample 1.00 — thin={} thick={}",
            combined_score(&thin),
            combined_score(&thick)
        );

        // And the discount is substantial, not cosmetic: 4 samples at 1.00
        // lands nearer 0.72 than 1.00.
        assert!((shrunk_rate(Some(1.0), 4) - 0.7222222222).abs() < 1e-6);
        assert!((shrunk_rate(Some(1.0), 30) - 0.9285714286).abs() < 1e-6);
    }

    /// A rate with NO measured runs is "no evidence", not "measured and failed".
    /// Scoring it 0.0 (the pre-CPROX-04 behaviour) asserted a failure the data
    /// never observed.
    #[test]
    fn missing_rates_degrade_to_the_prior_not_to_zero() {
        let a = CodeAggregate {
            model_id: "m".into(),
            backend_tag: None,
            mem_config: None,
            run_count: 0,
            avg_effective_score: None,
            compile_pass_rate: None,
            test_pass_rate: None,
            measured_compile_count: 0,
            measured_test_count: 0,
            measured_score_count: 0,
        };
        // 0.60*0 + 0.25*0.5 + 0.15*0.5 = 0.20 — and crucially not a panic/NaN.
        assert!((combined_score(&a) - 0.20).abs() < 1e-9);
        assert!(combined_score(&a).is_finite());
    }

    #[test]
    fn candidates_never_blend_mem_config() {
        // Same model_id, wildly different scores per mem_config — mirrors the
        // real qwen3-coder:30b split (dynamic_gtt ~1.75 vs legacy ~4.19).
        let aggregates = vec![
            agg("qwen3-coder:30b", Some("dynamic_gtt"), 1.75, 1.0, 0.89),
            agg("qwen3-coder:30b", None, 4.19, 0.94, 0.93),
        ];
        let ranked = rank_candidates(&aggregates, ContextDepthNeed::Short, &Default::default());
        assert_eq!(ranked.len(), 2, "each mem_config must remain its own candidate");

        let dynamic = ranked
            .iter()
            .find(|c| c.mem_config.as_deref() == Some("dynamic_gtt"))
            .expect("dynamic_gtt candidate present");
        let legacy = ranked
            .iter()
            .find(|c| c.mem_config.is_none())
            .expect("legacy/untagged candidate present");

        // The scores must be independently computed from each row's OWN data,
        // not an average of the two — this is the load-bearing assertion.
        assert!(
            (dynamic.combined_score - combined_score(&aggregates[0])).abs() < 1e-9,
            "dynamic_gtt candidate score must come from ITS OWN row only"
        );
        assert!(
            (legacy.combined_score - combined_score(&aggregates[1])).abs() < 1e-9,
            "legacy candidate score must come from ITS OWN row only"
        );
        assert!(legacy.combined_score > dynamic.combined_score);
        // The better (legacy) row must rank first.
        assert_eq!(ranked[0].mem_config, None);
    }

    #[test]
    fn ranking_is_best_first_and_deterministic_on_ties() {
        let aggregates = vec![
            agg("model-b", None, 3.0, 0.5, 0.5),
            agg("model-a", None, 3.0, 0.5, 0.5),
            agg("model-c", None, 5.0, 1.0, 1.0),
        ];
        let ranked = rank_candidates(&aggregates, ContextDepthNeed::Short, &Default::default());
        assert_eq!(ranked[0].model_id, "model-c");
        // Tie between a and b broken deterministically by model_id.
        assert_eq!(ranked[1].model_id, "model-a");
        assert_eq!(ranked[2].model_id, "model-b");
    }

    #[test]
    fn yarn_bonus_only_applied_for_long_context_need_with_data() {
        let aggregates = vec![
            agg("no-yarn-data", None, 4.0, 1.0, 1.0),
            agg("has-yarn-data", None, 3.9, 1.0, 1.0),
        ];
        let mut yarn = std::collections::HashMap::new();
        yarn.insert(("has-yarn-data".to_string(), None), 131072.0);

        // Long context need + data present ⇒ bonus applied, can overtake a
        // slightly-better short-context-only score.
        let ranked_long = rank_candidates(&aggregates, ContextDepthNeed::Long, &yarn);
        let has_data = ranked_long.iter().find(|c| c.model_id == "has-yarn-data").unwrap();
        let no_data = ranked_long.iter().find(|c| c.model_id == "no-yarn-data").unwrap();
        assert!(has_data.yarn_bonus_applied);
        assert!(!no_data.yarn_bonus_applied);
        assert!(has_data.combined_score > no_data.combined_score);
        assert_eq!(ranked_long[0].model_id, "has-yarn-data");

        // Short context need ⇒ no bonus applied even though yarn data exists.
        let ranked_short = rank_candidates(&aggregates, ContextDepthNeed::Short, &yarn);
        let has_data_short = ranked_short.iter().find(|c| c.model_id == "has-yarn-data").unwrap();
        assert!(!has_data_short.yarn_bonus_applied);
        assert_eq!(ranked_short[0].model_id, "no-yarn-data");
    }

    #[test]
    fn missing_yarn_data_degrades_gracefully_no_error() {
        // The common case today: NO model has dim7_yarn_depth data yet.
        let aggregates = vec![agg("model-x", None, 4.0, 1.0, 1.0)];
        let ranked = rank_candidates(&aggregates, ContextDepthNeed::Long, &Default::default());
        assert!(!ranked[0].yarn_bonus_applied);
        assert!((ranked[0].combined_score - combined_score(&aggregates[0])).abs() < 1e-9);
    }

    #[test]
    fn moe_tagged_candidates_are_excluded_entirely_not_flagged() {
        // A tag-flagged MoE model (a3b-class) even with a TOP score must never
        // appear in the returned list at all — not with a warning, not as a
        // lower-ranked entry, not anywhere. This is the blocking-bug regression
        // test: an MoE candidate that scores well must not become "the pick".
        let aggregates = vec![
            agg("qwen3-a3b-coder:30b", None, 5.0, 1.0, 1.0), // MoE-tagged, best score
            agg("llama3.3:70b", None, 3.0, 0.8, 0.8),        // dense, lower score
        ];
        let ranked = rank_candidates(&aggregates, ContextDepthNeed::Short, &Default::default());

        assert_eq!(ranked.len(), 1, "the MoE-tagged candidate must be excluded, not just flagged");
        assert_eq!(ranked[0].model_id, "llama3.3:70b");
        assert!(
            ranked.iter().all(|c| c.model_id != "qwen3-a3b-coder:30b"),
            "an MoE candidate must never appear in the ranked list, regardless of score"
        );
        // No safety flag is exposed at all — there's nothing to flag once
        // exclusion is real (see the module docs on why this field was removed).
    }

    #[test]
    fn dense_non_32b_class_models_are_not_wrongly_excluded() {
        // Regression guard for the original bug's root cause: using
        // `is_vulkan_candidate` (which also gates on the 32B/34B/70B/72B size
        // allowlist) as the exclusion signal would wrongly drop real, non-MoE
        // dense models that just aren't in that size class. `is_moe_tagged`
        // must NOT exclude these.
        let aggregates = vec![
            agg("devstral:24b", None, 3.5, 0.8, 0.8),
            agg("gemma3:12b", None, 3.2, 0.7, 0.7),
            agg("codestral:latest", None, 4.0, 0.9, 0.9),
        ];
        let ranked = rank_candidates(&aggregates, ContextDepthNeed::Short, &Default::default());
        assert_eq!(ranked.len(), 3, "non-MoE dense models below the vulkan size allowlist must survive");
    }

    #[tokio::test]
    async fn rank_for_work_type_end_to_end_with_fixture_source() {
        let source = StaticCodeProfileSource::new(vec![
            agg("model-a", None, 4.0, 1.0, 1.0),
            agg("model-b", Some("dynamic_gtt"), 4.5, 1.0, 1.0),
        ])
        .with_yarn("model-b", Some("dynamic_gtt"), 65536.0);

        let wtc = WorkTypeCode {
            language: crate::models::work_type::Language::Rust,
            task_shape: crate::models::work_type::TaskShape::MultiFileBuild,
            reasoning_need: crate::models::work_type::ReasoningNeed::Enrich,
            context_depth_need: ContextDepthNeed::Long,
        };
        let ranked = rank_for_work_type(&source, &wtc).await.expect("ranks");
        assert_eq!(ranked.candidates.len(), 2);
        // model-b has both a higher base score AND the yarn bonus.
        assert_eq!(ranked.candidates[0].model_id, "model-b");
        assert!(ranked.candidates[0].yarn_bonus_applied);
        // …and the selection reports the evidence it actually rests on: no
        // `multi_file` fixture was registered, so the per-category load is
        // empty, and the language-wide set has only 2 candidates — below
        // MIN_QUALIFYING_CANDIDATES. That is InsufficientData, not a silently
        // presented per-category ranking, and the requested shape is named.
        assert_eq!(ranked.basis, SelectionBasis::InsufficientData);
        assert_eq!(ranked.requested_category, "multi_file");
    }

    #[tokio::test]
    #[ignore = "gated integration test — requires a live read-only intake DB \
                connection; run with `cargo test -- --ignored` and \
                INTAKE_DATABASE_URL (or DATABASE_URL) set"]
    async fn live_db_rust_aggregates_are_never_blended_across_mem_config() {
        let url = std::env::var("INTAKE_DATABASE_URL")
            .or_else(|_| std::env::var("DATABASE_URL"))
            .expect("INTAKE_DATABASE_URL or DATABASE_URL must be set for this ignored test");
        let pool = sqlx::PgPool::connect(&url).await.expect("connect");
        let source = DbCodeProfileSource::new(pool);
        let aggregates = source.load_aggregates("rust", None).await.expect("query ok");
        assert!(!aggregates.is_empty(), "expected live rust aggregates");

        // Group by model_id, collecting (mem_config, avg_effective_score) per row.
        let mut by_model: std::collections::BTreeMap<&str, Vec<(Option<String>, Option<f64>)>> =
            Default::default();
        for a in &aggregates {
            by_model
                .entry(a.model_id.as_str())
                .or_default()
                .push((a.mem_config.clone(), a.avg_effective_score));
        }

        // For EVERY model, each distinct mem_config value it has data under
        // must have produced its OWN row — a regression that dropped
        // `mem_config` from the SQL `GROUP BY` would collapse these into fewer
        // rows than distinct configs (or fail with a Postgres "column must
        // appear in GROUP BY" error before we even get here).
        let mut found_dual_config_model = false;
        for (model, rows) in &by_model {
            let distinct_configs: std::collections::BTreeSet<&Option<String>> =
                rows.iter().map(|(c, _)| c).collect();
            assert!(
                rows.len() >= distinct_configs.len(),
                "model {model} produced fewer aggregate rows ({}) than distinct \
                 mem_config values ({}) it has data under — mem_config may have been \
                 dropped from the GROUP BY",
                rows.len(),
                distinct_configs.len()
            );

            // The load-bearing assertion: a model with BOTH a `dynamic_gtt` row
            // and a legacy/untagged row must show DIFFERENT scores. As of this
            // writing several real models qualify (e.g. `qwen3-coder:30b`:
            // ~4.19 untagged vs. ~1.75 under `dynamic_gtt`) — if a future
            // regression silently blended them, this is exactly the check that
            // would fail (blended rows either collapse to one row, tripping the
            // assertion above, or — if the bug is instead in the AVG itself —
            // would produce the SAME score for both configs, tripping this one).
            if distinct_configs.len() >= 2 {
                found_dual_config_model = true;
                let scores: Vec<f64> = rows.iter().filter_map(|(_, s)| *s).collect();
                if scores.len() >= 2 {
                    let all_equal = scores.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-9);
                    assert!(
                        !all_equal,
                        "model {model} has DIFFERENT mem_config rows but IDENTICAL \
                         avg_effective_score values ({scores:?}) — this is exactly the \
                         blending regression this test exists to catch"
                    );
                }
            }
        }
        assert!(
            found_dual_config_model,
            "expected at least one live Rust model with rows under BOTH a mem_config \
             value and legacy/untagged (e.g. qwen3-coder:30b as of this writing) — \
             without one, this test cannot actually exercise the non-blending guarantee. \
             If the live sweep data has changed shape, update this test's expectations \
             rather than deleting the assertion."
        );
    }
}

#[cfg(test)]
mod cprox04_basis_tests {
    use super::*;
    use crate::models::work_type::{Language, ReasoningNeed, TaskShape};

    fn wt(shape: TaskShape) -> WorkTypeCode {
        WorkTypeCode {
            language: Language::Rust,
            task_shape: shape,
            reasoning_need: ReasoningNeed::Execute,
            context_depth_need: ContextDepthNeed::Short,
        }
    }

    fn measured(model: &str, eff: f64, n: i64) -> CodeAggregate {
        CodeAggregate {
            model_id: model.to_string(),
            backend_tag: None,
            mem_config: Some("carveout".into()),
            run_count: n,
            avg_effective_score: Some(eff),
            compile_pass_rate: Some(1.0),
            test_pass_rate: Some(1.0),
            measured_compile_count: n,
            measured_test_count: n,
            measured_score_count: n,
        }
    }

    /// The live `multi_file` / `deep` shape: rows EXIST (run_count > 0) and
    /// carry a real `avg_effective_score` of 0.0, but nothing was ever measured.
    /// This is the trap the basis field exists for — it looks like data.
    fn unmeasured(model: &str, n: i64) -> CodeAggregate {
        CodeAggregate {
            model_id: model.to_string(),
            backend_tag: None,
            mem_config: Some("carveout".into()),
            run_count: n,
            avg_effective_score: Some(0.0),
            compile_pass_rate: None,
            test_pass_rate: None,
            measured_compile_count: 0,
            measured_test_count: 0,
            measured_score_count: 0,
        }
    }

    #[tokio::test]
    async fn well_measured_category_ranks_on_itself() {
        let source = StaticCodeProfileSource::new(vec![measured("lang-wide", 1.0, 40)])
            .with_category(
                "blitz",
                vec![
                    measured("blitz-a", 4.9, 12),
                    measured("blitz-b", 4.5, 20),
                    measured("blitz-c", 3.8, 12),
                ],
            );

        let sel = rank_for_work_type(&source, &wt(TaskShape::QuickEdit)).await.unwrap();

        assert_eq!(sel.basis, SelectionBasis::PerCategory);
        assert_eq!(sel.requested_category, "blitz");
        assert_eq!(sel.candidates[0].model_id, "blitz-a");
        // The language-wide row must NOT leak into a per-category ranking.
        assert!(sel.candidates.iter().all(|c| c.model_id != "lang-wide"));
    }

    /// The OpenHands case. `multi_file` has rows but zero measurement, so the
    /// ranking must come from language-wide data AND must say so — a caller
    /// that reads only `candidates[0]` would otherwise believe Chord had picked
    /// a model on multi-file evidence that does not exist.
    #[tokio::test]
    async fn unmeasured_category_falls_back_to_language_and_reports_it() {
        let source = StaticCodeProfileSource::new(vec![
            measured("lang-a", 4.2, 30),
            measured("lang-b", 3.1, 30),
            measured("lang-c", 2.0, 30),
        ])
        .with_category(
            "multi_file",
            vec![unmeasured("mf-a", 18), unmeasured("mf-b", 12), unmeasured("mf-c", 12)],
        );

        let sel = rank_for_work_type(&source, &wt(TaskShape::MultiFileBuild)).await.unwrap();

        assert_eq!(
            sel.basis,
            SelectionBasis::LanguageFallback,
            "an all-unmeasured category must never rank as PerCategory"
        );
        assert_eq!(sel.requested_category, "multi_file");
        assert_eq!(sel.candidates[0].model_id, "lang-a");
        assert!(sel.candidates.iter().all(|c| !c.model_id.starts_with("mf-")));
    }

    /// Every zero-scored unmeasured row still produces a NON-EMPTY, ordered
    /// candidate list. That is exactly why ordering alone cannot be trusted as
    /// a signal, and why the basis field carries the verdict instead.
    #[tokio::test]
    async fn nothing_measured_anywhere_reports_insufficient_data() {
        let source = StaticCodeProfileSource::new(vec![unmeasured("x", 10), unmeasured("y", 10)])
            .with_category("deep", vec![unmeasured("d", 12)]);

        let sel = rank_for_work_type(&source, &wt(TaskShape::Deep)).await.unwrap();

        assert_eq!(sel.basis, SelectionBasis::InsufficientData);
        assert_eq!(sel.requested_category, "deep");
        assert!(
            !sel.candidates.is_empty(),
            "a ranked list still comes back — the basis, not emptiness, is what warns the caller"
        );
    }

    /// One well-measured model is not a ranking. Below the qualifying-candidate
    /// floor the category must not be ranked on its own, or a single model with
    /// a handful of runs becomes the permanent answer for that shape.
    #[tokio::test]
    async fn a_single_measured_candidate_does_not_qualify_a_category() {
        let source = StaticCodeProfileSource::new(vec![
            measured("lang-a", 4.0, 30),
            measured("lang-b", 3.0, 30),
            measured("lang-c", 2.0, 30),
        ])
        .with_category("blitz", vec![measured("only-one", 5.0, 40), unmeasured("z", 10)]);

        let sel = rank_for_work_type(&source, &wt(TaskShape::QuickEdit)).await.unwrap();

        assert_eq!(sel.basis, SelectionBasis::LanguageFallback);
        assert!(sel.candidates.iter().all(|c| c.model_id != "only-one"));
    }

    /// A candidate measured on fewer runs than the per-candidate floor does not
    /// count toward category coverage, however many such candidates there are.
    #[test]
    fn coverage_gate_requires_real_depth_not_just_breadth() {
        let barely: Vec<CodeAggregate> = (0..10)
            .map(|i| {
                let mut a = measured(&format!("m{i}"), 4.0, 3);
                a.measured_compile_count = MIN_MEASURED_RUNS_PER_CANDIDATE - 1;
                a
            })
            .collect();
        assert!(!category_coverage_is_usable(&barely));

        let enough: Vec<CodeAggregate> =
            (0..MIN_QUALIFYING_CANDIDATES).map(|i| measured(&format!("m{i}"), 4.0, 8)).collect();
        assert!(category_coverage_is_usable(&enough));
    }
}
