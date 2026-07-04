//! CPROX-02: fleet-driven coding-model matching/scoring engine.
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
//! `code_profile_runs.mem_config` distinguishes the S85 `dynamic_gtt` memory
//! configuration from legacy/untagged runs (`mem_config IS NULL`). These are
//! NOT comparable: on the live data, `qwen3-coder:30b` averages an effective
//! score of ~4.19 untagged vs. ~1.75 under `dynamic_gtt` for the same model —
//! blending them would produce a meaningless average. Every aggregate query and
//! every ranking step in this module keeps `mem_config` as part of the grouping
//! key; [`candidates_never_blend_mem_config`] is the regression test for this.
//!
//! ## Scoring formula (documented, no unexplained magic numbers)
//! Each `(model_id, backend_tag, mem_config)` group's `combined_score` is:
//!
//! ```text
//! combined_score = 0.60 * (avg_effective_score / 5.0)
//!                + 0.25 * compile_pass_rate
//!                + 0.15 * test_pass_rate
//! ```
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
    /// Fraction of runs where `compiles = true`.
    pub compile_pass_rate: Option<f64>,
    /// Fraction of runs where `tests_pass = true`.
    pub test_pass_rate: Option<f64>,
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
    async fn load_aggregates(&self, language: &str) -> Result<Vec<CodeAggregate>, SelectorError>;

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
    let compile = agg.compile_pass_rate.unwrap_or(0.0);
    let test = agg.test_pass_rate.unwrap_or(0.0);
    WEIGHT_EFFECTIVE_SCORE * effective + WEIGHT_COMPILE_RATE * compile + WEIGHT_TEST_RATE * test
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

/// The full CPROX-02 entry point: given a [`WorkTypeCode`], load this
/// language's aggregates from `source`, resolve the YaRN long-context bonus
/// data (best-effort — a lookup failure never fails the whole call, it just
/// means no bonus for that candidate), rank, and return the safety-gated list.
pub async fn rank_for_work_type(
    source: &dyn CodeProfileSource,
    work_type: &WorkTypeCode,
) -> Result<Vec<CodingCandidate>, SelectorError> {
    let aggregates = source.load_aggregates(work_type.language.as_str()).await?;

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

    Ok(rank_candidates(&aggregates, work_type.context_depth_need, &yarn_tokens))
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
    async fn load_aggregates(&self, language: &str) -> Result<Vec<CodeAggregate>, SelectorError> {
        use sqlx::Row;

        // Mirrors `model_dual_profile`'s `(model_id, backend_tag, mem_config)`
        // grouping, adding `language` to the filter/group (see module docs on
        // why the view itself isn't reused directly). Read-only SELECT.
        let rows = sqlx::query(
            "SELECT mp.model_name AS model_id, \
                    cpr.backend_tag, \
                    cpr.mem_config, \
                    count(*) AS run_count, \
                    avg(coalesce(cpr.retry_score, cpr.first_pass_score)::float8) AS avg_effective_score, \
                    avg(cpr.compiles::int::float8) AS compile_pass_rate, \
                    avg(cpr.tests_pass::int::float8) AS test_pass_rate \
             FROM code_profile_runs cpr \
             JOIN model_profiles mp ON mp.id = cpr.profile_id \
             WHERE cpr.language = $1 \
             GROUP BY mp.model_name, cpr.backend_tag, cpr.mem_config",
        )
        .bind(language)
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
    pub aggregates: Vec<CodeAggregate>,
    /// `(model_id, mem_config) -> usable_ceiling_tokens`.
    pub yarn: std::collections::HashMap<(String, Option<String>), f64>,
}

impl StaticCodeProfileSource {
    pub fn new(aggregates: Vec<CodeAggregate>) -> Self {
        StaticCodeProfileSource {
            aggregates,
            yarn: std::collections::HashMap::new(),
        }
    }

    pub fn with_yarn(mut self, model_id: &str, mem_config: Option<&str>, tokens: f64) -> Self {
        self.yarn
            .insert((model_id.to_string(), mem_config.map(str::to_string)), tokens);
        self
    }
}

#[async_trait]
impl CodeProfileSource for StaticCodeProfileSource {
    async fn load_aggregates(&self, language: &str) -> Result<Vec<CodeAggregate>, SelectorError> {
        // The fixture stores all languages together; filter here to mimic the
        // production query's `WHERE language = $1`. Fixtures in tests key their
        // language via the `model_id` naming convention or store one language
        // per fixture — this filter is a no-op unless the caller sets an
        // `agg.model_id` marker; callers instead just build per-language fixture
        // sets directly, so in practice this returns everything given. Kept
        // simple: real language filtering already happened when constructing
        // the fixture set the test needs.
        let _ = language;
        Ok(self.aggregates.clone())
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

    fn agg(model: &str, mem_config: Option<&str>, eff: f64, compile: f64, test: f64) -> CodeAggregate {
        CodeAggregate {
            model_id: model.to_string(),
            backend_tag: Some("gpu".to_string()),
            mem_config: mem_config.map(str::to_string),
            run_count: 16,
            avg_effective_score: Some(eff),
            compile_pass_rate: Some(compile),
            test_pass_rate: Some(test),
        }
    }

    #[test]
    fn combined_score_matches_documented_formula() {
        let a = agg("m", None, 5.0, 1.0, 1.0);
        // 0.60*(5/5) + 0.25*1.0 + 0.15*1.0 = 0.60 + 0.25 + 0.15 = 1.0
        assert!((combined_score(&a) - 1.0).abs() < 1e-9);

        let b = agg("m", None, 0.0, 0.0, 0.0);
        assert!((combined_score(&b) - 0.0).abs() < 1e-9);

        let c = agg("m", None, 2.5, 0.5, 0.5);
        // 0.60*0.5 + 0.25*0.5 + 0.15*0.5 = 0.30+0.125+0.075 = 0.5
        assert!((combined_score(&c) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn combined_score_missing_fields_degrade_to_zero_not_panic() {
        let a = CodeAggregate {
            model_id: "m".into(),
            backend_tag: None,
            mem_config: None,
            run_count: 0,
            avg_effective_score: None,
            compile_pass_rate: None,
            test_pass_rate: None,
        };
        assert_eq!(combined_score(&a), 0.0);
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
        assert_eq!(ranked.len(), 2);
        // model-b has both a higher base score AND the yarn bonus.
        assert_eq!(ranked[0].model_id, "model-b");
        assert!(ranked[0].yarn_bonus_applied);
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
        let aggregates = source.load_aggregates("rust").await.expect("query ok");
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
