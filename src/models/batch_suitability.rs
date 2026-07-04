//! BSUIT-01: batch-suitability scoring engine.
//!
//! Given a target `language` (a coder-sweep language tag: `rust` / `python` /
//! `bash` / `typescript`), rank the REAL benchmarked fleet — the intake
//! `model_language_stats` materialized view, joined to `model_profiles` for the
//! human model name — by a composite "how well does this model do BATCH coding
//! work in this language" score. This is the batch-job sibling of
//! [`crate::models::coding_selector`]: that module ranks raw `code_profile_runs`
//! for a per-request pick; this one ranks the pre-aggregated per-language stats
//! for a "which local model should own a whole batch job" decision, folding in
//! throughput/latency/malformed/error signals the per-request path doesn't need.
//!
//! ## Data source: the `model_language_stats` view (not raw runs)
//! The intake schema already exposes `model_language_stats`, one row per
//! `(profile_id, language)` with `n_scored, mean_score, stddev_score,
//! retry_lift, mean_throughput, mean_latency_ms, p95_latency_ms, malformed_rate,
//! error_rate` pre-computed. Reusing that view (rather than re-aggregating
//! `code_profile_runs` a second time, the way `coding_selector` must because it
//! also needs `mem_config`-split grouping the view doesn't carry) keeps this
//! module a thin ranking layer over an existing aggregate — same "prefer the
//! existing view when it already answers the question" discipline documented in
//! `coding_selector`'s module header, applied the other way round because here
//! the view DOES answer the question.
//!
//! ## Scoring formula (documented, no unexplained magic numbers)
//! For a candidate set (all rows for one `language`), each candidate's
//! `suitability_score` is a weighted sum of terms normalized ACROSS that set —
//! `mean_throughput` (tokens/s) and `mean_latency_ms` live on wildly different
//! scales, so a min-max normalization within the language cohort is what makes
//! them comparable:
//!
//! ```text
//! score =  W_MEAN_SCORE   * norm(mean_score)
//!        - W_STDDEV_PEN    * norm(stddev_score)
//!        + W_THROUGHPUT    * norm(mean_throughput)
//!        - W_LATENCY_PEN   * norm(mean_latency_ms)
//!        + W_RELIABILITY   * (1 - malformed_rate)
//!        - W_ERROR_PEN     * error_rate
//! ```
//!
//! Weights (documented rationale, deliberately quality-first — this is a
//! defensible STARTING composite, not a tuned one):
//!   - `W_MEAN_SCORE = 0.40` — the sweep's own 0-5 quality score is the single
//!     most important signal for "will this model produce good code", so it
//!     dominates.
//!   - `W_STDDEV_PEN = 0.20` — consistency is the second concern: a model that
//!     is sometimes great and sometimes terrible is worse for an unattended
//!     BATCH job than a slightly-lower but steady one, so run-to-run variance
//!     is penalized heavily (second-highest weight).
//!   - `W_RELIABILITY = 0.15` — `(1 - malformed_rate)`: a model that emits
//!     unparseable/malformed output wastes the whole batch slot; rewarded
//!     directly (not just via the score) so well-formedness is its own credit.
//!   - `W_THROUGHPUT = 0.10` / `W_LATENCY_PEN = 0.10` — speed/cost matter for a
//!     batch but are explicitly LOWER priority than quality+consistency: a fast
//!     wrong answer is worthless. Throughput rewarded, latency penalized, equal
//!     small weights.
//!   - `W_ERROR_PEN = 0.05` — hard error rate (request-level failures) gets a
//!     small extra penalty on top of whatever it already did to the quality
//!     aggregate; kept small to avoid double-counting failures the mean_score
//!     already reflects.
//!
//! The positive weights don't sum to 1 and don't need to — `suitability_score`
//! is only ever used to RANK candidates within one language cohort relative to
//! each other, never as an absolute probability.
//!
//! ## Normalization edge cases
//! `norm(v, min, max)` is min-max within the cohort. When every candidate shares
//! a value (or there is a single candidate), `max == min` and the term returns
//! `0.0` for everyone — carrying no discriminating signal, which is correct: a
//! dimension on which all candidates are identical must not change their
//! relative order. A candidate missing a secondary field (`NULL`
//! `stddev_score`/`throughput`/`latency` while still having a `mean_score`)
//! contributes `0.0` for just that term rather than crashing.
//!
//! ## Missing/empty stats — EXCLUSION, documented
//! A `model_language_stats` row with `n_scored = 0` carries `NULL` `mean_score`
//! (and NULL stddev/throughput/latency), `error_rate = 1.0`: the model was
//! attempted for this language but produced NO scorable result. Such a row has
//! no quality signal to rank on, so it is **excluded from the ranked list
//! entirely** (same "an unrankable candidate never becomes a candidate"
//! discipline as `coding_selector`'s MoE exclusion) — [`rank_batch_candidates`]
//! drops every row whose `mean_score` is `NULL` or whose `n_scored` is `0`
//! before scoring. This never crashes the ranking and never invents a score for
//! a model that has none; the excluded count is returned so a caller can report
//! "N models had no data for `rust`" honestly.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Weight of the sweep's own graduated 0-5 mean quality score. See the
/// module-level "Scoring formula" doc comment for the full rationale.
const W_MEAN_SCORE: f64 = 0.40;
/// Penalty weight on run-to-run score variance (consistency).
const W_STDDEV_PEN: f64 = 0.20;
/// Reward weight on well-formedness `(1 - malformed_rate)`.
const W_RELIABILITY: f64 = 0.15;
/// Reward weight on throughput (tokens/s).
const W_THROUGHPUT: f64 = 0.10;
/// Penalty weight on mean latency.
const W_LATENCY_PEN: f64 = 0.10;
/// Penalty weight on request-level error rate.
const W_ERROR_PEN: f64 = 0.05;

/// One row from `model_language_stats`, joined to `model_profiles` for the
/// human-readable `model_id` (`model_profiles.model_name`). This is the
/// pre-ranking data shape; [`BatchStatsSource`] implementations produce these,
/// [`rank_batch_candidates`] turns the ones with real data into scored,
/// best-first [`BatchCandidate`]s.
///
/// `mean_score`/`stddev_score`/`mean_throughput`/`mean_latency_ms` are `Option`
/// because a row with `n_scored = 0` has them `NULL` in the view — see the
/// module docs' exclusion rule.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LanguageStat {
    /// `model_profiles.model_name` — the human model id (e.g. `qwen3-coder:30b`).
    pub model_id: String,
    /// `model_language_stats.profile_id` (a `model_profiles.id` UUID), stringified.
    pub profile_id: String,
    /// The coder-sweep language tag this stat row is for.
    pub language: String,
    /// Number of scored runs backing this aggregate. `0` ⇒ excluded (no data).
    pub n_scored: i64,
    /// Mean graduated 0-5 quality score. `None` ⇒ `n_scored = 0` ⇒ excluded.
    pub mean_score: Option<f64>,
    /// Standard deviation of the per-run score (consistency signal).
    pub stddev_score: Option<f64>,
    /// Mean throughput, tokens/sec.
    pub mean_throughput: Option<f64>,
    /// Mean end-to-end latency, milliseconds.
    pub mean_latency_ms: Option<f64>,
    /// Fraction `[0,1]` of runs whose output was malformed/unparseable.
    pub malformed_rate: f64,
    /// Fraction `[0,1]` of runs that hard-errored (request-level failure).
    pub error_rate: f64,
}

/// A ranked batch-suitability candidate — the model kept its own row's data and
/// earned a composite [`Self::suitability_score`]. Only rows with real measured
/// data become one of these (see the module docs' exclusion rule).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchCandidate {
    pub model_id: String,
    pub profile_id: String,
    pub n_scored: i64,
    pub mean_score: Option<f64>,
    pub stddev_score: Option<f64>,
    pub mean_throughput: Option<f64>,
    pub mean_latency_ms: Option<f64>,
    pub malformed_rate: f64,
    pub error_rate: f64,
    /// Composite ranking score (see the module docs' formula). Higher is better.
    /// Relative-only — meaningful for ordering within one language cohort, not
    /// as an absolute number.
    pub suitability_score: f64,
}

/// The full ranking result: the best-first candidate list plus how many rows
/// were dropped for having no scorable data (so a caller can report the
/// exclusion honestly rather than silently). See the module docs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchRanking {
    pub language: String,
    pub candidates: Vec<BatchCandidate>,
    /// Count of `model_language_stats` rows dropped because `mean_score` was
    /// `NULL` / `n_scored` was `0` (no data to rank on for this language).
    pub excluded_no_data: usize,
}

/// Per-cohort min/max bounds for the four unbounded fields, used by
/// [`suitability_score`] to normalize them into `[0,1]`. `malformed_rate` and
/// `error_rate` are already `[0,1]` and are NOT normalized (a 5% error rate
/// should mean the same thing regardless of the cohort). Computed over the
/// candidate set's non-`NULL` values only.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    pub mean_score: (f64, f64),
    pub stddev_score: (f64, f64),
    pub throughput: (f64, f64),
    pub latency: (f64, f64),
}

/// Min-max normalize `v` into `[0,1]` given inclusive bounds. Returns `0.0`
/// when `max <= min` (a degenerate cohort where this dimension carries no
/// discriminating signal — see the module docs) rather than dividing by zero.
pub fn normalize(v: f64, min: f64, max: f64) -> f64 {
    if max <= min {
        0.0
    } else {
        ((v - min) / (max - min)).clamp(0.0, 1.0)
    }
}

/// Compute per-field [`Bounds`] over a set of stats, considering only the
/// non-`NULL` value of each field. A field for which no candidate has a value
/// gets a degenerate `(0.0, 0.0)` bound (every `normalize` against it returns
/// `0.0`).
pub fn compute_bounds(stats: &[LanguageStat]) -> Bounds {
    fn range(vals: impl Iterator<Item = f64>) -> (f64, f64) {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for v in vals {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }
        if min.is_finite() && max.is_finite() {
            (min, max)
        } else {
            (0.0, 0.0)
        }
    }
    Bounds {
        mean_score: range(stats.iter().filter_map(|s| s.mean_score)),
        stddev_score: range(stats.iter().filter_map(|s| s.stddev_score)),
        throughput: range(stats.iter().filter_map(|s| s.mean_throughput)),
        latency: range(stats.iter().filter_map(|s| s.mean_latency_ms)),
    }
}

/// Compute the documented composite suitability score for one stat, given the
/// cohort [`Bounds`]. Pure — no I/O, no randomness, fully unit-testable. A
/// missing (`None`) unbounded field contributes `0.0` for its term (never
/// `NaN`, never a panic). `malformed_rate`/`error_rate` are used directly
/// (already `[0,1]`), NOT normalized.
pub fn suitability_score(stat: &LanguageStat, bounds: &Bounds) -> f64 {
    let mean = stat
        .mean_score
        .map(|v| normalize(v, bounds.mean_score.0, bounds.mean_score.1))
        .unwrap_or(0.0);
    let stddev = stat
        .stddev_score
        .map(|v| normalize(v, bounds.stddev_score.0, bounds.stddev_score.1))
        .unwrap_or(0.0);
    let throughput = stat
        .mean_throughput
        .map(|v| normalize(v, bounds.throughput.0, bounds.throughput.1))
        .unwrap_or(0.0);
    let latency = stat
        .mean_latency_ms
        .map(|v| normalize(v, bounds.latency.0, bounds.latency.1))
        .unwrap_or(0.0);

    W_MEAN_SCORE * mean - W_STDDEV_PEN * stddev + W_THROUGHPUT * throughput
        - W_LATENCY_PEN * latency
        + W_RELIABILITY * (1.0 - stat.malformed_rate)
        - W_ERROR_PEN * stat.error_rate
}

/// Rank a language cohort's stats into best-first [`BatchCandidate`]s. Rows with
/// no scorable data (`mean_score` `NULL` or `n_scored == 0`) are EXCLUDED before
/// scoring — never ranked, never crash the ranking (see the module docs). The
/// returned [`BatchRanking`] carries the excluded count so the exclusion is
/// reportable rather than silent. Pure (no I/O) — the DB lookup happens in
/// [`rank_batch_for_language`], this is the testable core.
pub fn rank_batch_candidates(language: &str, stats: &[LanguageStat]) -> BatchRanking {
    let (rankable, excluded): (Vec<&LanguageStat>, Vec<&LanguageStat>) = stats
        .iter()
        .partition(|s| s.n_scored > 0 && s.mean_score.is_some());

    // Bounds are computed over ONLY the rankable rows so an excluded "no data"
    // row can't skew the normalization range for the real candidates.
    let rankable_owned: Vec<LanguageStat> = rankable.iter().map(|s| (*s).clone()).collect();
    let bounds = compute_bounds(&rankable_owned);

    let mut candidates: Vec<BatchCandidate> = rankable
        .iter()
        .map(|s| BatchCandidate {
            model_id: s.model_id.clone(),
            profile_id: s.profile_id.clone(),
            n_scored: s.n_scored,
            mean_score: s.mean_score,
            stddev_score: s.stddev_score,
            mean_throughput: s.mean_throughput,
            mean_latency_ms: s.mean_latency_ms,
            malformed_rate: s.malformed_rate,
            error_rate: s.error_rate,
            suitability_score: suitability_score(s, &bounds),
        })
        .collect();

    // Best-first, stable on ties (by model_id) so the ranking is deterministic
    // for tests and for any downstream fallback ordering.
    candidates.sort_by(|a, b| {
        b.suitability_score
            .partial_cmp(&a.suitability_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.model_id.cmp(&b.model_id))
    });

    BatchRanking {
        language: language.to_string(),
        candidates,
        excluded_no_data: excluded.len(),
    }
}

/// The full BSUIT-01 entry point: load this `language`'s stats from `source`
/// and rank them. Async I/O lives here; the ranking itself is the pure
/// [`rank_batch_candidates`].
pub async fn rank_batch_for_language(
    source: &dyn BatchStatsSource,
    language: &str,
) -> Result<BatchRanking, BatchStatsError> {
    let stats = source.load_language_stats(language).await?;
    Ok(rank_batch_candidates(language, &stats))
}

/// A batch-stats data-source failure. Carries no infra detail (host/DSN) — same
/// discipline as [`crate::models::coding_selector::SelectorError`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchStatsError {
    NotConfigured,
    StoreUnavailable,
}

impl std::fmt::Display for BatchStatsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BatchStatsError::NotConfigured => {
                f.write_str("batch-suitability store is not configured")
            }
            BatchStatsError::StoreUnavailable => {
                f.write_str("batch-suitability store is temporarily unavailable")
            }
        }
    }
}

impl std::error::Error for BatchStatsError {}

/// Source of `model_language_stats` rows. Abstracted (mirrors
/// [`crate::models::coding_selector::CodeProfileSource`]'s established pattern)
/// so unit tests use fixtures and only a gated integration test hits the real
/// read-only intake DB.
#[async_trait]
pub trait BatchStatsSource: Send + Sync {
    /// Load every `model_language_stats` row for one `language`, joined to
    /// `model_profiles` for the model name.
    async fn load_language_stats(
        &self,
        language: &str,
    ) -> Result<Vec<LanguageStat>, BatchStatsError>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Production data source (Postgres, read-only)
// ─────────────────────────────────────────────────────────────────────────────

/// Production [`BatchStatsSource`]: reads `model_language_stats` joined to
/// `model_profiles` over a `sqlx::PgPool`. NO literal DSN/host — matches the
/// established pattern in [`crate::models::coding_selector::DbCodeProfileSource`].
/// The pool is built by the caller (e.g. from `terminus_rs::config::
/// intake_database_url()`); this struct only wraps it.
pub struct DbBatchStatsSource {
    pool: sqlx::PgPool,
}

impl DbBatchStatsSource {
    pub fn new(pool: sqlx::PgPool) -> Self {
        DbBatchStatsSource { pool }
    }
}

#[async_trait]
impl BatchStatsSource for DbBatchStatsSource {
    async fn load_language_stats(
        &self,
        language: &str,
    ) -> Result<Vec<LanguageStat>, BatchStatsError> {
        use sqlx::Row;

        // Read-only SELECT over the pre-aggregated view, joined to model_profiles
        // for the human name. `mls.profile_id` is a `model_profiles.id` UUID.
        let rows = sqlx::query(
            "SELECT mp.model_name AS model_id, \
                    mls.profile_id::text AS profile_id, \
                    mls.language, \
                    mls.n_scored, \
                    mls.mean_score::float8      AS mean_score, \
                    mls.stddev_score::float8    AS stddev_score, \
                    mls.mean_throughput::float8 AS mean_throughput, \
                    mls.mean_latency_ms::float8 AS mean_latency_ms, \
                    mls.malformed_rate::float8  AS malformed_rate, \
                    mls.error_rate::float8      AS error_rate \
             FROM model_language_stats mls \
             JOIN model_profiles mp ON mp.id = mls.profile_id \
             WHERE mls.language = $1",
        )
        .bind(language)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| {
            tracing::error!(error = %e, "batch-suitability stats query failed");
            BatchStatsError::StoreUnavailable
        })?;

        Ok(rows
            .into_iter()
            .map(|r| LanguageStat {
                model_id: r.get("model_id"),
                profile_id: r.get("profile_id"),
                language: r.get("language"),
                n_scored: r.get("n_scored"),
                mean_score: r.get("mean_score"),
                stddev_score: r.get("stddev_score"),
                mean_throughput: r.get("mean_throughput"),
                mean_latency_ms: r.get("mean_latency_ms"),
                malformed_rate: r.get::<Option<f64>, _>("malformed_rate").unwrap_or(0.0),
                error_rate: r.get::<Option<f64>, _>("error_rate").unwrap_or(0.0),
            })
            .collect())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test/fixture data source
// ─────────────────────────────────────────────────────────────────────────────

/// Fixed-fixture [`BatchStatsSource`] for unit tests — no Postgres needed.
#[derive(Debug, Clone, Default)]
pub struct StaticBatchStatsSource {
    pub stats: Vec<LanguageStat>,
}

impl StaticBatchStatsSource {
    pub fn new(stats: Vec<LanguageStat>) -> Self {
        StaticBatchStatsSource { stats }
    }
}

#[async_trait]
impl BatchStatsSource for StaticBatchStatsSource {
    async fn load_language_stats(
        &self,
        language: &str,
    ) -> Result<Vec<LanguageStat>, BatchStatsError> {
        Ok(self
            .stats
            .iter()
            .filter(|s| s.language == language)
            .cloned()
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stat(
        model: &str,
        mean: Option<f64>,
        stddev: Option<f64>,
        thru: Option<f64>,
        lat: Option<f64>,
        malformed: f64,
        error: f64,
    ) -> LanguageStat {
        let n = if mean.is_some() { 10 } else { 0 };
        LanguageStat {
            model_id: model.to_string(),
            profile_id: format!("id-{model}"),
            language: "rust".to_string(),
            n_scored: n,
            mean_score: mean,
            stddev_score: stddev,
            mean_throughput: thru,
            mean_latency_ms: lat,
            malformed_rate: malformed,
            error_rate: error,
        }
    }

    #[test]
    fn normalize_basic_and_degenerate() {
        assert!((normalize(5.0, 0.0, 10.0) - 0.5).abs() < 1e-9);
        assert_eq!(normalize(0.0, 0.0, 10.0), 0.0);
        assert_eq!(normalize(10.0, 0.0, 10.0), 1.0);
        // Degenerate range (all identical / single candidate) ⇒ no signal ⇒ 0.
        assert_eq!(normalize(7.0, 7.0, 7.0), 0.0);
        // Out-of-range clamps rather than exceeding [0,1].
        assert_eq!(normalize(20.0, 0.0, 10.0), 1.0);
        assert_eq!(normalize(-5.0, 0.0, 10.0), 0.0);
    }

    #[test]
    fn suitability_score_matches_documented_formula() {
        // A cohort where THIS candidate is the max on every good axis and the
        // min on every bad axis, with perfect malformed/error, hitting every
        // weight at its cap:
        //   norm(mean)=1, norm(stddev)=0, norm(thru)=1, norm(lat)=0,
        //   (1-malformed)=1, error=0
        // score = 0.40*1 - 0.20*0 + 0.10*1 - 0.10*0 + 0.15*1 - 0.05*0
        //       = 0.40 + 0.10 + 0.15 = 0.65
        let bounds = Bounds {
            mean_score: (0.0, 5.0),
            stddev_score: (0.0, 2.0),
            throughput: (0.0, 100.0),
            latency: (0.0, 40000.0),
        };
        let best = stat("m", Some(5.0), Some(0.0), Some(100.0), Some(0.0), 0.0, 0.0);
        assert!((suitability_score(&best, &bounds) - 0.65).abs() < 1e-9);

        // The worst possible in this cohort: min quality/thru, max stddev/lat,
        // fully malformed and fully erroring:
        //   0.40*0 - 0.20*1 + 0.10*0 - 0.10*1 + 0.15*0 - 0.05*1 = -0.35
        let worst = stat("m", Some(0.0), Some(2.0), Some(0.0), Some(40000.0), 1.0, 1.0);
        assert!((suitability_score(&worst, &bounds) + 0.35).abs() < 1e-9);
    }

    #[test]
    fn missing_secondary_field_degrades_to_zero_not_panic() {
        let bounds = Bounds {
            mean_score: (0.0, 5.0),
            stddev_score: (0.0, 2.0),
            throughput: (0.0, 100.0),
            latency: (0.0, 40000.0),
        };
        // mean present but every secondary field NULL: only the mean term and
        // the (1-malformed) reward contribute; nothing panics.
        let s = stat("m", Some(5.0), None, None, None, 0.0, 0.0);
        // 0.40*1 + 0.15*1 = 0.55
        assert!((suitability_score(&s, &bounds) - 0.55).abs() < 1e-9);
    }

    #[test]
    fn higher_quality_ranks_first() {
        let stats = vec![
            stat("low-q", Some(2.0), Some(0.5), Some(50.0), Some(10000.0), 0.0, 0.0),
            stat("high-q", Some(4.5), Some(0.5), Some(50.0), Some(10000.0), 0.0, 0.0),
        ];
        let ranking = rank_batch_candidates("rust", &stats);
        assert_eq!(ranking.candidates[0].model_id, "high-q");
        assert_eq!(ranking.excluded_no_data, 0);
    }

    #[test]
    fn consistency_breaks_a_quality_tie() {
        // Same mean_score; the steadier (lower stddev) model must win because
        // the stddev penalty is the second-heaviest weight.
        let stats = vec![
            stat("swingy", Some(4.0), Some(2.0), Some(50.0), Some(10000.0), 0.0, 0.0),
            stat("steady", Some(4.0), Some(0.1), Some(50.0), Some(10000.0), 0.0, 0.0),
        ];
        let ranking = rank_batch_candidates("rust", &stats);
        assert_eq!(ranking.candidates[0].model_id, "steady");
    }

    #[test]
    fn faster_wins_when_quality_and_consistency_equal() {
        let stats = vec![
            stat("slow", Some(4.0), Some(0.5), Some(20.0), Some(40000.0), 0.0, 0.0),
            stat("fast", Some(4.0), Some(0.5), Some(90.0), Some(5000.0), 0.0, 0.0),
        ];
        let ranking = rank_batch_candidates("rust", &stats);
        assert_eq!(ranking.candidates[0].model_id, "fast");
    }

    #[test]
    fn malformed_and_error_rate_penalize() {
        let stats = vec![
            stat("clean", Some(4.0), Some(0.5), Some(50.0), Some(10000.0), 0.0, 0.0),
            stat("messy", Some(4.0), Some(0.5), Some(50.0), Some(10000.0), 0.5, 0.3),
        ];
        let ranking = rank_batch_candidates("rust", &stats);
        assert_eq!(ranking.candidates[0].model_id, "clean");
    }

    #[test]
    fn no_data_rows_are_excluded_not_ranked_or_crashing() {
        // The n_scored=0 / NULL mean_score shape seen in the live view
        // (error_rate 1.0). Must be dropped and counted, never ranked, never
        // panic the normalization.
        let stats = vec![
            stat("real", Some(3.5), Some(0.5), Some(50.0), Some(10000.0), 0.0, 0.0),
            stat("nodata", None, None, None, None, 0.0, 1.0),
            stat("nodata2", None, None, None, None, 0.0, 1.0),
        ];
        let ranking = rank_batch_candidates("rust", &stats);
        assert_eq!(ranking.candidates.len(), 1);
        assert_eq!(ranking.candidates[0].model_id, "real");
        assert_eq!(ranking.excluded_no_data, 2);
        assert!(ranking.candidates.iter().all(|c| c.model_id != "nodata"));
    }

    #[test]
    fn single_candidate_does_not_panic_on_degenerate_bounds() {
        // Only one candidate ⇒ every min==max ⇒ every normalized term is 0;
        // score is just the two non-normalized reward/penalty terms. No NaN.
        let stats = vec![stat("solo", Some(4.0), Some(0.5), Some(50.0), Some(10000.0), 0.1, 0.05)];
        let ranking = rank_batch_candidates("rust", &stats);
        assert_eq!(ranking.candidates.len(), 1);
        let s = ranking.candidates[0].suitability_score;
        assert!(s.is_finite());
        // 0.15*(1-0.1) - 0.05*0.05 = 0.135 - 0.0025 = 0.1325
        assert!((s - 0.1325).abs() < 1e-9);
    }

    #[test]
    fn ranking_is_deterministic_on_ties() {
        // Identical stats ⇒ identical scores ⇒ tie broken by model_id asc.
        let stats = vec![
            stat("model-b", Some(4.0), Some(0.5), Some(50.0), Some(10000.0), 0.0, 0.0),
            stat("model-a", Some(4.0), Some(0.5), Some(50.0), Some(10000.0), 0.0, 0.0),
        ];
        let ranking = rank_batch_candidates("rust", &stats);
        assert_eq!(ranking.candidates[0].model_id, "model-a");
        assert_eq!(ranking.candidates[1].model_id, "model-b");
    }

    #[test]
    fn all_rows_no_data_yields_empty_ranking_not_error() {
        let stats = vec![
            stat("a", None, None, None, None, 0.0, 1.0),
            stat("b", None, None, None, None, 0.0, 1.0),
        ];
        let ranking = rank_batch_candidates("rust", &stats);
        assert!(ranking.candidates.is_empty());
        assert_eq!(ranking.excluded_no_data, 2);
    }

    #[tokio::test]
    async fn rank_batch_for_language_end_to_end_with_fixture_source() {
        let source = StaticBatchStatsSource::new(vec![
            stat("winner", Some(4.5), Some(0.2), Some(80.0), Some(6000.0), 0.0, 0.0),
            stat("loser", Some(2.0), Some(1.5), Some(30.0), Some(30000.0), 0.2, 0.1),
            // A different language must be filtered out by the source.
            LanguageStat {
                language: "python".to_string(),
                ..stat("other-lang", Some(9.9), Some(0.0), Some(999.0), Some(1.0), 0.0, 0.0)
            },
        ]);
        let ranking = rank_batch_for_language(&source, "rust").await.expect("ranks");
        assert_eq!(ranking.candidates.len(), 2, "python row must be filtered out");
        assert_eq!(ranking.candidates[0].model_id, "winner");
    }

    #[tokio::test]
    #[ignore = "gated integration test — requires a live read-only intake DB \
                connection; run with `cargo test -- --ignored` and \
                INTAKE_DATABASE_URL (or DATABASE_URL) set"]
    async fn live_db_rust_ranking_has_candidates_and_excludes_no_data_rows() {
        let url = std::env::var("INTAKE_DATABASE_URL")
            .or_else(|_| std::env::var("DATABASE_URL"))
            .expect("INTAKE_DATABASE_URL or DATABASE_URL must be set for this ignored test");
        let pool = sqlx::PgPool::connect(&url).await.expect("connect");
        let source = DbBatchStatsSource::new(pool);
        let ranking = rank_batch_for_language(&source, "rust").await.expect("query ok");
        // The live view has real rust rows; the ranking must be non-empty and
        // strictly descending by score, and no ranked candidate may have a NULL
        // mean_score (those are the excluded-no-data rows).
        assert!(!ranking.candidates.is_empty(), "expected live rust candidates");
        for c in &ranking.candidates {
            assert!(c.mean_score.is_some(), "ranked candidate must have a mean_score");
            assert!(c.n_scored > 0, "ranked candidate must have scored runs");
        }
        for w in ranking.candidates.windows(2) {
            assert!(
                w[0].suitability_score >= w[1].suitability_score,
                "candidates must be sorted best-first"
            );
        }
    }
}
