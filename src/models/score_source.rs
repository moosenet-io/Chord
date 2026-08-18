//! CHRD-MINTSEL: MINT operational-**score** reader for the unified selector.
//!
//! The unified [`resolve`](crate::models::selector::resolve) ranks candidates by
//! [`ModelCandidate::score`](crate::models::selector::ModelCandidate) — but nothing ever
//! populated that score for non-code tasks, so every non-code resolution silently fell
//! through to the lexicographic name tiebreak. This module closes the loop the CH-SEL-01 doc
//! comment always promised: it reads MINT's measured `assistant_dimension_score` rows (the
//! same intake Postgres the coding selector already uses via
//! [`terminus_rs::config::intake_database_url`]) and turns them into a single
//! `[0,1]`-normalized, higher-is-better operational score per `(model, task)`.
//!
//! It mirrors [`crate::models::coding_selector::DbCodeProfileSource`]'s established pattern:
//! a trait ([`ScoreSource`]) so unit tests inject fixtures and only a gated integration test
//! hits the live DB; a production [`DbScoreSource`] over a `sqlx::PgPool` with NO literal
//! DSN/host; and a [`StaticScoreSource`] fixture.
//!
//! ## Task → (category, dimension, metric, direction) mapping
//! Verified against the live intake schema (`assistant_dimension_score`) and the Terminus
//! `intake::newcats` / `intake::assistant` category constants:
//!
//! | Task | task_category | dimension | primary metric | direction |
//! |---|---|---|---|---|
//! | Chat / Reasoning | `assistant` | `personality_prompted` | *(avg all metrics)* | 1–5 → `/5` |
//! | Code | *(delegated to the coding aggregate — see [`DbScoreSource::code_score`])* | | | best combined_score |
//! | Embed | *(never scored — pinned out of the dynamic selector)* | | | — |
//! | Rerank | `reranking` | `rerank_relevance` | `ndcg_uplift` | higher=better |
//! | VisionQa | `image_parsing` | `vision_description` | `caption_similarity` | higher=better |
//! | Ocr / DocParse | `document_parsing` | `ocr_extraction` | `field_accuracy` | higher=better |
//! | ImageGen | `image_generation` | `text_to_image` | `generation_success` | higher=better |
//! | Tts | `tts` | `tts_intelligibility` | `loopback_accuracy` | higher=better |
//! | Stt | `voice_transcription` | `asr_transcription` | `word_error_rate` | lower=better → `1−wer` |
//! | ToolRoute | `tool_routing` | `tool_routing` | `correct_tool_at_1` | higher=better |
//! | Diffusion | *(no profiling category yet)* | | | — |
//!
//! An unprofiled model (or a task with no category) yields `None` — the selector treats that
//! as "no score" and ranks it LAST (fail-open), never a panic and never a fabricated number.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::models::coding_selector::{combined_score, CodeAggregate};
use crate::models::selector::Task;

/// Shared, hot-swappable score source for the `AppState`: `None` ⇒ the intake DB isn't
/// configured/reachable, so enrichment is skipped and every candidate keeps `score: None`
/// (fail-open — same discipline as [`crate::coding_proxy::SharedCodingProfileSource`]).
pub type SharedScoreSource = Arc<tokio::sync::Mutex<Option<Arc<dyn ScoreSource>>>>;

/// In-memory cache TTL. Intake tables update on the sweep cadence (not per-request), so a
/// short TTL avoids a DB round-trip per candidate per resolve without ever serving a score
/// more than this stale.
const SCORE_CACHE_TTL: Duration = Duration::from_secs(60);

/// How a raw metric value is mapped onto the selector's `[0,1]`, higher-is-better score.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Norm {
    /// Already a `[0,1]` fraction, higher=better: clamp defensively.
    Direct,
    /// A 1–5 graded score (the assistant-persona scale): divide by 5, then clamp.
    ScaleByFive,
    /// A `[0,1]` error rate, lower=better: invert to `1 − v`.
    InvertUnit,
}

fn normalize(norm: Norm, v: f64) -> f64 {
    match norm {
        Norm::Direct => v.clamp(0.0, 1.0),
        Norm::ScaleByFive => (v / 5.0).clamp(0.0, 1.0),
        Norm::InvertUnit => 1.0 - v.clamp(0.0, 1.0),
    }
}

/// The `assistant_dimension_score` coordinates a non-code task's primary quality signal
/// lives under. `metric == None` ⇒ average across every metric of the dimension (used for the
/// assistant-persona aggregate that has many sibling metrics).
struct Mapping {
    category: &'static str,
    dimension: &'static str,
    metric: Option<&'static str>,
    norm: Norm,
}

/// Map a [`Task`] to its MINT measurement coordinates. `Code` (delegated to the coding
/// aggregate), `Embed` (pinned out of the dynamic selector), and `Diffusion` (no category
/// yet) return `None` here.
fn mint_mapping(task: Task) -> Option<Mapping> {
    let m = match task {
        Task::Chat | Task::Reasoning => Mapping {
            category: "assistant",
            dimension: "personality_prompted",
            metric: None,
            norm: Norm::ScaleByFive,
        },
        Task::Rerank => Mapping {
            category: "reranking",
            dimension: "rerank_relevance",
            metric: Some("ndcg_uplift"),
            norm: Norm::Direct,
        },
        Task::VisionQa => Mapping {
            category: "image_parsing",
            dimension: "vision_description",
            metric: Some("caption_similarity"),
            norm: Norm::Direct,
        },
        Task::Ocr | Task::DocParse => Mapping {
            category: "document_parsing",
            dimension: "ocr_extraction",
            metric: Some("field_accuracy"),
            norm: Norm::Direct,
        },
        Task::ImageGen => Mapping {
            category: "image_generation",
            dimension: "text_to_image",
            metric: Some("generation_success"),
            norm: Norm::Direct,
        },
        Task::Tts => Mapping {
            category: "tts",
            dimension: "tts_intelligibility",
            metric: Some("loopback_accuracy"),
            norm: Norm::Direct,
        },
        Task::Stt => Mapping {
            category: "voice_transcription",
            dimension: "asr_transcription",
            metric: Some("word_error_rate"),
            norm: Norm::InvertUnit,
        },
        Task::ToolRoute => Mapping {
            category: "tool_routing",
            dimension: "tool_routing",
            metric: Some("correct_tool_at_1"),
            norm: Norm::Direct,
        },
        Task::Code | Task::Embed | Task::Diffusion => return None,
    };
    Some(m)
}

/// A source of MINT operational scores for the unified selector. Abstracted (mirrors
/// [`crate::models::coding_selector::CodeProfileSource`]) so unit tests use a fixture and
/// only a gated integration test touches the live intake DB.
#[async_trait]
pub trait ScoreSource: Send + Sync {
    /// The MINT operational score for `model` on `task`, normalized to `[0,1]` with
    /// higher=better, or `None` when the model is unprofiled for that task (fail-open —
    /// the selector ranks a `None` score last, never a panic).
    async fn score_for(&self, model: &str, task: Task) -> Option<f64>;
}

/// Production [`ScoreSource`]: reads `assistant_dimension_score` (and, for `Code`,
/// `code_profile_runs`/`model_profiles`) over a `sqlx::PgPool`. NO literal DSN/host — the
/// pool is built by the caller from [`terminus_rs::config::intake_database_url`], exactly
/// like [`crate::models::coding_selector::DbCodeProfileSource`]. Carries a short TTL cache
/// keyed by `(model, task)`.
pub struct DbScoreSource {
    pool: sqlx::PgPool,
    cache: tokio::sync::Mutex<HashMap<(String, Task), (Instant, Option<f64>)>>,
}

impl DbScoreSource {
    pub fn new(pool: sqlx::PgPool) -> Self {
        DbScoreSource {
            pool,
            cache: tokio::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Non-code path: average `value` over the model's rows for the mapping's
    /// `(task_category, dimension[, metric])`, then normalize by direction. Read-only SELECT.
    async fn dimension_score(&self, model: &str, m: &Mapping) -> Option<f64> {
        use sqlx::Row;
        let row = if let Some(metric) = m.metric {
            sqlx::query(
                "SELECT avg(value)::float8 AS v FROM assistant_dimension_score \
                 WHERE model_id = $1 AND task_category = $2 AND dimension = $3 AND metric = $4",
            )
            .bind(model)
            .bind(m.category)
            .bind(m.dimension)
            .bind(metric)
            .fetch_optional(&self.pool)
            .await
        } else {
            sqlx::query(
                "SELECT avg(value)::float8 AS v FROM assistant_dimension_score \
                 WHERE model_id = $1 AND task_category = $2 AND dimension = $3",
            )
            .bind(model)
            .bind(m.category)
            .bind(m.dimension)
            .fetch_optional(&self.pool)
            .await
        };
        let row = match row {
            Ok(r) => r?,
            Err(e) => {
                tracing::warn!(error = %e, "MINT score dimension query failed");
                return None;
            }
        };
        let v: Option<f64> = row.try_get("v").ok().flatten();
        v.map(|x| normalize(m.norm, x))
    }

    /// Code path: DELEGATE to the coding aggregate so both selectors agree on "how good a
    /// coder is this model". Reuses [`combined_score`] over the model's per-`(language,
    /// backend_tag, mem_config)` aggregates (never blended — each row scored on its own,
    /// mirroring `coding_selector`) and returns the model's BEST such score.
    async fn code_score(&self, model: &str) -> Option<f64> {
        use sqlx::Row;
        // The measured-count columns and the per-metric `finalized` discipline
        // are NOT optional decoration here: `combined_score` shrinks each rate
        // toward the 0.5 prior by the count that backs it, so a count must be
        // drawn from exactly the same population as the average it qualifies.
        // Passing 0 for these (the only other way to satisfy the struct) would
        // claim "no evidence" for every model and collapse every MINT code
        // score toward the prior. See the long rationale on the identical
        // aggregate in `coding_selector::DbCodeProfileSource::load_aggregates`:
        // `count(<col>)` counts NON-NULL values only, and `finalized` gates the
        // Phase-2-adjusted score term ONLY — `compiles`/`tests_pass` are
        // Phase-1 facts and every row counts for them.
        let rows = sqlx::query(
            "SELECT cpr.backend_tag, cpr.mem_config, \
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
             WHERE mp.model_name = $1 \
             GROUP BY cpr.language, cpr.backend_tag, cpr.mem_config",
        )
        .bind(model)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| tracing::warn!(error = %e, "MINT code score query failed"))
        .ok()?;

        let mut best: Option<f64> = None;
        for r in rows {
            let agg = CodeAggregate {
                model_id: model.to_string(),
                backend_tag: r.get("backend_tag"),
                mem_config: r.get("mem_config"),
                run_count: r.get("run_count"),
                avg_effective_score: r.get("avg_effective_score"),
                compile_pass_rate: r.get("compile_pass_rate"),
                test_pass_rate: r.get("test_pass_rate"),
                measured_compile_count: r.get("measured_compile_count"),
                measured_test_count: r.get("measured_test_count"),
                measured_score_count: r.get("measured_score_count"),
            };
            let s = combined_score(&agg);
            best = Some(best.map_or(s, |b: f64| b.max(s)));
        }
        best
    }

    async fn compute(&self, model: &str, task: Task) -> Option<f64> {
        match task {
            Task::Code => self.code_score(model).await,
            _ => {
                let m = mint_mapping(task)?;
                self.dimension_score(model, &m).await
            }
        }
    }
}

#[async_trait]
impl ScoreSource for DbScoreSource {
    async fn score_for(&self, model: &str, task: Task) -> Option<f64> {
        let key = (model.to_string(), task);
        {
            let cache = self.cache.lock().await;
            if let Some((at, v)) = cache.get(&key) {
                if at.elapsed() < SCORE_CACHE_TTL {
                    return *v;
                }
            }
        }
        let v = self.compute(model, task).await;
        self.cache.lock().await.insert(key, (Instant::now(), v));
        v
    }
}

/// Fixed-fixture [`ScoreSource`] for unit tests — no Postgres needed. Scores are stored
/// pre-normalized (`[0,1]`), keyed by `(model, task)`.
#[derive(Debug, Clone, Default)]
pub struct StaticScoreSource {
    scores: HashMap<(String, Task), f64>,
}

impl StaticScoreSource {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_score(mut self, model: &str, task: Task, score: f64) -> Self {
        self.scores.insert((model.to_string(), task), score);
        self
    }
}

#[async_trait]
impl ScoreSource for StaticScoreSource {
    async fn score_for(&self, model: &str, task: Task) -> Option<f64> {
        self.scores.get(&(model.to_string(), task)).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_directions() {
        // Direct: clamp only.
        assert!((normalize(Norm::Direct, 0.75) - 0.75).abs() < 1e-9);
        assert_eq!(normalize(Norm::Direct, 1.4), 1.0);
        assert_eq!(normalize(Norm::Direct, -0.2), 0.0);
        // ScaleByFive: 1-5 grade → [0.2, 1.0].
        assert!((normalize(Norm::ScaleByFive, 5.0) - 1.0).abs() < 1e-9);
        assert!((normalize(Norm::ScaleByFive, 1.0) - 0.2).abs() < 1e-9);
        // InvertUnit: WER lower=better.
        assert!((normalize(Norm::InvertUnit, 0.0) - 1.0).abs() < 1e-9);
        assert!((normalize(Norm::InvertUnit, 0.6) - 0.4).abs() < 1e-9);
    }

    #[test]
    fn every_non_pinned_task_has_a_mapping_or_is_delegated() {
        // Embed/Code/Diffusion are intentionally None (pinned / delegated / unprofiled);
        // every other task MUST resolve to a concrete MINT coordinate.
        for t in [
            Task::Chat,
            Task::Reasoning,
            Task::Rerank,
            Task::VisionQa,
            Task::Ocr,
            Task::DocParse,
            Task::ImageGen,
            Task::Tts,
            Task::Stt,
            Task::ToolRoute,
        ] {
            assert!(mint_mapping(t).is_some(), "task {t:?} must map to a MINT metric");
        }
        assert!(mint_mapping(Task::Embed).is_none());
        assert!(mint_mapping(Task::Code).is_none()); // delegated to code_score, not a dim mapping
        assert!(mint_mapping(Task::Diffusion).is_none());
    }

    #[test]
    fn stt_maps_to_inverted_wer() {
        let m = mint_mapping(Task::Stt).unwrap();
        assert_eq!(m.category, "voice_transcription");
        assert_eq!(m.metric, Some("word_error_rate"));
        assert_eq!(m.norm, Norm::InvertUnit);
    }

    #[tokio::test]
    async fn static_source_returns_seeded_and_none() {
        let src = StaticScoreSource::new()
            .with_score("m-hi", Task::VisionQa, 0.9)
            .with_score("m-lo", Task::VisionQa, 0.2);
        assert_eq!(src.score_for("m-hi", Task::VisionQa).await, Some(0.9));
        assert_eq!(src.score_for("m-lo", Task::VisionQa).await, Some(0.2));
        // Unprofiled model/task ⇒ None (fail-open).
        assert_eq!(src.score_for("m-hi", Task::Tts).await, None);
        assert_eq!(src.score_for("unknown", Task::VisionQa).await, None);
    }

    #[tokio::test]
    #[ignore = "gated integration test — requires a live read-only intake DB; run with \
                `cargo test -- --ignored` and INTAKE_DATABASE_URL (or DATABASE_URL) set"]
    async fn live_db_tool_routing_scores_are_normalized() {
        let url = std::env::var("INTAKE_DATABASE_URL")
            .or_else(|_| std::env::var("DATABASE_URL"))
            .expect("INTAKE_DATABASE_URL or DATABASE_URL must be set");
        let pool = sqlx::PgPool::connect(&url).await.expect("connect");
        let source = DbScoreSource::new(pool);
        // A model known to have tool_routing rows in the live sweep.
        let s = source.score_for("qwen2.5:32b", Task::ToolRoute).await;
        if let Some(v) = s {
            assert!((0.0..=1.0).contains(&v), "score must be normalized to [0,1], got {v}");
        }
    }
}
