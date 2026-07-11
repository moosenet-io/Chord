//! DOCGEN-04 → Postgres persistence for router-eval sweep results.
//!
//! Mirrors the SAME conventions the other Chord sweep/profiling persistence
//! uses (see `crate::snap::storage`, `crate::models::coding_selector`,
//! `crate::serving::profile`): a `sqlx 0.8` [`PgPool`] sourced from the shared
//! intake DB URL via [`terminus_rs::config::intake_database_url`] (NO literal
//! DSN/host — pii_gate), one row per scored decision, idempotent
//! `CREATE TABLE IF NOT EXISTS` DDL applied in-code by [`migrate`] (no
//! `migrations/` dir), and a `mem_config` tag column so router-eval results
//! never blend the current `dynamic_gtt` memory configuration with
//! legacy/untagged runs — the same non-blending rule
//! `code_profile_runs.mem_config` enforces for the coder sweep
//! (`crate::models::coding_selector`).
//!
//! `model_id` is normalized via [`normalize_model_id`] — trim + lowercase —
//! the same shape other sweep readers expect from `model_profiles.model_name`
//! /`code_profile_runs`-style rows (case/whitespace-insensitive matching so a
//! candidate router's model id always groups with itself across runs).
//!
//! ## Failure policy
//! Best-effort, like SNAP: a DB error must never crash a sweep run. Callers
//! `warn!` and the run proceeds without that row persisted.

use chrono::{DateTime, Utc};
use sqlx::PgPool;
use thiserror::Error;
use uuid::Uuid;

use super::eval::{CandidateEvalSummary, DecisionScore};

/// Errors raised by the router-eval persistence layer. Detail is for logs
/// only — the DB URL/host is never surfaced (pii_gate); connection failures
/// collapse to a generic message.
#[derive(Debug, Error)]
pub enum RouterEvalDbError {
    /// No intake DB URL resolved (neither `INTAKE_DATABASE_URL` nor
    /// `DATABASE_URL`). The caller runs with persistence disabled — never panics.
    #[error("router-eval persistence not configured: {0}")]
    NotConfigured(String),

    /// A connection or query failure. The inner string is a scrubbed,
    /// log-only summary — no DSN/host.
    #[error("router-eval database error: {0}")]
    Database(String),
}

/// Normalize a raw candidate/model identifier to the shape other sweep
/// tables expect: trimmed, lowercased. Keeps `router_eval_runs.model_id`
/// consistent with `code_profile_runs`/`model_profiles.model_name`-style
/// grouping keys used elsewhere in this crate (see `models::coding_selector`,
/// `models::batch_suitability`) so a router-eval result for, e.g., `" Qwen2.5:20B "`
/// groups with a `"qwen2.5:20b"` row from the same candidate on a later run.
pub fn normalize_model_id(raw: &str) -> String {
    raw.trim().to_lowercase()
}

/// Connect a pool to the intake DB — reuses the SAME URL resolver every other
/// Chord DB writer/reader uses (`terminus_rs::config::intake_database_url`).
/// NO literal host/DSN. `NotConfigured` when nothing resolves, so the caller
/// disables persistence rather than guessing at a default.
pub async fn get_pool() -> Result<PgPool, RouterEvalDbError> {
    let url = terminus_rs::config::intake_database_url().ok_or_else(|| {
        RouterEvalDbError::NotConfigured("neither INTAKE_DATABASE_URL nor DATABASE_URL set".into())
    })?;
    PgPool::connect(&url).await.map_err(|e| {
        tracing::error!(error = %e, "router-eval persistence DB connect failed");
        RouterEvalDbError::Database("connect failed".into())
    })
}

/// Apply the router-eval persistence schema. Idempotent: safe to call on
/// every startup. Creates `router_eval_runs` (one row per scored decision)
/// with `CREATE TABLE/INDEX IF NOT EXISTS`. No migration files.
pub async fn migrate(pool: &PgPool) -> Result<(), RouterEvalDbError> {
    let stmts: &[&str] = &[
        "CREATE TABLE IF NOT EXISTS router_eval_runs ( \
            id                     BIGSERIAL PRIMARY KEY, \
            run_id                 UUID        NOT NULL, \
            candidate_name         TEXT        NOT NULL, \
            model_id               TEXT        NOT NULL, \
            mem_config             TEXT, \
            request_id             TEXT        NOT NULL, \
            destination            TEXT        NOT NULL, \
            expected_destination   TEXT        NOT NULL, \
            appropriateness_score  DOUBLE PRECISION NOT NULL, \
            doc_quality_score      DOUBLE PRECISION NOT NULL, \
            cost_score             DOUBLE PRECISION NOT NULL, \
            latency_score          DOUBLE PRECISION NOT NULL, \
            composite_score        DOUBLE PRECISION NOT NULL, \
            flagged                BOOLEAN     NOT NULL, \
            flag_reason            TEXT, \
            evaluated_at           TIMESTAMPTZ NOT NULL, \
            created_at             TIMESTAMPTZ NOT NULL DEFAULT now() \
        )",
        "CREATE INDEX IF NOT EXISTS idx_router_eval_run \
            ON router_eval_runs (run_id)",
        // Mirrors code_profile_runs' non-blending grouping key: candidate,
        // model, and mem_config together — never blended.
        "CREATE INDEX IF NOT EXISTS idx_router_eval_candidate_model_mem \
            ON router_eval_runs (candidate_name, model_id, mem_config)",
        "CREATE INDEX IF NOT EXISTS idx_router_eval_evaluated_at \
            ON router_eval_runs (evaluated_at)",
    ];
    for stmt in stmts {
        sqlx::query(stmt)
            .execute(pool)
            .await
            .map_err(|e| RouterEvalDbError::Database(format!("migrate: {e}")))?;
    }
    Ok(())
}

fn destination_str(d: super::policy::RoutingDestination) -> &'static str {
    match d {
        super::policy::RoutingDestination::LocalHighContext => "local_high_context",
        super::policy::RoutingDestination::LocalCheap => "local_cheap",
        super::policy::RoutingDestination::CloudFrontierFree => "cloud_frontier_free",
    }
}

/// Insert one scored decision under `run_id`/`candidate_name`/`model_id`,
/// tagged with `mem_config` (the current memory configuration under test —
/// e.g. `"dynamic_gtt"` — or `None` for an untagged/legacy run, mirroring
/// `code_profile_runs.mem_config`'s nullable convention).
///
/// `model_id` is normalized via [`normalize_model_id`] before the insert —
/// callers do not need to pre-normalize.
pub async fn insert_eval_run(
    pool: &PgPool,
    run_id: Uuid,
    candidate_name: &str,
    model_id: &str,
    mem_config: Option<&str>,
    score: &DecisionScore,
    evaluated_at: DateTime<Utc>,
) -> Result<(), RouterEvalDbError> {
    let normalized_model_id = normalize_model_id(model_id);
    sqlx::query(
        "INSERT INTO router_eval_runs \
            (run_id, candidate_name, model_id, mem_config, request_id, destination, \
             expected_destination, appropriateness_score, doc_quality_score, cost_score, \
             latency_score, composite_score, flagged, flag_reason, evaluated_at) \
         VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15)",
    )
    .bind(run_id)
    .bind(candidate_name)
    .bind(&normalized_model_id)
    .bind(mem_config)
    .bind(score.request_id)
    .bind(destination_str(score.destination))
    .bind(destination_str(score.expected_destination))
    .bind(score.appropriateness_score)
    .bind(score.doc_quality_score)
    .bind(score.cost_score)
    .bind(score.latency_score)
    .bind(score.composite_score)
    .bind(score.flag_reason.is_some())
    .bind(score.flag_reason.as_deref())
    .bind(evaluated_at)
    .execute(pool)
    .await
    .map_err(|e| RouterEvalDbError::Database(format!("insert_eval_run: {e}")))?;
    Ok(())
}

/// Insert every score in `summary` under one `run_id`, tagged with the same
/// `model_id`/`mem_config`. Best-effort per row: a single row's failure is
/// logged and does not abort the remaining inserts, mirroring SNAP's "a DB
/// error must never fail or slow the caller" posture — one bad row should
/// not lose the rest of a sweep's results.
pub async fn insert_candidate_summary(
    pool: &PgPool,
    run_id: Uuid,
    model_id: &str,
    mem_config: Option<&str>,
    summary: &CandidateEvalSummary,
    evaluated_at: DateTime<Utc>,
) -> usize {
    let mut inserted = 0usize;
    for score in &summary.scores {
        match insert_eval_run(
            pool,
            run_id,
            &summary.candidate_name,
            model_id,
            mem_config,
            score,
            evaluated_at,
        )
        .await
        {
            Ok(()) => inserted += 1,
            Err(e) => {
                tracing::warn!(
                    candidate = %summary.candidate_name,
                    request_id = %score.request_id,
                    error = %e,
                    "router-eval: failed to persist one decision score"
                );
            }
        }
    }
    inserted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_model_id_trims_and_lowercases() {
        assert_eq!(normalize_model_id("  Qwen2.5:20B  "), "qwen2.5:20b");
        assert_eq!(normalize_model_id("already-normal"), "already-normal");
    }

    #[test]
    fn destination_str_is_stable_for_all_variants() {
        use super::super::policy::RoutingDestination;
        assert_eq!(destination_str(RoutingDestination::LocalCheap), "local_cheap");
        assert_eq!(
            destination_str(RoutingDestination::LocalHighContext),
            "local_high_context"
        );
        assert_eq!(
            destination_str(RoutingDestination::CloudFrontierFree),
            "cloud_frontier_free"
        );
    }

    #[test]
    fn test_no_hardcoded_infrastructure_values() {
        let src = include_str!("eval_storage.rs");
        let private_ip_prefix = ["192", "168", "."].concat();
        let org_domain = ["moosenet", ".online"].concat();
        assert!(!src.contains(&private_ip_prefix));
        assert!(!src.contains(&org_domain));
    }

    // ── DB-backed tests: require INTAKE_DATABASE_URL/DATABASE_URL; ignored by
    // default, mirroring the convention in models::coding_selector /
    // models::batch_suitability (`#[ignore]` + a doc string naming the env
    // var). Not run in this sweep's mocked-only test gate.
    #[tokio::test]
    #[ignore = "requires INTAKE_DATABASE_URL (or DATABASE_URL) set"]
    async fn migrate_and_insert_round_trip() {
        use sqlx::Row;
        let url = std::env::var("INTAKE_DATABASE_URL")
            .or_else(|_| std::env::var("DATABASE_URL"))
            .expect("DB url env var set");
        let pool = sqlx::PgPool::connect(&url).await.expect("connect");
        migrate(&pool).await.expect("migrate");

        let request = super::super::eval::DocGenEvalRequest {
            id: "round-trip-test",
            prompt: "x",
            estimated_tokens: 100,
            expected_destination: super::super::policy::RoutingDestination::LocalCheap,
        };
        let decision = crate::router::slm_router::RoutingDecision {
            destination: super::super::policy::RoutingDestination::LocalCheap,
            model: "test-model".into(),
            reason: "test".into(),
            fallback_from: None,
        };
        let score = super::super::eval::score_decision(&request, &decision, 0.9);

        let run_id = Uuid::new_v4();
        insert_eval_run(&pool, run_id, "test-candidate", "Test-Model", Some("dynamic_gtt"), &score, Utc::now())
            .await
            .expect("insert");

        let row = sqlx::query("SELECT model_id, mem_config FROM router_eval_runs WHERE run_id = $1")
            .bind(run_id)
            .fetch_one(&pool)
            .await
            .expect("fetch");
        let model_id: String = row.get("model_id");
        let mem_config: Option<String> = row.get("mem_config");
        assert_eq!(model_id, "test-model");
        assert_eq!(mem_config.as_deref(), Some("dynamic_gtt"));
    }
}
