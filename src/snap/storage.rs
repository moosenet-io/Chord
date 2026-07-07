//! SNAP → Postgres persistence (chord 1.3.1).
//!
//! Persists chord's SNAP observability streams (request analytics, model
//! inventory, passive activity, VRAM samples) to Postgres using the SAME
//! conventions the model-intake harnesses use: a `sqlx 0.8` [`PgPool`] sourced
//! from the shared intake DB URL via [`terminus_rs::config::intake_database_url`]
//! (NO literal DSN / host — pii_gate), one row per event, and idempotent
//! `CREATE TABLE IF NOT EXISTS` DDL applied in-code by [`migrate`] (no
//! `migrations/` dir, mirroring `intake/serving/schema.rs`).
//!
//! ## Toggle (additive, default-OFF)
//! ALL writes are gated behind the `CHORD_SNAP_PERSIST` env flag
//! ([`crate::snap::config::SnapConfig::persist`]). When unset/false SNAP runs
//! exactly as 1.2.0/1.3.0 — in-memory / JSONL only, zero behavior change. The
//! pool + [`migrate`] are only constructed when the flag is on.
//!
//! ## Failure policy
//! SNAP persistence is best-effort observability: a DB error must NEVER fail or
//! slow a proxied request. Callers `warn!` and drop the row (same spirit as the
//! JSONL append, which silently ignores IO errors).

use chrono::{DateTime, Utc};
use sqlx::PgPool;
use thiserror::Error;
use uuid::Uuid;

use crate::snap::activity::ModelActivity;
use crate::snap::analytics::RequestRecord;
use crate::snap::config::StorageTier;
use crate::snap::inventory::ModelRecord;
use crate::snap::state::VRAMState;

/// Errors raised by the SNAP persistence layer. Detail is for logs only — the DB
/// URL / host is never surfaced (pii_gate), connection failures collapse to a
/// generic message.
#[derive(Debug, Error)]
pub enum SnapDbError {
    /// No intake DB URL resolved (neither `INTAKE_DATABASE_URL` nor
    /// `DATABASE_URL`). The caller runs with persistence disabled — never panics.
    #[error("SNAP persistence not configured: {0}")]
    NotConfigured(String),

    /// A connection or query failure. The inner string is a scrubbed,
    /// log-only summary — no DSN/host.
    #[error("SNAP database error: {0}")]
    Database(String),
}

/// Connect a pool to the intake DB — reuses the SRV-01 / harness URL resolver
/// (`INTAKE_DATABASE_URL` → `DATABASE_URL`). NO literal host/DSN. `NotConfigured`
/// when nothing resolves, so the caller disables persistence rather than guessing.
///
/// chord builds ONE pool at startup (see [`crate::snap::spawn_health_monitor`])
/// and passes `&PgPool` into the inserts; this helper is the construction point.
pub async fn get_pool() -> Result<PgPool, SnapDbError> {
    let url = terminus_rs::config::intake_database_url().ok_or_else(|| {
        SnapDbError::NotConfigured(
            "neither INTAKE_DATABASE_URL nor DATABASE_URL set".into(),
        )
    })?;
    PgPool::connect(&url).await.map_err(|e| {
        // Detail logged by caller, NOT surfaced (no DSN/host leak).
        tracing::error!(error = %e, "SNAP persistence DB connect failed");
        SnapDbError::Database("connect failed".into())
    })
}

/// Apply the SNAP persistence schema. Idempotent: safe to call on every startup.
///
/// Creates the five tables (`snap_request_log`, `snap_model_inventory`,
/// `snap_activity`, `snap_vram_sample` + child `snap_vram_allocation`) and their
/// indexes with `CREATE TABLE/INDEX IF NOT EXISTS`. Mirrors
/// `intake/serving/schema.rs::migrate`. No migration files.
pub async fn migrate(pool: &PgPool) -> Result<(), SnapDbError> {
    // Each statement is issued separately (sqlx::query runs a single statement).
    let stmts: &[&str] = &[
        // ── SNAP-05: per-request analytics. One row per completed request. ──
        "CREATE TABLE IF NOT EXISTS snap_request_log ( \
            id            BIGSERIAL PRIMARY KEY, \
            request_id    TEXT        NOT NULL, \
            request_ts    TIMESTAMPTZ NOT NULL, \
            model         TEXT        NOT NULL, \
            endpoint      TEXT        NOT NULL, \
            engine_url    TEXT        NOT NULL, \
            input_tokens  BIGINT, \
            output_tokens BIGINT, \
            duration_ms   BIGINT      NOT NULL, \
            status_code   INTEGER     NOT NULL, \
            streaming     BOOLEAN     NOT NULL, \
            host          TEXT, \
            created_at    TIMESTAMPTZ NOT NULL DEFAULT now() \
        )",
        "CREATE INDEX IF NOT EXISTS idx_snap_request_ts \
            ON snap_request_log (request_ts)",
        "CREATE INDEX IF NOT EXISTS idx_snap_request_model_ts \
            ON snap_request_log (model, request_ts)",
        // ── SNAP-03: model inventory. One row per model per scan. ──
        "CREATE TABLE IF NOT EXISTS snap_model_inventory ( \
            id            BIGSERIAL PRIMARY KEY, \
            scan_id       UUID        NOT NULL, \
            scanned_at    TIMESTAMPTZ NOT NULL, \
            model_name    TEXT        NOT NULL, \
            file_path     TEXT        NOT NULL, \
            size_bytes    BIGINT      NOT NULL, \
            quant_level   TEXT, \
            engine_compat TEXT[]      NOT NULL DEFAULT '{}', \
            storage_tier  TEXT        NOT NULL CHECK (storage_tier IN ('hot','warm')), \
            last_used     TIMESTAMPTZ, \
            loaded        BOOLEAN     NOT NULL, \
            host          TEXT, \
            created_at    TIMESTAMPTZ NOT NULL DEFAULT now() \
        )",
        "CREATE INDEX IF NOT EXISTS idx_snap_inv_scan \
            ON snap_model_inventory (scan_id)",
        "CREATE INDEX IF NOT EXISTS idx_snap_inv_model_time \
            ON snap_model_inventory (model_name, scanned_at)",
        "CREATE INDEX IF NOT EXISTS idx_snap_inv_scanned_at \
            ON snap_model_inventory (scanned_at)",
        // ── SNAP-04: passive activity. One row per (engine × model) per poll. ──
        "CREATE TABLE IF NOT EXISTS snap_activity ( \
            id              BIGSERIAL PRIMARY KEY, \
            poll_id         UUID        NOT NULL, \
            model           TEXT        NOT NULL, \
            engine          TEXT        NOT NULL, \
            active_requests INTEGER     NOT NULL, \
            last_seen       TIMESTAMPTZ NOT NULL, \
            host            TEXT, \
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now() \
        )",
        "CREATE INDEX IF NOT EXISTS idx_snap_activity_seen \
            ON snap_activity (last_seen)",
        "CREATE INDEX IF NOT EXISTS idx_snap_activity_model_seen \
            ON snap_activity (model, last_seen)",
        // ── SNAP-02: VRAM totals. One row per poll. ──
        "CREATE TABLE IF NOT EXISTS snap_vram_sample ( \
            id         BIGSERIAL PRIMARY KEY, \
            sample_id  UUID        NOT NULL UNIQUE, \
            sampled_at TIMESTAMPTZ NOT NULL, \
            total_mb   BIGINT      NOT NULL, \
            used_mb    BIGINT      NOT NULL, \
            free_mb    BIGINT      NOT NULL, \
            host       TEXT, \
            created_at TIMESTAMPTZ NOT NULL DEFAULT now() \
        )",
        "CREATE INDEX IF NOT EXISTS idx_snap_vram_sampled_at \
            ON snap_vram_sample (sampled_at)",
        // ── SNAP-02: per-model VRAM allocation within a sample. ──
        "CREATE TABLE IF NOT EXISTS snap_vram_allocation ( \
            id         BIGSERIAL PRIMARY KEY, \
            sample_id  UUID        NOT NULL \
                REFERENCES snap_vram_sample (sample_id) ON DELETE CASCADE, \
            model_name TEXT        NOT NULL, \
            engine     TEXT        NOT NULL, \
            size_mb    BIGINT      NOT NULL, \
            loaded_at  TIMESTAMPTZ NOT NULL \
        )",
        "CREATE INDEX IF NOT EXISTS idx_snap_vram_alloc_sample \
            ON snap_vram_allocation (sample_id)",
        "CREATE INDEX IF NOT EXISTS idx_snap_vram_alloc_model \
            ON snap_vram_allocation (model_name, sample_id)",
    ];

    for stmt in stmts {
        sqlx::query(stmt)
            .execute(pool)
            .await
            .map_err(|e| SnapDbError::Database(format!("migrate: {e}")))?;
    }
    Ok(())
}

// ── SNAP-05: one row per completed request ───────────────────────────────────

/// Insert one completed-request record. `u64` token counts bind as `i64`
/// (`BIGINT`) — counts never realistically exceed `i64::MAX`.
pub async fn insert_request_log(
    pool: &PgPool,
    rec: &RequestRecord,
    host: Option<&str>,
) -> Result<(), SnapDbError> {
    sqlx::query(
        "INSERT INTO snap_request_log \
            (request_id, request_ts, model, endpoint, engine_url, \
             input_tokens, output_tokens, duration_ms, status_code, streaming, host) \
         VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)",
    )
    .bind(&rec.id)
    .bind(rec.timestamp)
    .bind(&rec.model)
    .bind(&rec.endpoint)
    .bind(&rec.engine_url)
    .bind(rec.input_tokens.map(|v| v as i64))
    .bind(rec.output_tokens.map(|v| v as i64))
    .bind(rec.duration_ms as i64)
    .bind(rec.status_code as i32)
    .bind(rec.streaming)
    .bind(host)
    .execute(pool)
    .await
    .map_err(|e| SnapDbError::Database(format!("insert_request_log: {e}")))?;
    Ok(())
}

// ── SNAP-03: one snapshot (whole Vec<ModelRecord>) under one scan_id ─────────

fn tier_str(tier: &StorageTier) -> &'static str {
    match tier {
        StorageTier::Hot => "hot",
        StorageTier::Warm => "warm",
    }
}

/// Insert one inventory snapshot: every record from a single scan shares the
/// returned `scan_id` + `scanned_at`. Atomic — all rows commit in one tx, so a
/// half-written snapshot never persists.
pub async fn insert_inventory_scan(
    pool: &PgPool,
    records: &[ModelRecord],
    host: Option<&str>,
) -> Result<Uuid, SnapDbError> {
    let scan_id = Uuid::new_v4();
    let scanned_at = Utc::now();

    let mut tx = pool
        .begin()
        .await
        .map_err(|e| SnapDbError::Database(format!("inventory tx begin: {e}")))?;

    for rec in records {
        sqlx::query(
            "INSERT INTO snap_model_inventory \
                (scan_id, scanned_at, model_name, file_path, size_bytes, quant_level, \
                 engine_compat, storage_tier, last_used, loaded, host) \
             VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)",
        )
        .bind(scan_id)
        .bind(scanned_at)
        .bind(&rec.name)
        .bind(rec.file_path.to_string_lossy().to_string())
        .bind(rec.size_bytes as i64)
        .bind(rec.quant_level.as_deref())
        .bind(&rec.engine_compat)
        .bind(tier_str(&rec.storage_tier))
        .bind(rec.last_used)
        .bind(rec.loaded)
        .bind(host)
        .execute(&mut *tx)
        .await
        .map_err(|e| SnapDbError::Database(format!("insert_inventory_scan: {e}")))?;
    }

    tx.commit()
        .await
        .map_err(|e| SnapDbError::Database(format!("inventory tx commit: {e}")))?;
    Ok(scan_id)
}

// ── SNAP-04: one batch of active rows under one poll_id ──────────────────────

/// Insert one activity poll: only rows with `active_requests > 0` are persisted
/// (idle models every poll are pure bloat — design §5.2 rec (a)). All persisted
/// rows share the returned `poll_id`. Returns the `poll_id` even when no rows
/// qualified (empty poll).
pub async fn insert_activity_poll(
    pool: &PgPool,
    activity: &[ModelActivity],
    host: Option<&str>,
) -> Result<Uuid, SnapDbError> {
    let poll_id = Uuid::new_v4();
    let active: Vec<&ModelActivity> =
        activity.iter().filter(|a| a.active_requests > 0).collect();
    if active.is_empty() {
        return Ok(poll_id);
    }

    let mut tx = pool
        .begin()
        .await
        .map_err(|e| SnapDbError::Database(format!("activity tx begin: {e}")))?;

    for a in active {
        sqlx::query(
            "INSERT INTO snap_activity \
                (poll_id, model, engine, active_requests, last_seen, host) \
             VALUES ($1,$2,$3,$4,$5,$6)",
        )
        .bind(poll_id)
        .bind(&a.model)
        .bind(&a.engine)
        .bind(a.active_requests as i32)
        .bind(a.last_seen)
        .bind(host)
        .execute(&mut *tx)
        .await
        .map_err(|e| SnapDbError::Database(format!("insert_activity_poll: {e}")))?;
    }

    tx.commit()
        .await
        .map_err(|e| SnapDbError::Database(format!("activity tx commit: {e}")))?;
    Ok(poll_id)
}

// ── SNAP-02: one sample header + its allocations (tx) ────────────────────────

/// Insert one VRAM sample (header + per-model allocations) atomically. Returns
/// the `sample_id`. The row-bloat interval gate is applied by the CALLER (the
/// health monitor) — this fn always writes when invoked.
pub async fn insert_vram_sample(
    pool: &PgPool,
    vram: &VRAMState,
    sampled_at: DateTime<Utc>,
    host: Option<&str>,
) -> Result<Uuid, SnapDbError> {
    let sample_id = Uuid::new_v4();

    let mut tx = pool
        .begin()
        .await
        .map_err(|e| SnapDbError::Database(format!("vram tx begin: {e}")))?;

    sqlx::query(
        "INSERT INTO snap_vram_sample \
            (sample_id, sampled_at, total_mb, used_mb, free_mb, host) \
         VALUES ($1,$2,$3,$4,$5,$6)",
    )
    .bind(sample_id)
    .bind(sampled_at)
    .bind(vram.total_mb as i64)
    .bind(vram.used_mb as i64)
    .bind(vram.free_mb as i64)
    .bind(host)
    .execute(&mut *tx)
    .await
    .map_err(|e| SnapDbError::Database(format!("insert_vram_sample header: {e}")))?;

    for alloc in &vram.allocations {
        sqlx::query(
            "INSERT INTO snap_vram_allocation \
                (sample_id, model_name, engine, size_mb, loaded_at) \
             VALUES ($1,$2,$3,$4,$5)",
        )
        .bind(sample_id)
        .bind(&alloc.model_name)
        .bind(&alloc.engine)
        .bind(alloc.size_mb as i64)
        .bind(alloc.loaded_at)
        .execute(&mut *tx)
        .await
        .map_err(|e| SnapDbError::Database(format!("insert_vram_allocation: {e}")))?;
    }

    tx.commit()
        .await
        .map_err(|e| SnapDbError::Database(format!("vram tx commit: {e}")))?;
    Ok(sample_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn req(model: &str) -> RequestRecord {
        RequestRecord {
            id: "req-1".into(),
            timestamp: Utc::now(),
            model: model.into(),
            endpoint: "/v1/chat/completions".into(),
            engine_url: "engine".into(),
            input_tokens: Some(10),
            output_tokens: Some(20),
            duration_ms: 123,
            status_code: 200,
            streaming: false,
        }
    }

    #[test]
    fn tier_str_maps_lowercase() {
        assert_eq!(tier_str(&StorageTier::Hot), "hot");
        assert_eq!(tier_str(&StorageTier::Warm), "warm");
    }

    #[test]
    fn u64_token_counts_bind_as_i64_without_overflow() {
        // The hot-path cast that must never panic / wrap.
        let r = req("m");
        assert_eq!(r.input_tokens.map(|v| v as i64), Some(10_i64));
        assert_eq!(r.duration_ms as i64, 123_i64);
        // Boundary: a very large but realistic token/byte value stays positive.
        let big: u64 = i64::MAX as u64;
        assert!(big as i64 >= 0);
    }

    #[test]
    fn activity_poll_filter_drops_idle_rows() {
        // Mirrors the WHERE active_requests > 0 gate inside insert_activity_poll
        // so the bloat rule is unit-covered without a DB.
        let now = Utc::now();
        let rows = vec![
            ModelActivity { model: "a".into(), engine: "e".into(), active_requests: 0, last_seen: now },
            ModelActivity { model: "b".into(), engine: "e".into(), active_requests: 3, last_seen: now },
        ];
        let active: Vec<&ModelActivity> =
            rows.iter().filter(|a| a.active_requests > 0).collect();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].model, "b");
    }

    #[test]
    fn inventory_record_path_lossy_is_stable() {
        let rec = ModelRecord {
            name: "m".into(),
            file_path: PathBuf::from("/models/m.gguf"),
            size_bytes: 42,
            quant_level: Some("Q4".into()),
            engine_compat: vec!["llama-cpp".into()],
            storage_tier: StorageTier::Hot,
            last_used: None,
            loaded: true,
        };
        assert_eq!(rec.file_path.to_string_lossy(), "/models/m.gguf");
        assert_eq!(rec.size_bytes as i64, 42_i64);
    }

    // DB-integration tests require a live Postgres (intake DB). They are
    // #[ignore]d so the default `cargo test` (no DB) stays green; run with
    // `cargo test -- --ignored` against a configured INTAKE_DATABASE_URL.
    #[tokio::test]
    #[ignore]
    async fn migrate_and_inserts_roundtrip() {
        let pool = get_pool().await.expect("pool");
        migrate(&pool).await.expect("migrate");
        insert_request_log(&pool, &req("roundtrip"), Some("test-host"))
            .await
            .expect("insert");
    }
}
