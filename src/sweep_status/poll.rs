//! The sweep-status background monitor: one tick every `poll_interval`
//! gathering DB + Ollama + GPU + systemd signals, computing verdicts,
//! appending to the JSONL log, and publishing the latest snapshot for the
//! HTTP endpoints to read.

use std::sync::Arc;

use once_cell::sync::Lazy;
use tokio::sync::RwLock;
use tracing::{info, warn};

use super::config::SweepMonitorConfig;
use super::db::{self, SweepDbStats, ASSISTANT_TABLE, CODER_TABLE};
use super::gpu;
use super::log::SweepStatusLog;
use super::ollama;
use super::snapshot::{SweepStatus, SweepStatusSnapshot};
use super::systemd;
use super::verdict::compute_verdict_optional;

/// Process-global latest snapshot, mirroring the SNAP subsystem's
/// `SHARED_STATE` pattern: a background writer (this module's tick loop) and
/// the HTTP read endpoints (`api.rs`) share this without needing a field on
/// chord's central `AppState`.
pub static LATEST_SNAPSHOT: Lazy<Arc<RwLock<Option<SweepStatusSnapshot>>>> =
    Lazy::new(|| Arc::new(RwLock::new(None)));

/// Spawn the sweep-status background monitor. Never blocks the caller and
/// never panics the process: a missing/unreachable intake DB degrades to
/// `db_configured: false` (or per-tick `available: false`) snapshots rather
/// than stopping the loop.
pub fn spawn(cfg: SweepMonitorConfig) {
    tokio::spawn(async move {
        run(cfg).await;
    });
}

async fn run(cfg: SweepMonitorConfig) {
    let log = SweepStatusLog::new(cfg.log_path.clone(), cfg.retention_days);
    let http_client = reqwest::Client::new();

    // `connect_lazy` never touches the network at construction time — a
    // briefly-down Postgres at startup can't block or fail this task; the
    // pool retries a real connection on each query, and every query already
    // handles its own error path (see `db::query_sweep_stats`).
    let pool = match super::config::intake_db_url() {
        Some(url) => match sqlx::PgPool::connect_lazy(&url) {
            Ok(p) => Some(p),
            Err(e) => {
                warn!(target: "chord.sweep_status", error = %e, "intake DB URL failed to parse — sweep-status DB monitoring disabled");
                None
            }
        },
        None => {
            info!(
                target: "chord.sweep_status",
                "INTAKE_DATABASE_URL/DATABASE_URL not set — sweep-status DB monitoring \
                 disabled (GPU/Ollama/systemd signals still collected)"
            );
            None
        }
    };
    let db_configured = pool.is_some();

    info!(
        target: "chord.sweep_status",
        interval_s = cfg.poll_interval.as_secs(),
        retention_days = cfg.retention_days,
        log = %cfg.log_path.display(),
        db_configured,
        "sweep-status monitor started"
    );

    let mut ticker = tokio::time::interval(cfg.poll_interval);
    loop {
        ticker.tick().await;
        let snapshot = tick(&cfg, pool.as_ref(), &http_client, db_configured).await;

        if let Err(e) = log.append(snapshot.timestamp, &snapshot).await {
            warn!(target: "chord.sweep_status", error = %e, "failed to append sweep-status snapshot to log");
        }
        if let Err(e) = log.enforce_retention(snapshot.timestamp).await {
            warn!(target: "chord.sweep_status", error = %e, "sweep-status log retention sweep failed");
        }

        *LATEST_SNAPSHOT.write().await = Some(snapshot);
    }
}

/// One poll tick: gather every signal and compute the full snapshot. Split
/// out from `run` so it's directly callable (no timer) — useful for a future
/// "force a check now" control endpoint, and keeps `run`'s loop trivial.
async fn tick(
    cfg: &SweepMonitorConfig,
    pool: Option<&sqlx::PgPool>,
    http_client: &reqwest::Client,
    db_configured: bool,
) -> SweepStatusSnapshot {
    let timestamp = chrono::Utc::now();

    let gpu_busy_percent = gpu::read_gpu_busy_percent(&cfg.drm_root).await;
    let ollama_stats = ollama::query_ollama_ps(http_client, &cfg.ollama_url).await;

    let coder_active = systemd::is_unit_active(&cfg.coder_service).await;
    let assistant_active = systemd::is_unit_active(&cfg.assistant_service).await;

    let coder_db = match pool {
        Some(p) => db::query_sweep_stats(p, CODER_TABLE).await,
        None => unconfigured_db_stats(),
    };
    let assistant_db = match pool {
        Some(p) => db::query_sweep_stats(p, ASSISTANT_TABLE).await,
        None => unconfigured_db_stats(),
    };

    let coder_verdict = compute_verdict_optional(
        coder_active,
        coder_db.latest_row_age_secs,
        gpu_busy_percent,
        cfg.stuck_age_secs,
        cfg.gpu_busy_threshold,
    );
    let assistant_verdict = compute_verdict_optional(
        assistant_active,
        assistant_db.latest_row_age_secs,
        gpu_busy_percent,
        cfg.stuck_age_secs,
        cfg.gpu_busy_threshold,
    );

    let coder = SweepStatus {
        name: "coder".to_string(),
        service_unit: cfg.coder_service.clone(),
        service_active: coder_active,
        db: coder_db,
        verdict: coder_verdict,
    };
    let assistant = SweepStatus {
        name: "assistant".to_string(),
        service_unit: cfg.assistant_service.clone(),
        service_active: assistant_active,
        db: assistant_db,
        verdict: assistant_verdict,
    };

    let overall_verdict = SweepStatusSnapshot::compute_overall(coder_verdict, Some(assistant_verdict));

    SweepStatusSnapshot {
        timestamp,
        coder,
        assistant: Some(assistant),
        gpu_busy_percent,
        ollama: ollama_stats,
        db_configured,
        overall_verdict,
    }
}

fn unconfigured_db_stats() -> SweepDbStats {
    SweepDbStats {
        available: false,
        error_message: Some("monitoring unavailable: INTAKE_DATABASE_URL (or DATABASE_URL) not set".to_string()),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end (minus DB/network) sanity check of a single tick with no
    /// Postgres pool configured: this is exactly the "INTAKE_DATABASE_URL
    /// unset" production degradation path, and must produce a usable,
    /// non-panicking snapshot rather than blocking or erroring out.
    #[tokio::test]
    async fn tick_without_db_pool_degrades_cleanly() {
        let tmp = tempfile::tempdir().unwrap();
        let mut cfg = SweepMonitorConfig::test_default(tmp.path().join("sweep-status.jsonl"));
        // Point GPU reading at an empty dir so it resolves to None deterministically,
        // and use a service name that will never be active in this sandbox.
        cfg.drm_root = tmp.path().join("drm");
        let client = reqwest::Client::new();

        let snapshot = tick(&cfg, None, &client, false).await;

        assert!(!snapshot.db_configured);
        assert!(!snapshot.coder.db.available);
        assert!(snapshot.coder.db.error_message.as_deref().unwrap().contains("INTAKE_DATABASE_URL"));
        assert!(!snapshot.assistant.as_ref().unwrap().db.available);
        assert!(snapshot.gpu_busy_percent.is_none());
        // No systemd unit named this exists (or systemctl is unavailable in the
        // sandbox) -> not confirmed active -> Idle, never Stuck, even though
        // the DB is "unavailable" (which maps to i64::MAX age).
        assert_eq!(snapshot.coder.verdict, super::super::verdict::Verdict::Idle);
        assert_eq!(snapshot.overall_verdict, super::super::verdict::Verdict::Idle);

        // The snapshot must be JSON-serializable (this is what gets logged +
        // served over HTTP).
        let json = serde_json::to_string(&snapshot).unwrap();
        assert!(json.contains("\"overall_verdict\":\"idle\""));
    }
}
