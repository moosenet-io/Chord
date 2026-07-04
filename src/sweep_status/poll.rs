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

/// Tracks how long a single sweep has been *continuously* observed in the
/// "service active + DB available + table still empty" state, so the
/// verdict layer (`verdict::compute_verdict_optional`) can bound its
/// start-up grace period instead of granting `Working` forever. Reset to
/// "not observed" the instant the sweep leaves that state — a real row
/// lands, the service goes inactive, or the DB becomes unreachable — since
/// none of those still describe the state being timed.
///
/// This is deliberately NOT part of the pure `verdict` module: `verdict.rs`
/// stays a dependency-free function of its inputs (easy to unit-test with
/// synthetic timestamps), while wall-clock bookkeeping across ticks lives
/// here in the stateful poll loop where it belongs.
#[derive(Debug, Default)]
struct EmptyTableTracker {
    first_observed: Option<chrono::DateTime<chrono::Utc>>,
}

impl EmptyTableTracker {
    /// Call once per tick with whether this sweep is currently in the
    /// "active + db-available + empty table" state. Returns how many
    /// seconds that state has been continuously observed (0 the tick it
    /// first appears), or `None` if the sweep isn't currently in that state.
    fn observe(&mut self, in_empty_state: bool, now: chrono::DateTime<chrono::Utc>) -> Option<i64> {
        if !in_empty_state {
            self.first_observed = None;
            return None;
        }
        let first = *self.first_observed.get_or_insert(now);
        Some((now - first).num_seconds().max(0))
    }
}

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

    let mut coder_empty_tracker = EmptyTableTracker::default();
    let mut assistant_empty_tracker = EmptyTableTracker::default();

    let mut ticker = tokio::time::interval(cfg.poll_interval);
    loop {
        ticker.tick().await;
        let snapshot = tick(
            &cfg,
            pool.as_ref(),
            &http_client,
            db_configured,
            &mut coder_empty_tracker,
            &mut assistant_empty_tracker,
        )
        .await;

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
    coder_empty_tracker: &mut EmptyTableTracker,
    assistant_empty_tracker: &mut EmptyTableTracker,
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

    let coder_empty_elapsed = coder_empty_tracker.observe(
        coder_active == Some(true) && coder_db.available && coder_db.latest_row_age_secs.is_none(),
        timestamp,
    );
    let assistant_empty_elapsed = assistant_empty_tracker.observe(
        assistant_active == Some(true) && assistant_db.available && assistant_db.latest_row_age_secs.is_none(),
        timestamp,
    );

    let coder_verdict = compute_verdict_optional(
        coder_active,
        coder_db.available,
        coder_db.latest_row_age_secs,
        gpu_busy_percent,
        cfg.stuck_age_secs,
        cfg.gpu_busy_threshold,
        coder_empty_elapsed,
        cfg.startup_grace_secs,
    );
    let assistant_verdict = compute_verdict_optional(
        assistant_active,
        assistant_db.available,
        assistant_db.latest_row_age_secs,
        gpu_busy_percent,
        cfg.stuck_age_secs,
        cfg.gpu_busy_threshold,
        assistant_empty_elapsed,
        cfg.startup_grace_secs,
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
        let mut coder_tracker = EmptyTableTracker::default();
        let mut assistant_tracker = EmptyTableTracker::default();

        let snapshot = tick(&cfg, None, &client, false, &mut coder_tracker, &mut assistant_tracker).await;

        assert!(!snapshot.db_configured);
        assert!(!snapshot.coder.db.available);
        assert!(snapshot.coder.db.error_message.as_deref().unwrap().contains("INTAKE_DATABASE_URL"));
        assert!(!snapshot.assistant.as_ref().unwrap().db.available);
        assert!(snapshot.gpu_busy_percent.is_none());
        // This test's own service name never exists, so `service_active` is
        // never confirmed-true, and — regardless of whether the *real*
        // `systemctl is-active` in whatever environment runs this test can
        // reach D-Bus — the resulting verdict must land on one of the two
        // "not confirmed active" outcomes: `Idle` (systemctl cleanly answered
        // "inactive") or `Unknown` (systemctl couldn't be queried at all,
        // e.g. no D-Bus in a sandbox/CI container). Which of the two depends
        // on host D-Bus availability, not on this tick's own logic, so this
        // test intentionally does not pin one — that distinction is already
        // exhaustively covered, environment-independently, by
        // `systemd::classify_is_active_output`'s own unit tests. What this
        // test verifies is the DB-pool-absent degrade path itself: no panic,
        // a usable snapshot, and never a false confirmed-active/Stuck read.
        assert_ne!(snapshot.coder.service_active, Some(true));
        assert!(matches!(
            snapshot.coder.verdict,
            super::super::verdict::Verdict::Idle | super::super::verdict::Verdict::Unknown
        ));
        assert_eq!(snapshot.coder.verdict, snapshot.overall_verdict);

        // The snapshot must be JSON-serializable (this is what gets logged +
        // served over HTTP).
        let json = serde_json::to_string(&snapshot).unwrap();
        assert!(json.contains("\"overall_verdict\":\"idle\"") || json.contains("\"overall_verdict\":\"unknown\""));
    }

    // ── EmptyTableTracker ────────────────────────────────────────────────

    #[test]
    fn empty_table_tracker_starts_at_zero_elapsed() {
        let mut tracker = EmptyTableTracker::default();
        let t0 = chrono::Utc::now();
        assert_eq!(tracker.observe(true, t0), Some(0));
    }

    #[test]
    fn empty_table_tracker_accumulates_elapsed_across_ticks() {
        let mut tracker = EmptyTableTracker::default();
        let t0 = chrono::Utc::now();
        assert_eq!(tracker.observe(true, t0), Some(0));
        assert_eq!(tracker.observe(true, t0 + chrono::Duration::seconds(30)), Some(30));
        assert_eq!(tracker.observe(true, t0 + chrono::Duration::seconds(400)), Some(400));
    }

    #[test]
    fn empty_table_tracker_resets_when_state_is_left() {
        // A row lands (or the service/DB signal changes) -> the very next
        // tick where `in_empty_state` is false must clear the clock, so a
        // *later* empty-table spell starts its own fresh grace window
        // instead of inheriting stale elapsed time.
        let mut tracker = EmptyTableTracker::default();
        let t0 = chrono::Utc::now();
        assert_eq!(tracker.observe(true, t0), Some(0));
        assert_eq!(tracker.observe(true, t0 + chrono::Duration::seconds(100)), Some(100));

        assert_eq!(tracker.observe(false, t0 + chrono::Duration::seconds(150)), None);

        // Re-entering the empty-table state later starts counting from zero
        // again, not from the old 100s.
        assert_eq!(tracker.observe(true, t0 + chrono::Duration::seconds(200)), Some(0));
    }

    #[test]
    fn empty_table_tracker_never_observed_returns_none() {
        let mut tracker = EmptyTableTracker::default();
        assert_eq!(tracker.observe(false, chrono::Utc::now()), None);
    }
}
