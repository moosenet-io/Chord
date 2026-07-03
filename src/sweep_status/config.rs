//! Sweep-status monitor configuration — entirely env-sourced, consistent with
//! chord's "no hardcoded secrets/paths, all config from env" rule (see
//! `crate::config`). No literal DB connection string, hostname, or path is
//! ever baked into source; every value here has a safe, non-secret default.

use std::time::Duration;

/// Env var holding the sweep-status JSONL log "template" path. The monitor
/// derives daily-rotated filenames from this (see `sweep_status::log`):
/// directory + stem + date + extension, e.g. a default of
/// `/var/log/chord/sweep-status.jsonl` yields
/// `/var/log/chord/sweep-status-2026-07-02.jsonl` etc.
pub const LOG_PATH_ENV: &str = "CHORD_SWEEP_STATUS_LOG";
const DEFAULT_LOG_PATH: &str = "/var/log/chord/sweep-status.jsonl";

/// Poll cadence (seconds). `CHORD_SWEEP_POLL_INTERVAL_SECS`, default 30 per
/// the design brief.
pub const POLL_INTERVAL_ENV: &str = "CHORD_SWEEP_POLL_INTERVAL_SECS";
const DEFAULT_POLL_INTERVAL_SECS: u64 = 30;

/// Retention window (days) for the JSONL history. `CHORD_SWEEP_RETENTION_DAYS`,
/// default 10.
pub const RETENTION_DAYS_ENV: &str = "CHORD_SWEEP_RETENTION_DAYS";
const DEFAULT_RETENTION_DAYS: u32 = 10;

/// "No fresh row" age threshold (seconds) used by the stuck heuristic.
/// `CHORD_SWEEP_STUCK_AGE_SECS`, default 360 (6 min), matching the host
/// watchdog already deployed for this exact failure mode.
pub const STUCK_AGE_SECS_ENV: &str = "CHORD_SWEEP_STUCK_AGE_SECS";

/// GPU-busy threshold (percent) used by the stuck heuristic.
/// `CHORD_SWEEP_GPU_BUSY_THRESHOLD`, default 70.0.
pub const GPU_BUSY_THRESHOLD_ENV: &str = "CHORD_SWEEP_GPU_BUSY_THRESHOLD";

/// Bound (seconds) on the "service active, DB reachable, table still empty"
/// start-up grace period — past this window an empty table no longer
/// unconditionally reports `Working` (see `verdict::compute_verdict_optional`).
/// `CHORD_SWEEP_STARTUP_GRACE_SECS`, default 360 (6 min), matching the
/// existing stuck-age default: there's no principled reason a sweep should
/// be allowed to sit at "zero rows ever" any longer than one that already
/// has a stale row before the stuck heuristic engages.
pub const STARTUP_GRACE_SECS_ENV: &str = "CHORD_SWEEP_STARTUP_GRACE_SECS";

/// Local Ollama base URL to poll `/api/ps` on. `CHORD_SWEEP_OLLAMA_URL`,
/// default `http://localhost:11434`.
pub const OLLAMA_URL_ENV: &str = "CHORD_SWEEP_OLLAMA_URL";
const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";

/// systemd unit name for the coder sweep. `CHORD_SWEEP_CODER_SERVICE`,
/// default `intake-coder-sweep.service`.
pub const CODER_SERVICE_ENV: &str = "CHORD_SWEEP_CODER_SERVICE";
const DEFAULT_CODER_SERVICE: &str = "intake-coder-sweep.service";

/// systemd unit name for the assistant sweep. `CHORD_SWEEP_ASSISTANT_SERVICE`,
/// default `intake-assistant-sweep.service`.
pub const ASSISTANT_SERVICE_ENV: &str = "CHORD_SWEEP_ASSISTANT_SERVICE";
const DEFAULT_ASSISTANT_SERVICE: &str = "intake-assistant-sweep.service";

/// Root directory to scan for `card*/device/gpu_busy_percent` sysfs nodes.
/// `CHORD_SWEEP_DRM_ROOT`, default `/sys/class/drm`. Overridable so tests can
/// point at a fake sysfs tree without touching the real one.
pub const DRM_ROOT_ENV: &str = "CHORD_SWEEP_DRM_ROOT";
const DEFAULT_DRM_ROOT: &str = "/sys/class/drm";

/// Full sweep-status monitor configuration, read once at spawn time.
#[derive(Debug, Clone)]
pub struct SweepMonitorConfig {
    pub log_path: std::path::PathBuf,
    pub poll_interval: Duration,
    pub retention_days: u32,
    pub stuck_age_secs: i64,
    pub gpu_busy_threshold: f64,
    pub startup_grace_secs: i64,
    pub ollama_url: String,
    pub coder_service: String,
    pub assistant_service: String,
    pub drm_root: std::path::PathBuf,
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| default.to_string())
}

fn env_parse_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(default)
}

impl SweepMonitorConfig {
    /// Build from environment variables. Never fails — every field has a
    /// documented, non-secret default so an unconfigured Chord still starts
    /// and runs the monitor (it just uses the stock production values).
    pub fn from_env() -> Self {
        SweepMonitorConfig {
            log_path: std::path::PathBuf::from(env_or(LOG_PATH_ENV, DEFAULT_LOG_PATH)),
            poll_interval: Duration::from_secs(
                env_parse_or(POLL_INTERVAL_ENV, DEFAULT_POLL_INTERVAL_SECS).max(1),
            ),
            retention_days: env_parse_or(RETENTION_DAYS_ENV, DEFAULT_RETENTION_DAYS).max(1),
            stuck_age_secs: env_parse_or(
                STUCK_AGE_SECS_ENV,
                crate::sweep_status::verdict::DEFAULT_STUCK_AGE_SECS,
            ),
            gpu_busy_threshold: env_parse_or(
                GPU_BUSY_THRESHOLD_ENV,
                crate::sweep_status::verdict::DEFAULT_GPU_BUSY_THRESHOLD_PERCENT,
            ),
            startup_grace_secs: env_parse_or(
                STARTUP_GRACE_SECS_ENV,
                crate::sweep_status::verdict::DEFAULT_STARTUP_GRACE_SECS,
            ),
            ollama_url: env_or(OLLAMA_URL_ENV, DEFAULT_OLLAMA_URL),
            coder_service: env_or(CODER_SERVICE_ENV, DEFAULT_CODER_SERVICE),
            assistant_service: env_or(ASSISTANT_SERVICE_ENV, DEFAULT_ASSISTANT_SERVICE),
            drm_root: std::path::PathBuf::from(env_or(DRM_ROOT_ENV, DEFAULT_DRM_ROOT)),
        }
    }

    /// Deterministic config for unit tests (no env reads).
    #[cfg(test)]
    pub fn test_default(log_path: std::path::PathBuf) -> Self {
        SweepMonitorConfig {
            log_path,
            poll_interval: Duration::from_secs(30),
            retention_days: 10,
            stuck_age_secs: crate::sweep_status::verdict::DEFAULT_STUCK_AGE_SECS,
            gpu_busy_threshold: crate::sweep_status::verdict::DEFAULT_GPU_BUSY_THRESHOLD_PERCENT,
            startup_grace_secs: crate::sweep_status::verdict::DEFAULT_STARTUP_GRACE_SECS,
            ollama_url: DEFAULT_OLLAMA_URL.to_string(),
            coder_service: DEFAULT_CODER_SERVICE.to_string(),
            assistant_service: DEFAULT_ASSISTANT_SERVICE.to_string(),
            drm_root: std::path::PathBuf::from(DEFAULT_DRM_ROOT),
        }
    }
}

/// Resolve the intake-DB URL used for sweep monitoring. Reuses chord's
/// already-established resolver (`terminus_rs::config::intake_database_url`,
/// i.e. `INTAKE_DATABASE_URL` falling back to `DATABASE_URL`) — the SAME
/// variable `main.rs`'s serving-profile routing map and `snap::storage`
/// already read, rather than inventing a second, parallel DB-URL env var for
/// the same database. `None` ⇒ DB monitoring disabled; callers must produce a
/// clear "monitoring unavailable" marker, never panic.
pub fn intake_db_url() -> Option<String> {
    terminus_rs::config::intake_database_url()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn defaults_when_unset() {
        std::env::remove_var(LOG_PATH_ENV);
        std::env::remove_var(POLL_INTERVAL_ENV);
        std::env::remove_var(RETENTION_DAYS_ENV);
        std::env::remove_var(STUCK_AGE_SECS_ENV);
        std::env::remove_var(GPU_BUSY_THRESHOLD_ENV);
        std::env::remove_var(STARTUP_GRACE_SECS_ENV);
        std::env::remove_var(OLLAMA_URL_ENV);
        std::env::remove_var(CODER_SERVICE_ENV);
        std::env::remove_var(ASSISTANT_SERVICE_ENV);
        std::env::remove_var(DRM_ROOT_ENV);

        let cfg = SweepMonitorConfig::from_env();
        assert_eq!(cfg.log_path, std::path::PathBuf::from(DEFAULT_LOG_PATH));
        assert_eq!(cfg.poll_interval, Duration::from_secs(30));
        assert_eq!(cfg.retention_days, 10);
        assert_eq!(cfg.stuck_age_secs, 360);
        assert_eq!(cfg.gpu_busy_threshold, 70.0);
        assert_eq!(cfg.startup_grace_secs, 360);
        assert_eq!(cfg.ollama_url, DEFAULT_OLLAMA_URL);
        assert_eq!(cfg.coder_service, DEFAULT_CODER_SERVICE);
        assert_eq!(cfg.assistant_service, DEFAULT_ASSISTANT_SERVICE);
        assert_eq!(cfg.drm_root, std::path::PathBuf::from(DEFAULT_DRM_ROOT));
    }

    #[test]
    #[serial]
    fn reads_custom_values() {
        std::env::set_var(LOG_PATH_ENV, "/tmp/custom-sweep.jsonl");
        std::env::set_var(POLL_INTERVAL_ENV, "15");
        std::env::set_var(RETENTION_DAYS_ENV, "3");
        std::env::set_var(STUCK_AGE_SECS_ENV, "120");
        std::env::set_var(GPU_BUSY_THRESHOLD_ENV, "50");
        std::env::set_var(STARTUP_GRACE_SECS_ENV, "90");
        std::env::set_var(OLLAMA_URL_ENV, "http://localhost:9999");
        std::env::set_var(CODER_SERVICE_ENV, "custom-coder.service");
        std::env::set_var(ASSISTANT_SERVICE_ENV, "custom-assistant.service");

        let cfg = SweepMonitorConfig::from_env();
        assert_eq!(cfg.log_path, std::path::PathBuf::from("/tmp/custom-sweep.jsonl"));
        assert_eq!(cfg.poll_interval, Duration::from_secs(15));
        assert_eq!(cfg.retention_days, 3);
        assert_eq!(cfg.stuck_age_secs, 120);
        assert_eq!(cfg.gpu_busy_threshold, 50.0);
        assert_eq!(cfg.startup_grace_secs, 90);
        assert_eq!(cfg.ollama_url, "http://localhost:9999");
        assert_eq!(cfg.coder_service, "custom-coder.service");
        assert_eq!(cfg.assistant_service, "custom-assistant.service");

        std::env::remove_var(LOG_PATH_ENV);
        std::env::remove_var(POLL_INTERVAL_ENV);
        std::env::remove_var(RETENTION_DAYS_ENV);
        std::env::remove_var(STUCK_AGE_SECS_ENV);
        std::env::remove_var(GPU_BUSY_THRESHOLD_ENV);
        std::env::remove_var(STARTUP_GRACE_SECS_ENV);
        std::env::remove_var(OLLAMA_URL_ENV);
        std::env::remove_var(CODER_SERVICE_ENV);
        std::env::remove_var(ASSISTANT_SERVICE_ENV);
    }

    #[test]
    #[serial]
    fn zero_poll_interval_is_clamped_to_one_second() {
        std::env::set_var(POLL_INTERVAL_ENV, "0");
        let cfg = SweepMonitorConfig::from_env();
        assert_eq!(cfg.poll_interval, Duration::from_secs(1));
        std::env::remove_var(POLL_INTERVAL_ENV);
    }
}
