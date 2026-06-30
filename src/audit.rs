//! Audit logging for chord-proxy.
//!
//! Writes one JSONL record per request to `${CHORD_AUDIT_PATH}/chord-audit.jsonl`
//! (default path: `/var/log/chord/audit.jsonl`).
//!
//! Sensitive content (tool arguments, LLM messages, memory content) is NEVER logged.
//! Only metadata is recorded.
//!
//! Log rotation: files are renamed to `.1`, `.2`, … `.10` when the active file
//! reaches 100 MiB; files beyond the 10th are removed.

use serde::{Deserialize, Serialize};
use std::io::Write as IoWrite;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

const MAX_ROTATION_FILES: u32 = 10;
const MAX_FILE_BYTES: u64 = 100 * 1024 * 1024; // 100 MiB

// ── Public types ──────────────────────────────────────────────────────────────

/// The kind of request being audited.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RequestType {
    Llm,
    ToolList,
    ToolCall,
    ToolDiscover,
    AuthFailure,
}

/// Outcome of the request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Success,
    Error,
    Timeout,
    Fallback,
}

/// One audit record (serialised as a JSONL line).
///
/// No sensitive fields: tool arguments, LLM messages, and memory content are
/// intentionally absent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// RFC-3339 UTC timestamp.
    pub timestamp: String,
    /// `sub` claim from JWT, or `"anonymous"` if JWT was absent/invalid.
    pub user_id: String,
    /// Category of request.
    pub request_type: RequestType,
    /// Model name (for LLM calls) or tool name (for tool calls). Empty for list/discover.
    pub target: String,
    /// Wall-clock duration of the handler in milliseconds.
    pub duration_ms: u64,
    /// Final outcome.
    pub status: Status,
    /// Human-readable error description when `status != Success`. Not present on success.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    /// First 8 hex chars of SHA-256(token), only present for `AuthFailure` entries.
    /// The raw token is NEVER stored.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_hash_prefix: Option<String>,
}

impl AuditEntry {
    /// Construct a successful entry (no error message).
    pub fn success(
        user_id: &str,
        request_type: RequestType,
        target: &str,
        duration_ms: u64,
    ) -> Self {
        Self {
            timestamp: utc_now_rfc3339(),
            user_id: user_id.to_string(),
            request_type,
            target: target.to_string(),
            duration_ms,
            status: Status::Success,
            error_message: None,
            token_hash_prefix: None,
        }
    }

    /// Construct an error/timeout/fallback entry.
    pub fn failed(
        user_id: &str,
        request_type: RequestType,
        target: &str,
        duration_ms: u64,
        status: Status,
        error_message: Option<String>,
    ) -> Self {
        Self {
            timestamp: utc_now_rfc3339(),
            user_id: user_id.to_string(),
            request_type,
            target: target.to_string(),
            duration_ms,
            status,
            error_message,
            token_hash_prefix: None,
        }
    }

    /// Construct an auth-failure entry. The raw token is hashed; only the first
    /// 8 hex characters of SHA-256 are stored.
    pub fn auth_failure(raw_token: Option<&str>, duration_ms: u64) -> Self {
        let hash_prefix = raw_token.map(|t| token_hash_prefix(t));
        Self {
            timestamp: utc_now_rfc3339(),
            user_id: "anonymous".to_string(),
            request_type: RequestType::AuthFailure,
            target: String::new(),
            duration_ms,
            status: Status::Error,
            error_message: Some("Authentication failed".to_string()),
            token_hash_prefix: hash_prefix,
        }
    }
}

// ── AuditLogger ───────────────────────────────────────────────────────────────

/// Thread-safe writer that appends JSONL records to the audit log.
///
/// Constructed once at startup and shared via `Arc<AuditLogger>`.
pub struct AuditLogger {
    log_path: PathBuf,
    inner: Mutex<LogInner>,
}

struct LogInner {
    file: Option<std::fs::File>,
}

impl AuditLogger {
    /// Create a new `AuditLogger`.
    /// Reads `CHORD_AUDIT_PATH` env var; defaults to `/var/log/chord`.
    /// The directory is created if it does not exist.
    pub fn from_env() -> Self {
        let dir = std::env::var("CHORD_AUDIT_PATH")
            .unwrap_or_else(|_| "/var/log/chord".to_string());
        let log_path = PathBuf::from(dir).join("chord-audit.jsonl");
        Self::new(log_path)
    }

    /// Create a new `AuditLogger` with an explicit path (useful in tests).
    pub fn new(log_path: PathBuf) -> Self {
        Self {
            log_path,
            inner: Mutex::new(LogInner { file: None }),
        }
    }

    // ── Public logging API ────────────────────────────────────────────────────

    /// Log a completed LLM inference call.
    pub fn log_llm_call(&self, user_id: &str, model: &str, duration_ms: u64, status: Status, error: Option<String>) {
        let entry = if status == Status::Success && error.is_none() {
            AuditEntry::success(user_id, RequestType::Llm, model, duration_ms)
        } else {
            AuditEntry::failed(user_id, RequestType::Llm, model, duration_ms, status, error)
        };
        self.write_entry(&entry);
    }

    /// Log a tool invocation (tool_call endpoint).
    pub fn log_tool_call(&self, user_id: &str, tool_name: &str, duration_ms: u64, status: Status, error: Option<String>) {
        let entry = if status == Status::Success && error.is_none() {
            AuditEntry::success(user_id, RequestType::ToolCall, tool_name, duration_ms)
        } else {
            AuditEntry::failed(user_id, RequestType::ToolCall, tool_name, duration_ms, status, error)
        };
        self.write_entry(&entry);
    }

    /// Log an authentication failure.
    /// `raw_token` is only used to compute a hash prefix; the value itself is never stored.
    pub fn log_auth_failure(&self, raw_token: Option<&str>, duration_ms: u64) {
        let entry = AuditEntry::auth_failure(raw_token, duration_ms);
        self.write_entry(&entry);
    }

    /// Log a generic entry (used by middleware for list/discover/health).
    pub fn log_entry(&self, entry: &AuditEntry) {
        self.write_entry(entry);
    }

    // ── Write path ────────────────────────────────────────────────────────────

    fn write_entry(&self, entry: &AuditEntry) {
        let line = match serde_json::to_string(entry) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("chord-audit: failed to serialise entry: {e}");
                return;
            }
        };

        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("chord-audit: mutex poisoned: {e}");
                return;
            }
        };

        // Ensure directory + file handle exist.
        if guard.file.is_none() {
            if let Err(e) = self.ensure_dir() {
                eprintln!("chord-audit: cannot create log directory: {e}");
                return;
            }
            match self.open_file() {
                Ok(f) => guard.file = Some(f),
                Err(e) => {
                    eprintln!("chord-audit: cannot open log file: {e}");
                    return;
                }
            }
        }

        // Rotate if needed.
        if let Some(ref f) = guard.file {
            let size = f.metadata().map(|m| m.len()).unwrap_or(0);
            if size >= MAX_FILE_BYTES {
                // Flush current handle before rotating.
                drop(guard.file.take());
                if let Err(e) = self.rotate() {
                    eprintln!("chord-audit: rotation failed: {e}");
                }
                match self.open_file() {
                    Ok(f) => guard.file = Some(f),
                    Err(e) => {
                        eprintln!("chord-audit: cannot open log file after rotation: {e}");
                        return;
                    }
                }
            }
        }

        if let Some(ref mut f) = guard.file {
            if let Err(e) = writeln!(f, "{line}") {
                eprintln!("chord-audit: write failed (disk full?): {e}");
                // Don't crash — drop the handle so next call retries.
                guard.file = None;
            }
        }
    }

    fn ensure_dir(&self) -> std::io::Result<()> {
        if let Some(parent) = self.log_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Ok(())
    }

    fn open_file(&self) -> std::io::Result<std::fs::File> {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_path)
    }

    /// Rotate: `.jsonl` → `.1.jsonl`, `.1` → `.2`, … `.9` → `.10`, remove `.10` if exists.
    fn rotate(&self) -> std::io::Result<()> {
        let base = self.log_path.with_extension(""); // strip .jsonl
        // Remove the oldest file if it would exceed our keep count.
        let oldest = PathBuf::from(format!("{}.{}.jsonl", base.display(), MAX_ROTATION_FILES));
        if oldest.exists() {
            let _ = std::fs::remove_file(&oldest);
        }
        // Rename .9 → .10, .8 → .9, … .1 → .2
        for i in (1..MAX_ROTATION_FILES).rev() {
            let from = PathBuf::from(format!("{}.{}.jsonl", base.display(), i));
            let to   = PathBuf::from(format!("{}.{}.jsonl", base.display(), i + 1));
            if from.exists() {
                let _ = std::fs::rename(&from, &to);
            }
        }
        // Rename active file → .1
        let rotated = PathBuf::from(format!("{}.1.jsonl", base.display()));
        if self.log_path.exists() {
            std::fs::rename(&self.log_path, &rotated)?;
        }
        Ok(())
    }

    // ── Summary ───────────────────────────────────────────────────────────────

    /// Produce a daily summary by reading the current log file.
    ///
    /// Returns counts by type and status for entries within the last 24 hours.
    /// Reads both the active log file and rotated files (.1..10) to cover
    /// entries that crossed a rotation boundary within the window.
    pub fn daily_summary(&self) -> AuditSummary {
        let cutoff = utc_now_secs().saturating_sub(86_400);
        let mut summary = AuditSummary::default();

        // Collect all files to scan: active + rotated (.1 through .10).
        // rotate() produces {stem}.{n}.jsonl (e.g. chord-audit.1.jsonl),
        // so strip the .jsonl extension first, then append .{n}.jsonl.
        let stem = self.log_path.with_extension("");
        let mut files_to_scan: Vec<std::path::PathBuf> = vec![self.log_path.clone()];
        for i in 1..=10u32 {
            let rotated = std::path::PathBuf::from(
                format!("{}.{}.jsonl", stem.display(), i)
            );
            if rotated.exists() {
                files_to_scan.push(rotated);
            }
        }

        for file_path in &files_to_scan {
            let Ok(contents) = std::fs::read_to_string(file_path) else {
                continue;
            };
            for line in contents.lines() {
                let Ok(entry) = serde_json::from_str::<AuditEntry>(line) else {
                    continue;
                };
                if let Some(ts) = parse_rfc3339_secs(&entry.timestamp) {
                    if ts < cutoff {
                        continue;
                    }
                }
                summary.total += 1;
                let type_key = serde_json::to_value(&entry.request_type)
                    .ok()
                    .and_then(|v| v.as_str().map(str::to_string))
                    .unwrap_or_default();
                let status_key = serde_json::to_value(&entry.status)
                    .ok()
                    .and_then(|v| v.as_str().map(str::to_string))
                    .unwrap_or_default();
                *summary.by_type.entry(type_key).or_insert(0) += 1;
                *summary.by_status.entry(status_key).or_insert(0) += 1;
            }
        }

        summary
    }
}

/// Aggregated counts for the `/v1/audit/summary` API response.
///
/// `by_user` is intentionally absent — this endpoint is unauthenticated and
/// should not expose JWT `sub` identities. Use the raw JSONL log for per-user
/// analysis (requires filesystem access).
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct AuditSummary {
    pub total: u64,
    pub by_type: std::collections::HashMap<String, u64>,
    pub by_status: std::collections::HashMap<String, u64>,
    pub window_hours: u64,
}

impl AuditSummary {
    pub fn new_with_window(window_hours: u64) -> Self {
        Self {
            window_hours,
            ..Default::default()
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Compute first 8 hex chars of SHA-256(token).
pub fn token_hash_prefix(token: &str) -> String {
    use sha2::{Digest, Sha256};
    let hash = Sha256::digest(token.as_bytes());
    // hex-encode the first 4 bytes (= 8 hex chars)
    format!("{:02x}{:02x}{:02x}{:02x}", hash[0], hash[1], hash[2], hash[3])
}

/// Current UTC time as RFC-3339 string (seconds precision).
fn utc_now_rfc3339() -> String {
    // Use SystemTime → format manually to avoid pulling in chrono.
    let secs = utc_now_secs();
    secs_to_rfc3339(secs)
}

fn utc_now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Minimal RFC-3339 formatter (UTC, seconds precision, no sub-seconds).
fn secs_to_rfc3339(secs: u64) -> String {
    // Days since Unix epoch
    let mut remaining = secs;
    let s = remaining % 60; remaining /= 60;
    let m = remaining % 60; remaining /= 60;
    let h = remaining % 24; remaining /= 24;

    // Gregorian calendar calculation from days since 1970-01-01
    let (year, month, day) = days_to_ymd(remaining as i64);

    format!("{year:04}-{month:02}-{day:02}T{h:02}:{m:02}:{s:02}Z")
}

fn days_to_ymd(days: i64) -> (i64, i64, i64) {
    // Algorithm from Richards (2013), adapted for epoch 1970-01-01.
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

/// Parse an RFC-3339 string back to unix seconds (best-effort).
fn parse_rfc3339_secs(ts: &str) -> Option<u64> {
    // Expected format: YYYY-MM-DDTHH:MM:SSZ (19+ chars)
    if ts.len() < 19 {
        return None;
    }
    let year:  u64 = ts[0..4].parse().ok()?;
    let month: u64 = ts[5..7].parse().ok()?;
    let day:   u64 = ts[8..10].parse().ok()?;
    let hour:  u64 = ts[11..13].parse().ok()?;
    let min:   u64 = ts[14..16].parse().ok()?;
    let sec:   u64 = ts[17..19].parse().ok()?;

    let days = days_from_epoch(year, month, day)?;
    Some(days * 86400 + hour * 3600 + min * 60 + sec)
}

/// Count calendar days from 1970-01-01 to given date.
fn days_from_epoch(year: u64, month: u64, day: u64) -> Option<u64> {
    // Cumulative days per month (non-leap year).
    const MONTHS: [u64; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    if month < 1 || month > 12 || day < 1 { return None; }
    // Reject dates before the Unix epoch.
    if year < 1970 { return None; }

    // Count leap years between 1970 (inclusive) and `year` (exclusive):
    //   leap_years_before(y) = y/4 - y/100 + y/400
    fn leap_years_before(y: u64) -> u64 {
        (y - 1) / 4 - (y - 1) / 100 + (y - 1) / 400
    }
    let leap_before_year  = leap_years_before(year);
    let leap_before_epoch = leap_years_before(1970);
    let extra_leaps = leap_before_year.saturating_sub(leap_before_epoch);
    let days_to_year = (year - 1970) * 365 + extra_leaps;

    let is_leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    let month_days = MONTHS[(month - 1) as usize]
        + if is_leap && month > 2 { 1 } else { 0 };
    let total = days_to_year + month_days + day - 1;
    Some(total)
}

/// A simple wall-clock timer helper used by middleware.
pub struct Timer(Instant);

impl Timer {
    pub fn start() -> Self { Self(Instant::now()) }
    pub fn elapsed_ms(&self) -> u64 { self.0.elapsed().as_millis() as u64 }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use tempfile::TempDir;

    fn temp_logger() -> (AuditLogger, TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("chord-audit.jsonl");
        (AuditLogger::new(path), dir)
    }

    // ── Basic entry construction ──────────────────────────────────────────────

    #[test]
    fn test_llm_call_produces_correct_fields() {
        let (logger, _dir) = temp_logger();
        logger.log_llm_call("lumina", "gpt-4o", 42, Status::Success, None);

        let contents = std::fs::read_to_string(&logger.log_path).unwrap();
        let entry: AuditEntry = serde_json::from_str(contents.trim()).unwrap();

        assert_eq!(entry.user_id, "lumina");
        assert_eq!(entry.request_type, RequestType::Llm);
        assert_eq!(entry.target, "gpt-4o");
        assert_eq!(entry.duration_ms, 42);
        assert_eq!(entry.status, Status::Success);
        assert!(entry.error_message.is_none());
        assert!(entry.token_hash_prefix.is_none());
    }

    #[test]
    fn test_tool_call_produces_correct_fields() {
        let (logger, _dir) = temp_logger();
        logger.log_tool_call("lumina", "get_briefing", 100, Status::Success, None);

        let contents = std::fs::read_to_string(&logger.log_path).unwrap();
        let entry: AuditEntry = serde_json::from_str(contents.trim()).unwrap();

        assert_eq!(entry.request_type, RequestType::ToolCall);
        assert_eq!(entry.target, "get_briefing");
        assert_eq!(entry.duration_ms, 100);
        assert_eq!(entry.status, Status::Success);
    }

    #[test]
    fn test_auth_failure_stores_hash_not_token() {
        let raw_token = "super-secret-jwt-token-value";
        let (logger, _dir) = temp_logger();
        logger.log_auth_failure(Some(raw_token), 5);

        let contents = std::fs::read_to_string(&logger.log_path).unwrap();
        let entry: AuditEntry = serde_json::from_str(contents.trim()).unwrap();

        assert_eq!(entry.request_type, RequestType::AuthFailure);
        assert_eq!(entry.user_id, "anonymous");
        assert_eq!(entry.status, Status::Error);

        // Token hash prefix must be present.
        let hash_prefix = entry.token_hash_prefix.as_ref().unwrap();
        assert_eq!(hash_prefix.len(), 8, "hash prefix must be 8 hex chars");

        // Must match what token_hash_prefix() computes.
        assert_eq!(hash_prefix, &token_hash_prefix(raw_token));

        // Raw token must NOT appear anywhere in the serialised line.
        assert!(!contents.contains(raw_token), "raw token must not appear in audit log");
    }

    #[test]
    fn test_sensitive_content_not_logged() {
        let (logger, _dir) = temp_logger();

        // Simulate a tool call — arguments are not part of the entry at all.
        let sensitive_args = r#"{"password":"hunter2","api_key":"sk-abc123"}"#;
        let entry = AuditEntry::success("lumina", RequestType::ToolCall, "do_something", 30);

        logger.log_entry(&entry);

        let contents = std::fs::read_to_string(&logger.log_path).unwrap();
        assert!(!contents.contains(sensitive_args));
        assert!(!contents.contains("hunter2"));
        assert!(!contents.contains("sk-abc123"));
        // Only metadata fields are present.
        assert!(contents.contains("tool_call"));
        assert!(contents.contains("do_something"));
    }

    #[test]
    fn test_anonymous_when_jwt_missing() {
        let entry = AuditEntry::auth_failure(None, 0);
        assert_eq!(entry.user_id, "anonymous");
        assert!(entry.token_hash_prefix.is_none());
    }

    // ── Log rotation ──────────────────────────────────────────────────────────

    #[test]
    fn test_log_rotation_renames_files() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("chord-audit.jsonl");
        let logger = AuditLogger::new(log_path.clone());

        // Write one entry to create the file.
        logger.log_llm_call("lumina", "test-model", 1, Status::Success, None);
        assert!(log_path.exists());

        // Manually trigger rotation.
        logger.rotate().unwrap();

        // Active file should be gone (moved to .1).
        assert!(!log_path.exists());
        let rotated = dir.path().join("chord-audit.1.jsonl");
        assert!(rotated.exists(), "rotated file should exist at .1");
    }

    #[test]
    fn test_rotation_keeps_at_most_10_files() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("chord-audit.jsonl");
        let logger = AuditLogger::new(log_path.clone());

        // Pre-create 10 rotated files.
        for i in 1..=10u32 {
            let p = dir.path().join(format!("chord-audit.{i}.jsonl"));
            std::fs::write(&p, format!("rotation-{i}")).unwrap();
        }

        // Write active file.
        std::fs::write(&log_path, "active").unwrap();
        logger.rotate().unwrap();

        // File .11 must not exist.
        let overflow = dir.path().join("chord-audit.11.jsonl");
        assert!(!overflow.exists(), "no more than 10 rotated files allowed");

        // File .10 should exist (shifted from .9).
        let tenth = dir.path().join("chord-audit.10.jsonl");
        assert!(tenth.exists());
    }

    // ── Daily summary ─────────────────────────────────────────────────────────

    #[test]
    fn test_daily_summary_aggregates_correctly() {
        let (logger, _dir) = temp_logger();

        logger.log_llm_call("lumina", "gpt-4o", 10, Status::Success, None);
        logger.log_tool_call("lumina", "ping", 5, Status::Success, None);
        logger.log_tool_call("axon", "ping", 5, Status::Error, Some("timeout".into()));
        logger.log_auth_failure(None, 1);

        let summary = logger.daily_summary();
        assert_eq!(summary.total, 4);

        // by_type
        assert_eq!(summary.by_type.get("llm").copied().unwrap_or(0), 1);
        assert_eq!(summary.by_type.get("tool_call").copied().unwrap_or(0), 2);
        assert_eq!(summary.by_type.get("auth_failure").copied().unwrap_or(0), 1);

        // by_user is intentionally not in AuditSummary (privacy: unauthenticated endpoint)

        // by_status
        assert_eq!(summary.by_status.get("success").copied().unwrap_or(0), 2);
        assert_eq!(summary.by_status.get("error").copied().unwrap_or(0), 2);
    }

    // ── No hardcoded paths ────────────────────────────────────────────────────

    #[test]
    #[serial]
    fn test_audit_path_from_env_var() {
        let dir = tempfile::tempdir().unwrap();
        let dir_str = dir.path().to_str().unwrap();
        std::env::set_var("CHORD_AUDIT_PATH", dir_str);

        let logger = AuditLogger::from_env();
        assert_eq!(logger.log_path, dir.path().join("chord-audit.jsonl"));

        std::env::remove_var("CHORD_AUDIT_PATH");
    }

    // ── RFC-3339 helpers ──────────────────────────────────────────────────────

    #[test]
    fn test_rfc3339_roundtrip() {
        // Known epoch second: 2024-01-15T12:34:56Z = 1705322096
        // Verified: 19737 days * 86400 + (12*3600 + 34*60 + 56) = 1705276800 + 45296 = 1705322096
        let formatted = secs_to_rfc3339(1705322096);
        assert_eq!(formatted, "2024-01-15T12:34:56Z");
        let parsed = parse_rfc3339_secs(&formatted).unwrap();
        assert_eq!(parsed, 1705322096);
    }

    // ── Token hash ────────────────────────────────────────────────────────────

    #[test]
    fn test_token_hash_prefix_length_and_determinism() {
        let h1 = token_hash_prefix("test-token");
        let h2 = token_hash_prefix("test-token");
        let h3 = token_hash_prefix("different-token");

        assert_eq!(h1.len(), 8);
        assert_eq!(h1, h2, "must be deterministic");
        assert_ne!(h1, h3, "different tokens → different hashes");
    }

    // ── JSONL format ──────────────────────────────────────────────────────────

    #[test]
    fn test_multiple_entries_one_per_line() {
        let (logger, _dir) = temp_logger();
        logger.log_llm_call("u1", "model-a", 10, Status::Success, None);
        logger.log_llm_call("u2", "model-b", 20, Status::Error, Some("err".into()));

        let contents = std::fs::read_to_string(&logger.log_path).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 2, "each entry must be on its own line");

        for line in &lines {
            let _: AuditEntry = serde_json::from_str(line)
                .expect("each line must be valid JSON");
        }
    }

    // ── Directory creation ────────────────────────────────────────────────────

    #[test]
    fn test_creates_directory_if_missing() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("a").join("b").join("c");
        let log_path = nested.join("chord-audit.jsonl");
        let logger = AuditLogger::new(log_path.clone());

        logger.log_llm_call("lumina", "model", 1, Status::Success, None);
        assert!(log_path.exists(), "log file should be created with parent dirs");
    }
}
