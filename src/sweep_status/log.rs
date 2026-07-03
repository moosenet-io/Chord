//! JSONL snapshot log: daily-rotated files + age-based retention.
//!
//! ## Rotation design (and why)
//! At a 30s cadence this is ~2,880 entries/day and ~28,800 over the full
//! 10-day retention window. Two designs were considered:
//!
//! 1. **Single file, trim-on-write.** Every tick reads however many lines
//!    have accumulated, drops everything older than the retention window, and
//!    rewrites the file. That's an O(n) read+rewrite *every 30 seconds*
//!    against a file that's ~28,800 lines by day 10 — real, pointless I/O
//!    that grows with retention, for a purely time-based trim.
//! 2. **Daily-rotated files, delete-whole-file when too old (chosen).** Each
//!    day's entries live in their own `sweep-status-YYYY-MM-DD.jsonl`.
//!    Writing is a pure O(1) append to today's file. Retention is an O(files)
//!    `read_dir` + delete-by-date-in-filename — cheap, and only needs to run
//!    occasionally (this implementation runs it once per tick anyway since
//!    `read_dir` over ~10 files is negligible, but it could be moved to
//!    "once per day" with zero change to correctness).
//!
//! Design (2) is the clear winner here: no rewriting of already-written data,
//! retention is a filesystem-metadata operation instead of a data-rewrite, and
//! history queries (`?hours=N`) only need to open the handful of files whose
//! date range overlaps the query instead of scanning one giant file.

use chrono::{DateTime, NaiveDate, Utc};
use std::path::PathBuf;
#[cfg(test)]
use std::path::Path;
use tokio::io::AsyncWriteExt;

/// A JSONL log using the daily-rotation + age-based-file-deletion scheme
/// described above. Constructed from a single "template" path (e.g.
/// `/var/log/chord/sweep-status.jsonl`); actual files on disk are
/// `{dir}/{stem}-{YYYY-MM-DD}.{ext}`.
#[derive(Debug, Clone)]
pub struct SweepStatusLog {
    template_path: PathBuf,
    retention_days: u32,
}

impl SweepStatusLog {
    pub fn new(template_path: PathBuf, retention_days: u32) -> Self {
        SweepStatusLog { template_path, retention_days }
    }

    fn dir(&self) -> PathBuf {
        self.template_path
            .parent()
            .map(|p| p.to_path_buf())
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| PathBuf::from("."))
    }

    fn stem(&self) -> String {
        self.template_path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "sweep-status".to_string())
    }

    fn ext(&self) -> String {
        self.template_path
            .extension()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "jsonl".to_string())
    }

    /// The path of the daily file for `date`.
    pub fn file_for_date(&self, date: NaiveDate) -> PathBuf {
        self.dir().join(format!("{}-{}.{}", self.stem(), date.format("%Y-%m-%d"), self.ext()))
    }

    /// Append one JSON-serializable snapshot as a line to today's (i.e.
    /// `timestamp`'s date) file, creating the directory and file as needed.
    pub async fn append<T: serde::Serialize>(&self, timestamp: DateTime<Utc>, snapshot: &T) -> std::io::Result<()> {
        let dir = self.dir();
        tokio::fs::create_dir_all(&dir).await?;
        let path = self.file_for_date(timestamp.date_naive());
        let mut line = serde_json::to_string(snapshot).map_err(std::io::Error::other)?;
        line.push('\n');
        let mut file = tokio::fs::OpenOptions::new().create(true).append(true).open(&path).await?;
        file.write_all(line.as_bytes()).await?;
        Ok(())
    }

    /// Delete daily files older than the retention window (relative to `now`).
    /// Returns the number of files removed. Best-effort: an unreadable
    /// directory yields `Ok(0)` (nothing to retain yet), individual delete
    /// failures are logged and skipped rather than aborting the sweep.
    pub async fn enforce_retention(&self, now: DateTime<Utc>) -> std::io::Result<usize> {
        let dir = self.dir();
        let mut entries = match tokio::fs::read_dir(&dir).await {
            Ok(e) => e,
            Err(_) => return Ok(0),
        };
        // `retention_days` calendar days are kept, counting today as one of
        // them: today, today-1, ..., today-(retention_days-1). A file dated
        // exactly `today - retention_days` is the first day *outside* that
        // window, so it (and anything older) must be deleted. Using
        // `today - retention_days` as the cutoff below with `date < cutoff`
        // gets exactly that: with retention_days=10 the oldest surviving date
        // is `today - 9`, i.e. 10 files total, not 11.
        let cutoff = now.date_naive() - chrono::Duration::days(self.retention_days as i64 - 1);
        let stem = self.stem();
        let ext = self.ext();
        let mut removed = 0usize;
        while let Ok(Some(entry)) = entries.next_entry().await {
            let name = entry.file_name().to_string_lossy().to_string();
            if let Some(date) = parse_dated_filename(&name, &stem, &ext) {
                if date < cutoff {
                    match tokio::fs::remove_file(entry.path()).await {
                        Ok(()) => removed += 1,
                        Err(e) => tracing::warn!(
                            target: "chord.sweep_status",
                            file = %name, error = %e,
                            "failed to remove expired sweep-status log file"
                        ),
                    }
                }
            }
        }
        Ok(removed)
    }

    /// Read every snapshot line logged between `since` and `until`
    /// (inclusive), across however many daily files that range touches.
    /// Malformed lines are skipped (logged at `warn`) rather than aborting
    /// the whole read.
    pub async fn read_range(&self, since: DateTime<Utc>, until: DateTime<Utc>) -> Vec<serde_json::Value> {
        let mut out = Vec::new();
        let mut date = since.date_naive();
        let end_date = until.date_naive();
        loop {
            let path = self.file_for_date(date);
            if let Ok(content) = tokio::fs::read_to_string(&path).await {
                for line in content.lines() {
                    if line.trim().is_empty() {
                        continue;
                    }
                    match serde_json::from_str::<serde_json::Value>(line) {
                        Ok(v) => {
                            if let Some(ts) = v.get("timestamp").and_then(|t| t.as_str()) {
                                if let Ok(ts) = DateTime::parse_from_rfc3339(ts) {
                                    let ts = ts.with_timezone(&Utc);
                                    if ts >= since && ts <= until {
                                        out.push(v);
                                    }
                                    continue;
                                }
                            }
                            // No parseable timestamp field — include it rather
                            // than silently dropping data.
                            out.push(v);
                        }
                        Err(e) => tracing::warn!(
                            target: "chord.sweep_status",
                            error = %e,
                            "skipping malformed sweep-status log line"
                        ),
                    }
                }
            }
            if date >= end_date {
                break;
            }
            date = date.succ_opt().unwrap_or(end_date);
        }
        out
    }

    /// Read the single most recent snapshot (the latest line of the latest
    /// non-empty daily file, searching back up to `retention_days` days).
    /// `None` if nothing has been logged yet.
    pub async fn read_latest(&self, now: DateTime<Utc>) -> Option<serde_json::Value> {
        let mut date = now.date_naive();
        for _ in 0..=self.retention_days {
            let path = self.file_for_date(date);
            if let Ok(content) = tokio::fs::read_to_string(&path).await {
                if let Some(last_line) = content.lines().rev().find(|l| !l.trim().is_empty()) {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(last_line) {
                        return Some(v);
                    }
                }
            }
            match date.pred_opt() {
                Some(d) => date = d,
                None => break,
            }
        }
        None
    }
}

/// Parse `"{stem}-YYYY-MM-DD.{ext}"` into the embedded date. `None` for any
/// filename that doesn't match this log's naming scheme (e.g. an unrelated
/// file the operator dropped in the same directory).
fn parse_dated_filename(name: &str, stem: &str, ext: &str) -> Option<NaiveDate> {
    let prefix = format!("{stem}-");
    let suffix = format!(".{ext}");
    let mid = name.strip_prefix(prefix.as_str())?.strip_suffix(suffix.as_str())?;
    NaiveDate::parse_from_str(mid, "%Y-%m-%d").ok()
}

#[cfg(test)]
fn path_from(dir: &Path, stem: &str, ext: &str) -> PathBuf {
    dir.join(format!("{stem}.{ext}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn tmpl(dir: &Path) -> PathBuf {
        path_from(dir, "sweep-status", "jsonl")
    }

    #[test]
    fn parses_dated_filename() {
        assert_eq!(
            parse_dated_filename("sweep-status-2026-07-02.jsonl", "sweep-status", "jsonl"),
            Some(NaiveDate::from_ymd_opt(2026, 7, 2).unwrap())
        );
    }

    #[test]
    fn rejects_unrelated_filenames() {
        assert_eq!(parse_dated_filename("sweep-status.jsonl", "sweep-status", "jsonl"), None);
        assert_eq!(parse_dated_filename("other-file-2026-07-02.jsonl", "sweep-status", "jsonl"), None);
        assert_eq!(parse_dated_filename("sweep-status-2026-07-02.txt", "sweep-status", "jsonl"), None);
        assert_eq!(parse_dated_filename("sweep-status-not-a-date.jsonl", "sweep-status", "jsonl"), None);
    }

    #[tokio::test]
    async fn append_writes_to_dated_file() {
        let tmp = tempfile::tempdir().unwrap();
        let log = SweepStatusLog::new(tmpl(tmp.path()), 10);
        let ts = Utc.with_ymd_and_hms(2026, 7, 2, 12, 0, 0).unwrap();
        log.append(ts, &serde_json::json!({"timestamp": ts.to_rfc3339(), "n": 1})).await.unwrap();
        log.append(ts, &serde_json::json!({"timestamp": ts.to_rfc3339(), "n": 2})).await.unwrap();

        let path = log.file_for_date(ts.date_naive());
        let content = tokio::fs::read_to_string(&path).await.unwrap();
        assert_eq!(content.lines().count(), 2);
    }

    #[tokio::test]
    async fn read_latest_returns_most_recent_line() {
        let tmp = tempfile::tempdir().unwrap();
        let log = SweepStatusLog::new(tmpl(tmp.path()), 10);
        let ts = Utc.with_ymd_and_hms(2026, 7, 2, 12, 0, 0).unwrap();
        log.append(ts, &serde_json::json!({"timestamp": ts.to_rfc3339(), "n": 1})).await.unwrap();
        let ts2 = ts + chrono::Duration::seconds(30);
        log.append(ts2, &serde_json::json!({"timestamp": ts2.to_rfc3339(), "n": 2})).await.unwrap();

        let latest = log.read_latest(ts2).await.unwrap();
        assert_eq!(latest.get("n").and_then(|v| v.as_i64()), Some(2));
    }

    #[tokio::test]
    async fn read_latest_searches_back_across_days() {
        let tmp = tempfile::tempdir().unwrap();
        let log = SweepStatusLog::new(tmpl(tmp.path()), 10);
        let yesterday = Utc.with_ymd_and_hms(2026, 7, 1, 23, 59, 0).unwrap();
        log.append(yesterday, &serde_json::json!({"timestamp": yesterday.to_rfc3339(), "n": 1})).await.unwrap();

        let today = Utc.with_ymd_and_hms(2026, 7, 2, 0, 5, 0).unwrap();
        let latest = log.read_latest(today).await.unwrap();
        assert_eq!(latest.get("n").and_then(|v| v.as_i64()), Some(1));
    }

    #[tokio::test]
    async fn read_latest_none_when_nothing_logged() {
        let tmp = tempfile::tempdir().unwrap();
        let log = SweepStatusLog::new(tmpl(tmp.path()), 10);
        assert!(log.read_latest(Utc::now()).await.is_none());
    }

    #[tokio::test]
    async fn read_range_filters_by_timestamp() {
        let tmp = tempfile::tempdir().unwrap();
        let log = SweepStatusLog::new(tmpl(tmp.path()), 10);
        let base = Utc.with_ymd_and_hms(2026, 7, 2, 0, 0, 0).unwrap();
        for i in 0..5 {
            let ts = base + chrono::Duration::hours(i);
            log.append(ts, &serde_json::json!({"timestamp": ts.to_rfc3339(), "n": i})).await.unwrap();
        }
        // Window covering hours 1..=3 inclusive.
        let since = base + chrono::Duration::hours(1);
        let until = base + chrono::Duration::hours(3);
        let rows = log.read_range(since, until).await;
        let ns: Vec<i64> = rows.iter().filter_map(|v| v.get("n").and_then(|n| n.as_i64())).collect();
        assert_eq!(ns, vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn read_range_spans_multiple_days() {
        let tmp = tempfile::tempdir().unwrap();
        let log = SweepStatusLog::new(tmpl(tmp.path()), 10);
        let day1 = Utc.with_ymd_and_hms(2026, 7, 1, 23, 0, 0).unwrap();
        let day2 = Utc.with_ymd_and_hms(2026, 7, 2, 1, 0, 0).unwrap();
        log.append(day1, &serde_json::json!({"timestamp": day1.to_rfc3339(), "n": 1})).await.unwrap();
        log.append(day2, &serde_json::json!({"timestamp": day2.to_rfc3339(), "n": 2})).await.unwrap();

        let rows = log.read_range(day1 - chrono::Duration::hours(1), day2 + chrono::Duration::hours(1)).await;
        assert_eq!(rows.len(), 2);
    }

    #[tokio::test]
    async fn enforce_retention_deletes_old_files_keeps_recent() {
        let tmp = tempfile::tempdir().unwrap();
        let log = SweepStatusLog::new(tmpl(tmp.path()), 10);
        let now = Utc.with_ymd_and_hms(2026, 7, 2, 12, 0, 0).unwrap();

        let old_date = (now - chrono::Duration::days(15)).date_naive();
        let recent_date = (now - chrono::Duration::days(2)).date_naive();
        let old_path = log.file_for_date(old_date);
        let recent_path = log.file_for_date(recent_date);
        tokio::fs::create_dir_all(tmp.path()).await.unwrap();
        tokio::fs::write(&old_path, "{}\n").await.unwrap();
        tokio::fs::write(&recent_path, "{}\n").await.unwrap();

        let removed = log.enforce_retention(now).await.unwrap();
        assert_eq!(removed, 1);
        assert!(!tokio::fs::try_exists(&old_path).await.unwrap());
        assert!(tokio::fs::try_exists(&recent_path).await.unwrap());
    }

    #[tokio::test]
    async fn enforce_retention_keeps_exactly_retention_days_files() {
        // Off-by-one regression test: with retention_days=10, exactly 10
        // daily files must survive (today through today-9 inclusive), and
        // the 11th-oldest (today-10) must be deleted.
        let tmp = tempfile::tempdir().unwrap();
        let retention_days = 10u32;
        let log = SweepStatusLog::new(tmpl(tmp.path()), retention_days);
        let now = Utc.with_ymd_and_hms(2026, 7, 2, 12, 0, 0).unwrap();
        tokio::fs::create_dir_all(tmp.path()).await.unwrap();

        // Create one file per day from today back 12 days (13 files total),
        // so the test also exercises files well past the retention window.
        let mut paths = Vec::new();
        for days_ago in 0..=12i64 {
            let date = (now - chrono::Duration::days(days_ago)).date_naive();
            let path = log.file_for_date(date);
            tokio::fs::write(&path, "{}\n").await.unwrap();
            paths.push((days_ago, path));
        }

        log.enforce_retention(now).await.unwrap();

        let mut survivors = 0usize;
        for (days_ago, path) in &paths {
            let exists = tokio::fs::try_exists(path).await.unwrap();
            if *days_ago < retention_days as i64 {
                assert!(exists, "day -{days_ago} should survive (within {retention_days}-day window)");
            } else {
                assert!(!exists, "day -{days_ago} should be deleted (outside {retention_days}-day window)");
            }
            if exists {
                survivors += 1;
            }
        }
        assert_eq!(survivors, retention_days as usize, "exactly retention_days files must survive");
    }

    #[tokio::test]
    async fn enforce_retention_ignores_unrelated_files() {
        let tmp = tempfile::tempdir().unwrap();
        let log = SweepStatusLog::new(tmpl(tmp.path()), 10);
        tokio::fs::create_dir_all(tmp.path()).await.unwrap();
        let unrelated = tmp.path().join("README.md");
        tokio::fs::write(&unrelated, "hello").await.unwrap();

        let removed = log.enforce_retention(Utc::now()).await.unwrap();
        assert_eq!(removed, 0);
        assert!(tokio::fs::try_exists(&unrelated).await.unwrap());
    }

    #[tokio::test]
    async fn enforce_retention_on_missing_dir_is_noop() {
        let log = SweepStatusLog::new(PathBuf::from("/nonexistent/chord/sweep-status.jsonl"), 10);
        let removed = log.enforce_retention(Utc::now()).await.unwrap();
        assert_eq!(removed, 0);
    }
}
