//! Postgres polling for the two intake-sweep tables.
//!
//! Read-only, best-effort: every query failure is caught and turned into an
//! `unavailable` [`SweepDbStats`] rather than propagated — a briefly
//! unreachable Postgres must never crash chord or stall the monitor tick.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::Row;

/// Static description of one sweep's result table: which timestamp column
/// marks "when this row landed", and (optionally) which column records a
/// per-row error so an error-rate can be computed. `error_column: None` means
/// the table has no error column (e.g. `assistant_profile_run` today) — the
/// resulting stats simply omit `errors_last_1h`/`error_rate_percent` rather
/// than inventing a column that doesn't exist.
#[derive(Debug, Clone, Copy)]
pub struct SweepTableSpec {
    pub table: &'static str,
    pub ts_column: &'static str,
    pub error_column: Option<&'static str>,
}

/// The coder-sweep table (`intake-coder-sweep.service` → `code_profile_runs`).
/// Confirmed live in `lumina_intake` with an `error` (text, nullable) column.
pub const CODER_TABLE: SweepTableSpec = SweepTableSpec {
    table: "code_profile_runs",
    ts_column: "created_at",
    error_column: Some("error"),
};

/// The assistant-sweep table (`intake-assistant-sweep.service` →
/// `assistant_profile_run`). Confirmed to exist in `lumina_intake` (S84/ASMT
/// schema) but as of this writing carries no per-row error column and only a
/// handful of rows — the S84 runtime-gap note in project memory records that
/// the assistant sweep has not actually been run continuously yet. Monitoring
/// it is still correct: an idle/never-populated table should show up as
/// `idle`/stale via the systemd-unit + row-age signals, not be hidden.
pub const ASSISTANT_TABLE: SweepTableSpec = SweepTableSpec {
    table: "assistant_profile_run",
    ts_column: "started_at",
    error_column: None,
};

/// One sweep's DB-derived stats for a single poll tick.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SweepDbStats {
    pub available: bool,
    pub error_message: Option<String>,
    pub latest_row_ts: Option<DateTime<Utc>>,
    /// Seconds between now and `latest_row_ts`; `None` iff `latest_row_ts` is
    /// `None` (unavailable, or the table has never had a row).
    pub latest_row_age_secs: Option<i64>,
    pub rows_last_5min: Option<i64>,
    pub rows_last_1h: Option<i64>,
    /// `None` when the table has no error column ([`SweepTableSpec::error_column`]).
    pub errors_last_1h: Option<i64>,
    /// `None` when `errors_last_1h` is `None`, or `rows_last_1h` is zero
    /// (avoids a division-by-zero "0/0 = 100%" artifact).
    pub error_rate_percent: Option<f64>,
}

impl SweepDbStats {
    fn unavailable(message: impl Into<String>) -> Self {
        SweepDbStats {
            available: false,
            error_message: Some(message.into()),
            ..Default::default()
        }
    }
}

/// Query one sweep table's stats. Never panics or propagates an error — a
/// query failure (unreachable DB, missing table, permission error) yields
/// `SweepDbStats { available: false, .. }` with the error logged at `warn`.
pub async fn query_sweep_stats(pool: &sqlx::PgPool, spec: SweepTableSpec) -> SweepDbStats {
    let error_select = match spec.error_column {
        Some(col) => format!(
            ", count(*) FILTER (WHERE {ts} > now() - interval '1 hour' AND {col} IS NOT NULL) AS errors_1h",
            ts = spec.ts_column,
            col = col
        ),
        None => String::new(),
    };
    let sql = format!(
        "SELECT max({ts}) AS latest, \
                count(*) FILTER (WHERE {ts} > now() - interval '5 minutes') AS last_5m, \
                count(*) FILTER (WHERE {ts} > now() - interval '1 hour') AS last_1h \
                {error_select} \
         FROM {table}",
        ts = spec.ts_column,
        table = spec.table,
    );

    let row = match sqlx::query(&sql).fetch_one(pool).await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(target: "chord.sweep_status", table = spec.table, error = %e, "sweep DB query failed");
            return SweepDbStats::unavailable(e.to_string());
        }
    };

    let latest_row_ts: Option<DateTime<Utc>> = row.try_get("latest").ok();
    let latest_row_age_secs = latest_row_ts.map(|ts| (Utc::now() - ts).num_seconds().max(0));
    let rows_last_5min: Option<i64> = row.try_get("last_5m").ok();
    let rows_last_1h: Option<i64> = row.try_get("last_1h").ok();
    let errors_last_1h: Option<i64> = if spec.error_column.is_some() {
        row.try_get("errors_1h").ok()
    } else {
        None
    };
    let error_rate_percent = match (errors_last_1h, rows_last_1h) {
        (Some(errs), Some(total)) if total > 0 => Some((errs as f64 / total as f64) * 100.0),
        _ => None,
    };

    SweepDbStats {
        available: true,
        error_message: None,
        latest_row_ts,
        latest_row_age_secs,
        rows_last_5min,
        rows_last_1h,
        errors_last_1h,
        error_rate_percent,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unavailable_stats_have_no_data() {
        let s = SweepDbStats::unavailable("connection refused");
        assert!(!s.available);
        assert_eq!(s.error_message.as_deref(), Some("connection refused"));
        assert!(s.latest_row_ts.is_none());
        assert!(s.latest_row_age_secs.is_none());
    }

    #[test]
    fn table_specs_are_the_confirmed_schema() {
        assert_eq!(CODER_TABLE.table, "code_profile_runs");
        assert_eq!(CODER_TABLE.ts_column, "created_at");
        assert_eq!(CODER_TABLE.error_column, Some("error"));

        assert_eq!(ASSISTANT_TABLE.table, "assistant_profile_run");
        assert_eq!(ASSISTANT_TABLE.ts_column, "started_at");
        assert_eq!(ASSISTANT_TABLE.error_column, None);
    }
}
