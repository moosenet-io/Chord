//! `GET /v1/sweep/status` and `GET /v1/sweep/status/history` — the fleet's
//! model-benchmarking-health observability surface.
//!
//! No auth, matching `/health` and `/v1/audit/summary` (also observability
//! endpoints, also no-auth): this surface returns aggregate sweep/GPU/Ollama
//! health, not user identities or secrets, so it's held to the same bar as
//! those two rather than the JWT-gated `/api/models*` control routes.

use axum::{
    extract::Query,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use serde::Deserialize;
use std::sync::Arc;

use super::config::SweepMonitorConfig;
use super::log::SweepStatusLog;
use super::poll::LATEST_SNAPSHOT;
use crate::routes::AppState;

/// Default history window (hours) when `?hours=` is omitted.
const DEFAULT_HISTORY_HOURS: u32 = 24;
/// Cap on the history window (hours) — 10 days, matching the log's retention.
const MAX_HISTORY_HOURS: u32 = 240;

/// Build the sweep-status routes. Merge straight into `build_router`'s
/// `Router<Arc<AppState>>` alongside `/health` / `/v1/audit/summary`.
pub fn sweep_status_routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/v1/sweep/status", get(get_status))
        .route("/v1/sweep/status/history", get(get_status_history))
}

/// `GET /v1/sweep/status` — the single latest snapshot.
///
/// 200 with the snapshot once the monitor has completed at least one tick;
/// 503 with an explanatory body before the first tick (a small, expected
/// startup window — the monitor ticks every `CHORD_SWEEP_POLL_INTERVAL_SECS`,
/// default 30s).
pub async fn get_status() -> Response {
    match LATEST_SNAPSHOT.read().await.clone() {
        Some(snapshot) => Json(snapshot).into_response(),
        None => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "status": "pending",
                "message": "sweep-status monitor has not completed its first tick yet"
            })),
        )
            .into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct HistoryQuery {
    pub hours: Option<u32>,
}

/// `GET /v1/sweep/status/history?hours=N` — matching snapshots from the
/// retained JSONL log. `hours` defaults to 24, capped at 240 (10 days, the
/// log's retention window — a caller can never get "not retained" silently
/// truncated to nothing, they just can't ask past 10 days).
pub async fn get_status_history(Query(q): Query<HistoryQuery>) -> Response {
    let hours = q.hours.unwrap_or(DEFAULT_HISTORY_HOURS).min(MAX_HISTORY_HOURS).max(1);
    let cfg = SweepMonitorConfig::from_env();
    let log = SweepStatusLog::new(cfg.log_path, cfg.retention_days);

    let until = chrono::Utc::now();
    let since = until - chrono::Duration::hours(hours as i64);
    let snapshots = log.read_range(since, until).await;

    Json(serde_json::json!({
        "hours": hours,
        "since": since.to_rfc3339(),
        "until": until.to_rfc3339(),
        "count": snapshots.len(),
        "snapshots": snapshots,
    }))
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn history_hours_default_and_cap() {
        assert_eq!(None::<u32>.unwrap_or(DEFAULT_HISTORY_HOURS), 24);
        assert_eq!(Some(1000u32).unwrap_or(DEFAULT_HISTORY_HOURS).min(MAX_HISTORY_HOURS), 240);
        assert_eq!(Some(5u32).unwrap_or(DEFAULT_HISTORY_HOURS).min(MAX_HISTORY_HOURS), 5);
    }
}
