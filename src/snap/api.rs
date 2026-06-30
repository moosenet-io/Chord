//! SNAP read-only observability HTTP API (additive control-plane routes).
//!
//! These handlers attach to chord's existing control router and reuse chord's
//! own JWT auth (`auth_check` / `auth_error_response` from `routes.rs`) — the
//! same gate as `/api/models`. They expose the SNAP subsystem on **new** paths
//! that do not collide with any existing chord route:
//!
//! | Method | Path                       | Source            |
//! |--------|----------------------------|-------------------|
//! | GET    | `/api/vram`                | [`crate::snap::vram`] via `SHARED_STATE` |
//! | GET    | `/api/activity`            | [`crate::snap::activity`] |
//! | GET    | `/api/inventory`           | [`crate::snap::inventory`] (env-configured locations) |
//! | GET    | `/api/analytics/requests`  | [`crate::snap::analytics`] request log |
//! | GET    | `/api/analytics/cost`      | daily imputed cost breakdown |
//! | GET    | `/api/analytics/savings`   | savings summary vs cloud pricing |
//!
//! Mutating lifecycle / config endpoints from harmony-chord are intentionally
//! NOT ported (see `snap` module docs).

use std::sync::Arc;

use axum::{
    extract::{Query, State},
    http::HeaderMap,
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use chrono::{DateTime, Utc};
use serde::Deserialize;

use crate::routes::{auth_check, auth_error_response, AppState};
use crate::snap::activity::ActivityTracker;
use crate::snap::analytics::RequestLogger;
use crate::snap::config::SnapConfig;
use crate::snap::inventory::ModelInventory;
use crate::snap::SHARED_STATE;

/// Merge the SNAP read-only routes into the supplied control router.
///
/// All routes share chord's `Arc<AppState>` for auth, so the returned router
/// can be `.merge()`d straight into `build_control_router`'s router.
pub fn snap_routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/vram", get(get_vram))
        .route("/api/activity", get(get_activity))
        .route("/api/inventory", get(get_inventory))
        .route("/api/analytics/requests", get(get_requests))
        .route("/api/analytics/cost", get(get_cost))
        .route("/api/analytics/savings", get(get_savings))
}

// ── GET /api/vram ────────────────────────────────────────────────────────────

/// Current GPU VRAM snapshot (total / used / free + per-model allocations).
pub async fn get_vram(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if let Err(e) = auth_check(&headers, &state.jwt_secret) {
        return auth_error_response(e);
    }
    let s = SHARED_STATE.read().await;
    Json(serde_json::json!(s.vram)).into_response()
}

// ── GET /api/activity ────────────────────────────────────────────────────────

/// Per-engine, per-model active-request observation.
pub async fn get_activity(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if let Err(e) = auth_check(&headers, &state.jwt_secret) {
        return auth_error_response(e);
    }
    let tracker = ActivityTracker::new(SHARED_STATE.clone());
    let grouped = tracker.activity_by_engine().await;
    // ModelActivity isn't Serialize (internal struct); project to JSON.
    let by_engine: serde_json::Map<String, serde_json::Value> = grouped
        .into_iter()
        .map(|(engine, models)| {
            let arr: Vec<serde_json::Value> = models
                .into_iter()
                .map(|m| {
                    serde_json::json!({
                        "model": m.model,
                        "engine": m.engine,
                        "active_requests": m.active_requests,
                        "last_seen": m.last_seen,
                    })
                })
                .collect();
            (engine, serde_json::Value::Array(arr))
        })
        .collect();
    Json(serde_json::json!({ "by_engine": by_engine })).into_response()
}

// ── GET /api/inventory ───────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct InventoryQuery {
    pub tier: Option<String>,
    pub loaded: Option<bool>,
    pub search: Option<String>,
}

/// Scan configured storage locations for GGUF / Ollama models.
pub async fn get_inventory(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(q): Query<InventoryQuery>,
) -> Response {
    if let Err(e) = auth_check(&headers, &state.jwt_secret) {
        return auth_error_response(e);
    }
    let cfg = SnapConfig::from_env();
    // Scanning touches the filesystem; keep it off the async reactor.
    let inv =
        tokio::task::spawn_blocking(move || ModelInventory::scan(&cfg.storage_locations))
            .await
            .unwrap_or_default();
    let records = inv.filter(q.tier.as_deref(), q.loaded, q.search.as_deref());
    let count = records.len();
    Json(serde_json::json!({ "models": records, "count": count })).into_response()
}

// ── GET /api/analytics/requests ──────────────────────────────────────────────

#[derive(Deserialize)]
pub struct RequestsQuery {
    pub from: Option<DateTime<Utc>>,
    pub to: Option<DateTime<Utc>>,
    pub model: Option<String>,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_limit() -> usize {
    100
}

/// Paginated request log (most-recent first capped at `limit`).
pub async fn get_requests(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(q): Query<RequestsQuery>,
) -> Response {
    if let Err(e) = auth_check(&headers, &state.jwt_secret) {
        return auth_error_response(e);
    }
    let cfg = SnapConfig::from_env();
    let logger = RequestLogger::new(&cfg.data_dir);
    let limit = q.limit.min(10_000);
    let records = logger.query(q.from, q.to, q.model.as_deref(), limit);
    let count = records.len();
    Json(serde_json::json!({ "records": records, "count": count, "limit": limit }))
        .into_response()
}

// ── GET /api/analytics/cost ──────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct CostQuery {
    pub period: Option<String>,
}

/// Daily imputed cost breakdown over a period (e.g. `?period=7d`).
pub async fn get_cost(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(q): Query<CostQuery>,
) -> Response {
    if let Err(e) = auth_check(&headers, &state.jwt_secret) {
        return auth_error_response(e);
    }
    let period = q.period.as_deref().unwrap_or("7d");
    let period_days: u64 = parse_period_days(period);
    let cfg = SnapConfig::from_env();
    let logger = RequestLogger::new(&cfg.data_dir);
    let costs = logger.daily_costs(period_days);
    Json(serde_json::json!({
        "period": period,
        "period_days": period_days,
        "daily_costs": costs,
    }))
    .into_response()
}

// ── GET /api/analytics/savings ───────────────────────────────────────────────

#[derive(Deserialize)]
pub struct SavingsQuery {
    pub period: Option<String>,
}

/// Imputed savings vs representative cloud pricing over a period.
pub async fn get_savings(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(q): Query<SavingsQuery>,
) -> Response {
    if let Err(e) = auth_check(&headers, &state.jwt_secret) {
        return auth_error_response(e);
    }
    let period = q.period.as_deref().unwrap_or("30d");
    let cfg = SnapConfig::from_env();
    let logger = RequestLogger::new(&cfg.data_dir);
    let summary = logger.savings_summary(period);
    Json(serde_json::json!(summary)).into_response()
}

fn parse_period_days(period: &str) -> u64 {
    if let Some(days) = period.strip_suffix('d') {
        days.parse().unwrap_or(7)
    } else if let Some(weeks) = period.strip_suffix('w') {
        weeks.parse::<u64>().unwrap_or(1) * 7
    } else {
        7
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_limit_is_100() {
        assert_eq!(default_limit(), 100);
    }

    #[test]
    fn period_parsing() {
        assert_eq!(parse_period_days("7d"), 7);
        assert_eq!(parse_period_days("30d"), 30);
        assert_eq!(parse_period_days("2w"), 14);
        assert_eq!(parse_period_days("bogus"), 7);
    }
}
