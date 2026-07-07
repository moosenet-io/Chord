//! SNAP — observability & inventory subsystem ported from harmony-chord.
//!
//! This module is **purely additive** to chord-proxy. It contributes the
//! unique "SNAP" features that lived only in harmony-chord, reconciled into the
//! published chord crate without removing or duplicating any existing chord
//! path:
//!
//! | SNAP item | Module          | What it adds |
//! |-----------|-----------------|--------------|
//! | SNAP-02   | [`vram`]        | Real GPU VRAM reader (sysfs / rocm-smi / Ollama). chord's `serving::memory_model` is *accounting*; this is the actual device read. |
//! | SNAP-02   | [`health`]      | Background engine health poller that populates the shared [`state::InferenceState`] (data source for vram/activity). |
//! | SNAP-03   | [`inventory`]   | On-disk GGUF + Ollama manifest scanner with quant detection and cleanup candidates. |
//! | SNAP-04   | [`activity`]    | Passive in-use observation per model/engine. |
//! | SNAP-05   | [`analytics`]   | `RequestLogger`: append-only request log + imputed cloud-cost / savings. |
//! | VLLM-01   | [`vllm`]        | vLLM `EngineAdapter` backend option (gfx1151 container lifecycle). |
//!
//! ## Overlap decisions (harmony-chord → chord)
//! - **proxy.rs (harmony streaming reverse proxy): NOT ported as a proxy.**
//!   chord already owns the request path (`routes.rs` `/v1/chat/completions`,
//!   `mcp_proxy.rs`, `routing/`). Only its *value* — the `RequestLogger`
//!   analytics hook — is ported, via [`analytics`]. Wiring the logger into the
//!   live proxy path is left for the later harmony-rewire step.
//! - **vram: additive, not a replacement.** chord's `serving::memory_model` is
//!   kept; [`vram`] is the missing *device reader* it can draw on.
//! - **api/{auth,config,storage,lifecycle,models full-CRUD}: NOT ported.**
//!   These duplicate chord's existing `/api/models`, `/api/storage`, JWT auth
//!   and config, and are coupled to harmony's `AppState`/`ApiKeyStore`. SNAP
//!   exposes only the *new* read-only observability surface ([`api`]) on
//!   distinct paths, gated by chord's own JWT `auth_check`.

use std::sync::Arc;

use async_trait::async_trait;
use once_cell::sync::Lazy;

pub mod activity;
pub mod analytics;
pub mod api;
pub mod config;
pub mod health;
pub mod inventory;
pub mod state;
pub mod storage;
pub mod vllm;
pub mod vram;

/// Minimal engine-adapter trait, vendored from harmony-chord's `lifecycle.rs`.
///
/// Only the [`vllm::VLLMAdapter`] implements it here. chord's own backend
/// launch/stop logic lives in `serving/` and is untouched; this trait exists so
/// the vLLM adapter can be offered as an additive backend option.
#[async_trait]
pub trait EngineAdapter: Send + Sync {
    async fn load_model(&self, model: &str, ctx_size: u32) -> Result<(), String>;
    async fn unload_model(&self, model: &str) -> Result<(), String>;
    async fn is_model_loaded(&self, model: &str) -> bool;
}

/// Process-global shared inference state.
///
/// Shared between the background [`health`] monitor (writer) and the SNAP
/// [`api`] read endpoints (readers), so the SNAP subsystem can be wired in
/// without modifying chord's central `AppState`.
pub static SHARED_STATE: Lazy<state::SharedInferenceState> = Lazy::new(state::new_shared_state);

/// Spawn the SNAP background health monitor, populating [`SHARED_STATE`].
///
/// Best-effort: if no engines are configured the poller simply records an empty
/// snapshot each tick. Returns immediately after spawning.
///
/// ## Persistence (default-OFF)
/// When `cfg.persist` (`CHORD_SNAP_PERSIST`) is true, this also builds ONE shared
/// intake-DB pool (via [`storage::get_pool`] — reuses the harness URL resolver,
/// no new secret), runs [`storage::migrate`] once, and hands the pool to the
/// health loop so it can persist VRAM samples (interval-gated) and activity polls.
/// When false — or when no DB URL resolves — the pool is `None` and the monitor
/// runs exactly as 1.2.0/1.3.0 (in-memory only, zero behavior change). A DB
/// failure here never blocks the monitor.
pub fn spawn_health_monitor(cfg: Arc<config::SnapConfig>) {
    let state = SHARED_STATE.clone();
    tokio::spawn(async move {
        let pool = if cfg.persist {
            match storage::get_pool().await {
                Ok(p) => match storage::migrate(&p).await {
                    Ok(()) => {
                        tracing::info!(
                            "SNAP persistence ENABLED (CHORD_SNAP_PERSIST): \
                             reusing intake DB, vram sample gate {}s",
                            cfg.vram_sample_secs
                        );
                        // SNAP-03: persist a one-shot inventory snapshot at boot
                        // when storage locations are configured (per-scan grain;
                        // inventory is cold — not on any hot path). Best-effort.
                        if !cfg.storage_locations.is_empty() {
                            let inv = inventory::ModelInventory::scan(&cfg.storage_locations);
                            match storage::insert_inventory_scan(&p, &inv.records, None).await {
                                Ok(scan_id) => tracing::info!(
                                    %scan_id,
                                    records = inv.records.len(),
                                    "SNAP inventory snapshot persisted"
                                ),
                                Err(e) => tracing::warn!(
                                    error = %e,
                                    "SNAP inventory snapshot persist failed (dropped)"
                                ),
                            }
                        }
                        Some(p)
                    }
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            "SNAP persistence migrate failed — running in-memory only"
                        );
                        None
                    }
                },
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "SNAP persistence enabled but no DB resolved — running in-memory only"
                    );
                    None
                }
            }
        } else {
            None
        };
        health::run_health_monitor(cfg, state, pool).await;
    });
}
