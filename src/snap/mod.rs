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
pub fn spawn_health_monitor(cfg: Arc<config::SnapConfig>) {
    let state = SHARED_STATE.clone();
    tokio::spawn(async move {
        health::run_health_monitor(cfg, state).await;
    });
}
