//! HRNS-03: VRAM lifecycle — on-demand load and immediate eviction for Harness-1.
//!
//! A research query drives a four-step VRAM rotation:
//!
//! ```text
//! personality(20B) → search(harness-1) → synthesis(20B|120B) → personality(20B)
//! ```
//!
//! Each transition is a *swap* performed through Chord's existing lifecycle
//! control API (FORGE-04): `POST {control_url}/api/lifecycle/swap {model, engine}`.
//! The [`HarnessVramManager`] reuses that mechanism rather than talking to Ollama
//! directly. All model names and the engine come from config/env — nothing is
//! hardcoded outside default-fallback strings.
//!
//! ## Failure handling (graceful degradation, never a crash)
//! - `load_search` fails  → [`SwapOutcome::Fallback`]: skip Harness-1, caller
//!   falls back to a regular searxng_search.
//! - `evict_search` fails → [`SwapOutcome::Degraded`]: Harness-1 stays resident,
//!   synthesis runs on it (lower quality, still functional).
//! - `load_synthesis` fails → [`SwapOutcome::Degraded`]: synthesize on whatever
//!   model is currently loaded.
//! - `restore_default` fails → logged warning only; the next user message will
//!   trigger a swap anyway.
//!
//! ## Concurrency
//! Only one rotation may touch VRAM at a time. A [`tokio::sync::Mutex`] serialises
//! rotations: concurrent research queries are *queued*, never rejected.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::{mpsc, Mutex};

// ── Swap client abstraction (injectable for tests) ────────────────────────────

/// Errors a swap backend may surface. Kept opaque so callers only branch on
/// success vs. failure, matching the spec's "graceful fallback on any failure".
#[derive(Debug, thiserror::Error)]
pub enum SwapError {
    #[error("swap timed out after {0:?}")]
    Timeout(Duration),
    #[error("swap transport error: {0}")]
    Transport(String),
    #[error("swap returned HTTP {0}")]
    Status(u16),
}

/// Abstracts the act of making a given model resident in VRAM via Chord's
/// lifecycle control API. Implemented over HTTP in production
/// ([`HttpSwapClient`]) and mocked in tests.
#[async_trait]
pub trait SwapClient: Send + Sync {
    /// Make `model` (loaded with `engine`) the resident VRAM model. This is the
    /// `POST /api/lifecycle/swap` call; it evicts whatever was previously loaded.
    async fn swap(&self, model: &str, engine: &str) -> Result<(), SwapError>;

    /// Restore the default/personality model. This is the
    /// `POST /api/lifecycle/restore` call.
    async fn restore(&self) -> Result<(), SwapError>;
}

/// Production [`SwapClient`] that talks to Chord's lifecycle control endpoint —
/// the same API FORGE-04's `LifecycleClient` uses.
pub struct HttpSwapClient {
    client: reqwest::Client,
    control_url: String,
    api_key: String,
    swap_timeout: Duration,
}

impl HttpSwapClient {
    pub fn new(control_url: String, api_key: String, swap_timeout: Duration) -> Self {
        let client = reqwest::Client::builder()
            .timeout(swap_timeout)
            .build()
            .expect("failed to build harness VRAM HTTP client");
        Self { client, control_url, api_key, swap_timeout }
    }

    fn map_err(&self, e: reqwest::Error) -> SwapError {
        if e.is_timeout() {
            SwapError::Timeout(self.swap_timeout)
        } else {
            SwapError::Transport(e.to_string())
        }
    }
}

#[async_trait]
impl SwapClient for HttpSwapClient {
    async fn swap(&self, model: &str, engine: &str) -> Result<(), SwapError> {
        let url = format!("{}/api/lifecycle/swap", self.control_url);
        let body = serde_json::json!({ "model": model, "engine": engine });
        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| self.map_err(e))?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(SwapError::Status(resp.status().as_u16()))
        }
    }

    async fn restore(&self) -> Result<(), SwapError> {
        let url = format!("{}/api/lifecycle/restore", self.control_url);
        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await
            .map_err(|e| self.map_err(e))?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(SwapError::Status(resp.status().as_u16()))
        }
    }
}

// ── Configuration (all from env, no hardcoded infra) ──────────────────────────

/// Model names + engine + timeout for the rotation. Sourced from env in
/// [`VramConfig::from_env`]; constructible directly in tests.
#[derive(Debug, Clone)]
pub struct VramConfig {
    /// Personality / default model (e.g. `gpt-oss:20b`). Restored at end.
    pub personality_model: String,
    /// The search subagent model (e.g. `harness-1`, alias `lumina-search`).
    pub search_model: String,
    /// Synthesis model when *not* deep (typically the personality 20B).
    pub synthesis_model: String,
    /// Synthesis model when deep (e.g. `gpt-oss:120b`).
    pub synthesis_deep_model: String,
    /// Ollama engine identifier for swaps.
    pub engine: String,
    /// Per-swap timeout (default 10s per spec).
    pub swap_timeout: Duration,
}

impl VramConfig {
    /// Build from environment. Falls back to documented defaults; the defaults
    /// are only ever fallback strings, never hardcoded infra (IPs/URLs).
    pub fn from_env() -> Self {
        let personality_model = std::env::var("CHORD_FAST_MODEL")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "gpt-oss:20b".to_string());
        let search_model = std::env::var("HARNESS_SEARCH_MODEL")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "harness-1".to_string());
        // Non-deep synthesis defaults to the personality model.
        let synthesis_model = std::env::var("HARNESS_SYNTHESIS_MODEL")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| personality_model.clone());
        // Deep synthesis defaults to the existing deep-model config.
        let synthesis_deep_model = std::env::var("CHORD_DEEP_MODEL")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "gpt-oss:120b".to_string());
        let engine = std::env::var("CHORD_SWAP_ENGINE")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "ollama_gpu".to_string());
        let swap_timeout = std::env::var("HARNESS_SWAP_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .map(Duration::from_secs)
            .unwrap_or_else(|| Duration::from_secs(10));
        Self {
            personality_model,
            search_model,
            synthesis_model,
            synthesis_deep_model,
            engine,
            swap_timeout,
        }
    }
}

// ── Streaming progress events ─────────────────────────────────────────────────

/// Phase of the rotation, surfaced to the streaming progress channel so the UI
/// can show "Loading research agent..." etc.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VramPhase {
    /// Swapping Harness-1 in. Message: "Loading research agent...".
    LoadingSearch,
    /// Evicting Harness-1 and loading synthesis. Message:
    /// "Research complete. Preparing analysis...".
    PreparingSynthesis,
    /// Restoring the personality model. Message: "Ready.".
    Restoring,
}

impl VramPhase {
    /// Human-facing progress text per the spec.
    pub fn message(&self) -> &'static str {
        match self {
            VramPhase::LoadingSearch => "Loading research agent...",
            VramPhase::PreparingSynthesis => "Research complete. Preparing analysis...",
            VramPhase::Restoring => "Ready.",
        }
    }
}

/// Outcome of a swap step — drives caller behaviour without ever being an error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SwapOutcome {
    /// Swap succeeded; the requested model is resident.
    Ok,
    /// Swap was a no-op because the model was already resident (warm).
    AlreadyWarm,
    /// Load failed; caller should fall back to the non-harness path.
    Fallback,
    /// Eviction/load failed but execution can continue on the currently loaded
    /// model (degraded quality, not a crash).
    Degraded,
}

// ── HarnessVramManager ────────────────────────────────────────────────────────

/// Drives the four-step VRAM rotation for a research query. Cloneable (cheap —
/// everything behind `Arc`) and safe to share across tasks; the internal mutex
/// serialises rotations so only one runs at a time.
#[derive(Clone)]
pub struct HarnessVramManager {
    client: Arc<dyn SwapClient>,
    config: Arc<VramConfig>,
    /// Serialises rotations. Concurrent callers queue rather than fail.
    rotation_lock: Arc<Mutex<()>>,
    /// Tracks the model currently believed resident, so back-to-back research
    /// queries can skip a redundant load ("model already warm").
    current_model: Arc<Mutex<String>>,
    /// Optional streaming progress sink.
    progress: Option<mpsc::UnboundedSender<VramPhase>>,
}

impl HarnessVramManager {
    /// Construct with an injected [`SwapClient`] and [`VramConfig`].
    pub fn new(client: Arc<dyn SwapClient>, config: VramConfig) -> Self {
        let personality = config.personality_model.clone();
        Self {
            client,
            config: Arc::new(config),
            rotation_lock: Arc::new(Mutex::new(())),
            current_model: Arc::new(Mutex::new(personality)),
            progress: None,
        }
    }

    /// Build the production manager from environment. Returns `None` when the
    /// lifecycle control API is not configured (`CHORD_CONTROL_URL` /
    /// `CHORD_API_KEY` unset) — mirroring FORGE-04's `from_env` contract.
    pub fn from_env() -> Option<Self> {
        let control_url = std::env::var("CHORD_CONTROL_URL")
            .ok()
            .filter(|s| !s.is_empty())?;
        let api_key = std::env::var("CHORD_API_KEY").ok().filter(|s| !s.is_empty())?;
        let config = VramConfig::from_env();
        let client = HttpSwapClient::new(control_url, api_key, config.swap_timeout);
        Some(Self::new(Arc::new(client), config))
    }

    /// Attach a streaming progress channel; phase events are emitted as the
    /// rotation advances.
    pub fn with_progress(mut self, tx: mpsc::UnboundedSender<VramPhase>) -> Self {
        self.progress = Some(tx);
        self
    }

    fn emit(&self, phase: VramPhase) {
        if let Some(tx) = &self.progress {
            // Best-effort: a closed receiver must never abort a swap.
            let _ = tx.send(phase);
        }
    }

    /// Acquire the rotation lock. Held for the duration of a guard's lifetime;
    /// callers should hold it across the full search→synthesis→restore sequence
    /// so the whole rotation is atomic w.r.t. other rotations.
    pub async fn acquire(&self) -> tokio::sync::MutexGuard<'_, ()> {
        self.rotation_lock.lock().await
    }

    /// Run `client.swap` under the per-swap timeout, mapping a timeout to
    /// [`SwapError::Timeout`].
    async fn timed_swap(&self, model: &str) -> Result<(), SwapError> {
        let fut = self.client.swap(model, &self.config.engine);
        match tokio::time::timeout(self.config.swap_timeout, fut).await {
            Ok(res) => res,
            Err(_) => Err(SwapError::Timeout(self.config.swap_timeout)),
        }
    }

    async fn timed_restore(&self) -> Result<(), SwapError> {
        let fut = self.client.restore();
        match tokio::time::timeout(self.config.swap_timeout, fut).await {
            Ok(res) => res,
            Err(_) => Err(SwapError::Timeout(self.config.swap_timeout)),
        }
    }

    /// Step 2 of the rotation: evict the personality model and load Harness-1.
    ///
    /// On success returns [`SwapOutcome::Ok`] (or [`SwapOutcome::AlreadyWarm`]
    /// when Harness-1 is already resident). On failure returns
    /// [`SwapOutcome::Fallback`] — the caller should use a regular search.
    pub async fn load_search(&self) -> SwapOutcome {
        self.emit(VramPhase::LoadingSearch);
        let target = self.config.search_model.clone();
        {
            let cur = self.current_model.lock().await;
            if *cur == target {
                return SwapOutcome::AlreadyWarm;
            }
        }
        match self.timed_swap(&target).await {
            Ok(()) => {
                *self.current_model.lock().await = target;
                SwapOutcome::Ok
            }
            Err(e) => {
                tracing::warn!(error = %e, "harness load_search failed; falling back to regular search");
                SwapOutcome::Fallback
            }
        }
    }

    /// Step 3a: evict Harness-1. Combined with [`load_synthesis`] this is the
    /// "Research complete. Preparing analysis..." phase. Eviction here is
    /// implicit in the subsequent synthesis load (a swap evicts the prior model),
    /// so this method only reports degradation if the *next* load fails. It is
    /// kept separate so callers can express the spec's eviction-failure path.
    ///
    /// Returns [`SwapOutcome::Ok`] always in isolation — actual eviction is
    /// performed by the swap in [`load_synthesis`]; failure surfaces there as
    /// [`SwapOutcome::Degraded`].
    pub async fn evict_search(&self) -> SwapOutcome {
        // No standalone "unload" verb in the lifecycle API: a swap to the next
        // model evicts the current one. We therefore treat eviction as part of
        // the synthesis load and report Ok here. See load_synthesis for the
        // degraded path.
        SwapOutcome::Ok
    }

    /// Step 3b: load the synthesis model (deep 120B or the non-deep model).
    /// This swap also evicts Harness-1.
    ///
    /// On failure returns [`SwapOutcome::Degraded`] — synthesis should run on
    /// whatever model is currently loaded (per spec).
    pub async fn load_synthesis(&self, deep: bool) -> SwapOutcome {
        self.emit(VramPhase::PreparingSynthesis);
        let target = if deep {
            self.config.synthesis_deep_model.clone()
        } else {
            self.config.synthesis_model.clone()
        };
        {
            let cur = self.current_model.lock().await;
            if *cur == target {
                return SwapOutcome::AlreadyWarm;
            }
        }
        match self.timed_swap(&target).await {
            Ok(()) => {
                *self.current_model.lock().await = target;
                SwapOutcome::Ok
            }
            Err(e) => {
                tracing::warn!(error = %e, "harness load_synthesis failed; synthesizing on currently loaded model (degraded)");
                SwapOutcome::Degraded
            }
        }
    }

    /// Step 4: restore the personality model. Failures are logged but never
    /// fatal — the next user message triggers a swap anyway.
    pub async fn restore_default(&self) -> SwapOutcome {
        self.emit(VramPhase::Restoring);
        let target = self.config.personality_model.clone();
        {
            let cur = self.current_model.lock().await;
            if *cur == target {
                return SwapOutcome::AlreadyWarm;
            }
        }
        // Prefer the explicit restore verb; it reloads the default model.
        match self.timed_restore().await {
            Ok(()) => {
                *self.current_model.lock().await = target;
                SwapOutcome::Ok
            }
            Err(e) => {
                tracing::warn!(error = %e, "harness restore_default failed; next message will re-trigger a swap");
                SwapOutcome::Degraded
            }
        }
    }

    /// Convenience: run the entire four-step rotation atomically (acquires the
    /// rotation lock for its whole duration). The `deep` flag selects the
    /// synthesis model. Returns the outcome of each phase in order:
    /// (load_search, load_synthesis, restore_default).
    ///
    /// HRNS-05 will interleave the actual search/synthesis work between these
    /// phases; this helper exists for tests and as a reference sequence.
    pub async fn full_rotation(&self, deep: bool) -> RotationOutcome {
        let _guard = self.acquire().await;
        let load = self.load_search().await;
        // If load failed we still attempt synthesis/restore on the default model.
        let _evict = self.evict_search().await;
        let synth = self.load_synthesis(deep).await;
        let restore = self.restore_default().await;
        RotationOutcome { load, synth, restore }
    }

    /// Test/inspection helper: the model currently believed resident.
    pub async fn current_model(&self) -> String {
        self.current_model.lock().await.clone()
    }
}

/// Outcome of a [`HarnessVramManager::full_rotation`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RotationOutcome {
    pub load: SwapOutcome,
    pub synth: SwapOutcome,
    pub restore: SwapOutcome,
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex as StdMutex;

    /// Records every swap/restore call for assertion. Configurable to fail or
    /// stall specific models.
    struct MockSwap {
        calls: StdMutex<Vec<String>>,
        restores: AtomicUsize,
        /// Models whose swap should fail.
        fail_models: Vec<String>,
        /// Models whose swap should stall past the timeout.
        stall_models: Vec<String>,
        /// Whether restore should fail.
        fail_restore: bool,
        stall: Duration,
    }

    impl MockSwap {
        fn new() -> Self {
            Self {
                calls: StdMutex::new(Vec::new()),
                restores: AtomicUsize::new(0),
                fail_models: Vec::new(),
                stall_models: Vec::new(),
                fail_restore: false,
                stall: Duration::from_secs(60),
            }
        }
        fn fail_on(mut self, model: &str) -> Self {
            self.fail_models.push(model.to_string());
            self
        }
        fn stall_on(mut self, model: &str) -> Self {
            self.stall_models.push(model.to_string());
            self
        }
        fn fail_restore(mut self) -> Self {
            self.fail_restore = true;
            self
        }
        fn calls(&self) -> Vec<String> {
            self.calls.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl SwapClient for MockSwap {
        async fn swap(&self, model: &str, _engine: &str) -> Result<(), SwapError> {
            self.calls.lock().unwrap().push(model.to_string());
            if self.stall_models.iter().any(|m| m == model) {
                tokio::time::sleep(self.stall).await;
            }
            if self.fail_models.iter().any(|m| m == model) {
                return Err(SwapError::Status(503));
            }
            Ok(())
        }
        async fn restore(&self) -> Result<(), SwapError> {
            self.restores.fetch_add(1, Ordering::SeqCst);
            self.calls.lock().unwrap().push("__restore__".to_string());
            if self.fail_restore {
                return Err(SwapError::Status(500));
            }
            Ok(())
        }
    }

    fn test_config() -> VramConfig {
        VramConfig {
            personality_model: "p20b".into(),
            search_model: "harness-1".into(),
            synthesis_model: "p20b".into(),
            synthesis_deep_model: "d120b".into(),
            engine: "ollama_gpu".into(),
            swap_timeout: Duration::from_millis(200),
        }
    }

    #[tokio::test]
    async fn full_rotation_calls_swaps_in_order() {
        let mock = Arc::new(MockSwap::new());
        let mgr = HarnessVramManager::new(mock.clone(), test_config());
        let out = mgr.full_rotation(true).await;
        assert_eq!(out.load, SwapOutcome::Ok);
        assert_eq!(out.synth, SwapOutcome::Ok);
        assert_eq!(out.restore, SwapOutcome::Ok);
        // search load → synthesis load → restore.
        assert_eq!(mock.calls(), vec!["harness-1", "d120b", "__restore__"]);
        assert_eq!(mgr.current_model().await, "p20b");
    }

    #[tokio::test]
    async fn non_deep_rotation_uses_synthesis_model() {
        let mock = Arc::new(MockSwap::new());
        let mgr = HarnessVramManager::new(mock.clone(), test_config());
        mgr.full_rotation(false).await;
        // synthesis_model == personality "p20b"; current already p20b after that
        // swap, so restore is a warm no-op but still issues the restore verb path
        // only if not already there. Here synth sets current to p20b, so restore
        // short-circuits to AlreadyWarm and issues no restore call.
        assert_eq!(mock.calls(), vec!["harness-1", "p20b"]);
    }

    #[tokio::test]
    async fn each_swap_targets_correct_model() {
        let mock = Arc::new(MockSwap::new());
        let mgr = HarnessVramManager::new(mock.clone(), test_config());
        assert_eq!(mgr.load_search().await, SwapOutcome::Ok);
        assert_eq!(mock.calls(), vec!["harness-1"]);
        assert_eq!(mgr.load_synthesis(true).await, SwapOutcome::Ok);
        assert_eq!(mock.calls(), vec!["harness-1", "d120b"]);
        assert_eq!(mgr.restore_default().await, SwapOutcome::Ok);
        assert_eq!(mock.calls(), vec!["harness-1", "d120b", "__restore__"]);
    }

    #[tokio::test]
    async fn swap_timeout_enforced() {
        let mock = Arc::new(MockSwap::new().stall_on("harness-1"));
        let mgr = HarnessVramManager::new(mock.clone(), test_config());
        let start = std::time::Instant::now();
        // load_search stalls past the 200ms timeout → Fallback.
        let out = mgr.load_search().await;
        assert_eq!(out, SwapOutcome::Fallback);
        assert!(start.elapsed() < Duration::from_secs(5), "must time out quickly");
    }

    #[tokio::test]
    async fn load_search_failure_falls_back() {
        let mock = Arc::new(MockSwap::new().fail_on("harness-1"));
        let mgr = HarnessVramManager::new(mock.clone(), test_config());
        assert_eq!(mgr.load_search().await, SwapOutcome::Fallback);
        // Current model unchanged (still personality).
        assert_eq!(mgr.current_model().await, "p20b");
    }

    #[tokio::test]
    async fn synthesis_failure_is_degraded_not_fatal() {
        let mock = Arc::new(MockSwap::new().fail_on("d120b"));
        let mgr = HarnessVramManager::new(mock.clone(), test_config());
        mgr.load_search().await;
        let out = mgr.load_synthesis(true).await;
        assert_eq!(out, SwapOutcome::Degraded);
        // Stays on harness-1 (degraded synthesis on the search model).
        assert_eq!(mgr.current_model().await, "harness-1");
    }

    #[tokio::test]
    async fn restore_failure_is_degraded_not_fatal() {
        let mock = Arc::new(MockSwap::new().fail_restore());
        let mgr = HarnessVramManager::new(mock.clone(), test_config());
        mgr.load_search().await; // move off personality so restore is attempted
        let out = mgr.restore_default().await;
        assert_eq!(out, SwapOutcome::Degraded);
    }

    #[tokio::test]
    async fn warm_model_skips_redundant_load() {
        let mock = Arc::new(MockSwap::new());
        let mgr = HarnessVramManager::new(mock.clone(), test_config());
        assert_eq!(mgr.load_search().await, SwapOutcome::Ok);
        // Second back-to-back research query: harness-1 already warm.
        assert_eq!(mgr.load_search().await, SwapOutcome::AlreadyWarm);
        // Only one actual swap issued.
        assert_eq!(mock.calls(), vec!["harness-1"]);
    }

    #[tokio::test]
    async fn concurrent_rotations_are_serialized() {
        let mock = Arc::new(MockSwap::new());
        let mgr = HarnessVramManager::new(mock.clone(), test_config());

        // Two rotations racing; the mutex must serialise them so swap sequences
        // never interleave. We assert the call log is two clean rotations
        // back-to-back, not interleaved.
        let m1 = mgr.clone();
        let m2 = mgr.clone();
        let h1 = tokio::spawn(async move { m1.full_rotation(true).await });
        let h2 = tokio::spawn(async move { m2.full_rotation(true).await });
        h1.await.unwrap();
        h2.await.unwrap();

        let calls = mock.calls();
        // Each rotation emits exactly [harness-1, d120b, __restore__].
        let one = ["harness-1", "d120b", "__restore__"];
        assert_eq!(calls.len(), 6);
        // Split into two halves; each half must be a complete, non-interleaved
        // rotation.
        assert_eq!(&calls[0..3], &one);
        assert_eq!(&calls[3..6], &one);
    }

    #[tokio::test]
    async fn streaming_events_emitted_per_phase() {
        let mock = Arc::new(MockSwap::new());
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mgr = HarnessVramManager::new(mock.clone(), test_config()).with_progress(tx);
        mgr.full_rotation(true).await;

        let mut phases = Vec::new();
        while let Ok(p) = rx.try_recv() {
            phases.push(p);
        }
        assert_eq!(
            phases,
            vec![
                VramPhase::LoadingSearch,
                VramPhase::PreparingSynthesis,
                VramPhase::Restoring,
            ]
        );
        assert_eq!(VramPhase::LoadingSearch.message(), "Loading research agent...");
        assert_eq!(
            VramPhase::PreparingSynthesis.message(),
            "Research complete. Preparing analysis..."
        );
        assert_eq!(VramPhase::Restoring.message(), "Ready.");
    }

    #[tokio::test]
    #[serial]
    async fn config_from_env_uses_overrides() {
        // Isolated env mutation — harness tests run with --test-threads=1.
        std::env::set_var("HARNESS_SEARCH_MODEL", "custom-search");
        std::env::set_var("CHORD_DEEP_MODEL", "custom-deep");
        std::env::set_var("CHORD_FAST_MODEL", "custom-fast");
        std::env::set_var("HARNESS_SWAP_TIMEOUT_SECS", "7");
        let cfg = VramConfig::from_env();
        assert_eq!(cfg.search_model, "custom-search");
        assert_eq!(cfg.synthesis_deep_model, "custom-deep");
        assert_eq!(cfg.personality_model, "custom-fast");
        // synthesis_model defaults to personality when unset.
        assert_eq!(cfg.synthesis_model, "custom-fast");
        assert_eq!(cfg.swap_timeout, Duration::from_secs(7));
        std::env::remove_var("HARNESS_SEARCH_MODEL");
        std::env::remove_var("CHORD_DEEP_MODEL");
        std::env::remove_var("CHORD_FAST_MODEL");
        std::env::remove_var("HARNESS_SWAP_TIMEOUT_SECS");
    }

    /// Env-gated integration test against a real Chord lifecycle control API.
    /// Skipped unless `HARNESS_VRAM_INTEGRATION=1` and the control env is set.
    #[tokio::test]
    #[ignore = "requires a live Chord lifecycle control API (set HARNESS_VRAM_INTEGRATION=1)"]
    #[serial]
    async fn integration_real_swap() {
        if std::env::var("HARNESS_VRAM_INTEGRATION").as_deref() != Ok("1") {
            return;
        }
        let mgr = HarnessVramManager::from_env()
            .expect("CHORD_CONTROL_URL / CHORD_API_KEY must be set for integration test");
        let out = mgr.full_rotation(false).await;
        assert_eq!(out.load, SwapOutcome::Ok);
    }

    #[tokio::test]
    async fn http_swap_client_posts_swap_and_restore() {
        use httpmock::prelude::*;
        let server = MockServer::start();
        let swap_mock = server.mock(|when, then| {
            when.method(POST)
                .path("/api/lifecycle/swap")
                .header("Authorization", "Bearer k")
                .json_body(serde_json::json!({"model": "harness-1", "engine": "ollama_gpu"}));
            then.status(200);
        });
        let restore_mock = server.mock(|when, then| {
            when.method(POST)
                .path("/api/lifecycle/restore")
                .header("Authorization", "Bearer k");
            then.status(200);
        });
        let client =
            HttpSwapClient::new(server.base_url(), "k".into(), Duration::from_secs(5));
        client.swap("harness-1", "ollama_gpu").await.unwrap();
        client.restore().await.unwrap();
        swap_mock.assert();
        restore_mock.assert();
    }
}
