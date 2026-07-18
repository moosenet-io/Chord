//! CHRD-DIFF-01: Chord-managed DiffusionGemma daemon lifecycle.
//!
//! Historically `llama-diffusion-daemon` (the DiffusionGemma inference binary,
//! a llama.cpp fork serving HTTP on `:8877`) ran as a perpetual standalone
//! systemd unit (`dgem.service`) that held VRAM forever, whether or not
//! anything was actually using it. Per the operator's hard requirement,
//! diffusion must NOT sit in VRAM perpetually — Chord owns it now, not a
//! standalone service.
//!
//! This module makes Chord the owner of that process:
//! - **Lazy start**: the daemon is spawned only when a request tagged for the
//!   diffusion model ([`is_diffusion_model`]) actually arrives
//!   ([`DiffusionManager::ensure_running`]), mirroring the
//!   `snap::vllm::VLLMAdapter` lazy-start / poll-`/health` / stop pattern, but
//!   for a bare (non-docker) process via `tokio::process`.
//! - **Idle eviction**: a background reaper ([`spawn_idle_reaper`]) stops the
//!   daemon (freeing VRAM) after `DIFFUSION_IDLE_SECS` of inactivity, using the
//!   same in-process last-activity clock the request path touches — the
//!   `snap::activity::ActivityTracker` pattern, specialized for a process this
//!   module owns directly rather than one polled via `SharedInferenceState`.
//! - **`gpu_exclusive`-aware**: [`DiffusionManager::ensure_running`] refuses to
//!   spawn while the GPU-exclusive lock ([`crate::gpu_exclusive`]) is held by a
//!   foreign holder, and the lock's acquire path calls
//!   [`DiffusionManager::stop`] so a fresh grant evicts this daemon exactly
//!   like it evicts resident Ollama models (see `gpu_exclusive_acquire` in
//!   `routes.rs` and the idle-mode release sequence in `admin/idle.rs`).
//! - **Best-effort VRAM-release confirmation**: `stop()` runs
//!   `serving::release_verify::verify_release` (via the `SysfsDeviceProbe`
//!   already used by the SRV-12 clean-swap barrier) after killing the process,
//!   purely for observability/logging — a failed/timed-out confirmation never
//!   fails the stop itself.
//!
//! ## Config (env, all with sane NVMe-deployment defaults — no literals baked
//! into call sites)
//! - `DIFFUSION_MODEL_ID` — the chat-completions model id that routes here
//!   (default `diffusion-gemma`).
//! - `DIFFUSION_DAEMON_BIN` — path to `llama-diffusion-daemon`.
//! - `DIFFUSION_MODEL_PATH` — path to the DiffusionGemma GGUF.
//! - `DIFFUSION_LD_LIBRARY_PATH` — `LD_LIBRARY_PATH` for the child process.
//! - `DIFFUSION_BIND` — bind host (default `127.0.0.1`).
//! - `DIFFUSION_PORT` — bind port (default `8877`).
//! - `DIFFUSION_IDLE_SECS` — idle window before eviction (default `300`).
//! - `DIFFUSION_START_TIMEOUT_SECS` — health-poll budget on lazy start
//!   (default `120`, matching `VLLMAdapter::start`'s wait).
//! - `DIFFUSION_EXTRA_ARGS` — space-separated extra CLI args appended after
//!   `-m <model>` (default matches the verified GPU-host launch:
//!   `-ngl 99 -t 4 --diffusion-eb auto -c 8192 -ub 8192 -b 8192`).
//!
//! ## Retiring the standalone service
//! Once this deploys, the standalone `dgem.service` must be **stopped and disabled**
//! — Chord now spawns/kills the process itself, so a lingering standalone unit
//! would race Chord for the daemon and re-introduce the "sits in VRAM
//! perpetually" problem this spec exists to fix. That systemctl action is an
//! ops/deploy step for the orchestrator, not something this build agent
//! performs (no ssh/systemctl from a build sandbox).

use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use once_cell::sync::Lazy;
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tracing::{info, warn};

/// Current wall-clock epoch seconds. Thin (not pure) wrapper, mirroring
/// `gpu_exclusive::now_epoch`, so the decision helpers below stay pure/testable.
fn now_epoch() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ── Config ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffusionConfig {
    pub model_id: String,
    pub bin: String,
    pub model_path: String,
    pub ld_library_path: String,
    pub bind: String,
    pub port: u16,
    pub idle_secs: u64,
    pub start_timeout_secs: u64,
    pub extra_args: Vec<String>,
}

impl Default for DiffusionConfig {
    fn default() -> Self {
        Self {
            model_id: "diffusion-gemma".to_string(),
            bin: "/opt/nvme-scratch/dgem/llama-diffusion/build-vulkan/bin/llama-diffusion-daemon"
                .to_string(),
            model_path:
                "/opt/nvme-scratch/dgem/diffusiongemma-eval/models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf"
                    .to_string(),
            ld_library_path: "/opt/nvme-scratch/dgem/llama-diffusion/build-vulkan/bin".to_string(),
            bind: "127.0.0.1".to_string(),
            port: 8877,
            idle_secs: 300,
            start_timeout_secs: 120,
            extra_args: vec![
                "-ngl".into(),
                "99".into(),
                "-t".into(),
                "4".into(),
                "--diffusion-eb".into(),
                "auto".into(),
                "-c".into(),
                "8192".into(),
                "-ub".into(),
                "8192".into(),
                "-b".into(),
                "8192".into(),
            ],
        }
    }
}

impl DiffusionConfig {
    pub fn from_env() -> Self {
        // Read the real process environment. All parsing lives in
        // [`Self::from_getter`] so tests can exercise it with a fake getter and
        // NEVER mutate process-global env — concurrent `std::env::set_var` +
        // `std::env::var` (which `chat_completions` now triggers via
        // `diffusion::global()`) is a data race that can corrupt UNRELATED
        // tests' env reads under the gate's high-parallelism run.
        Self::from_getter(|k| std::env::var(k).ok())
    }

    /// Env-parsing core, parameterized over a getter so it is testable without
    /// touching process-global environment state.
    pub fn from_getter<F: Fn(&str) -> Option<String>>(get: F) -> Self {
        let mut cfg = Self::default();
        if let Some(v) = get("DIFFUSION_MODEL_ID") {
            if !v.trim().is_empty() {
                cfg.model_id = v.trim().to_string();
            }
        }
        if let Some(v) = get("DIFFUSION_DAEMON_BIN") {
            if !v.trim().is_empty() {
                cfg.bin = v.trim().to_string();
            }
        }
        if let Some(v) = get("DIFFUSION_MODEL_PATH") {
            if !v.trim().is_empty() {
                cfg.model_path = v.trim().to_string();
            }
        }
        if let Some(v) = get("DIFFUSION_LD_LIBRARY_PATH") {
            if !v.trim().is_empty() {
                cfg.ld_library_path = v.trim().to_string();
            }
        }
        if let Some(v) = get("DIFFUSION_BIND") {
            if !v.trim().is_empty() {
                cfg.bind = v.trim().to_string();
            }
        }
        if let Some(v) = get("DIFFUSION_PORT") {
            if let Ok(p) = v.trim().parse() {
                cfg.port = p;
            }
        }
        if let Some(v) = get("DIFFUSION_IDLE_SECS") {
            if let Ok(n) = v.trim().parse() {
                cfg.idle_secs = n;
            }
        }
        if let Some(v) = get("DIFFUSION_START_TIMEOUT_SECS") {
            if let Ok(n) = v.trim().parse() {
                cfg.start_timeout_secs = n;
            }
        }
        if let Some(v) = get("DIFFUSION_EXTRA_ARGS") {
            let parsed: Vec<String> = v.split_whitespace().map(str::to_string).collect();
            if !parsed.is_empty() {
                cfg.extra_args = parsed;
            }
        }
        cfg
    }

    pub fn base_url(&self) -> String {
        format!("http://{}:{}", self.bind, self.port)
    }

    /// The FULL OpenAI-compatible chat-completions endpoint the router forwards
    /// to. Returned by [`DiffusionManager::ensure_running`] so its contract is
    /// IDENTICAL to `models::routing::resolve_and_ensure` (which likewise
    /// returns a full `.../v1/chat/completions` URL) — the caller in
    /// `chat_completions` POSTs this verbatim and MUST NOT append a path, so no
    /// call site can ever produce a doubled `/v1/chat/completions/v1/chat/completions`.
    pub fn chat_completions_url(&self) -> String {
        format!("{}/v1/chat/completions", self.base_url())
    }

    fn health_url(&self) -> String {
        format!("{}/health", self.base_url())
    }
}

/// Is `model` (the client-supplied / alias-resolved chat-completions model
/// name) the managed DiffusionGemma model? Compares the untagged base name
/// (`diffusion-gemma:latest` → `diffusion-gemma`) so Ollama-style tag suffixes
/// don't cause a miss, same normalization `chat_completions` already applies
/// to the Ollama registry key.
pub fn is_diffusion_model(cfg: &DiffusionConfig, model: &str) -> bool {
    let base = model.split(':').next().unwrap_or(model);
    base.eq_ignore_ascii_case(&cfg.model_id)
}

// ── Pure decisions (no IO, no process, no clock — exhaustively unit-testable) ─

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StartDecision {
    /// Already tracked as running — just refresh activity and return the URL.
    AlreadyRunning,
    /// GPU-exclusive lock is held by a live foreign holder — refuse to spawn.
    Blocked,
    /// Not running, GPU free — spawn it.
    Start,
}

/// The lazy-start decision. Separated from `DiffusionManager::ensure_running`
/// so it is testable with no process spawn and no real `gpu_exclusive` state.
pub fn decide_start(gpu_exclusively_held: bool, already_running: bool) -> StartDecision {
    if already_running {
        StartDecision::AlreadyRunning
    } else if gpu_exclusively_held {
        StartDecision::Blocked
    } else {
        StartDecision::Start
    }
}

/// The idle-evict decision: stop only a running daemon whose idle window has
/// elapsed. `idle_timeout_secs == 0` disables eviction entirely (never evict).
pub fn decide_evict(running: bool, idle_secs: u64, idle_timeout_secs: u64) -> bool {
    running && idle_timeout_secs > 0 && idle_secs >= idle_timeout_secs
}

// ── DiffusionManager ────────────────────────────────────────────────────────

struct Inner {
    /// The `llama-diffusion-daemon` process Chord itself spawned and OWNS
    /// (can kill for idle-eviction / gpu-exclusive). `None` ⇒ Chord is not
    /// managing a child right now. A tracked-but-dead child (crashed, or
    /// bind-failed against an ambient daemon) is reaped to `None` by
    /// [`DiffusionManager::reap_dead_child`] before any "is it running?"
    /// decision, so a stale handle never reads as "running".
    child: Option<Child>,
    /// One-shot latch so the "an unmanaged daemon is already on the port"
    /// warning is logged once per occupancy, not on every forwarded request.
    ambient_warned: bool,
}

/// Owns the `llama-diffusion-daemon` child process lifecycle. One instance per
/// Chord process (like [`crate::gpu_exclusive::GPU_EXCLUSIVE`]) — there is one
/// physical GPU and one managed daemon.
pub struct DiffusionManager {
    cfg: DiffusionConfig,
    inner: Mutex<Inner>,
    last_activity_epoch: AtomicU64,
    client: reqwest::Client,
}

impl DiffusionManager {
    pub fn new(cfg: DiffusionConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .expect("reqwest client");
        Self {
            cfg,
            inner: Mutex::new(Inner {
                child: None,
                ambient_warned: false,
            }),
            last_activity_epoch: AtomicU64::new(0),
            client,
        }
    }

    pub fn config(&self) -> &DiffusionConfig {
        &self.cfg
    }

    pub fn model_id(&self) -> &str {
        &self.cfg.model_id
    }

    pub fn is_diffusion_model(&self, model: &str) -> bool {
        is_diffusion_model(&self.cfg, model)
    }

    /// Mark the daemon as just-used (called on every request forwarded to it,
    /// and right after a successful lazy start).
    pub fn touch_activity(&self) {
        self.last_activity_epoch.store(now_epoch(), Ordering::Relaxed);
    }

    /// Seconds since the daemon was last used. Meaningless (but harmless) if
    /// it was never started this boot — the idle reaper only acts when
    /// [`Self::is_running`] is also true.
    fn idle_secs(&self) -> u64 {
        now_epoch().saturating_sub(self.last_activity_epoch.load(Ordering::Relaxed))
    }

    /// Reap a tracked child that has already exited (crashed, or failed to bind
    /// because an ambient daemon holds the port) so a dead handle never reads as
    /// "running". Idempotent; operates on the already-held guard.
    fn reap_dead_child(guard: &mut Inner) {
        if let Some(child) = guard.child.as_mut() {
            match child.try_wait() {
                Ok(Some(status)) => {
                    warn!(?status, "diffusion: tracked llama-diffusion-daemon has exited — clearing handle");
                    guard.child = None;
                }
                Ok(None) => { /* still alive */ }
                Err(e) => {
                    // Can't determine liveness — leave the handle in place rather
                    // than dropping a possibly-live process we still own.
                    warn!(error = %e, "diffusion: could not poll daemon liveness (leaving handle)");
                }
            }
        }
    }

    /// True only when Chord OWNS a currently-alive child (reaps a dead handle
    /// first). An ambient/unmanaged daemon on the port is deliberately NOT
    /// "running" here — Chord can't kill what it didn't spawn, so the idle
    /// reaper must not treat it as an evictable managed process.
    pub async fn is_running(&self) -> bool {
        let mut guard = self.inner.lock().await;
        Self::reap_dead_child(&mut guard);
        guard.child.is_some()
    }

    /// Best-effort single `GET /health` probe with a short timeout, used to
    /// detect an ALREADY-listening daemon on the port before we try to spawn.
    /// `true` ⇒ something answered health OK on the port.
    async fn port_is_serving(&self) -> bool {
        matches!(
            self.client
                .get(self.cfg.health_url())
                .timeout(Duration::from_secs(2))
                .send()
                .await,
            Ok(r) if r.status().is_success()
        )
    }

    /// Lazily start the daemon if it isn't already running, gated on the
    /// process-global `gpu_exclusive` lock. Returns the FULL
    /// `.../v1/chat/completions` endpoint to forward to — identical in shape to
    /// `models::routing::resolve_and_ensure`, so the `chat_completions` caller
    /// POSTs it verbatim and never appends a second path segment.
    ///
    /// Thin wrapper over [`Self::ensure_running_gated`]: it reads the current
    /// GPU-exclusive holder state from the process-global lock and delegates.
    /// The split keeps the process-spawn + gate LOGIC testable via
    /// `ensure_running_gated(gpu_held)` WITHOUT any test ever touching the
    /// process-global `GPU_EXCLUSIVE` lock (which would race the parallel
    /// `routes::` tests that assert the gate is free) — the same
    /// isolated-state discipline the `gpu_exclusive` unit tests follow.
    pub async fn ensure_running(&self) -> Result<String, String> {
        let gpu_held = crate::gpu_exclusive::GPU_EXCLUSIVE
            .active_holder(crate::gpu_exclusive::now_epoch())
            .is_some();
        self.ensure_running_gated(gpu_held).await
    }

    /// The gate + lazy-spawn logic, with the GPU-exclusive-held decision passed
    /// in rather than read from global state (so it is race-free to unit-test).
    /// Returns the FULL chat-completions endpoint URL on success.
    pub async fn ensure_running_gated(&self, gpu_held: bool) -> Result<String, String> {
        let mut guard = self.inner.lock().await;

        // Liveness: a tracked child that already died must not read as "running"
        // (a crashed daemon, or a spawn that lost the port bind race to an
        // ambient daemon). Reap it before the start decision.
        Self::reap_dead_child(&mut guard);

        let already_running = guard.child.is_some();
        match decide_start(gpu_held, already_running) {
            StartDecision::AlreadyRunning => {
                drop(guard);
                self.touch_activity();
                return Ok(self.cfg.chat_completions_url());
            }
            StartDecision::Blocked => {
                return Err(
                    "diffusion: GPU exclusively held — refusing to start llama-diffusion-daemon"
                        .to_string(),
                );
            }
            StartDecision::Start => {}
        }

        // Ambient-daemon guard: we own no live child, but something may ALREADY
        // be listening & healthy on the port — e.g. the standalone `dgem.service`
        // that this feature is meant to retire but which may not be stopped yet.
        // Spawning would just lose the port-bind race and exit. Adopt it for
        // serving (availability), but do NOT record a child: Chord did not spawn
        // it and cannot kill it for idle-eviction, so `is_running()` stays false
        // and the idle reaper won't try to evict a process it doesn't own. Warn
        // once so the operator knows to stop/disable the unmanaged unit.
        if self.port_is_serving().await {
            if !guard.ambient_warned {
                warn!(
                    port = self.cfg.port,
                    "diffusion: an UNMANAGED daemon is already serving on the port \
                     (Chord did not spawn it — likely the standalone dgem.service that \
                     must be stopped+disabled). Forwarding to it, but Chord CANNOT \
                     idle-evict an unmanaged process; free the port so Chord can own \
                     the lifecycle and honor the no-perpetual-VRAM requirement."
                );
                guard.ambient_warned = true;
            }
            drop(guard);
            self.touch_activity();
            return Ok(self.cfg.chat_completions_url());
        }

        info!(
            bin = %self.cfg.bin,
            model = %self.cfg.model_path,
            port = self.cfg.port,
            "diffusion: lazy-starting llama-diffusion-daemon"
        );

        let mut cmd = Command::new(&self.cfg.bin);
        cmd.arg("-m").arg(&self.cfg.model_path);
        for arg in &self.cfg.extra_args {
            cmd.arg(arg);
        }
        cmd.env("DGEM_BIND", &self.cfg.bind)
            .env("DGEM_HTTP_PORT", self.cfg.port.to_string())
            // Chord's idle reaper owns eviction now — disable the daemon's own
            // self-timeout (0 = daemon-side default of "never") so the two
            // idle policies never race each other.
            .env("DGEM_IDLE_TIMEOUT_SECS", "0")
            .env("LD_LIBRARY_PATH", &self.cfg.ld_library_path)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true);

        let child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                return Err(format!("diffusion: failed to spawn llama-diffusion-daemon: {e}"));
            }
        };
        guard.child = Some(child);
        // Reset the ambient latch — we now own a freshly-spawned child.
        guard.ambient_warned = false;
        drop(guard);

        if let Err(e) = self.wait_for_health(self.cfg.start_timeout_secs).await {
            warn!(error = %e, "diffusion: daemon did not become healthy — tearing down");
            self.stop().await;
            return Err(e);
        }

        self.touch_activity();
        Ok(self.cfg.chat_completions_url())
    }

    /// Poll `GET /health` every 2s up to `timeout_secs`, mirroring
    /// `snap::vllm::VLLMAdapter::wait_for_health`.
    async fn wait_for_health(&self, timeout_secs: u64) -> Result<(), String> {
        let deadline = Instant::now() + Duration::from_secs(timeout_secs);
        let url = self.cfg.health_url();
        loop {
            if let Ok(r) = self.client.get(&url).send().await {
                if r.status().is_success() {
                    info!("diffusion: llama-diffusion-daemon healthy");
                    return Ok(());
                }
            }
            if Instant::now() >= deadline {
                return Err(format!(
                    "diffusion: llama-diffusion-daemon did not become healthy within {timeout_secs}s"
                ));
            }
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
    }

    /// Stop (kill) the daemon if running — freeing VRAM. Idempotent no-op when
    /// not running. Returns whether a process was actually stopped.
    pub async fn stop(&self) -> bool {
        let mut guard = self.inner.lock().await;
        let Some(mut child) = guard.child.take() else {
            return false;
        };
        drop(guard);

        info!("diffusion: stopping llama-diffusion-daemon");
        if let Err(e) = child.start_kill() {
            warn!(error = %e, "diffusion: kill signal failed (process may already be gone)");
        }
        let _ = child.wait().await;

        // Best-effort VRAM-release confirmation (SRV-12 machinery) — purely for
        // observability; never fails/blocks the stop that already happened.
        let cfg = crate::config::release_config();
        match crate::serving::release_verify::verify_release(
            &crate::serving::release_verify::SysfsDeviceProbe,
            &cfg,
        )
        .await
        {
            crate::serving::release_verify::ReleaseOutcome::Clean { baseline_ms } => {
                info!(baseline_ms, "diffusion: VRAM confirmed released after daemon stop");
            }
            other => {
                warn!(?other, "diffusion: VRAM release not confirmed clean after daemon stop (best-effort)");
            }
        }
        true
    }
}

/// Process-global managed-diffusion instance.
pub static DIFFUSION: Lazy<DiffusionManager> =
    Lazy::new(|| DiffusionManager::new(DiffusionConfig::from_env()));

pub fn global() -> &'static DiffusionManager {
    &DIFFUSION
}

/// Background idle-eviction reaper: every `check_interval`, stop the daemon if
/// it has been idle for at least `DiffusionConfig::idle_secs`. Mirrors
/// `models::routing::idle_stop_sweep`'s shape for on-demand `llama-server`
/// backends, specialized for the process this module owns directly. Spawned
/// once from `main.rs`; runs for the life of the Chord process.
pub fn spawn_idle_reaper(check_interval: Duration) {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(check_interval).await;
            let mgr = global();
            let running = mgr.is_running().await;
            if decide_evict(running, mgr.idle_secs(), mgr.cfg.idle_secs) {
                info!(
                    idle_secs = mgr.idle_secs(),
                    idle_timeout = mgr.cfg.idle_secs,
                    "diffusion: idle window elapsed — evicting llama-diffusion-daemon"
                );
                mgr.stop().await;
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── DiffusionConfig::from_env ────────────────────────────────────────────

    #[test]
    fn default_config_matches_expected_nvme_paths() {
        let cfg = DiffusionConfig::default();
        assert_eq!(cfg.model_id, "diffusion-gemma");
        assert_eq!(cfg.port, 8877);
        assert_eq!(cfg.bind, "127.0.0.1");
        assert_eq!(cfg.idle_secs, 300);
        assert!(cfg.bin.contains("llama-diffusion-daemon"));
        assert!(cfg.model_path.ends_with(".gguf"));
        assert!(cfg.extra_args.contains(&"--diffusion-eb".to_string()));
    }

    #[test]
    fn base_url_and_health_url() {
        let cfg = DiffusionConfig {
            bind: "127.0.0.1".into(),
            port: 8877,
            ..DiffusionConfig::default()
        };
        assert_eq!(cfg.base_url(), "http://127.0.0.1:8877");
        assert_eq!(cfg.health_url(), "http://127.0.0.1:8877/health");
    }

    // ── Issue 1 (codex HIGH): the forwarded URL must contain EXACTLY ONE
    // `/v1/chat/completions`. `ensure_running` returns this full endpoint and
    // `chat_completions` POSTs it verbatim (no append), so a doubled
    // `/v1/chat/completions/v1/chat/completions` (→ 404) is structurally
    // impossible. These assert the contract at the string level AND end-to-end.

    #[test]
    fn chat_completions_url_has_exactly_one_path_segment() {
        let cfg = DiffusionConfig {
            bind: "127.0.0.1".into(),
            port: 8877,
            ..DiffusionConfig::default()
        };
        let url = cfg.chat_completions_url();
        assert_eq!(url, "http://127.0.0.1:8877/v1/chat/completions");
        assert_eq!(
            url.matches("/v1/chat/completions").count(),
            1,
            "URL must carry exactly one chat-completions path, got: {url}"
        );
    }

    // NOTE: the heavier end-to-end forwarding assertion (spin an httpmock
    // "ambient daemon", adopt it, POST to the returned URL, and prove the daemon
    // saw EXACTLY ONE `/v1/chat/completions`) lives in the separate integration
    // test `tests/diffusion_forwarding.rs`. It's kept out of this in-process lib
    // test binary on purpose: its httpmock+reqwest timing jitter, running
    // concurrently with the `routes::` tests, widened a PRE-EXISTING global-lock
    // race (`test_embeddings_not_gpu_exclusive_gated` holds the process-global
    // `GPU_EXCLUSIVE` during a request, so any concurrent `chat_completions`
    // test then sees the gate held → 503). A separate test binary is its own
    // process, adding no concurrency pressure to the lib suite.

    // IMPORTANT: this test exercises the env PARSING via `from_getter` with a
    // fake map — it does NOT call `std::env::set_var`/`remove_var`. Mutating the
    // process-global environment concurrently with the `std::env::var` reads that
    // `chat_completions` now triggers (via `diffusion::global()`) is a data race
    // that corrupts UNRELATED tests' env reads under the gate's high-parallelism
    // run — the exact class of flake that motivated `from_getter`. Keep this test
    // env-free.
    #[test]
    fn from_getter_overrides_defaults_and_ignores_junk_port() {
        let env: std::collections::HashMap<&str, &str> = [
            ("DIFFUSION_MODEL_ID", "test-diffusion"),
            ("DIFFUSION_PORT", "9999"),
            ("DIFFUSION_IDLE_SECS", "42"),
            ("DIFFUSION_EXTRA_ARGS", "-x 1 -y 2"),
        ]
        .into_iter()
        .collect();
        let cfg = DiffusionConfig::from_getter(|k| env.get(k).map(|s| s.to_string()));
        assert_eq!(cfg.model_id, "test-diffusion");
        assert_eq!(cfg.port, 9999);
        assert_eq!(cfg.idle_secs, 42);
        assert_eq!(cfg.extra_args, vec!["-x", "1", "-y", "2"]);

        // Junk port value: falls back to the default rather than parsing garbage.
        let junk: std::collections::HashMap<&str, &str> =
            [("DIFFUSION_PORT", "not-a-port")].into_iter().collect();
        let cfg2 = DiffusionConfig::from_getter(|k| junk.get(k).map(|s| s.to_string()));
        assert_eq!(cfg2.port, 8877);

        // Empty getter ⇒ pure defaults.
        let cfg3 = DiffusionConfig::from_getter(|_| None);
        assert_eq!(cfg3, DiffusionConfig::default());
    }

    // ── is_diffusion_model ───────────────────────────────────────────────────

    #[test]
    fn is_diffusion_model_matches_exact_and_tagged() {
        let cfg = DiffusionConfig::default();
        assert!(is_diffusion_model(&cfg, "diffusion-gemma"));
        assert!(is_diffusion_model(&cfg, "diffusion-gemma:latest"));
        assert!(is_diffusion_model(&cfg, "Diffusion-Gemma"));
        assert!(!is_diffusion_model(&cfg, "qwen3-coder:30b"));
        assert!(!is_diffusion_model(&cfg, "diffusion-gemma-eval"));
    }

    // ── decide_start (the lazy-start decision) ──────────────────────────────

    #[test]
    fn decide_start_already_running_short_circuits_regardless_of_gpu_lock() {
        assert_eq!(decide_start(true, true), StartDecision::AlreadyRunning);
        assert_eq!(decide_start(false, true), StartDecision::AlreadyRunning);
    }

    #[test]
    fn decide_start_blocked_when_gpu_exclusively_held_and_not_running() {
        assert_eq!(decide_start(true, false), StartDecision::Blocked);
    }

    #[test]
    fn decide_start_starts_when_free_and_not_running() {
        assert_eq!(decide_start(false, false), StartDecision::Start);
    }

    // ── decide_evict (the idle-evict trigger) ───────────────────────────────

    #[test]
    fn decide_evict_true_when_running_and_idle_past_threshold() {
        assert!(decide_evict(true, 300, 300));
        assert!(decide_evict(true, 301, 300));
    }

    #[test]
    fn decide_evict_false_when_not_yet_idle() {
        assert!(!decide_evict(true, 299, 300));
    }

    #[test]
    fn decide_evict_false_when_not_running() {
        assert!(!decide_evict(false, 10_000, 300));
    }

    #[test]
    fn decide_evict_disabled_when_idle_timeout_zero() {
        // idle_timeout_secs == 0 ⇒ eviction disabled entirely, no matter how
        // long it's been idle.
        assert!(!decide_evict(true, 1_000_000, 0));
    }

    // ── DiffusionManager: gpu_exclusive interaction (no process spawn) ──────
    //
    // These exercise `ensure_running_gated`'s gate by PASSING the GPU-held
    // decision in directly, so they never touch (or race on) the process-global
    // `gpu_exclusive::GPU_EXCLUSIVE` lock — the same isolated-state discipline
    // the `gpu_exclusive` unit tests follow (using `GpuExclusive::new()`
    // instances rather than the global). Touching the global here widened the
    // race window with the parallel `routes::` chat-completions tests that
    // assert the GPU gate is free, flaking them under a full-workspace run.

    #[tokio::test]
    async fn ensure_running_blocked_while_gpu_exclusively_held() {
        let mgr = DiffusionManager::new(DiffusionConfig {
            // A bin that would fail to spawn even if we reached that far,
            // belt-and-suspenders proof the gate short-circuits before spawn.
            bin: "/nonexistent/llama-diffusion-daemon".into(),
            ..DiffusionConfig::default()
        });
        // gpu_held = true ⇒ must refuse before ever reaching spawn.
        let result = mgr.ensure_running_gated(true).await;
        assert!(result.is_err(), "must refuse to start while GPU exclusively held");
        assert!(!mgr.is_running().await);
    }

    #[tokio::test]
    async fn ensure_running_surfaces_spawn_failure_when_gpu_free() {
        // gpu_held = false, nothing listening on the port (so the ambient probe
        // finds nothing), and the binary path is bogus — spawn itself must fail
        // cleanly rather than panic, and no child is left tracked running.
        let mgr = DiffusionManager::new(DiffusionConfig {
            bin: "/definitely/not/a/real/binary/llama-diffusion-daemon".into(),
            // A port very unlikely to be occupied, so `port_is_serving` is false.
            port: 59_337,
            ..DiffusionConfig::default()
        });
        let result = mgr.ensure_running_gated(false).await;
        assert!(result.is_err());
        assert!(!mgr.is_running().await);
    }

    #[tokio::test]
    async fn stop_on_never_started_manager_is_a_noop() {
        let mgr = DiffusionManager::new(DiffusionConfig::default());
        assert!(!mgr.stop().await);
    }

    // ── Issue 2 (codex HIGH): liveness — a tracked child that has EXITED must
    // be reaped so it never reads as "managed and running" (which would make
    // the idle reaper try to evict a process that no longer exists, and mask a
    // crashed daemon behind a coincidentally-listening ambient one).
    #[tokio::test]
    async fn dead_tracked_child_is_reaped_and_not_running() {
        let mgr = DiffusionManager::new(DiffusionConfig::default());
        // Track a real process that exits immediately as our "owned" child.
        let child = tokio::process::Command::new("sh")
            .arg("-c")
            .arg("exit 0")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn short-lived child");
        {
            let mut guard = mgr.inner.lock().await;
            guard.child = Some(child);
        }
        // Let it actually exit before we poll liveness.
        tokio::time::sleep(Duration::from_millis(200)).await;
        // is_running() reaps the dead handle → not running, nothing to evict.
        assert!(
            !mgr.is_running().await,
            "a tracked child that has exited must not read as running"
        );
    }
}
