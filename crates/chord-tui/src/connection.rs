//! Async multi-instance connection manager (S91 CTUI-01).
//!
//! Every configured instance is polled for health on its own tokio task with a
//! hard per-request timeout. Results are published through a shared, lock-light
//! snapshot map that the UI reads each frame. CRITICAL INVARIANT: a slow or dead
//! instance must NEVER freeze the event loop — all I/O is async + timeout-bounded
//! and the UI only ever reads the last known snapshot, never blocks on a probe.
//!
//! Auth: the instance's token is resolved from the [`SecretManager`] (vault) at
//! request time and attached as a bearer header. The value is never logged and
//! never stored in the snapshot (only its presence/status if needed elsewhere).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;

use crate::config::InstanceConfig;
use crate::secret::{SecretManager, SecretRef};

/// Per-instance health as seen by the UI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Health {
    /// Never polled yet.
    Unknown,
    /// Last probe succeeded (2xx).
    Connected,
    /// Reachable but returned a non-2xx (e.g. auth failure, 5xx).
    Degraded,
    /// Timed out or connection error.
    Unreachable,
}

impl Health {
    pub fn label(self) -> &'static str {
        match self {
            Health::Unknown => "unknown",
            Health::Connected => "connected",
            Health::Degraded => "degraded",
            Health::Unreachable => "unreachable",
        }
    }
}

/// A point-in-time view of one instance's connection state. Carries NO secret
/// material.
#[derive(Clone, Debug)]
pub struct InstanceStatus {
    pub name: String,
    pub base_url: String,
    pub health: Health,
    /// Last observed round-trip in ms (best effort).
    pub latency_ms: Option<u64>,
    /// Short human note (e.g. HTTP status, error kind) — never a secret.
    pub detail: Option<String>,
    /// Version string reported by the instance `/health`, if any.
    pub version: Option<String>,
}

impl InstanceStatus {
    fn initial(cfg: &InstanceConfig) -> Self {
        InstanceStatus {
            name: cfg.name.clone(),
            base_url: cfg.base_url.clone(),
            health: Health::Unknown,
            latency_ms: None,
            detail: None,
            version: None,
        }
    }
}

/// Abstraction over the health probe so tests can inject fast/slow/failing
/// backends without real sockets. Returns the observed status for one instance.
#[async_trait::async_trait]
pub trait HealthProbe: Send + Sync {
    async fn probe(&self, cfg: &InstanceConfig, token: Option<&str>, timeout: Duration) -> InstanceStatus;
}

/// Real HTTP probe against an instance's `/health` endpoint.
pub struct HttpHealthProbe {
    client: reqwest::Client,
}

impl HttpHealthProbe {
    pub fn new() -> Self {
        HttpHealthProbe { client: reqwest::Client::new() }
    }
}

impl Default for HttpHealthProbe {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl HealthProbe for HttpHealthProbe {
    async fn probe(&self, cfg: &InstanceConfig, token: Option<&str>, timeout: Duration) -> InstanceStatus {
        let mut status = InstanceStatus::initial(cfg);
        let url = format!("{}/health", cfg.base_url.trim_end_matches('/'));
        let started = std::time::Instant::now();

        let mut req = self.client.get(&url).timeout(timeout);
        if let Some(t) = token {
            // Bearer value attached here and nowhere logged.
            req = req.bearer_auth(t);
        }

        match req.send().await {
            Ok(resp) => {
                status.latency_ms = Some(started.elapsed().as_millis() as u64);
                let code = resp.status();
                if code.is_success() {
                    status.health = Health::Connected;
                    if let Ok(v) = resp.json::<serde_json::Value>().await {
                        status.version = v
                            .get("version")
                            .and_then(|x| x.as_str())
                            .map(|s| s.to_string());
                    }
                } else {
                    status.health = Health::Degraded;
                    status.detail = Some(format!("HTTP {}", code.as_u16()));
                }
            }
            Err(e) => {
                // Timeout OR connection error → unreachable, never a panic.
                status.health = Health::Unreachable;
                status.detail = Some(if e.is_timeout() { "timeout".into() } else { "unreachable".into() });
            }
        }
        status
    }
}

/// Shared snapshot store, keyed by instance name. The UI clones cheaply.
pub type StatusMap = Arc<RwLock<HashMap<String, InstanceStatus>>>;

/// The connection manager: owns the snapshot map and spawns one poll loop per
/// instance. Dropping it aborts the poll tasks.
pub struct ConnectionManager {
    statuses: StatusMap,
    tasks: Vec<tokio::task::JoinHandle<()>>,
}

impl ConnectionManager {
    /// Spawn poll loops for every instance. `probe` and `secrets` are shared.
    pub fn spawn(
        instances: Vec<InstanceConfig>,
        probe: Arc<dyn HealthProbe>,
        secrets: Arc<dyn SecretManager>,
        poll_interval: Duration,
        request_timeout: Duration,
    ) -> Self {
        let statuses: StatusMap = Arc::new(RwLock::new(HashMap::new()));
        {
            // Seed initial Unknown entries so the UI has rows immediately.
            let mut init = HashMap::new();
            for cfg in &instances {
                init.insert(cfg.name.clone(), InstanceStatus::initial(cfg));
            }
            // Blocking-free: we're in a sync fn, use try_write on a fresh lock.
            *statuses.try_write().expect("fresh lock uncontended") = init;
        }

        let mut tasks = Vec::new();
        for cfg in instances {
            let statuses = statuses.clone();
            let probe = probe.clone();
            let secrets = secrets.clone();
            let auth_ref: Option<SecretRef> = cfg.auth_secret_ref.clone();
            let handle = tokio::spawn(async move {
                let mut ticker = tokio::time::interval(poll_interval);
                loop {
                    ticker.tick().await;
                    // Resolve token from vault per-poll (rotations picked up).
                    let token = match &auth_ref {
                        Some(r) => secrets.resolve(r).await,
                        None => None,
                    };
                    let token_str = token.as_ref().map(|v| v.expose().to_string());
                    let status = probe.probe(&cfg, token_str.as_deref(), request_timeout).await;
                    // token_str dropped here; never logged.
                    let mut guard = statuses.write().await;
                    guard.insert(cfg.name.clone(), status);
                }
            });
            tasks.push(handle);
        }

        ConnectionManager { statuses, tasks }
    }

    /// Cheap handle the UI holds to read snapshots each frame.
    pub fn statuses(&self) -> StatusMap {
        self.statuses.clone()
    }

    /// Non-blocking read of the current snapshot for the UI. Never awaits I/O.
    pub async fn snapshot(&self) -> Vec<InstanceStatus> {
        let g = self.statuses.read().await;
        let mut v: Vec<_> = g.values().cloned().collect();
        v.sort_by(|a, b| a.name.cmp(&b.name));
        v
    }
}

impl Drop for ConnectionManager {
    fn drop(&mut self) {
        for t in &self.tasks {
            t.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::InstanceKind;
    use crate::secret::EnvSecretManager;
    use std::sync::atomic::{AtomicU32, Ordering};

    fn cfg(name: &str) -> InstanceConfig {
        InstanceConfig {
            name: name.into(),
            base_url: "http://example.invalid".into(),
            kind: InstanceKind::Chord,
            auth_secret_ref: None,
        }
    }

    /// A probe that sleeps forever for one instance ("dead") and returns fast
    /// for another ("live"). Used to prove a dead instance can't block the loop.
    struct SplitProbe {
        live_polls: Arc<AtomicU32>,
    }

    #[async_trait::async_trait]
    impl HealthProbe for SplitProbe {
        async fn probe(&self, cfg: &InstanceConfig, _t: Option<&str>, timeout: Duration) -> InstanceStatus {
            let mut s = InstanceStatus::initial(cfg);
            if cfg.name == "dead" {
                // Simulate a hang bounded by the caller-provided timeout: the
                // manager passes request_timeout, so we honor it like reqwest.
                tokio::time::sleep(timeout).await;
                s.health = Health::Unreachable;
                s.detail = Some("timeout".into());
            } else {
                self.live_polls.fetch_add(1, Ordering::SeqCst);
                s.health = Health::Connected;
                s.latency_ms = Some(1);
            }
            s
        }
    }

    /// NEGATIVE TEST: a dead/hanging instance must NEVER freeze polling of a
    /// healthy instance (the event loop equivalent). We run both concurrently
    /// and assert the live one keeps getting polled while the dead one hangs.
    #[tokio::test(start_paused = true)]
    async fn dead_instance_does_not_block_event_loop() {
        let live_polls = Arc::new(AtomicU32::new(0));
        let probe = Arc::new(SplitProbe { live_polls: live_polls.clone() });
        let secrets = Arc::new(EnvSecretManager::from_env());
        let mgr = ConnectionManager::spawn(
            vec![cfg("dead"), cfg("live")],
            probe,
            secrets,
            Duration::from_millis(100),   // poll interval
            Duration::from_secs(3600),    // dead instance "hangs" this long
        );

        // Advance virtual time; the live instance should poll many times while
        // the dead one is still stuck in its 1h sleep.
        for _ in 0..5 {
            tokio::time::advance(Duration::from_millis(100)).await;
            tokio::task::yield_now().await;
        }
        tokio::time::advance(Duration::from_millis(50)).await;
        tokio::task::yield_now().await;

        let snap = mgr.snapshot().await;
        let live = snap.iter().find(|s| s.name == "live").unwrap();
        let dead = snap.iter().find(|s| s.name == "dead").unwrap();

        assert_eq!(live.health, Health::Connected, "live instance kept being polled");
        assert!(live_polls.load(Ordering::SeqCst) >= 3, "live polled repeatedly despite dead hang");
        // The dead instance is still stuck → never advanced past Unknown.
        assert_eq!(dead.health, Health::Unknown, "dead instance hung but did not block others");
    }

    #[tokio::test(start_paused = true)]
    async fn snapshot_seeds_unknown_immediately() {
        let probe = Arc::new(SplitProbe { live_polls: Arc::new(AtomicU32::new(0)) });
        let secrets = Arc::new(EnvSecretManager::from_env());
        let mgr = ConnectionManager::spawn(
            vec![cfg("a"), cfg("b")],
            probe,
            secrets,
            Duration::from_secs(1),
            Duration::from_secs(1),
        );
        let snap = mgr.snapshot().await;
        assert_eq!(snap.len(), 2);
        assert!(snap.iter().all(|s| s.health == Health::Unknown));
    }
}
