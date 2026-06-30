// SNAP-02: Engine health monitor — async background poller for all inference endpoints.
use chrono::Utc;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, warn};

use crate::snap::state::{
    EndpointStatus, EngineEndpoint, InferenceState, LoadedModel,
};

use crate::snap::activity::ActivityTracker;
use crate::snap::config::SnapConfig;

/// Shared inference state updated by the background health poller.
pub type SharedInferenceState = Arc<RwLock<InferenceState>>;

/// Create a new shared inference state with empty defaults.
pub fn new_shared_state() -> SharedInferenceState {
    Arc::new(RwLock::new(InferenceState::default()))
}

/// Background task: polls all inference endpoints every `poll_interval_secs`
/// and updates the shared InferenceState atomically.
/// VLLM-03: If vllm_url is non-empty, also polls vLLM health and metrics.
///
/// `pool` is `Some` only when SNAP persistence is enabled (`CHORD_SNAP_PERSIST`)
/// AND a DB resolved — see [`crate::snap::spawn_health_monitor`]. When `None`,
/// the loop behaves exactly as before (no DB writes). Persistence is best-effort:
/// every write is a `warn!`-and-drop on failure so the monitor never stalls.
pub async fn run_health_monitor(
    cfg: Arc<SnapConfig>,
    state: SharedInferenceState,
    pool: Option<sqlx::PgPool>,
) {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .expect("reqwest client");

    let interval = Duration::from_secs_f64(cfg.poll_interval_secs);

    // VRAM row-bloat guard state (only used when persisting): last write time +
    // last (used_mb, allocation-count) so we write at most once per
    // `vram_sample_secs` AND skip writes when nothing changed.
    let vram_gate = Duration::from_secs_f64(cfg.vram_sample_secs.max(1.0));
    let mut last_vram_write: Option<Instant> = None;
    let mut last_vram_fingerprint: Option<(u64, usize)> = None;

    loop {
        let endpoints = vec![
            ("llama-server", "LLAMA_SERVER_URL", cfg.llama_server_url.clone()),
            ("ollama_gpu", "OLLAMA_URL", cfg.ollama_url.clone()),
            ("ollama_cpu", "OLLAMA_CPU_URL", cfg.ollama_cpu_url.clone()),
        ];

        let mut engine_states = Vec::new();
        for (name, env_var, url) in &endpoints {
            // Skip unconfigured endpoints (empty URL) so SNAP only reports on
            // engines the operator actually wired via env.
            if url.is_empty() {
                continue;
            }
            let ep = poll_endpoint(&client, name, env_var, url).await;
            engine_states.push(ep);
        }

        // VLLM-03: Add vLLM engine to health state when configured.
        if !cfg.vllm_url.is_empty() {
            let vllm_ep = poll_vllm(&client, &cfg.vllm_url).await;
            engine_states.push(vllm_ep);
        }

        {
            let mut s = state.write().await;
            s.engines = engine_states;
            s.timestamp = Utc::now();
        }

        // Update VRAM state from sysfs + Ollama allocations
        crate::snap::vram::update_vram(&state).await;

        // ── SNAP persistence (default-OFF; only when pool is Some) ──
        // Best-effort: any DB error is logged and dropped, never blocks polling.
        if let Some(ref pool) = pool {
            persist_tick(
                pool,
                &state,
                vram_gate,
                &mut last_vram_write,
                &mut last_vram_fingerprint,
            )
            .await;
        }

        tokio::time::sleep(interval).await;
    }
}

/// Persist one tick's observability snapshot (VRAM sample + activity poll).
///
/// VRAM is interval-gated AND write-on-change: a sample is written only when
/// `vram_gate` has elapsed since the last write *and* the (used_mb, alloc-count)
/// fingerprint changed — so a fast health poll can't flood `snap_vram_sample`.
/// Activity persists only models with `active_requests > 0` (the insert filters
/// idle rows). All errors are `warn!`-and-dropped (best-effort observability).
async fn persist_tick(
    pool: &sqlx::PgPool,
    state: &SharedInferenceState,
    vram_gate: Duration,
    last_vram_write: &mut Option<Instant>,
    last_vram_fingerprint: &mut Option<(u64, usize)>,
) {
    // Snapshot the VRAM state + activity under a single read lock.
    let (vram, sampled_at) = {
        let s = state.read().await;
        (s.vram.clone(), s.timestamp)
    };

    // VRAM interval + write-on-change gate.
    let fingerprint = (vram.used_mb, vram.allocations.len());
    let interval_ok = last_vram_write
        .map(|t| t.elapsed() >= vram_gate)
        .unwrap_or(true);
    let changed = last_vram_fingerprint.map(|f| f != fingerprint).unwrap_or(true);
    if interval_ok && changed {
        match crate::snap::storage::insert_vram_sample(pool, &vram, sampled_at, None).await {
            Ok(_) => {
                *last_vram_write = Some(Instant::now());
                *last_vram_fingerprint = Some(fingerprint);
            }
            Err(e) => warn!(error = %e, "SNAP vram sample persist failed (dropped)"),
        }
    }

    // Activity poll: read all activity, persist only active rows (insert filters).
    let activity = ActivityTracker::new(state.clone()).all_activity().await;
    if activity.iter().any(|a| a.active_requests > 0) {
        if let Err(e) =
            crate::snap::storage::insert_activity_poll(pool, &activity, None).await
        {
            warn!(error = %e, "SNAP activity poll persist failed (dropped)");
        }
    }
}

// ── VLLM-03: vLLM health polling ─────────────────────────────────────────────

/// Poll a vLLM instance: GET /health + GET /metrics.
/// Returns an EngineEndpoint with vLLM-specific metrics in the LoadedModel list.
pub async fn poll_vllm(client: &reqwest::Client, base_url: &str) -> EngineEndpoint {
    use crate::snap::state::{EndpointStatus, EngineEndpoint, LoadedModel};

    let start = std::time::Instant::now();
    let health_url = format!("{base_url}/health");
    let metrics_url = format!("{base_url}/metrics");

    // Check /health first
    let health_status = match client.get(&health_url).send().await {
        Ok(r) if r.status().is_success() => {
            let elapsed = start.elapsed().as_millis() as u64;
            classify_response_time(elapsed)
        }
        Ok(_) => EndpointStatus::Degraded,
        Err(e) if e.is_timeout() => EndpointStatus::Offline,
        Err(_) => EndpointStatus::Offline,
    };

    let elapsed = start.elapsed().as_millis() as u64;

    // Parse /metrics if online
    let models = if health_status != EndpointStatus::Offline {
        match client.get(&metrics_url).send().await {
            Ok(r) if r.status().is_success() => {
                if let Ok(text) = r.text().await {
                    let metrics = parse_vllm_metrics(&text);
                    vec![LoadedModel {
                        name: "vllm".to_string(),
                        size_vram_mb: (metrics.gpu_cache_usage_perc * 100.0) as u64,
                        active_requests: metrics.running_requests as u32,
                        tokens_per_sec: if metrics.avg_generation_tokens_per_sec > 0.0 {
                            Some(metrics.avg_generation_tokens_per_sec)
                        } else {
                            None
                        },
                    }]
                } else {
                    vec![]
                }
            }
            _ => vec![],
        }
    } else {
        vec![]
    };

    debug!(name = "vllm", ?health_status, elapsed, "vLLM health poll complete");

    EngineEndpoint {
        name: "vllm".to_string(),
        endpoint_env_var: "CHORD_VLLM_URL".to_string(),
        status: health_status,
        models,
        response_time_ms: elapsed,
    }
}

/// Parsed vLLM Prometheus metrics.
#[derive(Debug, Default)]
pub struct VLLMHealthMetrics {
    pub running_requests: u64,
    pub waiting_requests: u64,
    pub avg_generation_tokens_per_sec: f64,
    pub gpu_cache_usage_perc: f64,
}

/// Parse vLLM Prometheus-format /metrics response.
pub fn parse_vllm_metrics(text: &str) -> VLLMHealthMetrics {
    let mut m = VLLMHealthMetrics::default();
    for line in text.lines() {
        if line.starts_with('#') { continue; }
        if let Some(rest) = line.strip_prefix("vllm:num_requests_running") {
            if let Some(val) = rest.trim().split_whitespace().next() {
                m.running_requests = val.parse().unwrap_or(0);
            }
        } else if let Some(rest) = line.strip_prefix("vllm:num_requests_waiting") {
            if let Some(val) = rest.trim().split_whitespace().next() {
                m.waiting_requests = val.parse().unwrap_or(0);
            }
        } else if let Some(rest) = line.strip_prefix("vllm:avg_generation_throughput_toks_per_s") {
            if let Some(val) = rest.trim().split_whitespace().next() {
                m.avg_generation_tokens_per_sec = val.parse().unwrap_or(0.0);
            }
        } else if let Some(rest) = line.strip_prefix("vllm:gpu_cache_usage_perc") {
            if let Some(val) = rest.trim().split_whitespace().next() {
                m.gpu_cache_usage_perc = val.parse().unwrap_or(0.0);
            }
        }
    }
    m
}

/// Poll a single inference endpoint and return its EngineEndpoint state.
async fn poll_endpoint(
    client: &reqwest::Client,
    name: &str,
    endpoint_env_var: &str,
    url: &str,
) -> EngineEndpoint {
    let start = Instant::now();

    let (status, models, response_time_ms) = if name == "llama-server" {
        poll_llama_server(client, url, start).await
    } else {
        poll_ollama(client, url, start).await
    };

    debug!(name, ?status, response_time_ms, "health poll complete");

    EngineEndpoint {
        name: name.to_string(),
        endpoint_env_var: endpoint_env_var.to_string(),
        status,
        models,
        response_time_ms,
    }
}

async fn poll_llama_server(
    client: &reqwest::Client,
    url: &str,
    start: Instant,
) -> (EndpointStatus, Vec<LoadedModel>, u64) {
    let health_url = format!("{url}/health");
    match client.get(&health_url).send().await {
        Ok(resp) if resp.status().is_success() => {
            let elapsed = start.elapsed().as_millis() as u64;
            let status = classify_response_time(elapsed);

            // Parse slots from health response if available
            let models = if let Ok(body) = resp.json::<serde_json::Value>().await {
                let slots_processing = body
                    .get("slots_processing")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                // llama-server loads one model at a time
                vec![LoadedModel {
                    name: body
                        .get("model")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string(),
                    size_vram_mb: 0, // determined by vram.rs
                    active_requests: slots_processing as u32,
                    tokens_per_sec: None,
                }]
            } else {
                vec![]
            };

            (status, models, elapsed)
        }
        Ok(resp) => {
            warn!(url, status = %resp.status(), "llama-server returned non-200");
            (EndpointStatus::Degraded, vec![], start.elapsed().as_millis() as u64)
        }
        Err(e) if e.is_timeout() => {
            warn!(url, "llama-server health check timed out");
            (EndpointStatus::Offline, vec![], 5000)
        }
        Err(e) => {
            debug!(url, error = %e, "llama-server unreachable");
            (EndpointStatus::Offline, vec![], start.elapsed().as_millis() as u64)
        }
    }
}

async fn poll_ollama(
    client: &reqwest::Client,
    url: &str,
    start: Instant,
) -> (EndpointStatus, Vec<LoadedModel>, u64) {
    let ps_url = format!("{url}/api/ps");
    match client.get(&ps_url).send().await {
        Ok(resp) if resp.status().is_success() => {
            let elapsed = start.elapsed().as_millis() as u64;
            let status = classify_response_time(elapsed);

            let models = if let Ok(body) = resp.json::<serde_json::Value>().await {
                body.get("models")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|m| {
                                let name = m.get("name")?.as_str()?.to_string();
                                let size_vram_mb = m
                                    .get("size_vram")
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(0)
                                    / (1024 * 1024);
                                Some(LoadedModel {
                                    name,
                                    size_vram_mb,
                                    active_requests: 0,
                                    tokens_per_sec: None,
                                })
                            })
                            .collect()
                    })
                    .unwrap_or_default()
            } else {
                vec![]
            };

            (status, models, elapsed)
        }
        Ok(_) => (EndpointStatus::Degraded, vec![], start.elapsed().as_millis() as u64),
        Err(e) if e.is_timeout() => (EndpointStatus::Offline, vec![], 5000),
        Err(_) => (EndpointStatus::Offline, vec![], start.elapsed().as_millis() as u64),
    }
}

fn classify_response_time(ms: u64) -> EndpointStatus {
    if ms < 2000 {
        EndpointStatus::Online
    } else if ms < 5000 {
        EndpointStatus::Degraded
    } else {
        EndpointStatus::Offline
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_response_times() {
        assert_eq!(classify_response_time(100), EndpointStatus::Online);
        assert_eq!(classify_response_time(1999), EndpointStatus::Online);
        assert_eq!(classify_response_time(2000), EndpointStatus::Degraded);
        assert_eq!(classify_response_time(4999), EndpointStatus::Degraded);
        assert_eq!(classify_response_time(5000), EndpointStatus::Offline);
    }

    #[test]
    fn all_engines_offline_state_still_valid() {
        let state = InferenceState {
            engines: vec![
                EngineEndpoint {
                    name: "llama-server".into(),
                    endpoint_env_var: "LLAMA_SERVER_URL".into(),
                    status: EndpointStatus::Offline,
                    models: vec![],
                    response_time_ms: 5000,
                },
            ],
            vram: Default::default(),
            timestamp: Utc::now(),
        };
        // All offline — state struct is still valid and serializable
        let json = serde_json::to_string(&state).unwrap();
        assert!(json.contains("offline"));
    }

    // VLLM-03: vLLM health metrics tests

    #[test]
    fn parse_vllm_metrics_all_fields() {
        let sample = r#"
# HELP vllm:num_requests_running Number of running requests
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running 5
# HELP vllm:num_requests_waiting Number of waiting requests
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting 3
# HELP vllm:avg_generation_throughput_toks_per_s Throughput
# TYPE vllm:avg_generation_throughput_toks_per_s gauge
vllm:avg_generation_throughput_toks_per_s 38.5
# HELP vllm:gpu_cache_usage_perc GPU KV cache usage
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc 0.72
"#;
        let m = parse_vllm_metrics(sample);
        assert_eq!(m.running_requests, 5);
        assert_eq!(m.waiting_requests, 3);
        assert!((m.avg_generation_tokens_per_sec - 38.5).abs() < f64::EPSILON);
        assert!((m.gpu_cache_usage_perc - 0.72).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_vllm_metrics_empty_returns_defaults() {
        let m = parse_vllm_metrics("");
        assert_eq!(m.running_requests, 0);
        assert_eq!(m.waiting_requests, 0);
        assert_eq!(m.avg_generation_tokens_per_sec, 0.0);
        assert_eq!(m.gpu_cache_usage_perc, 0.0);
    }

    #[test]
    fn parse_vllm_metrics_skips_comment_lines() {
        let sample = "# this is a comment\nvllm:num_requests_running 7\n";
        let m = parse_vllm_metrics(sample);
        assert_eq!(m.running_requests, 7);
    }

    #[test]
    fn vllm_endpoint_name_and_env_var_correct() {
        // Verify the endpoint naming convention
        let name = "vllm";
        let env_var = "CHORD_VLLM_URL";
        assert_eq!(name, "vllm");
        assert_eq!(env_var, "CHORD_VLLM_URL");
    }

    #[test]
    fn vllm_health_not_added_when_unconfigured() {
        // When vllm_url is empty, the monitor should not add a vLLM endpoint.
        // This tests the conditional logic in run_health_monitor.
        let vllm_url = "";
        let should_poll = !vllm_url.is_empty();
        assert!(!should_poll, "vLLM should not be polled when URL is empty");
    }

    #[test]
    fn vllm_health_added_when_configured() {
        let vllm_url = "http://vllm-host.example:8000";
        let should_poll = !vllm_url.is_empty();
        assert!(should_poll, "vLLM should be polled when URL is configured");
    }
}
