// adapters/vllm.rs — VLLMAdapter: manages the kyuz0/vllm-therock-gfx1151 container.
// VLLM-01: VLLMAdapter implementation.
//
// vLLM on Strix Halo (gfx1151) has specific requirements:
//   - --enforce-eager: required to prevent HIP Graph capture driver timeouts
//   - GPU device passthrough: /dev/kfd and /dev/dri
//   - Model changes require container restart (not hot-swappable)
//   - Health check via GET /health, metrics via GET /metrics (Prometheus format)

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::process::Command;
use tracing::{info, warn};

use super::EngineAdapter;

// ── VLLMConfig ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VLLMConfig {
    /// Docker image to use. Default: "kyuz0/vllm-therock-gfx1151:stable"
    pub container_image: String,
    /// Name of the running container. Default: "chord-vllm"
    pub container_name: String,
    /// Port to expose on the host. Default: 8000
    pub port: u16,
    /// HuggingFace model ID (e.g., "Qwen/Qwen3.6-27B")
    pub model: String,
    /// Fraction of GPU memory for vLLM. Default: 0.90
    pub gpu_memory_utilization: f64,
    /// Maximum sequence length. Default: 32768
    pub max_model_len: u32,
    /// Required on gfx1151 to avoid HIP Graph driver timeouts. Default: true
    pub enforce_eager: bool,
    /// Tensor parallelism degree. Default: 1.
    /// Set to 2 for TP=2 cluster across two Framework Desktops (512GB unified total).
    /// Requires Ray cluster running: RAY_ADDRESS=ray://{node1}:10001
    /// See docs/vllm-tp2-cluster.md for full setup guide.
    pub tensor_parallel_size: u8,
    /// Extra vLLM CLI flags appended verbatim.
    pub extra_args: Vec<String>,
    /// Host path for HuggingFace model cache (mounted into container).
    pub model_cache_path: String,
}

impl Default for VLLMConfig {
    fn default() -> Self {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
        Self {
            container_image: "kyuz0/vllm-therock-gfx1151:stable".into(),
            container_name: "chord-vllm".into(),
            port: 8000,
            model: String::new(),
            gpu_memory_utilization: 0.90,
            max_model_len: 32768,
            enforce_eager: true,
            tensor_parallel_size: 1,
            extra_args: vec![],
            model_cache_path: format!("{home}/.cache/huggingface"),
        }
    }
}

impl VLLMConfig {
    /// Load from environment variables with defaults.
    pub fn from_env() -> Self {
        let mut cfg = Self::default();
        if let Ok(v) = std::env::var("VLLM_CONTAINER_IMAGE") { cfg.container_image = v; }
        if let Ok(v) = std::env::var("VLLM_CONTAINER_NAME") { cfg.container_name = v; }
        if let Ok(v) = std::env::var("VLLM_PORT") { if let Ok(p) = v.parse() { cfg.port = p; } }
        if let Ok(v) = std::env::var("VLLM_MODEL") { cfg.model = v; }
        if let Ok(v) = std::env::var("VLLM_GPU_MEMORY_UTILIZATION") {
            if let Ok(f) = v.parse() { cfg.gpu_memory_utilization = f; }
        }
        if let Ok(v) = std::env::var("VLLM_MAX_MODEL_LEN") {
            if let Ok(n) = v.parse() { cfg.max_model_len = n; }
        }
        if let Ok(v) = std::env::var("VLLM_ENFORCE_EAGER") {
            cfg.enforce_eager = v != "false" && v != "0";
        }
        if let Ok(v) = std::env::var("VLLM_TENSOR_PARALLEL_SIZE") {
            if let Ok(n) = v.parse() { cfg.tensor_parallel_size = n; }
        }
        if let Ok(v) = std::env::var("VLLM_MODEL_CACHE_PATH") { cfg.model_cache_path = v; }
        cfg
    }

    /// Build the docker run argument list (without "docker run" itself).
    /// Exposed for testing.
    pub fn build_docker_run_args(&self) -> Vec<String> {
        let mut args = vec![
            "run".into(),
            "-d".into(),
            "--name".into(),
            self.container_name.clone(),
            // GPU device access
            "--device".into(), "/dev/kfd".into(),
            "--device".into(), "/dev/dri".into(),
            "--group-add".into(), "video".into(),
            "--group-add".into(), "render".into(),
            "--security-opt".into(), "seccomp=unconfined".into(),
            // Model cache volume
            "-v".into(),
            format!("{}:/root/.cache/huggingface", self.model_cache_path),
            // Port mapping
            "-p".into(),
            format!("{}:8000", self.port),
            // Container image
            self.container_image.clone(),
            // vLLM arguments
            "--model".into(), self.model.clone(),
            "--gpu-memory-utilization".into(), format!("{:.2}", self.gpu_memory_utilization),
            "--max-model-len".into(), self.max_model_len.to_string(),
            "--host".into(), "0.0.0.0".into(),
            "--port".into(), "8000".into(),
        ];
        if self.enforce_eager {
            args.push("--enforce-eager".into());
        }
        if self.tensor_parallel_size > 1 {
            args.push("--tensor-parallel-size".into());
            args.push(self.tensor_parallel_size.to_string());
        }
        for extra in &self.extra_args {
            args.push(extra.clone());
        }
        args
    }
}

// ── VLLMMetrics ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VLLMMetrics {
    pub running_requests: u64,
    pub waiting_requests: u64,
    pub avg_generation_tokens_per_sec: f64,
}

impl VLLMMetrics {
    /// Parse Prometheus-format /metrics response.
    pub fn parse_prometheus(text: &str) -> Self {
        let mut m = VLLMMetrics::default();
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
            }
        }
        m
    }
}

// ── VLLMAdapter ───────────────────────────────────────────────────────────────

pub struct VLLMAdapter {
    pub config: VLLMConfig,
    client: reqwest::Client,
}

impl VLLMAdapter {
    pub fn new(config: VLLMConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("reqwest client");
        Self { config, client }
    }

    /// Health endpoint URL.
    fn health_url(&self) -> String {
        format!("http://localhost:{}/health", self.config.port)
    }

    /// Metrics endpoint URL.
    fn metrics_url(&self) -> String {
        format!("http://localhost:{}/metrics", self.config.port)
    }

    /// Start the vLLM Docker container with the current config.
    /// Waits up to 120s for the container to become healthy.
    pub async fn start(&self) -> Result<(), String> {
        info!(
            container = self.config.container_name,
            image = self.config.container_image,
            model = self.config.model,
            "Starting vLLM container"
        );

        let args = self.config.build_docker_run_args();
        let output = Command::new("docker")
            .args(&args)
            .output()
            .await
            .map_err(|e| format!("docker run failed: {e}"))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("docker run failed: {stderr}"));
        }

        // Wait for health — vLLM is slow to start (model load + compilation)
        self.wait_for_health(120).await
    }

    /// Stop and remove the vLLM Docker container.
    pub async fn stop(&self) -> Result<(), String> {
        info!(container = self.config.container_name, "Stopping vLLM container");

        let stop = Command::new("docker")
            .args(["stop", &self.config.container_name])
            .output()
            .await
            .map_err(|e| format!("docker stop failed: {e}"))?;

        if !stop.status.success() {
            let stderr = String::from_utf8_lossy(&stop.stderr);
            warn!(container = self.config.container_name, "docker stop warning: {}", stderr);
        }

        let rm = Command::new("docker")
            .args(["rm", &self.config.container_name])
            .output()
            .await
            .map_err(|e| format!("docker rm failed: {e}"))?;

        if !rm.status.success() {
            let stderr = String::from_utf8_lossy(&rm.stderr);
            return Err(format!("docker rm failed: {stderr}"));
        }
        Ok(())
    }

    /// Poll GET /health every 5s up to `timeout_secs`.
    async fn wait_for_health(&self, timeout_secs: u64) -> Result<(), String> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
        let url = self.health_url();
        loop {
            match self.client.get(&url).send().await {
                Ok(r) if r.status().is_success() => {
                    info!(container = self.config.container_name, "vLLM container healthy");
                    return Ok(());
                }
                _ => {}
            }
            if std::time::Instant::now() >= deadline {
                return Err(format!(
                    "vLLM container did not become healthy within {timeout_secs}s"
                ));
            }
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        }
    }

    /// Fetch and parse vLLM Prometheus metrics.
    pub async fn get_metrics(&self) -> Result<VLLMMetrics, String> {
        let resp = self.client.get(self.metrics_url())
            .send()
            .await
            .map_err(|e| format!("metrics request failed: {e}"))?;
        if !resp.status().is_success() {
            return Err(format!("metrics returned {}", resp.status()));
        }
        let text = resp.text().await.map_err(|e| format!("metrics body error: {e}"))?;
        Ok(VLLMMetrics::parse_prometheus(&text))
    }
}

// ── EngineAdapter impl ────────────────────────────────────────────────────────

#[async_trait]
impl EngineAdapter for VLLMAdapter {
    /// Load a model: stop current container, update model in config, start fresh.
    /// vLLM does not support hot-swapping — a container restart is required.
    async fn load_model(&self, model: &str, ctx_size: u32) -> Result<(), String> {
        info!(model, ctx_size, "vLLM load_model: restarting container with new model");
        // Stop existing container if running (best-effort)
        let _ = self.stop().await;
        // Create a temporary adapter with the new model config
        let mut new_cfg = self.config.clone();
        new_cfg.model = model.to_string();
        if ctx_size > 0 {
            new_cfg.max_model_len = ctx_size;
        }
        let new_adapter = VLLMAdapter::new(new_cfg);
        new_adapter.start().await
    }

    /// Unload model: stop the container.
    async fn unload_model(&self, model: &str) -> Result<(), String> {
        info!(model, "vLLM unload_model: stopping container");
        self.stop().await
    }

    /// Check if a model is currently loaded and the container is healthy.
    async fn is_model_loaded(&self, model: &str) -> bool {
        // Check Docker container is running
        let inspect = Command::new("docker")
            .args(["inspect", "--format", "{{.State.Running}}", &self.config.container_name])
            .output()
            .await;

        let running = inspect
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim() == "true")
            .unwrap_or(false);

        if !running { return false; }

        // Check /health endpoint
        let healthy = self.client.get(self.health_url())
            .send()
            .await
            .ok()
            .map(|r| r.status().is_success())
            .unwrap_or(false);

        if !healthy { return false; }

        // Verify the configured model matches the requested model
        // (vLLM serves exactly one model at a time)
        model == self.config.model || self.config.model.contains(model) || model.contains(&self.config.model)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vllm_config_defaults_are_correct() {
        let cfg = VLLMConfig::default();
        assert_eq!(cfg.container_image, "kyuz0/vllm-therock-gfx1151:stable");
        assert_eq!(cfg.container_name, "chord-vllm");
        assert_eq!(cfg.port, 8000);
        assert_eq!(cfg.gpu_memory_utilization, 0.90);
        assert_eq!(cfg.max_model_len, 32768);
        assert!(cfg.enforce_eager, "enforce_eager must default to true for gfx1151");
        assert_eq!(cfg.tensor_parallel_size, 1);
        assert!(cfg.extra_args.is_empty());
    }

    #[test]
    fn docker_run_args_include_enforce_eager() {
        let cfg = VLLMConfig {
            enforce_eager: true,
            model: "Qwen/Qwen3.6-27B".into(),
            ..VLLMConfig::default()
        };
        let args = cfg.build_docker_run_args();
        assert!(
            args.contains(&"--enforce-eager".to_string()),
            "enforce_eager=true must produce --enforce-eager flag"
        );
    }

    #[test]
    fn docker_run_args_omit_enforce_eager_when_false() {
        let cfg = VLLMConfig {
            enforce_eager: false,
            model: "some-model".into(),
            ..VLLMConfig::default()
        };
        let args = cfg.build_docker_run_args();
        assert!(
            !args.contains(&"--enforce-eager".to_string()),
            "enforce_eager=false must not produce --enforce-eager flag"
        );
    }

    #[test]
    fn docker_run_args_include_gpu_devices() {
        let cfg = VLLMConfig {
            model: "Qwen/Qwen3.6-27B".into(),
            ..VLLMConfig::default()
        };
        let args = cfg.build_docker_run_args();
        // Must include both GPU device passthrough flags
        assert!(args.contains(&"/dev/kfd".to_string()), "must include /dev/kfd");
        assert!(args.contains(&"/dev/dri".to_string()), "must include /dev/dri");
        assert!(args.contains(&"video".to_string()), "must add video group");
        assert!(args.contains(&"render".to_string()), "must add render group");
    }

    #[test]
    fn docker_run_args_container_name_configurable() {
        let cfg = VLLMConfig {
            container_name: "my-custom-vllm".into(),
            model: "some-model".into(),
            ..VLLMConfig::default()
        };
        let args = cfg.build_docker_run_args();
        // --name must be followed by the container name
        let name_pos = args.iter().position(|a| a == "--name").expect("must have --name");
        assert_eq!(args[name_pos + 1], "my-custom-vllm");
    }

    #[test]
    fn docker_run_args_tensor_parallel_included_when_gt_1() {
        let cfg = VLLMConfig {
            tensor_parallel_size: 2,
            model: "big-model".into(),
            ..VLLMConfig::default()
        };
        let args = cfg.build_docker_run_args();
        assert!(args.contains(&"--tensor-parallel-size".to_string()));
        assert!(args.contains(&"2".to_string()));
    }

    #[test]
    fn docker_run_args_tensor_parallel_omitted_when_1() {
        let cfg = VLLMConfig {
            tensor_parallel_size: 1,
            model: "small-model".into(),
            ..VLLMConfig::default()
        };
        let args = cfg.build_docker_run_args();
        assert!(!args.contains(&"--tensor-parallel-size".to_string()));
    }

    #[test]
    fn docker_run_args_port_mapped_correctly() {
        let cfg = VLLMConfig {
            port: 8001,
            model: "test-model".into(),
            ..VLLMConfig::default()
        };
        let args = cfg.build_docker_run_args();
        // -p 8001:8000
        assert!(args.contains(&"8001:8000".to_string()), "must map custom port");
    }

    #[test]
    fn vllm_config_serializes_and_deserializes() {
        let cfg = VLLMConfig {
            model: "Qwen/Qwen3.6-27B".into(),
            ..VLLMConfig::default()
        };
        let json = serde_json::to_string(&cfg).expect("must serialize");
        let back: VLLMConfig = serde_json::from_str(&json).expect("must deserialize");
        assert_eq!(back.model, "Qwen/Qwen3.6-27B");
        assert_eq!(back.container_image, cfg.container_image);
        assert_eq!(back.enforce_eager, cfg.enforce_eager);
    }

    #[test]
    fn prometheus_metrics_parsing() {
        let sample = r#"
# HELP vllm:num_requests_running Number of running requests
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running 3
# HELP vllm:num_requests_waiting Number of waiting requests
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting 7
# HELP vllm:avg_generation_throughput_toks_per_s Throughput
# TYPE vllm:avg_generation_throughput_toks_per_s gauge
vllm:avg_generation_throughput_toks_per_s 42.5
"#;
        let m = VLLMMetrics::parse_prometheus(sample);
        assert_eq!(m.running_requests, 3);
        assert_eq!(m.waiting_requests, 7);
        assert!((m.avg_generation_tokens_per_sec - 42.5).abs() < f64::EPSILON);
    }

    #[test]
    fn prometheus_metrics_empty_input_returns_defaults() {
        let m = VLLMMetrics::parse_prometheus("");
        assert_eq!(m.running_requests, 0);
        assert_eq!(m.waiting_requests, 0);
        assert_eq!(m.avg_generation_tokens_per_sec, 0.0);
    }
}
