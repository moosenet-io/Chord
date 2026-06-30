//! Inference backends — first-class, hardware-tagged (GPU vs CPU).
//!
//! Chord historically forwarded every request to one `CHORD_LLM_URL`. P5 makes
//! the real backends explicit so models can be **tagged** to the hardware they
//! belong on and routed accordingly:
//!
//!   - `ollama` / `ollama-cpu`  — Ollama HTTP (`/api/*`), CPU tier on this host
//!     (Ollama's ROCm path does not engage on gfx1151 — it offloads 0 layers).
//!   - `lemonade-coder`         — a llama.cpp `llama-server` (OpenAI `/v1/*`)
//!     pinned to one model on the **GPU**.
//!   - `llama-gpu`              — a *generic* on-demand `llama-server` that loads
//!     ANY requested model's Ollama blob on the GPU (`-m <blob> -ngl 999`).
//!
//! Backends are **demand-driven by tags**: an on-demand backend is only started
//! when a model tagged for it is requested, and stopped when idle (see
//! `lifecycle.rs`). `always_on` backends (the primary Ollama) are assumed up.
//!
//! This module only defines the data model + env seeding. Routing lives in
//! `routes.rs`, lifecycle in `lifecycle.rs`, metric-capturing inference in
//! `infer.rs`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Hardware a backend runs on. Used to test/route a model on the right device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Hardware {
    Gpu,
    Cpu,
}

/// Wire protocol a backend speaks (determines request shape + timings parsing).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BackendKind {
    /// Ollama HTTP API (`/api/generate`, `/api/ps`).
    Ollama,
    /// llama.cpp `llama-server` (OpenAI `/v1/*` + `/completion`, `timings`).
    LlamaServer,
    /// Externally-managed daemon (e.g. DiffusionGemma); not load/unload-managed.
    Daemon,
}

/// How to spawn a unit-less on-demand backend (the generic `llama-gpu`): a base
/// binary + args, with the requested model's resolved blob appended via
/// `model_arg`. `model_from` selects how to resolve the blob path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchSpec {
    /// Executable, e.g. `/opt/lemonade/b1258/llama-server`.
    pub bin: String,
    /// Fixed args (host/port/ngl/ctx/...). The model flag is appended at launch.
    pub args: Vec<String>,
    /// Flag used to pass the model file, e.g. `-m`.
    pub model_arg: String,
    /// How to resolve the model path. `"ollama-blob"` = the model's largest
    /// local Ollama blob (the GGUF weights).
    #[serde(default = "default_model_from")]
    pub model_from: String,
}

fn default_model_from() -> String {
    "ollama-blob".to_string()
}

/// A single inference backend.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Backend {
    /// Stable backend name (referenced by `ModelRecord::backend`).
    pub name: String,
    /// Base URL, e.g. `http://localhost:11434` (no trailing path).
    pub url: String,
    /// Hardware this backend runs on.
    pub hardware: Hardware,
    /// Wire protocol.
    pub kind: BackendKind,
    /// systemd unit managing this backend, if any (None ⇒ spawned via `launch`).
    #[serde(default)]
    pub unit: Option<String>,
    /// True ⇒ assumed always running (the primary Ollama). False ⇒ started on
    /// demand and idle-stopped.
    #[serde(default)]
    pub always_on: bool,
    /// Idle seconds before an on-demand backend is stopped (0 ⇒ never auto-stop).
    #[serde(default)]
    pub idle_stop_secs: u64,
    /// Spawn spec for unit-less on-demand backends (the generic `llama-gpu`).
    #[serde(default)]
    pub launch: Option<LaunchSpec>,
}

impl Backend {
    /// Whether this backend is managed on-demand (start/stop) vs assumed up.
    pub fn on_demand(&self) -> bool {
        !self.always_on && self.kind != BackendKind::Daemon
    }
}

/// Read an env var, trimmed, non-empty.
fn env_url(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|v| v.trim().trim_end_matches('/').to_string())
        .filter(|v| !v.is_empty())
}

/// Seed the default backend catalogue from the existing env vars, used the first
/// time a registry is loaded without a `backends` section. Mirrors the live
/// Inference host topology: two CPU Ollama instances, the dedicated GPU coder, and the
/// generic on-demand GPU llama-server. Only backends whose URL/host is known are
/// seeded; the generic `llama-gpu` is always offered (started on demand).
pub fn seed_from_env() -> HashMap<String, Backend> {
    let mut out: HashMap<String, Backend> = HashMap::new();

    // Primary Ollama (lumina/general). CPU tier on gfx1151 (ROCm won't engage).
    if let Some(url) = env_url("OLLAMA_URL").or_else(|| env_url("OLLAMA_BASE_URL")) {
        out.insert(
            "ollama".into(),
            Backend {
                name: "ollama".into(),
                url,
                hardware: Hardware::Cpu,
                kind: BackendKind::Ollama,
                unit: Some("ollama.service".into()),
                always_on: true,
                idle_stop_secs: 0,
                launch: None,
            },
        );
    }
    // Resident CPU Ollama (embeddings / scheduled micro jobs).
    if let Some(url) = env_url("OLLAMA_CPU_URL") {
        out.insert(
            "ollama-cpu".into(),
            Backend {
                name: "ollama-cpu".into(),
                url,
                hardware: Hardware::Cpu,
                kind: BackendKind::Ollama,
                unit: Some("ollama-cpu.service".into()),
                always_on: true,
                idle_stop_secs: 0,
                launch: None,
            },
        );
    }
    // Dedicated GPU coder (one fixed model), managed by its systemd unit.
    if let Some(url) = env_url("LLAMA_SERVER_URL") {
        out.insert(
            "lemonade-coder".into(),
            Backend {
                name: "lemonade-coder".into(),
                url,
                hardware: Hardware::Gpu,
                kind: BackendKind::LlamaServer,
                unit: Some("lemonade-coder.service".into()),
                always_on: false,
                idle_stop_secs: 900,
                launch: None,
            },
        );
    }
    // Generic on-demand GPU backend: serves ANY requested model's blob on GPU.
    let llama_bin = std::env::var("LLAMA_GPU_BIN")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "/opt/lemonade/b1258/llama-server".to_string());
    let llama_port = std::env::var("LLAMA_GPU_PORT")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "8082".to_string());
    out.insert(
        "llama-gpu".into(),
        Backend {
            name: "llama-gpu".into(),
            url: format!("http://localhost:{llama_port}"),
            hardware: Hardware::Gpu,
            kind: BackendKind::LlamaServer,
            unit: None,
            always_on: false,
            idle_stop_secs: 600,
            launch: Some(LaunchSpec {
                bin: llama_bin,
                args: vec![
                    "-c".into(), "32768".into(),
                    "-ngl".into(), "999".into(),
                    "-fa".into(), "1".into(),
                    "--no-mmap".into(),
                    "--host".into(), "0.0.0.0".into(),
                    "--port".into(), llama_port,
                ],
                model_arg: "-m".into(),
                model_from: default_model_from(),
            }),
        },
    );

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn on_demand_flag() {
        let gpu = Backend {
            name: "llama-gpu".into(),
            url: "http://localhost:8082".into(),
            hardware: Hardware::Gpu,
            kind: BackendKind::LlamaServer,
            unit: None,
            always_on: false,
            idle_stop_secs: 600,
            launch: None,
        };
        assert!(gpu.on_demand());

        let primary = Backend { always_on: true, ..gpu.clone() };
        assert!(!primary.on_demand());

        let daemon = Backend { kind: BackendKind::Daemon, ..gpu.clone() };
        assert!(!daemon.on_demand());
    }

    #[test]
    fn seed_from_env_includes_generic_gpu() {
        // llama-gpu is always offered regardless of env.
        std::env::remove_var("OLLAMA_URL");
        std::env::remove_var("OLLAMA_CPU_URL");
        std::env::remove_var("LLAMA_SERVER_URL");
        let b = seed_from_env();
        assert!(b.contains_key("llama-gpu"));
        let g = &b["llama-gpu"];
        assert_eq!(g.hardware, Hardware::Gpu);
        assert_eq!(g.kind, BackendKind::LlamaServer);
        assert!(g.on_demand());
        assert!(g.launch.is_some());
    }

    #[test]
    fn hardware_and_kind_serde_lowercase_kebab() {
        assert_eq!(serde_json::to_string(&Hardware::Gpu).unwrap(), "\"gpu\"");
        assert_eq!(
            serde_json::to_string(&BackendKind::LlamaServer).unwrap(),
            "\"llama-server\""
        );
    }
}
