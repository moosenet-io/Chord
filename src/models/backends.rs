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
//!   - `vulkan`                 — a `llama-server` built with the Vulkan/RADV
//!     (Mesa) backend (`-DGGML_VULKAN=ON`), a *driver-stable* alternative to the
//!     ROCm-only lemonade build for dense large models when ROCm is unavailable
//!     or unstable. Same generic on-demand shape as `llama-gpu`.
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

    // Vulkan/RADV (Mesa) GPU backend: a llama.cpp `llama-server` built with
    // `-DGGML_VULKAN=ON` (Mesa 25.0.7 RADV on gfx1151). A *driver-stable*
    // alternative to the ROCm-only `lemonade-coder`/`llama-gpu` (b1258) build —
    // useful when ROCm is unavailable or unstable. Memory-bound like HIP/ROCm
    // (~5 tok/s at 70B), so it is intended for dense large models in batch/async
    // mode, not latency-sensitive interactive traffic. Same generic on-demand
    // shape as `llama-gpu`: serves ANY requested model's Ollama blob on the GPU.
    //
    // Validated on pvf1: llama3.3:70b (Q4_K_M, 42.5GB) — cold-load ~13s, peak
    // VRAM 50.6GB/96GB at 32k context, generation 5.3 tok/s (prompt 22–24
    // tok/s), dmesg clean.
    let vk_bin = std::env::var("VULKAN_LLAMA_BIN")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "/root/llama-vk/build/bin/llama-server".to_string());
    let vk_port = std::env::var("VULKAN_LLAMA_PORT")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "8083".to_string());
    out.insert(
        "vulkan".into(),
        Backend {
            name: "vulkan".into(),
            url: format!("http://127.0.0.1:{vk_port}"),
            hardware: Hardware::Gpu,
            kind: BackendKind::LlamaServer,
            unit: None,
            always_on: false,
            idle_stop_secs: 600,
            launch: Some(LaunchSpec {
                // Validated flags: RADV/Mesa gfx1151, fits 70B dense at 32k
                // context in ~51GB VRAM. `--no-warmup` keeps cold-load ~13s.
                bin: vk_bin,
                args: vec![
                    "-c".into(), "32768".into(),
                    "-ngl".into(), "99".into(),
                    "--no-mmap".into(),
                    "--no-warmup".into(),
                    "--host".into(), "127.0.0.1".into(),
                    "--port".into(), vk_port,
                ],
                model_arg: "-m".into(),
                model_from: default_model_from(),
            }),
        },
    );

    out
}

/// Backend name of the Vulkan/RADV (Mesa) GPU serving backend.
pub const VULKAN_BACKEND: &str = "vulkan";

/// Whether a model is a **dense large** model (70B- or 32B-dense class) and thus
/// a candidate for the driver-stable [`VULKAN_BACKEND`] serving backend.
///
/// This mirrors the dense-retest shortlist: Vulkan is memory-bound (~5 tok/s at
/// 70B) but driver-stable, so it is offered for dense large models that run in
/// batch/async mode. MoE / small models keep their default (ROCm/Ollama) routing.
/// Name matching is on the `:<size>` tag suffix and is deliberately conservative
/// — MoE tags (e.g. containing `a3b`, `moe`) are excluded even at large sizes.
pub fn is_vulkan_candidate(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    // Exclude Mixture-of-Experts tags — Vulkan is for *dense* large models.
    if lower.contains("moe") || lower.contains("a3b") || lower.contains("a22b") {
        return false;
    }
    // llama3.3:70b is the confirmed dense-large validation model.
    if lower.contains("llama3.3:70b") || lower.contains("llama3.3-70b") {
        return true;
    }
    // 70B- and 32B-dense class by tag suffix.
    let Some((_, tag)) = lower.rsplit_once(':') else {
        return false;
    };
    matches!(
        tag.trim_end_matches("-instruct").trim_end_matches("-q4_k_m"),
        "70b" | "72b" | "32b" | "34b"
    )
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
    fn seed_from_env_includes_vulkan() {
        // vulkan is always offered regardless of env (like llama-gpu).
        std::env::remove_var("VULKAN_LLAMA_BIN");
        std::env::remove_var("VULKAN_LLAMA_PORT");
        let b = seed_from_env();
        assert!(b.contains_key("vulkan"));
        let v = &b["vulkan"];
        assert_eq!(v.name, "vulkan");
        assert_eq!(v.hardware, Hardware::Gpu);
        assert_eq!(v.kind, BackendKind::LlamaServer);
        assert!(v.on_demand());
        let l = v.launch.as_ref().expect("vulkan has a launch spec");
        assert_eq!(l.bin, "/root/llama-vk/build/bin/llama-server");
        assert_eq!(l.model_arg, "-m");
        assert!(l.args.contains(&"--no-mmap".to_string()));
        assert!(l.args.contains(&"--no-warmup".to_string()));
    }

    #[test]
    fn vulkan_candidate_dense_large_only() {
        assert!(is_vulkan_candidate("llama3.3:70b"));
        assert!(is_vulkan_candidate("qwen2.5:72b-instruct"));
        assert!(is_vulkan_candidate("qwen2.5-coder:32b"));
        // MoE and small models are NOT candidates.
        assert!(!is_vulkan_candidate("qwen3-a3b:30b"));
        assert!(!is_vulkan_candidate("qwen3-coder:30b"));
        assert!(!is_vulkan_candidate("qwen3:8b"));
        assert!(!is_vulkan_candidate("untagged-name"));
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
