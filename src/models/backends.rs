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
    /// A remote, bearer-token-authenticated cloud HTTP API (OpenAI-compatible
    /// `/v1/chat/completions`), e.g. OpenRouter. Like `Daemon`, has no local
    /// process lifecycle — always assumed reachable over the network. Unlike
    /// every other kind, requests to it need an `Authorization: Bearer <key>`
    /// header; see `Backend::api_key_env`.
    OpenRouter,
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
    /// For `BackendKind::OpenRouter` (and any future bearer-authenticated
    /// remote kind): the NAME of the environment variable holding the API key
    /// — never the key value itself. The registry file is plain JSON on local
    /// disk (see module docs on `ModelRegistry`); storing a secret *value* in
    /// it would defeat "no hardcoded secrets, everything from env/<secret-manager> at
    /// runtime". The value is read fresh from this env var at dispatch time
    /// (`models::routing::resolve_and_ensure`). `None` for every local/
    /// unauthenticated backend (Ollama, llama-server, daemon).
    #[serde(default)]
    pub api_key_env: Option<String>,
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
                api_key_env: None,
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
                api_key_env: None,
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
                api_key_env: None,
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
            api_key_env: None,
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
    // Validated on the GPU inference host: llama3.3:70b (Q4_K_M, 42.5GB) — cold-load ~13s, peak
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
            api_key_env: None,
        },
    );

    // OpenRouter: a remote, bearer-authenticated cloud API — fundamentally
    // different from every other backend here (no local process, no VRAM, no
    // lifecycle to start/stop). Always offered, like `llama-gpu`/`vulkan`, since
    // there is nothing to seed conditionally on: it's a fixed public HTTPS
    // endpoint. NOTE the URL intentionally omits the trailing `/v1` — routing.rs
    // appends `/v1/chat/completions` to every backend's `url` uniformly, and
    // OpenRouter's real endpoint is `https://openrouter.ai/api/v1/chat/completions`.
    //
    // Registered here for the "Owl Alpha" slot (`OWL_ALPHA_MODEL_ID`,
    // `registry::register_openrouter_owl_alpha_from_env`) per operator request.
    // The original target, `openrouter/owl-alpha`, was verified 2026-07-03
    // directly against OpenRouter's own API to genuinely exist (created
    // 2026-04-28, 1,048,576-token context, matching the operator's "1M
    // context" claim) but to have ZERO active serving endpoints
    // (`"endpoints":[]`) — confirmed dead, not just unpriced, by an
    // independent live 404 from a sibling project's authenticated call.
    // Retargeted (2026-07-03, operator-directed) to
    // `nvidia/nemotron-3-ultra-550b-a55b:free` — same $0 pricing tier, same
    // ~1M context, confirmed LIVE (a real chat-completion round trip
    // succeeded). See `OWL_ALPHA_MODEL_ID`'s docs for the full history and
    // the `OPENROUTER_OWL_ALPHA_MODEL` override if this ever needs to move
    // again — that's a config flip, not a code change.
    out.insert(
        "openrouter".into(),
        Backend {
            name: "openrouter".into(),
            url: std::env::var("OPENROUTER_URL")
                .ok()
                .filter(|s| !s.trim().is_empty())
                .unwrap_or_else(|| "https://openrouter.ai/api".to_string()),
            hardware: Hardware::Cpu, // remote call — no local GPU/CPU cost
            kind: BackendKind::OpenRouter,
            unit: None,
            always_on: true, // no process lifecycle: it's a remote HTTP API
            idle_stop_secs: 0,
            launch: None,
            api_key_env: Some(
                std::env::var("OPENROUTER_API_KEY_ENV_NAME")
                    .ok()
                    .filter(|s| !s.trim().is_empty())
                    .unwrap_or_else(|| "OPENROUTER_API_KEY_CHORDHARMONY".to_string()),
            ),
        },
    );

    out
}

/// Backend name of the Vulkan/RADV (Mesa) GPU serving backend.
pub const VULKAN_BACKEND: &str = "vulkan";

/// Whether `model`'s tag names it as a Mixture-of-Experts architecture, by the
/// same name-substring convention [`is_vulkan_candidate`] has always used
/// (`moe`, `a3b`, `a22b`). Factored out to its own function (CPROX-02 fix) so
/// callers that need a PURE MoE signal — not entangled with `is_vulkan_candidate`'s
/// separate "is this tag one of the large DENSE size classes" size gate — can
/// reuse the exact same detection logic without reimplementing it.
///
/// **Known limitation, not introduced by this function**: this is a naming
/// convention, not a true architecture read from the model's config/weights.
/// Ollama's stored tag for at least one real MoE coder in the fleet
/// (`qwen3-coder:30b`, a genuine 30B-total/3B-active MoE model — see the
/// `tag_vulkan_candidates_dense_large_only` registry test, which already
/// labels it "a MoE coder" in its comment) does NOT contain `moe`/`a3b`/`a22b`
/// literally, so this substring check does not catch it. Closing that gap
/// needs either a curated model-family list or a real per-model architecture
/// signal ingested by the sweep — out of scope for this factoring-out.
pub fn is_moe_tagged(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    lower.contains("moe") || lower.contains("a3b") || lower.contains("a22b")
}

/// Whether a model is a **dense large** model (70B- or 32B-dense class) and thus
/// a candidate for the driver-stable [`VULKAN_BACKEND`] serving backend.
///
/// This mirrors the dense-retest shortlist: Vulkan is memory-bound (~5 tok/s at
/// 70B) but driver-stable, so it is offered for dense large models that run in
/// batch/async mode. MoE / small models keep their default (ROCm/Ollama) routing.
/// Name matching is on the `:<size>` tag suffix and is deliberately conservative
/// — MoE tags (e.g. containing `a3b`, `moe`) are excluded even at large sizes.
///
/// NOTE: this answers "is this tag BOTH non-MoE AND one of the large dense size
/// classes" — it is a vulkan-tier ELIGIBILITY gate, not a general model-safety
/// verdict. A model this returns `false` for is not necessarily MoE or unsafe —
/// most `false` results here are simply "not 32B/34B/70B/72B", which says
/// nothing about safety. Callers that need a pure MoE-only safety signal
/// (e.g. `models::coding_selector`, which must not exclude every non-32B+
/// dense model from ranking) should call [`is_moe_tagged`] directly instead of
/// inferring MoE-ness from a `false` return here.
pub fn is_vulkan_candidate(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    // Exclude Mixture-of-Experts tags — Vulkan is for *dense* large models.
    if is_moe_tagged(model) {
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
            api_key_env: None,
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
    fn seed_from_env_includes_openrouter() {
        // openrouter is always offered regardless of env (like llama-gpu/vulkan).
        std::env::remove_var("OPENROUTER_URL");
        std::env::remove_var("OPENROUTER_API_KEY_ENV_NAME");
        let b = seed_from_env();
        assert!(b.contains_key("openrouter"));
        let o = &b["openrouter"];
        assert_eq!(o.name, "openrouter");
        assert_eq!(o.kind, BackendKind::OpenRouter);
        assert_eq!(o.hardware, Hardware::Cpu);
        assert!(o.always_on, "no process lifecycle to manage");
        assert!(!o.on_demand());
        assert!(o.launch.is_none());
        assert_eq!(o.url, "https://openrouter.ai/api");
        // Default env var name, never an actual key value.
        assert_eq!(o.api_key_env.as_deref(), Some("OPENROUTER_API_KEY_CHORDHARMONY"));
    }

    #[test]
    fn seed_from_env_openrouter_respects_env_overrides() {
        std::env::set_var("OPENROUTER_URL", "https://openrouter.example/api");
        std::env::set_var("OPENROUTER_API_KEY_ENV_NAME", "MY_CUSTOM_KEY_VAR");
        let b = seed_from_env();
        let o = &b["openrouter"];
        assert_eq!(o.url, "https://openrouter.example/api");
        assert_eq!(o.api_key_env.as_deref(), Some("MY_CUSTOM_KEY_VAR"));
        std::env::remove_var("OPENROUTER_URL");
        std::env::remove_var("OPENROUTER_API_KEY_ENV_NAME");
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
    fn is_moe_tagged_matches_known_substrings_only() {
        assert!(is_moe_tagged("qwen3-a3b:30b"));
        assert!(is_moe_tagged("some-model-moe:latest"));
        assert!(is_moe_tagged("model-a22b:1"));
        // Dense models, including ones that are NOT vulkan-eligible by size,
        // must NOT be flagged as MoE by this narrower check.
        assert!(!is_moe_tagged("devstral:24b"));
        assert!(!is_moe_tagged("gemma3:12b"));
        assert!(!is_moe_tagged("codestral:latest"));
        // Known limitation (documented on the function): qwen3-coder:30b is a
        // genuine MoE model but its stored tag doesn't literally say so.
        assert!(!is_moe_tagged("qwen3-coder:30b"));
    }

    #[test]
    fn hardware_and_kind_serde_lowercase_kebab() {
        assert_eq!(serde_json::to_string(&Hardware::Gpu).unwrap(), "\"gpu\"");
        assert_eq!(
            serde_json::to_string(&BackendKind::LlamaServer).unwrap(),
            "\"llama-server\""
        );
        assert_eq!(
            serde_json::to_string(&BackendKind::OpenRouter).unwrap(),
            "\"open-router\""
        );
    }
}
