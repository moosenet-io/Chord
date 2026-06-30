use std::collections::HashMap;

use crate::error::ProxyError;

/// Rate limit thresholds per user role.
///
/// All values are daily call counts. Admin is handled separately (always unlimited).
/// Populated from env vars via `RateLimitConfig::from_env()`.
///
/// Env vars:
///   CHORD_RATE_LLM_USER   (default 200)
///   CHORD_RATE_TOOL_USER  (default 500)
///   CHORD_RATE_DEEP_USER  (default 50)
///   CHORD_RATE_LLM_GUEST  (default 20)
///   CHORD_RATE_TOOL_GUEST (default 50)
///   CHORD_RATE_DEEP_GUEST (default 5)
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    pub user_llm_limit: u32,
    pub user_tool_limit: u32,
    pub user_deep_limit: u32,
    pub guest_llm_limit: u32,
    pub guest_tool_limit: u32,
    pub guest_deep_limit: u32,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        RateLimitConfig {
            user_llm_limit: 200,
            user_tool_limit: 500,
            user_deep_limit: 50,
            guest_llm_limit: 20,
            guest_tool_limit: 50,
            guest_deep_limit: 5,
        }
    }
}

impl RateLimitConfig {
    pub fn from_env() -> Self {
        fn read_u32(var: &str, default: u32) -> u32 {
            std::env::var(var)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        }
        RateLimitConfig {
            user_llm_limit: read_u32("CHORD_RATE_LLM_USER", 200),
            user_tool_limit: read_u32("CHORD_RATE_TOOL_USER", 500),
            user_deep_limit: read_u32("CHORD_RATE_DEEP_USER", 50),
            guest_llm_limit: read_u32("CHORD_RATE_LLM_GUEST", 20),
            guest_tool_limit: read_u32("CHORD_RATE_TOOL_GUEST", 50),
            guest_deep_limit: read_u32("CHORD_RATE_DEEP_GUEST", 5),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    /// URL of the MCP backend — reads MCP_BACKEND_URL env var
    pub mcp_backend_url: String,
    /// JWT secret for validating incoming requests — reads CHORD_JWT_SECRET env var
    pub jwt_secret: String,
    /// Per-tool call timeout in seconds — reads CHORD_TOOL_TIMEOUT_SECS (default 30)
    pub tool_timeout_secs: u64,
    /// Tool catalog cache TTL in seconds — reads CHORD_CATALOG_CACHE_SECS (default 300)
    pub catalog_cache_secs: u64,
    /// Port the proxy listens on — reads CHORD_PROXY_PORT (default 9099)
    pub listen_port: u16,
    /// Port the TIER-05 model-tier control API listens on — reads
    /// CHORD_CONTROL_PORT (default 8090). Bound by a second axum listener
    /// independent of the proxy port; a bind failure there does not take down
    /// the main proxy.
    pub control_port: u16,
    /// Per-user rate limit thresholds
    pub rate_limits: RateLimitConfig,
    /// Upstream LLM backend URL for the `/v1/chat/completions` proxy.
    /// Reads CHORD_LLM_URL env var (e.g. `http://localhost:11434/v1/chat/completions`).
    /// `None` (or empty) means the proxy endpoint is disabled and returns 503.
    pub llm_backend_url: Option<String>,
    /// Model alias → real backend model name map. Reads CHORD_MODEL_ALIASES env var
    /// (a JSON object, e.g. `{"lumina-fast":"gpt-oss:20b","lumina-deep":"gpt-oss:120b"}`).
    /// Used to rewrite the `model` field before forwarding to Ollama so that
    /// lumina-core's `lumina-fast`/`lumina-deep` aliases resolve to real models.
    /// An unset, empty, or malformed value yields an empty map (no rewriting).
    pub model_aliases: HashMap<String, String>,
    /// Archive (e.g. NFS) root that holds cold-tier Ollama models.
    /// Reads MODEL_ARCHIVE_PATH (default `/var/lib/model-archive`).
    pub model_archive_path: String,
    /// Local Ollama models root holding warm/hot models.
    /// Reads MODEL_LOCAL_PATH (default `/opt/ollama-models`).
    pub model_local_path: String,
    /// Protected model names that are never auto-archived. Reads MODEL_PROTECTED
    /// (comma-separated). Default: the core Lumina + qwen models.
    pub model_protected: Vec<String>,
    /// Maximum duration (seconds) for a cold→warm archive pull before it aborts
    /// and cleans up partial files. Reads MODEL_PULL_TIMEOUT_SECS (default 600).
    pub model_pull_timeout_secs: u64,
    /// Path to the JSON file backing the model registry (tier/size/timestamps).
    /// Reads MODEL_REGISTRY_PATH (default `/opt/chord/model-registry.json`).
    pub model_registry_path: String,
    /// Local-disk used-percentage threshold above which the TIER-03 eviction
    /// sweep archives warm models (warm → cold). Reads MODEL_DISK_PRESSURE_PERCENT
    /// (default 80).
    pub model_disk_pressure_percent: u8,
    /// Interval (seconds) between background disk-pressure eviction sweeps.
    /// Reads MODEL_SWEEP_INTERVAL_SECS (default 1800 = 30 min).
    pub model_sweep_interval_secs: u64,
    /// TIER-04 cooldown: a warm, non-protected model whose `last_requested` is
    /// older than this many hours is archived (warm → cold) regardless of disk
    /// pressure. Reads MODEL_WARM_COOLDOWN_HOURS (default 168 = 7 days). A value
    /// of `0` disables cooldown eviction entirely (a startup warning is logged).
    pub model_warm_cooldown_hours: u64,
}

/// Parse a comma-separated `MODEL_PROTECTED` value into a list of names,
/// trimming whitespace and dropping empties.
pub fn parse_protected_models(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Parse a raw `CHORD_MODEL_ALIASES` value (a JSON object of string→string) into a
/// map. A missing, empty, whitespace-only, or malformed value yields an empty map
/// so a bad config never aborts startup — it just disables alias rewriting.
pub fn parse_model_aliases(raw: Option<String>) -> HashMap<String, String> {
    let Some(text) = raw.map(|v| v.trim().to_string()).filter(|v| !v.is_empty()) else {
        return HashMap::new();
    };
    serde_json::from_str::<HashMap<String, String>>(&text).unwrap_or_default()
}

/// Resolve a model name through an alias map. Returns the mapped backend model when
/// `model` is a known alias, otherwise returns `model` unchanged (pass-through for
/// real model names already understood by the backend).
pub fn resolve_model_alias<'a>(aliases: &'a HashMap<String, String>, model: &'a str) -> &'a str {
    aliases.get(model).map(String::as_str).unwrap_or(model)
}

/// Normalize a raw `CHORD_LLM_URL` value: a missing, empty, or whitespace-only
/// value yields `None` (endpoint disabled → 503); otherwise the trimmed URL.
fn normalize_llm_url(raw: Option<String>) -> Option<String> {
    raw.map(|v| v.trim().to_string()).filter(|v| !v.is_empty())
}

impl Config {
    pub fn from_env() -> Result<Self, ProxyError> {
        let mcp_backend_url = std::env::var("MCP_BACKEND_URL").map_err(|_| {
            ProxyError::Config("MCP_BACKEND_URL env var not set".into())
        })?;

        let jwt_secret = std::env::var("CHORD_JWT_SECRET").unwrap_or_default();

        let tool_timeout_secs = std::env::var("CHORD_TOOL_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(30u64);

        let catalog_cache_secs = std::env::var("CHORD_CATALOG_CACHE_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(300u64);

        let listen_port = std::env::var("CHORD_PROXY_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(9099u16);

        let control_port = std::env::var("CHORD_CONTROL_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8090u16);

        // Treat an empty/blank CHORD_LLM_URL the same as unset → endpoint disabled (503).
        let llm_backend_url = normalize_llm_url(std::env::var("CHORD_LLM_URL").ok());

        // Model alias map (CHORD_MODEL_ALIASES JSON). Bad/empty → no rewriting.
        let model_aliases = parse_model_aliases(std::env::var("CHORD_MODEL_ALIASES").ok());

        let model_archive_path = std::env::var("MODEL_ARCHIVE_PATH")
            .unwrap_or_else(|_| "/var/lib/model-archive".into());

        let model_local_path =
            std::env::var("MODEL_LOCAL_PATH").unwrap_or_else(|_| "/opt/ollama-models".into());

        let model_protected = parse_protected_models(
            &std::env::var("MODEL_PROTECTED").unwrap_or_else(|_| {
                "lumina,lumina-fast,lumina-deep,qwen3-coder:30b,qwen3.6:35b-a3b,qwen3:8b".into()
            }),
        );

        let model_pull_timeout_secs = std::env::var("MODEL_PULL_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(600u64);

        let model_registry_path = std::env::var("MODEL_REGISTRY_PATH")
            .unwrap_or_else(|_| "/opt/chord/model-registry.json".into());

        let model_disk_pressure_percent = std::env::var("MODEL_DISK_PRESSURE_PERCENT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(80u8);

        let model_sweep_interval_secs = std::env::var("MODEL_SWEEP_INTERVAL_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1800u64);

        let model_warm_cooldown_hours = std::env::var("MODEL_WARM_COOLDOWN_HOURS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(168u64);

        Ok(Config {
            mcp_backend_url,
            jwt_secret,
            tool_timeout_secs,
            catalog_cache_secs,
            listen_port,
            control_port,
            rate_limits: RateLimitConfig::from_env(),
            llm_backend_url,
            model_aliases,
            model_archive_path,
            model_local_path,
            model_protected,
            model_pull_timeout_secs,
            model_registry_path,
            model_disk_pressure_percent,
            model_sweep_interval_secs,
            model_warm_cooldown_hours,
        })
    }
}

// ── S85 SRV-05: residency / VRAM-admission config helpers ─────────────────────
//
// The residency manager must read the host's FREE VRAM counter and persist a
// residency state file — both via env-sourced paths, NEVER a literal (S77 /
// pii_gate). A `None`/unreadable free-VRAM value is the FAIL-SAFE trigger (the
// caller treats it as "won't fit" and queues rather than risk an OOM launch), so
// these helpers must never invent a path or a number.

/// Read a trimmed, non-empty env var; `None` when unset or blank.
fn srv05_env_nonempty(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

/// Max cold-load (seconds) a model may have to be eligible as the PINNED,
/// interactive Lumina chat alias (SRV-06's latency guard). A model whose serving
/// profile cold-loads slower than this is too slow on first use to be the live
/// chat model — UNLESS it is `keep_warm` (held resident), in which case it is
/// allowed but flagged (warm residency mitigates steady-state latency; the first
/// cold-start still applies). From `CHORD_CHAT_PIN_MAX_COLD_LOAD_S`, default 300
/// (5 min) — generous enough for a warmed mid-size model, far below the ~8–10 min
/// big-MoE cold load that makes an interactive alias wrong.
pub fn chat_pin_max_cold_load_s() -> f64 {
    srv05_env_nonempty("CHORD_CHAT_PIN_MAX_COLD_LOAD_S")
        .and_then(|v| v.parse().ok())
        .unwrap_or(300.0)
}

/// Filesystem path of the sysfs counter exposing the GPU's FREE VRAM in BYTES
/// (e.g. the amdgpu `mem_info_vram_used`/`..._total` pair surfaced as a single
/// free counter by the operator's wrapper). From `CHORD_VRAM_FREE_SYSFS_PATH`;
/// `None` ⇒ no counter configured → the residency manager FAILS SAFE (treats free
/// VRAM as unreadable). No path is ever guessed (pii_gate).
pub fn vram_free_sysfs_path() -> Option<String> {
    srv05_env_nonempty("CHORD_VRAM_FREE_SYSFS_PATH")
}

/// Read the host's FREE VRAM in GB via the sysfs counter at
/// [`vram_free_sysfs_path`]. The counter file holds an integer number of BYTES.
///
/// Returns `None` (the FAIL-SAFE signal) when: no path is configured, the file
/// cannot be read (sysfs hiccup), or its contents do not parse as a byte count.
/// The residency manager turns any `None` into "won't fit → queue", never an
/// OOM-risking launch.
pub fn read_free_vram_gb() -> Option<f64> {
    let path = vram_free_sysfs_path()?;
    let raw = std::fs::read_to_string(&path).ok()?;
    let bytes: f64 = raw.trim().parse().ok()?;
    if bytes.is_finite() && bytes >= 0.0 {
        Some(bytes / 1_073_741_824.0) // bytes → GiB
    } else {
        None
    }
}

/// The SRV-12 release-verification tunables, from env (no literals): device idle
/// baseline + tolerance (GB), and the wait budget / poll interval. Defaults are
/// conservative — a generous timeout (cold backends can be slow to release) and a
/// 500 ms poll.
pub fn release_config() -> crate::serving::release_verify::ReleaseConfig {
    crate::serving::release_verify::ReleaseConfig {
        baseline_gb: srv05_env_nonempty("CHORD_RELEASE_BASELINE_GB")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.3),
        tolerance_gb: srv05_env_nonempty("CHORD_RELEASE_TOLERANCE_GB")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.5),
        timeout: std::time::Duration::from_millis(
            srv05_env_nonempty("CHORD_RELEASE_TIMEOUT_MS")
                .and_then(|v| v.parse().ok())
                .unwrap_or(30_000),
        ),
        poll_interval: std::time::Duration::from_millis(
            srv05_env_nonempty("CHORD_RELEASE_POLL_MS")
                .and_then(|v| v.parse().ok())
                .unwrap_or(500),
        ),
    }
}

/// The SRV-12 explicit-context defaults used when a model's profile carries no
/// `n_ctx` — a SAFE explicit context (never the backend's auto-fit). From env.
pub fn swap_context_defaults() -> crate::serving::swap::ContextDefaults {
    crate::serving::swap::ContextDefaults {
        base_ctx: srv05_env_nonempty("CHORD_SWAP_BASE_CTX")
            .and_then(|v| v.parse().ok())
            .unwrap_or(32768),
        min_ctx: srv05_env_nonempty("CHORD_SWAP_MIN_CTX")
            .and_then(|v| v.parse().ok())
            .unwrap_or(8192),
        large_model_gb: srv05_env_nonempty("CHORD_SWAP_LARGE_MODEL_GB")
            .and_then(|v| v.parse().ok())
            .unwrap_or(40.0),
    }
}

/// Free system RAM in GB, from `/proc/meminfo`'s `MemAvailable` line — the CPU
/// pool's free counter for the SRV-11 memory model (the genuine-CPU backend draws
/// system RAM). `/proc/meminfo` is a universal kernel interface, not infra, so it
/// is read directly. `None` ⇒ unreadable → the memory model FAILS SAFE.
pub fn read_cpu_free_gb() -> Option<f64> {
    let raw = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in raw.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            // Format: `MemAvailable:   28986336 kB`
            let kb: f64 = rest.split_whitespace().next()?.parse().ok()?;
            if kb.is_finite() && kb >= 0.0 {
                return Some(kb / 1_048_576.0); // kB → GiB
            }
        }
    }
    None
}

/// Sysfs path to the GPU's TOTAL VRAM carveout counter (`mem_info_vram_total`),
/// from `CHORD_VRAM_TOTAL_SYSFS_PATH`. Used only at boot for SRV-11 substrate
/// detection; `None` ⇒ detection cannot run → caller defaults to the safer model.
pub fn vram_total_sysfs_path() -> Option<String> {
    srv05_env_nonempty("CHORD_VRAM_TOTAL_SYSFS_PATH")
}

/// Sysfs path to the GPU's TOTAL GTT counter (`mem_info_gtt_total`), from
/// `CHORD_GTT_TOTAL_SYSFS_PATH`. Boot-time SRV-11 detection only.
pub fn gtt_total_sysfs_path() -> Option<String> {
    srv05_env_nonempty("CHORD_GTT_TOTAL_SYSFS_PATH")
}

/// Read a sysfs byte counter into GiB (shared by the total-VRAM/GTT readers).
fn read_sysfs_bytes_gb(path: &str) -> Option<f64> {
    let raw = std::fs::read_to_string(path).ok()?;
    let bytes: f64 = raw.trim().parse().ok()?;
    (bytes.is_finite() && bytes >= 0.0).then_some(bytes / 1_073_741_824.0)
}

/// Total system RAM in GB from `/proc/meminfo` `MemTotal` (boot-time detection).
fn read_system_ram_gb() -> Option<f64> {
    let raw = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in raw.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kb: f64 = rest.split_whitespace().next()?.parse().ok()?;
            return (kb.is_finite() && kb >= 0.0).then_some(kb / 1_048_576.0);
        }
    }
    None
}

/// Read the host's substrate facts (VRAM carveout, GTT total, system RAM) for
/// boot-time SRV-11 memory-model detection. `None` if any counter is unconfigured
/// or unreadable — the caller then defaults to the safer accounting model.
pub fn read_substrate_info() -> Option<crate::serving::memory_model::SubstrateInfo> {
    let vram = read_sysfs_bytes_gb(&vram_total_sysfs_path()?)?;
    let gtt = read_sysfs_bytes_gb(&gtt_total_sysfs_path()?)?;
    let ram = read_system_ram_gb()?;
    Some(crate::serving::memory_model::SubstrateInfo {
        vram_carveout_gb: vram,
        gtt_total_gb: gtt,
        system_ram_gb: ram,
    })
}

/// Path to the JSON residency-state file the manager writes atomically
/// (tempfile + rename) on every admit/queue/evict. From
/// `CHORD_RESIDENCY_STATE_PATH`; `None` ⇒ state-file persistence is disabled (the
/// manager still admits/evicts in memory — it just does not mirror to disk). No
/// path is guessed (pii_gate).
pub fn residency_state_path() -> Option<String> {
    srv05_env_nonempty("CHORD_RESIDENCY_STATE_PATH")
}

/// Bounded wait (milliseconds) the manager queues a keep-warm-contended launch
/// before it falls back to evicting an LRU keep-warm resident (and, if nothing is
/// evictable, denies admission). From `CHORD_RESIDENCY_WAIT_THRESHOLD_MS`,
/// default 30000 (30 s) — long enough to let a short generation finish, short
/// enough not to stall live Lumina.
pub fn residency_wait_threshold_ms() -> u64 {
    srv05_env_nonempty("CHORD_RESIDENCY_WAIT_THRESHOLD_MS")
        .and_then(|v| v.parse().ok())
        .unwrap_or(30_000)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_config_defaults_for_optional_fields() {
        // Set only the required field
        std::env::set_var("MCP_BACKEND_URL", "http://mcp-test-backend:3200");
        std::env::remove_var("CHORD_JWT_SECRET");
        std::env::remove_var("CHORD_TOOL_TIMEOUT_SECS");
        std::env::remove_var("CHORD_CATALOG_CACHE_SECS");
        std::env::remove_var("CHORD_PROXY_PORT");
        std::env::remove_var("CHORD_CONTROL_PORT");
        std::env::remove_var("CHORD_RATE_LLM_USER");
        std::env::remove_var("CHORD_RATE_TOOL_USER");
        std::env::remove_var("CHORD_RATE_DEEP_USER");
        std::env::remove_var("CHORD_RATE_LLM_GUEST");
        std::env::remove_var("CHORD_RATE_TOOL_GUEST");
        std::env::remove_var("CHORD_RATE_DEEP_GUEST");
        std::env::remove_var("CHORD_LLM_URL");

        let cfg = Config::from_env().unwrap();
        assert_eq!(cfg.mcp_backend_url, "http://mcp-test-backend:3200");
        assert_eq!(cfg.jwt_secret, "");
        assert_eq!(cfg.tool_timeout_secs, 30);
        assert_eq!(cfg.catalog_cache_secs, 300);
        assert_eq!(cfg.listen_port, 9099);
        assert_eq!(cfg.control_port, 8090);
        // Note: llm_backend_url is not asserted here — it derives from CHORD_LLM_URL
        // which other parallel env-based tests mutate. The normalize_llm_url unit
        // tests cover that logic deterministically.
        // Rate limit defaults
        assert_eq!(cfg.rate_limits.user_llm_limit, 200);
        assert_eq!(cfg.rate_limits.user_tool_limit, 500);
        assert_eq!(cfg.rate_limits.user_deep_limit, 50);
        assert_eq!(cfg.rate_limits.guest_llm_limit, 20);
        assert_eq!(cfg.rate_limits.guest_tool_limit, 50);
        assert_eq!(cfg.rate_limits.guest_deep_limit, 5);

        std::env::remove_var("MCP_BACKEND_URL");
    }

    #[test]
    #[serial]
    fn test_config_reads_custom_values() {
        std::env::set_var("MCP_BACKEND_URL", "http://custom-mcp:4000");
        std::env::set_var("CHORD_JWT_SECRET", "my-secret");
        std::env::set_var("CHORD_TOOL_TIMEOUT_SECS", "60");
        std::env::set_var("CHORD_CATALOG_CACHE_SECS", "120");
        std::env::set_var("CHORD_PROXY_PORT", "8888");
        std::env::set_var("CHORD_CONTROL_PORT", "8091");

        let cfg = Config::from_env().unwrap();
        assert_eq!(cfg.mcp_backend_url, "http://custom-mcp:4000");
        assert_eq!(cfg.jwt_secret, "my-secret");
        assert_eq!(cfg.tool_timeout_secs, 60);
        assert_eq!(cfg.catalog_cache_secs, 120);
        assert_eq!(cfg.listen_port, 8888);
        assert_eq!(cfg.control_port, 8091);

        std::env::remove_var("MCP_BACKEND_URL");
        std::env::remove_var("CHORD_JWT_SECRET");
        std::env::remove_var("CHORD_TOOL_TIMEOUT_SECS");
        std::env::remove_var("CHORD_CATALOG_CACHE_SECS");
        std::env::remove_var("CHORD_PROXY_PORT");
        std::env::remove_var("CHORD_CONTROL_PORT");
    }

    // `normalize_llm_url` is tested directly (no process-env mutation) to avoid
    // races with the other env-based config tests running in parallel.
    #[test]
    fn test_normalize_llm_url_keeps_real_value() {
        assert_eq!(
            normalize_llm_url(Some("http://localhost:11434/v1/chat/completions".into())).as_deref(),
            Some("http://localhost:11434/v1/chat/completions")
        );
    }

    #[test]
    fn test_normalize_llm_url_trims_whitespace() {
        assert_eq!(
            normalize_llm_url(Some("  http://host:11434/v1/chat/completions  ".into())).as_deref(),
            Some("http://host:11434/v1/chat/completions")
        );
    }

    #[test]
    fn test_normalize_llm_url_none_for_missing_or_blank() {
        assert!(normalize_llm_url(None).is_none());
        assert!(normalize_llm_url(Some(String::new())).is_none());
        assert!(
            normalize_llm_url(Some("   ".into())).is_none(),
            "blank CHORD_LLM_URL must be treated as None (endpoint disabled)"
        );
    }

    #[test]
    #[serial]
    fn test_config_missing_required_field_fails() {
        std::env::remove_var("MCP_BACKEND_URL");
        let result = Config::from_env();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("MCP_BACKEND_URL"));
    }

    #[test]
    fn test_parse_model_aliases_valid_json() {
        let m = parse_model_aliases(Some(
            r#"{"lumina-fast":"gpt-oss:20b","lumina-deep":"gpt-oss:120b"}"#.into(),
        ));
        assert_eq!(m.get("lumina-fast").map(String::as_str), Some("gpt-oss:20b"));
        assert_eq!(m.get("lumina-deep").map(String::as_str), Some("gpt-oss:120b"));
    }

    #[test]
    fn test_parse_model_aliases_missing_or_blank_or_bad_is_empty() {
        assert!(parse_model_aliases(None).is_empty());
        assert!(parse_model_aliases(Some(String::new())).is_empty());
        assert!(parse_model_aliases(Some("   ".into())).is_empty());
        // Malformed JSON must not panic — yields empty map (alias rewriting disabled).
        assert!(parse_model_aliases(Some("{not json".into())).is_empty());
    }

    #[test]
    fn test_resolve_model_alias_maps_known_passes_through_unknown() {
        let mut m = HashMap::new();
        m.insert("lumina-fast".to_string(), "gpt-oss:20b".to_string());
        // Known alias is rewritten.
        assert_eq!(resolve_model_alias(&m, "lumina-fast"), "gpt-oss:20b");
        // Unknown / already-real model passes through unchanged.
        assert_eq!(resolve_model_alias(&m, "gpt-oss:120b"), "gpt-oss:120b");
        // Empty alias map is pure pass-through.
        assert_eq!(resolve_model_alias(&HashMap::new(), "lumina-fast"), "lumina-fast");
    }

    #[test]
    fn test_parse_protected_models_trims_and_drops_empties() {
        let v = parse_protected_models(" lumina, lumina-fast ,, qwen3:8b , ");
        assert_eq!(v, vec!["lumina", "lumina-fast", "qwen3:8b"]);
        assert!(parse_protected_models("").is_empty());
        assert!(parse_protected_models("  , ,").is_empty());
    }

    #[test]
    #[serial]
    fn test_config_model_tier_defaults() {
        std::env::set_var("MCP_BACKEND_URL", "http://mcp-test-backend:3200");
        std::env::remove_var("MODEL_ARCHIVE_PATH");
        std::env::remove_var("MODEL_LOCAL_PATH");
        std::env::remove_var("MODEL_PROTECTED");

        std::env::remove_var("MODEL_PULL_TIMEOUT_SECS");
        std::env::remove_var("MODEL_DISK_PRESSURE_PERCENT");
        std::env::remove_var("MODEL_SWEEP_INTERVAL_SECS");
        std::env::remove_var("MODEL_WARM_COOLDOWN_HOURS");
        let cfg = Config::from_env().unwrap();
        assert_eq!(cfg.model_archive_path, "/var/lib/model-archive");
        assert_eq!(cfg.model_local_path, "/opt/ollama-models");
        assert!(cfg.model_protected.contains(&"lumina".to_string()));
        assert!(cfg.model_protected.contains(&"qwen3-coder:30b".to_string()));
        assert_eq!(cfg.model_protected.len(), 6);
        // Pull timeout default (TIER-02).
        assert_eq!(cfg.model_pull_timeout_secs, 600);
        // Eviction defaults (TIER-03).
        assert_eq!(cfg.model_disk_pressure_percent, 80);
        assert_eq!(cfg.model_sweep_interval_secs, 1800);
        // Cooldown default (TIER-04): 168h / 7 days.
        assert_eq!(cfg.model_warm_cooldown_hours, 168);

        std::env::remove_var("MCP_BACKEND_URL");
    }

    #[test]
    #[serial]
    fn test_config_model_tier_reads_env() {
        std::env::set_var("MCP_BACKEND_URL", "http://mcp-test-backend:3200");
        std::env::set_var("MODEL_ARCHIVE_PATH", "/custom/archive");
        std::env::set_var("MODEL_LOCAL_PATH", "/custom/local");
        std::env::set_var("MODEL_PROTECTED", "a,b , c");

        let cfg = Config::from_env().unwrap();
        assert_eq!(cfg.model_archive_path, "/custom/archive");
        assert_eq!(cfg.model_local_path, "/custom/local");
        assert_eq!(cfg.model_protected, vec!["a", "b", "c"]);

        std::env::set_var("MODEL_PULL_TIMEOUT_SECS", "120");
        std::env::set_var("MODEL_DISK_PRESSURE_PERCENT", "90");
        std::env::set_var("MODEL_SWEEP_INTERVAL_SECS", "60");
        std::env::set_var("MODEL_WARM_COOLDOWN_HOURS", "24");
        let cfg = Config::from_env().unwrap();
        assert_eq!(cfg.model_pull_timeout_secs, 120);
        assert_eq!(cfg.model_disk_pressure_percent, 90);
        assert_eq!(cfg.model_sweep_interval_secs, 60);
        assert_eq!(cfg.model_warm_cooldown_hours, 24);

        std::env::remove_var("MCP_BACKEND_URL");
        std::env::remove_var("MODEL_ARCHIVE_PATH");
        std::env::remove_var("MODEL_LOCAL_PATH");
        std::env::remove_var("MODEL_PROTECTED");
        std::env::remove_var("MODEL_PULL_TIMEOUT_SECS");
        std::env::remove_var("MODEL_DISK_PRESSURE_PERCENT");
        std::env::remove_var("MODEL_SWEEP_INTERVAL_SECS");
        std::env::remove_var("MODEL_WARM_COOLDOWN_HOURS");
    }

    #[test]
    #[serial]
    fn test_rate_limit_config_reads_env_vars() {
        std::env::set_var("CHORD_RATE_LLM_USER", "99");
        std::env::set_var("CHORD_RATE_TOOL_USER", "88");
        std::env::set_var("CHORD_RATE_DEEP_USER", "11");
        std::env::set_var("CHORD_RATE_LLM_GUEST", "7");
        std::env::set_var("CHORD_RATE_TOOL_GUEST", "14");
        std::env::set_var("CHORD_RATE_DEEP_GUEST", "2");

        let rl = RateLimitConfig::from_env();
        assert_eq!(rl.user_llm_limit, 99);
        assert_eq!(rl.user_tool_limit, 88);
        assert_eq!(rl.user_deep_limit, 11);
        assert_eq!(rl.guest_llm_limit, 7);
        assert_eq!(rl.guest_tool_limit, 14);
        assert_eq!(rl.guest_deep_limit, 2);

        std::env::remove_var("CHORD_RATE_LLM_USER");
        std::env::remove_var("CHORD_RATE_TOOL_USER");
        std::env::remove_var("CHORD_RATE_DEEP_USER");
        std::env::remove_var("CHORD_RATE_LLM_GUEST");
        std::env::remove_var("CHORD_RATE_TOOL_GUEST");
        std::env::remove_var("CHORD_RATE_DEEP_GUEST");
    }
}
