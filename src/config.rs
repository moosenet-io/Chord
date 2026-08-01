use std::collections::HashMap;
use std::path::PathBuf;

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

#[derive(Clone)]
pub struct Config {
    /// URL of the MCP backend — reads MCP_BACKEND_URL env var
    pub mcp_backend_url: String,
    /// JWT secret for validating incoming requests — reads CHORD_JWT_SECRET env var
    pub jwt_secret: String,
    /// Shared bearer token sent as `Authorization: Bearer <token>` on every outbound
    /// MCP request (initialize/tools-list/tools-call) to `MCP_BACKEND_URL`. Reads
    /// `MCP_BACKEND_TOKEN`. `None` (unset/blank) preserves the pre-hardening
    /// behavior of sending no `Authorization` header at all — this is intentional
    /// for backward-compatible rollout, not a silent failure. Never log this value;
    /// only ever log whether it is configured (see `config::mcp_backend_token`).
    pub mcp_backend_token: Option<String>,
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
    /// An unset or empty value yields an empty map (no rewriting). CHRD-82: a
    /// MALFORMED value no longer fails silently — it is logged loudly, reported on
    /// `/health`, and (for a partially-valid map) only the bad entries are dropped.
    /// See [`parse_model_aliases`].
    pub model_aliases: HashMap<String, String>,
    /// Archive (e.g. NFS) root that holds cold-tier Ollama models.
    /// Reads MODEL_ARCHIVE_PATH (default `/var/lib/model-archive`).
    pub model_archive_path: String,
    /// Local Ollama models root holding warm/hot models.
    /// Reads MODEL_LOCAL_PATH (default `/opt/ollama-models`).
    pub model_local_path: String,
    /// Protected model names that are never auto-archived. Reads MODEL_PROTECTED
    /// (comma-separated). Default: the core Lumina + qwen models.
    ///
    /// **DISK-tiering only.** This governs the warm→cold archive (eviction.rs) — a
    /// protected model's *blobs* stay on local disk. It says NOTHING about VRAM
    /// residency; VRAM residency is owned solely by `crate::routing::resident_set`.
    pub model_protected: Vec<String>,
    // CHRD-PIN-01: `model_keep_resident` / `model_keep_resident_rewarm_secs` /
    // `gpu_exclusive_exempt_keep_resident` (the old `MODEL_KEEP_RESIDENT`
    // `keep_alive:-1` pinning mechanism) were REMOVED here. VRAM residency now has
    // exactly ONE owner — `crate::routing::resident_set` — which holds its role
    // models on a BOUNDED `keep_alive` and RELEASES them on a mode swap. See the
    // module docs there.
    /// Maximum duration (seconds) for a cold→warm archive pull before it aborts
    /// and cleans up partial files. Reads MODEL_PULL_TIMEOUT_SECS (default 600).
    pub model_pull_timeout_secs: u64,
    /// Path to the JSON file backing the model registry (tier/size/timestamps).
    /// Reads MODEL_REGISTRY_PATH (default `<path>/model-registry.json`).
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
    /// MSM-02: maximum duration (seconds) for a single warm→cold eviction copy
    /// (the reverse of the TIER-02 pull) before it aborts, cleans up any partial
    /// archive files it wrote, and leaves the model Warm for retry on the next
    /// sweep. Reads MODEL_ARCHIVE_COPY_TIMEOUT_SECS (default 1800 = 30 min) —
    /// mirrors `model_pull_timeout_secs` for the opposite direction.
    pub model_archive_copy_timeout_secs: u64,
    /// MSM-03 (B1 defense-in-depth): minimum age (seconds) a local blob must
    /// have before the orphan-GC pass will consider deleting it. A blob whose
    /// mtime is within this window is skipped — an in-flight archive pull writes
    /// blobs to their final path before the referencing manifest lands, so a
    /// too-young "orphan" may actually be mid-copy. The primary guard is the
    /// shared disk-op lock (the pull-copy phase holds it), but this grace window
    /// protects against any future path that forgets the lock. Reads
    /// MODEL_GC_MIN_AGE_SECS (default 300 = 5 min).
    pub model_gc_min_age_secs: u64,
    /// S88 ISO-01: the egress allow-list of model-source hosts/domains a `Pull`
    /// runtime may reach. Reads `MODEL_SOURCE_ALLOWLIST` (comma/space separated).
    /// **UNSET → empty**, which makes every pull `Denied` (FAIL CLOSED). This is
    /// CONFIG, never a baked-in host; an empty list NEVER means allow-all.
    pub model_source_allowlist: Vec<String>,
    /// S88 ISO-01: an explicit Chord outbound proxy URL. Reads `CHORD_OUTBOUND_PROXY`.
    /// `None` (unset/blank) → runtime launches strip ALL proxy vars (they are never
    /// inherited from the supervisor). When set, an allow-listed `Pull` launch is
    /// given this proxy. `Serve` launches never get a proxy (no egress).
    pub outbound_proxy: Option<String>,
    /// S88 ISO-01: master toggle for the launch-env telemetry-off / offline opt-outs.
    /// Reads `CHORD_RUNTIME_TELEMETRY_OFF` (default `true`). When `false` the
    /// telemetry-off vars are still applied by [`crate::supervisor::launch_env`]
    /// unless a caller honours this flag — it is exposed for operators who must
    /// debug a runtime that misbehaves with the opt-outs set.
    pub runtime_telemetry_off: bool,
    /// Task 2 (federation): base URL of the standalone `terminus_personal`
    /// Rust MCP binary. Reads `PERSONAL_BACKEND_URL`. `None` (unset/blank) →
    /// the `/v1/personal/*` routes are disabled and return a clean 503, and
    /// Chord's second `McpProxy` instance is never constructed. This is
    /// deliberately optional — Chord must run fine with this unset, no hard
    /// dependency on `terminus_personal` being reachable.
    pub personal_backend_url: Option<String>,
    /// Bearer token attached as `Authorization: Bearer <token>` to every outbound
    /// request to `personal_backend_url`. Reads `PERSONAL_BACKEND_TOKEN`. This is
    /// `terminus_personal`'s own `TERMINUS_PERSONAL_TOKEN`, read from its `.env`
    /// at deploy time — never fetched fresh from <secret-manager> by Chord. Never logged.
    pub personal_backend_token: Option<String>,
}

/// Manual `Debug` impl: every field is passed through as the derive would,
/// EXCEPT `mcp_backend_token`, which is always redacted. This is a landmine
/// otherwise — nothing currently does `{:?}` on a whole `Config`, but a future
/// debug-log of it must never print the bearer token verbatim. Redaction text
/// mirrors the existing `chord-tui::secret::SecretValue` convention
/// (`"***redacted***"`) used elsewhere in this workspace for secret values.
impl std::fmt::Debug for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Config")
            .field("mcp_backend_url", &self.mcp_backend_url)
            .field("jwt_secret", &self.jwt_secret)
            .field(
                "mcp_backend_token",
                &self.mcp_backend_token.as_ref().map(|_| "***redacted***"),
            )
            .field("tool_timeout_secs", &self.tool_timeout_secs)
            .field("catalog_cache_secs", &self.catalog_cache_secs)
            .field("listen_port", &self.listen_port)
            .field("control_port", &self.control_port)
            .field("rate_limits", &self.rate_limits)
            .field("llm_backend_url", &self.llm_backend_url)
            .field("model_aliases", &self.model_aliases)
            .field("model_archive_path", &self.model_archive_path)
            .field("model_local_path", &self.model_local_path)
            .field("model_protected", &self.model_protected)
            .field("model_pull_timeout_secs", &self.model_pull_timeout_secs)
            .field("model_registry_path", &self.model_registry_path)
            .field(
                "model_disk_pressure_percent",
                &self.model_disk_pressure_percent,
            )
            .field(
                "model_sweep_interval_secs",
                &self.model_sweep_interval_secs,
            )
            .field(
                "model_warm_cooldown_hours",
                &self.model_warm_cooldown_hours,
            )
            .field(
                "model_archive_copy_timeout_secs",
                &self.model_archive_copy_timeout_secs,
            )
            .field("model_gc_min_age_secs", &self.model_gc_min_age_secs)
            .field("model_source_allowlist", &self.model_source_allowlist)
            .field("outbound_proxy", &self.outbound_proxy)
            .field("runtime_telemetry_off", &self.runtime_telemetry_off)
            .field("personal_backend_url", &self.personal_backend_url)
            .field(
                "personal_backend_token",
                &self.personal_backend_token.as_ref().map(|_| "***redacted***"),
            )
            .finish()
    }
}

/// Parse a comma/space-separated `MODEL_SOURCE_ALLOWLIST` value into a list of
/// host/domain strings, trimming whitespace and dropping empties. An empty result
/// (unset or blank) means NO source is allowed — the caller fails closed (never
/// allow-all). This is the S88 ISO-01 egress config surface.
pub fn parse_model_source_allowlist(raw: &str) -> Vec<String> {
    raw.split([',', ' ', '\t', '\n'])
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Parse a comma-separated `MODEL_PROTECTED` value into a list of names,
/// trimming whitespace and dropping empties.
pub fn parse_protected_models(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

// CHRD-PIN-01: the `model_keep_resident()` / `gpu_exclusive_exempt_keep_resident()`
// readers were REMOVED with the pinning mechanism they served. Nothing in Chord
// may pin a model with an indefinite `keep_alive` any more; residency is owned by
// `crate::routing::resident_set` alone, on a bounded keep_alive with an explicit
// mode-swap release.

// ── CHRD-82: CHORD_MODEL_ALIASES parsing (fails LOUD, never silently empty) ───
//
// `CHORD_MODEL_ALIASES` is a single hand-edited env value that backs EVERY alias
// on the host (`lumina-fast`, `lumina-deep`, `lumina-embed`, `coder-*`, `laguna`,
// `122b`, `llama`, `lumina-search`, …). Before CHRD-82 a parse error was swallowed
// with `unwrap_or_default()`, so one typo silently disabled all of them at once and
// the symptom surfaced far away — every resident role logging "alias resolves to no
// target", every alias caller missing — which reads as a residency/model bug and
// sends the investigation into `crate::routing::resident_set` instead of at config
// parsing. That misdirection was the real cost, not the lost rewriting.
//
// The posture now (same shape the fleet already uses for its other JSON-in-env
// config; see terminus-rs `AllowlistPolicy::from_env` / `PrincipalResolver::from_env`):
//   * The parse itself is PURE — it returns a structured [`AliasConfig`] carrying
//     both the map and a [`AliasConfigStatus`]; it never logs and never panics, so
//     it stays trivially unit-testable.
//   * Callers must handle the status. Startup logs a loud `ERROR` naming the env
//     var and the parse position, states the blast radius in the message, and says
//     in so many words that this is a CONFIG failure rather than a residency bug.
//   * The degraded state is visible on `/health` (`model_aliases.status`), so it is
//     discoverable without reading logs.
//   * A validation path (`--check-config`) lets an operator catch the typo BEFORE
//     restarting the service — see [`check_config_report`].
//
// Why degrade-and-shout rather than refuse to start: this fleet restarts services
// routinely (the nightly constellation-updater health-gates and restarts every
// module), so making a hand-edited env typo abort startup converts a config mistake
// into a Chord outage plus an updater rollback cycle. A loud, health-visible,
// pre-checkable degradation keeps inference serving while making the fault
// impossible to misattribute.

/// What happened while parsing `CHORD_MODEL_ALIASES`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AliasConfigStatus {
    /// Parsed cleanly. This INCLUDES the legitimately-unset case (no env var at
    /// all ⇒ zero aliases, which is a valid configuration, not a fault).
    Ok,
    /// The value was a well-formed JSON object, but individual entries were
    /// unusable and were dropped. The remaining entries are live. Each element is
    /// an operator-readable description of one dropped entry.
    EntriesDropped { dropped: Vec<String> },
    /// The value could not be parsed as a JSON object at all — no alias survived.
    /// `error` is serde's message, which carries the line/column of the fault.
    Unparseable { error: String },
}

impl AliasConfigStatus {
    /// `true` for anything other than a clean parse.
    pub fn is_degraded(&self) -> bool {
        !matches!(self, AliasConfigStatus::Ok)
    }

    /// A single-line, operator-facing explanation. `None` when not degraded.
    /// Never includes the raw env value — only the parse position and the names
    /// of the entries that were dropped.
    pub fn detail(&self) -> Option<String> {
        match self {
            AliasConfigStatus::Ok => None,
            AliasConfigStatus::EntriesDropped { dropped } => Some(format!(
                "CHORD_MODEL_ALIASES: {} malformed entr{} dropped ({}); the remaining aliases ARE live",
                dropped.len(),
                if dropped.len() == 1 { "y was" } else { "ies were" },
                dropped.join("; ")
            )),
            AliasConfigStatus::Unparseable { error } => Some(format!(
                "CHORD_MODEL_ALIASES is not a valid JSON object ({error}) — ALL model alias \
                 rewriting is DISABLED for this process (0 aliases loaded). Every alias \
                 (lumina-fast/lumina-deep/lumina-embed/coder-*/…) will MISS and every resident \
                 role will report that its alias resolves to no target. This is a CONFIG PARSE \
                 failure, NOT a residency/model/backend bug — fix the JSON in the service's \
                 environment file and restart. Validate it BEFORE restarting with: \
                 `harmony-chord --check-config`"
            )),
        }
    }
}

/// The parsed alias map plus how the parse went. See the module-level CHRD-82 note.
#[derive(Debug, Clone)]
pub struct AliasConfig {
    /// The usable alias → backend-model map. Empty only when nothing was
    /// configured or nothing survived parsing (see `status` to tell those apart).
    pub aliases: HashMap<String, String>,
    pub status: AliasConfigStatus,
}

impl AliasConfig {
    /// Emit the loud operator-facing log for a degraded parse. A no-op when the
    /// parse was clean. `source` names the call site (e.g. `"startup"`) so a
    /// duplicated warning from a second reader is attributable.
    ///
    /// A wholly-unparseable value is an `ERROR` (every alias lost); dropped
    /// individual entries are a `WARN` (the rest still work).
    pub fn log_if_degraded(&self, source: &str) {
        match &self.status {
            AliasConfigStatus::Ok => {}
            AliasConfigStatus::EntriesDropped { .. } => {
                if let Some(detail) = self.status.detail() {
                    tracing::warn!("{source}: {detail}");
                }
            }
            AliasConfigStatus::Unparseable { .. } => {
                if let Some(detail) = self.status.detail() {
                    tracing::error!("{source}: {detail}");
                }
            }
        }
    }
}

/// Process-wide record of how the live alias config parsed, so `/health` can
/// surface a degraded aliasing state without threading a new field through
/// `AppState` (and through its ~19 construction sites). Written once by
/// [`Config::from_env`]; read by the health handler.
static ALIAS_HEALTH: std::sync::RwLock<Option<(AliasConfigStatus, usize)>> =
    std::sync::RwLock::new(None);

/// Record the live alias-config outcome for `/health`. Last write wins (a
/// poisoned lock is ignored rather than panicking a request path).
pub fn record_alias_config_health(cfg: &AliasConfig) {
    if let Ok(mut slot) = ALIAS_HEALTH.write() {
        *slot = Some((cfg.status.clone(), cfg.aliases.len()));
    }
}

/// The recorded alias-config outcome and loaded-alias count, or `None` when the
/// process never parsed one (e.g. a test binary that never built a `Config`).
pub fn alias_config_health() -> Option<(AliasConfigStatus, usize)> {
    ALIAS_HEALTH.read().ok().and_then(|slot| slot.clone())
}

/// A JSON value's kind, for error messages (never the value itself).
fn json_kind(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "a boolean",
        serde_json::Value::Number(_) => "a number",
        serde_json::Value::String(_) => "a string",
        serde_json::Value::Array(_) => "an array",
        serde_json::Value::Object(_) => "an object",
    }
}

/// Bound an operator-supplied name before putting it in a log/report line.
fn truncate_for_report(s: &str) -> String {
    const MAX: usize = 64;
    if s.chars().count() <= MAX {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(MAX).collect::<String>())
    }
}

/// Parse a raw `CHORD_MODEL_ALIASES` value (a JSON object of alias→model) into an
/// [`AliasConfig`].
///
/// * Missing / empty / whitespace-only ⇒ empty map, [`AliasConfigStatus::Ok`]
///   (a host with no aliases configured is a valid configuration).
/// * Well-formed JSON object with some unusable entries ⇒ the usable entries are
///   KEPT and the bad ones are dropped individually
///   ([`AliasConfigStatus::EntriesDropped`]). Salvaging is deliberate here: the
///   whole point of CHRD-82 is that one bad entry must not take down every other
///   alias on the host. (Note this is stricter than terminus-rs's grant map, where
///   a single malformed entry does fail the whole map — that policy is right for a
///   security allowlist, where partial application would be a privilege question,
///   and wrong here, where partial application is strictly less damage.)
/// * Anything else — a syntax error, or valid JSON that is not an object ⇒ empty
///   map, [`AliasConfigStatus::Unparseable`] carrying serde's line/column.
///
/// This function NEVER logs and NEVER panics; the caller decides how to shout.
pub fn parse_model_aliases(raw: Option<String>) -> AliasConfig {
    let Some(text) = raw.map(|v| v.trim().to_string()).filter(|v| !v.is_empty()) else {
        return AliasConfig {
            aliases: HashMap::new(),
            status: AliasConfigStatus::Ok,
        };
    };

    let value: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => {
            return AliasConfig {
                aliases: HashMap::new(),
                status: AliasConfigStatus::Unparseable {
                    error: e.to_string(),
                },
            }
        }
    };

    let serde_json::Value::Object(obj) = value else {
        return AliasConfig {
            aliases: HashMap::new(),
            status: AliasConfigStatus::Unparseable {
                error: format!(
                    "expected a JSON object mapping alias → model, found {}",
                    json_kind(&value)
                ),
            },
        };
    };

    let mut aliases = HashMap::new();
    let mut dropped = Vec::new();
    for (key, val) in obj {
        let alias = key.trim();
        if alias.is_empty() {
            dropped.push("\"\": blank alias name".to_string());
            continue;
        }
        match val {
            serde_json::Value::String(target) if !target.trim().is_empty() => {
                aliases.insert(alias.to_string(), target.trim().to_string());
            }
            serde_json::Value::String(_) => dropped.push(format!(
                "\"{}\": empty target model",
                truncate_for_report(alias)
            )),
            other => dropped.push(format!(
                "\"{}\": target is not a JSON string (found {})",
                truncate_for_report(alias),
                json_kind(&other)
            )),
        }
    }

    let status = if dropped.is_empty() {
        AliasConfigStatus::Ok
    } else {
        AliasConfigStatus::EntriesDropped { dropped }
    };
    AliasConfig { aliases, status }
}

/// The `--check-config` validation path: parse `raw` exactly as the service would
/// and render an operator-facing report plus a process exit code (`0` clean,
/// `1` degraded in any way). Pure, so the flag's behaviour is unit-testable.
///
/// Reports alias NAMES (which are not secrets) so an operator can confirm the entry
/// they just hand-edited actually landed; never echoes the raw env value.
pub fn check_config_report(raw: Option<String>) -> (String, i32) {
    let cfg = parse_model_aliases(raw);
    let mut names: Vec<&str> = cfg.aliases.keys().map(String::as_str).collect();
    names.sort_unstable();

    match &cfg.status {
        AliasConfigStatus::Ok => (
            format!(
                "CHORD_MODEL_ALIASES: OK — {} alias(es) loaded: {}",
                names.len(),
                if names.is_empty() {
                    "(none configured)".to_string()
                } else {
                    names.join(", ")
                }
            ),
            0,
        ),
        AliasConfigStatus::EntriesDropped { dropped } => (
            format!(
                "CHORD_MODEL_ALIASES: DEGRADED — {} alias(es) would load: {}\n  \
                 dropped {} malformed entr{}:\n{}\n  \
                 Fix these before restarting; the listed aliases would NOT resolve.",
                names.len(),
                names.join(", "),
                dropped.len(),
                if dropped.len() == 1 { "y" } else { "ies" },
                dropped
                    .iter()
                    .map(|d| format!("    - {d}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            ),
            1,
        ),
        AliasConfigStatus::Unparseable { error } => (
            format!(
                "CHORD_MODEL_ALIASES: INVALID — {error}\n  \
                 0 aliases would load: EVERY alias on this host would MISS \
                 (lumina-fast/lumina-deep/lumina-embed/coder-*/…) and every resident role \
                 would report that its alias resolves to no target.\n  \
                 DO NOT restart the service until this is fixed."
            ),
            1,
        ),
    }
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

        // Outbound MCP auth. Loud-if-missing (mirrors the ISO-01 allow-list posture)
        // but NOT fail-closed: an unset token preserves today's unauthenticated
        // behavior so rollout can happen without a coordinated flag-day. The token
        // value itself is never logged — only whether it is present.
        let mcp_backend_token = mcp_backend_token(&mcp_backend_url);

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

        // Model alias map (CHORD_MODEL_ALIASES JSON). CHRD-82: a parse failure is
        // LOUD (ERROR naming the var + parse position, WARN per dropped entry) and
        // is recorded for `/health`, instead of silently yielding an empty map.
        // Deliberately NOT a startup abort — see the CHRD-82 note above
        // `parse_model_aliases`.
        let alias_config = parse_model_aliases(std::env::var("CHORD_MODEL_ALIASES").ok());
        alias_config.log_if_degraded("startup config");
        record_alias_config_health(&alias_config);
        let model_aliases = alias_config.aliases;

        let model_archive_path = std::env::var("MODEL_ARCHIVE_PATH")
            .unwrap_or_else(|_| "/var/lib/model-archive".into());

        let model_local_path =
            std::env::var("MODEL_LOCAL_PATH").unwrap_or_else(|_| "/opt/ollama-models".into());

        let model_protected = parse_protected_models(
            &std::env::var("MODEL_PROTECTED").unwrap_or_else(|_| {
                "lumina-fast,lumina-deep,qwen3-coder:30b,qwen3.6:35b-a3b,qwen3:8b".into()
            }),
        );


        let model_pull_timeout_secs = std::env::var("MODEL_PULL_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(600u64);

        let model_registry_path = std::env::var("MODEL_REGISTRY_PATH")
            .unwrap_or_else(|_| "<path>/model-registry.json".into());

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

        let model_archive_copy_timeout_secs = std::env::var("MODEL_ARCHIVE_COPY_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1800u64);

        let model_gc_min_age_secs = std::env::var("MODEL_GC_MIN_AGE_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(300u64);

        // S88 ISO-01: egress config surface. Allow-list is loud-on-empty (fail closed).
        let model_source_allowlist = model_source_allowlist();
        let outbound_proxy = std::env::var("CHORD_OUTBOUND_PROXY")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty());
        let runtime_telemetry_off = runtime_telemetry_off();

        // Task 2: optional federation to `terminus_personal`. Unset →
        // `/v1/personal/*` disabled, no second McpProxy instantiated.
        let personal_backend_url = normalize_llm_url(std::env::var("PERSONAL_BACKEND_URL").ok());
        let personal_backend_token = std::env::var("PERSONAL_BACKEND_TOKEN")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty());
        if personal_backend_url.is_some() && personal_backend_token.is_none() {
            tracing::warn!(
                target: "chord.personal",
                "PERSONAL_BACKEND_URL is set but PERSONAL_BACKEND_TOKEN is not — \
                 outbound requests to terminus_personal will be UNAUTHENTICATED"
            );
        }

        Ok(Config {
            mcp_backend_url,
            jwt_secret,
            mcp_backend_token,
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
            model_archive_copy_timeout_secs,
            model_gc_min_age_secs,
            model_source_allowlist,
            outbound_proxy,
            runtime_telemetry_off,
            personal_backend_url,
            personal_backend_token,
        })
    }

    /// A minimal `Config` for unit tests that need a `Config` value without reading
    /// the process environment. All fields take their documented defaults; ISO-01
    /// fields default to the FAIL-CLOSED posture (empty allow-list, no proxy).
    #[cfg(test)]
    pub fn test_default() -> Self {
        Config {
            mcp_backend_url: "http://mcp.invalid:3200".into(),
            jwt_secret: String::new(),
            mcp_backend_token: None,
            tool_timeout_secs: 30,
            catalog_cache_secs: 300,
            listen_port: 9099,
            control_port: 8090,
            rate_limits: RateLimitConfig::default(),
            llm_backend_url: None,
            model_aliases: HashMap::new(),
            model_archive_path: "/var/lib/model-archive".into(),
            model_local_path: "/opt/ollama-models".into(),
            model_protected: Vec::new(),
            model_pull_timeout_secs: 600,
            model_registry_path: "<path>/model-registry.json".into(),
            model_disk_pressure_percent: 80,
            model_sweep_interval_secs: 1800,
            model_warm_cooldown_hours: 168,
            model_archive_copy_timeout_secs: 1800,
            model_gc_min_age_secs: 300,
            model_source_allowlist: Vec::new(),
            outbound_proxy: None,
            runtime_telemetry_off: true,
            personal_backend_url: None,
            personal_backend_token: None,
        }
    }
}

/// S88 ISO-01 egress config surface: read `MODEL_SOURCE_ALLOWLIST` into the list of
/// model-source hosts a `Pull` may reach (comma/space split).
///
/// **UNSET → empty list + a loud `tracing::warn!`** that all pulls will be Denied
/// until the allow-list is configured. This NEVER defaults to allow-all: an
/// unconfigured Chord fails closed. A deployment's real value comes from its egress
/// audit; the examples in `.env.example` (registry.ollama.ai, huggingface.co) are
/// public placeholders only.
pub fn model_source_allowlist() -> Vec<String> {
    match std::env::var("MODEL_SOURCE_ALLOWLIST") {
        Ok(raw) => {
            let list = parse_model_source_allowlist(&raw);
            if list.is_empty() {
                tracing::warn!(
                    target: "chord.supervisor",
                    "MODEL_SOURCE_ALLOWLIST is set but empty after parsing — all model \
                     pulls will be DENIED (egress fail-closed) until it is configured"
                );
            }
            list
        }
        Err(_) => {
            tracing::warn!(
                target: "chord.supervisor",
                "MODEL_SOURCE_ALLOWLIST is unset — all model pulls will be DENIED \
                 (egress fail-closed) until it is configured; ISO-01 never defaults \
                 to allow-all"
            );
            Vec::new()
        }
    }
}

/// S88 ISO-01 telemetry toggle: read `CHORD_RUNTIME_TELEMETRY_OFF` (default `true`).
/// Controls whether runtime launches advertise the telemetry-off / offline opt-outs.
/// Any value other than a case-insensitive `false`/`0`/`no` keeps the opt-outs ON.
/// Read `MCP_BACKEND_TOKEN`: the shared bearer token attached as
/// `Authorization: Bearer <token>` to every outbound MCP request. Unset or
/// blank → `None`, which preserves today's unauthenticated behavior (backward
/// compatible during rollout) — this is deliberately NOT fail-closed like the
/// ISO-01 egress allow-list, since the remote backend does not enforce auth
/// yet. A missing token still gets a loud one-line `tracing::warn!` at startup
/// so the pre-hardening state is visible in logs. **The token value itself is
/// never included in any log line, here or at any call site.**
pub fn mcp_backend_token(mcp_backend_url: &str) -> Option<String> {
    let token = std::env::var("MCP_BACKEND_TOKEN")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());

    if token.is_none() && !mcp_backend_url.trim().is_empty() {
        tracing::warn!(
            target: "chord.mcp",
            "MCP_BACKEND_URL is set but MCP_BACKEND_TOKEN is not — outbound MCP \
             requests to the backend are UNAUTHENTICATED (pre-hardening state); \
             set MCP_BACKEND_TOKEN once the backend enforces bearer auth"
        );
    }

    token
}

pub fn runtime_telemetry_off() -> bool {
    match std::env::var("CHORD_RUNTIME_TELEMETRY_OFF") {
        Ok(v) => !matches!(v.trim().to_ascii_lowercase().as_str(), "false" | "0" | "no" | "off"),
        Err(_) => true,
    }
}

/// S88 ISO-02: the `ip` (iproute2) binary used to create/configure/teardown the
/// per-runtime network namespace. Reads `CHORD_IP_BIN`; when unset, falls back to a
/// bare `ip` (resolved on `PATH`). Returns `None` only when explicitly set to blank
/// (an operator disabling the iproute2 path). Never a hardcoded absolute path.
pub fn ip_bin() -> Option<String> {
    match std::env::var("CHORD_IP_BIN") {
        Ok(v) if v.trim().is_empty() => None,
        Ok(v) => Some(v.trim().to_string()),
        Err(_) => Some("ip".to_string()),
    }
}

/// S88 ISO-02: the `nft` (nftables) binary used to apply the `Pull` namespace's
/// egress allow-list filter. Reads `CHORD_NFT_BIN`; when unset, falls back to a
/// bare `nft` (resolved on `PATH`). `None` only when explicitly blanked.
pub fn nft_bin() -> Option<String> {
    match std::env::var("CHORD_NFT_BIN") {
        Ok(v) if v.trim().is_empty() => None,
        Ok(v) => Some(v.trim().to_string()),
        Err(_) => Some("nft".to_string()),
    }
}

// ── RESIL-01: durable runtime-state directory ─────────────────────────────────
//
// A small directory for Chord's durable runtime state files that must survive a
// Chord process restart — currently just the persisted GPU-exclusive lease
// (`gpu_exclusive.rs`). Resolved from `CHORD_STATE_DIR`; `None` when unset or
// blank, so callers degrade to in-memory-only behavior rather than inventing a
// path (S1 — never a hardcoded/guessed absolute path).

/// The durable runtime-state directory, from `CHORD_STATE_DIR`. `None` when
/// unset or blank — callers must treat that as "no persistence", never a guess.
pub fn chord_state_dir() -> Option<PathBuf> {
    std::env::var("CHORD_STATE_DIR")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
}

/// Path to the persisted GPU-exclusive lease file within [`chord_state_dir`].
/// `None` when `CHORD_STATE_DIR` is unset/blank (persistence disabled).
pub fn gpu_exclusive_state_path() -> Option<PathBuf> {
    chord_state_dir().map(|d| d.join("gpu_exclusive_lease.json"))
}

/// Path to the persisted sweep-session store (RESIL-02) within
/// [`chord_state_dir`]. `None` when `CHORD_STATE_DIR` is unset/blank (the
/// session store is then in-memory only, lost on restart).
pub fn sweep_session_state_path() -> Option<PathBuf> {
    chord_state_dir().map(|d| d.join("sweep_sessions.json"))
}

/// BLD-09: path to the persisted idle-mode resume manifest within
/// [`chord_state_dir`]. `None` when `CHORD_STATE_DIR` is unset/blank — idle-mode
/// state is then in-memory only (a crash mid-idle relies on the watchdog rather
/// than a reloaded manifest). Same "never a hardcoded/guessed path" discipline
/// (S1) as the other state-file helpers above.
pub fn admin_idle_state_path() -> Option<PathBuf> {
    chord_state_dir().map(|d| d.join("admin_idle_state.json"))
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

    #[test]
    fn test_config_debug_redacts_mcp_backend_token() {
        let mut cfg = Config::test_default();
        cfg.mcp_backend_token = Some("hunter2-super-secret".to_string());
        let debug_str = format!("{cfg:?}");
        assert!(!debug_str.contains("hunter2-super-secret"));
        assert!(debug_str.contains("***redacted***"));

        cfg.mcp_backend_token = None;
        let debug_str = format!("{cfg:?}");
        assert!(!debug_str.contains("***redacted***"));
        assert!(debug_str.contains("mcp_backend_token: None"));
    }

    #[test]
    #[serial]
    fn test_mcp_backend_token_absent_when_env_unset() {
        std::env::remove_var("MCP_BACKEND_TOKEN");
        std::env::set_var("MCP_BACKEND_URL", "http://mcp-test-backend:3200");

        let cfg = Config::from_env().unwrap();
        assert_eq!(cfg.mcp_backend_token, None);

        std::env::remove_var("MCP_BACKEND_URL");
    }

    #[test]
    #[serial]
    fn test_mcp_backend_token_absent_when_env_blank() {
        std::env::set_var("MCP_BACKEND_URL", "http://mcp-test-backend:3200");
        std::env::set_var("MCP_BACKEND_TOKEN", "   ");

        let cfg = Config::from_env().unwrap();
        assert_eq!(cfg.mcp_backend_token, None);

        std::env::remove_var("MCP_BACKEND_URL");
        std::env::remove_var("MCP_BACKEND_TOKEN");
    }

    #[test]
    #[serial]
    fn test_mcp_backend_token_present_when_env_set() {
        std::env::set_var("MCP_BACKEND_URL", "http://mcp-test-backend:3200");
        std::env::set_var("MCP_BACKEND_TOKEN", "shared-secret-123");

        let cfg = Config::from_env().unwrap();
        assert_eq!(cfg.mcp_backend_token, Some("shared-secret-123".to_string()));

        std::env::remove_var("MCP_BACKEND_URL");
        std::env::remove_var("MCP_BACKEND_TOKEN");
    }

    #[test]
    #[serial]
    fn test_mcp_backend_token_fn_trims_and_filters_blank() {
        // Direct unit test of the free function (no shared process-env access
        // beyond MCP_BACKEND_TOKEN itself) — kept separate from the `#[serial]`
        // Config::from_env tests above since it does not touch MCP_BACKEND_URL.
        std::env::set_var("MCP_BACKEND_TOKEN", "  padded-token  ");
        assert_eq!(
            mcp_backend_token("http://x"),
            Some("padded-token".to_string())
        );
        std::env::remove_var("MCP_BACKEND_TOKEN");

        std::env::remove_var("MCP_BACKEND_TOKEN");
        assert_eq!(mcp_backend_token("http://x"), None);
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
        let cfg = parse_model_aliases(Some(
            r#"{"lumina-fast":"gpt-oss:20b","lumina-deep":"gpt-oss:120b"}"#.into(),
        ));
        let m = &cfg.aliases;
        assert_eq!(m.get("lumina-fast").map(String::as_str), Some("gpt-oss:20b"));
        assert_eq!(m.get("lumina-deep").map(String::as_str), Some("gpt-oss:120b"));
        assert_eq!(cfg.status, AliasConfigStatus::Ok);
    }

    #[test]
    fn test_parse_model_aliases_missing_or_blank_is_empty_and_ok() {
        // An unconfigured host is NOT a fault: empty map, clean status, no shouting.
        for raw in [None, Some(String::new()), Some("   ".to_string())] {
            let cfg = parse_model_aliases(raw);
            assert!(cfg.aliases.is_empty());
            assert_eq!(cfg.status, AliasConfigStatus::Ok);
            assert!(!cfg.status.is_degraded());
        }
    }

    // ── CHRD-82 ──────────────────────────────────────────────────────────────
    // The guard: a malformed CHORD_MODEL_ALIASES must NEVER resolve to a silent
    // empty map. Before CHRD-82 this exact input returned `HashMap::new()` with no
    // signal at all, which disabled every alias on the host at once and made the
    // failure look like a residency/model bug.

    #[test]
    fn test_wholly_unparseable_json_is_loud_never_a_silent_empty_map() {
        let cfg = parse_model_aliases(Some(r#"{"lumina-fast":"gpt-oss:20b",}"#.into()));
        // No aliases survive a broken map...
        assert!(cfg.aliases.is_empty());
        // ...but the failure is REPORTED, which is the whole point.
        assert!(cfg.status.is_degraded(), "must not fail silently");
        let AliasConfigStatus::Unparseable { ref error } = cfg.status else {
            panic!("expected Unparseable, got {:?}", cfg.status);
        };
        // serde's message carries the parse position.
        assert!(error.contains("line 1"), "no parse position in {error:?}");
        let detail = cfg.status.detail().expect("degraded status must explain itself");
        assert!(detail.contains("CHORD_MODEL_ALIASES"), "must name the env var");
        assert!(
            detail.contains("DISABLED"),
            "must state the blast radius: {detail}"
        );
        // The misdirection killer: say out loud this is NOT a residency bug.
        assert!(
            detail.contains("NOT a residency"),
            "must point away from resident_set: {detail}"
        );
    }

    #[test]
    fn test_valid_json_that_is_not_an_object_is_unparseable() {
        for raw in [r#"["lumina-fast"]"#, r#""lumina-fast""#, "42", "null"] {
            let cfg = parse_model_aliases(Some(raw.to_string()));
            assert!(cfg.aliases.is_empty());
            assert!(
                matches!(cfg.status, AliasConfigStatus::Unparseable { .. }),
                "{raw} should be Unparseable, got {:?}",
                cfg.status
            );
        }
    }

    #[test]
    fn test_single_malformed_entry_drops_only_that_entry() {
        // Well-formed JSON object, one bad value: every OTHER alias must survive.
        // This is the deliberate divergence from an all-or-nothing parse — one typo
        // must not take the whole host's aliasing down.
        let cfg = parse_model_aliases(Some(
            r#"{"lumina-fast":"gpt-oss:20b","lumina-embed":123,"lumina-deep":"gpt-oss:120b","coder-fast":"","":"x"}"#
                .into(),
        ));
        assert_eq!(
            cfg.aliases.get("lumina-fast").map(String::as_str),
            Some("gpt-oss:20b")
        );
        assert_eq!(
            cfg.aliases.get("lumina-deep").map(String::as_str),
            Some("gpt-oss:120b")
        );
        assert_eq!(cfg.aliases.len(), 2, "only the good entries survive");
        assert!(!cfg.aliases.contains_key("lumina-embed"));
        assert!(!cfg.aliases.contains_key("coder-fast"));

        let AliasConfigStatus::EntriesDropped { ref dropped } = cfg.status else {
            panic!("expected EntriesDropped, got {:?}", cfg.status);
        };
        assert_eq!(dropped.len(), 3, "dropped: {dropped:?}");
        let joined = dropped.join(" | ");
        // Each dropped entry is NAMED, so the operator knows which alias went dark.
        assert!(joined.contains("lumina-embed"), "{joined}");
        assert!(joined.contains("coder-fast"), "{joined}");
        assert!(cfg.status.is_degraded());
    }

    #[test]
    fn test_positive_control_realistic_map_parses_every_alias() {
        // Proves the guard did not just start rejecting everything: the shape of a
        // real host's CHORD_MODEL_ALIASES must still parse, all of it.
        let cfg = parse_model_aliases(Some(
            r#"{"lumina":"granite4.1","lumina-fast":"granite4.1","lumina-deep":"gpt-oss:120b",
                "lumina-embed":"nomic-embed-text","lumina-search":"granite4.1",
                "coder-fast":"qwen3-coder:30b","coder-deep":"qwen3-coder:30b",
                "laguna":"granite4.1","122b":"gpt-oss:120b","llama":"llama3.1:8b"}"#
                .into(),
        ));
        assert_eq!(cfg.status, AliasConfigStatus::Ok);
        assert!(!cfg.status.is_degraded());
        assert_eq!(cfg.aliases.len(), 10);
        for (alias, target) in [
            ("lumina", "granite4.1"),
            ("lumina-fast", "granite4.1"),
            ("lumina-deep", "gpt-oss:120b"),
            ("lumina-embed", "nomic-embed-text"),
            ("lumina-search", "granite4.1"),
            ("coder-fast", "qwen3-coder:30b"),
            ("coder-deep", "qwen3-coder:30b"),
            ("laguna", "granite4.1"),
            ("122b", "gpt-oss:120b"),
            ("llama", "llama3.1:8b"),
        ] {
            assert_eq!(
                resolve_model_alias(&cfg.aliases, alias),
                target,
                "alias {alias} must resolve"
            );
        }
        // And an unknown model still passes through untouched.
        assert_eq!(resolve_model_alias(&cfg.aliases, "granite4.1"), "granite4.1");
    }

    #[test]
    fn test_check_config_rejects_a_bad_file_and_accepts_a_good_one() {
        // The operator-facing validation path: non-zero exit on anything degraded,
        // zero on a good config, BEFORE the service is restarted.
        let (good, code) =
            check_config_report(Some(r#"{"lumina-fast":"granite4.1","122b":"gpt-oss:120b"}"#.into()));
        assert_eq!(code, 0, "good config must pass: {good}");
        assert!(good.contains("OK"), "{good}");
        assert!(good.contains("lumina-fast") && good.contains("122b"), "{good}");

        let (bad, code) = check_config_report(Some(r#"{"lumina-fast":"granite4.1",}"#.into()));
        assert_eq!(code, 1, "unparseable config must fail: {bad}");
        assert!(bad.contains("INVALID"), "{bad}");
        assert!(bad.contains("DO NOT restart"), "{bad}");

        let (partial, code) =
            check_config_report(Some(r#"{"lumina-fast":"granite4.1","lumina-embed":123}"#.into()));
        assert_eq!(code, 1, "a dropped entry must also fail the check: {partial}");
        assert!(partial.contains("DEGRADED"), "{partial}");
        assert!(partial.contains("lumina-embed"), "{partial}");

        // An unconfigured host is legitimately clean.
        let (none, code) = check_config_report(None);
        assert_eq!(code, 0, "{none}");
        assert!(none.contains("none configured"), "{none}");
    }

    #[test]
    fn test_alias_entries_are_trimmed() {
        let cfg = parse_model_aliases(Some(r#"{" lumina-fast ":"  granite4.1 "}"#.into()));
        assert_eq!(
            cfg.aliases.get("lumina-fast").map(String::as_str),
            Some("granite4.1")
        );
        assert_eq!(cfg.status, AliasConfigStatus::Ok);
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
        std::env::remove_var("MODEL_ARCHIVE_COPY_TIMEOUT_SECS");
        std::env::remove_var("MODEL_GC_MIN_AGE_SECS");
        let cfg = Config::from_env().unwrap();
        assert_eq!(cfg.model_archive_path, "/var/lib/model-archive");
        assert_eq!(cfg.model_local_path, "/opt/ollama-models");
        // The retired `lumina` main alias was dropped from the default set.
        assert!(!cfg.model_protected.contains(&"lumina".to_string()));
        assert!(cfg.model_protected.contains(&"lumina-fast".to_string()));
        assert!(cfg.model_protected.contains(&"lumina-deep".to_string()));
        assert!(cfg.model_protected.contains(&"qwen3-coder:30b".to_string()));
        assert_eq!(cfg.model_protected.len(), 5);
        // Pull timeout default (TIER-02).
        assert_eq!(cfg.model_pull_timeout_secs, 600);
        // Eviction defaults (TIER-03).
        assert_eq!(cfg.model_disk_pressure_percent, 80);
        assert_eq!(cfg.model_sweep_interval_secs, 1800);
        // Cooldown default (TIER-04): 168h / 7 days.
        assert_eq!(cfg.model_warm_cooldown_hours, 168);
        // MSM-02/MSM-03 defaults.
        assert_eq!(cfg.model_archive_copy_timeout_secs, 1800);
        assert_eq!(cfg.model_gc_min_age_secs, 300);

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
        std::env::set_var("MODEL_ARCHIVE_COPY_TIMEOUT_SECS", "900");
        std::env::set_var("MODEL_GC_MIN_AGE_SECS", "60");
        let cfg = Config::from_env().unwrap();
        assert_eq!(cfg.model_pull_timeout_secs, 120);
        assert_eq!(cfg.model_disk_pressure_percent, 90);
        assert_eq!(cfg.model_sweep_interval_secs, 60);
        assert_eq!(cfg.model_warm_cooldown_hours, 24);
        assert_eq!(cfg.model_archive_copy_timeout_secs, 900);
        assert_eq!(cfg.model_gc_min_age_secs, 60);

        std::env::remove_var("MCP_BACKEND_URL");
        std::env::remove_var("MODEL_ARCHIVE_PATH");
        std::env::remove_var("MODEL_LOCAL_PATH");
        std::env::remove_var("MODEL_PROTECTED");
        std::env::remove_var("MODEL_PULL_TIMEOUT_SECS");
        std::env::remove_var("MODEL_DISK_PRESSURE_PERCENT");
        std::env::remove_var("MODEL_SWEEP_INTERVAL_SECS");
        std::env::remove_var("MODEL_WARM_COOLDOWN_HOURS");
        std::env::remove_var("MODEL_ARCHIVE_COPY_TIMEOUT_SECS");
        std::env::remove_var("MODEL_GC_MIN_AGE_SECS");
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
