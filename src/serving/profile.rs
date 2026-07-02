//! Serving-profile reader → in-memory routing map (S85 SRV-04, consume side).
//!
//! SRV-01 (`terminus-rs::intake::serving`) defines the `serving_profile` table
//! and the shared [`ServingProfile`] / [`Runtime`] / [`ServingBackend`] /
//! [`ExclusionReason`] types — the data contract. This module is the Chord-side
//! *reader*: it loads those rows into a [`RoutingMap`] keyed by `model_id`, so a
//! caller that asks for a model can be routed to the correct runtime + env
//! without re-deriving any serving facts.
//!
//! ## Where the rows come from
//! Loading goes through a [`ProfileSource`] trait, not a hardwired DB call, so:
//!   - production uses [`DbProfileSource`] (a `sqlx::PgPool` to the intake DB,
//!     reusing the SRV-01 connection helper — NO literals: the URL is sourced via
//!     `terminus_rs::config::intake_database_url`);
//!   - tests inject a deterministic [`StaticProfileSource`] (no Postgres needed).
//!
//! ## The launch env (`env_json`)
//! Each row carries an `env_json` JSON object the SRV-02 runner recorded: the
//! gfx override, the mmap flag, flash-attn, and the cpu lib. Chord REPLAYS it
//! verbatim into the launch command (see [`launcher`](super::launcher)). The
//! shape is captured by [`EnvSpec`] so the launcher reads typed fields instead of
//! poking at raw JSON. An absent/blank/malformed `env_json` yields an all-default
//! [`EnvSpec`] (no flags) rather than a hard failure — a row missing optional
//! launch hints is still routable.
//!
//! ## Refresh
//! [`RoutingMap`] is rebuilt wholesale on startup and on every refresh
//! ([`RoutingMap::load`]); there is no partial mutation, so a refresh can never
//! leave a half-updated map.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use terminus_rs::intake::serving::{ModelId, Runtime, ServingProfile};

/// Typed view of a serving row's `env_json` launch hints.
///
/// SRV-02 writes these as a JSON object; SRV-04 replays them. To keep the
/// producer/consumer data contract honest, the parser accepts BOTH the runner's
/// native key/type shape AND an explicit-string shape — a mismatch here silently
/// drops the load-bearing `--no-mmap` flag, exactly the v1 false-hang the v2 sweep
/// fixed. Recognized keys (all optional):
///   - `gfx_override` — apply the host's `HSA_OVERRIDE_GFX_VERSION` for the HIP
///     `llama-server` / ollama-rocm bring-up on gfx1151. Accepts a **bool** (the
///     runner's shape: `true` ⇒ apply the host gfx override from
///     [`terminus_rs::config::gfx_override_version`]; `false` ⇒ omit) OR an explicit **string**
///     value (`""` ⇒ the CPU tier's deliberate empty override, distinct from
///     absent; non-empty ⇒ that literal value).
///   - `mmap` (bool) / `mmap_flag` (int) — llama.cpp mmap toggle. `mmap: false`
///     or `mmap_flag: 0` ⇒ the launcher passes `--no-mmap` (the v2 lesson: mmap
///     over NFS page-faults into a false hang for staged/large weights). `true` /
///     non-zero ⇒ mmap stays on (no flag). Absent ⇒ runtime default.
///   - `flash_attn` (bool) — llama.cpp flash-attention toggle. `true` ⇒
///     `--flash-attn`.
///   - `cpu_library` / `cpu_lib` (string) — explicit cpu-runtime library for the
///     genuine-CPU tier (the empty-gfx-override path).
///   - `rope_scaling` (object, YARN-01) — an optional context-extension block for
///     the llama.cpp tier: `method` (`none`/`linear`/`yarn`, default `none`),
///     `rope_scale`, `yarn_orig_ctx`, `target_ctx`, `ext_factor`, `attn_factor`,
///     `beta_slow`, `beta_fast`, and `validated` (bool, default `false`). See
///     [`RopeScaling`] for the emission rules the launcher applies.
///   - `thinking` (object, YARN-05) — an optional thinking-trace-preservation
///     capability block for the llama.cpp tier: `supports_thinking` (bool,
///     default `false` — whether the model itself can emit/preserve a
///     reasoning trace), `preserve_thinking` (bool, default `false` — whether
///     Chord should ask the runtime to preserve it across turns),
///     `requires_prefix_caching` (bool, default `false` — whether preservation
///     needs prefix caching to avoid re-processing the trace each turn), and
///     `validated` (bool, default `false`, same gfx1151-validation gate as
///     [`RopeScaling`]). See [`ThinkingConfig`] for the emission rules the
///     launcher applies.
///
/// Unknown keys are ignored (forward-compatible: a newer runner can add hints an
/// older Chord harmlessly skips).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EnvSpec {
    /// Explicit `HSA_OVERRIDE_GFX_VERSION` value. `Some("")` is meaningful (the CPU
    /// tier's deliberate empty override); `None` ⇒ no explicit value (see
    /// [`gfx_apply_host_default`](Self::gfx_apply_host_default)).
    pub gfx_override: Option<String>,
    /// The runner's `gfx_override: true` shape: apply the host's configured gfx
    /// override at launch (resolved from [`terminus_rs::config::gfx_override_version`], never a
    /// literal). Mutually exclusive with an explicit [`gfx_override`](Self::gfx_override).
    pub gfx_apply_host_default: bool,
    /// llama.cpp mmap toggle. `Some(false)` ⇒ emit `--no-mmap`; `Some(true)` ⇒
    /// leave mmap on (no flag); `None` ⇒ key absent (runtime default).
    pub mmap: Option<bool>,
    /// llama.cpp flash-attention toggle. `Some(true)` ⇒ emit `--flash-attn`.
    pub flash_attn: Option<bool>,
    /// Explicit cpu-runtime library override (CPU tier).
    pub cpu_library: Option<String>,
    /// Explicit context window (`-c`). When present, the launcher pins the context
    /// so llama.cpp's auto-fit never sizes it against the UMA-misread free memory
    /// (the SRV-12 context-slash sidestep). Parsed from `n_ctx` / `ctx`. `None` ⇒
    /// the swap barrier computes a safe default from the model's footprint.
    pub n_ctx: Option<u32>,
    /// YARN-01: optional context-extension config for the llama.cpp tier. `None`
    /// ⇒ no `rope_scaling` key present (unchanged behavior — no scaling flags).
    /// `Some(rope)` with `rope.method == RopeScalingMethod::None` is equivalent to
    /// absent. See [`RopeScaling`] for the launcher's emission rules.
    pub rope_scaling: Option<RopeScaling>,
    /// YARN-05: optional thinking-trace-preservation capability for the
    /// llama.cpp tier. `None` ⇒ no `thinking` key present (unchanged behavior —
    /// no preservation/prefix-caching flags). See [`ThinkingConfig`] for the
    /// launcher's emission rules.
    pub thinking: Option<ThinkingConfig>,
}

/// YARN-01: the llama.cpp RoPE context-extension method a [`RopeScaling`] block
/// requests. Mirrors llama.cpp's `--rope-scaling` argument vocabulary.
///
/// YARN-02 adds `Serialize`/`Deserialize` so the ingestion pipeline can persist
/// a pre-filled block on a [`crate::models::registry::ModelRecord`] (the local
/// model registry's own JSON file) — a separate persistence path from the
/// `env_json` string this module otherwise parses by hand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RopeScalingMethod {
    /// No context extension (unchanged native-context behavior).
    #[default]
    None,
    /// Linear RoPE scaling (`--rope-scaling linear`).
    Linear,
    /// YaRN RoPE scaling (`--rope-scaling yarn`), with the yarn-specific
    /// fine-tune flags (orig-ctx / ext-factor / attn-factor / beta-slow /
    /// beta-fast).
    Yarn,
}

impl RopeScalingMethod {
    /// The literal value the llama.cpp `--rope-scaling` flag expects.
    pub fn as_str(self) -> &'static str {
        match self {
            RopeScalingMethod::None => "none",
            RopeScalingMethod::Linear => "linear",
            RopeScalingMethod::Yarn => "yarn",
        }
    }

    /// Parse the `env_json` `method` string. Unknown values are rejected (`None`
    /// return, not defaulted) so a typo'd method never silently becomes a no-op —
    /// the caller treats a parse failure as an invalid config (warn, don't emit).
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "none" => Some(RopeScalingMethod::None),
            "linear" => Some(RopeScalingMethod::Linear),
            "yarn" => Some(RopeScalingMethod::Yarn),
            _ => None,
        }
    }
}

/// YARN-01: a per-model RoPE context-extension config (llama.cpp tier only). Read
/// from the serving profile's `env_json.rope_scaling` object; the ollama and CPU
/// launchers ignore it (context extension is unavailable on those tiers — see
/// [`super::launcher`]).
///
/// All numeric values are sourced from the profile row — the launcher never
/// invents or hardcodes a scaling constant. `validated` gates emission: a
/// `method != none` block that has not been validated on gfx1151 serves at native
/// context (no yarn flags) rather than risk an unvalidated launch.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct RopeScaling {
    /// The requested scaling method. `None` (the enum variant) ⇒ no extension.
    pub method: RopeScalingMethod,
    /// `--rope-scale` value (linear and yarn).
    pub rope_scale: f64,
    /// `--yarn-orig-ctx` value (yarn only).
    pub yarn_orig_ctx: u32,
    /// The desired extended context; becomes `--ctx-size` when the block is
    /// emitted.
    pub target_ctx: u32,
    /// `--yarn-ext-factor` value (yarn only).
    pub ext_factor: f64,
    /// `--yarn-attn-factor` value (yarn only).
    pub attn_factor: f64,
    /// `--yarn-beta-slow` value (yarn only).
    pub beta_slow: f64,
    /// `--yarn-beta-fast` value (yarn only).
    pub beta_fast: f64,
    /// Whether this config has been validated on gfx1151. `false` ⇒ the launcher
    /// refuses to emit yarn flags even though the block is present (serves at
    /// native context, with a warning).
    pub validated: bool,
}

impl RopeScaling {
    /// YARN-01 plausibility gate: a `validated=true` block can still carry
    /// garbage numbers (e.g. a bad manual edit, a unit mix-up) — this is the
    /// last line of defense before the launcher emits them verbatim to
    /// `llama-server`. Deliberately conservative sanity bounds, NOT a physics
    /// model: `rope_scale` must be positive; for `yarn`, `ext_factor` /
    /// `attn_factor` must fall in `[0.0, 1.0]` and both betas must be
    /// non-negative. `method == None` is trivially plausible (nothing is
    /// emitted for it anyway).
    pub fn is_plausible(&self) -> bool {
        match self.method {
            RopeScalingMethod::None => true,
            RopeScalingMethod::Linear => self.rope_scale > 0.0,
            RopeScalingMethod::Yarn => {
                self.rope_scale > 0.0
                    && (0.0..=1.0).contains(&self.ext_factor)
                    && (0.0..=1.0).contains(&self.attn_factor)
                    && self.beta_slow >= 0.0
                    && self.beta_fast >= 0.0
            }
        }
    }
}

/// Parse the `rope_scaling` object out of an already-parsed `env_json` map.
///
/// Returns `None` (no block, i.e. unchanged native-context behavior) when the key
/// is absent, malformed, names an unknown method, or is missing a field the
/// chosen method requires — every one of those is an INVALID config, logged and
/// treated as absent rather than emitted partially or guessed.
fn parse_rope_scaling(obj: &serde_json::Map<String, serde_json::Value>) -> Option<RopeScaling> {
    let rs = obj.get("rope_scaling")?.as_object()?;

    let method = match rs.get("method").and_then(|v| v.as_str()) {
        Some(s) => match RopeScalingMethod::parse(s) {
            Some(m) => m,
            None => {
                tracing::warn!(
                    target: "chord.serving.profile",
                    method = s,
                    "rope_scaling.method unrecognized — ignoring config"
                );
                return None;
            }
        },
        None => RopeScalingMethod::None,
    };
    let validated = rs.get("validated").and_then(|v| v.as_bool()).unwrap_or(false);

    if method == RopeScalingMethod::None {
        return Some(RopeScaling {
            method,
            validated,
            ..Default::default()
        });
    }

    let rope_scale = rs.get("rope_scale").and_then(|v| v.as_f64());
    let target_ctx = rs
        .get("target_ctx")
        .and_then(|v| v.as_u64())
        .and_then(|n| u32::try_from(n).ok());

    // Linear needs only `rope_scale` + `target_ctx`; yarn additionally needs the
    // original-context + fine-tune quartet.
    let (rope_scale, target_ctx) = match (rope_scale, target_ctx) {
        (Some(a), Some(b)) => (a, b),
        _ => {
            tracing::warn!(
                target: "chord.serving.profile",
                method = method.as_str(),
                "rope_scaling missing required field (rope_scale/target_ctx) — ignoring config"
            );
            return None;
        }
    };

    if method == RopeScalingMethod::Linear {
        return Some(RopeScaling {
            method,
            rope_scale,
            target_ctx,
            validated,
            ..Default::default()
        });
    }

    // method == Yarn: the orig-ctx + fine-tune quartet is required.
    let yarn_orig_ctx = rs
        .get("yarn_orig_ctx")
        .and_then(|v| v.as_u64())
        .and_then(|n| u32::try_from(n).ok());
    let ext_factor = rs.get("ext_factor").and_then(|v| v.as_f64());
    let attn_factor = rs.get("attn_factor").and_then(|v| v.as_f64());
    let beta_slow = rs.get("beta_slow").and_then(|v| v.as_f64());
    let beta_fast = rs.get("beta_fast").and_then(|v| v.as_f64());

    match (yarn_orig_ctx, ext_factor, attn_factor, beta_slow, beta_fast) {
        (Some(yarn_orig_ctx), Some(ext_factor), Some(attn_factor), Some(beta_slow), Some(beta_fast)) => {
            Some(RopeScaling {
                method,
                rope_scale,
                yarn_orig_ctx,
                target_ctx,
                ext_factor,
                attn_factor,
                beta_slow,
                beta_fast,
                validated,
            })
        }
        _ => {
            tracing::warn!(
                target: "chord.serving.profile",
                "rope_scaling method=yarn missing a required field (yarn_orig_ctx/ext_factor/attn_factor/beta_slow/beta_fast) — ignoring config"
            );
            None
        }
    }
}

/// YARN-05: a per-model thinking-trace-preservation capability (llama.cpp tier
/// only). Read from the serving profile's `env_json.thinking` object; the
/// ollama and CPU launchers ignore it (context/prompt-cache preservation is
/// unavailable on those tiers — see [`super::launcher`]).
///
/// All flags are sourced from the profile row — the launcher never invents
/// whether a model supports thinking (that is populated by ingestion, e.g. a
/// small YARN-02-style addition to the model-config reader, not a new pipeline
/// this item builds). `validated` gates emission exactly like [`RopeScaling`]:
/// a `preserve_thinking` request that has not been validated on gfx1151 serves
/// without preservation (no flags) rather than risk an unvalidated launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ThinkingConfig {
    /// Whether the model itself is capable of emitting/preserving a thinking
    /// (reasoning) trace. `false` ⇒ nothing to preserve — a `preserve_thinking`
    /// request on such a model is an invalid config (see
    /// [`super::launcher::thinking_args`]).
    pub supports_thinking: bool,
    /// Whether Chord should ask the runtime to preserve the thinking trace
    /// across turns rather than discard it after each response. Requesting
    /// this with `supports_thinking == false` never emits anything — the
    /// launcher refuses and logs a note.
    pub preserve_thinking: bool,
    /// Whether preservation needs prefix caching to avoid re-processing the
    /// (potentially large) trace on every turn. Without this, `preserve_thinking`
    /// still may be requested, but forfeits the point on a tier that supports it
    /// — the launcher notes the memory-growth cost when it enables caching.
    pub requires_prefix_caching: bool,
    /// Whether this config has been validated on gfx1151. `false` ⇒ the
    /// launcher refuses to emit the preservation/prefix-caching flags even
    /// though the block is present (serves without preservation, with a
    /// warning) — mirrors [`RopeScaling::validated`] exactly.
    pub validated: bool,
}

/// Parse the `thinking` object out of an already-parsed `env_json` map.
///
/// Returns `None` (no block, i.e. unchanged behavior — no preservation flags)
/// when the key is absent or malformed. Unlike [`parse_rope_scaling`], every
/// field here is an optional bool with a safe `false` default, so there is no
/// "missing required field" rejection case — an object with unrecognized or
/// partially-present keys still yields a (conservatively all-`false`) config
/// rather than being discarded outright.
fn parse_thinking(obj: &serde_json::Map<String, serde_json::Value>) -> Option<ThinkingConfig> {
    let t = obj.get("thinking")?.as_object()?;
    Some(ThinkingConfig {
        supports_thinking: t.get("supports_thinking").and_then(|v| v.as_bool()).unwrap_or(false),
        preserve_thinking: t.get("preserve_thinking").and_then(|v| v.as_bool()).unwrap_or(false),
        requires_prefix_caching: t
            .get("requires_prefix_caching")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        validated: t.get("validated").and_then(|v| v.as_bool()).unwrap_or(false),
    })
}

impl EnvSpec {
    /// Parse a raw `env_json` string into a typed spec. A missing, blank, or
    /// malformed value yields the all-default spec (no flags) — never an error,
    /// so a row without launch hints is still routable.
    pub fn parse(env_json: &str) -> Self {
        let trimmed = env_json.trim();
        if trimmed.is_empty() {
            return EnvSpec::default();
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) else {
            return EnvSpec::default();
        };
        let obj = match v.as_object() {
            Some(o) => o,
            None => return EnvSpec::default(),
        };

        // gfx_override: a string is an explicit value; a bool is the runner's
        // "apply the host default" flag (true) or "omit" (false).
        let (gfx_override, gfx_apply_host_default) = match obj.get("gfx_override") {
            Some(serde_json::Value::String(s)) => (Some(s.clone()), false),
            Some(serde_json::Value::Bool(b)) => (None, *b),
            _ => (None, false),
        };

        // mmap: prefer the explicit `mmap` bool; otherwise accept the runner's
        // `mmap_flag` integer (0 ⇒ --no-mmap, non-zero ⇒ mmap on).
        let mmap = obj
            .get("mmap")
            .and_then(|x| x.as_bool())
            .or_else(|| obj.get("mmap_flag").and_then(|x| x.as_u64()).map(|n| n != 0));

        // cpu library: accept either key name.
        let cpu_library = obj
            .get("cpu_library")
            .and_then(|x| x.as_str())
            .or_else(|| obj.get("cpu_lib").and_then(|x| x.as_str()))
            .map(|s| s.to_string());

        // explicit context window: accept `n_ctx` or `ctx`.
        let n_ctx = obj
            .get("n_ctx")
            .and_then(|x| x.as_u64())
            .or_else(|| obj.get("ctx").and_then(|x| x.as_u64()))
            .and_then(|n| u32::try_from(n).ok());

        let rope_scaling = parse_rope_scaling(obj);
        let thinking = parse_thinking(obj);

        EnvSpec {
            gfx_override,
            gfx_apply_host_default,
            mmap,
            flash_attn: obj.get("flash_attn").and_then(|x| x.as_bool()),
            cpu_library,
            n_ctx,
            rope_scaling,
            thinking,
        }
    }
}

/// One model's resolved routing entry: the full SRV-01 profile row plus its
/// parsed [`EnvSpec`]. Held in the [`RoutingMap`]; the launcher reads from it.
#[derive(Debug, Clone)]
pub struct RouteEntry {
    /// The serving-profile row as loaded from SRV-01.
    pub profile: ServingProfile,
    /// The parsed launch env (gfx / mmap / flash-attn / cpu lib).
    pub env: EnvSpec,
}

impl RouteEntry {
    /// Build a route entry from a serving row, parsing its `env_json`.
    pub fn from_profile(profile: ServingProfile) -> Self {
        let env = EnvSpec::parse(&profile.env_json);
        RouteEntry { profile, env }
    }

    /// The runtime Chord should launch this model under.
    pub fn best_runtime(&self) -> Runtime {
        self.profile.best_runtime
    }

    /// The fallback runtime to try if `best_runtime` launch/health fails.
    pub fn fallback_runtime(&self) -> Option<Runtime> {
        self.profile.fallback_runtime
    }

    /// Whether this model is held resident (must go through the residency manager,
    /// never cold-launched inline on the request hot path).
    pub fn keep_warm(&self) -> bool {
        self.profile.keep_warm
    }

    /// The model's VRAM/RAM footprint in GB (from the profile), if measured. Used
    /// by the residency manager's admission check for keep-warm models.
    pub fn vram_gb(&self) -> Option<f64> {
        self.profile.vram_or_ram_peak_gb
    }
}

/// Source of serving-profile rows. Abstracted so production reads Postgres while
/// tests inject a static set — the routing map never hardwires a DB call.
#[async_trait::async_trait]
pub trait ProfileSource: Send + Sync {
    /// Load ALL serving-profile rows (one per model × backend). The routing map
    /// keys them by `model_id`; a model with multiple backend rows resolves to
    /// its best routable row (see [`RoutingMap::load_from`]).
    async fn load_all(&self) -> Result<Vec<ServingProfile>, ProfileLoadError>;
}

/// A profile-load failure, already genericized — carries NO infra detail (host,
/// path, DSN), only a stable reason code, so it is safe to surface (S77).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProfileLoadError {
    /// The intake DB connection is not configured (no URL helper resolved).
    NotConfigured,
    /// The profile store could not be read (connection / query failure). The
    /// underlying detail is logged internally, never carried here.
    StoreUnavailable,
}

impl std::fmt::Display for ProfileLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProfileLoadError::NotConfigured => {
                f.write_str("serving profile store is not configured")
            }
            ProfileLoadError::StoreUnavailable => {
                f.write_str("serving profile store is temporarily unavailable")
            }
        }
    }
}

impl std::error::Error for ProfileLoadError {}

/// In-memory routing map: `model_id` → [`RouteEntry`]. Rebuilt wholesale on
/// startup and refresh; lookups are a simple map get.
///
/// When a model has rows for more than one backend, the map keeps the most
/// preferred routable row in tier order (llama-gpu → ollama-gpu → cpu), skipping
/// rows whose `exclusion_reason` marks the runtime unusable — so a lookup yields
/// the row Chord should actually launch.
#[derive(Debug, Clone, Default)]
pub struct RoutingMap {
    by_model: HashMap<String, RouteEntry>,
}

impl RoutingMap {
    /// Build an empty map (no profiles loaded yet).
    pub fn empty() -> Self {
        RoutingMap {
            by_model: HashMap::new(),
        }
    }

    /// Load (or refresh) the map from a [`ProfileSource`]. Replaces the whole map
    /// atomically from the caller's perspective: it builds a fresh map and only
    /// returns it on success, so a failed refresh never corrupts a live map.
    pub async fn load(source: &dyn ProfileSource) -> Result<Self, ProfileLoadError> {
        let rows = source.load_all().await?;
        Ok(Self::load_from(rows))
    }

    /// Build a map from an already-loaded set of rows (the testable core).
    ///
    /// Backend preference: when two rows share a `model_id`, the more preferred
    /// tier wins (llama-gpu < ollama-gpu < cpu), but a row whose `best_runtime` is
    /// excluded for a non-`none` reason loses to any usable row. This keeps a
    /// lookup pointing at the row Chord can actually launch.
    pub fn load_from(rows: Vec<ServingProfile>) -> Self {
        use terminus_rs::intake::serving::{ExclusionReason, ServingBackend};

        // Lower rank = more preferred.
        fn tier_rank(b: ServingBackend) -> u8 {
            match b {
                ServingBackend::LlamaGpu => 0,
                ServingBackend::OllamaGpu => 1,
                ServingBackend::Cpu => 2,
            }
        }

        let mut by_model: HashMap<String, RouteEntry> = HashMap::new();
        for row in rows {
            let key = row.model_id.as_str().to_string();
            let candidate = RouteEntry::from_profile(row);

            match by_model.get(&key) {
                None => {
                    by_model.insert(key, candidate);
                }
                Some(existing) => {
                    let cand_usable =
                        candidate.profile.exclusion_reason == ExclusionReason::None;
                    let ex_usable = existing.profile.exclusion_reason == ExclusionReason::None;
                    let replace = match (cand_usable, ex_usable) {
                        // Usable beats unusable.
                        (true, false) => true,
                        (false, true) => false,
                        // Both same usability → prefer the better tier.
                        _ => {
                            tier_rank(candidate.profile.backend_tag)
                                < tier_rank(existing.profile.backend_tag)
                        }
                    };
                    if replace {
                        by_model.insert(key, candidate);
                    }
                }
            }
        }
        RoutingMap { by_model }
    }

    /// Look up a model's route entry by `model_id`. `None` ⇒ unprofiled model
    /// (the launcher turns this into a clear "unprofiled" error — never a guess).
    pub fn get(&self, model_id: &ModelId) -> Option<&RouteEntry> {
        self.by_model.get(model_id.as_str())
    }

    /// Number of distinct models routable.
    pub fn len(&self) -> usize {
        self.by_model.len()
    }

    /// Whether the map has no routable models.
    pub fn is_empty(&self) -> bool {
        self.by_model.is_empty()
    }
}

/// Production [`ProfileSource`]: reads the SRV-01 `serving_profile` table over a
/// `sqlx::PgPool`. The pool is built from the intake DB URL via the SRV-01
/// connection helper — NO literal DSN/host here (S77 / pii_gate).
pub struct DbProfileSource {
    pool: sqlx::PgPool,
}

impl DbProfileSource {
    /// Wrap an existing pool (caller owns connection lifecycle).
    pub fn new(pool: sqlx::PgPool) -> Self {
        DbProfileSource { pool }
    }

    /// Connect a pool via the SRV-01 helper (intake DB URL from config/vault, no
    /// literal). `NotConfigured` when no URL is resolvable — never guesses a host.
    pub async fn connect() -> Result<Self, ProfileLoadError> {
        let url = terminus_rs::config::intake_database_url()
            .ok_or(ProfileLoadError::NotConfigured)?;
        let pool = sqlx::PgPool::connect(&url)
            .await
            .map_err(|e| {
                // Detail is logged, NOT surfaced (no DSN/host leak).
                tracing::error!(error = %e, "serving profile DB connect failed");
                ProfileLoadError::StoreUnavailable
            })?;
        Ok(DbProfileSource { pool })
    }
}

#[async_trait::async_trait]
impl ProfileSource for DbProfileSource {
    async fn load_all(&self) -> Result<Vec<ServingProfile>, ProfileLoadError> {
        use sqlx::Row;
        use terminus_rs::intake::serving::{ExclusionReason, RecheckTrigger, Runtime, ServingBackend};

        let rows = sqlx::query(
            "SELECT model_id, backend_tag, best_runtime, env_json::text AS env_json, \
                    tok_s, vram_or_ram_peak_gb, cold_load_s, keep_warm, \
                    fallback_runtime, exclusion_reason, recheck_trigger, provenance \
             FROM serving_profile",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| {
            tracing::error!(error = %e, "serving profile query failed");
            ProfileLoadError::StoreUnavailable
        })?;

        let mut out = Vec::with_capacity(rows.len());
        for r in rows {
            let backend_tag = ServingBackend::parse(r.get::<String, _>("backend_tag").as_str())
                .ok_or(ProfileLoadError::StoreUnavailable)?;
            let best_runtime = Runtime::parse(r.get::<String, _>("best_runtime").as_str())
                .ok_or(ProfileLoadError::StoreUnavailable)?;
            let fallback_runtime = r
                .get::<Option<String>, _>("fallback_runtime")
                .and_then(|s| Runtime::parse(&s));
            let exclusion_reason =
                ExclusionReason::parse(r.get::<String, _>("exclusion_reason").as_str())
                    .ok_or(ProfileLoadError::StoreUnavailable)?;
            let recheck_trigger =
                RecheckTrigger::parse(r.get::<String, _>("recheck_trigger").as_str())
                    .ok_or(ProfileLoadError::StoreUnavailable)?;

            out.push(ServingProfile {
                model_id: ModelId::from_registry_key(r.get::<String, _>("model_id")),
                backend_tag,
                best_runtime,
                env_json: r.get::<Option<String>, _>("env_json").unwrap_or_default(),
                tok_s: r.get("tok_s"),
                vram_or_ram_peak_gb: r.get("vram_or_ram_peak_gb"),
                cold_load_s: r.get("cold_load_s"),
                keep_warm: r.get("keep_warm"),
                fallback_runtime,
                exclusion_reason,
                recheck_trigger,
                provenance: r.get("provenance"),
            });
        }
        Ok(out)
    }
}

/// Test/seed [`ProfileSource`]: serves a fixed set of rows with no DB. Used by the
/// SRV-04 tests and available for a config-seeded routing map.
pub struct StaticProfileSource {
    rows: Vec<ServingProfile>,
}

impl StaticProfileSource {
    /// Wrap a fixed set of rows.
    pub fn new(rows: Vec<ServingProfile>) -> Self {
        StaticProfileSource { rows }
    }
}

#[async_trait::async_trait]
impl ProfileSource for StaticProfileSource {
    async fn load_all(&self) -> Result<Vec<ServingProfile>, ProfileLoadError> {
        Ok(self.rows.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use terminus_rs::intake::serving::{
        ExclusionReason, RecheckTrigger, ServingBackend,
    };

    fn row(
        model: &str,
        backend: ServingBackend,
        best: Runtime,
        env_json: &str,
        excl: ExclusionReason,
    ) -> ServingProfile {
        ServingProfile {
            model_id: ModelId::from(model),
            backend_tag: backend,
            best_runtime: best,
            env_json: env_json.into(),
            tok_s: Some(30.0),
            vram_or_ram_peak_gb: Some(8.0),
            cold_load_s: Some(10.0),
            keep_warm: false,
            fallback_runtime: Some(Runtime::Ollama),
            exclusion_reason: excl,
            recheck_trigger: RecheckTrigger::None,
            provenance: None,
        }
    }

    #[test]
    fn envspec_parses_all_known_keys() {
        let spec = EnvSpec::parse(
            r#"{"gfx_override":"11.0.0","mmap":false,"flash_attn":true,"cpu_library":"cpu_avx2"}"#,
        );
        assert_eq!(spec.gfx_override.as_deref(), Some("11.0.0"));
        assert_eq!(spec.mmap, Some(false));
        assert_eq!(spec.flash_attn, Some(true));
        assert_eq!(spec.cpu_library.as_deref(), Some("cpu_avx2"));
    }

    #[test]
    fn envspec_parses_runner_native_schema() {
        // The SRV-02 runner / serving_seed.json shape: gfx_override as a BOOL,
        // mmap as `mmap_flag` (0 ⇒ --no-mmap), cpu_lib (not cpu_library). This is
        // the contract that previously silently dropped --no-mmap.
        let spec = EnvSpec::parse(
            r#"{"gfx_override":true,"mmap_flag":0,"flash_attn":false,"cpu_lib":"cpu_avx2","standard_prompt_id":"sp1"}"#,
        );
        // bool true ⇒ apply host default (no explicit string), NOT a literal value.
        assert_eq!(spec.gfx_override, None);
        assert!(spec.gfx_apply_host_default);
        // mmap_flag 0 ⇒ mmap off ⇒ launcher emits --no-mmap (the load-bearing fix).
        assert_eq!(spec.mmap, Some(false));
        assert_eq!(spec.flash_attn, Some(false));
        assert_eq!(spec.cpu_library.as_deref(), Some("cpu_avx2"));
    }

    #[test]
    fn envspec_parses_explicit_n_ctx() {
        // SRV-12: an explicit context pins -c so auto-fit never trips the UMA slash.
        assert_eq!(EnvSpec::parse(r#"{"n_ctx":131072}"#).n_ctx, Some(131072));
        // `ctx` alias accepted too.
        assert_eq!(EnvSpec::parse(r#"{"ctx":32768}"#).n_ctx, Some(32768));
        // absent ⇒ None (swap computes a safe default).
        assert_eq!(EnvSpec::parse("{}").n_ctx, None);
    }

    #[test]
    fn envspec_runner_mmap_flag_one_keeps_mmap_on() {
        let spec = EnvSpec::parse(r#"{"mmap_flag":1}"#);
        assert_eq!(spec.mmap, Some(true));
        // gfx_override:false ⇒ neither explicit nor apply-default.
        let spec2 = EnvSpec::parse(r#"{"gfx_override":false}"#);
        assert_eq!(spec2.gfx_override, None);
        assert!(!spec2.gfx_apply_host_default);
    }

    #[test]
    fn envspec_empty_blank_or_bad_is_all_default() {
        assert_eq!(EnvSpec::parse(""), EnvSpec::default());
        assert_eq!(EnvSpec::parse("   "), EnvSpec::default());
        assert_eq!(EnvSpec::parse("{not json"), EnvSpec::default());
        // A JSON non-object is also all-default (not a crash).
        assert_eq!(EnvSpec::parse("[1,2,3]"), EnvSpec::default());
    }

    #[test]
    fn envspec_empty_gfx_is_distinct_from_absent() {
        // The CPU tier records a deliberate EMPTY gfx override — Some("") — which
        // must NOT collapse to None (absent).
        let spec = EnvSpec::parse(r#"{"gfx_override":""}"#);
        assert_eq!(spec.gfx_override.as_deref(), Some(""));
        let absent = EnvSpec::parse("{}");
        assert_eq!(absent.gfx_override, None);
    }

    #[test]
    fn routing_map_keys_by_model_id() {
        let map = RoutingMap::load_from(vec![row(
            "qwen3:8b",
            ServingBackend::LlamaGpu,
            Runtime::LlamaCpp,
            "{}",
            ExclusionReason::None,
        )]);
        let entry = map.get(&ModelId::from("qwen3:8b")).expect("routed");
        assert_eq!(entry.best_runtime(), Runtime::LlamaCpp);
        // Unprofiled model → None (the launcher turns this into a clear error).
        assert!(map.get(&ModelId::from("not-a-model")).is_none());
    }

    #[test]
    fn routing_map_prefers_usable_row_then_better_tier() {
        // Same model: llama-gpu row is EXCLUDED (arch reject), ollama-gpu is usable.
        // The usable ollama row must win even though llama is the better tier.
        let rows = vec![
            row(
                "gemma4:9b",
                ServingBackend::LlamaGpu,
                Runtime::LlamaCpp,
                "{}",
                ExclusionReason::BuildConditional,
            ),
            row(
                "gemma4:9b",
                ServingBackend::OllamaGpu,
                Runtime::Ollama,
                "{}",
                ExclusionReason::None,
            ),
        ];
        // BuildConditional needs the version-bump trigger to validate, but the
        // routing map does not validate (it trusts the store); set trigger so the
        // fixture is internally coherent anyway.
        let mut rows = rows;
        rows[0].recheck_trigger = RecheckTrigger::LlamaCppVersionBump;

        let map = RoutingMap::load_from(rows);
        let entry = map.get(&ModelId::from("gemma4:9b")).unwrap();
        assert_eq!(entry.best_runtime(), Runtime::Ollama);
        assert_eq!(entry.profile.backend_tag, ServingBackend::OllamaGpu);
    }

    #[test]
    fn routing_map_prefers_better_tier_when_both_usable() {
        let rows = vec![
            row(
                "qwen3:8b",
                ServingBackend::OllamaGpu,
                Runtime::Ollama,
                "{}",
                ExclusionReason::None,
            ),
            row(
                "qwen3:8b",
                ServingBackend::LlamaGpu,
                Runtime::LlamaCpp,
                "{}",
                ExclusionReason::None,
            ),
        ];
        let map = RoutingMap::load_from(rows);
        let entry = map.get(&ModelId::from("qwen3:8b")).unwrap();
        // llama-gpu is the more preferred tier.
        assert_eq!(entry.profile.backend_tag, ServingBackend::LlamaGpu);
    }

    #[tokio::test]
    async fn load_from_static_source() {
        let src = StaticProfileSource::new(vec![row(
            "qwen3:8b",
            ServingBackend::LlamaGpu,
            Runtime::LlamaCpp,
            "{}",
            ExclusionReason::None,
        )]);
        let map = RoutingMap::load(&src).await.unwrap();
        assert_eq!(map.len(), 1);
        assert!(!map.is_empty());
    }

    #[test]
    fn envspec_rope_scaling_absent_is_none() {
        assert_eq!(EnvSpec::parse("{}").rope_scaling, None);
    }

    #[test]
    fn envspec_parses_valid_yarn_rope_scaling() {
        let spec = EnvSpec::parse(
            r#"{"rope_scaling":{"method":"yarn","rope_scale":4.0,"yarn_orig_ctx":32768,
                "target_ctx":131072,"ext_factor":1.0,"attn_factor":1.0,"beta_slow":1.0,
                "beta_fast":32.0,"validated":true}}"#,
        );
        let rope = spec.rope_scaling.expect("yarn block parsed");
        assert_eq!(rope.method, RopeScalingMethod::Yarn);
        assert_eq!(rope.rope_scale, 4.0);
        assert_eq!(rope.yarn_orig_ctx, 32768);
        assert_eq!(rope.target_ctx, 131072);
        assert_eq!(rope.ext_factor, 1.0);
        assert_eq!(rope.attn_factor, 1.0);
        assert_eq!(rope.beta_slow, 1.0);
        assert_eq!(rope.beta_fast, 32.0);
        assert!(rope.validated);
    }

    #[test]
    fn envspec_parses_valid_linear_rope_scaling() {
        let spec = EnvSpec::parse(
            r#"{"rope_scaling":{"method":"linear","rope_scale":2.0,"target_ctx":16384,"validated":true}}"#,
        );
        let rope = spec.rope_scaling.expect("linear block parsed");
        assert_eq!(rope.method, RopeScalingMethod::Linear);
        assert_eq!(rope.rope_scale, 2.0);
        assert_eq!(rope.target_ctx, 16384);
        // yarn-only fields default to zero — the launcher never emits them for linear.
        assert_eq!(rope.yarn_orig_ctx, 0);
    }

    #[test]
    fn envspec_rope_scaling_missing_required_field_is_invalid() {
        // yarn missing the fine-tune quartet.
        let spec = EnvSpec::parse(
            r#"{"rope_scaling":{"method":"yarn","rope_scale":4.0,"target_ctx":131072,"validated":true}}"#,
        );
        assert_eq!(spec.rope_scaling, None);
        // linear missing target_ctx.
        let spec2 =
            EnvSpec::parse(r#"{"rope_scaling":{"method":"linear","rope_scale":2.0,"validated":true}}"#);
        assert_eq!(spec2.rope_scaling, None);
    }

    #[test]
    fn envspec_rope_scaling_unknown_method_is_invalid() {
        let spec = EnvSpec::parse(r#"{"rope_scaling":{"method":"bogus"}}"#);
        assert_eq!(spec.rope_scaling, None);
    }

    #[test]
    fn envspec_rope_scaling_method_none_is_default_shape() {
        let spec = EnvSpec::parse(r#"{"rope_scaling":{"method":"none"}}"#);
        let rope = spec.rope_scaling.expect("present, but a no-op method");
        assert_eq!(rope.method, RopeScalingMethod::None);
    }

    #[test]
    fn rope_scaling_is_plausible_accepts_sane_yarn_values() {
        let rope = RopeScaling {
            method: RopeScalingMethod::Yarn,
            rope_scale: 4.0,
            yarn_orig_ctx: 32768,
            target_ctx: 131072,
            ext_factor: 1.0,
            attn_factor: 1.0,
            beta_slow: 1.0,
            beta_fast: 32.0,
            validated: true,
        };
        assert!(rope.is_plausible());
    }

    #[test]
    fn rope_scaling_is_plausible_rejects_nonpositive_rope_scale() {
        let mut rope = RopeScaling {
            method: RopeScalingMethod::Yarn,
            rope_scale: 0.0,
            yarn_orig_ctx: 32768,
            target_ctx: 131072,
            ext_factor: 1.0,
            attn_factor: 1.0,
            beta_slow: 1.0,
            beta_fast: 32.0,
            validated: true,
        };
        assert!(!rope.is_plausible());
        rope.rope_scale = -2.0;
        assert!(!rope.is_plausible());
    }

    #[test]
    fn rope_scaling_is_plausible_rejects_out_of_bounds_yarn_factors() {
        let base = RopeScaling {
            method: RopeScalingMethod::Yarn,
            rope_scale: 4.0,
            yarn_orig_ctx: 32768,
            target_ctx: 131072,
            ext_factor: 1.0,
            attn_factor: 1.0,
            beta_slow: 1.0,
            beta_fast: 32.0,
            validated: true,
        };
        let mut bad_ext = base;
        bad_ext.ext_factor = 5.0; // way outside [0,1]
        assert!(!bad_ext.is_plausible());

        let mut bad_attn = base;
        bad_attn.attn_factor = -0.5;
        assert!(!bad_attn.is_plausible());

        let mut bad_beta = base;
        bad_beta.beta_slow = -1.0;
        assert!(!bad_beta.is_plausible());
    }

    // ── YARN-05: thinking-preservation parsing ─────────────────────────────

    #[test]
    fn envspec_thinking_absent_is_none() {
        assert_eq!(EnvSpec::parse("{}").thinking, None);
    }

    #[test]
    fn envspec_parses_full_thinking_block() {
        let spec = EnvSpec::parse(
            r#"{"thinking":{"supports_thinking":true,"preserve_thinking":true,
                "requires_prefix_caching":true,"validated":true}}"#,
        );
        let t = spec.thinking.expect("thinking block parsed");
        assert!(t.supports_thinking);
        assert!(t.preserve_thinking);
        assert!(t.requires_prefix_caching);
        assert!(t.validated);
    }

    #[test]
    fn envspec_thinking_missing_fields_default_false() {
        // An object present but with only one field set — every other field is a
        // safe `false` default, not a rejection (unlike rope_scaling's required
        // fields, every thinking field is an optional bool).
        let spec = EnvSpec::parse(r#"{"thinking":{"supports_thinking":true}}"#);
        let t = spec.thinking.expect("thinking block parsed");
        assert!(t.supports_thinking);
        assert!(!t.preserve_thinking);
        assert!(!t.requires_prefix_caching);
        assert!(!t.validated);
    }

    #[test]
    fn envspec_thinking_malformed_is_none() {
        // `thinking` present but not an object.
        assert_eq!(EnvSpec::parse(r#"{"thinking":"yes"}"#).thinking, None);
    }

    #[test]
    fn profile_load_error_messages_carry_no_infra() {
        for e in [ProfileLoadError::NotConfigured, ProfileLoadError::StoreUnavailable] {
            let msg = e.to_string();
            assert!(!msg.contains("postgres"));
            assert!(!msg.contains("://"));
            assert!(!msg.contains('@'));
        }
    }
}
