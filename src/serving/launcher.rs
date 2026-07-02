//! Per-profile runtime launcher (S85 SRV-04, launch side).
//!
//! Given a model's [`RouteEntry`] (from [`profile`](super::profile)), build the
//! correct runtime launch command + env, start it, and health-check the endpoint
//! — falling back to `fallback_runtime` on failure, and refusing cleanly when a
//! model is unprofiled or cannot fit.
//!
//! ## The three runtimes (command + env construction)
//! Every binary/endpoint comes from the SRV-01 config helpers — NEVER a literal
//! (S77 / pii_gate). The launcher only assembles ARGUMENTS from the row's
//! [`EnvSpec`]:
//!   - **llama.cpp-rocm** ([`Runtime::LlamaCpp`]): `llama_server_bin()` with
//!     `--model <gguf>`; `HSA_OVERRIDE_GFX_VERSION` env from `gfx_override`;
//!     **`--no-mmap` when `mmap == Some(false)`** (the v2 NFS-page-fault lesson);
//!     `--flash-attn` when `flash_attn == Some(true)`; YARN-01: `--rope-scaling`
//!     / `--rope-scale` / `--yarn-*` / `--ctx-size` when the row's
//!     `rope_scaling` block is present, `validated`, and an extension is
//!     actually needed (see [`rope_scaling_args`]); YARN-05: `--preserve-thinking`
//!     / `--prompt-cache-all` when the row's `thinking` block requests
//!     preservation on a model that `supports_thinking` and is `validated` (see
//!     [`thinking_args`]). Health-checks `llama_server_url()`.
//!   - **ollama-rocm** ([`Runtime::Ollama`]): `ollama_bin()` `serve`; the GPU
//!     `gfx_override` env; health-checks `ollama_primary_url()`.
//!   - **genuine CPU** ([`Runtime::Cpu`]): `ollama_bin()` `serve` against the
//!     SECONDARY unit; the deliberate EMPTY gfx override; the cpu-runtime library
//!     from `cpu_library` (falling back to `ollama_cpu_library()`). Health-checks
//!     `ollama_secondary_url()`.
//!
//! ## keep_warm models never cold-launch inline
//! A `keep_warm` model is big and slow to cold-load; it must be served by a
//! resident slot, NOT cold-launched on the request hot path. SRV-04 routes those
//! requests through the [`ResidencyManager`] trait (implemented for real by
//! SRV-05). A keep_warm request that reaches the inline cold-launch path is a bug
//! — the launcher refuses it ([`LaunchError::KeepWarmMustUseResidency`]) so the
//! negative test can catch a regression even if SRV-05 is not wired yet.
//!
//! ## Genericized errors (S77)
//! Every [`LaunchError`] surfaced to a caller carries only a stable reason — no
//! host, path, DSN, or binary path. Underlying detail is logged internally and
//! recorded as a failure, never returned.

use async_trait::async_trait;

use terminus_rs::config;
use terminus_rs::intake::serving::{ModelId, Runtime};

use crate::config::Config;
use crate::supervisor::egress_policy::posture_for;
use crate::supervisor::launch_isolation::{decide_isolation, IsolationDecision};
use crate::supervisor::{build_runtime_env, RuntimeClass};

use super::profile::{
    EnvSpec, ProfileLoadError, RopeScaling, RopeScalingMethod, RouteEntry, ThinkingConfig,
};

/// A constructed runtime launch: the binary, its argv, the env pairs, and the
/// endpoint to health-check. Pure data — building it touches no process, so it is
/// unit-testable without launching anything.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchCommand {
    /// The runtime tier this command launches.
    pub runtime: Runtime,
    /// The binary/command to exec (from a config helper — never a literal).
    pub bin: String,
    /// Positional + flag arguments (the part SRV-04 actually constructs).
    pub args: Vec<String>,
    /// Environment variables to set for the launch (e.g. the gfx override).
    pub env: Vec<(String, String)>,
    /// The HTTP endpoint to health-check once started (from a config helper).
    pub health_url: String,
}

/// Env var name for the ROCm gfx override (the only env key the launcher sets).
/// Named here once so the string is not repeated; it is an env-var NAME, not an
/// infra value.
const GFX_OVERRIDE_ENV: &str = "HSA_OVERRIDE_GFX_VERSION";
/// Env var name for an explicit cpu-runtime library override on the CPU tier.
const CPU_LIBRARY_ENV: &str = "OLLAMA_LLM_LIBRARY";

/// A genericized launch failure (S77): a stable reason, NO infra leakage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LaunchError {
    /// No serving_profile row for the requested model — Chord will NOT guess a
    /// runtime.
    UnprofiledModel(String),
    /// A required runtime endpoint/binary is not configured (the config helper
    /// returned `None`). No host is guessed.
    RuntimeNotConfigured,
    /// `best_runtime` failed (launch or health), and either there was no
    /// `fallback_runtime` or the fallback also failed. The terminal user-facing
    /// error after the fallback chain is exhausted.
    AllRuntimesFailed(String),
    /// The model is `keep_warm` and reached the inline cold-launch path. It MUST
    /// be admitted through the [`ResidencyManager`]. A bug if ever returned on a
    /// path that was supposed to delegate.
    KeepWarmMustUseResidency(String),
    /// A CPU-only profile whose weights exceed host RAM — refuse with a clear
    /// reason, do not attempt a doomed launch.
    CpuModelExceedsHostRam(String),
    /// The serving profile store could not be read.
    ProfileStoreUnavailable,
    /// S88 ISO-02: network-namespace isolation could not be established and there
    /// was no explicit operator override → the launch is REFUSED (fail closed). The
    /// runtime is NOT spawned with full host egress.
    IsolationRefused(String),
}

impl std::fmt::Display for LaunchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LaunchError::UnprofiledModel(m) => {
                write!(f, "model '{m}' has no serving profile; refusing to guess a runtime")
            }
            LaunchError::RuntimeNotConfigured => {
                f.write_str("the requested serving runtime is not configured")
            }
            LaunchError::AllRuntimesFailed(m) => {
                write!(f, "could not serve model '{m}': all configured runtimes failed")
            }
            LaunchError::KeepWarmMustUseResidency(m) => write!(
                f,
                "model '{m}' is keep-warm and must be admitted via the residency manager, \
                 not cold-launched"
            ),
            LaunchError::CpuModelExceedsHostRam(m) => {
                write!(f, "model '{m}' does not fit available memory on the CPU tier")
            }
            LaunchError::ProfileStoreUnavailable => {
                f.write_str("serving profile store is temporarily unavailable")
            }
            LaunchError::IsolationRefused(m) => write!(
                f,
                "refusing to serve model '{m}': network-namespace isolation unavailable \
                 and no operator override (failing closed, not launching with host egress)"
            ),
        }
    }
}

impl std::error::Error for LaunchError {}

impl From<ProfileLoadError> for LaunchError {
    fn from(_: ProfileLoadError) -> Self {
        // Collapse any store error to a single genericized variant (the detailed
        // cause is already logged at the source).
        LaunchError::ProfileStoreUnavailable
    }
}

/// A handle to a launched, health-checked runtime serving one model. Opaque to
/// callers; carries which runtime ultimately served (so a fallback is observable)
/// and the health endpoint that passed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServeHandle {
    /// The model now being served.
    pub model_id: String,
    /// The runtime that ultimately served it (may be the fallback).
    pub runtime: Runtime,
    /// The endpoint that passed the health check.
    pub endpoint: String,
    /// True when this serve came from a warm residency slot rather than a cold
    /// launch (keep-warm models).
    pub from_warm_slot: bool,
    /// S88 ISO-02: the network namespace this runtime was spawned into, if any
    /// (`Some(name)` when isolated; `None` when isolation was disabled by config or
    /// the operator override was used). The SRV-12 clean swap tears this down when
    /// the runtime is swapped out — see [`netns_to_teardown`].
    pub netns: Option<String>,
}

/// An acquired warm-residency slot for a keep-warm model. Returned by
/// [`ResidencyManager::acquire_warm_slot`]. SRV-05 fills in the real
/// admission/eviction semantics; SRV-04 only needs the endpoint to serve from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Slot {
    /// The model this slot serves.
    pub model_id: String,
    /// The runtime backing the resident model.
    pub runtime: Runtime,
    /// The (already health-checked, resident) endpoint to route requests to.
    pub endpoint: String,
    /// S88 ISO-02: the network namespace the resident runtime was spawned into, if
    /// any. Torn down when the slot is evicted/swapped out.
    pub netns: Option<String>,
}

// ─────────────────────────────────────────────────────────────────────────────
// The SRV-05 interface (defined HERE by SRV-04, implemented by SRV-05)
// ─────────────────────────────────────────────────────────────────────────────

/// VRAM-admission + tier-aware residency for keep-warm models.
///
/// **This is the contract SRV-04 needs and SRV-05 implements.** SRV-04 (this
/// crate, launcher) must NEVER cold-launch a `keep_warm` model on the request hot
/// path — those big MoEs take ~8–10 min to cold-load and must be served from a
/// model that is already resident. Instead, SRV-04 asks the residency manager for
/// a warm slot via [`acquire_warm_slot`](ResidencyManager::acquire_warm_slot).
///
/// SRV-05 implements this trait with the real behavior described in the S85 spec:
///   - **Admission:** if the model already fits free VRAM (or is already resident)
///     → return a [`Slot`] immediately.
///   - **Eviction (tier-aware):** if it does not fit, evict transient (build)
///     residents first; the pinned chat-role model is NEVER evicted; keep-warm vs
///     keep-warm contention QUEUEs first (bounded wait) and only evicts LRU after
///     the wait threshold.
///   - **Fail-safe:** if free VRAM is unreadable or the bounded wait expires with
///     nothing evictable, return [`ResidencyError::CannotAdmit`] rather than risk
///     an OOM launch — never force-evict the pinned chat model.
///
/// SRV-04 ships a [`PassThroughResidency`] stub so this crate compiles and tests
/// before SRV-05 lands. The stub does NO admission/eviction — it just launches
/// (via an injected launch closure) and wraps the result as a slot. SRV-05
/// REPLACES the stub; SRV-04 does not change.
#[async_trait]
pub trait ResidencyManager: Send + Sync {
    /// Acquire a warm slot for `model_id` (whose footprint is `vram_gb`, if known
    /// from the profile). Returns a ready-to-serve [`Slot`] on admission, or a
    /// genericized [`ResidencyError`] when the model cannot be admitted.
    ///
    /// CONTRACT: this is the ONLY entry point by which a keep-warm model is
    /// served. It must health-check the resident endpoint before returning a slot
    /// (so a returned slot is always serveable).
    async fn acquire_warm_slot(
        &self,
        model_id: &ModelId,
        vram_gb: Option<f64>,
    ) -> Result<Slot, ResidencyError>;
}

/// A genericized residency-admission failure (S77): a stable reason, no infra.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResidencyError {
    /// The model could not be admitted within the bounded wait (e.g. all
    /// residents pinned/keep-warm and it still doesn't fit), or free VRAM was
    /// unreadable and the manager failed safe rather than risk an OOM launch.
    CannotAdmit(String),
    /// The resident endpoint failed its health check after admission.
    SlotUnhealthy(String),
}

impl std::fmt::Display for ResidencyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResidencyError::CannotAdmit(m) => {
                write!(f, "model '{m}' cannot currently be admitted to a warm slot")
            }
            ResidencyError::SlotUnhealthy(m) => {
                write!(f, "warm slot for model '{m}' did not become healthy")
            }
        }
    }
}

impl std::error::Error for ResidencyError {}

/// Health-checker abstraction so tests don't hit a real endpoint. Production wires
/// an HTTP probe; tests inject a deterministic result.
#[async_trait]
pub trait HealthChecker: Send + Sync {
    /// Return `true` iff the runtime at `endpoint` is serving (health passed).
    async fn check(&self, endpoint: &str) -> bool;
}

/// Launcher abstraction so tests don't spawn real processes. Production spawns the
/// runtime from the [`LaunchCommand`]; tests inject a scripted launcher that
/// records what was asked and returns success/failure per runtime.
#[async_trait]
pub trait RuntimeSpawner: Send + Sync {
    /// Start the runtime described by `cmd`. Returns `Ok(())` once the process is
    /// spawned (health is checked separately by the launcher). An `Err` carries a
    /// genericized reason (the detailed cause is logged, not surfaced).
    async fn spawn(&self, cmd: &LaunchCommand) -> Result<(), String>;
}

/// Records a launch failure for observability (the "record the failure" half of
/// the fallback path). Implementations sanitize per S6/S77 before persisting.
pub trait FailureRecorder: Send + Sync {
    /// Record that serving `model_id` via `runtime` failed for `reason` (a short,
    /// already-genericized phrase).
    fn record_failure(&self, model_id: &str, runtime: Runtime, reason: &str);
}

/// A no-op failure recorder (used when no sink is wired).
pub struct NoopFailureRecorder;
impl FailureRecorder for NoopFailureRecorder {
    fn record_failure(&self, _model_id: &str, _runtime: Runtime, _reason: &str) {}
}

// ─────────────────────────────────────────────────────────────────────────────
// Command + env construction (the testable core — no process, no network)
// ─────────────────────────────────────────────────────────────────────────────

/// Resolve the gfx override value to apply for a GPU-tier launch, honouring both
/// `env_json` shapes: an explicit string (returned as-is, incl. the meaningful
/// empty value) takes precedence; otherwise the runner's `gfx_override: true`
/// flag pulls the host's configured value from [`config::gfx_override_version`]
/// (no literal — `None` ⇒ omit the override rather than guess).
fn resolve_gfx_override(env: &EnvSpec) -> Option<String> {
    if let Some(explicit) = &env.gfx_override {
        return Some(explicit.clone());
    }
    if env.gfx_apply_host_default {
        return config::gfx_override_version();
    }
    None
}

/// YARN-01: build the `--rope-scaling` / `--yarn-*` / `--ctx-size` arguments for
/// the llama.cpp tier, or nothing at all.
///
/// Emission rules (all values sourced from `rope` — never invented):
///   - `method == None` ⇒ no flags (unchanged native-context behavior).
///   - `method != None` but `!validated` ⇒ no flags; the caller logs the
///     "configured but not validated" warning and serves at native context.
///   - `validated` but [`RopeScaling::is_plausible`] fails (e.g. a bad manual
///     edit) ⇒ no flags; refuses to emit garbage to `llama-server`, same
///     native-context fallback as the unvalidated path.
///   - `method == Yarn` and `target_ctx <= yarn_orig_ctx` ⇒ no extension
///     needed; no-op even when validated (nothing to gain from scaling
///     down/flat). Yarn-specific: `yarn_orig_ctx` defaults to 0 for linear,
///     where there is no original-context concept.
///   - otherwise: `--rope-scaling <method>`, `--rope-scale <rope_scale>`, and
///     `--ctx-size <target_ctx>`; `method == Yarn` additionally gets the
///     yarn-specific fine-tune flags (`--yarn-orig-ctx`, `--yarn-ext-factor`,
///     `--yarn-attn-factor`, `--yarn-beta-slow`, `--yarn-beta-fast`) — linear
///     scaling never gets these.
fn rope_scaling_args(rope: &RopeScaling) -> Vec<String> {
    if rope.method == RopeScalingMethod::None {
        return Vec::new();
    }
    if !rope.validated {
        tracing::warn!(
            target: "chord.serving.launcher",
            method = rope.method.as_str(),
            "yarn configured but not validated on gfx1151 — serving at native context"
        );
        return Vec::new();
    }
    // Plausibility gate: `validated=true` is not a license to emit whatever
    // numbers are in the row verbatim — a bad manual edit or unit mix-up must
    // never reach `llama-server` as-is. Same fallback as the unvalidated path:
    // native context, no flags.
    if !rope.is_plausible() {
        tracing::warn!(
            target: "chord.serving.launcher",
            method = rope.method.as_str(),
            rope_scale = rope.rope_scale,
            ext_factor = rope.ext_factor,
            attn_factor = rope.attn_factor,
            "yarn validated but parameters implausible — refusing to emit, serving at native context"
        );
        return Vec::new();
    }
    // "No extension needed" is a yarn-specific check: `yarn_orig_ctx` defaults
    // to 0 for linear (no original-context concept there), so scoping this to
    // `Yarn` avoids a spurious no-op read for the linear method.
    if rope.method == RopeScalingMethod::Yarn && rope.target_ctx <= rope.yarn_orig_ctx {
        tracing::warn!(
            target: "chord.serving.launcher",
            target_ctx = rope.target_ctx,
            yarn_orig_ctx = rope.yarn_orig_ctx,
            "rope-scaling target_ctx does not exceed yarn_orig_ctx — no context extension needed, skipping"
        );
        return Vec::new();
    }

    let mut args = vec![
        "--rope-scaling".to_string(),
        rope.method.as_str().to_string(),
        "--rope-scale".to_string(),
        rope.rope_scale.to_string(),
    ];
    if rope.method == RopeScalingMethod::Yarn {
        args.push("--yarn-orig-ctx".to_string());
        args.push(rope.yarn_orig_ctx.to_string());
        args.push("--yarn-ext-factor".to_string());
        args.push(rope.ext_factor.to_string());
        args.push("--yarn-attn-factor".to_string());
        args.push(rope.attn_factor.to_string());
        args.push("--yarn-beta-slow".to_string());
        args.push(rope.beta_slow.to_string());
        args.push("--yarn-beta-fast".to_string());
        args.push(rope.beta_fast.to_string());
    }
    args.push("--ctx-size".to_string());
    args.push(rope.target_ctx.to_string());
    args
}

/// YARN-01: log that a non-`none` `rope_scaling` block was present on a tier that
/// cannot apply it (ollama / CPU). Never crashes, never applies the config — just
/// a clear note that context extension is unavailable on that tier.
fn warn_rope_scaling_unsupported(rope: &RopeScaling) {
    if rope.method != RopeScalingMethod::None {
        tracing::warn!(
            target: "chord.serving.launcher",
            method = rope.method.as_str(),
            "rope-scaling not supported on this tier — context extension unavailable"
        );
    }
}

/// YARN-05: build the `--preserve-thinking` / `--prompt-cache-all` arguments
/// for the llama.cpp tier, or nothing at all. Mirrors [`rope_scaling_args`]'s
/// gate discipline exactly.
///
/// Emission rules (all values sourced from `thinking` — never invented):
///   - `preserve_thinking == false` ⇒ no flags (nothing requested — the
///     unchanged, no-op baseline, same as a model with no `thinking` block).
///   - `preserve_thinking == true` but `supports_thinking == false` ⇒ refuse:
///     no flags, warn. A model that cannot produce a thinking trace has
///     nothing to preserve — emitting the flag would be meaningless.
///   - `preserve_thinking && supports_thinking` but `!validated` ⇒ no flags;
///     the "configured but not validated" warning, served without it — same
///     unvalidated-yarn fallback as YARN-01.
///   - otherwise (validated): emit `--preserve-thinking`; additionally emit
///     `--prompt-cache-all` when `requires_prefix_caching` is set, with a log
///     note that the prompt cache grows with conversation length (a
///     memory-cost surface, not something this item accounts for in bytes).
fn thinking_args(thinking: &ThinkingConfig) -> Vec<String> {
    if !thinking.preserve_thinking {
        return Vec::new();
    }
    if !thinking.supports_thinking {
        tracing::warn!(
            target: "chord.serving.launcher",
            "preserve_thinking requested but model does not support thinking — refusing, no flags emitted"
        );
        return Vec::new();
    }
    if !thinking.validated {
        tracing::warn!(
            target: "chord.serving.launcher",
            "thinking preservation configured but not validated on gfx1151 — serving without it"
        );
        return Vec::new();
    }
    let mut args = vec!["--preserve-thinking".to_string()];
    if thinking.requires_prefix_caching {
        args.push("--prompt-cache-all".to_string());
        tracing::warn!(
            target: "chord.serving.launcher",
            "prefix caching enabled for thinking preservation — prompt cache grows with \
             conversation length, monitor VRAM/RAM headroom"
        );
    }
    args
}

/// YARN-05: log that a `preserve_thinking` request was present on a tier that
/// cannot apply it (ollama / CPU — no prefix-caching support). Never crashes,
/// never applies the config — just a clear note that thinking preservation is
/// unavailable on that tier.
fn warn_thinking_unsupported(thinking: &ThinkingConfig) {
    if thinking.preserve_thinking {
        tracing::warn!(
            target: "chord.serving.launcher",
            "thinking preservation requested but not supported on this tier (no prefix \
             caching) — serving without it"
        );
    }
}

/// Build the [`LaunchCommand`] for serving `entry` under a specific `runtime`.
///
/// Reads every binary/endpoint from the SRV-01 config helpers (no literals) and
/// assembles only the arguments/env from the row's [`EnvSpec`]. Returns
/// [`LaunchError::RuntimeNotConfigured`] when the runtime's endpoint helper yields
/// `None` (no host is guessed).
///
/// `gguf_path` is the resolved model weights path the caller provides (already
/// acquired); it is a runtime argument, not an infra literal.
pub fn build_launch_command(
    entry: &RouteEntry,
    runtime: Runtime,
    gguf_path: &str,
) -> Result<LaunchCommand, LaunchError> {
    let env = &entry.env;
    match runtime {
        Runtime::LlamaCpp => {
            let bin = config::llama_server_bin();
            let health_url = config::llama_server_url().ok_or(LaunchError::RuntimeNotConfigured)?;
            let mut args = vec!["--model".to_string(), gguf_path.to_string()];
            // mmap == Some(false) → --no-mmap (the v2 NFS page-fault lesson).
            if env.mmap == Some(false) {
                args.push("--no-mmap".to_string());
            }
            // flash-attn opt-in.
            if env.flash_attn == Some(true) {
                args.push("--flash-attn".to_string());
            }
            // Explicit context window pins `-c` so auto-fit never sizes the context
            // against the UMA-misread free memory (the SRV-12 context-slash sidestep).
            if let Some(n) = env.n_ctx {
                args.push("-c".to_string());
                args.push(n.to_string());
            }
            // YARN-01: emit --rope-scaling/--yarn-*/--ctx-size when configured,
            // validated, and an extension is actually needed (llama.cpp tier only).
            if let Some(rope) = &env.rope_scaling {
                args.extend(rope_scaling_args(rope));
            }
            // YARN-05: emit --preserve-thinking/--prompt-cache-all when
            // configured, supported, and validated (llama.cpp tier only).
            if let Some(thinking) = &env.thinking {
                args.extend(thinking_args(thinking));
            }
            let mut envs = Vec::new();
            if let Some(gfx) = resolve_gfx_override(env) {
                // Empty string is a meaningful (CPU) value, but on llama.cpp tier
                // an empty override is just omitted; a non-empty one is set.
                if !gfx.is_empty() {
                    envs.push((GFX_OVERRIDE_ENV.to_string(), gfx));
                }
            }
            Ok(LaunchCommand {
                runtime,
                bin,
                args,
                env: envs,
                health_url,
            })
        }
        Runtime::Ollama => {
            let bin = config::ollama_bin();
            let health_url =
                config::ollama_primary_url().ok_or(LaunchError::RuntimeNotConfigured)?;
            let args = vec!["serve".to_string()];
            // YARN-01: this tier cannot apply rope scaling — ignore, log, never crash.
            if let Some(rope) = &env.rope_scaling {
                warn_rope_scaling_unsupported(rope);
            }
            // YARN-05: this tier has no prefix-caching support — ignore, log,
            // never crash.
            if let Some(thinking) = &env.thinking {
                warn_thinking_unsupported(thinking);
            }
            let mut envs = Vec::new();
            if let Some(gfx) = resolve_gfx_override(env) {
                if !gfx.is_empty() {
                    envs.push((GFX_OVERRIDE_ENV.to_string(), gfx));
                }
            }
            Ok(LaunchCommand {
                runtime,
                bin,
                args,
                env: envs,
                health_url,
            })
        }
        Runtime::Cpu => {
            let bin = config::ollama_bin();
            let health_url =
                config::ollama_secondary_url().ok_or(LaunchError::RuntimeNotConfigured)?;
            let args = vec!["serve".to_string()];
            // YARN-01: this tier cannot apply rope scaling — ignore, log, never crash.
            if let Some(rope) = &env.rope_scaling {
                warn_rope_scaling_unsupported(rope);
            }
            // YARN-05: this tier has no prefix-caching support — ignore, log,
            // never crash.
            if let Some(thinking) = &env.thinking {
                warn_thinking_unsupported(thinking);
            }
            let mut envs = Vec::new();
            // The CPU tier sets a DELIBERATE empty gfx override (the empty-override
            // CPU path). Some("") → set it empty; absent → fall through to empty
            // too (CPU is the genuine-CPU tier and must not inherit a GPU override).
            let gfx_value = env.gfx_override.clone().unwrap_or_default();
            envs.push((GFX_OVERRIDE_ENV.to_string(), gfx_value));
            // cpu library: row override first, then the config helper.
            let cpu_lib = env
                .cpu_library
                .clone()
                .or_else(config::ollama_cpu_library);
            if let Some(lib) = cpu_lib {
                envs.push((CPU_LIBRARY_ENV.to_string(), lib));
            }
            Ok(LaunchCommand {
                runtime,
                bin,
                args,
                env: envs,
                health_url,
            })
        }
    }
}

/// Compose a [`LaunchCommand`] that carries the **scrubbed** S88 ISO-01 launch
/// environment for a runtime of `class`, with the command's own runtime-specific
/// env (the gfx override etc.) layered ON TOP.
///
/// This is the ISO-01 integration seam: the scrubbed base (minimal env +
/// telemetry-off/offline opt-outs + proxy-strip) is built by
/// [`build_runtime_env`], then `cmd.env` (the gfx/cpu-lib pairs the launcher
/// already computed) is appended so a per-launch override always wins over the
/// scrubbed base. Behaviour is additive: the only NEW vars on an existing launch
/// are the telemetry-off/offline ones; nothing the launcher previously set is
/// dropped.
///
/// `class` is [`RuntimeClass::Serve`] for the normal cold-launch / warm-slot serve
/// path (a serving runtime needs no egress). A model-PULL path (not the launcher's
/// concern today) would pass [`RuntimeClass::Pull`].
pub fn scrub_launch_env(mut cmd: LaunchCommand, class: RuntimeClass, cfg: &Config) -> LaunchCommand {
    let mut scrubbed = build_runtime_env(class, cfg);
    // The launcher's own runtime env (gfx override / cpu lib) layers on top so an
    // explicit per-launch value always wins over the scrubbed base.
    scrubbed.append(&mut cmd.env);
    cmd.env = scrubbed;
    cmd
}

/// The SRV-04 launcher: holds the injected spawner / health-checker / failure
/// recorder and the residency manager. `serve_model` is the single entry point.
pub struct Launcher<'a> {
    spawner: &'a dyn RuntimeSpawner,
    health: &'a dyn HealthChecker,
    recorder: &'a dyn FailureRecorder,
    residency: &'a dyn ResidencyManager,
    /// S88 ISO-01 config for launch-env scrubbing. `None` ⇒ no scrub is applied
    /// (the legacy behaviour, used by the pre-ISO-01 unit tests); `Some(cfg)` ⇒
    /// every cold-launch's env is scrubbed via [`scrub_launch_env`] before spawn.
    cfg: Option<&'a Config>,
}

impl<'a> Launcher<'a> {
    /// Build a launcher from its collaborators (no ISO-01 env scrub — legacy ctor).
    pub fn new(
        spawner: &'a dyn RuntimeSpawner,
        health: &'a dyn HealthChecker,
        recorder: &'a dyn FailureRecorder,
        residency: &'a dyn ResidencyManager,
    ) -> Self {
        Launcher {
            spawner,
            health,
            recorder,
            residency,
            cfg: None,
        }
    }

    /// Build a launcher that applies the S88 ISO-01 launch-env scrub (telemetry-off
    /// / offline opt-outs + proxy-strip) to every cold-launch using `cfg`.
    pub fn with_scrub(
        spawner: &'a dyn RuntimeSpawner,
        health: &'a dyn HealthChecker,
        recorder: &'a dyn FailureRecorder,
        residency: &'a dyn ResidencyManager,
        cfg: &'a Config,
    ) -> Self {
        Launcher {
            spawner,
            health,
            recorder,
            residency,
            cfg: Some(cfg),
        }
    }

    /// Serve `model_id` using its route `entry` and resolved `gguf_path`.
    ///
    /// Routing:
    ///   1. keep_warm → delegate to the residency manager (NEVER cold-launch
    ///      inline). A keep_warm model that ever reaches the cold-launch branch is
    ///      a bug.
    ///   2. CPU-only profile whose weights exceed host RAM → refuse cleanly.
    ///   3. otherwise → cold-launch `best_runtime`; on launch/health failure try
    ///      `fallback_runtime`; both fail → genericized [`LaunchError`] + record.
    pub async fn serve_model(
        &self,
        model_id: &ModelId,
        entry: &RouteEntry,
        gguf_path: &str,
    ) -> Result<ServeHandle, LaunchError> {
        let model_str = model_id.as_str().to_string();

        // (1) keep_warm models are admitted through the residency manager.
        if entry.keep_warm() {
            let slot = self
                .residency
                .acquire_warm_slot(model_id, entry.vram_gb())
                .await
                .map_err(|e| {
                    self.recorder
                        .record_failure(&model_str, entry.best_runtime(), "residency-admission");
                    LaunchError::AllRuntimesFailed(format!("{} ({})", model_str, generic(&e)))
                })?;
            return Ok(ServeHandle {
                model_id: slot.model_id,
                runtime: slot.runtime,
                endpoint: slot.endpoint,
                from_warm_slot: true,
                netns: slot.netns,
            });
        }

        // (2) CPU-only profile whose weights exceed host RAM → refuse, do not try.
        if entry.best_runtime() == Runtime::Cpu && entry.fallback_runtime().is_none() {
            if let (Some(peak), Some(budget)) =
                (entry.vram_gb(), config_host_ram_budget_gb())
            {
                if peak > budget {
                    self.recorder
                        .record_failure(&model_str, Runtime::Cpu, "cpu-exceeds-host-ram");
                    return Err(LaunchError::CpuModelExceedsHostRam(model_str));
                }
            }
        }

        // (3) cold-launch best_runtime, then fallback_runtime.
        let mut tried: Vec<Runtime> = Vec::new();
        let best = entry.best_runtime();
        match self.try_runtime(&model_str, entry, best, gguf_path).await {
            Ok(handle) => return Ok(handle),
            Err(_) => tried.push(best),
        }

        if let Some(fallback) = entry.fallback_runtime() {
            // A keep_warm model would never reach here (handled in step 1), so a
            // fallback cold-launch is safe.
            match self.try_runtime(&model_str, entry, fallback, gguf_path).await {
                Ok(handle) => return Ok(handle),
                Err(_) => tried.push(fallback),
            }
        }

        let _ = tried;
        self.recorder
            .record_failure(&model_str, best, "all-runtimes-failed");
        Err(LaunchError::AllRuntimesFailed(model_str))
    }

    /// Cold-launch a single runtime and health-check it. Failure (build error,
    /// spawn error, or health fail) → genericized `Err`, with the cause recorded.
    async fn try_runtime(
        &self,
        model_str: &str,
        entry: &RouteEntry,
        runtime: Runtime,
        gguf_path: &str,
    ) -> Result<ServeHandle, LaunchError> {
        // Defense in depth: a keep_warm model must never be cold-launched here.
        if entry.keep_warm() {
            return Err(LaunchError::KeepWarmMustUseResidency(model_str.to_string()));
        }

        let cmd = match build_launch_command(entry, runtime, gguf_path) {
            Ok(c) => c,
            Err(e) => {
                self.recorder
                    .record_failure(model_str, runtime, "runtime-not-configured");
                return Err(e);
            }
        };

        // S88 ISO-01: scrub the launch env (telemetry-off / offline opt-outs +
        // proxy-strip) before spawning. Serving a model needs no egress → Serve
        // class. Additive: when no cfg is wired the legacy env is used unchanged.
        let cmd = match self.cfg {
            Some(cfg) => scrub_launch_env(cmd, RuntimeClass::Serve, cfg),
            None => cmd,
        };

        // S88 ISO-02: enforce the egress posture with a per-runtime network
        // namespace. A serving runtime is the `Serve` class → `Denied` posture →
        // a netns with NO route (the kernel guarantee). FAIL CLOSED: a `Refused`
        // decision aborts the launch rather than spawn with full host egress. Only
        // applied when a cfg is wired (the legacy ctor keeps the un-isolated path
        // for the existing unit tests / non-privileged dev).
        let (cmd, netns_name) = match self.cfg {
            Some(cfg) => match self.isolate_command(model_str, runtime, cmd, RuntimeClass::Serve, cfg)
            {
                Ok(pair) => pair,
                Err(e) => return Err(e),
            },
            None => (cmd, None),
        };

        if let Err(_detail) = self.spawner.spawn(&cmd).await {
            // _detail is intentionally dropped from the surfaced error (S77).
            // A namespace we created for a launch that then failed to spawn must
            // not leak — tear it down (idempotent) before returning.
            teardown_netns(netns_name.as_deref());
            self.recorder
                .record_failure(model_str, runtime, "launch-failed");
            return Err(LaunchError::AllRuntimesFailed(model_str.to_string()));
        }

        if !self.health.check(&cmd.health_url).await {
            teardown_netns(netns_name.as_deref());
            self.recorder
                .record_failure(model_str, runtime, "health-check-failed");
            return Err(LaunchError::AllRuntimesFailed(model_str.to_string()));
        }

        Ok(ServeHandle {
            model_id: model_str.to_string(),
            runtime,
            endpoint: cmd.health_url,
            from_warm_slot: false,
            netns: netns_name,
        })
    }

    /// S88 ISO-02 seam: given a built+scrubbed `cmd`, establish the network
    /// namespace for `class`'s egress posture and rewrite the command to spawn
    /// INSIDE it. Returns the (possibly rewritten) command and the namespace name
    /// to record for teardown.
    ///
    /// FAIL CLOSED: an [`IsolationDecision::Refused`] returns
    /// [`LaunchError::IsolationRefused`] — the runtime is never spawned with full
    /// host egress. `DisabledByConfig` / `UnisolatedOverride` return the command
    /// unchanged with no namespace (the explicit opt-out / override paths).
    fn isolate_command(
        &self,
        model_str: &str,
        runtime: Runtime,
        cmd: LaunchCommand,
        class: RuntimeClass,
        cfg: &Config,
    ) -> Result<(LaunchCommand, Option<String>), LaunchError> {
        let posture = posture_for(class, cfg);
        let decision = decide_isolation(model_str, &posture);
        match decision {
            IsolationDecision::Isolated(handle) => {
                // Resolve the `ip` binary and rewrite bin+args to `ip netns exec …`.
                let ip = match crate::config::ip_bin() {
                    Some(b) => b,
                    None => {
                        // Tooling vanished between prepare and wrap — fail closed,
                        // tearing the just-created namespace down.
                        let _ = handle.teardown();
                        self.recorder
                            .record_failure(model_str, runtime, "isolation-tool-unavailable");
                        return Err(LaunchError::IsolationRefused(model_str.to_string()));
                    }
                };
                let name = handle.name().to_string();
                let (bin, args) = handle.wrap_command(&ip, &cmd.bin, &cmd.args);
                Ok((
                    LaunchCommand {
                        bin,
                        args,
                        ..cmd
                    },
                    Some(name),
                ))
            }
            IsolationDecision::DisabledByConfig | IsolationDecision::UnisolatedOverride(_) => {
                // Explicit dev opt-out or loud operator override: spawn as-is, no ns.
                Ok((cmd, None))
            }
            IsolationDecision::Refused(_) => {
                self.recorder
                    .record_failure(model_str, runtime, "isolation-refused");
                Err(LaunchError::IsolationRefused(model_str.to_string()))
            }
        }
    }
}

/// Tear down a network namespace by name (idempotent; no-op when `None`). Used to
/// reap a namespace whose launch failed after the namespace was created, and as the
/// hook the SRV-12 clean swap calls when a runtime is swapped out.
pub fn teardown_netns(name: Option<&str>) {
    if let Some(n) = name {
        // Reconstruct a handle purely to call its idempotent teardown. A wrong
        // posture here is irrelevant — teardown only needs the name.
        let handle = crate::supervisor::netns::NetnsHandle::for_teardown(n);
        if let Err(e) = handle.teardown() {
            tracing::debug!(
                target: "chord.serving.launcher",
                reason = %e,
                "network-namespace teardown reported an error (already idempotent-safe)"
            );
        }
    }
}

/// Genericize a residency error into a one-word reason for the terminal launch
/// error (no infra detail crosses over).
fn generic(e: &ResidencyError) -> &'static str {
    match e {
        ResidencyError::CannotAdmit(_) => "cannot-admit",
        ResidencyError::SlotUnhealthy(_) => "slot-unhealthy",
    }
}

/// Host-RAM budget (GB) for the CPU-tier fit check, from config (`HOST_RAM_BUDGET_GB`).
/// `None` ⇒ the check is skipped (no literal default host size guessed).
fn config_host_ram_budget_gb() -> Option<f64> {
    std::env::var("HOST_RAM_BUDGET_GB")
        .ok()
        .and_then(|v| v.trim().parse().ok())
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass-through residency stub (SRV-04 placeholder; SRV-05 replaces it)
// ─────────────────────────────────────────────────────────────────────────────

/// Minimal [`ResidencyManager`] stub so SRV-04 compiles and tests before SRV-05.
///
/// It does NO admission or eviction. It runs an injected launch closure to bring
/// the keep-warm model up, health-checks it, and wraps the result as a [`Slot`].
/// This is deliberately the trivial pass-through: SRV-05 replaces it with the real
/// VRAM-admission + tier-aware eviction (`residency.rs` / `eviction.rs`) WITHOUT
/// changing SRV-04 — the trait boundary is the seam.
///
/// NOTE: even the stub goes through the trait, so a keep-warm model is STILL not
/// cold-launched on the inline `serve_model` path — it is "launched" here behind
/// the residency seam, which is the contract SRV-05 hardens.
pub struct PassThroughResidency<'a> {
    spawner: &'a dyn RuntimeSpawner,
    health: &'a dyn HealthChecker,
    /// Resolved weights path for the keep-warm model (caller-provided).
    gguf_path: String,
    /// The route entry to launch (so the stub can build the command).
    entry: RouteEntry,
}

impl<'a> PassThroughResidency<'a> {
    /// Build the stub for one keep-warm model's route.
    pub fn new(
        spawner: &'a dyn RuntimeSpawner,
        health: &'a dyn HealthChecker,
        entry: RouteEntry,
        gguf_path: impl Into<String>,
    ) -> Self {
        PassThroughResidency {
            spawner,
            health,
            gguf_path: gguf_path.into(),
            entry,
        }
    }
}

#[async_trait]
impl<'a> ResidencyManager for PassThroughResidency<'a> {
    async fn acquire_warm_slot(
        &self,
        model_id: &ModelId,
        _vram_gb: Option<f64>,
    ) -> Result<Slot, ResidencyError> {
        let model_str = model_id.as_str().to_string();
        let runtime = self.entry.best_runtime();
        let cmd = build_launch_command(&self.entry, runtime, &self.gguf_path)
            .map_err(|_| ResidencyError::CannotAdmit(model_str.clone()))?;
        self.spawner
            .spawn(&cmd)
            .await
            .map_err(|_| ResidencyError::CannotAdmit(model_str.clone()))?;
        if !self.health.check(&cmd.health_url).await {
            return Err(ResidencyError::SlotUnhealthy(model_str));
        }
        Ok(Slot {
            model_id: model_str,
            runtime,
            endpoint: cmd.health_url,
            // The pass-through stub does not isolate (SRV-05's real residency
            // manager owns warm-slot isolation); no namespace to record here.
            netns: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serving::profile::RouteEntry;
    use serial_test::serial;
    use std::sync::Mutex;
    use terminus_rs::intake::serving::{
        ExclusionReason, RecheckTrigger, ServingBackend, ServingProfile,
    };

    fn entry(
        model: &str,
        backend: ServingBackend,
        best: Runtime,
        env_json: &str,
        keep_warm: bool,
        fallback: Option<Runtime>,
    ) -> RouteEntry {
        RouteEntry::from_profile(ServingProfile {
            model_id: ModelId::from(model),
            backend_tag: backend,
            best_runtime: best,
            env_json: env_json.into(),
            tok_s: Some(30.0),
            vram_or_ram_peak_gb: Some(8.0),
            cold_load_s: Some(10.0),
            keep_warm,
            fallback_runtime: fallback,
            exclusion_reason: ExclusionReason::None,
            recheck_trigger: RecheckTrigger::None,
            provenance: None,
        })
    }

    /// Set the three runtime endpoint env vars so config helpers resolve. Returns
    /// nothing — caller uses `#[serial]` to avoid env races.
    fn set_runtime_endpoints() {
        std::env::set_var("LLAMA_SERVER_URL", "http://llama.invalid/health");
        std::env::set_var("OLLAMA_URL", "http://ollama.invalid/health");
        std::env::set_var("OLLAMA_CPU_URL", "http://ollama-cpu.invalid/health");
    }

    #[test]
    #[serial]
    fn llama_command_has_no_mmap_and_flash_when_set() {
        set_runtime_endpoints();
        let e = entry(
            "big:70b",
            ServingBackend::LlamaGpu,
            Runtime::LlamaCpp,
            r#"{"gfx_override":"11.0.0","mmap":false,"flash_attn":true}"#,
            false,
            None,
        );
        let cmd = build_launch_command(&e, Runtime::LlamaCpp, "/w/model.gguf").unwrap();
        assert!(cmd.args.contains(&"--no-mmap".to_string()));
        assert!(cmd.args.contains(&"--flash-attn".to_string()));
        assert!(cmd.args.contains(&"/w/model.gguf".to_string()));
        assert!(cmd
            .env
            .iter()
            .any(|(k, v)| k == GFX_OVERRIDE_ENV && v == "11.0.0"));
    }

    #[test]
    #[serial]
    fn llama_command_emits_explicit_ctx_flag() {
        set_runtime_endpoints();
        let e = entry(
            "qwen3-coder:30b",
            ServingBackend::LlamaGpu,
            Runtime::LlamaCpp,
            r#"{"n_ctx":98304}"#,
            false,
            None,
        );
        let cmd = build_launch_command(&e, Runtime::LlamaCpp, "/w/m.gguf").unwrap();
        // -c <n> present (the SRV-12 explicit-context sidestep).
        let pos = cmd.args.iter().position(|a| a == "-c").expect("-c emitted");
        assert_eq!(cmd.args.get(pos + 1).map(String::as_str), Some("98304"));
    }

    // ── YARN-01: rope-scaling flag emission ────────────────────────────────────

    #[test]
    #[serial]
    fn llama_command_emits_yarn_flags_when_validated() {
        set_runtime_endpoints();
        let e = entry(
            "big:coder",
            ServingBackend::LlamaGpu,
            Runtime::LlamaCpp,
            r#"{"rope_scaling":{"method":"yarn","rope_scale":4.0,"yarn_orig_ctx":32768,
                "target_ctx":131072,"ext_factor":1.0,"attn_factor":1.0,"beta_slow":1.0,
                "beta_fast":32.0,"validated":true}}"#,
            false,
            None,
        );
        let cmd = build_launch_command(&e, Runtime::LlamaCpp, "/w/m.gguf").unwrap();
        // Exact expected flag string (order matters — this is what's exec'd).
        let expected: Vec<String> = [
            "--model", "/w/m.gguf",
            "--rope-scaling", "yarn",
            "--rope-scale", "4",
            "--yarn-orig-ctx", "32768",
            "--yarn-ext-factor", "1",
            "--yarn-attn-factor", "1",
            "--yarn-beta-slow", "1",
            "--yarn-beta-fast", "32",
            "--ctx-size", "131072",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        assert_eq!(cmd.args, expected);
    }

    #[test]
    #[serial]
    fn llama_command_unvalidated_yarn_emits_no_flags_native_context() {
        // NEGATIVE TEST: method=yarn but validated=false → no yarn/ctx-size flags,
        // native context (the "configured but not validated" path).
        set_runtime_endpoints();
        let e = entry(
            "big:coder",
            ServingBackend::LlamaGpu,
            Runtime::LlamaCpp,
            r#"{"rope_scaling":{"method":"yarn","rope_scale":4.0,"yarn_orig_ctx":32768,
                "target_ctx":131072,"ext_factor":1.0,"attn_factor":1.0,"beta_slow":1.0,
                "beta_fast":32.0,"validated":false}}"#,
            false,
            None,
        );
        let cmd = build_launch_command(&e, Runtime::LlamaCpp, "/w/m.gguf").unwrap();
        assert_eq!(cmd.args, vec!["--model".to_string(), "/w/m.gguf".to_string()]);
        assert!(!cmd.args.iter().any(|a| a.starts_with("--rope") || a.starts_with("--yarn")));
        assert!(!cmd.args.contains(&"--ctx-size".to_string()));
    }

    #[test]
    #[serial]
    fn llama_command_implausible_validated_yarn_emits_no_flags() {
        // NEGATIVE TEST: validated=true does NOT license emitting garbage numbers
        // verbatim to llama-server. A nonsensical rope_scale (<=0) must refuse to
        // emit, same as the unvalidated path (native context, no flags).
        set_runtime_endpoints();
        let e = entry(
            "big:coder",
            ServingBackend::LlamaGpu,
            Runtime::LlamaCpp,
            r#"{"rope_scaling":{"method":"yarn","rope_scale":-1.0,"yarn_orig_ctx":32768,
                "target_ctx":131072,"ext_factor":1.0,"attn_factor":1.0,"beta_slow":1.0,
                "beta_fast":32.0,"validated":true}}"#,
            false,
            None,
        );
        let cmd = build_launch_command(&e, Runtime::LlamaCpp, "/w/m.gguf").unwrap();
        assert_eq!(cmd.args, vec!["--model".to_string(), "/w/m.gguf".to_string()]);
        assert!(!cmd.args.iter().any(|a| a.starts_with("--rope") || a.starts_with("--yarn")));
        assert!(!cmd.args.contains(&"--ctx-size".to_string()));

        // An ext_factor wildly outside [0,1] is equally implausible.
        let e2 = entry(
            "big:coder",
            ServingBackend::LlamaGpu,
            Runtime::LlamaCpp,
            r#"{"rope_scaling":{"method":"yarn","rope_scale":4.0,"yarn_orig_ctx":32768,
                "target_ctx":131072,"ext_factor":50.0,"attn_factor":1.0,"beta_slow":1.0,
                "beta_fast":32.0,"validated":true}}"#,
            false,
            None,
        );
        let cmd2 = build_launch_command(&e2, Runtime::LlamaCpp, "/w/m.gguf").unwrap();
        assert_eq!(cmd2.args, vec!["--model".to_string(), "/w/m.gguf".to_string()]);
    }

    #[test]
    #[serial]
    fn llama_command_method_none_emits_no_scaling_flags() {
        // method=none (or the block absent entirely) → unchanged behavior.
        set_runtime_endpoints();
        let e = entry(
            "m",
            ServingBackend::LlamaGpu,
            Runtime::LlamaCpp,
            r#"{"rope_scaling":{"method":"none"}}"#,
            false,
            None,
        );
        let cmd = build_launch_command(&e, Runtime::LlamaCpp, "/w/m.gguf").unwrap();
        assert_eq!(cmd.args, vec!["--model".to_string(), "/w/m.gguf".to_string()]);

        // No block at all — identical (unchanged) behavior.
        let e2 = entry("m", ServingBackend::LlamaGpu, Runtime::LlamaCpp, "{}", false, None);
        let cmd2 = build_launch_command(&e2, Runtime::LlamaCpp, "/w/m.gguf").unwrap();
        assert_eq!(cmd2.args, vec!["--model".to_string(), "/w/m.gguf".to_string()]);
    }

    #[test]
    #[serial]
    fn llama_command_emits_linear_flags_without_yarn_finetune() {
        set_runtime_endpoints();
        let e = entry(
            "m",
            ServingBackend::LlamaGpu,
            Runtime::LlamaCpp,
            r#"{"rope_scaling":{"method":"linear","rope_scale":2.0,"target_ctx":16384,"validated":true}}"#,
            false,
            None,
        );
        let cmd = build_launch_command(&e, Runtime::LlamaCpp, "/w/m.gguf").unwrap();
        let expected: Vec<String> = [
            "--model", "/w/m.gguf",
            "--rope-scaling", "linear",
            "--rope-scale", "2",
            "--ctx-size", "16384",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        assert_eq!(cmd.args, expected);
        // No yarn-specific fine-tune flags for linear.
        assert!(!cmd.args.iter().any(|a| a.starts_with("--yarn")));
    }

    #[test]
    #[serial]
    fn llama_command_no_op_when_target_ctx_not_greater_than_orig() {
        // target_ctx <= yarn_orig_ctx → no extension needed, no-op even validated.
        set_runtime_endpoints();
        let e = entry(
            "m",
            ServingBackend::LlamaGpu,
            Runtime::LlamaCpp,
            r#"{"rope_scaling":{"method":"yarn","rope_scale":4.0,"yarn_orig_ctx":32768,
                "target_ctx":32768,"ext_factor":1.0,"attn_factor":1.0,"beta_slow":1.0,
                "beta_fast":32.0,"validated":true}}"#,
            false,
            None,
        );
        let cmd = build_launch_command(&e, Runtime::LlamaCpp, "/w/m.gguf").unwrap();
        assert_eq!(cmd.args, vec!["--model".to_string(), "/w/m.gguf".to_string()]);
    }

    #[test]
    #[serial]
    fn ollama_command_ignores_yarn_rope_scaling_never_crashes() {
        // NEGATIVE TEST: ollama tier cannot apply rope scaling — ignored, logged,
        // and the command builds successfully (no panic, no flags).
        set_runtime_endpoints();
        let e = entry(
            "gemma4:9b",
            ServingBackend::OllamaGpu,
            Runtime::Ollama,
            r#"{"rope_scaling":{"method":"yarn","rope_scale":4.0,"yarn_orig_ctx":32768,
                "target_ctx":131072,"ext_factor":1.0,"attn_factor":1.0,"beta_slow":1.0,
                "beta_fast":32.0,"validated":true}}"#,
            false,
            None,
        );
        let cmd = build_launch_command(&e, Runtime::Ollama, "ignored").unwrap();
        assert_eq!(cmd.args, vec!["serve".to_string()]);
    }

    #[test]
    #[serial]
    fn cpu_command_ignores_yarn_rope_scaling_never_crashes() {
        set_runtime_endpoints();
        let e = entry(
            "small:3b",
            ServingBackend::Cpu,
            Runtime::Cpu,
            r#"{"rope_scaling":{"method":"yarn","rope_scale":4.0,"yarn_orig_ctx":32768,
                "target_ctx":131072,"ext_factor":1.0,"attn_factor":1.0,"beta_slow":1.0,
                "beta_fast":32.0,"validated":true}}"#,
            false,
            None,
        );
        let cmd = build_launch_command(&e, Runtime::Cpu, "ignored").unwrap();
        assert_eq!(cmd.args, vec!["serve".to_string()]);
    }

    // ── YARN-05: thinking-preservation flag emission ───────────────────────────

    #[test]
    #[serial]
    fn llama_command_emits_thinking_flags_when_validated() {
        set_runtime_endpoints();
        let e = entry(
            "reasoner:30b",
            ServingBackend::LlamaGpu,
            Runtime::LlamaCpp,
            r#"{"thinking":{"supports_thinking":true,"preserve_thinking":true,
                "requires_prefix_caching":true,"validated":true}}"#,
            false,
            None,
        );
        let cmd = build_launch_command(&e, Runtime::LlamaCpp, "/w/m.gguf").unwrap();
        assert!(cmd.args.contains(&"--preserve-thinking".to_string()));
        assert!(cmd.args.contains(&"--prompt-cache-all".to_string()));
    }

    #[test]
    #[serial]
    fn llama_command_emits_thinking_flag_without_cache_flag_when_not_required() {
        set_runtime_endpoints();
        let e = entry(
            "reasoner:30b",
            ServingBackend::LlamaGpu,
            Runtime::LlamaCpp,
            r#"{"thinking":{"supports_thinking":true,"preserve_thinking":true,
                "requires_prefix_caching":false,"validated":true}}"#,
            false,
            None,
        );
        let cmd = build_launch_command(&e, Runtime::LlamaCpp, "/w/m.gguf").unwrap();
        assert!(cmd.args.contains(&"--preserve-thinking".to_string()));
        assert!(!cmd.args.contains(&"--prompt-cache-all".to_string()));
    }

    #[test]
    #[serial]
    fn llama_command_unvalidated_thinking_emits_no_flags() {
        set_runtime_endpoints();
        let e = entry(
            "reasoner:30b",
            ServingBackend::LlamaGpu,
            Runtime::LlamaCpp,
            r#"{"thinking":{"supports_thinking":true,"preserve_thinking":true,
                "requires_prefix_caching":true,"validated":false}}"#,
            false,
            None,
        );
        let cmd = build_launch_command(&e, Runtime::LlamaCpp, "/w/m.gguf").unwrap();
        // Unvalidated ⇒ served without preservation (warning logged, no flags).
        assert!(!cmd.args.contains(&"--preserve-thinking".to_string()));
        assert!(!cmd.args.contains(&"--prompt-cache-all".to_string()));
    }

    #[test]
    #[serial]
    fn llama_command_no_supports_thinking_emits_no_flags_unchanged_baseline() {
        set_runtime_endpoints();
        // No `thinking` block at all — the unchanged, no-op baseline.
        let e = entry("plain:7b", ServingBackend::LlamaGpu, Runtime::LlamaCpp, "{}", false, None);
        let cmd = build_launch_command(&e, Runtime::LlamaCpp, "/w/m.gguf").unwrap();
        assert!(!cmd.args.contains(&"--preserve-thinking".to_string()));
        assert!(!cmd.args.contains(&"--prompt-cache-all".to_string()));
    }

    #[test]
    #[serial]
    fn llama_command_preserve_requested_on_non_supporting_model_refuses() {
        set_runtime_endpoints();
        // preserve_thinking=true but supports_thinking=false ⇒ refuse, no flags,
        // never emit a meaningless preservation flag.
        let e = entry(
            "plain:7b",
            ServingBackend::LlamaGpu,
            Runtime::LlamaCpp,
            r#"{"thinking":{"supports_thinking":false,"preserve_thinking":true,
                "requires_prefix_caching":true,"validated":true}}"#,
            false,
            None,
        );
        let cmd = build_launch_command(&e, Runtime::LlamaCpp, "/w/m.gguf").unwrap();
        assert!(!cmd.args.contains(&"--preserve-thinking".to_string()));
        assert!(!cmd.args.contains(&"--prompt-cache-all".to_string()));
    }

    #[test]
    #[serial]
    fn ollama_command_ignores_thinking_never_crashes() {
        set_runtime_endpoints();
        let e = entry(
            "reasoner:30b",
            ServingBackend::OllamaGpu,
            Runtime::Ollama,
            r#"{"thinking":{"supports_thinking":true,"preserve_thinking":true,
                "requires_prefix_caching":true,"validated":true}}"#,
            false,
            None,
        );
        let cmd = build_launch_command(&e, Runtime::Ollama, "ignored").unwrap();
        // No crash, no invented support — plain serve command, same as baseline.
        assert_eq!(cmd.args, vec!["serve".to_string()]);
    }

    #[test]
    #[serial]
    fn cpu_command_ignores_thinking_never_crashes() {
        set_runtime_endpoints();
        let e = entry(
            "small:3b",
            ServingBackend::Cpu,
            Runtime::Cpu,
            r#"{"thinking":{"supports_thinking":true,"preserve_thinking":true,
                "requires_prefix_caching":true,"validated":true}}"#,
            false,
            None,
        );
        let cmd = build_launch_command(&e, Runtime::Cpu, "ignored").unwrap();
        assert_eq!(cmd.args, vec!["serve".to_string()]);
    }

    #[test]
    #[serial]
    fn llama_command_replays_runner_native_env_schema() {
        // The SRV-02/seed shape: gfx_override bool + mmap_flag int. Previously this
        // dropped --no-mmap silently (the bug this fix closes). With the host gfx
        // override configured, `gfx_override: true` applies it.
        set_runtime_endpoints();
        std::env::set_var("CHORD_GFX_OVERRIDE_VERSION", "11.5.1");
        let e = entry(
            "minimax-m2.7",
            ServingBackend::LlamaGpu,
            Runtime::LlamaCpp,
            r#"{"gfx_override":true,"mmap_flag":0,"flash_attn":false,"cpu_lib":null}"#,
            false,
            None,
        );
        let cmd = build_launch_command(&e, Runtime::LlamaCpp, "/w/m.gguf").unwrap();
        assert!(
            cmd.args.contains(&"--no-mmap".to_string()),
            "mmap_flag:0 must replay as --no-mmap"
        );
        assert!(cmd
            .env
            .iter()
            .any(|(k, v)| k == GFX_OVERRIDE_ENV && v == "11.5.1"));
        std::env::remove_var("CHORD_GFX_OVERRIDE_VERSION");
    }

    #[test]
    #[serial]
    fn llama_gfx_apply_default_omitted_when_host_unset() {
        // gfx_override:true but no host value configured ⇒ omit (never guess).
        set_runtime_endpoints();
        std::env::remove_var("CHORD_GFX_OVERRIDE_VERSION");
        let e = entry(
            "m",
            ServingBackend::LlamaGpu,
            Runtime::LlamaCpp,
            r#"{"gfx_override":true}"#,
            false,
            None,
        );
        let cmd = build_launch_command(&e, Runtime::LlamaCpp, "/w/m.gguf").unwrap();
        assert!(!cmd.env.iter().any(|(k, _)| k == GFX_OVERRIDE_ENV));
    }

    #[test]
    #[serial]
    fn llama_command_omits_mmap_flag_when_mmap_true_or_absent() {
        set_runtime_endpoints();
        // mmap absent → no flag.
        let e = entry("m", ServingBackend::LlamaGpu, Runtime::LlamaCpp, "{}", false, None);
        let cmd = build_launch_command(&e, Runtime::LlamaCpp, "/w/m.gguf").unwrap();
        assert!(!cmd.args.contains(&"--no-mmap".to_string()));
        assert!(!cmd.args.contains(&"--flash-attn".to_string()));
        // mmap == true → still no flag (mmap stays on).
        let e2 = entry(
            "m",
            ServingBackend::LlamaGpu,
            Runtime::LlamaCpp,
            r#"{"mmap":true}"#,
            false,
            None,
        );
        let cmd2 = build_launch_command(&e2, Runtime::LlamaCpp, "/w/m.gguf").unwrap();
        assert!(!cmd2.args.contains(&"--no-mmap".to_string()));
    }

    #[test]
    #[serial]
    fn ollama_command_is_serve_with_gfx() {
        set_runtime_endpoints();
        let e = entry(
            "gemma4:9b",
            ServingBackend::OllamaGpu,
            Runtime::Ollama,
            r#"{"gfx_override":"11.0.0"}"#,
            false,
            None,
        );
        let cmd = build_launch_command(&e, Runtime::Ollama, "ignored").unwrap();
        assert_eq!(cmd.args, vec!["serve".to_string()]);
        assert!(cmd.env.iter().any(|(k, v)| k == GFX_OVERRIDE_ENV && v == "11.0.0"));
    }

    #[test]
    #[serial]
    fn cpu_command_sets_empty_gfx_and_cpu_lib() {
        set_runtime_endpoints();
        let e = entry(
            "small:3b",
            ServingBackend::Cpu,
            Runtime::Cpu,
            r#"{"gfx_override":"","cpu_library":"cpu_avx2"}"#,
            false,
            None,
        );
        let cmd = build_launch_command(&e, Runtime::Cpu, "ignored").unwrap();
        // Empty gfx override is SET (the genuine-CPU empty-override path).
        assert!(cmd.env.iter().any(|(k, v)| k == GFX_OVERRIDE_ENV && v.is_empty()));
        assert!(cmd.env.iter().any(|(k, v)| k == CPU_LIBRARY_ENV && v == "cpu_avx2"));
    }

    #[test]
    #[serial]
    fn build_command_unconfigured_runtime_errors_without_host() {
        std::env::remove_var("LLAMA_SERVER_URL");
        let e = entry("m", ServingBackend::LlamaGpu, Runtime::LlamaCpp, "{}", false, None);
        let err = build_launch_command(&e, Runtime::LlamaCpp, "/w/m.gguf").unwrap_err();
        assert_eq!(err, LaunchError::RuntimeNotConfigured);
        // The error string carries no infra.
        let s = err.to_string();
        assert!(!s.contains("://") && !s.contains("invalid"));
        set_runtime_endpoints();
    }

    #[test]
    fn launch_errors_carry_no_infra() {
        let errs = [
            LaunchError::UnprofiledModel("m".into()),
            LaunchError::RuntimeNotConfigured,
            LaunchError::AllRuntimesFailed("m".into()),
            LaunchError::KeepWarmMustUseResidency("m".into()),
            LaunchError::CpuModelExceedsHostRam("m".into()),
            LaunchError::ProfileStoreUnavailable,
            LaunchError::IsolationRefused("m".into()),
        ];
        for e in errs {
            let s = e.to_string();
            assert!(!s.contains("192.168."), "{s}");
            assert!(!s.contains("://"), "{s}");
            assert!(!s.contains('@'), "{s}");
            assert!(!s.contains("llama-server"), "{s}");
            assert!(!s.contains(".gguf"), "{s}");
        }
    }

    // ── scripted collaborators ────────────────────────────────────────────────

    /// Spawner scripted per-runtime: succeed/fail by runtime; records calls.
    struct ScriptedSpawner {
        fail_runtimes: Vec<Runtime>,
        calls: Mutex<Vec<Runtime>>,
    }
    #[async_trait]
    impl RuntimeSpawner for ScriptedSpawner {
        async fn spawn(&self, cmd: &LaunchCommand) -> Result<(), String> {
            self.calls.lock().unwrap().push(cmd.runtime);
            if self.fail_runtimes.contains(&cmd.runtime) {
                Err("scripted spawn failure".into())
            } else {
                Ok(())
            }
        }
    }

    /// Health checker scripted by which runtimes' endpoints are healthy.
    struct ScriptedHealth {
        healthy: bool,
    }
    #[async_trait]
    impl HealthChecker for ScriptedHealth {
        async fn check(&self, _endpoint: &str) -> bool {
            self.healthy
        }
    }

    struct CountingRecorder {
        failures: Mutex<Vec<(String, Runtime, String)>>,
    }
    impl FailureRecorder for CountingRecorder {
        fn record_failure(&self, model_id: &str, runtime: Runtime, reason: &str) {
            self.failures
                .lock()
                .unwrap()
                .push((model_id.to_string(), runtime, reason.to_string()));
        }
    }

    /// A residency manager that records whether it was called (for the keep_warm
    /// delegation test) and returns a fixed healthy slot.
    struct RecordingResidency {
        called: Mutex<bool>,
        endpoint: String,
    }
    #[async_trait]
    impl ResidencyManager for RecordingResidency {
        async fn acquire_warm_slot(
            &self,
            model_id: &ModelId,
            _vram_gb: Option<f64>,
        ) -> Result<Slot, ResidencyError> {
            *self.called.lock().unwrap() = true;
            Ok(Slot {
                model_id: model_id.as_str().to_string(),
                runtime: Runtime::LlamaCpp,
                endpoint: self.endpoint.clone(),
                netns: None,
            })
        }
    }

    /// A residency manager that PANICS if called — proves the inline path never
    /// touches it for a non-keep-warm model.
    struct NeverResidency;
    #[async_trait]
    impl ResidencyManager for NeverResidency {
        async fn acquire_warm_slot(
            &self,
            _model_id: &ModelId,
            _vram_gb: Option<f64>,
        ) -> Result<Slot, ResidencyError> {
            panic!("residency manager must not be called for a non-keep-warm model");
        }
    }

    #[tokio::test]
    #[serial]
    async fn request_launches_best_runtime_and_health_checks() {
        set_runtime_endpoints();
        let spawner = ScriptedSpawner {
            fail_runtimes: vec![],
            calls: Mutex::new(vec![]),
        };
        let health = ScriptedHealth { healthy: true };
        let rec = CountingRecorder {
            failures: Mutex::new(vec![]),
        };
        let resi = NeverResidency;
        let l = Launcher::new(&spawner, &health, &rec, &resi);
        let e = entry(
            "qwen3:8b",
            ServingBackend::LlamaGpu,
            Runtime::LlamaCpp,
            "{}",
            false,
            Some(Runtime::Ollama),
        );
        let h = l
            .serve_model(&ModelId::from("qwen3:8b"), &e, "/w/q.gguf")
            .await
            .unwrap();
        assert_eq!(h.runtime, Runtime::LlamaCpp);
        assert!(!h.from_warm_slot);
        assert_eq!(*spawner.calls.lock().unwrap(), vec![Runtime::LlamaCpp]);
        assert!(rec.failures.lock().unwrap().is_empty());
    }

    #[tokio::test]
    #[serial]
    async fn best_fails_then_fallback_succeeds() {
        set_runtime_endpoints();
        let spawner = ScriptedSpawner {
            fail_runtimes: vec![Runtime::LlamaCpp],
            calls: Mutex::new(vec![]),
        };
        let health = ScriptedHealth { healthy: true };
        let rec = CountingRecorder {
            failures: Mutex::new(vec![]),
        };
        let resi = NeverResidency;
        let l = Launcher::new(&spawner, &health, &rec, &resi);
        let e = entry(
            "m",
            ServingBackend::LlamaGpu,
            Runtime::LlamaCpp,
            "{}",
            false,
            Some(Runtime::Ollama),
        );
        let h = l.serve_model(&ModelId::from("m"), &e, "/w/m.gguf").await.unwrap();
        // Fell back to ollama.
        assert_eq!(h.runtime, Runtime::Ollama);
        assert_eq!(
            *spawner.calls.lock().unwrap(),
            vec![Runtime::LlamaCpp, Runtime::Ollama]
        );
        // The best-runtime failure was recorded.
        assert!(rec
            .failures
            .lock()
            .unwrap()
            .iter()
            .any(|(_, rt, reason)| *rt == Runtime::LlamaCpp && reason == "launch-failed"));
    }

    #[tokio::test]
    #[serial]
    async fn both_fail_gives_genericized_error_and_records() {
        set_runtime_endpoints();
        let spawner = ScriptedSpawner {
            fail_runtimes: vec![Runtime::LlamaCpp, Runtime::Ollama],
            calls: Mutex::new(vec![]),
        };
        let health = ScriptedHealth { healthy: true };
        let rec = CountingRecorder {
            failures: Mutex::new(vec![]),
        };
        let resi = NeverResidency;
        let l = Launcher::new(&spawner, &health, &rec, &resi);
        let e = entry(
            "m",
            ServingBackend::LlamaGpu,
            Runtime::LlamaCpp,
            "{}",
            false,
            Some(Runtime::Ollama),
        );
        let err = l
            .serve_model(&ModelId::from("m"), &e, "/w/m.gguf")
            .await
            .unwrap_err();
        assert!(matches!(err, LaunchError::AllRuntimesFailed(_)));
        // Terminal "all failed" recorded plus both per-runtime failures.
        let f = rec.failures.lock().unwrap();
        assert!(f.iter().any(|(_, _, r)| r == "all-runtimes-failed"));
        let s = err.to_string();
        assert!(!s.contains("://") && !s.contains(".gguf"));
    }

    #[tokio::test]
    #[serial]
    async fn health_fail_on_best_triggers_fallback() {
        set_runtime_endpoints();
        // spawn succeeds for both, but health fails → both treated as failed.
        let spawner = ScriptedSpawner {
            fail_runtimes: vec![],
            calls: Mutex::new(vec![]),
        };
        let health = ScriptedHealth { healthy: false };
        let rec = CountingRecorder {
            failures: Mutex::new(vec![]),
        };
        let resi = NeverResidency;
        let l = Launcher::new(&spawner, &health, &rec, &resi);
        let e = entry(
            "m",
            ServingBackend::LlamaGpu,
            Runtime::LlamaCpp,
            "{}",
            false,
            Some(Runtime::Ollama),
        );
        let err = l
            .serve_model(&ModelId::from("m"), &e, "/w/m.gguf")
            .await
            .unwrap_err();
        assert!(matches!(err, LaunchError::AllRuntimesFailed(_)));
        // Both runtimes were attempted (health failed on each).
        assert_eq!(
            *spawner.calls.lock().unwrap(),
            vec![Runtime::LlamaCpp, Runtime::Ollama]
        );
        assert!(rec
            .failures
            .lock()
            .unwrap()
            .iter()
            .any(|(_, _, r)| r == "health-check-failed"));
    }

    #[tokio::test]
    #[serial]
    async fn keep_warm_delegates_to_residency_not_cold_launch() {
        set_runtime_endpoints();
        // The spawner would PANIC-equivalent: if it is ever called for this
        // keep_warm model the test fails because we assert zero spawn calls AND
        // the residency manager (not the spawner) produced the endpoint.
        let spawner = ScriptedSpawner {
            fail_runtimes: vec![],
            calls: Mutex::new(vec![]),
        };
        let health = ScriptedHealth { healthy: true };
        let rec = CountingRecorder {
            failures: Mutex::new(vec![]),
        };
        let resi = RecordingResidency {
            called: Mutex::new(false),
            endpoint: "http://warm.invalid/health".into(),
        };
        let l = Launcher::new(&spawner, &health, &rec, &resi);
        let e = entry(
            "big:120b",
            ServingBackend::LlamaGpu,
            Runtime::LlamaCpp,
            "{}",
            true, // keep_warm
            Some(Runtime::Ollama),
        );
        let h = l
            .serve_model(&ModelId::from("big:120b"), &e, "/w/big.gguf")
            .await
            .unwrap();
        // Served from the warm slot.
        assert!(h.from_warm_slot);
        assert_eq!(h.endpoint, "http://warm.invalid/health");
        // Residency manager WAS consulted.
        assert!(*resi.called.lock().unwrap());
        // And the inline cold-launch spawner was NEVER called (the negative test).
        assert!(
            spawner.calls.lock().unwrap().is_empty(),
            "keep_warm model must NOT be cold-launched inline"
        );
    }

    #[tokio::test]
    #[serial]
    async fn try_runtime_refuses_keep_warm_defense_in_depth() {
        set_runtime_endpoints();
        let spawner = ScriptedSpawner {
            fail_runtimes: vec![],
            calls: Mutex::new(vec![]),
        };
        let health = ScriptedHealth { healthy: true };
        let rec = CountingRecorder {
            failures: Mutex::new(vec![]),
        };
        let resi = RecordingResidency {
            called: Mutex::new(false),
            endpoint: "http://warm.invalid/health".into(),
        };
        let l = Launcher::new(&spawner, &health, &rec, &resi);
        let e = entry("k", ServingBackend::LlamaGpu, Runtime::LlamaCpp, "{}", true, None);
        // Direct call to the private cold path must refuse keep_warm.
        let err = l
            .try_runtime("k", &e, Runtime::LlamaCpp, "/w/k.gguf")
            .await
            .unwrap_err();
        assert!(matches!(err, LaunchError::KeepWarmMustUseResidency(_)));
        assert!(spawner.calls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    #[serial]
    async fn cpu_only_model_exceeding_host_ram_is_refused() {
        set_runtime_endpoints();
        std::env::set_var("HOST_RAM_BUDGET_GB", "31");
        let spawner = ScriptedSpawner {
            fail_runtimes: vec![],
            calls: Mutex::new(vec![]),
        };
        let health = ScriptedHealth { healthy: true };
        let rec = CountingRecorder {
            failures: Mutex::new(vec![]),
        };
        let resi = NeverResidency;
        let l = Launcher::new(&spawner, &health, &rec, &resi);
        // CPU-only (no fallback), peak 8GB default from `entry`; bump to 64GB.
        let mut e = entry("huge", ServingBackend::Cpu, Runtime::Cpu, "{}", false, None);
        e.profile.vram_or_ram_peak_gb = Some(64.0);
        let err = l
            .serve_model(&ModelId::from("huge"), &e, "ignored")
            .await
            .unwrap_err();
        assert!(matches!(err, LaunchError::CpuModelExceedsHostRam(_)));
        // Never attempted a launch.
        assert!(spawner.calls.lock().unwrap().is_empty());
        std::env::remove_var("HOST_RAM_BUDGET_GB");
    }

    #[test]
    #[serial]
    fn scrub_layers_telemetry_off_under_runtime_env() {
        // The gfx override (a runtime-specific env pair) must SURVIVE the scrub and
        // sit ON TOP of the telemetry-off base.
        set_runtime_endpoints();
        let e = entry(
            "g",
            ServingBackend::OllamaGpu,
            Runtime::Ollama,
            r#"{"gfx_override":"11.0.0"}"#,
            false,
            None,
        );
        let cmd = build_launch_command(&e, Runtime::Ollama, "ignored").unwrap();
        let cfg = Config::test_default();
        let scrubbed = scrub_launch_env(cmd, RuntimeClass::Serve, &cfg);
        // Telemetry-off vars present.
        assert!(scrubbed
            .env
            .iter()
            .any(|(k, v)| k == "DO_NOT_TRACK" && v == "1"));
        assert!(scrubbed
            .env
            .iter()
            .any(|(k, v)| k == "HF_HUB_OFFLINE" && v == "1"));
        // The runtime-specific gfx override survived the layering.
        assert!(scrubbed
            .env
            .iter()
            .any(|(k, v)| k == GFX_OVERRIDE_ENV && v == "11.0.0"));
    }

    #[tokio::test]
    #[serial]
    async fn launcher_with_scrub_spawns_scrubbed_env() {
        // End-to-end through the launcher: the spawned command carries the ISO-01
        // telemetry-off vars when the launcher is built `with_scrub`.
        set_runtime_endpoints();
        struct CapturingSpawner {
            env: Mutex<Vec<(String, String)>>,
        }
        #[async_trait]
        impl RuntimeSpawner for CapturingSpawner {
            async fn spawn(&self, cmd: &LaunchCommand) -> Result<(), String> {
                *self.env.lock().unwrap() = cmd.env.clone();
                Ok(())
            }
        }
        let spawner = CapturingSpawner { env: Mutex::new(vec![]) };
        let health = ScriptedHealth { healthy: true };
        let rec = CountingRecorder { failures: Mutex::new(vec![]) };
        let resi = NeverResidency;
        let cfg = Config::test_default();
        // ISO-02: on this unprivileged host the netns can't be created. Use the dev
        // opt-out so the ISO-01 env scrub is still exercised end-to-end (the
        // isolation fail-closed path is asserted by its own tests below).
        std::env::set_var("CHORD_NETNS_ISOLATION", "0");
        let l = Launcher::with_scrub(&spawner, &health, &rec, &resi, &cfg);
        let e = entry("m", ServingBackend::LlamaGpu, Runtime::LlamaCpp, "{}", false, None);
        l.serve_model(&ModelId::from("m"), &e, "/w/m.gguf").await.unwrap();
        std::env::remove_var("CHORD_NETNS_ISOLATION");
        let env = spawner.env.lock().unwrap();
        assert!(env.iter().any(|(k, v)| k == "DO_NOT_TRACK" && v == "1"));
        assert!(env.iter().any(|(k, v)| k == "TRANSFORMERS_OFFLINE" && v == "1"));
    }

    // ── S88 ISO-02: launcher isolation integration ─────────────────────────────

    #[tokio::test]
    #[serial]
    async fn with_scrub_fails_closed_when_isolation_unavailable_and_no_override() {
        // NEGATIVE TEST (the load-bearing one): a `with_scrub` launcher on a host
        // WITHOUT CAP_NET_ADMIN and with isolation ON and NO override must REFUSE to
        // launch (IsolationRefused) — it must NEVER spawn the runtime with full host
        // egress. We assert the spawner was never called.
        set_runtime_endpoints();
        std::env::remove_var("CHORD_NETNS_ISOLATION"); // default ON
        std::env::remove_var("CHORD_ALLOW_UNISOLATED"); // no override
        let spawner = ScriptedSpawner { fail_runtimes: vec![], calls: Mutex::new(vec![]) };
        let health = ScriptedHealth { healthy: true };
        let rec = CountingRecorder { failures: Mutex::new(vec![]) };
        let resi = NeverResidency;
        let cfg = Config::test_default();
        let l = Launcher::with_scrub(&spawner, &health, &rec, &resi, &cfg);
        // No fallback so both attempts hit the same refusal; the error after the
        // chain is AllRuntimesFailed, but crucially NOTHING was spawned.
        let e = entry("m", ServingBackend::LlamaGpu, Runtime::LlamaCpp, "{}", false, None);
        let err = l.serve_model(&ModelId::from("m"), &e, "/w/m.gguf").await.unwrap_err();
        assert!(matches!(err, LaunchError::AllRuntimesFailed(_)));
        assert!(
            spawner.calls.lock().unwrap().is_empty(),
            "fail-closed: the runtime must NOT be spawned when isolation is unavailable"
        );
        // The isolation refusal was recorded.
        assert!(rec
            .failures
            .lock()
            .unwrap()
            .iter()
            .any(|(_, _, r)| r == "isolation-refused"));
    }

    #[tokio::test]
    #[serial]
    async fn with_scrub_override_permits_unisolated_spawn_loudly() {
        // With the explicit operator override, the same unprivileged host DOES
        // spawn (unisolated) — the sanctioned bypass. The serve handle carries no
        // namespace (none was created).
        set_runtime_endpoints();
        std::env::remove_var("CHORD_NETNS_ISOLATION");
        std::env::set_var("CHORD_ALLOW_UNISOLATED", "1");
        let spawner = ScriptedSpawner { fail_runtimes: vec![], calls: Mutex::new(vec![]) };
        let health = ScriptedHealth { healthy: true };
        let rec = CountingRecorder { failures: Mutex::new(vec![]) };
        let resi = NeverResidency;
        let cfg = Config::test_default();
        let l = Launcher::with_scrub(&spawner, &health, &rec, &resi, &cfg);
        let e = entry("m", ServingBackend::LlamaGpu, Runtime::LlamaCpp, "{}", false, None);
        let h = l.serve_model(&ModelId::from("m"), &e, "/w/m.gguf").await.unwrap();
        std::env::remove_var("CHORD_ALLOW_UNISOLATED");
        assert_eq!(*spawner.calls.lock().unwrap(), vec![Runtime::LlamaCpp]);
        assert!(h.netns.is_none(), "override path spawns without a namespace");
    }

    #[tokio::test]
    #[serial]
    async fn with_scrub_disabled_by_config_spawns_unisolated_legacy_path() {
        // CHORD_NETNS_ISOLATION=0 is the dev/CI opt-out: spawn, no namespace.
        set_runtime_endpoints();
        std::env::set_var("CHORD_NETNS_ISOLATION", "0");
        let spawner = ScriptedSpawner { fail_runtimes: vec![], calls: Mutex::new(vec![]) };
        let health = ScriptedHealth { healthy: true };
        let rec = CountingRecorder { failures: Mutex::new(vec![]) };
        let resi = NeverResidency;
        let cfg = Config::test_default();
        let l = Launcher::with_scrub(&spawner, &health, &rec, &resi, &cfg);
        let e = entry("m", ServingBackend::LlamaGpu, Runtime::LlamaCpp, "{}", false, None);
        let h = l.serve_model(&ModelId::from("m"), &e, "/w/m.gguf").await.unwrap();
        std::env::remove_var("CHORD_NETNS_ISOLATION");
        assert_eq!(*spawner.calls.lock().unwrap(), vec![Runtime::LlamaCpp]);
        assert!(h.netns.is_none());
    }

    #[tokio::test]
    #[serial]
    async fn passthrough_residency_stub_launches_and_healthchecks() {
        set_runtime_endpoints();
        let spawner = ScriptedSpawner {
            fail_runtimes: vec![],
            calls: Mutex::new(vec![]),
        };
        let health = ScriptedHealth { healthy: true };
        let e = entry("k", ServingBackend::LlamaGpu, Runtime::LlamaCpp, "{}", true, None);
        let stub = PassThroughResidency::new(&spawner, &health, e, "/w/k.gguf");
        let slot = stub
            .acquire_warm_slot(&ModelId::from("k"), Some(8.0))
            .await
            .unwrap();
        assert_eq!(slot.runtime, Runtime::LlamaCpp);
        // The stub DID launch (it is a pass-through), behind the residency seam.
        assert_eq!(*spawner.calls.lock().unwrap(), vec![Runtime::LlamaCpp]);
    }
}
