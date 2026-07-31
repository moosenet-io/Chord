//! TRTR-07: the assistant-mode RESIDENT MODEL SET, with mode-swap release.
//!
//! ## Why this exists
//! The tool router moved out of Chord and into Terminus (TERM #553). Every tool
//! turn therefore now costs a Chord inference call for the tool-SELECTING
//! sub-agent, on top of what Lumina already needs. In assistant mode Chord must
//! hold three models resident:
//!
//! | role | what it serves |
//! |---|---|
//! | [`Role::Personality`] | Lumina's voice (the chat turn the human reads) |
//! | [`Role::Router`] | the tool-selecting sub-agent (Terminus's router) |
//! | [`Role::Embedding`] | Engram memory |
//!
//! Before this, nothing was pinned: `MODEL_PROTECTED` was empty, Ollama's
//! `keep_alive` expires, and the TIER-03/04 sweep reclaims warm models — so a
//! single tool turn could cold-load several GB.
//!
//! ## Roles resolve through Chord ALIASES, never a concrete model name
//! Each role names a Chord **alias key** (default: personality → `lumina-deep`,
//! router → `lumina-fast` — the alias Terminus's router already asks for via
//! `TERMINUS_ROUTER_MODEL` — embedding → `lumina-embed`), overridable per role by
//! env. The alias is resolved through the dynamic [`LuminaAliasStore`] first and
//! the static `CHORD_MODEL_ALIASES` map second. A role whose alias resolves to
//! nothing is [`RoleState::Unresolved`] and simply degrades — the resident set
//! NEVER hard-wires a model name, because Chord owns model selection (north-star
//! Module Contract clause 1) and the dynamic alias updater must stay free to
//! repoint a role under us.
//!
//! ## Presence is VERIFIED, never assumed
//! This fleet has already been bitten by "protecting" a model that had never been
//! pulled (`qwen3-embedding:0.6b` sat in `MODEL_PROTECTED` while absent from
//! disk). So a resolved target is looked up in the model registry before it is
//! warmed or exempted; an unknown target is [`RoleState::Missing`], logged, and
//! skipped. We never exempt a model that does not exist.
//!
//! ## Mode swap = RELEASE, and the primitive already exists
//! Harmony's heavy builds acquire the BLD-11 idle lease and MINT's sweep enters
//! idle — both land on Chord's existing `POST /admin/idle`. So [`ResidentSet::release`]
//! is called from [`crate::admin::idle::enter_idle`] AFTER its closed-world drain
//! (an in-flight turn always finishes; we never yank a model out from under a
//! live request) and BEFORE it unloads VRAM. [`crate::admin::idle::activate`]
//! re-warms. No new mode-swap primitive is invented here.
//!
//! ## Lifecycle concurrency: RELEASE ALWAYS WINS (TRTR-07b)
//! A warm pass is mostly slow network I/O (an Ollama load can take minutes) and
//! it CANNOT hold a lock across that. Three callers can start one — startup,
//! `activate`, and the periodic reconcile — and a mode swap (`release`) can land
//! at any instant in the middle of any of them. The original shape read state,
//! dropped the lock, did the I/O, and then UNCONDITIONALLY installed the
//! exemption and set `active = true`; a release that landed mid-I/O was silently
//! overwritten, re-pinning models the idle path had already begun unloading. That
//! is precisely the failure the shared-GPU design forbids: a resident set that
//! refuses to yield.
//!
//! The guard is a **generation counter** plus a **single in-flight pass**:
//!
//! - [`Inner::generation`] is bumped by EVERY release (including a release of an
//!   already-inactive set, so even a first-ever warm in flight is cancelled).
//! - A warm pass captures the generation while admitting itself, does its I/O
//!   with no lock held, and then COMMITS under the `inner` lock: it re-reads the
//!   generation and, if it changed, **discards the entire pass** — no exemption
//!   is installed, `active` is not set, slots are not touched.
//! - The commit is ATOMIC with respect to a release because both take the `inner`
//!   lock and perform their registry mutation while still holding it (lock order
//!   is always `inner` → `model_registry`, so the two can never interleave and
//!   never deadlock).
//! - Therefore: once `release` has returned, its generation bump is visible to
//!   every in-flight pass, and each of them will discard at commit. No in-flight
//!   warm can re-install an exemption after a release. That is the invariant.
//! - `reconcile` additionally captures the generation at the moment it observes
//!   `active` and passes it in as an EXPECTATION, so a release landing between
//!   "observed active" and "started the pass" also cancels it.
//! - Concurrent warms COALESCE: the first pass registers an in-flight broadcast
//!   channel under the lock, and any concurrent caller subscribes and awaits that
//!   pass's report instead of issuing a duplicate set of warm requests.
//!
//! ## Residency is not lost on an ordinary tick
//! The models the set already holds are CONSUMING the free VRAM the planner
//! reads, so a naive re-plan on a steady-state tick classifies its own residents
//! as `DroppedVram` and then replaces the exemption with an incomplete set —
//! silently losing residency during normal operation. [`plan_warm`] therefore
//! takes the currently-resident model set and charges those models **zero** (they
//! are already in VRAM), and takes a `retainable` set of held models that are not
//! yet due for a keep-alive re-assert, which are [`WarmDecision::Retain`]ed with
//! NO warm request at all. A steady-state tick is consequently a no-op.
//!
//! ## A failed refresh never downgrades working residency
//! A transient Ollama hiccup on a re-assert must not turn a slot that is warm and
//! valid into `WarmFailed`/`DroppedVram` and drop it from the exemption — the
//! model is still loaded, and dropping the pin hands it to the eviction sweep for
//! nothing. At commit, a role whose new state would be `WarmFailed` or
//! `DroppedVram` but whose PREVIOUS state was `Warm` on the SAME model keeps its
//! `Warm` state and its exemption (counted as `preserved`).
//!
//! ## Fail SOFT, everywhere
//! Nothing on this path can block startup or refuse a turn. An unresolvable
//! alias, an absent model, an unreachable Ollama, an unreadable VRAM counter — all
//! degrade to a logged, observable role state. A cold turn is slower, not broken.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde::Serialize;
use tokio::sync::{broadcast, Mutex};
use tracing::{info, warn};

use super::lumina_alias::LuminaAliasStore;

/// The three role slots of the assistant-mode resident set.
///
/// **Declaration order is the documented degradation priority**: when VRAM cannot
/// hold all three, roles are warmed personality-first and the lowest-priority
/// role that does not fit is dropped (logged, reported) — never thrashed.
/// Rationale: a cold personality model is the only one the human directly feels
/// as latency; a cold router costs one extra load per tool turn; a cold embedding
/// call is the cheapest of the three.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    Personality,
    Router,
    Embedding,
}

impl Role {
    /// Every role in DEGRADATION PRIORITY order (highest priority first).
    pub const PRIORITY: [Role; 3] = [Role::Personality, Role::Router, Role::Embedding];

    /// Stable lowercase id used in logs and the control-API report.
    pub fn id(self) -> &'static str {
        match self {
            Role::Personality => "personality",
            Role::Router => "router",
            Role::Embedding => "embedding",
        }
    }

    /// Env var naming this role's ALIAS KEY (never a model name).
    fn alias_env(self) -> &'static str {
        match self {
            Role::Personality => "CHORD_RESIDENT_ROLE_PERSONALITY",
            Role::Router => "CHORD_RESIDENT_ROLE_ROUTER",
            Role::Embedding => "CHORD_RESIDENT_ROLE_EMBEDDING",
        }
    }

    /// Default ALIAS KEY for this role. These are Chord alias keys — logical
    /// routes Chord owns — not model names, so the dynamic alias updater stays in
    /// charge of what actually serves them.
    fn default_alias(self) -> &'static str {
        match self {
            // The deliberate chat tier — Lumina's voice.
            Role::Personality => "lumina-deep",
            // The alias Terminus's tool-selecting sub-agent already requests.
            Role::Router => "lumina-fast",
            // Engram memory. Unconfigured on a fleet with no embedding alias ⇒
            // the role degrades to `unresolved` rather than pinning a guess.
            Role::Embedding => "lumina-embed",
        }
    }
}

/// What a role slot is doing right now. Every non-`Warm` value is a DEGRADED but
/// entirely serviceable state — the turn still runs, just colder.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RoleState {
    /// Held resident (warmed with a long `keep_alive` + eviction-exempt).
    Warm,
    /// Deliberately not held: released by a mode swap, or never warmed yet.
    Released,
    /// The role's alias key resolves to no target (alias unconfigured).
    Unresolved,
    /// The alias resolved, but the model is unknown to the registry — i.e. never
    /// pulled. Guarded explicitly: we never exempt a model that does not exist.
    Missing,
    /// The warm call did not succeed (Ollama unreachable/refused). Best-effort.
    WarmFailed,
    /// Skipped this pass because VRAM could not hold it at this priority.
    DroppedVram,
    /// The whole resident set is switched off (`CHORD_RESIDENT_SET_ENABLED=0`).
    Disabled,
}

impl RoleState {
    /// Whether this state means the role's model is actually being held.
    pub fn is_held(&self) -> bool {
        matches!(self, RoleState::Warm)
    }
}

/// One role's observable status, as reported on the control API.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RoleStatus {
    pub role: Role,
    /// The alias key this role resolves through (never a hard-wired model name).
    pub alias: String,
    /// The concrete model the alias currently points at, when resolvable.
    pub model: Option<String>,
    pub state: RoleState,
    /// Convenience mirror of `state == warm` for scrapers.
    pub warm: bool,
    /// Registry `last_requested` (epoch seconds) for the resolved model.
    pub last_used: Option<i64>,
    /// Registry size, when known — what the VRAM budget is computed against.
    pub size_gb: Option<f64>,
}

/// The whole resident set's observable state.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ResidentSetStatus {
    pub enabled: bool,
    /// True while the set is held (assistant mode); false after a mode-swap
    /// release, until a re-warm.
    pub active: bool,
    /// TRTR-07b: the lifecycle generation. Bumped by every release; a warm pass
    /// that completes against a stale generation discards itself. Exposed so a
    /// release/re-warm cycle is observable rather than inferred.
    pub generation: u64,
    pub keep_alive: String,
    pub roles: Vec<RoleStatus>,
    /// The registry residency exemption currently in force (deduped model names).
    pub exempt: Vec<String>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

/// Env-driven config. No infrastructure literals: the Ollama base comes from the
/// existing `OLLAMA_URL` helper, everything else is a tunable with a documented
/// default.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidentSetConfig {
    /// `CHORD_RESIDENT_SET_ENABLED` (default on). Off ⇒ every role `disabled`.
    pub enabled: bool,
    /// role → ALIAS KEY, in priority order.
    pub aliases: Vec<(Role, String)>,
    /// `CHORD_RESIDENT_KEEP_ALIVE` — Ollama keep_alive for a held model
    /// (default `24h`, i.e. "resident until a mode swap releases it", rather
    /// than the ~30m default that was silently letting the set expire).
    pub keep_alive: String,
    /// `CHORD_RESIDENT_REWARM_DEBOUNCE_SECS` (default 30) — a rapid
    /// acquire/release/acquire lease cycle must not thrash-warm.
    pub rewarm_debounce: Duration,
    /// `CHORD_RESIDENT_WARM_TIMEOUT_SECS` (default 300) — per-model warm bound.
    pub warm_timeout: Duration,
    /// `CHORD_RESIDENT_REFRESH_SECS` (default 300) — background reconcile tick
    /// (catches an alias repoint and a keep_alive that has drifted cold).
    pub refresh: Duration,
    /// `CHORD_RESIDENT_REASSERT_SECS` (default 3600) — how old a role's last
    /// successful warm must be before an ordinary reconcile tick re-issues the
    /// keep-alive call for it. Shorter than `keep_alive` (so drift is corrected
    /// well before expiry) but MUCH longer than `refresh`, so the periodic tick
    /// is a cheap no-op instead of a warm storm every `refresh` seconds.
    pub reassert: Duration,
}

fn env_flag(key: &str, default: bool) -> bool {
    match std::env::var(key) {
        Ok(v) => !matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "false" | "no" | "off"),
        Err(_) => default,
    }
}

fn env_string(key: &str, default: &str) -> String {
    std::env::var(key)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| default.to_string())
}

fn env_secs(key: &str, default: u64) -> Duration {
    let v = std::env::var(key)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(default);
    Duration::from_secs(v)
}

impl Default for ResidentSetConfig {
    fn default() -> Self {
        ResidentSetConfig {
            enabled: true,
            aliases: Role::PRIORITY
                .iter()
                .map(|r| (*r, r.default_alias().to_string()))
                .collect(),
            keep_alive: "24h".to_string(),
            rewarm_debounce: Duration::from_secs(30),
            warm_timeout: Duration::from_secs(300),
            refresh: Duration::from_secs(300),
            reassert: Duration::from_secs(3600),
        }
    }
}

impl ResidentSetConfig {
    pub fn from_env() -> Self {
        ResidentSetConfig {
            enabled: env_flag("CHORD_RESIDENT_SET_ENABLED", true),
            aliases: Role::PRIORITY
                .iter()
                .map(|r| (*r, env_string(r.alias_env(), r.default_alias())))
                .collect(),
            keep_alive: env_string("CHORD_RESIDENT_KEEP_ALIVE", "24h"),
            rewarm_debounce: env_secs("CHORD_RESIDENT_REWARM_DEBOUNCE_SECS", 30),
            warm_timeout: env_secs("CHORD_RESIDENT_WARM_TIMEOUT_SECS", 300),
            refresh: env_secs("CHORD_RESIDENT_REFRESH_SECS", 300),
            reassert: env_secs("CHORD_RESIDENT_REASSERT_SECS", 3600),
        }
    }

    fn alias_for(&self, role: Role) -> &str {
        self.aliases
            .iter()
            .find(|(r, _)| *r == role)
            .map(|(_, a)| a.as_str())
            .unwrap_or("")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pure decision helpers (unit-tested without a GPU, an Ollama, or a registry)
// ─────────────────────────────────────────────────────────────────────────────

/// Resolve an alias KEY to its current concrete target: the runtime lumina store
/// first (so a dynamic repoint wins), then the static `CHORD_MODEL_ALIASES` map.
///
/// A key present in neither map returns `None` — deliberately NOT the key itself.
/// Falling back to the key would let a typo'd or unconfigured role silently
/// become a hard-wired "model name", which is exactly what this design forbids.
pub fn resolve_alias(
    alias: &str,
    dynamic: &LuminaAliasStore,
    statics: &HashMap<String, String>,
) -> Option<String> {
    if alias.is_empty() {
        return None;
    }
    dynamic
        .resolve(alias)
        .or_else(|| statics.get(alias).cloned())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// A role's resolution outcome, before any VRAM budgeting.
#[derive(Debug, Clone, PartialEq)]
pub struct Resolved {
    pub role: Role,
    pub alias: String,
    /// `None` ⇒ the alias resolves to nothing.
    pub model: Option<String>,
    /// `None` ⇒ the model is unknown to the registry (never pulled).
    pub size_gb: Option<f64>,
    /// Whether the registry knows this model at all.
    pub present: bool,
    pub last_used: Option<i64>,
}

/// What the warm pass decided for one role.
#[derive(Debug, Clone, PartialEq)]
pub enum WarmDecision {
    /// Warm this model (first role to claim it).
    Warm { model: String, size_gb: Option<f64> },
    /// Already held and not yet due for a keep-alive re-assert: keep it, issue
    /// NO warm request. This is what makes an ordinary reconcile tick free.
    Retain { model: String },
    /// Another, higher-priority role already claims the SAME model — it is held
    /// once and shared. Not a degradation.
    Shared { model: String },
    /// Skipped, with the state to record.
    Skip(RoleState),
}

/// Plan a warm pass in priority order against a VRAM budget.
///
/// - `free_gb == None` (counter unreadable): fail SOFT and attempt every role.
///   Refusing to warm on an unreadable counter would make an ordinary sensor gap
///   permanently disable residency; a warm that does not fit fails on its own and
///   is reported, which is the cheaper wrong answer here (unlike an admission
///   decision, an over-optimistic warm cannot corrupt anything).
/// - `free_gb == Some(g)`: spend the budget highest-priority-first. A role that
///   does not fit is `DroppedVram` and we CONTINUE to lower-priority roles — a
///   small embedding model should still land when a big personality model didn't.
/// - `resident`: models the set is ALREADY holding. They are already occupying
///   VRAM, so the free-VRAM reading the caller passed in has ALREADY been reduced
///   by them — charging them again would make a steady state look like a
///   shortfall and drop residency on an ordinary tick. They cost ZERO.
/// - `retainable`: held models not yet due for a keep-alive re-assert. These are
///   `Retain`ed: kept, exempt, and NOT re-warmed — no redundant warm storm.
/// - A model already claimed by a higher-priority role is `Shared` and costs
///   nothing (the three roles may legitimately point at two distinct models).
///
/// The plan is computed once per pass and never retried inside the pass — that is
/// the anti-thrash property.
pub fn plan_warm(
    resolved: &[Resolved],
    free_gb: Option<f64>,
    resident: &HashSet<String>,
    retainable: &HashSet<String>,
) -> Vec<(Role, WarmDecision)> {
    let mut budget = free_gb;
    let mut claimed: Vec<String> = Vec::new();
    let mut out = Vec::with_capacity(resolved.len());

    for r in resolved {
        let decision = match (&r.model, r.present) {
            (None, _) => WarmDecision::Skip(RoleState::Unresolved),
            (Some(_), false) => WarmDecision::Skip(RoleState::Missing),
            (Some(model), true) => {
                if claimed.iter().any(|c| c == model) {
                    WarmDecision::Shared {
                        model: model.clone(),
                    }
                } else if retainable.contains(model) {
                    // Already warm and not due for a re-assert: free, no I/O.
                    claimed.push(model.clone());
                    WarmDecision::Retain {
                        model: model.clone(),
                    }
                } else {
                    // An already-resident model is already inside the reported
                    // free-VRAM figure — it costs nothing to keep holding it.
                    let cost = if resident.contains(model) {
                        0.0
                    } else {
                        r.size_gb.unwrap_or(0.0)
                    };
                    match budget {
                        Some(free) if cost > free => WarmDecision::Skip(RoleState::DroppedVram),
                        _ => {
                            if let Some(free) = budget.as_mut() {
                                *free = (*free - cost).max(0.0);
                            }
                            claimed.push(model.clone());
                            WarmDecision::Warm {
                                model: model.clone(),
                                size_gb: r.size_gb,
                            }
                        }
                    }
                }
            }
        };
        out.push((r.role, decision));
    }
    out
}

/// Debounce a re-warm: `true` when enough time has passed since the last warm
/// pass. A rapid lease acquire → release → acquire must not re-warm three times.
pub fn should_rewarm(last_warm: Option<Instant>, now: Instant, debounce: Duration) -> bool {
    match last_warm {
        None => true,
        Some(t) => now.saturating_duration_since(t) >= debounce,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The side-effect seam
// ─────────────────────────────────────────────────────────────────────────────

/// Everything a warm/release pass does to the outside world, behind one trait.
///
/// This exists so the LIFECYCLE (the generation guard, the in-flight coalescing,
/// the commit) is testable deterministically — a test injects an env whose warm
/// call blocks on a barrier and forces the exact interleaving a race needs,
/// instead of racing a real Ollama and a real GPU and hoping.
#[async_trait]
pub trait ResidentEnv: Send + Sync {
    /// Resolve every role against the CURRENT aliases + registry, verifying
    /// presence (the never-pulled guard).
    async fn resolve(&self, aliases: &[(Role, String)]) -> Vec<Resolved>;
    /// Free VRAM in GB; `None` when the counter is unreadable (fail soft).
    fn free_vram_gb(&self) -> Option<f64>;
    /// Load `model` and hold it for `keep_alive`. Errors are genericized (S77).
    async fn warm_one(&self, role: Role, model: &str, keep_alive: &str) -> Result<(), String>;
    /// Replace the registry residency exemption with exactly `models`.
    async fn set_exempt(&self, models: &[String]);
    /// Drop the registry residency exemption entirely.
    async fn clear_exempt(&self);
}

/// The production [`ResidentEnv`]: the real alias store, registry, VRAM counter,
/// and Ollama.
pub struct AppStateEnv<'a> {
    state: &'a Arc<crate::routes::AppState>,
    warm_timeout: Duration,
}

impl<'a> AppStateEnv<'a> {
    pub fn new(state: &'a Arc<crate::routes::AppState>, warm_timeout: Duration) -> Self {
        AppStateEnv {
            state,
            warm_timeout,
        }
    }
}

#[async_trait]
impl<'a> ResidentEnv for AppStateEnv<'a> {
    async fn resolve(&self, aliases: &[(Role, String)]) -> Vec<Resolved> {
        let state = self.state;
        let targets: Vec<(Role, String, Option<String>)> = aliases
            .iter()
            .map(|(role, alias)| {
                let model = resolve_alias(alias, &state.lumina_aliases, &state.model_aliases);
                (*role, alias.clone(), model)
            })
            .collect();

        // ONE brief registry lock for the whole resolution (never held across an
        // await that does I/O).
        let reg = state.model_registry.lock().await;
        let mut out = Vec::with_capacity(targets.len());
        for (role, alias, model) in targets {
            let (present, size_gb, last_used) = match model.as_deref().and_then(|m| reg.get(m)) {
                Some(rec) => (
                    true,
                    Some(rec.size_bytes as f64 / 1_073_741_824.0),
                    rec.last_requested,
                ),
                None => (false, None, None),
            };
            out.push(Resolved {
                role,
                alias,
                model,
                size_gb,
                present,
                last_used,
            });
        }
        out
    }

    fn free_vram_gb(&self) -> Option<f64> {
        crate::config::read_free_vram_gb()
    }

    /// Issue the actual Ollama warm call for one role.
    ///
    /// Role-shaped: an embedding model is loaded through `/api/embed` (it cannot
    /// serve `/api/generate`), everything else through `/api/generate` with no
    /// prompt — Ollama's documented "load this model and hold it for keep_alive"
    /// call. Never fatal: every failure is a genericized `Err` string (S77 — no
    /// infrastructure in the message) the caller logs and degrades on.
    async fn warm_one(&self, role: Role, model: &str, keep_alive: &str) -> Result<(), String> {
        let Some(base) = crate::gpu_exclusive::ollama_base_from_env() else {
            return Err("ollama base not configured".to_string());
        };
        let base = base.trim_end_matches('/');
        let (url, body) = match role {
            Role::Embedding => (
                format!("{base}/api/embed"),
                serde_json::json!({
                    "model": model,
                    "input": "",
                    "keep_alive": keep_alive,
                }),
            ),
            _ => (
                format!("{base}/api/generate"),
                serde_json::json!({
                    "model": model,
                    "keep_alive": keep_alive,
                }),
            ),
        };
        match self
            .state
            .http_client
            .post(&url)
            .json(&body)
            .timeout(self.warm_timeout)
            .send()
            .await
        {
            Ok(r) if r.status().is_success() => Ok(()),
            Ok(r) => Err(format!("warm rejected with status {}", r.status().as_u16())),
            Err(_) => Err("warm request failed".to_string()),
        }
    }

    async fn set_exempt(&self, models: &[String]) {
        let mut reg = self.state.model_registry.lock().await;
        reg.set_residency_exempt(models.iter().cloned());
    }

    async fn clear_exempt(&self) {
        let mut reg = self.state.model_registry.lock().await;
        reg.clear_residency_exempt();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The manager
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Slot {
    alias: String,
    model: Option<String>,
    state: RoleState,
    last_used: Option<i64>,
    size_gb: Option<f64>,
    /// When this slot was last successfully warmed — drives the re-assert clock.
    warmed_at: Option<Instant>,
}

struct Inner {
    /// True while the set is held. `false` after a release, so a second release
    /// is a cheap no-op (idempotence) and the background reconcile stays quiet.
    active: bool,
    /// TRTR-07b lifecycle guard. Bumped by EVERY release. A warm pass captures it
    /// on admission and discards itself at commit if it has moved.
    generation: u64,
    slots: Vec<(Role, Slot)>,
    last_warm: Option<Instant>,
    /// Set for the duration of a warm pass. A concurrent caller subscribes to it
    /// and awaits that pass's report rather than duplicating the warm requests.
    in_flight: Option<broadcast::Sender<WarmReport>>,
}

/// Outcome of a warm pass (logged + returned for observability).
#[derive(Debug, Clone, PartialEq, Serialize, Default)]
pub struct WarmReport {
    pub warmed: usize,
    pub shared: usize,
    /// Held models kept without a re-warm (not yet due for a re-assert).
    pub retained: usize,
    pub dropped: usize,
    pub failed: usize,
    pub skipped: usize,
    /// Roles whose working residency was PRESERVED against a would-be downgrade
    /// (a failed re-assert or a VRAM shortfall on an already-held model).
    pub preserved: usize,
    /// `false` when the pass was debounced away or the set is disabled.
    pub changed: bool,
    /// TRTR-07b: the pass completed but a release had bumped the generation, so
    /// every result was thrown away and nothing was re-pinned. Release wins.
    pub discarded: bool,
    /// This caller joined an already-in-flight pass instead of duplicating it.
    pub coalesced: bool,
}

/// Outcome of a release (logged + returned for observability).
#[derive(Debug, Clone, PartialEq, Serialize, Default)]
pub struct ReleaseReport {
    /// Distinct models whose residency exemption was dropped.
    pub released: usize,
    /// `false` when already released (idempotent no-op).
    pub changed: bool,
    /// The lifecycle generation AFTER this release. Every warm pass in flight at
    /// this point is now stale and will discard.
    pub generation: u64,
}

/// What one role's pass produced, before the commit applies downgrade protection.
struct PassOutcome {
    role: Role,
    alias: String,
    model: Option<String>,
    size_gb: Option<f64>,
    last_used: Option<i64>,
    state: RoleState,
    /// `Some` ⇒ this pass warmed it now; `None` ⇒ carry the previous timestamp.
    warmed_at: Option<Instant>,
}

/// Snapshot of what the set already holds, taken atomically on admission.
struct Snapshot {
    resident: HashSet<String>,
    retainable: HashSet<String>,
}

enum Admission {
    Proceed {
        generation: u64,
        snapshot: Snapshot,
    },
    Join(broadcast::Receiver<WarmReport>),
    Skip(WarmReport),
}

/// The assistant-mode resident set. A process-global singleton, following the
/// same pattern as `GPU_EXCLUSIVE` / the managed DiffusionGemma daemon — so it is
/// reachable from the idle-mode path without threading a new field through every
/// `AppState` construction site.
pub struct ResidentSet {
    cfg: ResidentSetConfig,
    inner: Mutex<Inner>,
}

/// The process-global resident set.
pub fn global() -> &'static ResidentSet {
    static GLOBAL: once_cell::sync::Lazy<ResidentSet> =
        once_cell::sync::Lazy::new(|| ResidentSet::new(ResidentSetConfig::from_env()));
    &GLOBAL
}

impl ResidentSet {
    pub fn new(cfg: ResidentSetConfig) -> Self {
        let slots = cfg
            .aliases
            .iter()
            .map(|(role, alias)| {
                (
                    *role,
                    Slot {
                        alias: alias.clone(),
                        model: None,
                        state: if cfg.enabled {
                            RoleState::Released
                        } else {
                            RoleState::Disabled
                        },
                        last_used: None,
                        size_gb: None,
                        warmed_at: None,
                    },
                )
            })
            .collect();
        ResidentSet {
            cfg,
            inner: Mutex::new(Inner {
                active: false,
                generation: 0,
                slots,
                last_warm: None,
                in_flight: None,
            }),
        }
    }

    pub fn config(&self) -> &ResidentSetConfig {
        &self.cfg
    }

    /// Current observable state (control API `GET /admin/resident-set`).
    pub async fn status(&self) -> ResidentSetStatus {
        let inner = self.inner.lock().await;
        let roles = inner
            .slots
            .iter()
            .map(|(role, s)| RoleStatus {
                role: *role,
                alias: s.alias.clone(),
                model: s.model.clone(),
                state: s.state.clone(),
                warm: s.state.is_held(),
                last_used: s.last_used,
                size_gb: s.size_gb,
            })
            .collect();
        let mut exempt: Vec<String> = inner
            .slots
            .iter()
            .filter(|(_, s)| s.state.is_held())
            .filter_map(|(_, s)| s.model.clone())
            .collect();
        exempt.sort();
        exempt.dedup();
        ResidentSetStatus {
            enabled: self.cfg.enabled,
            active: inner.active,
            generation: inner.generation,
            keep_alive: self.cfg.keep_alive.clone(),
            roles,
            exempt,
        }
    }

    // ── Public entry points (production env) ────────────────────────────────

    /// Warm every role and hold it resident. Best-effort and idempotent; safe to
    /// call at startup, after an `activate`, and on the background tick.
    ///
    /// `force` bypasses the re-warm debounce (used for the startup pass).
    pub async fn warm(
        &self,
        state: &Arc<crate::routes::AppState>,
        trigger: &str,
        force: bool,
    ) -> WarmReport {
        let env = AppStateEnv::new(state, self.cfg.warm_timeout);
        self.warm_with(&env, trigger, force, None).await
    }

    /// RELEASE the whole set for a mode swap (Harmony BLD-11 idle lease, MINT
    /// `enter_idle`). See [`ResidentSet::release_with`].
    pub async fn release(
        &self,
        state: &Arc<crate::routes::AppState>,
        reason: &str,
    ) -> ReleaseReport {
        let env = AppStateEnv::new(state, self.cfg.warm_timeout);
        self.release_with(&env, reason).await
    }

    /// Re-warm after a mode swap ends (idle lease released / sweep finished).
    /// Debounced, so a rapid acquire/release cycle does not thrash-warm.
    pub async fn rewarm(&self, state: &Arc<crate::routes::AppState>, trigger: &str) -> WarmReport {
        self.warm(state, trigger, false).await
    }

    /// Background reconcile. See [`ResidentSet::reconcile_with`].
    pub async fn reconcile(&self, state: &Arc<crate::routes::AppState>) -> WarmReport {
        let env = AppStateEnv::new(state, self.cfg.warm_timeout);
        self.reconcile_with(&env).await
    }

    // ── The lifecycle, against any env ──────────────────────────────────────

    /// Admit (or refuse) a warm pass. Everything here happens under ONE `inner`
    /// lock acquisition, so the generation capture, the debounce decision, the
    /// in-flight registration, and the resident snapshot are atomic with respect
    /// to a release and to another warm.
    async fn admit(&self, trigger: &str, force: bool, expect_gen: Option<u64>) -> Admission {
        let mut inner = self.inner.lock().await;

        // A caller that observed state BEFORE deciding to warm (reconcile) tells
        // us which generation it observed; if a release has landed since, its
        // premise is void and the pass never starts.
        if let Some(g) = expect_gen {
            if inner.generation != g {
                info!(
                    trigger,
                    observed_generation = g,
                    current_generation = inner.generation,
                    "resident-set: warm pass abandoned before it started — a mode-swap release landed first"
                );
                return Admission::Skip(WarmReport {
                    discarded: true,
                    ..Default::default()
                });
            }
        }

        // Coalesce: one pass at a time, everyone else awaits its report.
        if let Some(tx) = inner.in_flight.as_ref() {
            return Admission::Join(tx.subscribe());
        }

        if !force && !should_rewarm(inner.last_warm, Instant::now(), self.cfg.rewarm_debounce) {
            info!(
                trigger,
                debounce_secs = self.cfg.rewarm_debounce.as_secs(),
                "resident-set: re-warm debounced (rapid mode-swap cycle)"
            );
            return Admission::Skip(WarmReport::default());
        }

        let now = Instant::now();
        let mut resident = HashSet::new();
        let mut retainable = HashSet::new();
        for (_, s) in inner.slots.iter() {
            if !s.state.is_held() {
                continue;
            }
            let Some(model) = s.model.clone() else { continue };
            let fresh = s
                .warmed_at
                .map(|t| now.saturating_duration_since(t) < self.cfg.reassert)
                .unwrap_or(false);
            if fresh {
                retainable.insert(model.clone());
            }
            resident.insert(model);
        }

        let (tx, _rx) = broadcast::channel(8);
        inner.in_flight = Some(tx);
        Admission::Proceed {
            generation: inner.generation,
            snapshot: Snapshot {
                resident,
                retainable,
            },
        }
    }

    /// The whole warm lifecycle against an injected env.
    ///
    /// `expect_gen`: the lifecycle generation the CALLER observed when it decided
    /// this pass was warranted (reconcile passes the generation it saw `active`
    /// under). A mismatch cancels the pass before any I/O.
    pub(crate) async fn warm_with(
        &self,
        env: &dyn ResidentEnv,
        trigger: &str,
        force: bool,
        expect_gen: Option<u64>,
    ) -> WarmReport {
        if !self.cfg.enabled {
            return WarmReport::default();
        }
        let (generation, snapshot) = match self.admit(trigger, force, expect_gen).await {
            Admission::Skip(r) => return r,
            Admission::Join(mut rx) => {
                // Join the in-flight pass: no duplicate warm requests, and we
                // report ITS outcome. `changed` is false because THIS caller
                // changed nothing.
                let mut report = rx.recv().await.unwrap_or_default();
                report.changed = false;
                report.coalesced = true;
                info!(
                    trigger,
                    "resident-set: warm coalesced into the in-flight pass (no duplicate warm requests)"
                );
                return report;
            }
            Admission::Proceed {
                generation,
                snapshot,
            } => (generation, snapshot),
        };

        self.run_pass(env, trigger, generation, snapshot).await
    }

    /// The slow half (resolution + warm I/O, NO lock held) followed by the
    /// atomic commit.
    async fn run_pass(
        &self,
        env: &dyn ResidentEnv,
        trigger: &str,
        generation: u64,
        snapshot: Snapshot,
    ) -> WarmReport {
        let resolved = env.resolve(&self.cfg.aliases).await;
        // Fail-soft VRAM read: `None` (unreadable counter) attempts every role.
        let free_gb = env.free_vram_gb();
        let plan = plan_warm(
            &resolved,
            free_gb,
            &snapshot.resident,
            &snapshot.retainable,
        );

        let mut report = WarmReport {
            changed: true,
            ..Default::default()
        };
        let mut outcomes: Vec<PassOutcome> = Vec::with_capacity(plan.len());

        for (r, (role, decision)) in resolved.iter().zip(plan.into_iter()) {
            debug_assert_eq!(r.role, role);
            let (state_for_slot, warmed_at) = match decision {
                WarmDecision::Warm { model, size_gb } => {
                    match env.warm_one(role, &model, &self.cfg.keep_alive).await {
                        Ok(()) => {
                            report.warmed += 1;
                            info!(
                                role = role.id(),
                                alias = %r.alias,
                                model = %model,
                                size_gb = size_gb.unwrap_or(0.0),
                                keep_alive = %self.cfg.keep_alive,
                                "resident-set: role held resident"
                            );
                            (RoleState::Warm, Some(Instant::now()))
                        }
                        Err(reason) => {
                            report.failed += 1;
                            warn!(
                                role = role.id(),
                                model = %model,
                                reason = %reason,
                                "resident-set: warm failed — continuing DEGRADED (a cold turn is slower, not broken)"
                            );
                            (RoleState::WarmFailed, None)
                        }
                    }
                }
                WarmDecision::Retain { model } => {
                    report.retained += 1;
                    info!(
                        role = role.id(),
                        model = %model,
                        "resident-set: role already resident and not due for a re-assert — retained without a warm request"
                    );
                    (RoleState::Warm, None)
                }
                WarmDecision::Shared { model } => {
                    report.shared += 1;
                    info!(
                        role = role.id(),
                        model = %model,
                        "resident-set: role shares an already-held model"
                    );
                    (RoleState::Warm, None)
                }
                WarmDecision::Skip(st) => {
                    match st {
                        RoleState::DroppedVram => {
                            report.dropped += 1;
                            warn!(
                                role = role.id(),
                                model = r.model.as_deref().unwrap_or(""),
                                needed_gb = r.size_gb.unwrap_or(0.0),
                                free_gb = free_gb.unwrap_or(0.0),
                                "resident-set: VRAM shortfall — DROPPING this role by priority (personality > router > embedding)"
                            );
                        }
                        RoleState::Missing => {
                            report.skipped += 1;
                            warn!(
                                role = role.id(),
                                alias = %r.alias,
                                model = r.model.as_deref().unwrap_or(""),
                                "resident-set: alias target is unknown to the registry (never pulled) — NOT exempting a model that does not exist"
                            );
                        }
                        _ => {
                            report.skipped += 1;
                            info!(
                                role = role.id(),
                                alias = %r.alias,
                                "resident-set: role alias resolves to no target — running degraded"
                            );
                        }
                    }
                    (st, None)
                }
            };
            outcomes.push(PassOutcome {
                role,
                alias: r.alias.clone(),
                model: r.model.clone(),
                size_gb: r.size_gb,
                last_used: r.last_used,
                state: state_for_slot,
                warmed_at,
            });
        }

        self.commit(env, trigger, generation, outcomes, report).await
    }

    /// Commit a completed pass — or throw it away because a release won.
    ///
    /// Held under ONE `inner` lock acquisition, and the registry mutation happens
    /// INSIDE it (lock order `inner` → registry, the same order `release_with`
    /// uses) so a release can never interleave between the generation check and
    /// the exemption install.
    async fn commit(
        &self,
        env: &dyn ResidentEnv,
        trigger: &str,
        generation: u64,
        outcomes: Vec<PassOutcome>,
        mut report: WarmReport,
    ) -> WarmReport {
        let mut inner = self.inner.lock().await;

        if inner.generation != generation {
            // RELEASE WON. Install nothing, touch nothing: the idle path has
            // already dropped the exemption and begun reclaiming this VRAM, and
            // re-pinning here is exactly the bug this guard exists to prevent.
            report = WarmReport {
                discarded: true,
                changed: false,
                ..report
            };
            warn!(
                trigger,
                pass_generation = generation,
                current_generation = inner.generation,
                "resident-set: warm pass DISCARDED — a mode-swap release landed mid-warm; the GPU stays released"
            );
            if let Some(tx) = inner.in_flight.take() {
                let _ = tx.send(report.clone());
            }
            return report;
        }

        let mut new_slots: Vec<(Role, Slot)> = Vec::with_capacity(outcomes.len());
        let mut held: Vec<String> = Vec::new();

        for out in outcomes {
            let prev = inner
                .slots
                .iter()
                .find(|(role, _)| *role == out.role)
                .map(|(_, s)| s.clone());

            let attempted = out.state.clone();
            let mut state = out.state;
            let mut warmed_at = out.warmed_at.or_else(|| prev.as_ref().and_then(|p| p.warmed_at));

            // A transient failure (or a momentary shortfall) must NOT downgrade a
            // slot that is currently warm and valid on the SAME model: the model
            // is still loaded, and dropping the pin only feeds it to the eviction
            // sweep. Keeping working residency beats replacing it with a failure.
            if matches!(state, RoleState::WarmFailed | RoleState::DroppedVram) {
                if let (Some(p), Some(model)) = (prev.as_ref(), out.model.as_ref()) {
                    if p.state.is_held() && p.model.as_deref() == Some(model.as_str()) {
                        warn!(
                            role = out.role.id(),
                            model = %model,
                            "resident-set: refresh did not succeed — PRESERVING the existing valid residency rather than downgrading it"
                        );
                        state = RoleState::Warm;
                        warmed_at = p.warmed_at;
                        report.preserved += 1;
                        match attempted {
                            RoleState::WarmFailed => report.failed = report.failed.saturating_sub(1),
                            _ => report.dropped = report.dropped.saturating_sub(1),
                        }
                    }
                }
            }

            if state.is_held() {
                if let Some(m) = out.model.clone() {
                    held.push(m);
                }
            }

            new_slots.push((
                out.role,
                Slot {
                    alias: out.alias,
                    model: out.model,
                    state,
                    last_used: out.last_used,
                    size_gb: out.size_gb,
                    warmed_at,
                },
            ));
        }

        // Apply the eviction exemption for exactly the models we actually hold.
        // Wholesale replacement, so a repointed alias leaves no stale pin behind.
        held.sort();
        held.dedup();
        env.set_exempt(&held).await;

        inner.slots = new_slots;
        inner.active = true;
        inner.last_warm = Some(Instant::now());
        if let Some(tx) = inner.in_flight.take() {
            let _ = tx.send(report.clone());
        }
        info!(
            trigger,
            warmed = report.warmed,
            shared = report.shared,
            retained = report.retained,
            dropped = report.dropped,
            failed = report.failed,
            skipped = report.skipped,
            preserved = report.preserved,
            held = held.len(),
            "resident-set: warm pass complete"
        );
        report
    }

    /// RELEASE the whole set for a mode swap: drop the eviction exemption so every
    /// held model becomes immediately reclaimable, and mark the roles released.
    ///
    /// Idempotent — a second release reports `changed: false`. It ALWAYS bumps the
    /// lifecycle generation though, even when the set was already inactive: a warm
    /// pass may be in flight at this instant (the very first startup warm, say),
    /// and the bump is what makes that pass discard itself instead of pinning
    /// models onto a host that is being handed to Harmony/MINT.
    ///
    /// This does NOT itself unload VRAM: the idle path's existing
    /// `gpu_exclusive::evict_resident_models` step does that, and it runs right
    /// after us. Calling release first is what lets that step actually reclaim
    /// these models instead of stepping around a pin.
    pub(crate) async fn release_with(&self, env: &dyn ResidentEnv, reason: &str) -> ReleaseReport {
        let mut inner = self.inner.lock().await;
        let was_active = inner.active;
        inner.generation = inner.generation.wrapping_add(1);
        let generation = inner.generation;

        let mut released: Vec<String> = inner
            .slots
            .iter()
            .filter(|(_, s)| s.state.is_held())
            .filter_map(|(_, s)| s.model.clone())
            .collect();
        released.sort();
        released.dedup();
        for (_, s) in inner.slots.iter_mut() {
            if s.state.is_held() {
                s.state = RoleState::Released;
            }
            s.warmed_at = None;
        }
        inner.active = false;

        // Inside the `inner` lock, same order the commit uses — so a warm pass can
        // neither observe a half-applied release nor slip its exemption in after
        // this clear.
        env.clear_exempt().await;
        drop(inner);

        info!(
            reason,
            released = released.len(),
            generation,
            "resident-set: RELEASED for a mode swap — models are immediately reclaimable"
        );
        ReleaseReport {
            released: released.len(),
            changed: was_active,
            generation,
        }
    }

    /// Background reconcile: catch a dynamic alias repoint and a keep_alive that
    /// has drifted. A no-op while released (a mode swap owns the host then).
    ///
    /// On a repoint the NEW target is warmed first and only then is the OLD one
    /// dropped from the exemption — never the reverse, so the role is never
    /// momentarily unbacked. (The commit applies the exemption wholesale, which
    /// performs the drop; the ordering falls out of doing the warm first.)
    ///
    /// The generation observed alongside `active` is threaded into the pass, so a
    /// release landing between the observation and the pass cancels it.
    pub(crate) async fn reconcile_with(&self, env: &dyn ResidentEnv) -> WarmReport {
        let (active, generation) = {
            let inner = self.inner.lock().await;
            (inner.active, inner.generation)
        };
        if !active {
            return WarmReport::default();
        }
        let resolved = env.resolve(&self.cfg.aliases).await;
        let repointed: Vec<(Role, Option<String>, Option<String>)> = {
            let inner = self.inner.lock().await;
            resolved
                .iter()
                .filter_map(|r| {
                    let old = inner
                        .slots
                        .iter()
                        .find(|(role, _)| *role == r.role)
                        .and_then(|(_, s)| s.model.clone());
                    (old != r.model).then_some((r.role, old, r.model.clone()))
                })
                .collect()
        };
        for (role, old, new) in &repointed {
            info!(
                role = role.id(),
                previous = old.as_deref().unwrap_or(""),
                current = new.as_deref().unwrap_or(""),
                "resident-set: alias repointed mid-residency — warming the new target, then dropping the old"
            );
        }
        // Force past the debounce ONLY for a genuine repoint; a plain periodic
        // tick stays debounced so it can never become a warm loop.
        self.warm_with(env, "reconcile", !repointed.is_empty(), Some(generation))
            .await
    }
}

/// Background loop: periodically reconcile the resident set so an alias repoint
/// or a drifted keep_alive is corrected without a restart. Best-effort and
/// no-op while released — it never contends with a mode swap.
pub async fn reconcile_loop(state: Arc<crate::routes::AppState>, interval: Duration) {
    info!(
        interval_secs = interval.as_secs(),
        "resident-set reconcile loop started"
    );
    loop {
        tokio::time::sleep(interval).await;
        let _ = global().reconcile(&state).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex as StdMutex;
    use tokio::sync::{mpsc, Semaphore};

    fn resolved(role: Role, model: Option<&str>, size: Option<f64>, present: bool) -> Resolved {
        Resolved {
            role,
            alias: role.default_alias().to_string(),
            model: model.map(|s| s.to_string()),
            size_gb: size,
            present,
            last_used: None,
        }
    }

    fn no_set() -> HashSet<String> {
        HashSet::new()
    }

    // ── Role resolution goes through aliases, never a hard-wired name ────────

    #[test]
    fn alias_resolves_dynamic_first_then_static() {
        let mut statics = HashMap::new();
        statics.insert("lumina-fast".to_string(), "static-model".to_string());
        statics.insert("lumina-embed".to_string(), "embed-model".to_string());
        let dynamic = LuminaAliasStore::from_static(&statics);
        dynamic.set("lumina-fast", "dynamic-model".to_string());

        assert_eq!(
            resolve_alias("lumina-fast", &dynamic, &statics).as_deref(),
            Some("dynamic-model"),
            "a runtime repoint must win over the static map"
        );
        assert_eq!(
            resolve_alias("lumina-embed", &dynamic, &statics).as_deref(),
            Some("embed-model"),
            "a non-lumina-tier alias still resolves via the static map"
        );
    }

    #[test]
    fn unknown_alias_never_falls_back_to_the_key_itself() {
        let statics = HashMap::new();
        let dynamic = LuminaAliasStore::empty();
        assert_eq!(
            resolve_alias("lumina-embed", &dynamic, &statics),
            None,
            "an unconfigured alias must degrade, NOT become a hard-wired model name"
        );
        assert_eq!(resolve_alias("", &dynamic, &statics), None);
    }

    // ── The never-pulled guard ──────────────────────────────────────────────

    #[test]
    fn absent_model_is_missing_and_never_warmed() {
        let plan = plan_warm(
            &[resolved(Role::Personality, Some("never-pulled:1"), None, false)],
            Some(96.0),
            &no_set(),
            &no_set(),
        );
        assert_eq!(plan[0].1, WarmDecision::Skip(RoleState::Missing));
    }

    #[test]
    fn unresolved_alias_is_skipped_not_guessed() {
        let plan = plan_warm(
            &[resolved(Role::Embedding, None, None, false)],
            Some(96.0),
            &no_set(),
            &no_set(),
        );
        assert_eq!(plan[0].1, WarmDecision::Skip(RoleState::Unresolved));
    }

    // ── VRAM shortfall degradation order ────────────────────────────────────

    #[test]
    fn vram_shortfall_drops_lowest_priority_and_keeps_personality() {
        // 20GB free; personality 18GB fits, router 10GB does not, embedding 1GB
        // still does — a lower-priority role that FITS is not punished for a
        // bigger sibling missing out.
        let plan = plan_warm(
            &[
                resolved(Role::Personality, Some("voice:1"), Some(18.0), true),
                resolved(Role::Router, Some("router:1"), Some(10.0), true),
                resolved(Role::Embedding, Some("embed:1"), Some(1.0), true),
            ],
            Some(20.0),
            &no_set(),
            &no_set(),
        );
        assert!(matches!(plan[0].1, WarmDecision::Warm { .. }), "personality held");
        assert_eq!(plan[1].1, WarmDecision::Skip(RoleState::DroppedVram));
        assert!(matches!(plan[2].1, WarmDecision::Warm { .. }), "small role still fits");
    }

    #[test]
    fn priority_order_is_personality_then_router_then_embedding() {
        assert_eq!(
            Role::PRIORITY,
            [Role::Personality, Role::Router, Role::Embedding]
        );
        // With room for exactly one, the highest-priority role wins.
        let plan = plan_warm(
            &[
                resolved(Role::Personality, Some("voice:1"), Some(10.0), true),
                resolved(Role::Router, Some("router:1"), Some(10.0), true),
                resolved(Role::Embedding, Some("embed:1"), Some(10.0), true),
            ],
            Some(10.0),
            &no_set(),
            &no_set(),
        );
        assert!(matches!(plan[0].1, WarmDecision::Warm { .. }));
        assert_eq!(plan[1].1, WarmDecision::Skip(RoleState::DroppedVram));
        assert_eq!(plan[2].1, WarmDecision::Skip(RoleState::DroppedVram));
    }

    #[test]
    fn unreadable_vram_counter_fails_soft_and_still_warms() {
        let plan = plan_warm(
            &[
                resolved(Role::Personality, Some("voice:1"), Some(400.0), true),
                resolved(Role::Router, Some("router:1"), Some(400.0), true),
            ],
            None,
            &no_set(),
            &no_set(),
        );
        assert!(plan.iter().all(|(_, d)| matches!(d, WarmDecision::Warm { .. })));
    }

    #[test]
    fn two_roles_on_one_model_are_held_once() {
        let plan = plan_warm(
            &[
                resolved(Role::Personality, Some("shared:1"), Some(10.0), true),
                resolved(Role::Router, Some("shared:1"), Some(10.0), true),
                resolved(Role::Embedding, Some("embed:1"), Some(1.0), true),
            ],
            // Only 11GB: if the shared model were double-counted, the embedding
            // role would be wrongly dropped.
            Some(11.0),
            &no_set(),
            &no_set(),
        );
        assert!(matches!(plan[0].1, WarmDecision::Warm { .. }));
        assert_eq!(
            plan[1].1,
            WarmDecision::Shared {
                model: "shared:1".to_string()
            }
        );
        assert!(matches!(plan[2].1, WarmDecision::Warm { .. }));
    }

    /// TRTR-07b finding 4, at the pure-planner level: a model the set ALREADY
    /// holds is already inside the free-VRAM reading, so it must cost zero. With
    /// the old accounting, a steady state (residents consuming the VRAM) re-plans
    /// its own residents as `DroppedVram`.
    #[test]
    fn already_resident_models_cost_nothing_against_the_reported_free_vram() {
        let resident: HashSet<String> = ["voice:1".to_string(), "router:1".to_string()]
            .into_iter()
            .collect();
        let plan = plan_warm(
            &[
                resolved(Role::Personality, Some("voice:1"), Some(18.0), true),
                resolved(Role::Router, Some("router:1"), Some(10.0), true),
            ],
            // Only 2GB free BECAUSE those two models are loaded.
            Some(2.0),
            &resident,
            &no_set(),
        );
        assert!(
            plan.iter().all(|(_, d)| matches!(d, WarmDecision::Warm { .. })),
            "residents must not be dropped for occupying the VRAM they already occupy"
        );
    }

    /// A held model that is not yet due for a re-assert is retained with NO warm
    /// request — that is what keeps the periodic tick from becoming a warm storm.
    #[test]
    fn retainable_models_are_kept_without_a_warm_request() {
        let resident: HashSet<String> = ["voice:1".to_string()].into_iter().collect();
        let plan = plan_warm(
            &[resolved(Role::Personality, Some("voice:1"), Some(18.0), true)],
            Some(0.0),
            &resident,
            &resident,
        );
        assert_eq!(
            plan[0].1,
            WarmDecision::Retain {
                model: "voice:1".to_string()
            }
        );
    }

    // ── Debounce ────────────────────────────────────────────────────────────

    #[test]
    fn rewarm_is_debounced_after_a_recent_warm() {
        let now = Instant::now();
        let debounce = Duration::from_secs(30);
        assert!(should_rewarm(None, now, debounce), "first warm always runs");
        assert!(
            !should_rewarm(Some(now), now, debounce),
            "a rapid acquire/release cycle must not thrash-warm"
        );
        assert!(should_rewarm(
            Some(now - Duration::from_secs(31)),
            now,
            debounce
        ));
    }

    // ── Config ──────────────────────────────────────────────────────────────

    #[test]
    fn default_roles_are_alias_keys_not_model_names() {
        let cfg = ResidentSetConfig::default();
        assert!(cfg.enabled);
        for (role, alias) in &cfg.aliases {
            // An alias key, never a concrete model tag (`name:tag` / a repo path).
            assert!(
                !alias.contains(':') && !alias.contains('/'),
                "role {} must name an ALIAS, got {alias}",
                role.id()
            );
        }
        assert_eq!(cfg.alias_for(Role::Router), "lumina-fast");
        assert_eq!(cfg.alias_for(Role::Personality), "lumina-deep");
        assert_eq!(cfg.alias_for(Role::Embedding), "lumina-embed");
    }

    #[test]
    fn reassert_window_is_much_longer_than_the_reconcile_tick() {
        let cfg = ResidentSetConfig::default();
        assert!(
            cfg.reassert > cfg.refresh * 2,
            "an ordinary reconcile tick must not re-warm — otherwise residency re-warms every tick"
        );
    }

    #[tokio::test]
    async fn disabled_set_reports_every_role_disabled() {
        let cfg = ResidentSetConfig {
            enabled: false,
            ..Default::default()
        };
        let set = ResidentSet::new(cfg);
        let status = set.status().await;
        assert!(!status.enabled);
        assert!(!status.active);
        assert!(status.roles.iter().all(|r| r.state == RoleState::Disabled));
        assert!(status.roles.iter().all(|r| !r.warm));
        assert_eq!(status.roles.len(), 3);
    }

    #[tokio::test]
    async fn status_reports_every_role_before_any_warm() {
        let set = ResidentSet::new(ResidentSetConfig::default());
        let status = set.status().await;
        assert_eq!(status.roles.len(), 3);
        assert_eq!(
            status.roles.iter().map(|r| r.role).collect::<Vec<_>>(),
            Role::PRIORITY.to_vec(),
            "status is reported in priority order"
        );
        assert!(status.roles.iter().all(|r| r.state == RoleState::Released));
        assert!(status.exempt.is_empty());
    }

    #[test]
    fn status_serializes_with_stable_role_and_state_ids() {
        let s = RoleStatus {
            role: Role::Embedding,
            alias: "lumina-embed".into(),
            model: Some("embed:1".into()),
            state: RoleState::DroppedVram,
            warm: false,
            last_used: Some(1),
            size_gb: Some(1.0),
        };
        let v = serde_json::to_value(&s).unwrap();
        assert_eq!(v.get("role").unwrap(), "embedding");
        assert_eq!(v.get("state").unwrap(), "dropped-vram");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // TRTR-07b lifecycle: the injected env + the concurrency tests
    // ─────────────────────────────────────────────────────────────────────────

    /// A fully deterministic [`ResidentEnv`]. The warm call can be GATED: it
    /// announces its entry on a channel and then blocks on a semaphore until the
    /// test opens it, which lets a test place a `release` exactly in the middle
    /// of an in-flight warm with no sleeps and no probabilistic looping.
    struct FakeEnv {
        resolved: StdMutex<Vec<Resolved>>,
        free_gb: StdMutex<Option<f64>>,
        warm_calls: StdMutex<Vec<String>>,
        exempt: StdMutex<Vec<String>>,
        exempt_ops: StdMutex<Vec<String>>,
        fail: AtomicBool,
        gated: AtomicBool,
        gate: Semaphore,
        entered: mpsc::UnboundedSender<String>,
    }

    impl FakeEnv {
        fn new(
            resolved: Vec<Resolved>,
            free_gb: Option<f64>,
        ) -> (Arc<FakeEnv>, mpsc::UnboundedReceiver<String>) {
            let (tx, rx) = mpsc::unbounded_channel();
            (
                Arc::new(FakeEnv {
                    resolved: StdMutex::new(resolved),
                    free_gb: StdMutex::new(free_gb),
                    warm_calls: StdMutex::new(Vec::new()),
                    exempt: StdMutex::new(Vec::new()),
                    exempt_ops: StdMutex::new(Vec::new()),
                    fail: AtomicBool::new(false),
                    gated: AtomicBool::new(false),
                    gate: Semaphore::new(0),
                    entered: tx,
                }),
                rx,
            )
        }

        fn gate_warms(&self) {
            self.gated.store(true, Ordering::SeqCst);
        }
        fn open_gate(&self, permits: usize) {
            self.gated.store(false, Ordering::SeqCst);
            self.gate.add_permits(permits);
        }
        fn set_free(&self, v: Option<f64>) {
            *self.free_gb.lock().unwrap() = v;
        }
        fn set_fail(&self, v: bool) {
            self.fail.store(v, Ordering::SeqCst);
        }
        fn warm_calls(&self) -> Vec<String> {
            self.warm_calls.lock().unwrap().clone()
        }
        fn exempt(&self) -> Vec<String> {
            self.exempt.lock().unwrap().clone()
        }
        fn exempt_ops(&self) -> Vec<String> {
            self.exempt_ops.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl ResidentEnv for FakeEnv {
        async fn resolve(&self, _aliases: &[(Role, String)]) -> Vec<Resolved> {
            self.resolved.lock().unwrap().clone()
        }
        fn free_vram_gb(&self) -> Option<f64> {
            *self.free_gb.lock().unwrap()
        }
        async fn warm_one(&self, _role: Role, model: &str, _ka: &str) -> Result<(), String> {
            self.warm_calls.lock().unwrap().push(model.to_string());
            let _ = self.entered.send(model.to_string());
            if self.gated.load(Ordering::SeqCst) {
                self.gate.acquire().await.expect("gate").forget();
            }
            if self.fail.load(Ordering::SeqCst) {
                Err("fake warm failure".to_string())
            } else {
                Ok(())
            }
        }
        async fn set_exempt(&self, models: &[String]) {
            *self.exempt.lock().unwrap() = models.to_vec();
            self.exempt_ops
                .lock()
                .unwrap()
                .push(format!("set:{}", models.join(",")));
        }
        async fn clear_exempt(&self) {
            self.exempt.lock().unwrap().clear();
            self.exempt_ops.lock().unwrap().push("clear".to_string());
        }
    }

    /// Three distinct, present roles: 8 / 6 / 1 GB.
    fn three_roles() -> Vec<Resolved> {
        vec![
            resolved(Role::Personality, Some("voice:1"), Some(8.0), true),
            resolved(Role::Router, Some("router:1"), Some(6.0), true),
            resolved(Role::Embedding, Some("embed:1"), Some(1.0), true),
        ]
    }

    /// Debounce off (so a reconcile tick genuinely runs), long re-assert window.
    fn live_cfg() -> ResidentSetConfig {
        ResidentSetConfig {
            rewarm_debounce: Duration::ZERO,
            reassert: Duration::from_secs(3600),
            ..Default::default()
        }
    }

    /// Let every already-scheduled task run to its next await point. With the
    /// single-threaded test runtime this is deterministic — no sleeps.
    async fn settle() {
        for _ in 0..64 {
            tokio::task::yield_now().await;
        }
    }

    /// **THE INVARIANT.** A mode-swap release that lands while a warm is mid-I/O
    /// must WIN: when everything settles, no exemption is installed and the set
    /// is inactive. Otherwise the completing warm re-pins models onto a GPU the
    /// idle path has already begun handing to Harmony/MINT.
    #[tokio::test]
    async fn release_wins_over_a_warm_that_is_already_in_flight() {
        let set = Arc::new(ResidentSet::new(live_cfg()));
        let (env, mut entered) = FakeEnv::new(three_roles(), Some(96.0));
        env.gate_warms();

        let (s, e) = (set.clone(), env.clone());
        let warming = tokio::spawn(async move { s.warm_with(&*e, "startup", true, None).await });

        // The warm pass is now provably INSIDE its network I/O.
        entered.recv().await.expect("warm pass reached its first warm call");

        // The mode swap lands.
        let rel = set.release_with(&*env, "harmony-idle-lease").await;
        assert!(rel.generation > 0, "a release always bumps the generation");

        // Let the warm finish.
        env.open_gate(16);
        let report = warming.await.unwrap();

        assert!(
            report.discarded,
            "the pass must discard itself once a release has landed"
        );
        assert!(
            env.exempt().is_empty(),
            "NO exemption may be installed after a release — got {:?}",
            env.exempt()
        );
        assert!(
            !env.exempt_ops().iter().any(|op| op.starts_with("set:")),
            "the registry must never be re-pinned after a release: {:?}",
            env.exempt_ops()
        );
        let status = set.status().await;
        assert!(!status.active, "the set must remain inactive after a release");
        assert!(
            status.roles.iter().all(|r| !r.warm),
            "no role may report warm after a release"
        );
    }

    /// The same invariant on the reconcile path: a release landing between
    /// "observed active" and the pass's I/O must also win.
    #[tokio::test]
    async fn release_wins_over_an_in_flight_reconcile() {
        let set = Arc::new(ResidentSet::new(live_cfg()));
        let (env, mut entered) = FakeEnv::new(three_roles(), Some(96.0));

        // Establish residency first.
        let first = set.warm_with(&*env, "startup", true, None).await;
        assert_eq!(first.warmed, 3);
        assert_eq!(env.exempt().len(), 3);
        // Drain the entry notifications the startup pass left behind, so the
        // `recv` below provably observes the RECONCILE pass's warm call.
        while entered.try_recv().is_ok() {}

        // Now a reconcile that has to re-assert (force a repoint by pointing the
        // personality alias at a different model).
        {
            let mut r = env.resolved.lock().unwrap();
            r[0].model = Some("voice:2".to_string());
        }
        env.gate_warms();
        let (s, e) = (set.clone(), env.clone());
        let reconciling = tokio::spawn(async move { s.reconcile_with(&*e).await });
        entered.recv().await.expect("reconcile reached its warm call");

        set.release_with(&*env, "mint-sweep").await;
        env.open_gate(16);
        let report = reconciling.await.unwrap();

        assert!(report.discarded, "reconcile must discard after a release");
        assert!(env.exempt().is_empty(), "reconcile must not re-pin after a release");
        assert!(!set.status().await.active);
    }

    /// Two concurrent warms for the same roles must COALESCE into one pass — one
    /// set of warm requests, not two.
    #[tokio::test]
    async fn concurrent_warms_coalesce_instead_of_duplicating() {
        let set = Arc::new(ResidentSet::new(live_cfg()));
        let (env, mut entered) = FakeEnv::new(three_roles(), Some(96.0));
        env.gate_warms();

        let (s1, e1) = (set.clone(), env.clone());
        let a = tokio::spawn(async move { s1.warm_with(&*e1, "startup", true, None).await });
        entered.recv().await.expect("pass A is in flight");

        let (s2, e2) = (set.clone(), env.clone());
        let b = tokio::spawn(async move { s2.warm_with(&*e2, "activate", true, None).await });
        // Let B run to its await point — it must be parked on A's pass, not
        // issuing its own warm calls.
        settle().await;

        env.open_gate(32);
        let ra = a.await.unwrap();
        let rb = b.await.unwrap();

        assert_eq!(
            env.warm_calls().len(),
            3,
            "exactly one warm request per role — got {:?}",
            env.warm_calls()
        );
        assert!(rb.coalesced, "the second caller must join the in-flight pass");
        assert!(!rb.changed, "a coalesced caller changed nothing itself");
        assert_eq!(ra.warmed, 3);
        assert_eq!(env.exempt().len(), 3);
    }

    /// Steady state: the three residents are consuming the VRAM the planner
    /// reads, so an ordinary reconcile tick must NOT reclassify them as a
    /// shortfall, must NOT drop the exemption, and must NOT re-warm anything.
    #[tokio::test]
    async fn steady_state_reconcile_preserves_residency_and_issues_no_warm_storm() {
        let set = Arc::new(ResidentSet::new(live_cfg()));
        let (env, _entered) = FakeEnv::new(three_roles(), Some(20.0));

        let first = set.warm_with(&*env, "startup", true, None).await;
        assert_eq!(first.warmed, 3);
        let baseline_calls = env.warm_calls().len();
        let baseline_exempt = env.exempt();
        assert_eq!(baseline_exempt.len(), 3);

        // The residents now occupy 15 of the 20 GB — this is what the counter
        // actually reports in a steady state.
        env.set_free(Some(5.0));

        for _ in 0..3 {
            let tick = set.reconcile_with(&*env).await;
            assert_eq!(tick.dropped, 0, "a steady-state tick must drop nothing");
            assert_eq!(tick.failed, 0);
        }

        assert_eq!(
            env.warm_calls().len(),
            baseline_calls,
            "a steady-state tick must issue NO redundant warm requests — got {:?}",
            env.warm_calls()
        );
        assert_eq!(
            env.exempt(),
            baseline_exempt,
            "the exemption must survive an ordinary tick intact"
        );
        let status = set.status().await;
        assert!(status.active);
        assert!(
            status.roles.iter().all(|r| r.warm),
            "every role must still be warm after a steady-state tick: {:?}",
            status.roles
        );
    }

    /// A refresh whose warm calls FAIL must not downgrade slots that are warm and
    /// valid: the models are still loaded, and dropping the pin only feeds them to
    /// the eviction sweep.
    #[tokio::test]
    async fn failed_refresh_does_not_downgrade_working_residency() {
        // reassert = 0 ⇒ every tick genuinely re-issues the warm calls.
        let cfg = ResidentSetConfig {
            rewarm_debounce: Duration::ZERO,
            reassert: Duration::ZERO,
            ..Default::default()
        };
        let set = Arc::new(ResidentSet::new(cfg));
        let (env, _entered) = FakeEnv::new(three_roles(), Some(96.0));

        assert_eq!(set.warm_with(&*env, "startup", true, None).await.warmed, 3);
        let baseline_exempt = env.exempt();
        assert_eq!(baseline_exempt.len(), 3);

        env.set_fail(true);
        let tick = set.reconcile_with(&*env).await;

        assert_eq!(tick.preserved, 3, "every working slot must be preserved");
        assert_eq!(
            env.exempt(),
            baseline_exempt,
            "a failed refresh must not drop the exemption of models that are still held"
        );
        let status = set.status().await;
        assert!(
            status.roles.iter().all(|r| r.warm),
            "a transient warm failure must not downgrade a valid warm slot: {:?}",
            status.roles
        );
    }

    /// **POSITIVE CONTROL.** An ordinary warm with no concurrent release still
    /// installs the exemption and marks the set active — proving the guards above
    /// did not simply disable residency.
    #[tokio::test]
    async fn ordinary_warm_still_installs_the_exemption_and_activates() {
        let set = Arc::new(ResidentSet::new(live_cfg()));
        let (env, _entered) = FakeEnv::new(three_roles(), Some(96.0));

        let report = set.warm_with(&*env, "startup", true, None).await;

        assert!(report.changed);
        assert!(!report.discarded);
        assert_eq!(report.warmed, 3);
        assert_eq!(
            env.exempt(),
            vec![
                "embed:1".to_string(),
                "router:1".to_string(),
                "voice:1".to_string()
            ]
        );
        assert_eq!(env.warm_calls().len(), 3);
        let status = set.status().await;
        assert!(status.active);
        assert!(status.roles.iter().all(|r| r.warm));
        assert_eq!(status.exempt.len(), 3);
    }

    /// A release must not permanently poison the set: the next activate re-warms
    /// normally. (The generation guard cancels IN-FLIGHT passes, not future ones.)
    #[tokio::test]
    async fn a_release_is_not_permanent_the_next_activate_rewarms() {
        let set = Arc::new(ResidentSet::new(live_cfg()));
        let (env, _entered) = FakeEnv::new(three_roles(), Some(96.0));

        assert_eq!(set.warm_with(&*env, "startup", true, None).await.warmed, 3);
        let rel = set.release_with(&*env, "harmony-idle-lease").await;
        assert!(rel.changed);
        assert_eq!(rel.released, 3);
        assert!(env.exempt().is_empty());

        let again = set.warm_with(&*env, "activate", false, None).await;
        assert_eq!(again.warmed, 3, "a post-release re-warm must work normally");
        assert!(!again.discarded);
        assert_eq!(env.exempt().len(), 3);
        assert!(set.status().await.active);
    }

    /// Releasing an already-inactive set is reported as a no-op but STILL bumps
    /// the generation — a first-ever startup warm in flight must also be cancelled.
    #[tokio::test]
    async fn release_of_an_inactive_set_still_bumps_the_generation() {
        let set = ResidentSet::new(live_cfg());
        let (env, _entered) = FakeEnv::new(three_roles(), Some(96.0));
        let before = set.status().await.generation;
        let rel = set.release_with(&*env, "idle").await;
        assert!(!rel.changed, "idempotent: nothing was held");
        assert_eq!(rel.generation, before + 1);
        assert_eq!(set.status().await.generation, before + 1);
    }

    /// A reconcile on a released set stays quiet (a mode swap owns the host).
    #[tokio::test]
    async fn reconcile_is_a_noop_while_released() {
        let set = ResidentSet::new(live_cfg());
        let (env, _entered) = FakeEnv::new(three_roles(), Some(96.0));
        let r = set.reconcile_with(&*env).await;
        assert!(!r.changed);
        assert!(env.warm_calls().is_empty());
    }
}
