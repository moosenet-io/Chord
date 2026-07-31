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
//! idle — both land on Chord's existing `POST /admin/idle`. So [`release`] is
//! called from [`crate::admin::idle::enter_idle`] AFTER its closed-world drain
//! (an in-flight turn always finishes; we never yank a model out from under a
//! live request) and BEFORE it unloads VRAM. [`crate::admin::idle::activate`]
//! re-warms. No new mode-swap primitive is invented here.
//!
//! ## Fail SOFT, everywhere
//! Nothing on this path can block startup or refuse a turn. An unresolvable
//! alias, an absent model, an unreachable Ollama, an unreadable VRAM counter — all
//! degrade to a logged, observable role state. A cold turn is slower, not broken.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Serialize;
use tokio::sync::Mutex;
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
/// - A model already claimed by a higher-priority role is `Shared` and costs
///   nothing (the three roles may legitimately point at two distinct models).
///
/// The plan is computed once per pass and never retried inside the pass — that is
/// the anti-thrash property.
pub fn plan_warm(resolved: &[Resolved], free_gb: Option<f64>) -> Vec<(Role, WarmDecision)> {
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
                } else {
                    let cost = r.size_gb.unwrap_or(0.0);
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
// The manager
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Slot {
    alias: String,
    model: Option<String>,
    state: RoleState,
    last_used: Option<i64>,
    size_gb: Option<f64>,
}

struct Inner {
    /// True while the set is held. `false` after a release, so a second release
    /// is a cheap no-op (idempotence) and the background reconcile stays quiet.
    active: bool,
    slots: Vec<(Role, Slot)>,
    last_warm: Option<Instant>,
}

/// Outcome of a warm pass (logged + returned for observability).
#[derive(Debug, Clone, PartialEq, Serialize, Default)]
pub struct WarmReport {
    pub warmed: usize,
    pub shared: usize,
    pub dropped: usize,
    pub failed: usize,
    pub skipped: usize,
    /// `false` when the pass was debounced away or the set is disabled.
    pub changed: bool,
}

/// Outcome of a release (logged + returned for observability).
#[derive(Debug, Clone, PartialEq, Serialize, Default)]
pub struct ReleaseReport {
    /// Distinct models whose residency exemption was dropped.
    pub released: usize,
    /// `false` when already released (idempotent no-op).
    pub changed: bool,
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
                    },
                )
            })
            .collect();
        ResidentSet {
            cfg,
            inner: Mutex::new(Inner {
                active: false,
                slots,
                last_warm: None,
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
            keep_alive: self.cfg.keep_alive.clone(),
            roles,
            exempt,
        }
    }

    /// Resolve every role against the CURRENT aliases + registry. Presence is
    /// verified here (the never-pulled guard).
    async fn resolve_all(&self, state: &Arc<crate::routes::AppState>) -> Vec<Resolved> {
        let mut out = Vec::with_capacity(self.cfg.aliases.len());
        let targets: Vec<(Role, String, Option<String>)> = self
            .cfg
            .aliases
            .iter()
            .map(|(role, alias)| {
                let model = resolve_alias(alias, &state.lumina_aliases, &state.model_aliases);
                (*role, alias.clone(), model)
            })
            .collect();

        // ONE brief registry lock for the whole resolution (never held across an
        // await that does I/O).
        let reg = state.model_registry.lock().await;
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
        if !self.cfg.enabled {
            return WarmReport::default();
        }
        {
            let inner = self.inner.lock().await;
            if !force && !should_rewarm(inner.last_warm, Instant::now(), self.cfg.rewarm_debounce) {
                info!(
                    trigger,
                    debounce_secs = self.cfg.rewarm_debounce.as_secs(),
                    "resident-set: re-warm debounced (rapid mode-swap cycle)"
                );
                return WarmReport::default();
            }
        }

        let resolved = self.resolve_all(state).await;
        // Fail-soft VRAM read: `None` (unreadable counter) attempts every role.
        let free_gb = crate::config::read_free_vram_gb();
        let plan = plan_warm(&resolved, free_gb);

        let mut report = WarmReport {
            changed: true,
            ..Default::default()
        };
        let mut new_slots: Vec<(Role, Slot)> = Vec::with_capacity(plan.len());
        let mut held: Vec<String> = Vec::new();

        for (r, (role, decision)) in resolved.iter().zip(plan.into_iter()) {
            debug_assert_eq!(r.role, role);
            let state_for_slot = match decision {
                WarmDecision::Warm { model, size_gb } => {
                    match self.warm_one(state, role, &model).await {
                        Ok(()) => {
                            report.warmed += 1;
                            held.push(model.clone());
                            info!(
                                role = role.id(),
                                alias = %r.alias,
                                model = %model,
                                size_gb = size_gb.unwrap_or(0.0),
                                keep_alive = %self.cfg.keep_alive,
                                "resident-set: role held resident"
                            );
                            RoleState::Warm
                        }
                        Err(reason) => {
                            report.failed += 1;
                            warn!(
                                role = role.id(),
                                model = %model,
                                reason = %reason,
                                "resident-set: warm failed — continuing DEGRADED (a cold turn is slower, not broken)"
                            );
                            RoleState::WarmFailed
                        }
                    }
                }
                WarmDecision::Shared { model } => {
                    report.shared += 1;
                    info!(
                        role = role.id(),
                        model = %model,
                        "resident-set: role shares an already-held model"
                    );
                    held.push(model.clone());
                    RoleState::Warm
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
                    st
                }
            };
            new_slots.push((
                role,
                Slot {
                    alias: r.alias.clone(),
                    model: r.model.clone(),
                    state: state_for_slot,
                    last_used: r.last_used,
                    size_gb: r.size_gb,
                },
            ));
        }

        // Apply the eviction exemption for exactly the models we actually hold.
        // Wholesale replacement, so a repointed alias leaves no stale pin behind.
        held.sort();
        held.dedup();
        {
            let mut reg = state.model_registry.lock().await;
            reg.set_residency_exempt(held.iter().cloned());
        }

        let mut inner = self.inner.lock().await;
        inner.slots = new_slots;
        inner.active = true;
        inner.last_warm = Some(Instant::now());
        info!(
            trigger,
            warmed = report.warmed,
            shared = report.shared,
            dropped = report.dropped,
            failed = report.failed,
            skipped = report.skipped,
            "resident-set: warm pass complete"
        );
        report
    }

    /// RELEASE the whole set for a mode swap (Harmony BLD-11 idle lease, MINT
    /// `enter_idle`): drop the eviction exemption so every held model becomes
    /// immediately reclaimable, and mark the roles released.
    ///
    /// Idempotent — a second release is a no-op with `changed: false`. This does
    /// NOT itself unload VRAM: the idle path's existing
    /// `gpu_exclusive::evict_resident_models` step does that, and it runs right
    /// after us. Calling release first is what lets that step actually reclaim
    /// these models instead of stepping around a pin.
    pub async fn release(
        &self,
        state: &Arc<crate::routes::AppState>,
        reason: &str,
    ) -> ReleaseReport {
        let mut inner = self.inner.lock().await;
        if !inner.active {
            return ReleaseReport {
                released: 0,
                changed: false,
            };
        }
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
        }
        inner.active = false;
        drop(inner);

        {
            let mut reg = state.model_registry.lock().await;
            reg.clear_residency_exempt();
        }
        info!(
            reason,
            released = released.len(),
            "resident-set: RELEASED for a mode swap — models are immediately reclaimable"
        );
        ReleaseReport {
            released: released.len(),
            changed: true,
        }
    }

    /// Re-warm after a mode swap ends (idle lease released / sweep finished).
    /// Debounced, so a rapid acquire/release cycle does not thrash-warm.
    pub async fn rewarm(&self, state: &Arc<crate::routes::AppState>, trigger: &str) -> WarmReport {
        self.warm(state, trigger, false).await
    }

    /// Background reconcile: catch a dynamic alias repoint and a keep_alive that
    /// has drifted. A no-op while released (a mode swap owns the host then).
    ///
    /// On a repoint the NEW target is warmed first and only then is the OLD one
    /// dropped from the exemption — never the reverse, so the role is never
    /// momentarily unbacked. (`warm` already applies the exemption wholesale,
    /// which performs the drop; the ordering falls out of doing the warm first.)
    pub async fn reconcile(&self, state: &Arc<crate::routes::AppState>) -> WarmReport {
        {
            let inner = self.inner.lock().await;
            if !inner.active {
                return WarmReport::default();
            }
        }
        let resolved = self.resolve_all(state).await;
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
        self.warm(state, "reconcile", !repointed.is_empty()).await
    }

    /// Issue the actual Ollama warm call for one role.
    ///
    /// Role-shaped: an embedding model is loaded through `/api/embed` (it cannot
    /// serve `/api/generate`), everything else through `/api/generate` with no
    /// prompt — Ollama's documented "load this model and hold it for keep_alive"
    /// call. Never fatal: every failure is a genericized `Err` string (S77 — no
    /// infrastructure in the message) the caller logs and degrades on.
    async fn warm_one(
        &self,
        state: &Arc<crate::routes::AppState>,
        role: Role,
        model: &str,
    ) -> Result<(), String> {
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
                    "keep_alive": self.cfg.keep_alive,
                }),
            ),
            _ => (
                format!("{base}/api/generate"),
                serde_json::json!({
                    "model": model,
                    "keep_alive": self.cfg.keep_alive,
                }),
            ),
        };
        match state
            .http_client
            .post(&url)
            .json(&body)
            .timeout(self.cfg.warm_timeout)
            .send()
            .await
        {
            Ok(r) if r.status().is_success() => Ok(()),
            Ok(r) => Err(format!("warm rejected with status {}", r.status().as_u16())),
            Err(_) => Err("warm request failed".to_string()),
        }
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
        );
        assert_eq!(plan[0].1, WarmDecision::Skip(RoleState::Missing));
    }

    #[test]
    fn unresolved_alias_is_skipped_not_guessed() {
        let plan = plan_warm(&[resolved(Role::Embedding, None, None, false)], Some(96.0));
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
}
