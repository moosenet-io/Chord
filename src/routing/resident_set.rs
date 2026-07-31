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
//! Each role names a Chord **alias key** (default: personality → `lumina-fast` —
//! the INTERACTIVE tier, see [`Role::default_alias`] — router → `lumina-fast`, the
//! alias Terminus's router already asks for via `TERMINUS_ROUTER_MODEL` — embedding
//! → `lumina-embed`, which falls back to the configured `EMBED_LOCAL_MODEL` when
//! that alias is absent, see [`resolve_role_target`]), overridable per role by env. The alias is resolved through the dynamic [`LuminaAliasStore`] first and
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
//! ## Lifecycle concurrency: RELEASE ALWAYS WINS (TRTR-07b/07d)
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
//!   warm can re-install an exemption after a release.
//! - `reconcile` additionally captures the generation at the moment it observes
//!   `active` and passes it in as an EXPECTATION, so a release landing between
//!   "observed active" and "started the pass" also cancels it.
//! - Concurrent warms COALESCE: the first pass registers an in-flight broadcast
//!   channel under the lock, and any concurrent caller subscribes and awaits that
//!   pass's report instead of issuing a duplicate set of warm requests.
//!
//! ## The generation guard alone is a BOOKKEEPING guarantee, not a GPU one (TRTR-07d)
//! Discarding a pass stops Chord from *recording* residency. It does not stop the
//! HTTP request that pass already put on the wire. So this remained possible:
//!
//! 1. a warm pass drops the lock and starts a slow `warm_one`;
//! 2. `release` takes the lock, bumps the generation, clears the exemption, returns;
//! 3. the idle path unloads VRAM;
//! 4. **the already-issued Ollama request completes and loads the model again with
//!    `keep_alive=24h`.**
//!
//! Chord's state says released; the GPU says occupied, for a day — starving exactly
//! the Harmony/MINT run the mode swap was making room for. Three mechanisms close
//! it, and the guarantee is stated per layer:
//!
//! - **Cancellation.** Every warm pass carries a [`CancelToken`] captured on
//!   admission. `release` cancels it, and [`AppStateEnv::warm_one`] `select!`s the
//!   HTTP future against it — so the request is DROPPED (connection torn down),
//!   not merely ignored. A fresh token is installed for the next epoch.
//! - **Bounded drain.** `release` WAITS for the in-flight pass to finish and undo
//!   itself before returning (and therefore before the idle path's VRAM unload):
//!   `CHORD_RESIDENT_RELEASE_DRAIN_SECS` (default 30) of graceful wait, then
//!   cancellation, then `CHORD_RESIDENT_RELEASE_CANCEL_GRACE_SECS` (default 5).
//!   A release that hangs is its own outage on a shared GPU, so the total wait is
//!   hard-bounded at 35s and release ALWAYS returns. Waiting rather than
//!   cancelling FIRST is deliberate: a warm that is allowed to finish is a warm
//!   whose load state we KNOW, so the unload that follows definitively undoes it.
//!   Cancellation is the escape hatch that keeps the bound, not the primary
//!   mechanism.
//! - **Compensating unload.** Every model a pass ISSUED a warm request for is
//!   remembered. When the pass discards, it POSTs the role-shaped
//!   `keep_alive: 0` unload for each of them — undoing a load that landed
//!   regardless of whether we saw its response. It never fights a legitimate
//!   concurrent re-warm: a model the set currently HOLDS is skipped, and the
//!   in-flight slot is held across the compensation so no new pass can start
//!   underneath it. On the drain-timeout fallback `release` issues the same
//!   compensation itself, under the same two guards.
//!
//! **What is guaranteed, and where.** At the BOOKKEEPING layer: once `release`
//! returns, no in-flight warm can re-install an exemption or mark the set active.
//! At the OLLAMA/GPU layer: once `release` returns *having drained* (the normal
//! path — `ReleaseReport::drained`), every warm request that pass issued has been
//! compensated with an explicit unload, so no load caused by the cancelled pass is
//! still in effect. **Not guaranteed:** a genuinely in-flight HTTP request cannot
//! be aborted *inside Ollama* — dropping the connection stops us waiting, but a
//! load already begun server-side may still complete. That case is covered by the
//! compensating unload, which is issued after the request settles. The one
//! irreducible residual is the drain-TIMEOUT path
//! ([`ReleaseReport::drain_timed_out`]): there, release compensates from its
//! snapshot and returns anyway, and a load that completes *after* both that
//! compensation and the idle path's own `evict_resident_models` sweep would
//! survive until the pass itself completes and compensates (it always does — the
//! unload is idempotent). Nothing here can outlive the pass.
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
use tokio::sync::{broadcast, watch, Mutex};
use tracing::{info, warn};

use super::lumina_alias::LuminaAliasStore;

// ─────────────────────────────────────────────────────────────────────────────
// Cancellation (TRTR-07d)
// ─────────────────────────────────────────────────────────────────────────────

/// A latch-once, cloneable cancellation signal for a warm pass.
///
/// Hand-rolled on `tokio::sync::watch` rather than adding a `tokio-util`
/// dependency for one type: the semantics needed here are exactly "latch a flag
/// once, wake everyone waiting, never un-latch".
#[derive(Clone, Debug)]
pub struct CancelToken {
    tx: Arc<watch::Sender<bool>>,
}

impl Default for CancelToken {
    fn default() -> Self {
        Self::new()
    }
}

impl CancelToken {
    pub fn new() -> Self {
        let (tx, _rx) = watch::channel(false);
        CancelToken { tx: Arc::new(tx) }
    }

    /// Latch the token. Idempotent.
    pub fn cancel(&self) {
        let _ = self.tx.send(true);
    }

    pub fn is_cancelled(&self) -> bool {
        *self.tx.borrow()
    }

    /// Resolves as soon as the token is cancelled — immediately if it already is.
    /// Never resolves otherwise, so it is safe as a `select!` arm.
    pub async fn cancelled(&self) {
        let mut rx = self.tx.subscribe();
        loop {
            if *rx.borrow_and_update() {
                return;
            }
            if rx.changed().await.is_err() {
                // The sender lives inside this token, so this is unreachable in
                // practice. Park rather than report a cancel we never received.
                std::future::pending::<()>().await;
            }
        }
    }
}

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
            // Lumina's voice — the chat turn a HUMAN waits on, so it resolves
            // through the INTERACTIVE tier.
            //
            // CHRD-PIN-01: this was `lumina-deep`. `lumina-deep` is by construction
            // the biggest/deepest tier (its blend is 0.65*q + 0.30*a + only 0.05*r,
            // i.e. responsiveness is almost unweighted), so it legitimately selects a
            // model that cannot produce a conversational turn in a realistic
            // interactive timeframe — and on the live fleet it resolved to a target
            // that was not even pulled (`warm rejected with status 404`). The
            // personality slot therefore points at the tier that carries a
            // responsiveness weight and that Lumina's own chat path already requests.
            // Still an ALIAS, never a model name: the dynamic updater stays in charge
            // of which model actually serves it, subject to its quality floor.
            // Override per deployment with `CHORD_RESIDENT_ROLE_PERSONALITY`.
            Role::Personality => "lumina-fast",
            // The alias Terminus's tool-selecting sub-agent already requests. Sharing
            // it with personality is not a degradation — `plan_warm` holds a shared
            // model ONCE (`WarmDecision::Shared`) and both roles read it warm.
            Role::Router => "lumina-fast",
            // Engram memory. There is deliberately no `lumina-embed` alias on this
            // fleet; rather than degrade to nothing, an unresolved embedding role
            // falls back to the CONFIGURED local embedding model
            // (`EMBED_LOCAL_MODEL` — the exact model `/v1/embeddings` serves). See
            // [`embedding_fallback_target`].
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
    /// `CHORD_RESIDENT_RELEASE_DRAIN_SECS` (default 30) — TRTR-07d. How long
    /// [`ResidentSet::release_with`] waits GRACEFULLY for an in-flight warm pass
    /// to finish and undo itself before escalating to cancellation.
    pub release_drain: Duration,
    /// `CHORD_RESIDENT_RELEASE_CANCEL_GRACE_SECS` (default 5) — TRTR-07d. How
    /// long release waits AFTER cancelling for the pass to compensate, before it
    /// compensates from its own snapshot and returns anyway. Release therefore
    /// blocks for at most `release_drain + cancel_grace` (35s by default) — a
    /// release that hangs is its own outage on a shared GPU.
    pub cancel_grace: Duration,
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
            release_drain: Duration::from_secs(30),
            cancel_grace: Duration::from_secs(5),
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
            release_drain: env_secs("CHORD_RESIDENT_RELEASE_DRAIN_SECS", 30),
            cancel_grace: env_secs("CHORD_RESIDENT_RELEASE_CANCEL_GRACE_SECS", 5),
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

/// Where a role's concrete target came from — reported so a fallback is visible
/// in the log rather than looking like a normal alias resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetSource {
    /// The role's alias key resolved (dynamic store or static map).
    Alias,
    /// The alias resolved to nothing and a CONFIGURED default was used instead.
    /// Only the embedding role has one (`EMBED_LOCAL_MODEL`).
    ConfiguredDefault,
    /// Nothing resolved — the role degrades.
    None,
}

/// CHRD-PIN-01 (Task B): resolve a role's concrete target, with the ONE sanctioned
/// fallback.
///
/// The rule is deliberately narrow. A role whose alias resolves to nothing normally
/// degrades LOUDLY and holds nothing — falling back to a guess would be exactly the
/// hard-wiring this module forbids. But the embedding role is different: Chord
/// ALREADY has a configured, authoritative answer to "which local embedding model do
/// we serve" — `EMBED_LOCAL_MODEL`, the model `/v1/embeddings` actually calls. Using
/// it is not a guess and not a hard-wired name; it is reading the config that is
/// already the source of truth, and it keeps the two from disagreeing about which
/// model Engram's memory vectors come from. Every other role has no such config, so
/// it gets no fallback.
///
/// `embedding_default` is passed in (never read from env here) so this stays pure.
pub fn resolve_role_target(
    role: Role,
    alias: &str,
    dynamic: &LuminaAliasStore,
    statics: &HashMap<String, String>,
    embedding_default: Option<&str>,
) -> (Option<String>, TargetSource) {
    if let Some(model) = resolve_alias(alias, dynamic, statics) {
        return (Some(model), TargetSource::Alias);
    }
    if role == Role::Embedding {
        if let Some(d) = embedding_default
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
        {
            return (Some(d.to_string()), TargetSource::ConfiguredDefault);
        }
    }
    (None, TargetSource::None)
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
    ///
    /// TRTR-07d: MUST be cancellation-aware — when `cancel` latches, abandon the
    /// request (drop the future / tear the connection down) and return `Err`
    /// rather than continuing to wait. An implementation that merely ignores the
    /// token re-opens the post-release-load race this seam exists to close.
    async fn warm_one(
        &self,
        role: Role,
        model: &str,
        keep_alive: &str,
        cancel: &CancelToken,
    ) -> Result<(), String>;
    /// TRTR-07d COMPENSATION: force `model` back out of VRAM immediately
    /// (role-shaped `keep_alive: 0`). Issued for any model whose warm request was
    /// put on the wire by a pass that a release then invalidated — the load may
    /// have landed server-side even though we never accepted its response. MUST
    /// be idempotent: unloading a model that is not loaded is a no-op.
    async fn unload_one(&self, role: Role, model: &str) -> Result<(), String>;
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

/// Build the role-shaped Ollama request. `keep_alive` is a JSON value so the same
/// builder serves both the warm (`"24h"`) and the TRTR-07d compensating unload
/// (`0`) — the two must hit the SAME endpoint per role or the undo silently misses
/// (an embedding model cannot be addressed through `/api/generate`).
fn ollama_role_request(
    base: &str,
    role: Role,
    model: &str,
    keep_alive: serde_json::Value,
) -> (String, serde_json::Value) {
    match role {
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
    }
}

#[async_trait]
impl<'a> ResidentEnv for AppStateEnv<'a> {
    async fn resolve(&self, aliases: &[(Role, String)]) -> Vec<Resolved> {
        let state = self.state;
        // The configured local embedding model — the same value `/v1/embeddings`
        // serves. Read once per pass, never hard-wired.
        let embedding_default = crate::embeddings::EmbeddingsConfig::from_env().local_model;
        let targets: Vec<(Role, String, Option<String>)> = aliases
            .iter()
            .map(|(role, alias)| {
                let (model, source) = resolve_role_target(
                    *role,
                    alias,
                    &state.lumina_aliases,
                    &state.model_aliases,
                    Some(embedding_default.as_str()),
                );
                if source == TargetSource::ConfiguredDefault {
                    // LOUD on purpose: the role is serviceable, but the operator
                    // should know the alias is missing and a config default stood in.
                    warn!(
                        role = role.id(),
                        alias = %alias,
                        model = model.as_deref().unwrap_or(""),
                        "resident-set: role alias resolves to no target — falling back to the CONFIGURED local embedding model (set the alias to silence this)"
                    );
                }
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
    ///
    /// TRTR-07d: the send future is raced against `cancel`. Dropping a `reqwest`
    /// send future tears the connection down, so a cancelled warm genuinely stops
    /// being an in-flight request instead of quietly completing after a release.
    /// It cannot un-issue a load Ollama already started, which is why the caller
    /// pairs cancellation with a compensating unload.
    async fn warm_one(
        &self,
        role: Role,
        model: &str,
        keep_alive: &str,
        cancel: &CancelToken,
    ) -> Result<(), String> {
        let Some(base) = crate::gpu_exclusive::ollama_base_from_env() else {
            return Err("ollama base not configured".to_string());
        };
        let base = base.trim_end_matches('/');
        let (url, body) = ollama_role_request(base, role, model, serde_json::json!(keep_alive));
        if cancel.is_cancelled() {
            return Err("warm cancelled before it was issued".to_string());
        }
        let send = self
            .state
            .http_client
            .post(&url)
            .json(&body)
            .timeout(self.warm_timeout)
            .send();
        tokio::pin!(send);
        let res = tokio::select! {
            biased;
            r = &mut send => r,
            _ = cancel.cancelled() => {
                // Dropping `send` here is what aborts the HTTP request.
                return Err("warm cancelled mid-flight by a mode-swap release".to_string());
            }
        };
        match res {
            Ok(r) if r.status().is_success() => Ok(()),
            Ok(r) => Err(format!("warm rejected with status {}", r.status().as_u16())),
            Err(_) => Err("warm request failed".to_string()),
        }
    }

    /// TRTR-07d compensation: the same role-shaped endpoint with `keep_alive: 0`,
    /// which is Ollama's documented "drop this model from VRAM now". Deliberately
    /// NOT cancellable — this is the undo, and it must be allowed to land.
    async fn unload_one(&self, role: Role, model: &str) -> Result<(), String> {
        let Some(base) = crate::gpu_exclusive::ollama_base_from_env() else {
            return Err("ollama base not configured".to_string());
        };
        let base = base.trim_end_matches('/');
        let (url, body) = ollama_role_request(base, role, model, serde_json::json!(0));
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
            Ok(r) => Err(format!("unload rejected with status {}", r.status().as_u16())),
            Err(_) => Err("unload request failed".to_string()),
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
    /// TRTR-07d. Cancellation token for the CURRENT lifecycle epoch. A warm pass
    /// clones it on admission; `release` cancels it (aborting the in-flight HTTP)
    /// and installs a fresh one for the next epoch.
    cancel: CancelToken,
    /// TRTR-07d. `(role, model)` for every warm request the in-flight pass has
    /// ISSUED and not yet reconciled. This is the compensation worklist: a request
    /// that went out may have loaded a model even if we never accepted its
    /// response, so it has to be undoable from outside the pass too.
    pending: Vec<(Role, String)>,
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
    /// TRTR-07d: roles whose warm request was never issued because a release had
    /// already cancelled this pass.
    pub cancelled: usize,
    /// TRTR-07d: models this pass had put a warm request on the wire for and then
    /// explicitly UNLOADED again, because a release invalidated the pass.
    pub compensated: usize,
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
    /// TRTR-07d: an in-flight warm pass was awaited to completion (and therefore
    /// to its compensating unloads) before this release returned. `false` when
    /// there was nothing in flight OR the bounded wait expired.
    pub drained: bool,
    /// TRTR-07d: the bounded wait (`release_drain` + `cancel_grace`) expired. The
    /// documented FALLBACK ran: release issued the compensating unloads itself
    /// from its own pending snapshot and returned rather than blocking a shared
    /// GPU. The pass will also compensate when it finishes; unloads are idempotent.
    pub drain_timed_out: bool,
    /// TRTR-07d: how many models were explicitly unloaded as compensation for
    /// warm requests that a cancelled pass had already put on the wire.
    pub compensated: usize,
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
        cancel: CancelToken,
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
                cancel: CancelToken::new(),
                pending: Vec::new(),
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
        // A fresh pass owns a fresh compensation worklist.
        inner.pending.clear();
        let cancel = inner.cancel.clone();
        Admission::Proceed {
            generation: inner.generation,
            cancel,
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
        let (generation, cancel, snapshot) = match self.admit(trigger, force, expect_gen).await {
            Admission::Skip(r) => return r,
            Admission::Join(mut rx) => {
                // Join the in-flight pass: no duplicate warm requests, and we
                // report ITS outcome. `changed` is false because THIS caller
                // changed nothing.
                //
                // TRTR-07d requirement 3: this wait is BOUNDED. A leader that is
                // cancelled always reaches its commit and broadcasts, so the
                // normal path returns promptly — but a subscriber must never be
                // able to hang forever on a leader that somehow never reports.
                let bound = self.cfg.warm_timeout
                    + self.cfg.release_drain
                    + self.cfg.cancel_grace
                    + Duration::from_secs(5);
                let mut report = match tokio::time::timeout(bound, rx.recv()).await {
                    Ok(r) => r.unwrap_or_default(),
                    Err(_) => {
                        warn!(
                            trigger,
                            bound_secs = bound.as_secs(),
                            "resident-set: coalesced warm gave up waiting for the leader's report — reporting nothing rather than hanging"
                        );
                        WarmReport::default()
                    }
                };
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
                cancel,
                snapshot,
            } => (generation, cancel, snapshot),
        };

        self.run_pass(env, trigger, generation, cancel, snapshot).await
    }

    /// The slow half (resolution + warm I/O, NO lock held) followed by the
    /// atomic commit.
    async fn run_pass(
        &self,
        env: &dyn ResidentEnv,
        trigger: &str,
        generation: u64,
        cancel: CancelToken,
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
        // TRTR-07d: every model this pass actually put a request on the wire for.
        // The compensation worklist — issuing the request is what can load a
        // model, so it is issuance (not success) that creates the obligation.
        let mut issued: Vec<(Role, String)> = Vec::new();

        for (r, (role, decision)) in resolved.iter().zip(plan.into_iter()) {
            debug_assert_eq!(r.role, role);
            let (state_for_slot, warmed_at) = match decision {
                WarmDecision::Warm { model, size_gb } if cancel.is_cancelled() => {
                    // A release has already landed. Do not open a NEW request we
                    // would only have to undo.
                    report.cancelled += 1;
                    let _ = size_gb;
                    info!(
                        trigger,
                        role = role.id(),
                        model = %model,
                        "resident-set: warm not issued — a mode-swap release cancelled this pass"
                    );
                    (RoleState::Released, None)
                }
                WarmDecision::Warm { model, size_gb } => {
                    // Record the obligation BEFORE the request goes out, so a
                    // release that lands mid-flight can compensate for it even if
                    // this pass never gets to run again.
                    {
                        let mut inner = self.inner.lock().await;
                        inner.pending.push((role, model.clone()));
                    }
                    issued.push((role, model.clone()));
                    match env
                        .warm_one(role, &model, &self.cfg.keep_alive, &cancel)
                        .await
                    {
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

        self.commit(env, trigger, generation, issued, outcomes, report)
            .await
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
        issued: Vec<(Role, String)>,
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
            // TRTR-07d: discarding is only the BOOKKEEPING half. Every warm
            // request this pass put on the wire may have loaded a model, so undo
            // each one explicitly — skipping any model the set currently HOLDS,
            // because that means a legitimate later pass owns it and we must not
            // fight it.
            let held_now: HashSet<String> = inner
                .slots
                .iter()
                .filter(|(_, s)| s.state.is_held())
                .filter_map(|(_, s)| s.model.clone())
                .collect();
            let mut seen: HashSet<String> = HashSet::new();
            let mut to_undo: Vec<(Role, String)> = Vec::new();
            for (role, model) in issued.iter() {
                if held_now.contains(model) || !seen.insert(model.clone()) {
                    continue;
                }
                to_undo.push((*role, model.clone()));
            }
            inner
                .pending
                .retain(|(_, m)| !to_undo.iter().any(|(_, u)| u == m));
            warn!(
                trigger,
                pass_generation = generation,
                current_generation = inner.generation,
                undo = to_undo.len(),
                "resident-set: warm pass DISCARDED — a mode-swap release landed mid-warm; unloading anything it loaded so the GPU really is released"
            );
            // NOTE: `in_flight` is deliberately NOT taken yet. Holding it keeps
            // any concurrent caller COALESCED onto this (discarded) pass, so no
            // new warm can start and race the compensating unloads below.
            drop(inner);

            for (role, model) in &to_undo {
                match env.unload_one(*role, model).await {
                    Ok(()) => {
                        report.compensated += 1;
                        warn!(
                            trigger,
                            role = role.id(),
                            model = %model,
                            "resident-set: compensating unload issued for a warm that a release invalidated"
                        );
                    }
                    Err(reason) => warn!(
                        trigger,
                        role = role.id(),
                        model = %model,
                        reason = %reason,
                        "resident-set: compensating unload did not succeed — the idle path's own VRAM eviction is the backstop"
                    ),
                }
            }

            let mut inner = self.inner.lock().await;
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
        // The pass committed: nothing is owed a compensating unload.
        inner.pending.clear();
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
        let (mut report, in_flight, pending, cancel, generation) = {
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

            // Inside the `inner` lock, same order the commit uses — so a warm pass
            // can neither observe a half-applied release nor slip its exemption in
            // after this clear.
            env.clear_exempt().await;

            // TRTR-07d: subscribe to the in-flight pass BEFORE dropping the lock
            // (so its report cannot be missed), snapshot its compensation
            // worklist, and take the epoch's cancel token. A pass admitted after
            // this point gets the FRESH token and is not cancelled by us.
            let in_flight = inner.in_flight.as_ref().map(|tx| tx.subscribe());
            let pending = inner.pending.clone();
            let cancel = inner.cancel.clone();
            inner.cancel = CancelToken::new();

            let report = ReleaseReport {
                released: released.len(),
                changed: was_active,
                generation,
                ..Default::default()
            };
            (report, in_flight, pending, cancel, generation)
        };

        if let Some(mut rx) = in_flight {
            // Phase 1 — GRACEFUL DRAIN. Let the in-flight pass finish naturally
            // and undo itself. Bounded, because release must never become the
            // outage: a blocked release blocks the whole mode swap.
            let drained = match tokio::time::timeout(self.cfg.release_drain, rx.recv()).await {
                Ok(Ok(pass)) => Some(pass.compensated),
                // Sender dropped without a report: the pass is gone, nothing more
                // can land from it.
                Ok(Err(_)) => Some(0),
                Err(_) => None,
            };
            match drained {
                Some(c) => {
                    report.drained = true;
                    report.compensated += c;
                }
                None => {
                    // Phase 2 — ESCALATE: abort the in-flight HTTP request itself,
                    // then give the pass a short grace to compensate.
                    warn!(
                        reason,
                        drain_secs = self.cfg.release_drain.as_secs(),
                        "resident-set: release drain bound exceeded — CANCELLING the in-flight warm"
                    );
                    cancel.cancel();
                    match tokio::time::timeout(self.cfg.cancel_grace, rx.recv()).await {
                        Ok(Ok(pass)) => {
                            report.drained = true;
                            report.compensated += pass.compensated;
                        }
                        Ok(Err(_)) => report.drained = true,
                        Err(_) => {
                            // Phase 3 — DOCUMENTED FALLBACK. Never block: return,
                            // but compensate from our own snapshot first so the
                            // GPU is not left holding a model we believe we
                            // released. The pass will compensate too when it
                            // finally completes; the unload is idempotent.
                            report.drain_timed_out = true;
                            warn!(
                                reason,
                                grace_secs = self.cfg.cancel_grace.as_secs(),
                                pending = pending.len(),
                                "resident-set: in-flight warm did not settle within the bounded wait — compensating from the release side and returning anyway"
                            );
                            report.compensated +=
                                self.compensate_pending(env, generation, &pending).await;
                        }
                    }
                }
            }
        }

        info!(
            reason,
            released = report.released,
            generation,
            drained = report.drained,
            drain_timed_out = report.drain_timed_out,
            compensated = report.compensated,
            "resident-set: RELEASED for a mode swap — models are immediately reclaimable"
        );
        report
    }

    /// TRTR-07d fallback compensation, issued by `release` itself when the bounded
    /// wait expired. Same two guards as the pass-side compensation: stop if a
    /// NEWER release has taken over (it owns the decision now), and never unload a
    /// model the set currently HOLDS (a legitimate re-warm owns it).
    async fn compensate_pending(
        &self,
        env: &dyn ResidentEnv,
        generation: u64,
        pending: &[(Role, String)],
    ) -> usize {
        let mut done = 0usize;
        let mut seen: HashSet<String> = HashSet::new();
        for (role, model) in pending {
            if !seen.insert(model.clone()) {
                continue;
            }
            {
                let inner = self.inner.lock().await;
                if inner.generation != generation {
                    break;
                }
                if inner
                    .slots
                    .iter()
                    .any(|(_, s)| s.state.is_held() && s.model.as_deref() == Some(model.as_str()))
                {
                    continue;
                }
            }
            match env.unload_one(*role, model).await {
                Ok(()) => {
                    done += 1;
                    warn!(
                        role = role.id(),
                        model = %model,
                        "resident-set: release-side compensating unload issued for an unsettled in-flight warm"
                    );
                }
                Err(reason) => warn!(
                    role = role.id(),
                    model = %model,
                    reason = %reason,
                    "resident-set: release-side compensating unload did not succeed — the idle path's VRAM eviction is the backstop"
                ),
            }
        }
        done
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

// ─────────────────────────────────────────────────────────────────────────────
// CHRD-PIN-01: startup pin convergence
// ─────────────────────────────────────────────────────────────────────────────
//
// The retired `MODEL_KEEP_RESIDENT` mechanism pinned models with `keep_alive:-1`
// (INDEFINITE). Deleting the code that issues those pins does NOT unpin what is
// already pinned: Ollama holds a `-1` model until the process is restarted or
// something explicitly unloads it, and a Chord restart does not touch Ollama. On
// the live host that is four models and ~60 GB of VRAM that no lifecycle owns and
// no mode swap can reclaim — exactly the starvation this change exists to end.
//
// So the fix CONVERGES the running host instead of requiring a manual unload: once
// at startup, after the resident set's own first warm, look at what Ollama actually
// has loaded and unload anything pinned indefinitely that the resident set does not
// hold. Deliberately conservative:
//   - a model the set HOLDS is never touched (its own warm has already re-asserted
//     the bounded `keep_alive`, so it no longer reads as an indefinite pin anyway —
//     the explicit skip is belt-and-braces);
//   - an expiry we cannot PARSE is left alone (never unload on ambiguity);
//   - only an expiry beyond the horizon counts. A bounded keep_alive — including
//     the resident set's own 24h — is never mistaken for a pin.
//   - best-effort throughout: an unreachable Ollama logs and does nothing.

/// `CHORD_RESIDENT_PIN_HORIZON_DAYS` (default 365). An `/api/ps` expiry further out
/// than this is a legacy INDEFINITE pin, not a keep_alive. Ollama renders `-1` as a
/// year-2318-style timestamp, so the horizon has enormous margin over any real
/// keep_alive while never catching one.
pub fn pin_horizon_days() -> i64 {
    std::env::var("CHORD_RESIDENT_PIN_HORIZON_DAYS")
        .ok()
        .and_then(|s| s.trim().parse::<i64>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(365)
}

/// Is this `/api/ps` `expires_at` an INDEFINITE pin (further out than `horizon_days`
/// from `now`)? Pure. Unparseable/absent ⇒ `false` — we never unload on ambiguity.
pub fn is_indefinite_pin(expires_at: Option<&str>, now_epoch_secs: i64, horizon_days: i64) -> bool {
    let Some(raw) = expires_at.map(|s| s.trim()).filter(|s| !s.is_empty()) else {
        return false;
    };
    let Ok(ts) = chrono::DateTime::parse_from_rfc3339(raw) else {
        return false;
    };
    ts.timestamp().saturating_sub(now_epoch_secs) > horizon_days.saturating_mul(86_400)
}

/// Which loaded models are legacy indefinite pins that must be released: pinned
/// beyond the horizon AND not held by the resident set. Pure — the whole policy in
/// one testable place, no network.
pub fn plan_unpin(
    loaded: &[(String, Option<String>)],
    held: &HashSet<String>,
    now_epoch_secs: i64,
    horizon_days: i64,
) -> Vec<String> {
    loaded
        .iter()
        .filter(|(name, _)| !name.trim().is_empty())
        .filter(|(name, _)| !held.contains(name.as_str()))
        .filter(|(_, exp)| is_indefinite_pin(exp.as_deref(), now_epoch_secs, horizon_days))
        .map(|(name, _)| name.clone())
        .collect()
}

/// Unload one arbitrary (role-unknown) model: `/api/generate` with `keep_alive: 0`,
/// falling back to the embeddings endpoint for an embedding model, which cannot be
/// addressed through `/api/generate` at all. Best-effort — returns whether it
/// landed, never errors.
async fn unload_untyped(client: &reqwest::Client, base: &str, model: &str) -> bool {
    async fn post(client: &reqwest::Client, url: &str, body: serde_json::Value) -> Option<(u16, String)> {
        let r = client
            .post(url)
            .json(&body)
            .timeout(Duration::from_secs(60))
            .send()
            .await
            .ok()?;
        let status = r.status().as_u16();
        let text = r.text().await.unwrap_or_default();
        Some((status, text))
    }
    let embed = || async {
        let body = serde_json::json!({ "model": model, "input": "", "keep_alive": 0 });
        matches!(
            post(client, &format!("{base}/api/embed"), body).await,
            Some((s, _)) if (200..300).contains(&s)
        )
    };
    if crate::gpu_exclusive::is_embedding_model(model) {
        return embed().await;
    }
    let body = serde_json::json!({ "model": model, "keep_alive": 0 });
    match post(client, &format!("{base}/api/generate"), body).await {
        Some((s, _)) if (200..300).contains(&s) => true,
        // An embedder whose NAME did not match the heuristic: Ollama says so
        // explicitly, and only then do we retry on the embeddings endpoint.
        Some((400, body)) if crate::gpu_exclusive::is_generate_unsupported_rejection(&body) => {
            embed().await
        }
        _ => false,
    }
}

/// How many models a convergence pass released (returned for observability/tests).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ConvergeReport {
    pub found: usize,
    pub unpinned: usize,
}

/// Run the startup convergence described above. Best-effort and idempotent.
pub async fn converge_legacy_pins(state: &Arc<crate::routes::AppState>) -> ConvergeReport {
    let Some(base) = crate::gpu_exclusive::ollama_base_from_env() else {
        info!("resident-set: OLLAMA_URL unset — skipping legacy pin convergence (best-effort)");
        return ConvergeReport::default();
    };
    let base = base.trim_end_matches('/').to_string();
    let stats = crate::sweep_status::ollama::query_ollama_ps(&state.http_client, &base).await;
    if !stats.available {
        info!("resident-set: Ollama /api/ps unavailable — skipping legacy pin convergence");
        return ConvergeReport::default();
    }
    let held: HashSet<String> = global()
        .status()
        .await
        .roles
        .into_iter()
        .filter(|r| r.warm)
        .filter_map(|r| r.model)
        .collect();
    let loaded: Vec<(String, Option<String>)> = stats
        .models
        .into_iter()
        .map(|m| (m.name, m.expires_at))
        .collect();
    let targets = plan_unpin(
        &loaded,
        &held,
        crate::gpu_exclusive::now_epoch() as i64,
        pin_horizon_days(),
    );
    let mut report = ConvergeReport {
        found: targets.len(),
        unpinned: 0,
    };
    for model in targets {
        warn!(
            model = %model,
            "resident-set: found a LEGACY INDEFINITE VRAM pin owned by no lifecycle (retired keep-resident mechanism) — releasing it"
        );
        if unload_untyped(&state.http_client, &base, &model).await {
            report.unpinned += 1;
        } else {
            warn!(
                model = %model,
                "resident-set: legacy pin release did not land (best-effort, continuing)"
            );
        }
    }
    if report.found > 0 {
        info!(
            found = report.found,
            unpinned = report.unpinned,
            "resident-set: legacy pin convergence complete — residency now has a single owner"
        );
    }
    report
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

    // ── CHRD-PIN-01 Task B: role targets ────────────────────────────────────

    #[test]
    fn personality_resolves_through_the_interactive_tier_not_the_deep_tier() {
        // The regression this guards: personality pointed at `lumina-deep`, whose
        // blend all but ignores responsiveness, so it selected a model that cannot
        // produce a conversational turn in a realistic timeframe (and on the live
        // fleet was not even pulled).
        assert_eq!(Role::Personality.default_alias(), "lumina-fast");
        assert_ne!(Role::Personality.default_alias(), "lumina-deep");
        // Still an ALIAS, never a model name.
        let a = Role::Personality.default_alias();
        assert!(!a.contains(':') && !a.contains('/'));
    }

    #[test]
    fn embedding_role_falls_back_to_the_configured_local_embedding_model() {
        let statics = HashMap::new();
        let dynamic = LuminaAliasStore::empty();
        // The alias does not exist (as on the live fleet) — but Chord already knows
        // which local embedding model it serves, so the role is served, not degraded.
        let (model, source) = resolve_role_target(
            Role::Embedding,
            "lumina-embed",
            &dynamic,
            &statics,
            Some("configured-embed-model:0.6b"),
        );
        assert_eq!(model.as_deref(), Some("configured-embed-model:0.6b"));
        assert_eq!(source, TargetSource::ConfiguredDefault);
    }

    #[test]
    fn a_resolvable_alias_always_wins_over_the_configured_default() {
        let mut statics = HashMap::new();
        statics.insert("lumina-embed".to_string(), "alias-model:1b".to_string());
        let dynamic = LuminaAliasStore::empty();
        let (model, source) = resolve_role_target(
            Role::Embedding,
            "lumina-embed",
            &dynamic,
            &statics,
            Some("configured-embed-model:0.6b"),
        );
        assert_eq!(model.as_deref(), Some("alias-model:1b"));
        assert_eq!(source, TargetSource::Alias);
    }

    #[test]
    fn only_the_embedding_role_has_a_configured_fallback() {
        let statics = HashMap::new();
        let dynamic = LuminaAliasStore::empty();
        for role in [Role::Personality, Role::Router] {
            let (model, source) = resolve_role_target(
                role,
                "no-such-alias",
                &dynamic,
                &statics,
                Some("configured-embed-model:0.6b"),
            );
            assert_eq!(model, None, "{} must NOT inherit a fallback", role.id());
            assert_eq!(source, TargetSource::None);
        }
        // …and an embedding role with no configured default degrades too, rather
        // than inventing a name.
        for empty in [None, Some(""), Some("   ")] {
            let (model, source) =
                resolve_role_target(Role::Embedding, "lumina-embed", &dynamic, &statics, empty);
            assert_eq!(model, None);
            assert_eq!(source, TargetSource::None);
        }
    }

    // ── CHRD-PIN-01 Task A: no indefinite pin survives ──────────────────────

    /// The load-bearing invariant, asserted against the SOURCE TREE rather than a
    /// behavior: **no code path anywhere in this crate may ask Ollama for an
    /// indefinite `keep_alive`.** A behavioral test can only cover the paths it
    /// knows about; the failure mode being guarded is a NEW (or resurrected) path
    /// nobody wired a test to — which is exactly how the retired keep-resident pass
    /// coexisted with the resident set in the first place.
    ///
    /// A genuinely bounded, explicitly-released phase may opt out by putting the
    /// marker `RESIDENCY-PIN-ALLOWED` on the line, which makes the exception
    /// deliberate, greppable, and reviewable instead of silent.
    #[test]
    fn no_code_path_pins_a_model_indefinitely() {
        // Built at runtime so this guard cannot match its own source.
        let key: String = ["keep", "_alive"].concat();
        let indefinite: String = ["-", "1"].concat();
        let allow = ["RESIDENCY", "-PIN-ALLOWED"].concat();

        fn rs_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            let Ok(entries) = std::fs::read_dir(dir) else {
                return;
            };
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    rs_files(&p, out);
                } else if p.extension().and_then(|s| s.to_str()) == Some("rs") {
                    out.push(p);
                }
            }
        }
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        rs_files(&root, &mut files);
        assert!(files.len() > 10, "source scan found suspiciously few files");

        let mut offenders = Vec::new();
        for f in files {
            let Ok(text) = std::fs::read_to_string(&f) else {
                continue;
            };
            for (n, line) in text.lines().enumerate() {
                let t = line.trim_start();
                // Prose (docs/comments) may DISCUSS the retired mechanism.
                if t.starts_with("//") || t.starts_with("*") {
                    continue;
                }
                if line.contains(&allow) {
                    continue;
                }
                if line.contains(&key) && line.contains(&indefinite) {
                    offenders.push(format!("{}:{}: {}", f.display(), n + 1, t));
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "an INDEFINITE keep_alive pin is outside every lifecycle — no mode swap can \
             reclaim it. Hold models through `ResidentSet` (bounded keep_alive + release) \
             instead. Offending lines:\n{}",
            offenders.join("\n")
        );
    }

    #[test]
    fn an_indefinite_expiry_is_a_pin_but_a_bounded_keep_alive_is_not() {
        let now = 1_800_000_000i64; // fixed epoch; no wall clock in the assertion
        let horizon = 365;
        // Ollama renders keep_alive:-1 as a year-2318-style timestamp.
        assert!(is_indefinite_pin(
            Some("2318-01-01T00:00:00Z"),
            now,
            horizon
        ));
        // The resident set's own 24h keep_alive must NEVER read as a pin.
        let in_24h = chrono::DateTime::from_timestamp(now + 86_400, 0)
            .unwrap()
            .to_rfc3339();
        assert!(!is_indefinite_pin(Some(&in_24h), now, horizon));
        // Ambiguity is never a reason to unload.
        assert!(!is_indefinite_pin(None, now, horizon));
        assert!(!is_indefinite_pin(Some(""), now, horizon));
        assert!(!is_indefinite_pin(Some("not-a-timestamp"), now, horizon));
        // Exactly at the horizon is NOT past it (strict >).
        let at_horizon = chrono::DateTime::from_timestamp(now + 365 * 86_400, 0)
            .unwrap()
            .to_rfc3339();
        assert!(!is_indefinite_pin(Some(&at_horizon), now, horizon));
    }

    #[test]
    fn convergence_releases_stranded_pins_and_never_touches_what_the_set_holds() {
        let now = 1_800_000_000i64;
        let far = "2318-01-01T00:00:00Z".to_string();
        let soon = chrono::DateTime::from_timestamp(now + 3600, 0)
            .unwrap()
            .to_rfc3339();
        let loaded = vec![
            // Pinned indefinitely by the retired mechanism, held by nobody.
            ("stranded-a:8b".to_string(), Some(far.clone())),
            ("stranded-b:30b".to_string(), Some(far.clone())),
            // Pinned indefinitely BUT the resident set holds it — its own warm owns
            // the lifecycle, so convergence must not fight it.
            ("held-by-the-set:30b".to_string(), Some(far.clone())),
            // An ordinary bounded model: not a pin, not our business.
            ("ordinary:3b".to_string(), Some(soon)),
            // Unparseable expiry: never unload on ambiguity.
            ("unknown-expiry:1b".to_string(), Some("???".to_string())),
            ("".to_string(), Some(far)),
        ];
        let held: HashSet<String> = ["held-by-the-set:30b".to_string()].into_iter().collect();
        let plan = plan_unpin(&loaded, &held, now, 365);
        assert_eq!(plan, vec!["stranded-a:8b", "stranded-b:30b"]);
    }

    #[test]
    fn convergence_is_a_noop_on_a_host_with_no_legacy_pins() {
        let now = 1_800_000_000i64;
        let soon = chrono::DateTime::from_timestamp(now + 86_400, 0)
            .unwrap()
            .to_rfc3339();
        let loaded = vec![("a:1b".to_string(), Some(soon))];
        assert!(plan_unpin(&loaded, &HashSet::new(), now, 365).is_empty());
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
        // CHRD-PIN-01: personality resolves through the INTERACTIVE tier, not the
        // deep tier (which selects for depth over responsiveness and cannot produce
        // a conversational turn in a realistic timeframe).
        assert_eq!(cfg.alias_for(Role::Personality), "lumina-fast");
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
        /// TRTR-07d: every compensating unload the manager issued, in order.
        unloads: StdMutex<Vec<String>>,
        exempt: StdMutex<Vec<String>>,
        exempt_ops: StdMutex<Vec<String>>,
        fail: AtomicBool,
        gated: AtomicBool,
        /// When set, a gated warm IGNORES the cancel token — modelling an
        /// in-flight warm that simply will not stop (an unresponsive Ollama, a
        /// stuck connection). This is the only situation release's bounded-wait
        /// FALLBACK exists for, so it is the only honest way to drive it.
        uncancellable: AtomicBool,
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
                    unloads: StdMutex::new(Vec::new()),
                    exempt: StdMutex::new(Vec::new()),
                    exempt_ops: StdMutex::new(Vec::new()),
                    fail: AtomicBool::new(false),
                    gated: AtomicBool::new(false),
                    uncancellable: AtomicBool::new(false),
                    gate: Semaphore::new(0),
                    entered: tx,
                }),
                rx,
            )
        }

        fn gate_warms(&self) {
            self.gated.store(true, Ordering::SeqCst);
        }
        /// Gate the warms AND make them unstoppable: cancellation will not free
        /// them, so release's bounded wait must expire.
        fn gate_warms_uncancellable(&self) {
            self.gated.store(true, Ordering::SeqCst);
            self.uncancellable.store(true, Ordering::SeqCst);
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
        fn unloads(&self) -> Vec<String> {
            self.unloads.lock().unwrap().clone()
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
        async fn warm_one(
            &self,
            _role: Role,
            model: &str,
            _ka: &str,
            cancel: &CancelToken,
        ) -> Result<(), String> {
            self.warm_calls.lock().unwrap().push(model.to_string());
            let _ = self.entered.send(model.to_string());
            if self.gated.load(Ordering::SeqCst) {
                // TRTR-07d: a gated warm is "mid network I/O". It must be
                // ABORTABLE, exactly as the production `AppStateEnv::warm_one`
                // races its send future against the token — otherwise a test env
                // that ignored cancellation would make the drain look like it
                // works when it only ever times out.
                if self.uncancellable.load(Ordering::SeqCst) {
                    self.gate.acquire().await.expect("gate").forget();
                } else {
                    tokio::select! {
                        p = self.gate.acquire() => { p.expect("gate").forget(); }
                        _ = cancel.cancelled() => {
                            return Err("fake warm cancelled".to_string());
                        }
                    }
                }
            }
            if self.fail.load(Ordering::SeqCst) {
                Err("fake warm failure".to_string())
            } else {
                Ok(())
            }
        }
        async fn unload_one(&self, _role: Role, model: &str) -> Result<(), String> {
            self.unloads.lock().unwrap().push(model.to_string());
            Ok(())
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
            // TRTR-07d: escalate to cancellation IMMEDIATELY rather than waiting
            // out a graceful drain. These tests gate the warm deliberately, so the
            // graceful phase can only ever expire; going straight to cancellation
            // keeps them deterministic AND with no wall-clock sleeping. The grace
            // that follows is generous and never actually elapses, because a
            // cancelled `FakeEnv::warm_one` returns at once.
            release_drain: Duration::ZERO,
            cancel_grace: Duration::from_secs(30),
            ..Default::default()
        }
    }

    /// Await a spawned pass that cancellation is supposed to unblock, FAILING
    /// loudly instead of hanging if it does not. A regression that made
    /// [`ResidentEnv::warm_one`] ignore its [`CancelToken`] would otherwise turn
    /// these tests into a hang, and a hang is not a red test.
    async fn join_unblocked<T>(h: tokio::task::JoinHandle<T>, what: &str) -> T {
        match tokio::time::timeout(Duration::from_secs(20), h).await {
            Ok(r) => r.unwrap(),
            Err(_) => panic!("{what} never completed — cancellation did not unblock it"),
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
        let report = join_unblocked(warming, "the cancelled startup pass").await;

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
        // TRTR-07d: bookkeeping is not enough. The one warm request that DID go
        // out must have been undone at the model layer too, and release must have
        // waited for that before returning.
        assert!(rel.drained, "release must drain the in-flight pass: {rel:?}");
        assert_eq!(
            env.unloads(),
            env.warm_calls(),
            "every warm request the cancelled pass issued must be compensated by an unload"
        );
        assert_eq!(rel.compensated, env.unloads().len());
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
        let report = join_unblocked(reconciling, "the cancelled reconcile pass").await;

        assert!(report.discarded, "reconcile must discard after a release");
        assert!(env.exempt().is_empty(), "reconcile must not re-pin after a release");
        assert!(
            env.unloads().contains(&"voice:2".to_string()),
            "the repointed target the cancelled reconcile had already asked for must be unloaded again: {:?}",
            env.unloads()
        );
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

    /// **TRTR-07d requirement 3.** Cancelling the LEADER of a coalesced group must
    /// not strand the subscribers awaiting its report: they must return, promptly,
    /// with a sensible (discarded, coalesced) verdict rather than hanging forever
    /// on a broadcast that never arrives.
    #[tokio::test]
    async fn all_three_roles_are_held_when_every_alias_resolves() {
        // The positive statement of the whole feature: three resolving aliases ⇒
        // three roles held, three models exempt, set active.
        let set = Arc::new(ResidentSet::new(live_cfg()));
        let (env, _entered) = FakeEnv::new(three_roles(), Some(96.0));
        let report = set.warm_with(&*env, "startup", true, None).await;

        assert_eq!(report.warmed, 3);
        assert_eq!(report.skipped, 0);
        let status = set.status().await;
        assert!(status.active);
        for r in &status.roles {
            assert_eq!(r.state, RoleState::Warm, "role {} not held", r.role.id());
        }
        assert_eq!(status.exempt.len(), 3);
    }

    #[tokio::test]
    async fn a_role_with_a_missing_alias_degrades_and_pins_nothing() {
        // An unresolvable role must be visibly `unresolved`, must issue NO warm
        // request, and must not appear in the exemption — while its siblings are
        // held normally (degrade the role, never the set).
        let set = Arc::new(ResidentSet::new(live_cfg()));
        let mut roles = three_roles();
        roles[0] = resolved(Role::Personality, None, None, false);
        let (env, _entered) = FakeEnv::new(roles, Some(96.0));

        let report = set.warm_with(&*env, "startup", true, None).await;

        assert_eq!(report.warmed, 2);
        assert_eq!(report.skipped, 1);
        let status = set.status().await;
        let personality = status
            .roles
            .iter()
            .find(|r| r.role == Role::Personality)
            .unwrap();
        assert_eq!(personality.state, RoleState::Unresolved);
        assert!(!personality.warm);
        assert_eq!(personality.model, None);
        // Nothing was warmed or exempted on its behalf.
        assert_eq!(env.warm_calls().len(), 2);
        assert!(!env.exempt().iter().any(|m| m == "voice:1"));
        assert_eq!(status.exempt.len(), 2);
    }

    #[tokio::test]
    async fn a_cancelled_leader_does_not_strand_its_coalesced_subscribers() {
        let set = Arc::new(ResidentSet::new(live_cfg()));
        let (env, mut entered) = FakeEnv::new(three_roles(), Some(96.0));
        env.gate_warms();

        let (s, e) = (set.clone(), env.clone());
        let leader = tokio::spawn(async move { s.warm_with(&*e, "startup", true, None).await });
        entered.recv().await.expect("the leader is inside its warm I/O");

        let (s1, e1) = (set.clone(), env.clone());
        let sub_a = tokio::spawn(async move { s1.warm_with(&*e1, "activate", true, None).await });
        let (s2, e2) = (set.clone(), env.clone());
        let sub_b = tokio::spawn(async move { s2.warm_with(&*e2, "reconcile", true, None).await });
        // Both subscribers are parked on the leader's broadcast, not warming.
        settle().await;

        // The mode swap cancels the leader. Nothing ever opens the gate.
        let rel = set.release_with(&*env, "harmony-idle-lease").await;

        let lead = join_unblocked(leader, "the cancelled leader").await;
        let ra = join_unblocked(sub_a, "coalesced subscriber a").await;
        let rb = join_unblocked(sub_b, "coalesced subscriber b").await;

        assert!(lead.discarded, "the cancelled leader discards its pass");
        for (name, r) in [("a", &ra), ("b", &rb)] {
            assert!(r.coalesced, "subscriber {name} must report as coalesced");
            assert!(
                r.discarded,
                "subscriber {name} must inherit the leader's discarded verdict: {r:?}"
            );
            assert!(!r.changed, "subscriber {name} changed nothing itself");
        }
        assert_eq!(
            env.warm_calls().len(),
            1,
            "only the leader ever issued a warm request: {:?}",
            env.warm_calls()
        );
        assert!(rel.drained);
        assert!(env.exempt().is_empty());
        assert!(!set.status().await.active);
    }

    /// **TRTR-07d requirement 2: the bounded wait and its documented FALLBACK.**
    ///
    /// When in-flight warm work will not settle inside
    /// `release_drain + cancel_grace` — here an env whose gated warm ignores
    /// cancellation outright — release must NOT hang (a stuck release is its own
    /// outage on a shared GPU). It must report `drain_timed_out`, issue the
    /// compensating unloads ITSELF from its pending snapshot, and return.
    #[tokio::test]
    async fn release_bounded_wait_expires_and_release_compensates_itself() {
        // Both phases expire immediately; the warm is unstoppable throughout, so
        // no report can arrive during either window. Deterministic, no sleeps.
        let cfg = ResidentSetConfig {
            rewarm_debounce: Duration::ZERO,
            release_drain: Duration::ZERO,
            cancel_grace: Duration::ZERO,
            ..Default::default()
        };
        let set = Arc::new(ResidentSet::new(cfg));
        let (env, mut entered) = FakeEnv::new(three_roles(), Some(96.0));
        env.gate_warms_uncancellable();

        let (s, e) = (set.clone(), env.clone());
        let warming = tokio::spawn(async move { s.warm_with(&*e, "startup", true, None).await });
        let stuck = entered
            .recv()
            .await
            .expect("a warm request is out and will not come back");

        let rel = set.release_with(&*env, "harmony-idle-lease").await;

        assert!(
            rel.drain_timed_out,
            "the bounded wait must EXPIRE and be reported, never silently extended: {rel:?}"
        );
        assert!(
            !rel.drained,
            "nothing was drained — the pass never settled: {rel:?}"
        );
        assert_eq!(
            rel.compensated, 1,
            "the fallback must compensate from release's own snapshot: {rel:?}"
        );
        assert_eq!(
            env.unloads(),
            vec![stuck.clone()],
            "release itself must unload the model the unsettled warm may have loaded"
        );
        assert!(env.exempt().is_empty());
        assert!(!set.status().await.active);

        // Let the stuck pass finish so the test leaves nothing running. It
        // discards (and re-issues an idempotent unload of its own).
        env.open_gate(16);
        let pass = join_unblocked(warming, "the unstoppable pass").await;
        assert!(pass.discarded, "the pass still discards: {pass:?}");
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

    // ─────────────────────────────────────────────────────────────────────────
    // TRTR-07c: PRODUCTION-SEAM tests — prove the models actually get warmed.
    //
    // Every test above drives `ResidentSet` against `FakeEnv`, so all of them
    // prove the MANAGER's behaviour: that it calls `warm_one` and then installs
    // an exemption. NONE of them prove that the production
    // [`AppStateEnv::warm_one`] causes Ollama to load and RETAIN anything. A
    // regression that made it a no-op, hit the wrong endpoint or HTTP path,
    // omitted `keep_alive` (so Ollama loads the model and then drops it on its
    // own default timer), or sent a chat-shaped request for the embedding model
    // would leave every test above GREEN while residency quietly degraded into
    // pure bookkeeping — the exemption set says the models are resident, the
    // assistant still cold-starts, and nothing detects it. The `ResidentEnv`
    // seam that made the concurrency tests deterministic is exactly what opens
    // that hole, so it has to be closed from the other side.
    //
    // These tests therefore run the REAL `AppStateEnv` (and the real public
    // `warm`/`release` entry points) against a STUB OLLAMA — `httpmock`, the
    // repo's existing HTTP test facility, already used by `session.rs`,
    // `mcp_proxy.rs`, `embeddings.rs` and `slm_router.rs` — and assert the
    // ACTUAL REQUEST CONTENT that goes out: method, path, the resolved model
    // name, and the PRESENCE AND VALUE of `keep_alive`.
    // ─────────────────────────────────────────────────────────────────────────

    /// Every request the stub Ollama received, in arrival order:
    /// `(method, path, parsed JSON body)`.
    ///
    /// A process `static` because httpmock's `MockMatcherFunction` is a plain
    /// `fn` pointer and cannot capture an environment. That is fine here: every
    /// seam test is `#[serial_test::serial]` anyway, because they all share the
    /// process-global `OLLAMA_URL` that `AppStateEnv::warm_one` reads.
    static OLLAMA_SEEN: once_cell::sync::Lazy<StdMutex<Vec<(String, String, serde_json::Value)>>> =
        once_cell::sync::Lazy::new(|| StdMutex::new(Vec::new()));

    /// httpmock matcher that RECORDS the request and always matches, so a single
    /// catch-all mock captures the exact bytes every warm actually sent.
    fn record_ollama_request(req: &httpmock::prelude::HttpMockRequest) -> bool {
        let body = req
            .body
            .as_deref()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(b).ok())
            .unwrap_or(serde_json::Value::Null);
        OLLAMA_SEEN
            .lock()
            .unwrap()
            .push((req.method.clone(), req.path.clone(), body));
        true
    }

    /// Drain and return everything the stub Ollama has seen since the last drain.
    fn seen_requests() -> Vec<(String, String, serde_json::Value)> {
        std::mem::take(&mut *OLLAMA_SEEN.lock().unwrap())
    }

    /// Placeholder model names — never a real fleet model.
    const SEAM_PERSONALITY_MODEL: &str = "test-personality-model:8b";
    const SEAM_ROUTER_MODEL: &str = "test-router-model:3b";
    const SEAM_EMBED_MODEL: &str = "test-embed-model:0.6b";

    /// A production `AppState` pointed at a stub Ollama on `base` (loopback),
    /// with all three role aliases resolving and all three models present in the
    /// registry (so the never-pulled guard does not skip them).
    async fn seam_state(base: &str) -> Arc<crate::routes::AppState> {
        std::env::set_var("OLLAMA_URL", base);
        // No VRAM counter configured ⇒ `free_vram_gb()` is `None` ⇒ the planner
        // fails SOFT and attempts every role. That keeps the plan deterministic
        // instead of depending on the build host's real free VRAM.
        std::env::remove_var("CHORD_VRAM_FREE_SYSFS_PATH");
        let state = crate::routes::tests::test_state("http://mcp.invalid:3200".to_string());
        state
            .lumina_aliases
            .set("lumina-deep", SEAM_PERSONALITY_MODEL.to_string());
        state
            .lumina_aliases
            .set("lumina-fast", SEAM_ROUTER_MODEL.to_string());
        state
            .lumina_aliases
            .set("lumina-embed", SEAM_EMBED_MODEL.to_string());
        {
            let mut reg = state.model_registry.lock().await;
            for m in [
                SEAM_PERSONALITY_MODEL,
                SEAM_ROUTER_MODEL,
                SEAM_EMBED_MODEL,
            ] {
                // 1 byte: present in the registry, and effectively free, so the
                // VRAM budget can never be what decides these tests.
                reg.register_external(m, "test", Some("/nonexistent/local".to_string()), None, 1);
            }
        }
        let _ = seen_requests();
        state
    }

    /// Config whose roles are the three real alias keys, so a full production
    /// pass resolves through `AppStateEnv::resolve`. `reassert` is left at its
    /// default (far longer than any test) — a re-warm after a RELEASE still
    /// re-issues, because release clears `warmed_at`.
    fn seam_cfg(keep_alive: &str) -> ResidentSetConfig {
        ResidentSetConfig {
            keep_alive: keep_alive.to_string(),
            rewarm_debounce: Duration::from_secs(0),
            // Personality is overridden to the DEEP alias here (the per-role env
            // override, `CHORD_RESIDENT_ROLE_PERSONALITY`, in config form) purely so
            // this harness keeps THREE DISTINCT role targets and can assert three
            // distinct Ollama requests. Production defaults personality and router to
            // the same interactive alias, which `plan_warm` holds once as `Shared`;
            // that path has its own test.
            aliases: vec![
                (Role::Personality, "lumina-deep".to_string()),
                (Role::Router, "lumina-fast".to_string()),
                (Role::Embedding, "lumina-embed".to_string()),
            ],
            ..Default::default()
        }
    }

    fn body_of<'a>(
        seen: &'a [(String, String, serde_json::Value)],
        model: &str,
    ) -> &'a (String, String, serde_json::Value) {
        seen.iter()
            .find(|(_, _, b)| b.get("model").and_then(|m| m.as_str()) == Some(model))
            .unwrap_or_else(|| {
                panic!("no warm request named model {model}; saw {seen:?}")
            })
    }

    /// **The core seam test.** All three roles must produce a real HTTP warm
    /// request, and the EMBEDDING role's request must be shaped differently from
    /// the chat-style personality/router ones (an embedding model cannot serve
    /// `/api/generate`, so a regression that sent it a chat-shaped request is a
    /// realistic silent failure).
    #[tokio::test]
    #[serial_test::serial]
    async fn production_warm_one_issues_a_role_shaped_ollama_request_per_role() {
        let server = httpmock::MockServer::start_async().await;
        let stub = server
            .mock_async(|when, then| {
                when.matches(record_ollama_request);
                then.status(200).json_body(serde_json::json!({"ok": true}));
            })
            .await;
        let state = seam_state(&server.base_url()).await;
        let env = AppStateEnv::new(&state, Duration::from_secs(10));

        assert!(
            env.warm_one(Role::Personality, SEAM_PERSONALITY_MODEL, "24h", &CancelToken::new())
                .await
                .is_ok(),
            "the personality warm must succeed against a 200 Ollama"
        );
        assert!(
            env.warm_one(Role::Router, SEAM_ROUTER_MODEL, "24h", &CancelToken::new())
                .await
                .is_ok(),
            "the router warm must succeed against a 200 Ollama"
        );
        assert!(
            env.warm_one(Role::Embedding, SEAM_EMBED_MODEL, "24h", &CancelToken::new())
                .await
                .is_ok(),
            "the embedding warm must succeed against a 200 Ollama"
        );

        assert_eq!(
            stub.hits_async().await,
            3,
            "three roles ⇒ three real HTTP warm requests; a no-op `warm_one` sends none"
        );
        let seen = seen_requests();
        assert_eq!(
            seen.len(),
            3,
            "every role must actually put a request on the wire — otherwise residency is pure bookkeeping. saw: {seen:?}"
        );

        // ── personality: chat-style load on /api/generate ────────────────────
        let (method, path, body) = body_of(&seen, SEAM_PERSONALITY_MODEL);
        assert_eq!(method, "POST", "the personality warm must be a POST");
        assert_eq!(
            path, "/api/generate",
            "the personality role loads through Ollama's generate endpoint"
        );
        assert_eq!(
            body["model"],
            serde_json::json!(SEAM_PERSONALITY_MODEL),
            "the request must name the RESOLVED personality model"
        );
        assert_eq!(
            body.get("keep_alive"),
            Some(&serde_json::json!("24h")),
            "keep_alive must be PRESENT and carry the configured value — without it Ollama loads the model and then drops it on its own default timer, which is exactly the silent no-op residency cannot survive"
        );
        assert!(
            body.get("input").is_none(),
            "a chat-style warm must not carry an embedding `input` field: {body}"
        );

        // ── router: same chat-style shape, its own model ─────────────────────
        let (method, path, body) = body_of(&seen, SEAM_ROUTER_MODEL);
        assert_eq!(method, "POST", "the router warm must be a POST");
        assert_eq!(
            path, "/api/generate",
            "the router role loads through Ollama's generate endpoint"
        );
        assert_eq!(
            body["model"],
            serde_json::json!(SEAM_ROUTER_MODEL),
            "the request must name the RESOLVED router model"
        );
        assert_eq!(
            body.get("keep_alive"),
            Some(&serde_json::json!("24h")),
            "the router warm must carry keep_alive"
        );

        // ── embedding: DIFFERENT call shape ──────────────────────────────────
        let (method, path, body) = body_of(&seen, SEAM_EMBED_MODEL);
        assert_eq!(method, "POST", "the embedding warm must be a POST");
        assert_eq!(
            path, "/api/embed",
            "an embedding model cannot serve /api/generate — it must be loaded through the embed endpoint"
        );
        assert_ne!(
            path, "/api/generate",
            "sending the chat-shaped warm for the embedding role is a realistic regression and must be caught"
        );
        assert_eq!(
            body["model"],
            serde_json::json!(SEAM_EMBED_MODEL),
            "the request must name the RESOLVED embedding model"
        );
        assert_eq!(
            body.get("keep_alive"),
            Some(&serde_json::json!("24h")),
            "the embedding warm must carry keep_alive too"
        );
        assert!(
            body.get("input").is_some(),
            "an /api/embed warm must carry the `input` field the endpoint requires: {body}"
        );

        // No two roles may collapse onto the same request shape+model.
        let paths: HashSet<&str> = seen.iter().map(|(_, p, _)| p.as_str()).collect();
        assert_eq!(
            paths.len(),
            2,
            "exactly two distinct endpoints are expected (generate for chat roles, embed for the embedding role); saw {paths:?}"
        );
    }

    /// `keep_alive` is THREADED from config, not a literal: a non-default value
    /// must appear verbatim in all three requests. This is the assertion that
    /// bites when someone drops the field or hardcodes a short default.
    #[tokio::test]
    #[serial_test::serial]
    async fn production_warm_one_threads_the_configured_long_keep_alive() {
        let server = httpmock::MockServer::start_async().await;
        let _stub = server
            .mock_async(|when, then| {
                when.matches(record_ollama_request);
                then.status(200).json_body(serde_json::json!({"ok": true}));
            })
            .await;
        let state = seam_state(&server.base_url()).await;
        let env = AppStateEnv::new(&state, Duration::from_secs(10));

        // Deliberately NOT the default "24h", so a hardcoded literal fails.
        for (role, model) in [
            (Role::Personality, SEAM_PERSONALITY_MODEL),
            (Role::Router, SEAM_ROUTER_MODEL),
            (Role::Embedding, SEAM_EMBED_MODEL),
        ] {
            assert!(env.warm_one(role, model, "17h", &CancelToken::new()).await.is_ok());
        }

        let seen = seen_requests();
        assert_eq!(seen.len(), 3);
        for (_, path, body) in &seen {
            assert_eq!(
                body.get("keep_alive"),
                Some(&serde_json::json!("17h")),
                "the CONFIGURED keep_alive must reach the wire for {path} — a missing or hardcoded value means the model is loaded and then silently dropped: {body}"
            );
        }
    }

    /// A non-2xx response is surfaced as a warm FAILURE for that role (and the
    /// error message carries no infrastructure detail, per S77).
    #[tokio::test]
    #[serial_test::serial]
    async fn production_warm_one_surfaces_a_non_2xx_as_a_warm_failure() {
        let server = httpmock::MockServer::start_async().await;
        let _stub = server
            .mock_async(|when, then| {
                when.matches(record_ollama_request);
                then.status(503).body("upstream unavailable");
            })
            .await;
        let state = seam_state(&server.base_url()).await;
        let env = AppStateEnv::new(&state, Duration::from_secs(10));

        let err = env
            .warm_one(Role::Personality, SEAM_PERSONALITY_MODEL, "24h", &CancelToken::new())
            .await
            .expect_err("a 503 must NOT be reported as a successful warm");
        assert!(
            err.contains("503"),
            "the failure must name the rejecting status: {err}"
        );
        assert!(
            !err.contains(&server.base_url()),
            "S77: the genericized error must not leak the endpoint: {err}"
        );
        assert_eq!(
            seen_requests().len(),
            1,
            "the request was still issued — it was the RESPONSE that failed"
        );
    }

    /// A connection error (nothing listening) is a warm failure, not a success.
    #[tokio::test]
    #[serial_test::serial]
    async fn production_warm_one_surfaces_a_connection_error_as_a_warm_failure() {
        // Loopback port 1: nothing listens there. No real infrastructure.
        let state = seam_state("http://127.0.0.1:1").await;
        let env = AppStateEnv::new(&state, Duration::from_secs(5));

        let err = env
            .warm_one(Role::Router, SEAM_ROUTER_MODEL, "24h", &CancelToken::new())
            .await
            .expect_err("an unreachable Ollama must NOT be reported as a successful warm");
        assert!(
            !err.contains("127.0.0.1"),
            "S77: the genericized error must not leak the endpoint: {err}"
        );
    }

    /// The FULL production wiring: a failing Ollama must land every role in
    /// `WarmFailed` with NO eviction exemption installed — a role that never
    /// actually loaded must never be reported warm.
    #[tokio::test]
    #[serial_test::serial]
    async fn production_warm_pass_reports_failed_roles_as_warm_failed_and_exempts_nothing() {
        let server = httpmock::MockServer::start_async().await;
        let _stub = server
            .mock_async(|when, then| {
                when.matches(record_ollama_request);
                then.status(500).body("boom");
            })
            .await;
        let state = seam_state(&server.base_url()).await;
        let set = ResidentSet::new(seam_cfg("24h"));

        let report = set.warm(&state, "seam-failure", true).await;

        assert_eq!(
            report.warmed, 0,
            "no role loaded, so none may be counted warm"
        );
        assert_eq!(report.failed, 3, "all three roles must be reported failed");
        assert_eq!(
            seen_requests().len(),
            3,
            "all three warm requests were genuinely attempted"
        );
        let status = set.status().await;
        assert!(
            status.roles.iter().all(|r| r.state == RoleState::WarmFailed),
            "every role must land in warm-failed: {:?}",
            status.roles
        );
        assert!(
            status.exempt.is_empty(),
            "a model that never loaded must NOT be pinned into the eviction exemption: {:?}",
            status.exempt
        );
        assert!(
            state
                .model_registry
                .lock()
                .await
                .residency_exempt()
                .is_empty(),
            "the REGISTRY exemption must be empty too"
        );
    }

    /// **Release/re-warm through the production wiring is genuinely
    /// re-driveable.** After a mode-swap release, a subsequent warm must issue
    /// the requests AGAIN — the production side must not be latched (an
    /// "already warmed, skip" regression would leave the fleet cold after every
    /// Harmony/MINT idle lease and no existing test would notice).
    #[tokio::test]
    #[serial_test::serial]
    async fn production_release_then_rewarm_reissues_the_ollama_requests() {
        let server = httpmock::MockServer::start_async().await;
        let stub = server
            .mock_async(|when, then| {
                when.matches(record_ollama_request);
                then.status(200).json_body(serde_json::json!({"ok": true}));
            })
            .await;
        let state = seam_state(&server.base_url()).await;
        let set = ResidentSet::new(seam_cfg("24h"));

        // ── first warm ───────────────────────────────────────────────────────
        let first = set.warm(&state, "startup", true).await;
        assert_eq!(first.warmed, 3, "the startup pass must warm all three roles");
        let first_seen = seen_requests();
        assert_eq!(first_seen.len(), 3, "three real requests on the wire");
        let mut first_models: Vec<String> = first_seen
            .iter()
            .map(|(_, _, b)| b["model"].as_str().unwrap_or_default().to_string())
            .collect();
        first_models.sort();
        assert_eq!(
            first_models,
            vec![
                SEAM_EMBED_MODEL.to_string(),
                SEAM_PERSONALITY_MODEL.to_string(),
                SEAM_ROUTER_MODEL.to_string(),
            ],
            "each resolved role model must be warmed exactly once"
        );
        let mut exempt = state.model_registry.lock().await.residency_exempt();
        exempt.sort();
        assert_eq!(
            exempt,
            vec![
                SEAM_EMBED_MODEL.to_string(),
                SEAM_PERSONALITY_MODEL.to_string(),
                SEAM_ROUTER_MODEL.to_string(),
            ],
            "the registry exemption must hold exactly the warmed models"
        );

        // ── release (mode swap) ──────────────────────────────────────────────
        let rel = set.release(&state, "harmony-idle-lease").await;
        assert!(rel.changed);
        assert_eq!(rel.released, 3);
        assert!(
            state
                .model_registry
                .lock()
                .await
                .residency_exempt()
                .is_empty(),
            "release must clear the REGISTRY exemption so the idle path can reclaim the VRAM"
        );
        assert!(
            seen_requests().is_empty(),
            "a release issues no Ollama warm requests"
        );

        // ── re-warm ──────────────────────────────────────────────────────────
        let again = set.rewarm(&state, "activate").await;
        assert_eq!(
            again.warmed, 3,
            "a post-release re-warm must genuinely re-warm, not report a latched success"
        );
        assert_eq!(again.retained, 0, "nothing may be retained across a release");
        let second_seen = seen_requests();
        assert_eq!(
            second_seen.len(),
            3,
            "the production side must RE-ISSUE the warm requests after a release; saw {second_seen:?}"
        );
        for (_, _, body) in &second_seen {
            assert_eq!(
                body.get("keep_alive"),
                Some(&serde_json::json!("24h")),
                "the re-warm must carry keep_alive as well: {body}"
            );
        }
        assert_eq!(
            stub.hits_async().await,
            6,
            "3 warm + 3 re-warm requests reached the stub Ollama"
        );
        let mut exempt = state.model_registry.lock().await.residency_exempt();
        exempt.sort();
        assert_eq!(exempt.len(), 3, "the exemption is restored by the re-warm");
        assert!(set.status().await.active);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // TRTR-07d: the POST-RELEASE LOAD. The generation guard is a bookkeeping
    // guarantee; these tests are about the GPU.
    //
    // None of the tests above can catch the residual race, and it is worth being
    // precise about why: `FakeEnv::warm_one` has no model-loading side effect at
    // all, and every httpmock seam test only ever releases AFTER its warm
    // requests have already completed. The real failure needs a warm request that
    // is STILL ON THE WIRE when the release lands and that then LOADS THE MODEL
    // ANYWAY — Ollama does not abandon a load because the client hung up.
    //
    // So the stub below is not httpmock: it is a loopback HTTP server whose
    // responses can be HELD OPEN, and which marks a model LOADED when its work
    // finishes whether or not the client is still there. That is the whole point.
    // Interleaving is driven by opening the gate, never by sleeping.
    // ─────────────────────────────────────────────────────────────────────────

    #[derive(Default)]
    struct HeldOllamaState {
        /// Models the stub considers resident in "VRAM". A warm marks its model
        /// loaded when the held request finally completes — EVEN IF the client
        /// already disconnected. A `keep_alive: 0` request unloads it.
        loaded: StdMutex<HashSet<String>>,
        /// Every request seen: `(path, model, keep_alive)`.
        seen: StdMutex<Vec<(String, String, serde_json::Value)>>,
    }

    /// A stub Ollama on loopback whose warm responses are gated.
    struct HeldOllama {
        base: String,
        state: Arc<HeldOllamaState>,
        gate: Arc<Semaphore>,
    }

    impl HeldOllama {
        /// Let `n` more held warm requests complete (and therefore load).
        fn open(&self, n: usize) {
            self.gate.add_permits(n);
        }
        fn loaded(&self) -> Vec<String> {
            let mut v: Vec<String> = self.state.loaded.lock().unwrap().iter().cloned().collect();
            v.sort();
            v
        }
        fn requests(&self) -> Vec<(String, String, serde_json::Value)> {
            self.state.seen.lock().unwrap().clone()
        }
        /// Unload requests (`keep_alive: 0`) seen, by model.
        fn unloaded_models(&self) -> Vec<String> {
            let mut v: Vec<String> = self
                .requests()
                .into_iter()
                .filter(|(_, _, ka)| *ka == serde_json::json!(0))
                .map(|(_, m, _)| m)
                .collect();
            v.sort();
            v.dedup();
            v
        }
    }

    fn headers_end(buf: &[u8]) -> Option<usize> {
        buf.windows(4).position(|w| w == b"\r\n\r\n").map(|i| i + 4)
    }

    async fn start_held_ollama() -> (HeldOllama, mpsc::UnboundedReceiver<String>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        // Loopback only, ephemeral port — never a real fleet address.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let state = Arc::new(HeldOllamaState::default());
        let gate = Arc::new(Semaphore::new(0));
        let (tx, rx) = mpsc::unbounded_channel();

        {
            let state = state.clone();
            let gate = gate.clone();
            tokio::spawn(async move {
                loop {
                    let Ok((mut sock, _)) = listener.accept().await else {
                        return;
                    };
                    let state = state.clone();
                    let gate = gate.clone();
                    let tx = tx.clone();
                    tokio::spawn(async move {
                        let mut buf: Vec<u8> = Vec::new();
                        let mut tmp = [0u8; 2048];
                        let (path, body) = loop {
                            let n = match sock.read(&mut tmp).await {
                                Ok(0) | Err(_) => return,
                                Ok(n) => n,
                            };
                            buf.extend_from_slice(&tmp[..n]);
                            let Some(end) = headers_end(&buf) else { continue };
                            let head = String::from_utf8_lossy(&buf[..end]).to_string();
                            let mut len = 0usize;
                            for line in head.lines() {
                                let lower = line.to_ascii_lowercase();
                                if let Some(v) = lower.strip_prefix("content-length:") {
                                    len = v.trim().parse().unwrap_or(0);
                                }
                            }
                            if buf.len() < end + len {
                                continue;
                            }
                            let path = head
                                .lines()
                                .next()
                                .and_then(|l| l.split_whitespace().nth(1))
                                .unwrap_or("/")
                                .to_string();
                            let body: serde_json::Value =
                                serde_json::from_slice(&buf[end..end + len])
                                    .unwrap_or(serde_json::Value::Null);
                            break (path, body);
                        };

                        let model = body
                            .get("model")
                            .and_then(|m| m.as_str())
                            .unwrap_or_default()
                            .to_string();
                        let keep_alive = body
                            .get("keep_alive")
                            .cloned()
                            .unwrap_or(serde_json::Value::Null);
                        state
                            .seen
                            .lock()
                            .unwrap()
                            .push((path, model.clone(), keep_alive.clone()));
                        let _ = tx.send(model.clone());

                        if keep_alive == serde_json::json!(0) {
                            // The compensating unload is never gated — it is the
                            // undo, and it has to be able to land.
                            state.loaded.lock().unwrap().remove(&model);
                        } else {
                            // HELD. When the test lets it through, the load
                            // completes SERVER-SIDE regardless of whether Chord is
                            // still waiting for the response. This is the exact
                            // behaviour that makes "release returned" insufficient.
                            if let Ok(p) = gate.acquire().await {
                                p.forget();
                            }
                            state.loaded.lock().unwrap().insert(model.clone());
                        }

                        let _ = sock
                            .write_all(
                                b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 11\r\nconnection: close\r\n\r\n{\"ok\":true}",
                            )
                            .await;
                        let _ = sock.flush().await;
                    });
                }
            });
        }

        (
            HeldOllama {
                base: format!("http://{addr}"),
                state,
                gate,
            },
            rx,
        )
    }

    /// Seam config with explicit release bounds.
    fn seam_cfg_bounds(drain: Duration, grace: Duration) -> ResidentSetConfig {
        ResidentSetConfig {
            release_drain: drain,
            cancel_grace: grace,
            // Same THREE-DISTINCT-TARGETS harness as `seam_cfg` (see its comment):
            // personality is overridden to the deep alias so these lifecycle tests
            // can still observe three separate in-flight warms.
            ..seam_cfg("24h")
        }
    }

    /// **THE MISSING TEST.** A warm request that completes only AFTER the release.
    ///
    /// Two roles are already loaded and a third is still on the wire when the mode
    /// swap lands. The generation guard alone would leave all three sitting in
    /// VRAM with `keep_alive=24h` while Chord reported "released" — the GPU
    /// starved for a day. What must happen instead: release WAITS for the pass
    /// (bounded), the pass discards itself AND unloads every model it loaded, and
    /// only then does release return, before the idle path's VRAM unload runs.
    #[tokio::test]
    #[serial_test::serial]
    async fn production_release_undoes_warms_that_complete_after_it() {
        let (ollama, mut entered) = start_held_ollama().await;
        let state = seam_state(&ollama.base).await;
        let set = Arc::new(ResidentSet::new(seam_cfg_bounds(
            Duration::from_secs(30),
            Duration::from_secs(5),
        )));

        let (s, st) = (set.clone(), state.clone());
        let warming = tokio::spawn(async move { s.warm(&st, "startup", true).await });

        // Role 1 reaches the stub and is held there; let it through so it is
        // genuinely LOADED, then let role 2 reach the stub and hold it.
        assert_eq!(
            entered.recv().await.as_deref(),
            Some(SEAM_PERSONALITY_MODEL),
            "the personality role warms first (declaration order is the priority)"
        );
        ollama.open(1);
        assert_eq!(
            entered.recv().await.as_deref(),
            Some(SEAM_ROUTER_MODEL),
            "the router role is now the one on the wire"
        );

        // ── the mode swap lands with a warm STILL IN FLIGHT ──────────────────
        let (s, st) = (set.clone(), state.clone());
        let releasing = tokio::spawn(async move { s.release(&st, "harmony-idle-lease").await });
        settle().await;

        // Ollama does what Ollama does: it finishes the loads anyway.
        ollama.open(8);

        let rel = releasing.await.unwrap();

        // ── the guarantee, at the GPU layer, AT THE MOMENT RELEASE RETURNS ───
        assert!(
            rel.drained,
            "release must have waited for the in-flight pass before returning: {rel:?}"
        );
        assert!(!rel.drain_timed_out, "the bounded wait was ample here: {rel:?}");
        assert_eq!(
            ollama.loaded(),
            Vec::<String>::new(),
            "NOTHING may still be loaded once release has returned — got {:?} (requests: {:?})",
            ollama.loaded(),
            ollama.requests()
        );
        assert_eq!(
            rel.compensated, 3,
            "every model the cancelled pass loaded must have been explicitly unloaded: {rel:?}"
        );
        let mut expected = vec![
            SEAM_EMBED_MODEL.to_string(),
            SEAM_PERSONALITY_MODEL.to_string(),
            SEAM_ROUTER_MODEL.to_string(),
        ];
        expected.sort();
        assert_eq!(
            ollama.unloaded_models(),
            expected,
            "each role's compensating unload must go out: {:?}",
            ollama.requests()
        );
        // The embedding unload must use the EMBED endpoint — a `/api/generate`
        // unload for an embedding model silently misses.
        let embed_unload = ollama
            .requests()
            .into_iter()
            .find(|(_, m, ka)| m == SEAM_EMBED_MODEL && *ka == serde_json::json!(0))
            .expect("an unload for the embedding model");
        assert_eq!(embed_unload.0, "/api/embed");

        assert!(
            state
                .model_registry
                .lock()
                .await
                .residency_exempt()
                .is_empty(),
            "no exemption may survive the release"
        );
        assert!(!set.status().await.active);

        let pass = warming.await.unwrap();
        assert!(pass.discarded, "the pass itself must report as discarded");
        assert_eq!(pass.compensated, 3);
    }

    /// **Production cancellation is real, not decorative.** With a held-open
    /// warm that Ollama will never answer, release escalates to cancelling the
    /// token — and [`AppStateEnv::warm_one`] must ABORT the HTTP request so the
    /// pass settles and compensates INSIDE the bounded wait. If it merely ignored
    /// the token the warm would sit there for its full 300s timeout, the grace
    /// would expire, and release would have to fall back — which is exactly what
    /// this asserts does NOT happen.
    #[tokio::test]
    #[serial_test::serial]
    async fn production_cancellation_aborts_the_request_so_release_drains_in_time() {
        let (ollama, mut entered) = start_held_ollama().await;
        let state = seam_state(&ollama.base).await;
        // Graceful phase expires at once (the stub will never answer, so no report
        // can arrive during it); the grace is generous and is only ever consumed
        // if cancellation does nothing.
        let set = Arc::new(ResidentSet::new(seam_cfg_bounds(
            Duration::ZERO,
            Duration::from_secs(5),
        )));

        let (s, st) = (set.clone(), state.clone());
        let warming = tokio::spawn(async move { s.warm(&st, "startup", true).await });
        assert_eq!(
            entered.recv().await.as_deref(),
            Some(SEAM_PERSONALITY_MODEL),
            "a warm is on the wire and the stub will never answer it"
        );

        // The gate is NEVER opened. Only cancellation can free this request.
        let rel = set.release(&state, "mint-sweep").await;

        assert!(
            rel.drained,
            "cancellation must free the in-flight warm so release drains: {rel:?}"
        );
        assert!(
            !rel.drain_timed_out,
            "the pass settled well inside the grace — no fallback needed: {rel:?}"
        );
        assert_eq!(
            rel.compensated, 1,
            "the one request that went out must still be compensated: {rel:?}"
        );
        assert!(
            ollama
                .unloaded_models()
                .contains(&SEAM_PERSONALITY_MODEL.to_string()),
            "the compensating unload must reach Ollama: {:?}",
            ollama.requests()
        );
        assert!(
            !ollama
                .requests()
                .iter()
                .any(|(_, m, ka)| m == SEAM_ROUTER_MODEL && *ka != serde_json::json!(0)),
            "a cancelled pass must not go on to issue the REMAINING roles' warms: {:?}",
            ollama.requests()
        );
        assert!(
            state
                .model_registry
                .lock()
                .await
                .residency_exempt()
                .is_empty()
        );

        let pass = join_unblocked(warming, "the cancelled warm pass").await;
        assert!(pass.discarded, "the cancelled pass discards: {pass:?}");
        assert!(pass.cancelled >= 1, "the remaining roles were skipped: {pass:?}");
    }

    /// **POSITIVE CONTROL at the production seam.** With no release anywhere near
    /// it, an ordinary warm must still load all three models and install the
    /// exemption — and must issue NO unload. Without this, every assertion above
    /// could be satisfied by a change that simply broke residency.
    #[tokio::test]
    #[serial_test::serial]
    async fn production_ordinary_warm_loads_all_three_and_installs_the_exemption() {
        let (ollama, _entered) = start_held_ollama().await;
        // Nothing is held: every warm completes immediately.
        ollama.open(64);
        let state = seam_state(&ollama.base).await;
        let set = ResidentSet::new(seam_cfg_bounds(
            Duration::from_secs(30),
            Duration::from_secs(5),
        ));

        let report = set.warm(&state, "startup", true).await;

        assert_eq!(report.warmed, 3, "all three roles warm: {report:?}");
        assert!(!report.discarded);
        assert_eq!(report.compensated, 0, "nothing to compensate: {report:?}");
        assert_eq!(report.cancelled, 0);
        let mut expected = vec![
            SEAM_EMBED_MODEL.to_string(),
            SEAM_PERSONALITY_MODEL.to_string(),
            SEAM_ROUTER_MODEL.to_string(),
        ];
        expected.sort();
        assert_eq!(
            ollama.loaded(),
            expected,
            "all three models must actually be loaded: {:?}",
            ollama.requests()
        );
        assert!(
            ollama.unloaded_models().is_empty(),
            "an ordinary warm must never unload anything: {:?}",
            ollama.requests()
        );
        for (_, _, ka) in ollama.requests() {
            assert_eq!(ka, serde_json::json!("24h"), "keep_alive must be threaded");
        }
        let mut exempt = state.model_registry.lock().await.residency_exempt();
        exempt.sort();
        assert_eq!(exempt, expected, "the registry exemption must be installed");
        assert!(set.status().await.active);
    }
}
