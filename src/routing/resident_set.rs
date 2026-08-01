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
//! → `lumina-embed`), overridable per role by env. The alias is resolved through
//! the dynamic [`LuminaAliasStore`] first and the static `CHORD_MODEL_ALIASES` map
//! second. The resident set NEVER hard-wires a model name, because Chord owns model
//! selection (north-star Module Contract clause 1) and the dynamic alias updater
//! must stay free to repoint a role under us.
//!
//! ## An unresolved alias degrades LOUDLY and pins NOTHING — uniformly, every role
//! A role whose alias resolves to nothing is [`RoleState::Unresolved`]: it is
//! WARNED about by name (role, the alias that failed, and what to set — see
//! [`unresolved_alias_remedy`]) and then holds nothing. There is no per-role
//! fallback, and deliberately so. A fallback that quietly substitutes a DIFFERENT
//! model than the operator configured re-creates the exact defect CHRD-PIN-01
//! exists to remove: a second, implicit source of residency truth. If a role
//! should hold a model, that belongs in config, where it is visible — set the
//! alias (`CHORD_MODEL_ALIASES`) or repoint the role
//! (`CHORD_RESIDENT_ROLE_*`), not in a code fallback.
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
use std::sync::atomic::{AtomicU8, Ordering as AtomicOrdering};
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
            // Engram memory. Like every other role this is an ALIAS KEY and gets no
            // code-level fallback: if `lumina-embed` is not configured, the role
            // warns and holds nothing. The operator sets the alias (normally to the
            // same model `EMBED_LOCAL_MODEL` names, so `/v1/embeddings` and the
            // resident set cannot disagree about which model Engram's vectors come
            // from) — visible config, not an implicit second source of truth.
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

/// The ACTIONABLE half of the unresolved-alias warning: what the operator should
/// actually do, named concretely for THIS role and THIS alias key.
///
/// Pure and public so the wording is a tested artifact rather than an incidental
/// string inside a macro. Every role gets the same treatment — there is no role
/// with a code-level fallback, so there is no role that degrades quietly.
pub fn unresolved_alias_remedy(role: Role, alias: &str) -> String {
    format!(
        "point the Chord alias '{alias}' at a pulled model (add it to CHORD_MODEL_ALIASES, \
         or let the dynamic lumina alias updater own it), or repoint this role with {env}; \
         until then the {id} role holds nothing and Chord pins nothing for it",
        alias = alias,
        env = role.alias_env(),
        id = role.id(),
    )
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
/// (`0`), and both hit the SAME endpoint for a given role, so the undo is always
/// the exact inverse of the load.
///
/// **Measured, not assumed (verified against the live Ollama 0.22.1, 2026-07-31).**
/// An embedding model is NOT rejected by `/api/generate`: `POST /api/generate` with
/// `{"model":"<an embedding model>","prompt":"","keep_alive":0}` returns **200** and
/// genuinely unloads the model (`/api/ps` drops it). `POST /api/embed` and the older
/// `POST /api/embeddings` with a `keep_alive` also return 200. So on 0.22.1 the
/// generate endpoint would work for every role.
///
/// The role shaping is kept anyway, as DEFENSIVE PORTABILITY, not a requirement:
/// older Ollama builds do answer `/api/generate` for an embedder with a 400
/// `"does not support generate"` (which is why
/// [`crate::gpu_exclusive::is_generate_unsupported_rejection`] exists), addressing a
/// model through the endpoint that actually serves it is the better-defined call,
/// and it costs nothing. Do NOT "simplify" this by asserting the 400 behaviour as
/// current fact — it is version-dependent and false on 0.22.1.
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
        let targets: Vec<(Role, String, Option<String>)> = aliases
            .iter()
            .map(|(role, alias)| {
                let model = resolve_alias(alias, &state.lumina_aliases, &state.model_aliases);
                if model.is_none() {
                    // ONE code path, EVERY role: an alias that resolves to nothing
                    // degrades LOUDLY and pins nothing. No role has a fallback, so
                    // no role degrades silently — the warning names the role, the
                    // alias that failed, and the fix.
                    warn!(
                        role = role.id(),
                        alias = %alias,
                        remedy = %unresolved_alias_remedy(*role, alias),
                        "resident-set: role alias resolves to NO target — this role is DEGRADED and holds nothing"
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
    /// Role-shaped: an embedding model is loaded through `/api/embed`, everything
    /// else through `/api/generate` with no prompt — Ollama's documented "load this
    /// model and hold it for keep_alive" call. The shaping is defensive
    /// portability, not a hard requirement: on Ollama 0.22.1 `/api/generate` accepts
    /// an embedding model too (measured — see [`ollama_role_request`]). Never fatal:
    /// every failure is a genericized `Err` string (S77 — no infrastructure in the
    /// message) the caller logs and degrades on.
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
    /// CHRD-83: OUTGOING models this pass unloaded because a role's target
    /// CHANGED and no remaining role held the old one. An alias repoint used to
    /// drop the outgoing model's residency exemption without unloading it, so it
    /// sat in VRAM at full size, held by nobody, until its bounded `keep_alive`
    /// expired (24h by default). Bounded is not free on a shared GPU.
    pub orphaned: usize,
    /// CHRD-83, GUARD 5: role slots whose residency claim was REPAIRED because
    /// the model they claimed had just been unloaded by this pass. The orphan
    /// decision is taken under the lock and the unload is issued outside it, so
    /// the decision can go stale in that window; rather than leave the set
    /// believing it holds a model that is no longer in VRAM — precisely the
    /// bookkeeping/GPU divergence TRTR-07d exists to eliminate — the slot is
    /// marked not-resident so the next pass re-warms it. Expected to be 0 on
    /// every steady-state tick.
    pub repaired: usize,
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
                            // Same posture as the two branches above: a role that is
                            // not held is WARNED about, never whispered. The
                            // resolution step already emitted the actionable remedy
                            // (`unresolved_alias_remedy`); this records the outcome
                            // of the pass itself.
                            warn!(
                                role = role.id(),
                                alias = %r.alias,
                                state = ?st,
                                "resident-set: role holds nothing this pass — running degraded"
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
        // CHRD-83: models a role is MOVING OFF. Deliberately narrow — see the
        // filter below the loop for why.
        let mut outgoing: Vec<(Role, String)> = Vec::new();

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

            // CHRD-83: this role was holding a model and is now holding a
            // DIFFERENT one. Note how narrow the condition is, on purpose:
            //
            //   * the role must have been held BEFORE (`p.state.is_held()`), so
            //     the outgoing model really is one this set loaded and owned;
            //   * the role must be held NOW on a different model — i.e. the new
            //     target is already confirmed warm. Doing this only on a
            //     SUCCESSFUL takeover is what keeps us from ever fighting an
            //     in-flight warm for the new target, and means a role that merely
            //     failed to resolve or failed to warm falls back to exactly the
            //     old behaviour (a bounded keep_alive expiry), never worse.
            //
            // A role that goes held → not-held is therefore NOT an orphan here:
            // that is the release/mode-swap path, which has its own compensating
            // unload and its own generation ownership.
            if state.is_held() {
                if let (Some(p), Some(new_model)) = (prev.as_ref(), out.model.as_ref()) {
                    if p.state.is_held() {
                        if let Some(old_model) = p.model.as_ref() {
                            if old_model != new_model {
                                outgoing.push((out.role, old_model.clone()));
                            }
                        }
                    }
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

        // CHRD-83: dropping the exemption is only the BOOKKEEPING half of a
        // repoint — exactly the same asymmetry TRTR-07d found on the release
        // path. The outgoing model keeps the 24h `keep_alive` this set gave it,
        // so it stays in VRAM at full size held by NOBODY until that expires.
        // Bounded, but the promoter repoints on its own schedule with no human
        // involved, so two large repoints can occupy most of a card for a day.
        //
        // DELIBERATE NON-FEATURE — we do NOT also shorten the outgoing model's
        // keep_alive as a belt-and-braces backstop. It looks free and is not:
        // shortening means issuing the SAME role-shaped request to the SAME
        // endpoint with a small `keep_alive` instead of `0`, so it fails in
        // exactly the circumstances the unload fails (Ollama unreachable or
        // refusing) — it buys nothing against the correlated failure it is meant
        // to hedge. Worse, it is not idempotent the way an unload is: if the
        // model has already left VRAM (the unload landed, or the eviction sweep
        // got there first), a keep_alive request RE-LOADS it, so a hedge against
        // a rare failed unload would routinely reload a 24 GB model for its own
        // short timer. The bounded expiry the model already carries is the
        // backstop, and it is the one we are strictly improving on.
        //
        // GUARD 1 — a shared target must survive one role moving off it.
        // personality and router share an alias in the live configuration, so
        // this filter is the one that matters most: only unload a model that NO
        // remaining role holds.
        let orphans: Vec<(Role, String)> = {
            let mut seen: HashSet<String> = HashSet::new();
            outgoing
                .into_iter()
                .filter(|(_, m)| !held.iter().any(|h| h == m))
                .filter(|(_, m)| seen.insert(m.clone()))
                .collect()
        };

        inner.slots = new_slots;
        inner.active = true;
        inner.last_warm = Some(Instant::now());
        // The pass committed: nothing is owed a compensating unload.
        inner.pending.clear();

        if orphans.is_empty() {
            if let Some(tx) = inner.in_flight.take() {
                let _ = tx.send(report.clone());
            }
        } else {
            // NOTE: `in_flight` is deliberately NOT taken yet — the same trick the
            // discard path uses. Holding it keeps any concurrent caller COALESCED
            // onto this pass, so no NEW warm can start and race these unloads.
            // GUARD 3, first half.
            drop(inner);

            for (role, model) in &orphans {
                // GUARD 2 — re-check immediately before each unload, per model,
                // exactly as `converge_legacy_pins` does. Between the decision
                // above and this await a warm could have made the model held
                // again. GUARD 3, second half — a newer generation means a
                // release owns the decision now, and it has its own compensation;
                // stop rather than issue an unload against a superseded pass.
                //
                // This check is necessary but NOT sufficient: it is taken under
                // the lock and the lock is dropped to issue the request, so the
                // decision can still go stale in the window between the two.
                // GUARD 5 below closes that window.
                {
                    let inner = self.inner.lock().await;
                    if inner.generation != generation {
                        info!(
                            trigger,
                            pass_generation = generation,
                            current_generation = inner.generation,
                            model = %model,
                            "resident-set: orphan unload skipped — a mode-swap release superseded this pass and owns the decision now"
                        );
                        break;
                    }
                    if inner
                        .slots
                        .iter()
                        .any(|(_, s)| s.state.is_held() && s.model.as_deref() == Some(model.as_str()))
                    {
                        info!(
                            trigger,
                            role = role.id(),
                            model = %model,
                            "resident-set: orphan unload skipped — the outgoing model is held again"
                        );
                        continue;
                    }
                }
                // GUARD 4 — soft. A failed unload costs us the OLD behaviour (the
                // model still has a bounded expiry), never a wedged reconcile.
                let outcome = env.unload_one(*role, model).await;

                // GUARD 5 — REVALIDATE AND REPAIR. The check above was taken
                // under the lock and then the lock was DROPPED to issue the
                // request, so the decision is only known-good at the moment it
                // was taken, not at the moment it lands. Re-validate now, when
                // the request has actually landed, and repair rather than
                // assume: see `repair_stale_residency` for why this is the shape
                // of the fix and not "hold the lock across the unload".
                report.repaired += self.repair_stale_residency(env, generation, model).await;

                match outcome {
                    Ok(()) => {
                        report.orphaned += 1;
                        info!(
                            trigger,
                            role = role.id(),
                            model = %model,
                            "resident-set: outgoing model unloaded — the role repointed and no remaining role holds it"
                        );
                    }
                    Err(reason) => warn!(
                        trigger,
                        role = role.id(),
                        model = %model,
                        reason = %reason,
                        "resident-set: outgoing model could not be unloaded — it keeps its bounded keep_alive and the eviction sweep is the backstop"
                    ),
                }
            }

            let mut inner = self.inner.lock().await;
            if let Some(tx) = inner.in_flight.take() {
                let _ = tx.send(report.clone());
            }
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
            orphaned = report.orphaned,
            repaired = report.repaired,
            held = held.len(),
            "resident-set: warm pass complete"
        );
        report
    }

    /// CHRD-83 GUARD 5 — make an orphan unload's decision valid at the moment the
    /// request LANDED, not merely at the moment it was taken.
    ///
    /// The orphan loop takes its decision under `inner` and then DROPS the lock to
    /// issue the unload. That window is real: a `release` can bump the generation
    /// in it, and a slot can come to claim the very model being unloaded. Retaining
    /// `in_flight` across the loop (GUARD 3) coalesces new warm PASSES, but it is a
    /// guarantee about one caller path, not an invariant of the data structure — so
    /// the set claiming residency for a model it just unloaded rests on an argument
    /// rather than on a check. This makes it a check.
    ///
    /// **Why repair-after rather than holding the lock across the unload.** The
    /// production unload is a bounded `keep_alive:0` POST (`AppStateEnv::unload_one`
    /// sets `.timeout(warm_timeout)`), so serializing it would *usually* be fine —
    /// but `ResidentEnv` is an injected trait and the bound is the implementation's
    /// promise, not the type's. Holding `inner` across it would put `status`,
    /// `release` and every coalescing warm behind an await this code cannot bound,
    /// which is the outage `release`'s own bounded-wait design exists to refuse.
    /// So the lock stays off the await and the pass repairs afterwards instead.
    ///
    /// **What "repair" means.** Any slot still claiming to HOLD the model we just
    /// unloaded is claiming residency for something no longer in VRAM. It is marked
    /// not-resident (`Released`, `warmed_at` cleared) so the next pass sees it as
    /// absent and re-warms it — the alias comparison in `reconcile_with` sees a
    /// slot pointing at the outgoing model while the alias points at the new one,
    /// so that reconcile forces past the debounce and the role is re-warmed on the
    /// very next tick. Erring this way is deliberate: a spurious repair costs one
    /// keep_alive re-assert against a model that is already loaded, while a missed
    /// one is exactly the bookkeeping/GPU divergence TRTR-07d spent several rounds
    /// eliminating.
    ///
    /// Repair runs whether the unload reported success or failure. A transport
    /// error or a timeout cannot tell us whether the request landed, so the only
    /// answer that is safe under every interleaving is to stop claiming residency
    /// we can no longer vouch for.
    ///
    /// **A release still wins (TRTR-07d).** A release marks every held slot
    /// `Released` before we get here, so there is normally nothing to repair. The
    /// generation check below is what makes that unconditional: once the generation
    /// has moved, the release owns the exemption and this pass must never re-apply
    /// one — the bookkeeping is still repaired, the exemption is left exactly as
    /// the release left it.
    ///
    /// **No `env` call happens under `inner`.** The repair decides everything —
    /// which slots are false, whether this pass still owns the exemption, and what
    /// the exemption should be — under the lock, then RELEASES it and installs the
    /// exemption outside. Applying it under the lock would put `status`, `release`
    /// and every coalescing lifecycle operation behind an await against an
    /// *injected* trait, which is the same objection this header raises to holding
    /// the lock across `unload_one`; it applies to `set_exempt` verbatim.
    ///
    /// **Two properties a future reader should not have to re-derive.**
    /// (a) A role that re-holds the model AFTER its repair and before the pass ends
    /// would leave the same false claim behind. That interleaving is unreachable
    /// in-tree while this loop retains `in_flight` — `admit` joins rather than
    /// starts, so no pass can install slots — and the deliberate bar here is the
    /// structural repair plus this disclosure, not an absolute guarantee under
    /// interleavings the design does not admit.
    /// (b) A release can supersede the generation between the re-check and the
    /// request, so an unload can go out on behalf of a superseded pass. That is
    /// benign: a release wants that model released anyway, and because it has
    /// already marked every slot `Released`, the repair that follows finds nothing
    /// to fix and leaves the exemption to the release.
    ///
    /// The two OTHER unload sites (`commit`'s discard path and
    /// `compensate_pending`) deliberately do NOT call this: both run only after a
    /// release has already marked every slot `Released`, and no pass can be
    /// admitted while `in_flight` is retained, so neither can end with the set
    /// claiming residency at all. This loop is the only one that runs on a
    /// COMMITTED pass with freshly-held slots installed.
    async fn repair_stale_residency(
        &self,
        env: &dyn ResidentEnv,
        generation: u64,
        model: &str,
    ) -> usize {
        // PHASE 1 — repair, and DECIDE the exemption, entirely under the lock.
        // Nothing in this block touches `env`.
        let (repaired, exempt) = {
            let mut inner = self.inner.lock().await;

            let mut repaired = 0usize;
            for (role, slot) in inner.slots.iter_mut() {
                if slot.state.is_held() && slot.model.as_deref() == Some(model) {
                    slot.state = RoleState::Released;
                    slot.warmed_at = None;
                    repaired += 1;
                    warn!(
                        role = role.id(),
                        model = %model,
                        "resident-set: STALE residency repaired — this role came to claim the model this pass was already unloading; marking it not-resident so it is re-warmed rather than believed resident"
                    );
                }
            }

            if repaired == 0 {
                return 0;
            }

            if inner.generation != generation {
                // A release owns the exemption now. Repairing the bookkeeping above
                // is still right (the claim was false either way); re-applying an
                // exemption here would resurrect a pin the release just dropped.
                warn!(
                    pass_generation = generation,
                    current_generation = inner.generation,
                    model = %model,
                    "resident-set: stale residency repaired under a superseded generation — leaving the exemption to the release that owns it"
                );
                return repaired;
            }

            // The exemption must track what we actually hold, so recompute it from
            // the REPAIRED slots — under the same lock acquisition and the same
            // generation check that authorised the repair. That is what makes the
            // list safe to install after the lock is dropped: every slot claiming
            // `model` was just marked `Released` a few lines up, and no new pass
            // can install slots while this loop retains `in_flight` (GUARD 3), so
            // `model` cannot appear in this list whenever the call lands. Dropping
            // the lock therefore cannot make the exemption stale in the one way
            // that would matter — it can never resurrect residency for the model
            // this pass just unloaded.
            let mut held: Vec<String> = inner
                .slots
                .iter()
                .filter(|(_, s)| s.state.is_held())
                .filter_map(|(_, s)| s.model.clone())
                .collect();
            held.sort();
            held.dedup();
            (repaired, held)
        };

        // PHASE 2 — install it with the lock RELEASED, so an injected or slow
        // `set_exempt` stalls nothing else.
        env.set_exempt(&exempt).await;

        // PHASE 3 — the one thing dropping the lock actually costs, closed. The
        // ONLY writer that can act in this window is `release_with` (a warm pass
        // cannot be admitted while `in_flight` is retained), and what it does is
        // bump the generation and CLEAR the exemption. Neither side can impose an
        // ordering on the two `env` calls, so if the release landed while ours was
        // in flight we may have just re-installed a pin it dropped. Re-check, and
        // if the generation moved, re-assert the release's intent — outside the
        // lock, again. Clearing is idempotent, and it cannot clobber a legitimate
        // later exemption: the only code that installs one is a warm commit, and
        // none can run until this pass drops `in_flight`.
        let superseded = {
            let inner = self.inner.lock().await;
            inner.generation != generation
        };
        if superseded {
            warn!(
                pass_generation = generation,
                model = %model,
                "resident-set: a release landed while the repaired exemption was being applied — clearing it again so the release, not this pass, owns the exemption"
            );
            env.clear_exempt().await;
        }

        repaired
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

/// The default pin horizon, in days: **3650 days (10 years)**.
///
/// **This is a HEURISTIC, and the code says so on purpose.** Ollama's `/api/ps` does
/// not report WHO loaded a model or WHETHER it was loaded with the indefinite
/// sentinel — see [`PROVENANCE`] below for the measured field list. Expiry distance
/// is the only signal available, so classification is inference, not fact.
///
/// **Why 10 years, measured against the live host (Ollama 0.22.1, 2026-07-31).**
/// `keep_alive:-1` is rendered by Ollama as `now + i64::MAX nanoseconds` — a fixed
/// **292.27 years**, observed on this host as four models all expiring
/// `2318-11-10T13:14:31`. The resident set's own bounded residency is **24 hours**.
/// So the two populations we must separate are ~24h apart from ~292 years, five
/// orders of magnitude:
///
/// | | value | ratio to the 10y horizon |
/// |---|---|---|
/// | resident set's own keep_alive | 24 h | 3650x BELOW |
/// | an absurdly long *deliberate* keep_alive (`8760h` = 1 year) | 1 y | 10x BELOW |
/// | the `-1` sentinel | 292.27 y | 29.2x ABOVE |
///
/// The old 365-day default sat 1x from a one-year keep_alive — i.e. a deliberate
/// `8760h` was a coin flip. Ten years cannot plausibly be a deliberate bounded
/// keep_alive (Ollama keep_alive is a duration string; real ones are minutes to
/// hours) while still clearing the sentinel by a factor of 29.
///
/// **Residual false-positive cost, stated plainly:** if someone really does set a
/// bounded keep_alive longer than 10 years, convergence unloads it once at startup.
/// That model is NOT deleted and NOT blocked — the next request reloads it on demand.
/// The cost is one cold load; it is recoverable, and it is the deliberate trade
/// against leaving tens of GB of VRAM pinned by nothing.
pub const DEFAULT_PIN_HORIZON_DAYS: i64 = 3650;

/// **Measured provenance evidence (Ollama 0.22.1, live host, 2026-07-31).** A
/// `GET /api/ps` entry carries exactly:
/// `name`, `model`, `size`, `digest`, `details{parent_model,format,family,families,
/// parameter_size,quantization_level}`, `expires_at`, `size_vram`, `context_length`.
///
/// There is **no** `keep_alive`, no `indefinite`/`pinned` flag, no loader identity,
/// and no session/owner field. `digest`/`details` identify WHAT is loaded, never HOW
/// it was pinned or by WHOM — two models loaded with `24h` and with `-1` are
/// byte-identical in every field except `expires_at`. So there is nothing cheaper or
/// better than the expiry to key on, and the horizon below stays a heuristic by
/// necessity, not by laziness. If a future Ollama exposes keep-alive provenance,
/// prefer it and demote the horizon to a fallback.
pub const PROVENANCE: &str = "ollama /api/ps 0.22.1 exposes no keep_alive/owner field; expiry distance is the only available signal";

/// `CHORD_RESIDENT_PIN_HORIZON_DAYS` (default [`DEFAULT_PIN_HORIZON_DAYS`], 10 years).
/// An `/api/ps` expiry further out than this is *inferred* to be a legacy INDEFINITE
/// pin rather than a keep_alive. See [`DEFAULT_PIN_HORIZON_DAYS`] for the margins on
/// each side and the false-positive cost.
pub fn pin_horizon_days() -> i64 {
    std::env::var("CHORD_RESIDENT_PIN_HORIZON_DAYS")
        .ok()
        .and_then(|s| s.trim().parse::<i64>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_PIN_HORIZON_DAYS)
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
/// using the embeddings endpoint for a model whose name says it is an embedder.
/// Best-effort — returns whether it landed, never errors.
///
/// **Measured (live Ollama 0.22.1, 2026-07-31):** `/api/generate` +
/// `keep_alive: 0` DOES unload an embedding model — 200, and `/api/ps` drops it. So
/// on this version the name pre-check and the 400-retry below are belt-and-braces,
/// not correctness requirements. They are retained because the 400
/// `"does not support generate"` rejection is real on older Ollama builds, and this
/// path runs once at startup against whatever version the host happens to have. The
/// 400 branch is therefore version-conditional, not dead: it cannot fire on 0.22.1
/// and is the only thing that makes convergence work where it can.
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

/// How a convergence pass ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvergeStatus {
    /// No `OLLAMA_URL`: there is nothing to converge against and retrying cannot
    /// change that. TERMINAL — never retried.
    Unconfigured,
    /// Ollama did not answer `/api/ps`. Transient by nature ⇒ worth a retry.
    Unreachable,
    /// The pass ran and nothing legacy-pinned is left. TERMINAL — the success case.
    Settled,
    /// The pass ran but at least one release did not land ⇒ worth a retry.
    Stranded,
}

impl ConvergeStatus {
    /// Is this a terminal outcome — i.e. must the bounded retry loop STOP here?
    /// Only `Settled` (done) and `Unconfigured` (nothing a retry can fix) are.
    pub fn is_terminal(self) -> bool {
        matches!(self, ConvergeStatus::Settled | ConvergeStatus::Unconfigured)
    }
}

/// The outcome of one convergence pass (returned for observability/tests).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvergeReport {
    pub status: ConvergeStatus,
    /// Legacy pins this pass identified.
    pub found: usize,
    /// …of which it successfully released.
    pub unpinned: usize,
    /// …and the names it could NOT release, so a give-up log can name them.
    pub stranded: Vec<String>,
    /// …and the names it deliberately did NOT attempt, because the resident set
    /// had taken them over between the plan snapshot and that model's turn in the
    /// unload loop. A normal, expected outcome — NOT a failure, and never a reason
    /// to leave the pass non-terminal.
    pub skipped_held: Vec<String>,
}

impl ConvergeReport {
    fn terminal(status: ConvergeStatus) -> Self {
        ConvergeReport {
            status,
            found: 0,
            unpinned: 0,
            stranded: Vec::new(),
            skipped_held: Vec::new(),
        }
    }
}

impl Default for ConvergeReport {
    fn default() -> Self {
        ConvergeReport::terminal(ConvergeStatus::Settled)
    }
}

/// The seam convergence runs against, so the whole policy (including the retry
/// loop) is testable without a real Ollama, a real `AppState`, or a network.
#[async_trait]
pub trait ConvergeEnv: Send + Sync {
    /// Currently loaded models as `(name, expires_at)`, or why we cannot tell.
    async fn loaded(&self) -> Result<Vec<(String, Option<String>)>, ConvergeStatus>;
    /// The models the resident set currently HOLDS — never touched by convergence.
    ///
    /// Re-read on every pass AND **immediately before every individual unload** —
    /// see `converge_once` for why the batch snapshot alone is a TOCTOU. Implementations
    /// must therefore be cheap enough to call once per unload target (the production
    /// impl reads in-process resident-set state, no network).
    async fn held(&self) -> HashSet<String>;
    /// Release one model from VRAM. Returns whether it landed. Never errors.
    async fn unload(&self, model: &str) -> bool;
    /// Wall clock, epoch seconds (injected so tests are deterministic).
    fn now_epoch_secs(&self) -> i64;
}

/// The production [`ConvergeEnv`]: the real Ollama + the live resident set.
pub struct AppStateConvergeEnv<'a> {
    state: &'a Arc<crate::routes::AppState>,
}

impl<'a> AppStateConvergeEnv<'a> {
    pub fn new(state: &'a Arc<crate::routes::AppState>) -> Self {
        AppStateConvergeEnv { state }
    }
    fn base(&self) -> Option<String> {
        crate::gpu_exclusive::ollama_base_from_env()
            .map(|b| b.trim_end_matches('/').to_string())
    }
}

#[async_trait]
impl<'a> ConvergeEnv for AppStateConvergeEnv<'a> {
    async fn loaded(&self) -> Result<Vec<(String, Option<String>)>, ConvergeStatus> {
        let Some(base) = self.base() else {
            return Err(ConvergeStatus::Unconfigured);
        };
        let stats =
            crate::sweep_status::ollama::query_ollama_ps(&self.state.http_client, &base).await;
        if !stats.available {
            return Err(ConvergeStatus::Unreachable);
        }
        Ok(stats
            .models
            .into_iter()
            .map(|m| (m.name, m.expires_at))
            .collect())
    }

    async fn held(&self) -> HashSet<String> {
        global()
            .status()
            .await
            .roles
            .into_iter()
            .filter(|r| r.warm)
            .filter_map(|r| r.model)
            .collect()
    }

    async fn unload(&self, model: &str) -> bool {
        let Some(base) = self.base() else {
            return false;
        };
        unload_untyped(&self.state.http_client, &base, model).await
    }

    fn now_epoch_secs(&self) -> i64 {
        crate::gpu_exclusive::now_epoch() as i64
    }
}

/// ONE convergence pass against an injected env. Best-effort and idempotent.
///
/// **The held-set is re-read immediately before EVERY individual unload, not once
/// for the batch.** The plan is computed from one `/api/ps` read plus one `held()`
/// snapshot, but issuing the unloads takes real time (each is an HTTP round-trip to
/// Ollama, up to 60s), and a resident-set warm or lifecycle transition can land in
/// the middle of that. Checking `held()` once for the batch means a model that the
/// set legitimately took over WHILE the loop was running would still be unloaded by
/// the very cleanup meant to protect the set's own residency — the worst possible
/// victim. So each target is re-checked at the last moment before its own unload,
/// and a model that has since become held is SKIPPED and logged at INFO (a normal,
/// expected race outcome, not a warning, and not a failure of the pass).
///
/// **The reverse ordering is deliberately NOT chased.** A model that was held at
/// snapshot time and is released before the loop reaches it stays untouched this
/// pass: it was excluded from `targets` by `plan_unpin` and is never re-planned
/// mid-loop. That is safe and intentional. (a) Unloading it would be harmless —
/// it is genuinely unpinned and reloads on demand — so there is no correctness
/// pressure to widen the batch. (b) Widening it WOULD mean re-planning against a
/// held-set read that can race the other way, turning a conservative snapshot into
/// one that can grow the unload set from a momentary observation — exactly the
/// hazard the per-unload re-check exists to remove. The two directions are
/// asymmetric: missing a target costs at most one more pass (convergence is
/// idempotent, and the reconcile re-attempt below re-runs it while a pass is
/// non-terminal), while unloading a wrongly-included one costs a live role its
/// VRAM residency. We accept the miss and refuse the over-reach. In practice the
/// case is close to vacuous anyway: a held model has the set's own BOUNDED 24h
/// keep_alive re-asserted on it, so it does not look like an indefinite pin at all
/// unless it also carries a past-horizon expiry.
pub async fn converge_once(env: &dyn ConvergeEnv) -> ConvergeReport {
    let loaded = match env.loaded().await {
        Ok(l) => l,
        Err(ConvergeStatus::Unconfigured) => {
            info!("resident-set: OLLAMA_URL unset — no legacy pin convergence possible (best-effort)");
            return ConvergeReport::terminal(ConvergeStatus::Unconfigured);
        }
        Err(_) => {
            info!("resident-set: Ollama /api/ps unavailable — legacy pin convergence deferred to a retry");
            return ConvergeReport::terminal(ConvergeStatus::Unreachable);
        }
    };
    let held = env.held().await;
    let targets = plan_unpin(&loaded, &held, env.now_epoch_secs(), pin_horizon_days());
    let mut report = ConvergeReport {
        status: ConvergeStatus::Settled,
        found: targets.len(),
        unpinned: 0,
        stranded: Vec::new(),
        skipped_held: Vec::new(),
    };
    for model in targets {
        // TOCTOU close: the snapshot above may be stale by now. Re-read the held
        // set for THIS model, immediately before its unload.
        if env.held().await.contains(&model) {
            info!(
                model = %model,
                "resident-set: model became held by the resident set after the convergence plan was taken — skipping its release (expected race; the set now owns its bounded lifecycle)"
            );
            report.skipped_held.push(model);
            continue;
        }
        warn!(
            model = %model,
            horizon_days = pin_horizon_days(),
            heuristic = true,
            "resident-set: expiry is past the pin horizon, so this is INFERRED to be a legacy indefinite VRAM pin owned by no lifecycle (retired keep-resident mechanism) — releasing it. Heuristic: Ollama exposes no keep-alive provenance; a false positive costs one on-demand reload, nothing is deleted"
        );
        if env.unload(&model).await {
            report.unpinned += 1;
        } else {
            warn!(
                model = %model,
                "resident-set: legacy pin release did not land — will retry (bounded)"
            );
            report.stranded.push(model);
        }
    }
    if !report.stranded.is_empty() {
        report.status = ConvergeStatus::Stranded;
    }
    if report.found > 0 && report.status == ConvergeStatus::Settled {
        info!(
            found = report.found,
            unpinned = report.unpinned,
            skipped_held = report.skipped_held.len(),
            "resident-set: legacy pin convergence complete — residency now has a single owner"
        );
    }
    report
}

/// Run the startup convergence against the live host. Best-effort and idempotent.
pub async fn converge_legacy_pins(state: &Arc<crate::routes::AppState>) -> ConvergeReport {
    converge_once(&AppStateConvergeEnv::new(state)).await
}

// ─────────────────────────────────────────────────────────────────────────────
// CHRD-PIN-01: BOUNDED convergence retry
// ─────────────────────────────────────────────────────────────────────────────
//
// A single best-effort pass strands a pin forever if Ollama happens to be down at
// startup, or if one unload fails. So convergence retries — but STRICTLY BOUNDED:
// a small number of attempts with exponential backoff, stopping the moment it
// succeeds once (or hits an outcome no retry can fix). It is deliberately NOT a
// background loop: nothing here can end up hammering Ollama for the process
// lifetime. Every attempt re-reads what the resident set holds, so a retry can
// never fight a legitimate re-warm.

/// Bounded retry policy for startup convergence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvergeRetryPolicy {
    /// TOTAL attempts, including the first. Always >= 1 and hard-capped.
    pub attempts: u32,
    pub base_backoff: Duration,
    pub max_backoff: Duration,
}

/// The hard ceiling on attempts, whatever the env says — the bound is not
/// operator-defeatable.
pub const CONVERGE_MAX_ATTEMPTS: u32 = 10;

impl Default for ConvergeRetryPolicy {
    fn default() -> Self {
        // 5 attempts, 30s → 60s → 120s → 240s: bounded at ~7.5 minutes total.
        ConvergeRetryPolicy {
            attempts: 5,
            base_backoff: Duration::from_secs(30),
            max_backoff: Duration::from_secs(300),
        }
    }
}

impl ConvergeRetryPolicy {
    pub fn from_env() -> Self {
        let d = ConvergeRetryPolicy::default();
        let attempts = std::env::var("CHORD_RESIDENT_PIN_CONVERGE_ATTEMPTS")
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .map(|n| n.clamp(1, CONVERGE_MAX_ATTEMPTS))
            .unwrap_or(d.attempts);
        let base = std::env::var("CHORD_RESIDENT_PIN_CONVERGE_BACKOFF_SECS")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .filter(|&n| n > 0)
            .map(Duration::from_secs)
            .unwrap_or(d.base_backoff);
        let max = std::env::var("CHORD_RESIDENT_PIN_CONVERGE_MAX_BACKOFF_SECS")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .filter(|&n| n > 0)
            .map(Duration::from_secs)
            .unwrap_or(d.max_backoff);
        ConvergeRetryPolicy {
            attempts,
            base_backoff: base,
            max_backoff: max.max(base),
        }
    }

    /// Backoff before the attempt AFTER `attempt_index` (0-based): exponential from
    /// `base_backoff`, capped at `max_backoff`. Pure.
    pub fn backoff(&self, attempt_index: u32) -> Duration {
        let shift = attempt_index.min(16);
        let secs = self
            .base_backoff
            .as_secs()
            .saturating_mul(1u64 << shift)
            .min(self.max_backoff.as_secs());
        Duration::from_secs(secs)
    }
}

/// Drive [`converge_once`] under a BOUNDED retry policy. Returns the final report
/// and how many attempts were spent. `attempt` and `sleep` are injected so the
/// bound itself is unit-testable without wall-clock time.
pub async fn converge_until_settled<A, AF, S, SF>(
    policy: &ConvergeRetryPolicy,
    mut attempt: A,
    mut sleep: S,
) -> (ConvergeReport, u32)
where
    A: FnMut() -> AF,
    AF: std::future::Future<Output = ConvergeReport>,
    S: FnMut(Duration) -> SF,
    SF: std::future::Future<Output = ()>,
{
    let total = policy.attempts.clamp(1, CONVERGE_MAX_ATTEMPTS);
    let mut report = ConvergeReport::terminal(ConvergeStatus::Unreachable);
    for i in 0..total {
        report = attempt().await;
        if report.status.is_terminal() {
            if i > 0 {
                info!(
                    attempts = i + 1,
                    "resident-set: legacy pin convergence succeeded on retry"
                );
            }
            return (report, i + 1);
        }
        if i + 1 < total {
            sleep(policy.backoff(i)).await;
        }
    }
    warn!(
        attempts = total,
        status = ?report.status,
        still_pinned = %if report.stranded.is_empty() { "unknown (Ollama unreachable)".to_string() } else { report.stranded.join(", ") },
        "resident-set: GIVING UP on legacy pin convergence after the bounded retry budget — the models named above are still holding VRAM that no lifecycle owns; an operator must unload them (ollama stop <model>) or restart Ollama"
    );
    (report, total)
}

// ─────────────────────────────────────────────────────────────────────────────
// CHRD-PIN-01: the reconcile-tick RE-ATTEMPT (what happens after the burst gives up)
// ─────────────────────────────────────────────────────────────────────────────
//
// The startup burst above must stay BOUNDED — an unbounded retry loop hammering
// Ollama would be its own outage. But permanently giving up after one burst is
// also wrong: a host whose Ollama was briefly unavailable at startup keeps tens of
// GB pinned indefinitely, with nothing but a log line to say so.
//
// So a non-terminal burst ARMS the existing reconcile loop to re-attempt
// convergence — **at most one attempt per reconcile tick, with no inner retry
// loop of any kind** — until convergence settles once, after which it is disarmed
// forever. Terminal outcomes (`Settled`, `Unconfigured`) never arm it at all.
//
// Why this cannot become a hot loop, structurally:
//   * the ONLY caller is the reconcile loop body, which is gated by that loop's own
//     `sleep(interval)` — so attempts are rate-limited by the reconcile interval,
//     not by anything convergence does;
//   * [`converge_reattempt_tick`] takes an `FnOnce` and contains no loop: one call
//     ⇒ at most one `converge_once`;
//   * the gate state machine is MONOTONE — `Pending → Armed → Done` — and `Done`
//     is absorbing, so the number of attempts after the first terminal outcome is
//     exactly zero, forever.
//
// Worst-case convergence latency is therefore: the bounded startup burst (default
// 5 attempts, 30/60/120/240s backoff ⇒ ~7.5 min) plus up to one reconcile interval
// (`CHORD_RESIDENT_REFRESH_SECS`, default 300s) per subsequent attempt, for as long
// as the host stays broken — one probe every 5 minutes, which is strictly cheaper
// than the reconcile tick it rides on.

/// The three states of the reconcile re-attempt gate. Monotone: it only ever moves
/// forward, and `Done` is absorbing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvergeGateState {
    /// The startup burst has not reported yet — the reconcile loop must NOT also
    /// be converging, or the two would run concurrently against the same host.
    Pending,
    /// The burst ended non-terminal (`Unreachable`/`Stranded`): re-attempt once per
    /// reconcile tick.
    Armed,
    /// Convergence settled (or is unfixable). Never attempt again.
    Done,
}

const GATE_PENDING: u8 = 0;
const GATE_ARMED: u8 = 1;
const GATE_DONE: u8 = 2;

/// Shared, lock-free state deciding whether a reconcile tick should also try to
/// converge. One per process, created by [`startup_residency`].
#[derive(Debug)]
pub struct ConvergeGate {
    state: AtomicU8,
}

impl Default for ConvergeGate {
    fn default() -> Self {
        ConvergeGate::new()
    }
}

impl ConvergeGate {
    pub fn new() -> Self {
        ConvergeGate {
            state: AtomicU8::new(GATE_PENDING),
        }
    }

    pub fn state(&self) -> ConvergeGateState {
        match self.state.load(AtomicOrdering::SeqCst) {
            GATE_ARMED => ConvergeGateState::Armed,
            GATE_DONE => ConvergeGateState::Done,
            _ => ConvergeGateState::Pending,
        }
    }

    /// Should this reconcile tick attempt a convergence? Only when ARMED.
    pub fn should_attempt(&self) -> bool {
        self.state() == ConvergeGateState::Armed
    }

    fn advance(&self, status: ConvergeStatus) {
        let next = if status.is_terminal() {
            GATE_DONE
        } else {
            GATE_ARMED
        };
        // `Done` is absorbing: never step back from it.
        let _ = self.state.fetch_update(
            AtomicOrdering::SeqCst,
            AtomicOrdering::SeqCst,
            |cur| {
                if cur == GATE_DONE || cur == next {
                    None
                } else {
                    Some(next)
                }
            },
        );
    }

    /// The startup burst finished with `status`. Non-terminal ⇒ arm the reconcile
    /// re-attempt; terminal ⇒ close the gate permanently.
    pub fn record_startup_outcome(&self, status: ConvergeStatus) {
        self.advance(status);
        if self.state() == ConvergeGateState::Armed {
            info!(
                status = ?status,
                "resident-set: startup convergence did not settle — re-attempting once per reconcile tick until it does (no retry storm; one attempt per tick)"
            );
        }
    }

    /// A reconcile-tick attempt finished with `status`.
    pub fn record_attempt(&self, status: ConvergeStatus) {
        let was = self.state();
        self.advance(status);
        if was != ConvergeGateState::Done && self.state() == ConvergeGateState::Done {
            info!("resident-set: legacy pin convergence settled on a reconcile re-attempt — no further attempts will be made");
        }
    }
}

/// One reconcile tick's worth of convergence: **at most one attempt**, and only
/// while the gate is armed. `FnOnce` + no loop is the structural guarantee that a
/// tick can never turn into a retry storm. Returns `None` when the gate is not
/// armed (nothing ran).
pub async fn converge_reattempt_tick<A, AF>(
    gate: &ConvergeGate,
    attempt: A,
) -> Option<ConvergeReport>
where
    A: FnOnce() -> AF,
    AF: std::future::Future<Output = ConvergeReport>,
{
    if !gate.should_attempt() {
        return None;
    }
    let report = attempt().await;
    if !report.status.is_terminal() {
        warn!(
            status = ?report.status,
            still_pinned = %if report.stranded.is_empty() { "unknown (Ollama unreachable)".to_string() } else { report.stranded.join(", ") },
            "resident-set: reconcile re-attempt of legacy pin convergence did not settle — the models named above are still holding VRAM that no lifecycle owns; will try again on the next reconcile tick"
        );
    }
    gate.record_attempt(report.status);
    Some(report)
}

/// Production convergence: bounded retries against the live host, real sleeps.
pub async fn converge_legacy_pins_bounded(state: Arc<crate::routes::AppState>) -> ConvergeReport {
    let policy = ConvergeRetryPolicy::from_env();
    let (report, _attempts) = converge_until_settled(
        &policy,
        || {
            let state = state.clone();
            async move { converge_legacy_pins(&state).await }
        },
        |d| tokio::time::sleep(d),
    )
    .await;
    if report.found > 0 {
        info!(
            found = report.found,
            unpinned = report.unpinned,
            "resident-set: released legacy indefinite VRAM pins at startup"
        );
    }
    report
}

// ─────────────────────────────────────────────────────────────────────────────
// CHRD-PIN-01: the startup residency task (EXPLICIT ordering)
// ─────────────────────────────────────────────────────────────────────────────

/// The single startup entry point for residency. Owns the ORDER of the three
/// startup concerns, explicitly, so none of it is incidental to how the calls
/// happen to be arranged in `main`:
///
/// 1. **If the resident set is ENABLED**, its first warm runs to COMPLETION first.
///    That re-asserts a bounded `keep_alive` on everything the set legitimately
///    holds, so step 2 sees those models as bounded and skips them. This ordering
///    is the reason convergence cannot unload the set's own models out from under it.
/// 2. **Convergence runs UNCONDITIONALLY** — it is a MIGRATION concern, not a
///    residency-feature concern. Gating it on `CHORD_RESIDENT_SET_ENABLED` would be
///    exactly backwards: with the set OFF, nothing else in the process will ever
///    release a stranded `keep_alive` pin, so that is precisely when convergence
///    matters most. Disabled ⇒ the set holds nothing ⇒ everything that looks like a
///    legacy pin converges.
/// 3. **The reconcile loop** then runs for the life of the process (itself a no-op
///    while the set is disabled or released).
///
/// Convergence is spawned rather than awaited so its bounded retry backoff can
/// never delay the reconcile loop starting — but it is spawned only AFTER the warm
/// in step 1 has returned, which is what makes the ordering a guarantee.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StartupPlan {
    /// Warm the set (and therefore wait for that warm) BEFORE converging.
    pub warm_first: bool,
    /// Run legacy-pin convergence at all.
    pub converge: bool,
}

/// The startup ordering, as DATA rather than as control flow buried in a spawn —
/// so "convergence does not depend on resident-set enablement" is a property a test
/// can assert directly instead of one a reader has to infer.
pub fn plan_startup(resident_set_enabled: bool) -> StartupPlan {
    StartupPlan {
        // Only an ENABLED set has a warm to wait for.
        warm_first: resident_set_enabled,
        // ALWAYS. Migration concern, not a residency-feature concern.
        converge: true,
    }
}

pub async fn startup_residency(state: Arc<crate::routes::AppState>) {
    let set = global();
    let plan = plan_startup(set.config().enabled);

    // Step 1 — ordering gate.
    if plan.warm_first {
        let _ = set.warm(&state, "startup", true).await;
    } else {
        info!(
            "resident-set: DISABLED (CHORD_RESIDENT_SET_ENABLED=0) — no warm; legacy-pin convergence still runs, because with the set off nothing else would ever release a stranded pin"
        );
    }

    // Step 2 — unconditional, and demonstrably after step 1.
    // The gate starts PENDING so the reconcile loop (step 3, which starts while
    // this burst is still running) never converges concurrently with it.
    let gate = Arc::new(ConvergeGate::new());
    if plan.converge {
        let converge_state = state.clone();
        let converge_gate = gate.clone();
        tokio::spawn(async move {
            let report = converge_legacy_pins_bounded(converge_state).await;
            // A non-terminal burst hands the problem to the reconcile loop rather
            // than leaving ~60GB pinned until an operator notices.
            converge_gate.record_startup_outcome(report.status);
        });
    }

    // Step 3.
    reconcile_loop(state, set.config().refresh, gate).await;
}

/// Background loop: periodically reconcile the resident set so an alias repoint
/// or a drifted keep_alive is corrected without a restart. Best-effort and
/// no-op while released — it never contends with a mode swap.
///
/// It also carries the convergence RE-ATTEMPT (see [`ConvergeGate`]): if the
/// bounded startup burst ended non-terminal, each tick makes exactly ONE further
/// convergence attempt until it settles, then stops forever. The tick's own
/// `sleep(interval)` is what rate-limits it — there is no inner retry loop here.
pub async fn reconcile_loop(
    state: Arc<crate::routes::AppState>,
    interval: Duration,
    converge_gate: Arc<ConvergeGate>,
) {
    info!(
        interval_secs = interval.as_secs(),
        "resident-set reconcile loop started"
    );
    loop {
        tokio::time::sleep(interval).await;
        let _ = global().reconcile(&state).await;
        // At most one convergence attempt per tick, and only while armed.
        let _ = converge_reattempt_tick(&converge_gate, || {
            let state = state.clone();
            async move { converge_legacy_pins(&state).await }
        })
        .await;
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

    /// CHRD-PIN-01 review FINDING 1: NO role has a code-level fallback.
    ///
    /// The earlier revision fell back to the configured `EMBED_LOCAL_MODEL` for the
    /// embedding role. That made an UNRESOLVED role warm and exempt a model anyway —
    /// a special-case lifecycle target, and a second implicit source of residency
    /// truth substituting a different model than the operator configured. It is
    /// gone: resolution is `resolve_alias` and nothing else, for every role.
    #[test]
    fn no_role_has_a_code_level_fallback_when_its_alias_is_unresolved() {
        let statics = HashMap::new();
        let dynamic = LuminaAliasStore::empty();
        for role in Role::PRIORITY {
            assert_eq!(
                resolve_alias(role.default_alias(), &dynamic, &statics),
                None,
                "{} must resolve to NOTHING — no role gets a substituted model",
                role.id()
            );
        }
    }

    /// The actionable half of the loud warning is UNIFORM: every role's remedy names
    /// that role, the alias that failed, and the env var that repoints it.
    #[test]
    fn the_unresolved_alias_remedy_is_actionable_for_every_role() {
        for role in Role::PRIORITY {
            let alias = role.default_alias();
            let msg = unresolved_alias_remedy(role, alias);
            assert!(
                msg.contains(alias),
                "{}: remedy must name the alias that failed: {msg}",
                role.id()
            );
            assert!(
                msg.contains(role.id()),
                "{}: remedy must name the role: {msg}",
                role.id()
            );
            assert!(
                msg.contains(role.alias_env()),
                "{}: remedy must name the env var that repoints the role: {msg}",
                role.id()
            );
            assert!(
                msg.contains("CHORD_MODEL_ALIASES"),
                "{}: remedy must say where the alias is configured: {msg}",
                role.id()
            );
            assert!(
                msg.contains("holds nothing"),
                "{}: remedy must state the consequence — nothing is pinned: {msg}",
                role.id()
            );
        }
    }

    /// Positive control for the removal: a RESOLVED alias still resolves normally,
    /// dynamic-store-first, for every role.
    #[test]
    fn a_resolved_alias_still_resolves_for_every_role() {
        let mut statics = HashMap::new();
        // Keyed by ALIAS, not by role — personality and router deliberately share
        // `lumina-fast`, and a shared alias must resolve identically for both.
        for role in Role::PRIORITY {
            let alias = role.default_alias();
            statics.insert(alias.to_string(), format!("static-{alias}:1b"));
        }
        let dynamic = LuminaAliasStore::from_static(&statics);
        for role in Role::PRIORITY {
            let alias = role.default_alias();
            assert_eq!(
                resolve_alias(alias, &dynamic, &statics).as_deref(),
                Some(format!("static-{alias}:1b").as_str()),
                "{} must still resolve through its alias",
                role.id()
            );
        }
        // …and a runtime repoint still wins over the static map.
        dynamic.set("lumina-embed", "dynamic-embed:1b".to_string());
        assert_eq!(
            resolve_alias("lumina-embed", &dynamic, &statics).as_deref(),
            Some("dynamic-embed:1b"),
            "a runtime repoint must still win"
        );
    }

    // ── CHRD-PIN-01 Task A: no indefinite pin survives ──────────────────────
    //
    // The load-bearing RULE this guard enforces (a scoped, best-effort enforcement
    // against the SOURCE TREE — NOT a proof of the absolute invariant; see the ⚠
    // block below): no code path anywhere in this crate may ask Ollama for an
    // indefinite `keep_alive`. A behavioral test can only cover the paths it knows about; the
    // failure mode being guarded is a NEW (or resurrected) path nobody wired a test
    // to — which is exactly how the retired keep-resident pass coexisted with the
    // resident set in the first place.
    //
    // ⚠ WHAT THIS GUARD IS AND IS NOT (read before trusting it).
    // It is a cheap LEXICAL scanner over the crate's `.rs` files, not a dataflow
    // analysis. It catches three shapes:
    //   1. the direct one — `keep_alive` and a `-1` on the same non-comment line;
    //   2. an INDIRECTION — a `-1` bound to a named constant/binding/field anywhere
    //      in the crate (`const FOREVER: i64 = -1;`, `let forever = -1;`,
    //      `some_field: -1,`), where that name later appears on a `keep_alive` line;
    //   3. a KEEP_ALIVE-NAMED binding or field assigned `-1` even with no literal
    //      `keep_alive` token on the line (`let keepAlive = -1;`).
    // It does NOT and CANNOT catch: a value computed at runtime (`0 - 1`,
    // `i64::MIN`, a parsed env var/config value), a `-1` passed through a function
    // parameter or a generic wrapper before reaching the request, a pin assembled
    // in a data structure many statements away with no shared name, a pin issued
    // from OUTSIDE this crate, or a pin issued by hand against the live Ollama.
    // Those residuals are real. This guard raises the cost of re-introducing the
    // retired mechanism; it is NOT a proof that no indefinite pin can exist. The
    // GPU-side backstop for everything it misses is `converge_legacy_pins`, which
    // reads what Ollama actually holds and unpins by observation rather than by
    // reading code.
    //
    // A genuinely bounded, explicitly-released phase may opt out by putting the
    // marker `RESIDENCY-PIN-ALLOWED` on the line, which makes the exception
    // deliberate, greppable, and reviewable instead of silent.

    /// The three needles, assembled at runtime so the guard can never match its own
    /// source. `(keep_alive, -1, RESIDENCY-PIN-ALLOWED)`.
    fn pin_needles() -> (String, String, String) {
        (
            ["keep", "_alive"].concat(),
            ["-", "1"].concat(),
            ["RESIDENCY", "-PIN-ALLOWED"].concat(),
        )
    }

    /// Is this line prose (a doc/comment) or explicitly opted out? Such lines may
    /// freely DISCUSS the retired mechanism.
    fn pin_scan_skips(line: &str, allow: &str) -> bool {
        let t = line.trim_start();
        t.starts_with("//") || t.starts_with("*") || line.contains(allow)
    }

    /// A `-1` that is a standalone numeric literal here, not the tail of `-100` or
    /// part of an identifier/date. Returns the byte offsets at which one starts.
    fn negative_one_positions(line: &str, neg: &str) -> Vec<usize> {
        let bytes = line.as_bytes();
        let mut out = Vec::new();
        let mut from = 0usize;
        while let Some(rel) = line[from..].find(neg) {
            let at = from + rel;
            let after = bytes.get(at + neg.len()).copied();
            let is_bare = match after {
                None => true,
                Some(c) => !(c as char).is_ascii_alphanumeric() && c != b'_' && c != b'.',
            };
            if is_bare {
                out.push(at);
            }
            from = at + 1;
        }
        out
    }

    /// Names bound to a bare `-1` on this line: `const FOREVER: i64 = -1;`,
    /// `let forever = -1;`, `field: -1,`. Cheap and deliberately shallow — it takes
    /// the identifier immediately left of the `=`/`:` that introduces the literal.
    fn pin_constant_names(line: &str) -> Vec<String> {
        let (_, neg, allow) = pin_needles();
        if pin_scan_skips(line, &allow) {
            return Vec::new();
        }
        let mut out = Vec::new();
        for at in negative_one_positions(line, &neg) {
            let mut head = line[..at].trim_end();
            // Step back over the introducer.
            if let Some(h) = head.strip_suffix('=') {
                head = h.trim_end();
                // `let x: i64 = -1` — drop the type annotation.
                if let Some((lhs, _ty)) = head.rsplit_once(':') {
                    if !lhs.trim_end().ends_with(':') {
                        head = lhs.trim_end();
                    }
                }
            } else if let Some(h) = head.strip_suffix(':') {
                head = h.trim_end();
            } else {
                continue;
            }
            let name: String = head
                .chars()
                .rev()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            let name = name.trim_matches('"').to_string();
            if name.len() >= 2 && !matches!(name.as_str(), "let" | "const" | "static" | "mut") {
                out.push(name);
            }
        }
        out
    }

    /// Does `line` reference `name` as a whole identifier token?
    fn mentions_ident(line: &str, name: &str) -> bool {
        let bytes = line.as_bytes();
        let mut from = 0usize;
        while let Some(rel) = line[from..].find(name) {
            let at = from + rel;
            let before_ok = at == 0 || {
                let c = bytes[at - 1];
                !(c as char).is_ascii_alphanumeric() && c != b'_'
            };
            let after = bytes.get(at + name.len()).copied();
            let after_ok = match after {
                None => true,
                Some(c) => !(c as char).is_ascii_alphanumeric() && c != b'_',
            };
            if before_ok && after_ok {
                return true;
            }
            from = at + 1;
        }
        false
    }

    /// `keep_alive` modulo case and separators — so `keepAlive`/`KEEP_ALIVE` count.
    fn is_keep_alive_named(name: &str) -> bool {
        let flat: String = name
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .map(|c| c.to_ascii_lowercase())
            .collect();
        flat.contains(&["keep", "alive"].concat())
    }

    /// The pure scanner. `pinned_names` are identifiers the crate binds to `-1`
    /// (from [`pin_constant_names`], collected crate-wide first so an indirection
    /// across files is still caught). Returns `(line_number_1_based, trimmed_line)`.
    fn indefinite_pin_offenders(
        text: &str,
        pinned_names: &HashSet<String>,
    ) -> Vec<(usize, String)> {
        let (key, neg, allow) = pin_needles();
        let mut out = Vec::new();
        for (n, line) in text.lines().enumerate() {
            if pin_scan_skips(line, &allow) {
                continue;
            }
            // (1) direct: keep_alive and a -1 on the same line. Kept LOOSE (a plain
            // substring `-1`) so a stringly-typed `"keep_alive": "-1"` is caught too.
            let direct = line.contains(&key) && line.contains(&neg);
            // (2) indirection: a keep_alive line that names a binding the crate
            // bound to -1 somewhere else.
            let via_const = line.contains(&key)
                && pinned_names.iter().any(|c| mentions_ident(line, c));
            // (3) a keep_alive-NAMED binding/field assigned -1, even if the literal
            // token `keep_alive` never appears (`let keepAlive = -1;`).
            let named = pin_constant_names(line).iter().any(|c| is_keep_alive_named(c));
            if direct || via_const || named {
                out.push((n + 1, line.trim_start().to_string()));
            }
        }
        out
    }

    fn crate_rs_files() -> Vec<std::path::PathBuf> {
        fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            let Ok(entries) = std::fs::read_dir(dir) else {
                return;
            };
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    walk(&p, out);
                } else if p.extension().and_then(|s| s.to_str()) == Some("rs") {
                    out.push(p);
                }
            }
        }
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        walk(&root, &mut files);
        files
    }

    #[test]
    fn no_code_path_pins_a_model_indefinitely() {
        let files = crate_rs_files();
        assert!(files.len() > 10, "source scan found suspiciously few files");

        // Pass 1, crate-wide: every identifier bound to a bare -1.
        let mut pinned_names: HashSet<String> = HashSet::new();
        for f in &files {
            let Ok(text) = std::fs::read_to_string(f) else {
                continue;
            };
            for line in text.lines() {
                pinned_names.extend(pin_constant_names(line));
            }
        }

        // Pass 2: flag the three shapes.
        let mut offenders = Vec::new();
        for f in &files {
            let Ok(text) = std::fs::read_to_string(f) else {
                continue;
            };
            for (n, line) in indefinite_pin_offenders(&text, &pinned_names) {
                offenders.push(format!("{}:{}: {}", f.display(), n, line));
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

    /// The guard's own unit tests: it must catch the direct shape, the two
    /// indirections, and NOT fire on prose, on the opt-out marker, or on a bounded
    /// keep_alive. Without these the guard is untested code asserting a property.
    #[test]
    fn the_pin_guard_catches_direct_and_indirect_shapes() {
        let none: HashSet<String> = HashSet::new();
        // Fixtures are ASSEMBLED at runtime for the same reason the needles are:
        // a literal `keep_alive` + `-1` written here would make this test file its
        // own offender under the crate-wide scan.
        let (ka, neg, _) = pin_needles();

        // (1) direct — the original case.
        let direct = format!("    let body = json!({{ \"{ka}\": {neg} }});");
        assert_eq!(indefinite_pin_offenders(&direct, &none).len(), 1);
        // …including the stringly-typed variant.
        let stringly = format!("    let body = json!({{ \"{ka}\": \"{neg}\" }});");
        assert_eq!(indefinite_pin_offenders(&stringly, &none).len(), 1);

        // (2) indirection through a named constant, ACROSS statements.
        let defs = format!("const FOREVER: i64 = {neg};");
        let names: HashSet<String> = pin_constant_names(&defs).into_iter().collect();
        assert!(names.contains("FOREVER"), "the -1 constant must be learned");
        let use_line = format!("    body.insert(\"{ka}\", FOREVER);");
        assert_eq!(
            indefinite_pin_offenders(&use_line, &names).len(),
            1,
            "a pin constant reaching a keep-alive line must be caught"
        );
        // …and with no such constant learned, that same line is clean (so the
        // detection genuinely comes from the constant, not from the word alone).
        assert!(indefinite_pin_offenders(&use_line, &none).is_empty());

        // (3) a keep_alive-NAMED binding assigned -1, with no keep_alive literal
        // in the request line. Case/separator-insensitive, so `keepAlive` counts.
        let named = format!("    let {}Alive = {neg};", "keep");
        assert_eq!(indefinite_pin_offenders(&named, &none).len(), 1);

        // Negatives.
        let bounded = format!("    let body = json!({{ \"{ka}\": \"24h\" }});");
        assert!(indefinite_pin_offenders(&bounded, &none).is_empty());
        let unload = format!("    let body = json!({{ \"{ka}\": 0 }});");
        assert!(indefinite_pin_offenders(&unload, &none).is_empty());
        let prose = format!("    // the retired path used \"{ka}\": {neg} forever");
        assert!(indefinite_pin_offenders(&prose, &none).is_empty());
        // `-1` as the tail of a larger literal is not a -1.
        let bigger = format!("    let timeout_ms = {neg}000;");
        assert!(pin_constant_names(&bigger).is_empty());
    }

    /// The documented escape hatch still works, and works for the NEW shapes too —
    /// otherwise a future bounded phase could not opt out of them.
    #[test]
    fn the_pin_guard_opt_out_marker_still_works() {
        let none: HashSet<String> = HashSet::new();
        let (ka, neg, marker) = pin_needles();

        let direct = format!("    json!({{ \"{ka}\": {neg} }}); // {marker}: bounded phase");
        assert!(
            indefinite_pin_offenders(&direct, &none).is_empty(),
            "the opt-out must still exempt the direct shape"
        );

        let def = format!("const FOREVER: i64 = {neg}; // {marker}");
        assert!(
            pin_constant_names(&def).is_empty(),
            "an opted-out constant must not be learned as a pin source"
        );

        let named = format!("    let {}Alive = {neg}; // {marker}", "keep");
        assert!(
            indefinite_pin_offenders(&named, &none).is_empty(),
            "the opt-out must exempt the keep_alive-named-binding shape"
        );
    }

    /// The guard's honesty test: it is lexical, and these shapes get through. This
    /// exists so nobody reads the guard as a total guarantee — if a future change
    /// makes one of these detectable, DELETE the corresponding line here rather
    /// than leaving a false claim of weakness. The GPU-side backstop for all of
    /// them is `converge_legacy_pins`.
    #[test]
    fn the_pin_guard_documents_what_it_cannot_catch() {
        let none: HashSet<String> = HashSet::new();
        let (ka, _neg, _) = pin_needles();

        // A runtime-computed -1: no literal anywhere.
        let computed = format!("    let v = zero - one; json!({{ \"{ka}\": v }});");
        assert!(
            indefinite_pin_offenders(&computed, &none).is_empty(),
            "documented residual: a computed -1 is invisible to a lexical scan"
        );

        // A -1 routed through a function parameter with an unrelated name.
        let routed = format!("    warm(model, forever_value); // reaches {ka} inside warm()");
        assert!(indefinite_pin_offenders(&routed, &none).is_empty());

        // A value read from config/env at runtime.
        let from_env = format!(
            "    let v = std::env::var(\"X\").unwrap(); json!({{ \"{ka}\": v }});"
        );
        assert!(indefinite_pin_offenders(&from_env, &none).is_empty());
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

    // ── CHRD-PIN-01 round 2: horizon, enablement-independence, bounded retry ──

    /// A fake [`ConvergeEnv`]: no Ollama, no `AppState`, no clock.
    struct FakeConvergeEnv {
        loaded: Result<Vec<(String, Option<String>)>, ConvergeStatus>,
        held: HashSet<String>,
        now: i64,
        /// Models whose unload must FAIL (to exercise the stranded path).
        fail_unload: HashSet<String>,
        unloaded: StdMutex<Vec<String>>,
    }

    impl FakeConvergeEnv {
        fn new(loaded: Vec<(String, Option<String>)>, held: &[&str], now: i64) -> Self {
            FakeConvergeEnv {
                loaded: Ok(loaded),
                held: held.iter().map(|s| s.to_string()).collect(),
                now,
                fail_unload: HashSet::new(),
                unloaded: StdMutex::new(Vec::new()),
            }
        }
        fn unloaded(&self) -> Vec<String> {
            self.unloaded.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl ConvergeEnv for FakeConvergeEnv {
        async fn loaded(&self) -> Result<Vec<(String, Option<String>)>, ConvergeStatus> {
            self.loaded.clone()
        }
        async fn held(&self) -> HashSet<String> {
            self.held.clone()
        }
        async fn unload(&self, model: &str) -> bool {
            if self.fail_unload.contains(model) {
                return false;
            }
            self.unloaded.lock().unwrap().push(model.to_string());
            true
        }
        fn now_epoch_secs(&self) -> i64 {
            self.now
        }
    }

    /// Ollama renders `keep_alive:-1` as `now + i64::MAX ns` ≈ 292.27 years.
    /// Measured live on Ollama 0.22.1 (2026-07-31) as `2318-11-10T13:14:31`.
    const SENTINEL_SECS: i64 = 9_223_372_036; // i64::MAX nanoseconds, in seconds

    fn at(now: i64, offset_secs: i64) -> String {
        chrono::DateTime::from_timestamp(now + offset_secs, 0)
            .unwrap()
            .to_rfc3339()
    }

    // FINDING 1 — convergence must NOT depend on resident-set enablement.

    #[test]
    fn convergence_is_planned_whether_or_not_the_resident_set_is_enabled() {
        // The whole point: disabling the set is precisely when nothing else will
        // ever release a stranded pin, so convergence must still be planned.
        assert!(plan_startup(false).converge);
        assert!(plan_startup(true).converge);
        // …and only an ENABLED set has a first warm to be sequenced after.
        assert!(!plan_startup(false).warm_first);
        assert!(plan_startup(true).warm_first);
    }

    #[tokio::test]
    async fn convergence_runs_with_the_resident_set_disabled() {
        // A disabled set holds nothing, so `held` is empty — and EVERYTHING that
        // looks like a legacy pin must converge.
        let now = 1_800_000_000i64;
        let env = FakeConvergeEnv::new(
            vec![
                ("stranded:30b".to_string(), Some(at(now, SENTINEL_SECS))),
                (
                    "would-be-held:8b".to_string(),
                    Some(at(now, SENTINEL_SECS)),
                ),
            ],
            &[], // disabled ⇒ the set reports no warm role
            now,
        );
        let report = converge_once(&env).await;
        assert_eq!(report.status, ConvergeStatus::Settled);
        assert_eq!(report.found, 2);
        assert_eq!(report.unpinned, 2);
        assert_eq!(env.unloaded(), vec!["stranded:30b", "would-be-held:8b"]);
    }

    #[tokio::test]
    async fn convergence_still_skips_what_the_set_holds_when_enabled() {
        // Positive control for the test above: the ONLY difference is that the
        // enabled set holds `would-be-held:8b`, and that model is then untouched.
        let now = 1_800_000_000i64;
        let env = FakeConvergeEnv::new(
            vec![
                ("stranded:30b".to_string(), Some(at(now, SENTINEL_SECS))),
                (
                    "would-be-held:8b".to_string(),
                    Some(at(now, SENTINEL_SECS)),
                ),
            ],
            &["would-be-held:8b"],
            now,
        );
        let report = converge_once(&env).await;
        assert_eq!(report.found, 1);
        assert_eq!(env.unloaded(), vec!["stranded:30b"]);
    }

    // FINDING 2 — the horizon must not confuse a long BOUNDED keep_alive with `-1`.

    #[test]
    fn the_pin_horizon_clears_the_sentinel_and_every_plausible_bounded_keep_alive() {
        let now = 1_800_000_000i64;
        let h = DEFAULT_PIN_HORIZON_DAYS;
        // Margin ABOVE: the sentinel is ~29x further out than the horizon.
        assert!(SENTINEL_SECS / (h * 86_400) >= 29);
        assert!(is_indefinite_pin(Some(&at(now, SENTINEL_SECS)), now, h));
        // Margin BELOW: an absurd but DELIBERATE bounded keep_alive of one year
        // (`8760h`) is 10x inside the horizon and must never be unloaded…
        assert_eq!(h / 365, 10);
        assert!(!is_indefinite_pin(Some(&at(now, 365 * 86_400)), now, h));
        // …including one deliberately LONGER than the old 365-day default, which
        // is the exact misclassification this raise fixes.
        let four_hundred_days = 400 * 86_400;
        assert!(!is_indefinite_pin(Some(&at(now, four_hundred_days)), now, h));
        assert!(
            is_indefinite_pin(Some(&at(now, four_hundred_days)), now, 365),
            "regression witness: the OLD 365-day horizon misclassified this bounded keep_alive"
        );
        // The set's own 24h residency is 3650x inside the horizon.
        assert!(!is_indefinite_pin(Some(&at(now, 86_400)), now, h));
    }

    #[tokio::test]
    async fn a_long_bounded_keep_alive_is_never_unloaded_but_the_sentinel_is() {
        let now = 1_800_000_000i64;
        let env = FakeConvergeEnv::new(
            vec![
                ("sentinel:30b".to_string(), Some(at(now, SENTINEL_SECS))),
                (
                    "long-bounded:8b".to_string(),
                    Some(at(now, 400 * 86_400)),
                ),
                ("ordinary:1b".to_string(), Some(at(now, 86_400))),
            ],
            &[],
            now,
        );
        let report = converge_once(&env).await;
        assert_eq!(report.found, 1);
        assert_eq!(env.unloaded(), vec!["sentinel:30b"]);
    }

    #[test]
    fn ollama_ps_exposes_no_better_provenance_than_the_expiry() {
        // Documents the measured evidence (0.22.1 live, 2026-07-31): every /api/ps
        // field is about WHAT is loaded, never HOW it was pinned. The classifier is
        // therefore a heuristic on purpose, and says so.
        assert!(PROVENANCE.contains("no keep_alive"));
        assert!(PROVENANCE.contains("only available signal"));
    }

    // FINDING 3 — bounded retry.

    #[test]
    fn the_retry_backoff_is_exponential_and_capped() {
        let p = ConvergeRetryPolicy::default();
        assert_eq!(p.backoff(0), Duration::from_secs(30));
        assert_eq!(p.backoff(1), Duration::from_secs(60));
        assert_eq!(p.backoff(2), Duration::from_secs(120));
        assert_eq!(p.backoff(3), Duration::from_secs(240));
        // Capped — it can never grow without bound.
        assert_eq!(p.backoff(20), p.max_backoff);
    }

    #[tokio::test]
    async fn a_failed_first_convergence_attempt_is_retried_and_succeeds() {
        use std::cell::RefCell;
        let calls = RefCell::new(0u32);
        let sleeps: RefCell<Vec<Duration>> = RefCell::new(Vec::new());
        let policy = ConvergeRetryPolicy::default();
        let (report, attempts) = converge_until_settled(
            &policy,
            || async {
                let mut c = calls.borrow_mut();
                *c += 1;
                if *c == 1 {
                    // Ollama was down at startup.
                    ConvergeReport {
                        status: ConvergeStatus::Unreachable,
                        found: 0,
                        unpinned: 0,
                        stranded: Vec::new(),
                        skipped_held: Vec::new(),
                    }
                } else {
                    ConvergeReport {
                        status: ConvergeStatus::Settled,
                        found: 1,
                        unpinned: 1,
                        stranded: Vec::new(),
                        skipped_held: Vec::new(),
                    }
                }
            },
            |d| {
                sleeps.borrow_mut().push(d);
                async {}
            },
        )
        .await;
        assert_eq!(attempts, 2, "retried exactly once, then stopped on success");
        assert_eq!(report.status, ConvergeStatus::Settled);
        assert_eq!(*calls.borrow(), 2);
        assert_eq!(sleeps.borrow().as_slice(), &[Duration::from_secs(30)]);
    }

    #[tokio::test]
    async fn exhausted_retries_stop_bounded_and_report_what_is_still_pinned() {
        use std::cell::RefCell;
        let calls = RefCell::new(0u32);
        let sleeps: RefCell<Vec<Duration>> = RefCell::new(Vec::new());
        let policy = ConvergeRetryPolicy {
            attempts: 3,
            base_backoff: Duration::from_secs(10),
            max_backoff: Duration::from_secs(15),
        };
        let (report, attempts) = converge_until_settled(
            &policy,
            || async {
                *calls.borrow_mut() += 1;
                ConvergeReport {
                    status: ConvergeStatus::Stranded,
                    found: 1,
                    unpinned: 0,
                    stranded: vec!["stuck:30b".to_string()],
                    skipped_held: Vec::new(),
                }
            },
            |d| {
                sleeps.borrow_mut().push(d);
                async {}
            },
        )
        .await;
        assert_eq!(attempts, 3, "BOUNDED: never more than the budget");
        assert_eq!(*calls.borrow(), 3);
        // One fewer sleep than attempts — it never sleeps after giving up.
        assert_eq!(
            sleeps.borrow().as_slice(),
            &[Duration::from_secs(10), Duration::from_secs(15)]
        );
        assert_eq!(report.stranded, vec!["stuck:30b".to_string()]);
    }

    #[tokio::test]
    async fn an_unconfigured_ollama_is_terminal_and_never_retried() {
        use std::cell::RefCell;
        let calls = RefCell::new(0u32);
        let (report, attempts) = converge_until_settled(
            &ConvergeRetryPolicy::default(),
            || async {
                *calls.borrow_mut() += 1;
                ConvergeReport {
                    status: ConvergeStatus::Unconfigured,
                    found: 0,
                    unpinned: 0,
                    stranded: Vec::new(),
                    skipped_held: Vec::new(),
                }
            },
            |_| async {},
        )
        .await;
        assert_eq!(attempts, 1);
        assert_eq!(*calls.borrow(), 1);
        assert_eq!(report.status, ConvergeStatus::Unconfigured);
    }

    // ── CHRD-PIN-01 round 4: the held-set TOCTOU, and the reconcile re-attempt ──

    /// A [`ConvergeEnv`] whose `held()` answer CHANGES between calls, so the window
    /// between the plan snapshot and each individual unload is directly testable.
    /// `held_script[0]` is what the snapshot sees; `held_script[i]` answers the i-th
    /// later call; the last entry repeats forever.
    struct ScriptedHeldEnv {
        loaded: Vec<(String, Option<String>)>,
        held_script: Vec<HashSet<String>>,
        held_calls: std::sync::atomic::AtomicUsize,
        now: i64,
        unloaded: StdMutex<Vec<String>>,
    }

    impl ScriptedHeldEnv {
        fn new(loaded: Vec<(String, Option<String>)>, script: &[&[&str]], now: i64) -> Self {
            ScriptedHeldEnv {
                loaded,
                held_script: script
                    .iter()
                    .map(|step| step.iter().map(|s| s.to_string()).collect())
                    .collect(),
                held_calls: std::sync::atomic::AtomicUsize::new(0),
                now,
                unloaded: StdMutex::new(Vec::new()),
            }
        }
        fn unloaded(&self) -> Vec<String> {
            self.unloaded.lock().unwrap().clone()
        }
        fn held_calls(&self) -> usize {
            self.held_calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl ConvergeEnv for ScriptedHeldEnv {
        async fn loaded(&self) -> Result<Vec<(String, Option<String>)>, ConvergeStatus> {
            Ok(self.loaded.clone())
        }
        async fn held(&self) -> HashSet<String> {
            let i = self.held_calls.fetch_add(1, Ordering::SeqCst);
            self.held_script[i.min(self.held_script.len() - 1)].clone()
        }
        async fn unload(&self, model: &str) -> bool {
            self.unloaded.lock().unwrap().push(model.to_string());
            true
        }
        fn now_epoch_secs(&self) -> i64 {
            self.now
        }
    }

    /// FIX 1 — the TOCTOU. The batch snapshot said nobody held `taken-over:8b`, but
    /// the resident set warmed it into a role while the loop was busy unloading the
    /// model ahead of it. Re-checking `held()` per-unload is the only thing that
    /// stops convergence from evicting a model a live role is holding.
    #[tokio::test]
    async fn a_model_that_becomes_held_between_the_snapshot_and_its_unload_is_not_unloaded() {
        let now = 1_800_000_000i64;
        let env = ScriptedHeldEnv::new(
            vec![
                ("stranded:30b".to_string(), Some(at(now, SENTINEL_SECS))),
                ("taken-over:8b".to_string(), Some(at(now, SENTINEL_SECS))),
            ],
            // snapshot: nothing held → BOTH are planned. Then the set warms
            // `taken-over:8b` into a role before the loop reaches it.
            &[&[], &["taken-over:8b"]],
            now,
        );
        let report = converge_once(&env).await;

        assert_eq!(
            report.found, 2,
            "both were legitimately planned at snapshot time"
        );
        assert_eq!(
            env.unloaded(),
            vec!["stranded:30b"],
            "the model the resident set took over mid-loop must NOT be unloaded"
        );
        assert_eq!(report.unpinned, 1);
        assert_eq!(report.skipped_held, vec!["taken-over:8b".to_string()]);
        assert!(report.stranded.is_empty(), "a skip is not a failure");
        assert_eq!(
            report.status,
            ConvergeStatus::Settled,
            "losing a target to a legitimate warm is a normal, TERMINAL outcome"
        );
        // 1 snapshot + 1 per target: the re-check is PER UNLOAD, not per batch.
        assert_eq!(env.held_calls(), 3);
    }

    /// FIX 1, the reverse ordering — chosen behaviour, stated: a model that was held
    /// at snapshot time and is RELEASED before the loop reaches it is left alone this
    /// pass. It was excluded from the plan and is never re-planned mid-loop.
    /// Unloading it would be harmless, but re-planning to catch it would mean growing
    /// the unload set from a momentary read — the exact hazard the per-unload
    /// re-check exists to remove. Missing it costs one more (idempotent) pass;
    /// over-reaching costs a live role its VRAM. We take the miss.
    #[tokio::test]
    async fn a_model_released_from_held_after_the_snapshot_is_left_for_a_later_pass_not_unloaded() {
        let now = 1_800_000_000i64;
        let env = ScriptedHeldEnv::new(
            vec![
                ("stranded:30b".to_string(), Some(at(now, SENTINEL_SECS))),
                ("was-held:8b".to_string(), Some(at(now, SENTINEL_SECS))),
            ],
            // snapshot: `was-held:8b` is held → excluded from the plan. It is then
            // released, but the loop does not go back for it.
            &[&["was-held:8b"], &[]],
            now,
        );
        let report = converge_once(&env).await;

        assert_eq!(report.found, 1, "only the unheld model was planned");
        assert_eq!(
            env.unloaded(),
            vec!["stranded:30b"],
            "a model released AFTER the snapshot is not chased mid-loop"
        );
        assert!(report.skipped_held.is_empty());
        assert_eq!(report.status, ConvergeStatus::Settled);
        // …and a later pass picks it up, because convergence is idempotent.
        let env2 = ScriptedHeldEnv::new(
            vec![("was-held:8b".to_string(), Some(at(now, SENTINEL_SECS)))],
            &[&[]],
            now,
        );
        assert_eq!(converge_once(&env2).await.unpinned, 1);
    }

    /// Positive control for FIX 1: with a stable held set, the per-unload re-check
    /// changes nothing — an ordinary settled pass still unloads EXACTLY the
    /// sentinel-pinned, unheld models and nothing else.
    #[tokio::test]
    async fn an_ordinary_settled_convergence_still_unloads_exactly_the_sentinel_pinned_unheld_models(
    ) {
        let now = 1_800_000_000i64;
        let env = ScriptedHeldEnv::new(
            vec![
                ("sentinel-a:30b".to_string(), Some(at(now, SENTINEL_SECS))),
                ("held-sentinel:8b".to_string(), Some(at(now, SENTINEL_SECS))),
                ("long-bounded:8b".to_string(), Some(at(now, 400 * 86_400))),
                ("ordinary:1b".to_string(), Some(at(now, 86_400))),
                ("sentinel-b:1b".to_string(), Some(at(now, SENTINEL_SECS))),
                ("no-expiry:1b".to_string(), None),
            ],
            &[&["held-sentinel:8b"]], // stable for the whole pass
            now,
        );
        let report = converge_once(&env).await;
        assert_eq!(report.status, ConvergeStatus::Settled);
        assert_eq!(report.found, 2);
        assert_eq!(report.unpinned, 2);
        assert!(report.skipped_held.is_empty());
        assert!(report.stranded.is_empty());
        assert_eq!(env.unloaded(), vec!["sentinel-a:30b", "sentinel-b:1b"]);
    }

    // FIX 2 — the reconcile-tick re-attempt.

    fn report_with(status: ConvergeStatus) -> ConvergeReport {
        ConvergeReport {
            status,
            found: 0,
            unpinned: 0,
            stranded: Vec::new(),
            skipped_held: Vec::new(),
        }
    }

    /// A non-terminal startup burst hands the problem to the reconcile loop: the
    /// gate arms, each tick makes ONE attempt, and it stops the moment one settles.
    #[tokio::test]
    async fn a_non_terminal_startup_outcome_re_attempts_on_the_reconcile_tick_and_stops_once_settled(
    ) {
        for burst in [ConvergeStatus::Unreachable, ConvergeStatus::Stranded] {
            let gate = ConvergeGate::new();
            // While the burst is still running the gate is PENDING — the reconcile
            // loop must not converge concurrently with it.
            assert_eq!(gate.state(), ConvergeGateState::Pending);
            assert!(converge_reattempt_tick(&gate, || async {
                panic!("must not attempt while the startup burst is still in flight")
            })
            .await
            .is_none());

            gate.record_startup_outcome(burst);
            assert_eq!(gate.state(), ConvergeGateState::Armed, "burst {burst:?}");

            // Tick 1: Ollama is still down.
            let calls = std::cell::RefCell::new(0u32);
            let r = converge_reattempt_tick(&gate, || async {
                *calls.borrow_mut() += 1;
                report_with(ConvergeStatus::Unreachable)
            })
            .await;
            assert_eq!(r.map(|r| r.status), Some(ConvergeStatus::Unreachable));
            assert_eq!(gate.state(), ConvergeGateState::Armed, "still not settled");

            // Tick 2: Ollama is back — it settles.
            let r = converge_reattempt_tick(&gate, || async {
                *calls.borrow_mut() += 1;
                report_with(ConvergeStatus::Settled)
            })
            .await;
            assert_eq!(r.map(|r| r.status), Some(ConvergeStatus::Settled));
            assert_eq!(gate.state(), ConvergeGateState::Done);
            assert_eq!(*calls.borrow(), 2);

            // Tick 3+: never again, forever.
            for _ in 0..100 {
                assert!(converge_reattempt_tick(&gate, || async {
                    panic!("a settled convergence must never be re-attempted")
                })
                .await
                .is_none());
            }
        }
    }

    /// A TERMINAL startup outcome must never arm the reconcile re-attempt: there is
    /// nothing left to do (`Settled`) or nothing a retry could fix (`Unconfigured`).
    #[tokio::test]
    async fn a_terminal_startup_outcome_never_re_arms_the_reconcile_re_attempt() {
        for terminal in [ConvergeStatus::Settled, ConvergeStatus::Unconfigured] {
            let gate = ConvergeGate::new();
            gate.record_startup_outcome(terminal);
            assert_eq!(gate.state(), ConvergeGateState::Done, "{terminal:?}");
            assert!(!gate.should_attempt());
            for _ in 0..50 {
                assert!(converge_reattempt_tick(&gate, || async {
                    panic!("a terminal startup outcome must never be re-attempted: {terminal:?}")
                })
                .await
                .is_none());
            }
            // And `Done` is ABSORBING — a stray non-terminal report cannot re-arm it.
            gate.record_attempt(ConvergeStatus::Unreachable);
            gate.record_startup_outcome(ConvergeStatus::Stranded);
            assert_eq!(gate.state(), ConvergeGateState::Done);
            assert!(!gate.should_attempt());
        }
    }

    /// NO STORM: however many ticks pass while convergence keeps failing, each tick
    /// costs EXACTLY ONE attempt — the rate limit is the reconcile interval itself,
    /// and there is no inner retry loop.
    #[tokio::test]
    async fn the_reconcile_re_attempt_makes_at_most_one_convergence_attempt_per_tick() {
        let gate = ConvergeGate::new();
        gate.record_startup_outcome(ConvergeStatus::Stranded);
        let calls = std::cell::RefCell::new(0u32);
        for tick in 1..=25u32 {
            converge_reattempt_tick(&gate, || async {
                *calls.borrow_mut() += 1;
                report_with(ConvergeStatus::Stranded)
            })
            .await;
            assert_eq!(
                *calls.borrow(),
                tick,
                "tick {tick} must cost exactly one convergence attempt, never a burst"
            );
        }
        assert_eq!(gate.state(), ConvergeGateState::Armed);
    }

    /// The two halves compose: the bounded burst is still bounded, and what it
    /// hands to the reconcile loop is bounded per tick. Worst case = burst budget +
    /// one attempt per reconcile interval, never a hot loop.
    #[tokio::test]
    async fn the_startup_burst_stays_bounded_and_the_reconcile_re_attempt_takes_over() {
        use std::cell::RefCell;
        let burst_calls = RefCell::new(0u32);
        let policy = ConvergeRetryPolicy::default();
        let (report, attempts) = converge_until_settled(
            &policy,
            || async {
                *burst_calls.borrow_mut() += 1;
                report_with(ConvergeStatus::Unreachable)
            },
            |_| async {},
        )
        .await;
        assert_eq!(attempts, policy.attempts, "the burst is BOUNDED, as before");
        assert_eq!(*burst_calls.borrow(), policy.attempts);

        // …and giving up no longer means giving up permanently.
        let gate = ConvergeGate::new();
        gate.record_startup_outcome(report.status);
        assert!(gate.should_attempt());
        let r = converge_reattempt_tick(&gate, || async { report_with(ConvergeStatus::Settled) })
            .await;
        assert_eq!(r.map(|r| r.status), Some(ConvergeStatus::Settled));
        assert_eq!(gate.state(), ConvergeGateState::Done);
    }

    #[test]
    fn the_retry_budget_is_hard_capped_regardless_of_configuration() {
        let p = ConvergeRetryPolicy {
            attempts: 10_000,
            ..ConvergeRetryPolicy::default()
        };
        // The driver clamps, so no env/config value can turn this into a loop.
        assert!(p.attempts.clamp(1, CONVERGE_MAX_ATTEMPTS) <= CONVERGE_MAX_ATTEMPTS);
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

    // ── CHRD-83: an alias repoint must not ORPHAN the outgoing model ────────
    //
    // Live, not hypothetical: within 20 minutes of the CHRD-PIN-01 deploy the
    // promoter repointed `lumina-fast` from a 24.64 GB model to a 9.59 GB one.
    // personality and router correctly followed; the 24.64 GB model stayed
    // LOADED, held by no role, carrying the 24h keep_alive the set had given it —
    // VRAM went 31.7 GB → 41.3 GB and stayed there.

    /// Personality and router BOTH move off the old model ⇒ nothing holds it
    /// afterwards ⇒ it is unloaded. This is the live failure verbatim.
    #[tokio::test]
    async fn alias_repoint_unloads_the_outgoing_model_when_no_role_still_holds_it() {
        let set = ResidentSet::new(live_cfg());
        // The live shape: personality + router SHARE one model, embedding is its own.
        let shared = vec![
            resolved(Role::Personality, Some("granite:30b"), Some(24.64), true),
            resolved(Role::Router, Some("granite:30b"), Some(24.64), true),
            resolved(Role::Embedding, Some("embed:1"), Some(1.0), true),
        ];
        let (env, _entered) = FakeEnv::new(shared, Some(96.0));

        let first = set.warm_with(&*env, "startup", true, None).await;
        assert_eq!(
            first.warmed, 2,
            "the shared model is warmed ONCE (+ the embedder): {first:?}"
        );
        assert_eq!(first.shared, 1, "router shares personality's model");
        assert_eq!(first.orphaned, 0, "nothing was outgoing on the first pass");
        assert_eq!(env.exempt(), vec!["embed:1".to_string(), "granite:30b".to_string()]);

        // The promoter repoints `lumina-fast`: BOTH roles follow.
        {
            let mut r = env.resolved.lock().unwrap();
            r[0].model = Some("granite:8b".to_string());
            r[0].size_gb = Some(9.59);
            r[1].model = Some("granite:8b".to_string());
            r[1].size_gb = Some(9.59);
        }

        let report = set.reconcile_with(&*env).await;

        assert_eq!(report.warmed, 1, "the new target is warmed: {report:?}");
        assert_eq!(
            report.orphaned, 1,
            "the outgoing model must be unloaded, not left to a 24h expiry: {report:?}"
        );
        assert_eq!(
            env.unloads(),
            vec!["granite:30b".to_string()],
            "exactly the outgoing model, exactly once: {:?}",
            env.unloads()
        );
        assert_eq!(
            env.exempt(),
            vec!["embed:1".to_string(), "granite:8b".to_string()],
            "and the exemption follows the new target"
        );
        let status = set.status().await;
        assert!(
            status.roles.iter().all(|r| r.warm),
            "every role must still be held after the repoint: {:?}",
            status.roles
        );
    }

    /// **The regression this is most likely to grow.** personality and router
    /// share a model in the LIVE configuration. When only ONE of them repoints,
    /// the other still holds the old model — it must NOT be unloaded out from
    /// under the role that is still using it.
    #[tokio::test]
    async fn a_repoint_never_unloads_a_model_another_role_still_holds() {
        let set = ResidentSet::new(live_cfg());
        let shared = vec![
            resolved(Role::Personality, Some("granite:30b"), Some(24.64), true),
            resolved(Role::Router, Some("granite:30b"), Some(24.64), true),
            resolved(Role::Embedding, Some("embed:1"), Some(1.0), true),
        ];
        let (env, _entered) = FakeEnv::new(shared, Some(96.0));

        set.warm_with(&*env, "startup", true, None).await;

        // ONLY personality repoints. Router stays on the big model.
        {
            let mut r = env.resolved.lock().unwrap();
            r[0].model = Some("granite:8b".to_string());
            r[0].size_gb = Some(9.59);
        }

        let report = set.reconcile_with(&*env).await;

        assert_eq!(
            report.orphaned, 0,
            "a model another role still holds is NOT an orphan: {report:?}"
        );
        assert!(
            env.unloads().is_empty(),
            "unloading a still-held model would tear residency out from under the router: {:?}",
            env.unloads()
        );
        let mut want = vec!["embed:1".to_string(), "granite:30b".to_string(), "granite:8b".to_string()];
        want.sort();
        assert_eq!(env.exempt(), want, "both models stay exempt");
        let status = set.status().await;
        assert!(status.roles.iter().all(|r| r.warm));
    }

    /// The outgoing model becomes HELD AGAIN between the decision and the
    /// unload ⇒ that unload is SKIPPED (logged at INFO), not treated as an error
    /// and not issued.
    ///
    /// Driven deterministically without sleeps or races: three roles repoint at
    /// once, so there are three orphans processed in order, and the env flips the
    /// SECOND orphan back to held from inside the FIRST orphan's unload call —
    /// which is precisely the window the per-model re-check exists to cover
    /// (`commit` holds no lock across an unload, so a warm genuinely can land
    /// there). The env then delegates, so if the re-check were removed the
    /// delegate would record the unload and this test would fail.
    #[tokio::test]
    async fn an_outgoing_model_that_becomes_held_again_is_skipped_not_unloaded() {
        struct HeldAgainEnv {
            inner: Arc<FakeEnv>,
            set: StdMutex<Option<Arc<ResidentSet>>>,
        }

        #[async_trait]
        impl ResidentEnv for HeldAgainEnv {
            async fn resolve(&self, a: &[(Role, String)]) -> Vec<Resolved> {
                self.inner.resolve(a).await
            }
            fn free_vram_gb(&self) -> Option<f64> {
                self.inner.free_vram_gb()
            }
            async fn warm_one(
                &self,
                role: Role,
                model: &str,
                ka: &str,
                c: &CancelToken,
            ) -> Result<(), String> {
                self.inner.warm_one(role, model, ka, c).await
            }
            async fn unload_one(&self, role: Role, model: &str) -> Result<(), String> {
                // Fire exactly once, from inside the FIRST orphan unload: make
                // the NEXT orphan held again before its re-check runs.
                let set = self.set.lock().unwrap().take();
                if let Some(set) = set {
                    let mut inner = set.inner.lock().await;
                    if let Some((_, slot)) = inner
                        .slots
                        .iter_mut()
                        .find(|(r, _)| *r == Role::Personality)
                    {
                        slot.model = Some("router:1".to_string());
                        slot.state = RoleState::Warm;
                    }
                }
                self.inner.unload_one(role, model).await
            }
            async fn set_exempt(&self, models: &[String]) {
                self.inner.set_exempt(models).await;
            }
            async fn clear_exempt(&self) {
                self.inner.clear_exempt().await;
            }
        }

        let set = Arc::new(ResidentSet::new(live_cfg()));
        let (fake, _entered) = FakeEnv::new(three_roles(), Some(96.0));

        let plain = HeldAgainEnv {
            inner: fake.clone(),
            set: StdMutex::new(None),
        };
        let first = set.warm_with(&plain, "startup", true, None).await;
        assert_eq!(first.warmed, 3);

        // All three roles repoint at once ⇒ three outgoing models.
        {
            let mut r = fake.resolved.lock().unwrap();
            r[0].model = Some("voice:2".to_string());
            r[1].model = Some("router:2".to_string());
            r[2].model = Some("embed:2".to_string());
        }

        let armed = HeldAgainEnv {
            inner: fake.clone(),
            set: StdMutex::new(Some(set.clone())),
        };
        let report = set.reconcile_with(&armed).await;

        assert!(
            !fake.unloads().contains(&"router:1".to_string()),
            "a model that became held again between the decision and the unload must be SKIPPED: {:?}",
            fake.unloads()
        );
        assert_eq!(
            fake.unloads(),
            vec!["voice:1".to_string(), "embed:1".to_string()],
            "the other orphans are still unloaded, in order: {:?}",
            fake.unloads()
        );
        assert_eq!(
            report.orphaned, 2,
            "a skipped orphan is not counted as reaped: {report:?}"
        );
        // A skip is not an error: the pass completed normally.
        assert!(!report.discarded, "a skipped orphan unload is not a failure");
        assert_eq!(report.warmed, 3, "and every new target was still warmed");
    }

    /// An unload that FAILS is soft: logged, the reconcile completes normally,
    /// and role state is still correct (the roles follow the new target).
    #[tokio::test]
    async fn a_failed_orphan_unload_is_soft_and_does_not_wedge_the_reconcile() {
        struct FailingUnloadEnv(Arc<FakeEnv>);

        #[async_trait]
        impl ResidentEnv for FailingUnloadEnv {
            async fn resolve(&self, a: &[(Role, String)]) -> Vec<Resolved> {
                self.0.resolve(a).await
            }
            fn free_vram_gb(&self) -> Option<f64> {
                self.0.free_vram_gb()
            }
            async fn warm_one(
                &self,
                role: Role,
                model: &str,
                ka: &str,
                c: &CancelToken,
            ) -> Result<(), String> {
                self.0.warm_one(role, model, ka, c).await
            }
            async fn unload_one(&self, role: Role, model: &str) -> Result<(), String> {
                let _ = self.0.unload_one(role, model).await;
                Err("unload request failed".to_string())
            }
            async fn set_exempt(&self, models: &[String]) {
                self.0.set_exempt(models).await;
            }
            async fn clear_exempt(&self) {
                self.0.clear_exempt().await;
            }
        }

        let set = ResidentSet::new(live_cfg());
        let shared = vec![
            resolved(Role::Personality, Some("granite:30b"), Some(24.64), true),
            resolved(Role::Router, Some("granite:30b"), Some(24.64), true),
            resolved(Role::Embedding, Some("embed:1"), Some(1.0), true),
        ];
        let (fake, _entered) = FakeEnv::new(shared, Some(96.0));
        let env = FailingUnloadEnv(fake.clone());

        set.warm_with(&env, "startup", true, None).await;
        {
            let mut r = fake.resolved.lock().unwrap();
            r[0].model = Some("granite:8b".to_string());
            r[1].model = Some("granite:8b".to_string());
        }

        let report = set.reconcile_with(&env).await;

        assert!(
            fake.unloads().contains(&"granite:30b".to_string()),
            "the unload must have been ATTEMPTED: {:?}",
            fake.unloads()
        );
        assert_eq!(
            report.orphaned, 0,
            "a failed unload is not counted as an orphan reaped: {report:?}"
        );
        assert!(!report.discarded, "a failed unload must not discard the pass");
        assert_eq!(report.warmed, 1, "the new target was still warmed");
        let status = set.status().await;
        assert!(
            status.active && status.roles.iter().all(|r| r.warm),
            "role state must still be correct after a failed unload: {status:?}"
        );
        assert_eq!(
            status.exempt,
            vec!["embed:1".to_string(), "granite:8b".to_string()],
            "and the exemption still follows the new target"
        );
    }

    /// **POSITIVE CONTROL.** A reconcile with NO target change must unload
    /// nothing and stay the steady-state no-op three consecutive live ticks
    /// currently show (`warmed=0 retained=N dropped=0`). If CHRD-83 ever turns
    /// the steady state into a churn loop, this is what catches it.
    #[tokio::test]
    async fn a_reconcile_with_no_target_change_stays_a_no_op_and_unloads_nothing() {
        let set = ResidentSet::new(live_cfg());
        let shared = vec![
            resolved(Role::Personality, Some("granite:30b"), Some(24.64), true),
            resolved(Role::Router, Some("granite:30b"), Some(24.64), true),
            resolved(Role::Embedding, Some("embed:1"), Some(1.0), true),
        ];
        let (env, _entered) = FakeEnv::new(shared, Some(96.0));

        set.warm_with(&*env, "startup", true, None).await;
        let after_startup = env.warm_calls().len();

        for tick in 0..3 {
            let report = set.reconcile_with(&*env).await;
            assert_eq!(report.warmed, 0, "tick {tick}: no re-warm: {report:?}");
            assert_eq!(report.dropped, 0, "tick {tick}: nothing dropped: {report:?}");
            assert_eq!(report.failed, 0, "tick {tick}: nothing failed: {report:?}");
            assert_eq!(report.orphaned, 0, "tick {tick}: nothing orphaned: {report:?}");
            assert_eq!(report.repaired, 0, "tick {tick}: nothing repaired: {report:?}");
            assert_eq!(
                report.retained + report.shared,
                3,
                "tick {tick}: every role retained/shared: {report:?}"
            );
            assert!(
                env.unloads().is_empty(),
                "tick {tick}: a steady-state tick must never unload: {:?}",
                env.unloads()
            );
            assert_eq!(
                env.warm_calls().len(),
                after_startup,
                "tick {tick}: a steady-state tick must issue NO warm requests"
            );
        }
        assert_eq!(
            env.exempt(),
            vec!["embed:1".to_string(), "granite:30b".to_string()],
            "and residency is unchanged throughout"
        );
    }

    // ── CHRD-83 GUARD 5: the CHECK-TO-REQUEST window ────────────────────────
    //
    // The orphan decision is taken under `inner` and the lock is DROPPED before
    // `unload_one` is awaited, so the decision is only known-good at the moment
    // it was TAKEN. The tests below drive that exact window deterministically —
    // no sleeps, no probabilistic looping: the env parks INSIDE the unload it is
    // told to intercept, signals that it is parked, and waits for the test to
    // land the interfering transition and open a gate.
    //
    // The existing `an_outgoing_model_..._is_skipped_not_unloaded` test does NOT
    // cover this: it flips the SECOND orphan back to held from inside the FIRST
    // orphan's unload, so the transition is always observed by a LATER
    // iteration's re-check. Here the transition lands against the model whose
    // unload is already on the wire.

    /// An env that parks inside `unload_one` for ONE named model, so a test can
    /// land a transition strictly between the orphan check and the unload
    /// request. Every other call delegates to the [`FakeEnv`] underneath.
    struct RaceEnv {
        inner: Arc<FakeEnv>,
        target: String,
        entered_unload: mpsc::UnboundedSender<String>,
        /// Fired from `clear_exempt` — i.e. from INSIDE `release_with`'s lock
        /// section, after the generation bump and after every held slot has been
        /// marked released. That is what makes "the release has landed" an
        /// observable event rather than a sleep.
        released: mpsc::UnboundedSender<()>,
        gate: Semaphore,
        intercept: AtomicBool,
        /// Make the intercepted unload REPORT failure after it has been released.
        fail_target: AtomicBool,
    }

    impl RaceEnv {
        #[allow(clippy::type_complexity)]
        fn new(
            inner: Arc<FakeEnv>,
            target: &str,
        ) -> (
            Arc<RaceEnv>,
            mpsc::UnboundedReceiver<String>,
            mpsc::UnboundedReceiver<()>,
        ) {
            let (utx, urx) = mpsc::unbounded_channel();
            let (rtx, rrx) = mpsc::unbounded_channel();
            (
                Arc::new(RaceEnv {
                    inner,
                    target: target.to_string(),
                    entered_unload: utx,
                    released: rtx,
                    gate: Semaphore::new(0),
                    intercept: AtomicBool::new(false),
                    fail_target: AtomicBool::new(false),
                }),
                urx,
                rrx,
            )
        }
        /// Arm the interception. Off during the startup warm so only the pass
        /// under test is ever parked.
        fn arm(&self) {
            self.intercept.store(true, Ordering::SeqCst);
        }
        fn open_unload(&self) {
            self.gate.add_permits(1);
        }
        fn fail_target_unload(&self) {
            self.fail_target.store(true, Ordering::SeqCst);
        }
    }

    #[async_trait]
    impl ResidentEnv for RaceEnv {
        async fn resolve(&self, a: &[(Role, String)]) -> Vec<Resolved> {
            self.inner.resolve(a).await
        }
        fn free_vram_gb(&self) -> Option<f64> {
            self.inner.free_vram_gb()
        }
        async fn warm_one(
            &self,
            role: Role,
            model: &str,
            ka: &str,
            c: &CancelToken,
        ) -> Result<(), String> {
            self.inner.warm_one(role, model, ka, c).await
        }
        async fn unload_one(&self, role: Role, model: &str) -> Result<(), String> {
            if self.intercept.load(Ordering::SeqCst) && model == self.target {
                // We are now strictly BETWEEN the orphan re-check (which passed)
                // and the request landing.
                let _ = self.entered_unload.send(model.to_string());
                self.gate.acquire().await.expect("unload gate").forget();
                if self.fail_target.load(Ordering::SeqCst) {
                    // Record the ATTEMPT, then report failure — the shape of a
                    // transport error or a timeout, where whether the request
                    // landed is exactly what we do not know.
                    let _ = self.inner.unload_one(role, model).await;
                    return Err("unload request failed".to_string());
                }
            }
            self.inner.unload_one(role, model).await
        }
        async fn set_exempt(&self, models: &[String]) {
            self.inner.set_exempt(models).await;
        }
        async fn clear_exempt(&self) {
            self.inner.clear_exempt().await;
            let _ = self.released.send(());
        }
    }

    /// The live shape (personality + router share the big model), warmed, then
    /// BOTH repointed — so `granite:30b` is the single orphan and the env is
    /// armed to park inside its unload.
    #[allow(clippy::type_complexity)]
    async fn armed_repoint_race() -> (
        Arc<ResidentSet>,
        Arc<FakeEnv>,
        Arc<RaceEnv>,
        mpsc::UnboundedReceiver<String>,
        mpsc::UnboundedReceiver<()>,
    ) {
        let set = Arc::new(ResidentSet::new(live_cfg()));
        let shared = vec![
            resolved(Role::Personality, Some("granite:30b"), Some(24.64), true),
            resolved(Role::Router, Some("granite:30b"), Some(24.64), true),
            resolved(Role::Embedding, Some("embed:1"), Some(1.0), true),
        ];
        let (fake, _entered) = FakeEnv::new(shared, Some(96.0));
        let (env, unload_rx, released_rx) = RaceEnv::new(fake.clone(), "granite:30b");

        let first = set.warm_with(&*env, "startup", true, None).await;
        assert_eq!(first.orphaned, 0, "nothing outgoing on the first pass");
        assert_eq!(first.repaired, 0, "nothing to repair on the first pass");

        {
            let mut r = fake.resolved.lock().unwrap();
            r[0].model = Some("granite:8b".to_string());
            r[0].size_gb = Some(9.59);
            r[1].model = Some("granite:8b".to_string());
            r[1].size_gb = Some(9.59);
        }
        env.arm();
        (set, fake, env, unload_rx, released_rx)
    }

    /// REQUIREMENT 1 as an assertion: after the pass, no role may claim
    /// residency for a model the pass unloaded, and no such model may still
    /// carry the residency exemption.
    async fn assert_claims_no_residency_for_unloaded(set: &ResidentSet, unloaded: &[String]) {
        let status = set.status().await;
        for r in status.roles.iter() {
            if !r.warm {
                continue;
            }
            if let Some(m) = r.model.as_deref() {
                assert!(
                    !unloaded.iter().any(|u| u == m),
                    "role {:?} still CLAIMS residency for {m}, which this pass unloaded — \
                     the resident set and the GPU now disagree until the next reconcile: {:?}",
                    r.role,
                    status.roles
                );
            }
        }
        for m in unloaded {
            assert!(
                !status.exempt.iter().any(|e| e == m),
                "{m} was unloaded but is still eviction-exempt: {:?}",
                status.exempt
            );
        }
    }

    /// **THE RACE TEST.** A warm makes the outgoing model HELD AGAIN strictly
    /// between the orphan re-check and the unload landing. The unload is already
    /// on the wire and cannot be recalled, so the only correct outcome is that
    /// the set does not end the pass believing it holds it.
    ///
    /// Fails against the pre-fix branch: there the re-check is the last word, so
    /// the slot keeps claiming `granite:30b` after that model has been unloaded.
    #[tokio::test]
    async fn a_rehold_between_the_check_and_the_unload_never_leaves_a_false_residency() {
        let (set, fake, env, mut unload_rx, _released_rx) = armed_repoint_race().await;

        let (s, e) = (set.clone(), env.clone());
        let pass = tokio::spawn(async move { s.reconcile_with(&*e).await });

        let parked = unload_rx.recv().await.expect("parked inside the unload");
        assert_eq!(parked, "granite:30b");

        // THE WINDOW. A warm has landed and this role now claims the very model
        // whose unload is in flight — the same mutation the sibling test uses,
        // aimed at the model already being unloaded rather than the next one.
        {
            let mut inner = set.inner.lock().await;
            let (_, slot) = inner
                .slots
                .iter_mut()
                .find(|(r, _)| *r == Role::Personality)
                .expect("personality slot");
            slot.model = Some("granite:30b".to_string());
            slot.state = RoleState::Warm;
        }
        // A warm that re-held the model would also have re-exempted it (the
        // commit applies the exemption wholesale from what it holds), so model
        // that too — otherwise the repair's exemption recompute would be
        // vacuously correct here and the test could not tell whether it ran.
        fake.set_exempt(&[
            "embed:1".to_string(),
            "granite:30b".to_string(),
            "granite:8b".to_string(),
        ])
        .await;
        env.open_unload();

        let report = join_unblocked(pass, "the reconcile pass").await;

        assert_eq!(
            fake.unloads(),
            vec!["granite:30b".to_string()],
            "the unload really did land — this is a genuine race, not a skip: {:?}",
            fake.unloads()
        );
        assert_eq!(
            report.repaired, 1,
            "the stale residency claim must be REPAIRED: {report:?}"
        );
        assert_claims_no_residency_for_unloaded(&set, &["granite:30b".to_string()]).await;
        assert_eq!(
            fake.exempt(),
            vec!["embed:1".to_string(), "granite:8b".to_string()],
            "the registry exemption tracks the repaired slots, not the unloaded model: {:?}",
            fake.exempt()
        );

        // And the repair must leave the role RE-WARMABLE, not wedged: the next
        // reconcile sees a slot pointing at the outgoing model while the alias
        // points at the new one, so it forces past the debounce and the role is
        // held again on the very next tick.
        let after = set.reconcile_with(&*fake).await;
        assert!(!after.discarded, "the follow-up tick ran: {after:?}");
        let status = set.status().await;
        assert!(
            status.roles.iter().all(|r| r.warm),
            "every role is held again after the next tick: {:?}",
            status.roles
        );
        assert_eq!(
            fake.unloads(),
            vec!["granite:30b".to_string()],
            "and the repair does not itself cause churn — no second unload: {:?}",
            fake.unloads()
        );
    }

    /// **THE RACE TEST, release edition.** A `release` bumps the generation
    /// strictly between the orphan re-check and the unload landing. Release must
    /// still WIN (TRTR-07d): the set ends released, nothing claims residency for
    /// the unloaded model, and — the part a careless fix gets wrong — the pass
    /// must NOT re-apply an exemption the release just dropped.
    #[tokio::test]
    async fn a_release_between_the_check_and_the_unload_still_wins() {
        let (set, fake, env, mut unload_rx, mut released_rx) = armed_repoint_race().await;

        let (s, e) = (set.clone(), env.clone());
        let pass = tokio::spawn(async move { s.reconcile_with(&*e).await });

        let parked = unload_rx.recv().await.expect("parked inside the unload");
        assert_eq!(parked, "granite:30b");

        // THE WINDOW. A mode-swap release lands. `clear_exempt` fires from inside
        // its lock section, so waiting on that receiver is a deterministic
        // "the generation has been bumped and the slots are released".
        let (s2, e2) = (set.clone(), env.clone());
        let rel = tokio::spawn(async move { s2.release_with(&*e2, "idle-lease").await });
        released_rx.recv().await.expect("the release landed");
        env.open_unload();

        let report = join_unblocked(pass, "the reconcile pass").await;
        let release = join_unblocked(rel, "the release").await;

        assert!(release.generation > 0, "the release bumped the generation");
        assert_eq!(
            fake.unloads(),
            vec!["granite:30b".to_string()],
            "the in-flight unload still landed: {:?}",
            fake.unloads()
        );
        assert_eq!(
            report.repaired, 0,
            "a release marks every slot released, so there is no false claim to repair: {report:?}"
        );
        assert_claims_no_residency_for_unloaded(&set, &["granite:30b".to_string()]).await;

        let status = set.status().await;
        assert!(!status.active, "the release wins: the set is not active");
        assert!(
            status.roles.iter().all(|r| !r.warm),
            "no role is held after a release: {:?}",
            status.roles
        );
        assert!(
            fake.exempt().is_empty(),
            "the superseded pass must NOT resurrect the registry exemption the release dropped: {:?}",
            fake.exempt()
        );
    }

    /// Wraps [`RaceEnv`] and, once armed, additionally parks inside `set_exempt`.
    /// That opens the window the repair now has BY CONSTRUCTION: it decides the
    /// exemption under `inner`, DROPS the lock, and only then applies it. A test
    /// can therefore land a release strictly between the drop and the install.
    struct ExemptRaceEnv {
        inner: Arc<RaceEnv>,
        entered_exempt: mpsc::UnboundedSender<Vec<String>>,
        gate: Semaphore,
        park: AtomicBool,
    }

    impl ExemptRaceEnv {
        fn new(inner: Arc<RaceEnv>) -> (Arc<ExemptRaceEnv>, mpsc::UnboundedReceiver<Vec<String>>) {
            let (tx, rx) = mpsc::unbounded_channel();
            (
                Arc::new(ExemptRaceEnv {
                    inner,
                    entered_exempt: tx,
                    gate: Semaphore::new(0),
                    park: AtomicBool::new(false),
                }),
                rx,
            )
        }
        /// Park the NEXT `set_exempt` only — the commit's wholesale install has
        /// already happened by the time a test arms this.
        fn arm_exempt(&self) {
            self.park.store(true, Ordering::SeqCst);
        }
        fn open_exempt(&self) {
            self.gate.add_permits(1);
        }
    }

    #[async_trait]
    impl ResidentEnv for ExemptRaceEnv {
        async fn resolve(&self, a: &[(Role, String)]) -> Vec<Resolved> {
            self.inner.resolve(a).await
        }
        fn free_vram_gb(&self) -> Option<f64> {
            self.inner.free_vram_gb()
        }
        async fn warm_one(
            &self,
            role: Role,
            model: &str,
            ka: &str,
            c: &CancelToken,
        ) -> Result<(), String> {
            self.inner.warm_one(role, model, ka, c).await
        }
        async fn unload_one(&self, role: Role, model: &str) -> Result<(), String> {
            self.inner.unload_one(role, model).await
        }
        async fn set_exempt(&self, models: &[String]) {
            if self.park.swap(false, Ordering::SeqCst) {
                let _ = self.entered_exempt.send(models.to_vec());
                self.gate.acquire().await.expect("exempt gate").forget();
            }
            self.inner.set_exempt(models).await;
        }
        async fn clear_exempt(&self) {
            self.inner.clear_exempt().await;
        }
    }

    /// **THE LOCK-DROP TEST.** The repair applies its recomputed exemption with
    /// `inner` RELEASED — that is the whole point of the shape (no await against
    /// an injected trait under the lock). This test asserts the window that opens
    /// is closed: a release lands *while the install is in flight*, and the pass
    /// must still not leave behind an exemption the release dropped.
    ///
    /// It also pins down the property the recompute gets for free: the list the
    /// repair installs is computed from the already-repaired slots, so it can
    /// never carry the unloaded model no matter when it lands.
    #[tokio::test]
    async fn a_release_during_the_repairs_exemption_install_still_wins() {
        let (set, fake, race, mut unload_rx, mut released_rx) = armed_repoint_race().await;
        let (env, mut exempt_rx) = ExemptRaceEnv::new(race.clone());

        let (s, e) = (set.clone(), env.clone());
        let pass = tokio::spawn(async move { s.reconcile_with(&*e).await });

        let parked = unload_rx.recv().await.expect("parked inside the unload");
        assert_eq!(parked, "granite:30b");

        // A warm re-holds the outgoing model (and re-exempts it, as a commit
        // would), so the repair has something real to fix and a real exemption to
        // recompute.
        {
            let mut inner = set.inner.lock().await;
            let (_, slot) = inner
                .slots
                .iter_mut()
                .find(|(r, _)| *r == Role::Personality)
                .expect("personality slot");
            slot.model = Some("granite:30b".to_string());
            slot.state = RoleState::Warm;
        }
        fake.set_exempt(&[
            "embed:1".to_string(),
            "granite:30b".to_string(),
            "granite:8b".to_string(),
        ])
        .await;

        env.arm_exempt();
        race.open_unload();

        // The repair has repaired the slots, dropped the lock, and is now INSIDE
        // `set_exempt`. This is the exact instant the old shape could not reach,
        // because it held `inner` across this await.
        let installing = exempt_rx.recv().await.expect("parked inside set_exempt");
        assert_eq!(
            installing,
            vec!["embed:1".to_string(), "granite:8b".to_string()],
            "the list is computed from the REPAIRED slots, so it cannot carry the unloaded model: {installing:?}"
        );

        // THE WINDOW. A mode-swap release lands while that install is in flight.
        let (s2, e2) = (set.clone(), race.clone());
        let rel = tokio::spawn(async move { s2.release_with(&*e2, "idle-lease").await });
        released_rx.recv().await.expect("the release landed");
        env.open_exempt();

        let report = join_unblocked(pass, "the reconcile pass").await;
        let release = join_unblocked(rel, "the release").await;

        assert!(release.generation > 0, "the release bumped the generation");
        assert_eq!(
            report.repaired, 1,
            "the stale residency claim was still repaired: {report:?}"
        );
        assert_claims_no_residency_for_unloaded(&set, &["granite:30b".to_string()]).await;

        let status = set.status().await;
        assert!(!status.active, "the release wins: the set is not active");
        assert!(
            status.roles.iter().all(|r| !r.warm),
            "no role is held after a release: {:?}",
            status.roles
        );
        assert!(
            fake.exempt().is_empty(),
            "dropping the lock before set_exempt must NOT let a superseded pass leave an exemption \
             the release dropped: {:?}",
            fake.exempt()
        );
        assert_eq!(
            fake.exempt_ops().last().map(String::as_str),
            Some("clear"),
            "the release, not the repair, must own the final word on the exemption: {:?}",
            fake.exempt_ops()
        );
    }

    /// The repair does NOT wait for the unload to report success. A transport
    /// error or a timeout cannot tell us whether the request landed, so the only
    /// answer safe under every interleaving is to stop claiming a residency we
    /// can no longer vouch for. The cost of being wrong here is one keep_alive
    /// re-assert against a model that is still loaded; the cost of the other
    /// choice is the divergence TRTR-07d eliminated.
    ///
    /// This test exists because it is the ONLY thing separating the shipped
    /// behaviour from the tempting narrower `if outcome.is_ok()` version.
    #[tokio::test]
    async fn a_rehold_is_repaired_even_when_the_unload_reports_failure() {
        let (set, fake, env, mut unload_rx, _released_rx) = armed_repoint_race().await;
        env.fail_target_unload();

        let (s, e) = (set.clone(), env.clone());
        let pass = tokio::spawn(async move { s.reconcile_with(&*e).await });

        assert_eq!(
            unload_rx.recv().await.expect("parked inside the unload"),
            "granite:30b"
        );
        {
            let mut inner = set.inner.lock().await;
            let (_, slot) = inner
                .slots
                .iter_mut()
                .find(|(r, _)| *r == Role::Personality)
                .expect("personality slot");
            slot.model = Some("granite:30b".to_string());
            slot.state = RoleState::Warm;
        }
        env.open_unload();

        let report = join_unblocked(pass, "the reconcile pass").await;

        assert_eq!(
            report.orphaned, 0,
            "a failed unload is still not counted as an orphan reaped: {report:?}"
        );
        assert_eq!(
            report.repaired, 1,
            "the stale claim is repaired regardless of what the unload REPORTED: {report:?}"
        );
        assert_claims_no_residency_for_unloaded(&set, &["granite:30b".to_string()]).await;
        assert!(
            !report.discarded,
            "and a failed unload still does not wedge the pass: {report:?}"
        );
        assert!(
            fake.unloads().contains(&"granite:30b".to_string()),
            "the unload was attempted: {:?}",
            fake.unloads()
        );
    }

    /// **POSITIVE CONTROL, through the same seam.** The gate is already open, so
    /// the unload runs with no concurrent transition at all: the orphan is still
    /// unloaded, nothing is repaired, and the remaining roles stay held.
    #[tokio::test]
    async fn an_orphan_unload_with_no_concurrent_transition_still_unloads_and_holds() {
        let (set, fake, env, mut unload_rx, _released_rx) = armed_repoint_race().await;
        env.open_unload();

        let report = set.reconcile_with(&*env).await;

        assert_eq!(
            unload_rx.recv().await.expect("the seam was exercised"),
            "granite:30b",
            "the control must go through the SAME intercepted path"
        );
        assert_eq!(report.warmed, 1, "the new target is warmed: {report:?}");
        assert_eq!(report.orphaned, 1, "the orphan is unloaded: {report:?}");
        assert_eq!(
            report.repaired, 0,
            "with no concurrent transition there is nothing to repair: {report:?}"
        );
        assert_eq!(
            fake.unloads(),
            vec!["granite:30b".to_string()],
            "exactly the outgoing model, exactly once: {:?}",
            fake.unloads()
        );
        let status = set.status().await;
        assert!(
            status.active && status.roles.iter().all(|r| r.warm),
            "every remaining role is still held: {status:?}"
        );
        assert_eq!(
            fake.exempt(),
            vec!["embed:1".to_string(), "granite:8b".to_string()],
            "and the exemption follows the new target: {:?}",
            fake.exempt()
        );
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
    /// the chat-style personality/router ones.
    ///
    /// This asserts the DELIBERATE role shaping, not a claim that `/api/generate`
    /// would fail: measured on live Ollama 0.22.1 (2026-07-31) `/api/generate`
    /// accepts an embedding model and its `keep_alive:0` really unloads it. The
    /// shaping is defensive portability across Ollama versions (older builds 400
    /// with "does not support generate"), and this test pins it so it is not
    /// dropped by accident.
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
            "the embedding role is deliberately addressed through the embed endpoint (defensive portability across Ollama versions — 0.22.1 would also accept /api/generate)"
        );
        assert_ne!(
            path, "/api/generate",
            "silently collapsing the embedding role onto the chat-shaped warm drops that portability and must be caught"
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

    // ── CHRD-PIN-01 review FINDINGS 1+2: unresolved ⇒ LOUD, and PIN NOTHING ──

    /// A `tracing` writer that captures emitted events into a buffer, so "degrades
    /// LOUDLY" is asserted as an observed log record rather than assumed from the
    /// presence of a `warn!` in the source.
    #[derive(Clone, Default)]
    struct LogCapture(Arc<StdMutex<Vec<u8>>>);

    impl LogCapture {
        fn text(&self) -> String {
            String::from_utf8_lossy(&self.0.lock().unwrap()).to_string()
        }
    }

    impl std::io::Write for LogCapture {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'w> tracing_subscriber::fmt::MakeWriter<'w> for LogCapture {
        type Writer = LogCapture;
        fn make_writer(&'w self) -> Self::Writer {
            self.clone()
        }
    }

    /// A `seam_state` whose three role aliases resolve to NOTHING (the alias keys
    /// are simply not configured, dynamically or statically).
    async fn unresolved_seam_state(base: &str) -> Arc<crate::routes::AppState> {
        std::env::set_var("OLLAMA_URL", base);
        std::env::remove_var("CHORD_VRAM_FREE_SYSFS_PATH");
        // The retired fallback read this. It is asserted here precisely because a
        // resurrection of the fallback would make this test pass a model through.
        std::env::set_var("EMBED_LOCAL_MODEL", "configured-embed-model:0.6b");
        let state = crate::routes::tests::test_state("http://mcp.invalid:3200".to_string());
        // Deliberately NO lumina_aliases.set(...) and no static alias map entries.
        let _ = seen_requests();
        state
    }

    /// **FINDING 1 + FINDING 2, end to end through the production wiring.** With
    /// every role's alias unresolved: every role must WARN by name, and the pass
    /// must warm nothing, exempt nothing, and put NO request on the wire — for the
    /// embedding role exactly as for the other two. The removed fallback made the
    /// embedding role warm and exempt a model anyway; a resurrection of it fails
    /// here (`EMBED_LOCAL_MODEL` is set, so the old code path would find a target).
    #[tokio::test]
    #[serial_test::serial]
    async fn production_unresolved_alias_warns_for_every_role_and_pins_nothing() {
        let server = httpmock::MockServer::start_async().await;
        let _stub = server
            .mock_async(|when, then| {
                when.matches(record_ollama_request);
                then.status(200).json_body(serde_json::json!({"ok": true}));
            })
            .await;
        let state = unresolved_seam_state(&server.base_url()).await;
        let set = ResidentSet::new(seam_cfg("24h"));

        let capture = LogCapture::default();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(capture.clone())
            .with_ansi(false)
            .with_max_level(tracing::Level::WARN)
            .finish();
        let report = {
            // `#[tokio::test]` is a current-thread runtime, so the thread-local
            // default subscriber stays installed across the awaits below.
            let _guard = tracing::subscriber::set_default(subscriber);
            set.warm(&state, "seam-unresolved", true).await
        };

        assert_eq!(report.warmed, 0, "an unresolved role must warm NOTHING");
        assert_eq!(report.failed, 0, "unresolved is not a warm failure");
        assert_eq!(report.skipped, 3, "all three roles must be skipped");
        assert!(
            seen_requests().is_empty(),
            "no Ollama request may be issued for a role with no target"
        );

        let status = set.status().await;
        assert!(
            status.roles.iter().all(|r| r.state == RoleState::Unresolved),
            "every role must land in unresolved: {:?}",
            status.roles
        );
        assert!(
            status.roles.iter().all(|r| r.model.is_none()),
            "no role may acquire a substituted model: {:?}",
            status.roles
        );
        assert!(
            status.exempt.is_empty(),
            "nothing may be exempted: {:?}",
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

        // …and it was LOUD, uniformly, naming role + alias + remedy.
        let logged = capture.text();
        for (role, alias) in [
            (Role::Personality, "lumina-deep"),
            (Role::Router, "lumina-fast"),
            (Role::Embedding, "lumina-embed"),
        ] {
            assert!(
                logged.contains(role.id()),
                "the {} role must be named in the warning:\n{logged}",
                role.id()
            );
            assert!(
                logged.contains(alias),
                "the failed alias {alias} must be named in the warning:\n{logged}"
            );
            assert!(
                logged.contains(role.alias_env()),
                "the warning must carry the actionable remedy for {}:\n{logged}",
                role.id()
            );
        }
        assert!(
            logged.contains("WARN"),
            "the degradation must be at WARN level, not whispered:\n{logged}"
        );
        assert!(
            !logged.contains("configured-embed-model"),
            "the retired EMBED_LOCAL_MODEL fallback must not reappear:\n{logged}"
        );
        std::env::remove_var("EMBED_LOCAL_MODEL");
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
