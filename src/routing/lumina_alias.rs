//! CHRD-91390429: dynamic lumina-proxy alias updater — a blended, assistant-fit
//! background task that repoints the three Lumina chat aliases (`lumina`,
//! `lumina-fast`, `lumina-deep`) at runtime, WITHOUT a restart.
//!
//! ## What this replaces
//! The three lumina alias targets used to be STATIC entries in
//! `CHORD_MODEL_ALIASES` (parsed once at startup, immutable). An interim ops
//! drop-in pinned them to a single hand-picked model. This module makes those
//! three (and ONLY those three) targets runtime-mutable: a background task
//! ranks the measured assistant fleet every `CHORD_ALIAS_REFRESH_SECS` and
//! hot-swaps the targets through an [`arc_swap::ArcSwap`], so the chat hot path
//! ([`crate::routes::chat_completions`]) resolves the current target with a
//! single lock-free `ArcSwap::load` — inference is NEVER blocked on the updater.
//! Every non-lumina alias stays in the static `CHORD_MODEL_ALIASES` map,
//! untouched.
//!
//! ## The blended metric (operator-chosen — see CHRD-91390429)
//! Each candidate model is scored as a weighted blend of three signals,
//! min-max **normalized to `[0,1]` across the surviving candidate set**:
//!   - `q` = `assistant_avg_value` from `model_dual_profile` — the operator's
//!     stated quality metric. A model has multiple `model_dual_profile` rows
//!     (one per backend_tag/mem_config); they are aggregated to ONE value by
//!     the **mean** of every row that has an assistant profile (documented
//!     choice — the mean, not the max, so a single lucky config can't inflate a
//!     model that is otherwise mediocre across its measured configs).
//!   - `a` = the dim-5 behavioral / prompted-adherence mean — the exact signal
//!     the already-built [`reporting::select_chat_role`] ranks by. Aggregated
//!     per model by the **max across its GUARD-ELIGIBLE rows only** (never a
//!     global max — an excluded backend's adherence must not stand in for the
//!     eligible one that actually qualifies the model). Composed INTO the blend
//!     rather than discarded, so both the operator's metric AND the built code's
//!     metric contribute.
//!   - `r` = responsiveness (tokens/sec) from `model_operational_profiles`
//!     (`throughput_at_2k`, falling back to the next-larger measured tier) —
//!     higher tok/s ⇒ higher `r`.
//!
//! ### Per-tier weights (env-configurable; defaults below)
//! `lumina` and `lumina-fast` favor responsiveness; `lumina-deep` favors
//! quality:
//! ```text
//! lumina / lumina-fast : 0.40*q + 0.30*a + 0.30*r
//! lumina-deep          : 0.60*q + 0.30*a + 0.10*r
//! ```
//!
//! ### Gates (a candidate is dropped BEFORE scoring if any fails)
//!   a. Not servable/known in the registry.
//!   b. Arch-excluded on all backends (per the RoutingMap serving exclusion —
//!      reuses [`crate::serving::profile::RoutingMap`]).
//!   c. Fails the existing latency/degradation guard that
//!      [`reporting::select_chat_role`] applies (we only accept models it marks
//!      [`GuardVerdict::Eligible`]).
//!   d. Below the assistant-quality floor `CHORD_ALIAS_MIN_QUALITY` (on `q`).
//!
//! ### Hysteresis
//! A tier only repoints when the new top model's blended score beats the CURRENT
//! target's blended score by ≥ `CHORD_ALIAS_SWITCH_MARGIN` (default 0.05) — this
//! prevents flapping between near-ties. A current target that is no longer a
//! candidate at all (e.g. it just became arch-excluded, or was never set) counts
//! as an unconditional loss, so the top candidate is adopted.
//!
//! ## Fail-safe
//! No assistant data / empty candidate set / DB unreachable ⇒ the current alias
//! targets are KEPT (never blanked), a warning is logged, and Chord's core proxy
//! function is unaffected — the same fail-open discipline as the serving-profile
//! routing map and the coding selector.
//!
//! ## No literals / secrets (S1/S7/S9)
//! The DB URL comes from [`terminus_rs::config::intake_database_url`] (same
//! source the coding selector / serving profile use); every threshold and weight
//! is read from an env var with a documented default. No host/IP/token literal
//! appears here.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use arc_swap::ArcSwap;
use async_trait::async_trait;

use terminus_rs::intake::assistant::reporting::{
    self, AssistantReport, GuardVerdict, ReportConfig,
};

/// The three lumina alias keys this updater owns. EVERY other alias in
/// `CHORD_MODEL_ALIASES` stays static and is never touched here.
pub const LUMINA_ALIAS_KEYS: [&str; 3] = ["lumina", "lumina-fast", "lumina-deep"];

// ─────────────────────────────────────────────────────────────────────────────
// Runtime-mutable store (lock-free hot-path reads)
// ─────────────────────────────────────────────────────────────────────────────

/// Runtime-mutable map of the three lumina alias keys → their current backend
/// model target, published through an [`ArcSwap`] so the chat hot path reads it
/// lock-free (`ArcSwap::load`) and the background updater repoints it without a
/// restart and without ever blocking a reader.
///
/// Cloning is cheap (an `Arc` bump) and every clone observes the same shared
/// cell, so the copy handed to the background task and the copy stored in
/// [`crate::routes::AppState`] stay in sync.
#[derive(Clone)]
pub struct LuminaAliasStore {
    inner: Arc<ArcSwap<HashMap<String, String>>>,
}

impl LuminaAliasStore {
    /// Seed from the static alias map: copy out ONLY the three lumina keys'
    /// current targets, so runtime resolution is byte-identical to the static
    /// `CHORD_MODEL_ALIASES` config until the updater first repoints a tier.
    pub fn from_static(aliases: &HashMap<String, String>) -> Self {
        let mut seed = HashMap::new();
        for key in LUMINA_ALIAS_KEYS {
            if let Some(target) = aliases.get(key) {
                seed.insert(key.to_string(), target.clone());
            }
        }
        LuminaAliasStore {
            inner: Arc::new(ArcSwap::from_pointee(seed)),
        }
    }

    /// An empty store (no lumina targets yet) — for tests and for the fail-open
    /// path when no static lumina alias was configured.
    pub fn empty() -> Self {
        LuminaAliasStore {
            inner: Arc::new(ArcSwap::from_pointee(HashMap::new())),
        }
    }

    /// Hot-path resolve: lock-free. Returns the current dynamic target for a
    /// lumina alias, or `None` when `model` is not one of the three lumina keys
    /// (the caller then falls through to the static alias map). This is the ONLY
    /// method the chat hot path calls — a single `ArcSwap::load` + map lookup.
    pub fn resolve(&self, model: &str) -> Option<String> {
        self.inner.load().get(model).cloned()
    }

    /// The current target for a specific lumina key (used by the updater for the
    /// hysteresis comparison and the repoint log line).
    pub fn current(&self, key: &str) -> Option<String> {
        self.inner.load().get(key).cloned()
    }

    /// Repoint ONE lumina key, leaving the other two untouched. Lock-free
    /// publish: [`ArcSwap::rcu`] clones the current map, updates the one key, and
    /// atomically swaps the new `Arc` in — concurrent readers see either the old
    /// or the new map, never a torn state.
    pub fn set(&self, key: &str, target: String) {
        self.inner.rcu(|current| {
            let mut next: HashMap<String, String> = (**current).clone();
            next.insert(key.to_string(), target.clone());
            next
        });
    }

    /// A snapshot of all current lumina targets (for tests / diagnostics).
    pub fn snapshot(&self) -> HashMap<String, String> {
        (*self.inner.load_full()).clone()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Config (env-driven weights + thresholds; documented defaults; no literals)
// ─────────────────────────────────────────────────────────────────────────────

/// The per-tier blend weights `w.q*q + w.a*a + w.r*r`. Not renormalized — the
/// operator's chosen values are used as-is (the defaults happen to sum to 1.0).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlendWeights {
    pub q: f64,
    pub a: f64,
    pub r: f64,
}

/// Full updater configuration, all env-driven with documented defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AliasUpdaterConfig {
    /// `CHORD_ALIAS_REFRESH_SECS` — tick interval (default 900).
    pub refresh_secs: u64,
    /// `CHORD_ALIAS_MIN_QUALITY` — minimum RAW `assistant_avg_value` a candidate
    /// must reach (gate d). `assistant_avg_value` is UN-normalized (observed in
    /// the hundreds, e.g. granite4.1 ≈ 489), so a meaningful threshold depends on
    /// the live distribution and cannot be known at build time. **Default 0.0 =
    /// DISABLED** (no quality filtering); an operator sets a real raw value (e.g.
    /// 50) from the observed distribution to exclude weak models.
    pub min_quality: f64,
    /// `CHORD_ALIAS_SWITCH_MARGIN` — hysteresis margin (default 0.05).
    pub switch_margin: f64,
    /// Weights for `lumina` and `lumina-fast` (responsiveness-favoring).
    pub fast_weights: BlendWeights,
    /// Weights for `lumina-deep` (quality-favoring).
    pub deep_weights: BlendWeights,
}

/// Lower bound on the refresh interval. `tokio::time::interval` PANICS on a
/// zero period, and a sub-minute alias flap makes no sense for a slow-moving
/// assistant ranking — so any smaller/zero value is floored to this.
const MIN_REFRESH_SECS: u64 = 60;

/// Parse a non-negative, finite f64 env value; anything malformed, NaN, ±inf, or
/// negative falls back to `default` (a NaN/inf/negative weight would corrupt or
/// invert the blend; a negative margin would defeat hysteresis).
fn env_nonneg_f64(key: &str, default: f64) -> f64 {
    match std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
    {
        Some(v) if v.is_finite() && v >= 0.0 => v,
        _ => default,
    }
}

/// Parse the refresh interval, floored to [`MIN_REFRESH_SECS`] so a `0`/tiny
/// value can never panic `tokio::time::interval`.
fn env_refresh_secs(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(default)
        .max(MIN_REFRESH_SECS)
}

impl AliasUpdaterConfig {
    /// Read the config from the environment, applying the documented defaults for
    /// any unset/malformed/invalid var (see [`env_nonneg_f64`] /
    /// [`env_refresh_secs`] — invalid weights/margins fall back to defaults, the
    /// interval is floored).
    pub fn from_env() -> Self {
        AliasUpdaterConfig {
            refresh_secs: env_refresh_secs("CHORD_ALIAS_REFRESH_SECS", 900),
            min_quality: env_nonneg_f64("CHORD_ALIAS_MIN_QUALITY", 0.0),
            switch_margin: env_nonneg_f64("CHORD_ALIAS_SWITCH_MARGIN", 0.05),
            fast_weights: BlendWeights {
                q: env_nonneg_f64("CHORD_ALIAS_W_FAST_Q", 0.40),
                a: env_nonneg_f64("CHORD_ALIAS_W_FAST_A", 0.30),
                r: env_nonneg_f64("CHORD_ALIAS_W_FAST_R", 0.30),
            },
            deep_weights: BlendWeights {
                q: env_nonneg_f64("CHORD_ALIAS_W_DEEP_Q", 0.60),
                a: env_nonneg_f64("CHORD_ALIAS_W_DEEP_A", 0.30),
                r: env_nonneg_f64("CHORD_ALIAS_W_DEEP_R", 0.10),
            },
        }
    }
}

impl Default for AliasUpdaterConfig {
    fn default() -> Self {
        AliasUpdaterConfig {
            refresh_secs: 900,
            min_quality: 0.0,
            switch_margin: 0.05,
            fast_weights: BlendWeights {
                q: 0.40,
                a: 0.30,
                r: 0.30,
            },
            deep_weights: BlendWeights {
                q: 0.60,
                a: 0.30,
                r: 0.10,
            },
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pure ranking core (no I/O — fully unit-testable with synthetic inputs)
// ─────────────────────────────────────────────────────────────────────────────

/// One model's raw (un-normalized, un-gated) blend inputs, extracted from the
/// assistant report + the operational-profile responsiveness map.
#[derive(Debug, Clone, PartialEq)]
pub struct RawSignal {
    pub model_id: String,
    /// Aggregated `assistant_avg_value` (mean of the model's `model_dual_profile`
    /// rows). `None` ⇒ no assistant-value data (fails the quality floor).
    pub q: Option<f64>, // RAW `assistant_avg_value` scale (un-normalized; ~hundreds).
    /// dim-5 behavioral/prompted-adherence mean, taken from a GUARD-ELIGIBLE
    /// backend row (max across the model's eligible rows only). Sourcing `a` from
    /// an eligible row — rather than a global max that could come from an
    /// excluded/ineligible backend — keeps it consistent with the backend that
    /// actually makes the model a candidate. `None` ⇒ no eligible row.
    pub a: Option<f64>,
    /// Responsiveness (tokens/sec) from `model_operational_profiles`. This table
    /// carries NO backend dimension (it is keyed per model_profile), so `r` is a
    /// model-level signal — unlike `a`/`guard`, it cannot be tied to a specific
    /// eligible backend. Documented limitation.
    pub r: Option<f64>,
    /// Whether the model cleared `select_chat_role`'s latency/degradation guard.
    pub guard_eligible: bool,
}

/// A gated candidate with concrete (missing→0.0) blend inputs, ready to rank.
#[derive(Debug, Clone, PartialEq)]
pub struct AliasCandidate {
    pub model_id: String,
    pub q: f64,
    pub a: f64,
    pub r: f64,
}

/// A scored candidate: the blended `score` plus the normalized components (kept
/// for logging / test introspection). Higher `score` is better.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredCandidate {
    pub model_id: String,
    pub score: f64,
    pub q_norm: f64,
    pub a_norm: f64,
    pub r_norm: f64,
}

/// The repoint decision for a single tier.
#[derive(Debug, Clone, PartialEq)]
pub enum Repoint {
    /// Repoint the tier's alias from `from` to `to`.
    Switch {
        from: Option<String>,
        to: String,
        new_score: f64,
        old_score: Option<f64>,
    },
    /// Leave the tier's alias unchanged. `reason` is logged at debug.
    Keep { reason: String },
}

/// Extract per-model [`RawSignal`]s from a built [`AssistantReport`] and a
/// responsiveness map. Pure. `q` is the MEAN of the model's assistant-profiled
/// `model_dual_profile` rows (RAW `assistant_avg_value` scale); `a` is the MAX
/// behavioral_mean across the model's GUARD-ELIGIBLE chat-role rows only (so an
/// excluded backend's adherence can't stand in for the eligible one);
/// `guard_eligible` is true if ANY chat-role row for the model cleared the guard.
pub fn extract_raw_signals(
    report: &AssistantReport,
    responsiveness: &HashMap<String, f64>,
) -> Vec<RawSignal> {
    // q: sum + count of assistant_avg_value across dual-profile rows → mean.
    let mut q_acc: HashMap<String, (f64, u32)> = HashMap::new();
    for row in &report.dual_profile {
        if row.has_assistant_profile {
            if let Some(v) = row.assistant_avg_value {
                let e = q_acc.entry(row.model_id.clone()).or_insert((0.0, 0));
                e.0 += v;
                e.1 += 1;
            }
        }
    }

    // a: max behavioral_mean over the model's GUARD-ELIGIBLE rows ONLY (not a
    // global max — a high-adherence but EXCLUDED backend row must not inflate a
    // model that only actually qualifies via a different, lower-adherence
    // backend). guard: any Eligible verdict per model.
    let mut a_eligible_max: HashMap<String, f64> = HashMap::new();
    let mut guard_ok: HashMap<String, bool> = HashMap::new();
    for cand in &report.chat_role.candidates {
        let is_eligible = matches!(cand.verdict, GuardVerdict::Eligible);
        if is_eligible {
            let a = a_eligible_max
                .entry(cand.key.model_id.clone())
                .or_insert(f64::NEG_INFINITY);
            if cand.behavioral_mean > *a {
                *a = cand.behavioral_mean;
            }
        }
        let g = guard_ok.entry(cand.key.model_id.clone()).or_insert(false);
        if is_eligible {
            *g = true;
        }
    }

    // Deterministic union of every model id we have any signal for.
    let mut ids: BTreeSet<String> = BTreeSet::new();
    ids.extend(q_acc.keys().cloned());
    ids.extend(guard_ok.keys().cloned());

    ids.into_iter()
        .map(|model_id| {
            let q =
                q_acc
                    .get(&model_id)
                    .and_then(|(sum, n)| if *n > 0 { Some(*sum / *n as f64) } else { None });
            // `a` from an eligible row only (see above).
            let a = a_eligible_max
                .get(&model_id)
                .copied()
                .filter(|v| v.is_finite());
            let r = responsiveness.get(&model_id).copied();
            let guard_eligible = guard_ok.get(&model_id).copied().unwrap_or(false);
            RawSignal {
                model_id,
                q,
                a,
                r,
                guard_eligible,
            }
        })
        .collect()
}

/// Apply the candidate gates and produce concrete-valued candidates.
///
/// Gates: (a+b) `eligible` membership — servable & not arch-excluded — computed
/// by the live caller from the registry + RoutingMap. `eligible` is an
/// `Option`: `Some(set)` ALWAYS filters (even an empty set drops everything —
/// the live path relies on this so an empty eligible set can't leak unservable
/// models); `None` disables the gate entirely (unit-test convenience only, never
/// used on the live path). (c) the chat-role guard (`guard_eligible`); (d) the
/// RAW-scale quality floor `min_quality` on `q` — a model with no `q` at all is
/// dropped (can't assess quality); the floor is disabled by default (0.0).
pub fn gate_and_build(
    raws: &[RawSignal],
    eligible: Option<&HashSet<String>>,
    min_quality: f64,
) -> Vec<AliasCandidate> {
    raws.iter()
        .filter_map(|raw| {
            // (a+b) servable & not arch-excluded — Some(set) always filters.
            if let Some(set) = eligible {
                if !set.contains(&raw.model_id) {
                    return None;
                }
            }
            // (c) chat-role latency/degradation guard.
            if !raw.guard_eligible {
                return None;
            }
            // (d) quality floor — requires a real q value at or above the floor.
            let q = raw.q?;
            if q < min_quality {
                return None;
            }
            Some(AliasCandidate {
                model_id: raw.model_id.clone(),
                q,
                a: raw.a.unwrap_or(0.0),
                r: raw.r.unwrap_or(0.0),
            })
        })
        .collect()
}

fn min_max<I: Iterator<Item = f64>>(vals: I) -> Option<(f64, f64)> {
    vals.fold(None, |acc, v| match acc {
        None => Some((v, v)),
        Some((lo, hi)) => Some((lo.min(v), hi.max(v))),
    })
}

/// Min-max normalize `v` into `[0,1]`. A degenerate range (all candidates equal
/// on this signal) maps to a neutral `0.5` — a constant signal cannot
/// discriminate, so it contributes the same to every candidate and does not
/// affect the ranking.
fn normalize(v: f64, lo: f64, hi: f64) -> f64 {
    if (hi - lo).abs() < 1e-12 {
        0.5
    } else {
        (v - lo) / (hi - lo)
    }
}

/// Normalize each signal across the candidate set and compute the weighted
/// blend, returning candidates sorted best-first (ties broken deterministically
/// by `model_id`). Pure.
pub fn rank(candidates: &[AliasCandidate], weights: BlendWeights) -> Vec<ScoredCandidate> {
    if candidates.is_empty() {
        return Vec::new();
    }
    let (q_lo, q_hi) = min_max(candidates.iter().map(|c| c.q)).unwrap();
    let (a_lo, a_hi) = min_max(candidates.iter().map(|c| c.a)).unwrap();
    let (r_lo, r_hi) = min_max(candidates.iter().map(|c| c.r)).unwrap();

    let mut scored: Vec<ScoredCandidate> = candidates
        .iter()
        .map(|c| {
            let q_norm = normalize(c.q, q_lo, q_hi);
            let a_norm = normalize(c.a, a_lo, a_hi);
            let r_norm = normalize(c.r, r_lo, r_hi);
            let score = weights.q * q_norm + weights.a * a_norm + weights.r * r_norm;
            ScoredCandidate {
                model_id: c.model_id.clone(),
                score,
                q_norm,
                a_norm,
                r_norm,
            }
        })
        .collect();

    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.model_id.cmp(&b.model_id))
    });
    scored
}

/// Apply hysteresis: pick the repoint decision for one tier given the ranked
/// candidates and the tier's CURRENT target. Only switches when the top
/// candidate beats the current target's score by ≥ `margin`. A current target
/// that is absent from the ranked set (unset, or now-ineligible) counts as an
/// unconditional loss, so the top candidate is adopted. Pure.
pub fn select_target(ranked: &[ScoredCandidate], current: Option<&str>, margin: f64) -> Repoint {
    let Some(top) = ranked.first() else {
        return Repoint::Keep {
            reason: "no eligible candidates — keeping current target".into(),
        };
    };

    // Already pointed at the best candidate → nothing to do.
    if current == Some(top.model_id.as_str()) {
        return Repoint::Keep {
            reason: format!("already pointed at top candidate '{}'", top.model_id),
        };
    }

    let current_score = current
        .and_then(|c| ranked.iter().find(|s| s.model_id == c))
        .map(|s| s.score);

    let beats = match current_score {
        Some(cs) => top.score >= cs + margin,
        // Current target isn't a candidate (unset or now-ineligible) → adopt top.
        None => true,
    };

    if beats {
        Repoint::Switch {
            from: current.map(str::to_string),
            to: top.model_id.clone(),
            new_score: top.score,
            old_score: current_score,
        }
    } else {
        Repoint::Keep {
            reason: format!(
                "top '{}' (score {:.3}) does not beat current '{}' (score {:.3}) by margin {:.3}",
                top.model_id,
                top.score,
                current.unwrap_or("<unset>"),
                current_score.unwrap_or(0.0),
                margin
            ),
        }
    }
}

/// A per-tier plan input: which alias key, which weights, and its current target.
#[derive(Debug, Clone, PartialEq)]
pub struct TierPlan {
    pub key: String,
    pub weights: BlendWeights,
    pub current: Option<String>,
}

/// Compute the repoint decision for every tier from one shared raw-signal set.
/// The gates are identical across tiers (they don't depend on the weights); only
/// the blend/ranking differs per tier's weights. Pure — the whole
/// decision core is testable without any I/O.
pub fn plan_repoints(
    raws: &[RawSignal],
    eligible: Option<&HashSet<String>>,
    cfg: &AliasUpdaterConfig,
    tiers: &[TierPlan],
) -> Vec<(String, Repoint)> {
    let candidates = gate_and_build(raws, eligible, cfg.min_quality);
    tiers
        .iter()
        .map(|tier| {
            let ranked = rank(&candidates, tier.weights);
            let repoint = select_target(&ranked, tier.current.as_deref(), cfg.switch_margin);
            (tier.key.clone(), repoint)
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Data source (assistant report + responsiveness) — abstracted, mirrors
// coding_selector::CodeProfileSource so tests use fixtures and only a live path
// hits Postgres.
// ─────────────────────────────────────────────────────────────────────────────

/// A source failure. Carries no infra detail (host/DSN) — same discipline as
/// [`crate::models::coding_selector::SelectorError`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AliasSourceError {
    NotConfigured,
    StoreUnavailable,
}

impl std::fmt::Display for AliasSourceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AliasSourceError::NotConfigured => {
                f.write_str("assistant-profile store is not configured")
            }
            AliasSourceError::StoreUnavailable => {
                f.write_str("assistant-profile store is temporarily unavailable")
            }
        }
    }
}

impl std::error::Error for AliasSourceError {}

/// Source of the assistant scoring inputs. Abstracted so unit tests inject
/// fixtures and only a gated live path touches the read-only intake DB.
#[async_trait]
pub trait AssistantAliasSource: Send + Sync {
    /// Build the ASMT-11 assistant report (provides `q` via `model_dual_profile`
    /// and `a`/guard via the chat-role selection) — reuses
    /// [`reporting::run_report`].
    async fn assistant_report(
        &self,
        cfg: &ReportConfig,
    ) -> Result<AssistantReport, AliasSourceError>;

    /// Per-model responsiveness (tokens/sec) from `model_operational_profiles`.
    /// A source with no operational data returns an empty map (responsiveness
    /// then degrades to `0` for every candidate — never an error).
    async fn responsiveness(&self) -> Result<HashMap<String, f64>, AliasSourceError>;
}

/// Production source: `assistant_report` reuses [`reporting::run_report`] (which
/// self-connects via `intake_database_url`, so no pool is needed for it); the
/// held pool serves the `model_operational_profiles` responsiveness query.
/// NO literal DSN/host — matches `DbCodeProfileSource` / `DbProfileSource`.
pub struct DbAssistantAliasSource {
    pool: sqlx::PgPool,
}

impl DbAssistantAliasSource {
    pub fn new(pool: sqlx::PgPool) -> Self {
        DbAssistantAliasSource { pool }
    }
}

#[async_trait]
impl AssistantAliasSource for DbAssistantAliasSource {
    async fn assistant_report(
        &self,
        cfg: &ReportConfig,
    ) -> Result<AssistantReport, AliasSourceError> {
        let (report, _md) = reporting::run_report(None, cfg).await.map_err(|e| {
            tracing::warn!(error = %e, "lumina alias updater: assistant report build failed");
            AliasSourceError::StoreUnavailable
        })?;
        Ok(report)
    }

    async fn responsiveness(&self) -> Result<HashMap<String, f64>, AliasSourceError> {
        use sqlx::Row;

        // Read-only. One row per model_profile; we coalesce the interactive
        // (small-context) throughput first, falling back to the next-larger
        // measured tier, and keep the highest tok/s on any duplicate model name.
        let rows = sqlx::query(
            "SELECT mp.model_name AS model_id, \
                    coalesce(op.throughput_at_2k, op.throughput_at_8k, op.throughput_at_16k, \
                             op.throughput_at_32k, op.throughput_at_64k) AS tok_s \
             FROM model_operational_profiles op \
             JOIN model_profiles mp ON mp.id = op.profile_id",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| {
            tracing::warn!(error = %e, "lumina alias updater: responsiveness query failed");
            AliasSourceError::StoreUnavailable
        })?;

        let mut out: HashMap<String, f64> = HashMap::new();
        for row in rows {
            let model_id: String = row.get("model_id");
            if let Ok(Some(tok_s)) = row.try_get::<Option<f64>, _>("tok_s") {
                let e = out.entry(model_id).or_insert(f64::NEG_INFINITY);
                if tok_s > *e {
                    *e = tok_s;
                }
            }
        }
        // Drop any NEG_INFINITY sentinels (shouldn't happen — guarded above).
        out.retain(|_, v| v.is_finite());
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Live orchestration
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the eligible (servable & has-a-usable-backend) model-id set, given a
/// non-empty `servable` name set and the serving RoutingMap, restricted to the
/// models we actually have signals for.
///
/// Gate (a) servable: the registry must know the model (the caller guarantees a
/// non-empty `servable` — an empty/not-ready registry is handled upstream by
/// keeping the current targets, never by scoring unservable models).
///
/// Gate (b) arch-exclusion — evaluated across the model's FULL backend set, not a
/// single row: a model is dropped ONLY when it is arch-excluded on ALL of its
/// available backends. `RoutingMap::load_from` collapses a model to its winning
/// row, preferring any usable (non-excluded) backend over an excluded one — so
/// the winning row is non-excluded IFF at least one backend is usable. A model
/// with NO routing entry at all (unprofiled) carries no arch verdict and is not
/// dropped on arch grounds. Thus: drop only when it HAS routing entries but the
/// winner is still excluded (⇒ every known backend is excluded).
fn compute_eligible(
    raws: &[RawSignal],
    servable: &HashSet<String>,
    rmap: &crate::serving::profile::RoutingMap,
) -> HashSet<String> {
    use terminus_rs::intake::serving::{ExclusionReason, ModelId};

    raws.iter()
        .filter_map(|raw| {
            // (a) servable per the registry.
            if !servable.contains(&raw.model_id) {
                return None;
            }
            // (b) usable on at least one backend.
            let mid = ModelId::from(raw.model_id.as_str());
            let entry = rmap.get(&mid);
            let has_routing = entry.is_some();
            let winner_usable = entry
                .map(|e| e.profile.exclusion_reason == ExclusionReason::None)
                .unwrap_or(false);
            let excluded_on_all_backends = has_routing && !winner_usable;
            if excluded_on_all_backends {
                return None;
            }
            Some(raw.model_id.clone())
        })
        .collect()
}

/// One refresh tick: fetch the assistant report + responsiveness, compute the
/// eligible set, plan the per-tier repoints, and apply them to the store,
/// logging every actual repoint at INFO. Fail-safe throughout: any fetch error
/// KEEPS the current targets (logs a warning) and returns.
pub async fn run_alias_refresh_tick(
    source: &dyn AssistantAliasSource,
    store: &LuminaAliasStore,
    registry: &Arc<tokio::sync::Mutex<crate::models::registry::ModelRegistry>>,
    routing: &Arc<tokio::sync::Mutex<crate::serving::profile::RoutingMap>>,
    cfg: &AliasUpdaterConfig,
) {
    let report = match source.assistant_report(&ReportConfig::default()).await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(
                "lumina alias updater: assistant report unavailable ({e}) — keeping current targets"
            );
            return;
        }
    };
    // Responsiveness is best-effort: a failure just means every candidate's r=0.
    let responsiveness = match source.responsiveness().await {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!(
                "lumina alias updater: responsiveness unavailable ({e}) — treating tok/s as unknown"
            );
            HashMap::new()
        }
    };

    let raws = extract_raw_signals(&report, &responsiveness);

    // FIX 4: servability must be established before we score anything. An
    // empty/not-yet-ready registry means we CANNOT judge which models are
    // servable — scoring then risks pointing a lumina alias at a model Chord
    // can't start. Fail-safe: keep the current targets, same posture as the
    // empty-candidate / DB-down paths.
    let servable: HashSet<String> = {
        let reg = registry.lock().await;
        reg.all_records().map(|r| r.name.clone()).collect()
    };
    if servable.is_empty() {
        tracing::warn!(
            "lumina alias updater: model registry not ready (no records) — keeping current \
             lumina targets"
        );
        return;
    }

    // Snapshot the routing map (drop the lock before any store writes).
    let eligible = {
        let rmap = routing.lock().await;
        compute_eligible(&raws, &servable, &rmap)
    };

    let tiers = vec![
        TierPlan {
            key: "lumina".to_string(),
            weights: cfg.fast_weights,
            current: store.current("lumina"),
        },
        TierPlan {
            key: "lumina-fast".to_string(),
            weights: cfg.fast_weights,
            current: store.current("lumina-fast"),
        },
        TierPlan {
            key: "lumina-deep".to_string(),
            weights: cfg.deep_weights,
            current: store.current("lumina-deep"),
        },
    ];

    for (key, repoint) in plan_repoints(&raws, Some(&eligible), cfg, &tiers) {
        match repoint {
            Repoint::Switch {
                from,
                to,
                new_score,
                ..
            } => {
                store.set(&key, to.clone());
                tracing::info!(
                    "lumina alias '{key}' repointed: {} -> {} (score {new_score:.3})",
                    from.as_deref().unwrap_or("<unset>"),
                    to
                );
            }
            Repoint::Keep { reason } => {
                tracing::debug!("lumina alias '{key}' unchanged: {reason}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(model: &str, q: Option<f64>, a: Option<f64>, r: Option<f64>, guard: bool) -> RawSignal {
        RawSignal {
            model_id: model.to_string(),
            q,
            a,
            r,
            guard_eligible: guard,
        }
    }

    // ── Store: lock-free resolve + isolated repoint ────────────────────────────

    #[test]
    fn store_seeds_only_lumina_keys_from_static() {
        let mut statics = HashMap::new();
        statics.insert("lumina".to_string(), "granite4.1:30b".to_string());
        statics.insert("lumina-fast".to_string(), "qwen3:8b".to_string());
        statics.insert("lumina-deep".to_string(), "qwen3:32b".to_string());
        statics.insert("some-other-alias".to_string(), "mystery:7b".to_string());

        let store = LuminaAliasStore::from_static(&statics);
        assert_eq!(store.resolve("lumina").as_deref(), Some("granite4.1:30b"));
        assert_eq!(store.resolve("lumina-fast").as_deref(), Some("qwen3:8b"));
        assert_eq!(store.resolve("lumina-deep").as_deref(), Some("qwen3:32b"));
        // Non-lumina aliases are NEVER captured by the dynamic store.
        assert_eq!(store.resolve("some-other-alias"), None);
        assert_eq!(store.snapshot().len(), 3);
    }

    #[test]
    fn store_set_repoints_one_key_only() {
        let mut statics = HashMap::new();
        statics.insert("lumina".to_string(), "a:1b".to_string());
        statics.insert("lumina-deep".to_string(), "b:2b".to_string());
        let store = LuminaAliasStore::from_static(&statics);

        store.set("lumina", "c:3b".to_string());
        assert_eq!(store.resolve("lumina").as_deref(), Some("c:3b"));
        // lumina-deep untouched.
        assert_eq!(store.resolve("lumina-deep").as_deref(), Some("b:2b"));
    }

    // ── Blend ranking: higher blended score wins ───────────────────────────────

    #[test]
    fn blend_picks_higher_blended_score() {
        // Two eligible candidates; with equal weights the one that dominates all
        // three normalized signals must rank first.
        let raws = vec![
            raw("winner", Some(5.0), Some(5.0), Some(100.0), true),
            raw("loser", Some(3.0), Some(3.0), Some(10.0), true),
        ];
        let cands = gate_and_build(&raws, None, 0.0);
        let ranked = rank(
            &cands,
            BlendWeights {
                q: 0.34,
                a: 0.33,
                r: 0.33,
            },
        );
        assert_eq!(ranked[0].model_id, "winner");
        assert!(ranked[0].score > ranked[1].score);
    }

    // ── Per-tier weights make fast vs deep diverge on a crafted set ─────────────

    #[test]
    fn fast_and_deep_weights_pick_different_winners() {
        // `fast_model`: slightly-lower quality but MUCH faster.
        // `deep_model`: highest quality but slow.
        let raws = vec![
            raw("fast_model", Some(4.0), Some(4.0), Some(120.0), true),
            raw("deep_model", Some(5.0), Some(4.0), Some(10.0), true),
        ];
        let cfg = AliasUpdaterConfig::default();
        let tiers = vec![
            TierPlan {
                key: "lumina-fast".into(),
                weights: cfg.fast_weights,
                current: None,
            },
            TierPlan {
                key: "lumina-deep".into(),
                weights: cfg.deep_weights,
                current: None,
            },
        ];
        let plans = plan_repoints(&raws, None, &cfg, &tiers);

        let fast = &plans[0].1;
        let deep = &plans[1].1;
        match fast {
            Repoint::Switch { to, .. } => {
                assert_eq!(
                    to, "fast_model",
                    "responsiveness-favoring tier picks the fast model"
                )
            }
            other => panic!("expected fast tier to switch, got {other:?}"),
        }
        match deep {
            Repoint::Switch { to, .. } => {
                assert_eq!(
                    to, "deep_model",
                    "quality-favoring tier picks the slow high-quality model"
                )
            }
            other => panic!("expected deep tier to switch, got {other:?}"),
        }
    }

    // ── Hysteresis: holds within margin, switches beyond it ────────────────────

    #[test]
    fn hysteresis_holds_within_margin() {
        // current = "b"; top = "a" but only marginally better than "b".
        let raws = vec![
            raw("a", Some(4.10), Some(4.0), Some(100.0), true),
            raw("b", Some(4.00), Some(4.0), Some(99.0), true),
        ];
        let cfg = AliasUpdaterConfig {
            switch_margin: 0.50, // large margin → a's tiny edge can't clear it
            ..AliasUpdaterConfig::default()
        };
        let tiers = vec![TierPlan {
            key: "lumina".into(),
            weights: cfg.fast_weights,
            current: Some("b".into()),
        }];
        let plans = plan_repoints(&raws, None, &cfg, &tiers);
        assert!(
            matches!(plans[0].1, Repoint::Keep { .. }),
            "within-margin near-tie must NOT switch (got {:?})",
            plans[0].1
        );
    }

    #[test]
    fn hysteresis_switches_beyond_margin() {
        let raws = vec![
            raw("a", Some(5.0), Some(5.0), Some(100.0), true),
            raw("b", Some(3.0), Some(3.0), Some(10.0), true),
        ];
        let cfg = AliasUpdaterConfig {
            switch_margin: 0.05,
            ..AliasUpdaterConfig::default()
        };
        let tiers = vec![TierPlan {
            key: "lumina".into(),
            weights: cfg.fast_weights,
            current: Some("b".into()),
        }];
        let plans = plan_repoints(&raws, None, &cfg, &tiers);
        match &plans[0].1 {
            Repoint::Switch { from, to, .. } => {
                assert_eq!(from.as_deref(), Some("b"));
                assert_eq!(to, "a");
            }
            other => panic!("expected a switch beyond margin, got {other:?}"),
        }
    }

    #[test]
    fn already_top_is_a_noop_keep() {
        let raws = vec![raw("a", Some(5.0), Some(5.0), Some(100.0), true)];
        let cfg = AliasUpdaterConfig::default();
        let tiers = vec![TierPlan {
            key: "lumina".into(),
            weights: cfg.fast_weights,
            current: Some("a".into()),
        }];
        let plans = plan_repoints(&raws, None, &cfg, &tiers);
        assert!(matches!(plans[0].1, Repoint::Keep { .. }));
    }

    // ── Gates ──────────────────────────────────────────────────────────────────

    #[test]
    fn gate_drops_guard_failing_candidates() {
        let raws = vec![
            raw("guarded_out", Some(5.0), Some(5.0), Some(100.0), false),
            raw("ok", Some(4.0), Some(4.0), Some(50.0), true),
        ];
        let cands = gate_and_build(&raws, None, 0.0);
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].model_id, "ok");
    }

    #[test]
    fn gate_drops_below_quality_floor() {
        let raws = vec![
            raw("too_low", Some(2.5), Some(5.0), Some(100.0), true),
            raw("ok", Some(3.5), Some(4.0), Some(50.0), true),
        ];
        let cands = gate_and_build(&raws, None, 3.0);
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].model_id, "ok");
    }

    #[test]
    fn gate_drops_models_without_quality_data() {
        // No q value at all → cannot clear the floor, dropped.
        let raws = vec![raw("no_q", None, Some(5.0), Some(100.0), true)];
        let cands = gate_and_build(&raws, None, 0.0);
        assert!(cands.is_empty());
    }

    #[test]
    fn gate_drops_non_eligible_when_eligibility_set_given() {
        // Simulates arch-excluded / non-servable filtering: only "servable" is in
        // the eligible set, so "arch_excluded" is dropped even though it scores.
        let raws = vec![
            raw("arch_excluded", Some(5.0), Some(5.0), Some(100.0), true),
            raw("servable", Some(4.0), Some(4.0), Some(50.0), true),
        ];
        let mut eligible = HashSet::new();
        eligible.insert("servable".to_string());
        let cands = gate_and_build(&raws, Some(&eligible), 0.0);
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].model_id, "servable");
    }

    // ── Empty candidate set → keep current ─────────────────────────────────────

    #[test]
    fn empty_candidates_keep_current() {
        // Every candidate fails the guard → nothing survives → keep current.
        let raws = vec![raw("x", Some(5.0), Some(5.0), Some(100.0), false)];
        let cfg = AliasUpdaterConfig::default();
        let tiers = vec![TierPlan {
            key: "lumina".into(),
            weights: cfg.fast_weights,
            current: Some("granite4.1:30b".into()),
        }];
        let plans = plan_repoints(&raws, None, &cfg, &tiers);
        assert!(
            matches!(plans[0].1, Repoint::Keep { .. }),
            "no candidates must keep the current target, never blank it"
        );
    }

    #[test]
    fn current_no_longer_a_candidate_adopts_top() {
        // current="stale" is not in the ranked set anymore → adopt the top pick.
        let raws = vec![raw("fresh", Some(4.5), Some(4.5), Some(80.0), true)];
        let cfg = AliasUpdaterConfig::default();
        let tiers = vec![TierPlan {
            key: "lumina".into(),
            weights: cfg.fast_weights,
            current: Some("stale".into()),
        }];
        let plans = plan_repoints(&raws, None, &cfg, &tiers);
        match &plans[0].1 {
            Repoint::Switch {
                from,
                to,
                old_score,
                ..
            } => {
                assert_eq!(from.as_deref(), Some("stale"));
                assert_eq!(to, "fresh");
                assert!(
                    old_score.is_none(),
                    "stale current has no score in the ranked set"
                );
            }
            other => panic!("expected switch to fresh top, got {other:?}"),
        }
    }

    // ── Aggregation / extraction from a report ─────────────────────────────────

    #[test]
    fn extract_a_from_eligible_row_not_global_max() {
        // FIX 5 regression: model "m" has an EXCLUDED backend row with a HIGHER
        // behavioral_mean (4.9) than its ELIGIBLE row (4.8). `a` must come from
        // the eligible row (4.8), NOT the global max (4.9) — the excluded
        // backend's adherence must never stand in for the one that qualifies.
        use terminus_rs::intake::assistant::reporting::{
            ChatRoleCandidate, ChatRoleSelection, DualProfileRow, GuardVerdict, ModelKey,
            MoneyQuery, OceanShortlist, PersonalityRead,
        };

        // Minimal empty scaffolding for the report fields this test doesn't
        // exercise (MoneyQuery / PersonalityRead aren't `Default`).
        let empty_money = |name: &str| MoneyQuery {
            name: name.into(),
            dimension: String::new(),
            metric: String::new(),
            higher_is_better: true,
            ranking: vec![],
        };
        let empty_personality = || PersonalityRead {
            shortlist: OceanShortlist {
                members: vec![],
                min_proximity: 3.0,
            },
            prompted_ranking: vec![],
        };

        // model "m": two dual-profile rows (values 4.0 and 2.0 → mean 3.0) and
        // two chat-role rows — an EXCLUDED gpu row (behavioral 4.9) and an
        // ELIGIBLE cpu row (behavioral 4.8). `a` must be 4.8 (eligible), not 4.9.
        let dual_profile = vec![
            DualProfileRow {
                model_id: "m".into(),
                backend_tag: Some("gpu".into()),
                mem_config: None,
                has_builder_profile: false,
                has_assistant_profile: true,
                builder_avg_quality: None,
                assistant_avg_value: Some(4.0),
            },
            DualProfileRow {
                model_id: "m".into(),
                backend_tag: Some("gpu".into()),
                mem_config: Some("dynamic_gtt".into()),
                has_builder_profile: false,
                has_assistant_profile: true,
                builder_avg_quality: None,
                assistant_avg_value: Some(2.0),
            },
        ];
        let candidates = vec![
            ChatRoleCandidate {
                key: ModelKey {
                    model_id: "m".into(),
                    backend_tag: "gpu".into(),
                },
                behavioral_mean: 4.9,
                recall_ceiling_turns: Some(40.0),
                latency_ms: Some(1000.0),
                verdict: GuardVerdict::Excluded {
                    reason: "some".into(),
                },
            },
            ChatRoleCandidate {
                key: ModelKey {
                    model_id: "m".into(),
                    backend_tag: "cpu".into(),
                },
                behavioral_mean: 4.8,
                recall_ceiling_turns: Some(40.0),
                latency_ms: Some(1000.0),
                verdict: GuardVerdict::Eligible,
            },
        ];

        let report = AssistantReport {
            best_conversation_depth: empty_money("best_conversation_depth"),
            best_tool_chaining: empty_money("best_tool_chaining"),
            best_memory_survival: empty_money("best_memory_survival"),
            embedding_leader: empty_money("embedding_leader"),
            embedding_public_vs_engram_delta: vec![],
            personality: empty_personality(),
            chat_role: ChatRoleSelection {
                candidates,
                selected: Some(ModelKey {
                    model_id: "m".into(),
                    backend_tag: "cpu".into(),
                }),
            },
            dual_profile,
        };

        let mut resp = HashMap::new();
        resp.insert("m".to_string(), 55.0);

        let raws = extract_raw_signals(&report, &resp);
        assert_eq!(raws.len(), 1);
        let s = &raws[0];
        assert_eq!(s.model_id, "m");
        assert!(
            (s.q.unwrap() - 3.0).abs() < 1e-9,
            "q is the MEAN of dual rows"
        );
        assert!(
            (s.a.unwrap() - 4.8).abs() < 1e-9,
            "a must come from the ELIGIBLE row (4.8), not the excluded row's 4.9"
        );
        assert!((s.r.unwrap() - 55.0).abs() < 1e-9);
        assert!(
            s.guard_eligible,
            "any Eligible chat-role row makes the model guard-eligible"
        );
    }

    // ── FIX 3: multi-backend arch-exclusion (usable-on-one survives) ───────────

    #[test]
    fn compute_eligible_keeps_model_usable_on_one_backend() {
        use crate::serving::profile::RoutingMap;
        use terminus_rs::intake::serving::{
            ExclusionReason, ModelId, RecheckTrigger, Runtime, ServingBackend, ServingProfile,
        };

        fn prof(model: &str, backend: ServingBackend, excl: ExclusionReason) -> ServingProfile {
            ServingProfile {
                model_id: ModelId::from(model),
                backend_tag: backend,
                best_runtime: Runtime::Ollama,
                env_json: String::new(),
                tok_s: Some(30.0),
                vram_or_ram_peak_gb: Some(8.0),
                cold_load_s: Some(10.0),
                keep_warm: false,
                fallback_runtime: None,
                exclusion_reason: excl,
                recheck_trigger: RecheckTrigger::None,
                provenance: None,
            }
        }

        // "mixed" is arch-excluded on llama-gpu but USABLE on ollama-gpu → survives.
        // "dead" is excluded on BOTH of its backends → dropped.
        // "unprofiled" has no routing row at all → survives (no arch verdict).
        let rmap = RoutingMap::load_from(vec![
            prof(
                "mixed",
                ServingBackend::LlamaGpu,
                ExclusionReason::PermanentUnknownArch,
            ),
            prof("mixed", ServingBackend::OllamaGpu, ExclusionReason::None),
            prof(
                "dead",
                ServingBackend::LlamaGpu,
                ExclusionReason::PermanentUnknownArch,
            ),
            prof(
                "dead",
                ServingBackend::OllamaGpu,
                ExclusionReason::PermanentUnknownArch,
            ),
        ]);

        let raws = vec![
            raw("mixed", Some(50.0), Some(4.0), Some(80.0), true),
            raw("dead", Some(60.0), Some(4.5), Some(90.0), true),
            raw("unprofiled", Some(40.0), Some(3.5), Some(70.0), true),
        ];
        let servable: HashSet<String> = ["mixed", "dead", "unprofiled"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        let eligible = compute_eligible(&raws, &servable, &rmap);
        assert!(
            eligible.contains("mixed"),
            "a model usable on ONE backend must survive arch-exclusion"
        );
        assert!(
            !eligible.contains("dead"),
            "a model excluded on ALL backends must be dropped"
        );
        assert!(
            eligible.contains("unprofiled"),
            "an unprofiled model carries no arch verdict and is not dropped"
        );
    }

    #[test]
    fn compute_eligible_requires_servability() {
        use crate::serving::profile::RoutingMap;
        let rmap = RoutingMap::empty();
        let raws = vec![raw("known", Some(50.0), Some(4.0), Some(80.0), true)];
        // "known" is NOT in the servable set → dropped even with clean routing.
        let servable: HashSet<String> = HashSet::new();
        // (compute_eligible itself never sees an empty registry live — the tick
        // bails first — but the servability filter must still hold here.)
        let eligible = compute_eligible(&raws, &servable, &rmap);
        assert!(
            eligible.is_empty(),
            "non-servable models must be filtered out"
        );
    }

    // ── FIX 6: invalid config falls back to defaults ───────────────────────────

    #[test]
    #[serial_test::serial]
    fn from_env_rejects_invalid_weights_and_refresh() {
        let keys = [
            "CHORD_ALIAS_REFRESH_SECS",
            "CHORD_ALIAS_MIN_QUALITY",
            "CHORD_ALIAS_SWITCH_MARGIN",
            "CHORD_ALIAS_W_FAST_Q",
            "CHORD_ALIAS_W_FAST_A",
            "CHORD_ALIAS_W_FAST_R",
            "CHORD_ALIAS_W_DEEP_Q",
            "CHORD_ALIAS_W_DEEP_A",
            "CHORD_ALIAS_W_DEEP_R",
        ];
        for k in keys {
            std::env::remove_var(k);
        }
        // Invalid values: NaN, negative, and a zero/too-small refresh interval.
        std::env::set_var("CHORD_ALIAS_REFRESH_SECS", "0"); // → floored to MIN_REFRESH_SECS
        std::env::set_var("CHORD_ALIAS_W_FAST_Q", "NaN"); // → default
        std::env::set_var("CHORD_ALIAS_W_FAST_A", "-1.0"); // → default
        std::env::set_var("CHORD_ALIAS_W_DEEP_Q", "inf"); // → default
        std::env::set_var("CHORD_ALIAS_SWITCH_MARGIN", "-0.5"); // → default

        let cfg = AliasUpdaterConfig::from_env();
        let def = AliasUpdaterConfig::default();

        assert_eq!(
            cfg.refresh_secs, MIN_REFRESH_SECS,
            "0 refresh floored, never panics interval"
        );
        assert_eq!(
            cfg.fast_weights.q, def.fast_weights.q,
            "NaN weight → default"
        );
        assert_eq!(
            cfg.fast_weights.a, def.fast_weights.a,
            "negative weight → default"
        );
        assert_eq!(
            cfg.deep_weights.q, def.deep_weights.q,
            "inf weight → default"
        );
        assert_eq!(
            cfg.switch_margin, def.switch_margin,
            "negative margin → default"
        );

        for k in keys {
            std::env::remove_var(k);
        }
    }

    #[test]
    fn default_min_quality_is_disabled() {
        // FIX 2: the raw-scale floor is OFF by default so it can't silently
        // filter on a wrong 1..5 assumption.
        assert_eq!(AliasUpdaterConfig::default().min_quality, 0.0);
    }
}
