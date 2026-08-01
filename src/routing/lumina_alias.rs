//! CHRD-91390429: dynamic lumina-proxy alias updater — a blended, assistant-fit
//! background task that repoints the two Lumina chat aliases (`lumina-fast`,
//! `lumina-deep`) at runtime, WITHOUT a restart. (The former `lumina` main/core
//! tier was removed as dead — `lumina-core` only requests fast/deep.)
//!
//! ## What this replaces
//! The lumina alias targets used to be STATIC entries in
//! `CHORD_MODEL_ALIASES` (parsed once at startup, immutable). An interim ops
//! drop-in pinned them to a single hand-picked model. This module makes those
//! two (and ONLY those two) targets runtime-mutable: a background task
//! ranks the measured assistant fleet every `CHORD_ALIAS_REFRESH_SECS` and
//! hot-swaps the targets through an [`arc_swap::ArcSwap`], so the chat hot path
//! ([`crate::routes::chat_completions`]) resolves the current target with a
//! single lock-free `ArcSwap::load` — inference is NEVER blocked on the updater.
//! Every non-lumina alias (and the retired `lumina` main key) stays in the static
//! `CHORD_MODEL_ALIASES` map, untouched.
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
//!     higher tok/s ⇒ higher `r`, but SATURATED at
//!     `CHORD_ALIAS_R_SATURATION_TOK_S` (metric-v2) so a tiny hyper-fast model
//!     earns no runaway speed advantage over a merely "fast enough" one.
//!
//! ## metric-v2 (post-mortem of the qwen2.5:0.5b regression)
//! The first live deploy crowned `qwen2.5:0.5b` (a 0.5B model!) for all three
//! tiers at an identical 0.700 — it had a deceptively high `assistant_avg_value`
//! and maximal responsiveness (tiny ⇒ fast), and the blend rewarded speed. The
//! redesign:
//!   - **PRIMARY: a hard capability/size gate** (`CHORD_ALIAS_MIN_SIZE_BYTES`,
//!     default ~5 GB) drops any model too small to be a real assistant BEFORE
//!     scoring. A 0.5B model can never be a candidate. Unknown/zero size ⇒
//!     excluded (fail-safe).
//!   - **q-heavy reweight** so responsiveness only breaks ties among CAPABLE
//!     models (see weights below).
//!   - **responsiveness saturation** so "fast enough" earns full `r` and being
//!     tiny-fast earns no more.
//!   - **`assistant_avg_value` verified** to be `avg(value)` (a genuine per-row
//!     average, NOT a count-inflated sum — terminus-rs `schema.rs` view SQL), so
//!     `q` is kept as-is (dividing by `assistant_score_count` would be wrong).
//!
//! ### Per-tier weights (env-configurable; metric-v2 defaults below)
//! Both tiers are q-heavy; `lumina-fast` keeps a small responsiveness
//! tie-breaker, `lumina-deep` almost none:
//! ```text
//! lumina-fast : 0.55*q + 0.30*a + 0.15*r
//! lumina-deep : 0.65*q + 0.30*a + 0.05*r
//! ```
//!
//! ### Gates (a candidate is dropped BEFORE scoring if any fails)
//!   a. Not servable/known in the registry.
//!   SIZE. Registry `size_bytes` below `CHORD_ALIAS_MIN_SIZE_BYTES`, or
//!      unknown/zero (metric-v2 PRIMARY gate — the decisive fix).
//!   b. Arch-excluded on ALL backends (per the RoutingMap serving exclusion —
//!      reuses [`crate::serving::profile::RoutingMap`]).
//!   c. Fails the existing latency/degradation guard that
//!      [`reporting::select_chat_role`] applies (we only accept models it marks
//!      [`GuardVerdict::Eligible`]).
//!   d. Below the assistant-quality floor (on `q`; disabled by default — the
//!      size gate is the real filter). The floor is PER-TIER: the global
//!      `CHORD_ALIAS_MIN_QUALITY` is overridable per tier by
//!      `CHORD_ALIAS_MIN_QUALITY_{FAST,DEEP}` (each defaulting to the global
//!      when unset), so e.g. the assistant tiers can stay floored while
//!      `lumina-deep` is unfloored to pick its best by the deep-weighted blend.
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

/// The two lumina alias keys this updater owns. EVERY other alias in
/// `CHORD_MODEL_ALIASES` stays static and is never touched here.
///
/// The former `"lumina"` (main/core) tier was removed — `lumina-core` only ever
/// requests `lumina-fast`/`lumina-deep`, so the main tier was dead weight and its
/// removal also cleanly separates concerns (`fast_weights` now affect ONLY
/// `lumina-fast`). The JWT `sub:"lumina"` identity is unrelated auth and untouched.
pub const LUMINA_ALIAS_KEYS: [&str; 2] = ["lumina-fast", "lumina-deep"];

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
    /// CQH-01 (F1/F2): a shared serialization point between an alias PUBLICATION
    /// ([`set`](Self::set)) and a consumer that must read the current targets and
    /// then commit a MULTI-STEP, `await`-crossing action atomically w.r.t. any
    /// repoint. The hot read path ([`resolve`](Self::resolve)) stays lock-free
    /// (`ArcSwap::load`) — this lock is taken ONLY on the rare write (`set`, a
    /// promotion event) and by a consumer that brackets its whole critical section
    /// via [`publish_lock_arc`](Self::publish_lock_arc)`.lock_owned()`.
    ///
    /// It is a `tokio::sync::Mutex` (not `std`) SPECIFICALLY so the cold-quota
    /// pruner can hold it ACROSS its async filesystem delete: the delete-time
    /// keep/resident re-check → `remove_record` → `remover.remove().await` must ALL
    /// be atomic w.r.t. a repoint, or an alias could repoint onto a candidate in
    /// the record-drop→fs-delete window and the already-scheduled delete would wipe
    /// a now-live target's archive. A plain registry mutex cannot order against a
    /// lock-free `ArcSwap` publish, and a `std` mutex cannot be held across the
    /// `.await`; this lock solves both. Lock order at the one site that takes both
    /// this and the registry lock (the prune commit) is ALWAYS publish→registry;
    /// nothing takes registry→publish (`set` and the registry methods each take
    /// only their own lock), so there is no inversion.
    publish_lock: Arc<tokio::sync::Mutex<()>>,
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
            publish_lock: Arc::new(tokio::sync::Mutex::new(())),
        }
    }

    /// An empty store (no lumina targets yet) — for tests and for the fail-open
    /// path when no static lumina alias was configured.
    pub fn empty() -> Self {
        LuminaAliasStore {
            inner: Arc::new(ArcSwap::from_pointee(HashMap::new())),
            publish_lock: Arc::new(tokio::sync::Mutex::new(())),
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
    pub async fn set(&self, key: &str, target: String) {
        // CQH-01 (F1/F2): serialize the publish against a consumer holding the
        // publish lock (the cold-quota prune commit), so a repoint cannot land
        // anywhere inside that consumer's read → record-drop → fs-delete critical
        // section. Writes are rare (promotion events); readers stay lock-free
        // (`resolve`/`snapshot` never take this lock). The guard scopes ONLY the
        // `ArcSwap` publish here — no other lock is nested under it in `set`.
        let _publish = self.publish_lock.lock().await;
        self.inner.rcu(|current| {
            let mut next: HashMap<String, String> = (**current).clone();
            next.insert(key.to_string(), target.clone());
            next
        });
    }

    /// CQH-01 (F1/F2): the alias-publish serialization lock, as an `Arc` the caller
    /// can `.lock_owned().await` to hold across a whole `await`-crossing critical
    /// section. The cold-quota pruner holds it across its delete-time keep/resident
    /// re-check → `remove_record` → `remover.remove().await`, so no repoint (also
    /// taking this lock in [`set`](Self::set)) can interleave any part of that
    /// section. Lock order where both this and the registry lock are held is
    /// ALWAYS publish→registry (see the field docs); `resolve`/`snapshot` never take
    /// it, so the hot path is unaffected.
    pub fn publish_lock_arc(&self) -> Arc<tokio::sync::Mutex<()>> {
        self.publish_lock.clone()
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
    ///
    /// This is the GLOBAL floor; each tier can OVERRIDE it via a per-tier var
    /// ([`min_quality_fast`](Self::min_quality_fast) /
    /// [`min_quality_deep`](Self::min_quality_deep)), each of which DEFAULTS to
    /// this global value when its own var is unset — so behavior is unchanged
    /// unless a per-tier var is explicitly set. The effective floor for a given
    /// alias key is resolved by [`min_quality_for`](Self::min_quality_for).
    pub min_quality: f64,
    /// `CHORD_ALIAS_MIN_QUALITY_FAST` — per-tier quality floor for the
    /// `lumina-fast` tier. Overrides [`min_quality`](Self::min_quality) for that
    /// tier only; UNSET ⇒ falls back to the global `min_quality`.
    pub min_quality_fast: f64,
    /// `CHORD_ALIAS_MIN_QUALITY_DEEP` — per-tier quality floor for the
    /// `lumina-deep` tier. Overrides [`min_quality`](Self::min_quality) for that
    /// tier only; UNSET ⇒ falls back to the global `min_quality`. Set to `0` to
    /// UNFLOOR the deep tier so it picks its best by the deep-weighted blend +
    /// size gate, unconstrained by the assistant-quality floor.
    pub min_quality_deep: f64,
    /// `CHORD_ALIAS_MIN_SIZE_BYTES` — PRIMARY capability gate (metric-v2). A model
    /// whose registry `size_bytes` is below this is too small to be a real
    /// conversational assistant and is EXCLUDED before scoring (no blend can
    /// crown it). Default `5_000_000_000` (~5 GB ≈ ~7B params at Q4) — keeps
    /// granite4.1:30b / command-r / phi4:14b / mistral-small3.2:24b / qwen2.5:32b,
    /// drops qwen2.5:0.5b (~0.4 GB) and other sub-~7B models. A model with
    /// unknown/zero `size_bytes` is EXCLUDED (fail-safe — never promote an unsized
    /// model to a primary assistant alias).
    pub min_size_bytes: u64,
    /// `CHORD_ALIAS_R_SATURATION_TOK_S` — responsiveness saturation target
    /// (tok/s). Raw `r` is capped at this BEFORE normalization, so a model gets
    /// FULL responsiveness credit once it is "fast enough" and a tiny model earns
    /// no runaway speed advantage for being sub-second. Default 40.0 tok/s
    /// (comfortably interactive). Set very high to effectively disable saturation.
    pub r_saturation_tok_s: f64,
    /// `CHORD_ALIAS_SWITCH_MARGIN` — hysteresis margin (default 0.05).
    pub switch_margin: f64,
    /// Weights for `lumina` and `lumina-fast` (responsiveness-favoring, but
    /// q-heavy in metric-v2 so speed only breaks ties among CAPABLE models).
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

/// Parse a strictly-positive u64 env value (a SAFETY floor like the size gate).
/// A `0` or malformed value falls back to `default` — it must NEVER silently
/// disable the guard (`0` would re-open the tiny-model hole). A genuinely lower
/// floor is an explicit small POSITIVE value.
fn env_pos_u64(key: &str, default: u64) -> u64 {
    match std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
    {
        Some(v) if v > 0 => v,
        _ => default,
    }
}

/// Parse a strictly-positive, finite f64 env value; anything malformed, NaN,
/// ±inf, or `<= 0` falls back to `default` (a zero saturation would divide-by-zero
/// or zero-out responsiveness).
fn env_pos_f64(key: &str, default: f64) -> f64 {
    match std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
    {
        Some(v) if v.is_finite() && v > 0.0 => v,
        _ => default,
    }
}

/// Default size gate: ~5 GB ≈ ~7B params at Q4 (see [`AliasUpdaterConfig::min_size_bytes`]).
const DEFAULT_MIN_SIZE_BYTES: u64 = 5_000_000_000;
/// Default responsiveness saturation target (tok/s).
const DEFAULT_R_SATURATION_TOK_S: f64 = 40.0;
/// Default hysteresis switch margin.
const DEFAULT_SWITCH_MARGIN: f64 = 0.05;

// ── Point-of-USE sanitizers ─────────────────────────────────────────────────
// Validation lives HERE, at the point a value is applied, not only in
// `from_env` — so ANY construction path (a struct literal in a test, a future
// caller, a hand-built config) still gets a sane value. A NaN/inf/negative
// weight or margin would corrupt/invert the blend; a `0`/invalid size floor or
// saturation would silently disable a SAFETY guard. Each invalid value falls
// back to its documented default.

/// The size floor is a safety guard: `0` or (impossible for u64) invalid ⇒ the
/// default, NEVER "disabled".
fn sane_min_size_bytes(v: u64) -> u64 {
    if v == 0 {
        DEFAULT_MIN_SIZE_BYTES
    } else {
        v
    }
}

/// Saturation must be finite and `> 0`; else the default (never zero-out `r`).
fn sane_saturation_tok_s(v: f64) -> f64 {
    if v.is_finite() && v > 0.0 {
        v
    } else {
        DEFAULT_R_SATURATION_TOK_S
    }
}

/// A blend weight must be finite and `>= 0`; an invalid weight is dropped to
/// `0.0` (removes that signal) rather than allowed to corrupt/invert the score.
fn sane_weight(w: f64) -> f64 {
    if w.is_finite() && w >= 0.0 {
        w
    } else {
        0.0
    }
}

/// The hysteresis margin must be finite and `>= 0`; else the default.
fn sane_margin(m: f64) -> f64 {
    if m.is_finite() && m >= 0.0 {
        m
    } else {
        DEFAULT_SWITCH_MARGIN
    }
}

/// The quality floor must be finite and `>= 0`; else `0.0` (disabled default).
fn sane_min_quality(q: f64) -> f64 {
    if q.is_finite() && q >= 0.0 {
        q
    } else {
        0.0
    }
}

impl AliasUpdaterConfig {
    /// Read the config from the environment, applying the documented defaults for
    /// any unset/malformed/invalid var (see [`env_nonneg_f64`] /
    /// [`env_refresh_secs`] — invalid weights/margins fall back to defaults, the
    /// interval is floored).
    pub fn from_env() -> Self {
        let min_quality = env_nonneg_f64("CHORD_ALIAS_MIN_QUALITY", 0.0);
        AliasUpdaterConfig {
            refresh_secs: env_refresh_secs("CHORD_ALIAS_REFRESH_SECS", 900),
            min_quality,
            // Per-tier floors each DEFAULT to the global `min_quality` when their
            // own var is unset — so unless an operator sets a per-tier var, every
            // tier keeps the exact global-only behavior.
            min_quality_fast: env_nonneg_f64("CHORD_ALIAS_MIN_QUALITY_FAST", min_quality),
            min_quality_deep: env_nonneg_f64("CHORD_ALIAS_MIN_QUALITY_DEEP", min_quality),
            min_size_bytes: env_pos_u64("CHORD_ALIAS_MIN_SIZE_BYTES", DEFAULT_MIN_SIZE_BYTES),
            r_saturation_tok_s: env_pos_f64(
                "CHORD_ALIAS_R_SATURATION_TOK_S",
                DEFAULT_R_SATURATION_TOK_S,
            ),
            switch_margin: env_nonneg_f64("CHORD_ALIAS_SWITCH_MARGIN", 0.05),
            // metric-v2 reweight: q-heavy so responsiveness only breaks ties among
            // capable (size-gated) models — it can no longer crown a weak model.
            fast_weights: BlendWeights {
                q: env_nonneg_f64("CHORD_ALIAS_W_FAST_Q", 0.55),
                a: env_nonneg_f64("CHORD_ALIAS_W_FAST_A", 0.30),
                r: env_nonneg_f64("CHORD_ALIAS_W_FAST_R", 0.15),
            },
            deep_weights: BlendWeights {
                q: env_nonneg_f64("CHORD_ALIAS_W_DEEP_Q", 0.65),
                a: env_nonneg_f64("CHORD_ALIAS_W_DEEP_A", 0.30),
                r: env_nonneg_f64("CHORD_ALIAS_W_DEEP_R", 0.05),
            },
        }
    }

    /// Resolve the effective quality floor for one alias tier by its key. The
    /// two lumina tiers map to their per-tier floor (each of which already
    /// defaulted to the global `min_quality` at construction if its own var was
    /// unset); any OTHER key falls back to the global `min_quality`. The returned
    /// value is sanitized at the point of use (see [`sane_min_quality`]) so a
    /// hand-built config with a bad value can never invert the gate.
    pub fn min_quality_for(&self, key: &str) -> f64 {
        let raw = match key {
            "lumina-fast" => self.min_quality_fast,
            "lumina-deep" => self.min_quality_deep,
            _ => self.min_quality,
        };
        sane_min_quality(raw)
    }
}

impl Default for AliasUpdaterConfig {
    fn default() -> Self {
        AliasUpdaterConfig {
            refresh_secs: 900,
            min_quality: 0.0,
            min_quality_fast: 0.0,
            min_quality_deep: 0.0,
            min_size_bytes: DEFAULT_MIN_SIZE_BYTES,
            r_saturation_tok_s: DEFAULT_R_SATURATION_TOK_S,
            switch_margin: 0.05,
            fast_weights: BlendWeights {
                q: 0.55,
                a: 0.30,
                r: 0.15,
            },
            deep_weights: BlendWeights {
                q: 0.65,
                a: 0.30,
                r: 0.05,
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
    ///
    /// metric-v2 semantics verification (FIX B): the `model_dual_profile` view
    /// defines this column as `avg(value)` over the assistant-category
    /// `assistant_dimension_score` rows — a genuine PER-ROW AVERAGE, **not** a
    /// count-inflated SUM (verified against terminus-rs `schema.rs`, view SQL
    /// `count(*) AS assistant_score_count, avg(value) AS assistant_avg_value`).
    /// So answer-count does NOT inflate it, and dividing by `assistant_score_count`
    /// would be WRONG. It IS a coarse signal (it averages `value` across several
    /// assistant dimensions on different magnitudes), which is exactly why the
    /// hard size gate — not `q` alone — is the decisive filter in metric-v2.
    pub q: Option<f64>, // RAW `assistant_avg_value` (avg(value); un-normalized).
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

/// A scored candidate: the blended `score` plus BOTH the raw and normalized
/// components, so a repoint's full rationale is auditable in the logs (FIX D).
/// Higher `score` is better.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredCandidate {
    pub model_id: String,
    pub score: f64,
    /// Raw (pre-normalization) blend inputs — `r` is post-saturation.
    pub q_raw: f64,
    pub a_raw: f64,
    pub r_raw: f64,
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
///
/// NOTE: the PRIMARY capability/size gate (metric-v2 FIX A) is applied UPSTREAM,
/// folded into the `eligible` set the live caller passes — a too-small model is
/// never in `eligible`, so it never reaches scoring here.
///
/// `r_saturation_tok_s` caps each candidate's raw responsiveness (FIX C) so a
/// tiny hyper-fast model earns no runaway speed advantage: once a model is
/// "fast enough" it gets the same `r` as any faster one. Both `min_quality` and
/// `r_saturation_tok_s` are sanitized HERE (point of use) so a bad value from ANY
/// caller — not just `from_env` — is corrected (invalid → documented default).
pub fn gate_and_build(
    raws: &[RawSignal],
    eligible: Option<&HashSet<String>>,
    min_quality: f64,
    r_saturation_tok_s: f64,
) -> Vec<AliasCandidate> {
    let min_quality = sane_min_quality(min_quality);
    let r_saturation_tok_s = sane_saturation_tok_s(r_saturation_tok_s);
    raws.iter()
        .filter_map(|raw| {
            // (a+b) servable, size-gated & not arch-excluded — Some(set) always filters.
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
            // (FIX C) saturate responsiveness at the "fast enough" target.
            let r = raw.r.unwrap_or(0.0).min(r_saturation_tok_s);
            Some(AliasCandidate {
                model_id: raw.model_id.clone(),
                q,
                a: raw.a.unwrap_or(0.0),
                r,
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
    // Sanitize weights at the point of use so a NaN/inf/negative from ANY caller
    // (not just from_env) can never corrupt or invert the blend.
    let w_q = sane_weight(weights.q);
    let w_a = sane_weight(weights.a);
    let w_r = sane_weight(weights.r);
    let (q_lo, q_hi) = min_max(candidates.iter().map(|c| c.q)).unwrap();
    let (a_lo, a_hi) = min_max(candidates.iter().map(|c| c.a)).unwrap();
    let (r_lo, r_hi) = min_max(candidates.iter().map(|c| c.r)).unwrap();

    let mut scored: Vec<ScoredCandidate> = candidates
        .iter()
        .map(|c| {
            let q_norm = normalize(c.q, q_lo, q_hi);
            let a_norm = normalize(c.a, a_lo, a_hi);
            let r_norm = normalize(c.r, r_lo, r_hi);
            let score = w_q * q_norm + w_a * a_norm + w_r * r_norm;
            ScoredCandidate {
                model_id: c.model_id.clone(),
                score,
                q_raw: c.q,
                a_raw: c.a,
                r_raw: c.r,
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
    // Sanitize the margin at the point of use (invalid → default) so a bad value
    // from any caller can't defeat hysteresis.
    let margin = sane_margin(margin);
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

/// One tier's full decision: the repoint plus the ranked candidate list it was
/// derived from (for the FIX D audit log — every candidate's q/a/r and score).
#[derive(Debug, Clone, PartialEq)]
pub struct TierDecision {
    pub key: String,
    pub repoint: Repoint,
    pub ranked: Vec<ScoredCandidate>,
}

/// Compute the repoint decision for every tier from one shared raw-signal set.
/// The size/arch/chat-role/q-exists gates and responsiveness saturation are
/// tier-independent and computed once into a shared base set; the QUALITY floor
/// (gate d) is applied PER-TIER via [`AliasUpdaterConfig::min_quality_for`], and
/// the blend/ranking differs per tier's weights. Pure — the whole decision core
/// is testable without any I/O. Returns each tier's ranked list so the caller can
/// log the full selection rationale.
pub fn plan_repoints(
    raws: &[RawSignal],
    eligible: Option<&HashSet<String>>,
    cfg: &AliasUpdaterConfig,
    tiers: &[TierPlan],
) -> Vec<TierDecision> {
    // Build the SHARED base candidate set with every gate EXCEPT the quality
    // floor (pass 0.0) — the floor is now applied PER-TIER below so each tier can
    // carry a different `CHORD_ALIAS_MIN_QUALITY_{MAIN,FAST,DEEP}`. The size /
    // arch / chat-role / q-exists gates and the responsiveness saturation are
    // tier-independent and stay here, computed once.
    let base = gate_and_build(raws, eligible, 0.0, cfg.r_saturation_tok_s);
    tiers
        .iter()
        .map(|tier| {
            // Per-tier quality floor: overrides the global for this tier, else
            // falls back to it. When all tiers resolve to the same floor (the
            // default, or a uniform global setting) each tier's filtered set is
            // identical to the old single shared set — behavior is unchanged.
            let floor = cfg.min_quality_for(&tier.key);
            let candidates: Vec<AliasCandidate> =
                base.iter().filter(|c| c.q >= floor).cloned().collect();
            let ranked = rank(&candidates, tier.weights);
            let repoint = select_target(&ranked, tier.current.as_deref(), cfg.switch_margin);
            TierDecision {
                key: tier.key.clone(),
                repoint,
                ranked,
            }
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

/// Compute the eligible model-id set, given the registry `sizes` map (name →
/// `size_bytes`), the serving RoutingMap, and the size floor, restricted to the
/// models we actually have signals for.
///
/// Gate (a) servable: the registry must know the model — presence in `sizes`
/// (the caller guarantees a non-empty registry; an empty/not-ready registry is
/// handled upstream by keeping the current targets, never by scoring unservable
/// models).
///
/// Gate (SIZE — metric-v2 FIX A, the PRIMARY capability gate): the model's
/// registry `size_bytes` must be `>= min_size_bytes`. A model with unknown/zero
/// size is EXCLUDED (fail-safe — never promote an unsized model). This is what
/// keeps a 0.5B model from ever being scored, let alone winning.
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
    sizes: &HashMap<String, u64>,
    min_size_bytes: u64,
    rmap: &crate::serving::profile::RoutingMap,
) -> HashSet<String> {
    use terminus_rs::intake::serving::{ExclusionReason, ModelId};

    // Sanitize the SAFETY floor at the point of use: a `0` (or any invalid value
    // reaching here from a non-`from_env` path) must NEVER disable the guard — it
    // falls back to the default floor, so a 0 can't silently re-open the
    // tiny-model hole.
    let min_size_bytes = sane_min_size_bytes(min_size_bytes);

    raws.iter()
        .filter_map(|raw| {
            // (a) servable per the registry.
            let size = match sizes.get(&raw.model_id) {
                Some(&s) => s,
                None => return None,
            };
            // (SIZE) hard capability gate — unknown/zero or too-small ⇒ excluded.
            if size == 0 || size < min_size_bytes {
                tracing::debug!(
                    "lumina alias updater: '{}' excluded by size gate ({} bytes < {} min)",
                    raw.model_id,
                    size,
                    min_size_bytes
                );
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
    // Registry name → size_bytes: powers BOTH servability (presence) and the
    // metric-v2 size gate (FIX A).
    let sizes: HashMap<String, u64> = {
        let reg = registry.lock().await;
        reg.all_records()
            .map(|r| (r.name.clone(), r.size_bytes))
            .collect()
    };
    if sizes.is_empty() {
        tracing::warn!(
            "lumina alias updater: model registry not ready (no records) — keeping current \
             lumina targets"
        );
        return;
    }

    // Snapshot the routing map (drop the lock before any store writes).
    let eligible = {
        let rmap = routing.lock().await;
        compute_eligible(&raws, &sizes, cfg.min_size_bytes, &rmap)
    };
    tracing::info!(
        "lumina alias updater: {} raw signal(s), {} eligible after gates (min_size={} bytes, \
         r_saturation={} tok/s)",
        raws.len(),
        eligible.len(),
        cfg.min_size_bytes,
        cfg.r_saturation_tok_s
    );

    let tiers = vec![
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

    for decision in plan_repoints(&raws, Some(&eligible), cfg, &tiers) {
        let TierDecision {
            key,
            repoint,
            ranked,
        } = decision;

        // FIX D: full per-candidate audit so the operator can verify the live pick
        // is sane. Every candidate's raw + normalized q/a/r and final score.
        for (rank_idx, sc) in ranked.iter().enumerate() {
            tracing::debug!(
                "lumina alias '{key}' cand #{rank_idx} {}: score={:.3} \
                 q(raw={:.1},norm={:.3}) a(raw={:.3},norm={:.3}) r(raw={:.1},norm={:.3})",
                sc.model_id,
                sc.score,
                sc.q_raw,
                sc.q_norm,
                sc.a_raw,
                sc.a_norm,
                sc.r_raw,
                sc.r_norm,
            );
        }

        match repoint {
            Repoint::Switch {
                from,
                to,
                new_score,
                ..
            } => {
                store.set(&key, to.clone()).await;
                // INFO rationale: the winning candidate's full breakdown so a live
                // repoint is verifiable without DEBUG logging enabled.
                let winner = ranked.iter().find(|s| s.model_id == to);
                match winner {
                    Some(w) => tracing::info!(
                        "lumina alias '{key}' repointed: {} -> {} | score={:.3} \
                         q(raw={:.1},norm={:.3}) a(raw={:.3},norm={:.3}) r(raw={:.1},norm={:.3}) \
                         [{} candidate(s)]",
                        from.as_deref().unwrap_or("<unset>"),
                        to,
                        new_score,
                        w.q_raw,
                        w.q_norm,
                        w.a_raw,
                        w.a_norm,
                        w.r_raw,
                        w.r_norm,
                        ranked.len(),
                    ),
                    None => tracing::info!(
                        "lumina alias '{key}' repointed: {} -> {} (score {new_score:.3})",
                        from.as_deref().unwrap_or("<unset>"),
                        to
                    ),
                }
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
        // The retired `lumina` main key is now just another static alias — it is
        // NOT one of the managed keys and must never be captured by the store.
        statics.insert("lumina".to_string(), "granite4.1:30b".to_string());
        statics.insert("lumina-fast".to_string(), "qwen3:8b".to_string());
        statics.insert("lumina-deep".to_string(), "qwen3:32b".to_string());
        statics.insert("some-other-alias".to_string(), "mystery:7b".to_string());

        let store = LuminaAliasStore::from_static(&statics);
        assert_eq!(store.resolve("lumina-fast").as_deref(), Some("qwen3:8b"));
        assert_eq!(store.resolve("lumina-deep").as_deref(), Some("qwen3:32b"));
        // The retired `lumina` main key and every other non-managed alias are
        // NEVER captured by the dynamic store.
        assert_eq!(store.resolve("lumina"), None);
        assert_eq!(store.resolve("some-other-alias"), None);
        assert_eq!(store.snapshot().len(), 2);
    }

    /// The updater manages EXACTLY the two live tiers {lumina-fast, lumina-deep}
    /// and the retired `lumina` main tier is gone — a regression guard so the dead
    /// main key can't silently creep back into the managed set.
    #[test]
    fn managed_keys_are_exactly_fast_and_deep() {
        let managed: BTreeSet<&str> = LUMINA_ALIAS_KEYS.iter().copied().collect();
        assert_eq!(managed, BTreeSet::from(["lumina-fast", "lumina-deep"]));
        assert!(
            !managed.contains("lumina"),
            "retired main tier must not be managed"
        );
    }

    #[tokio::test]
    async fn store_set_repoints_one_key_only() {
        let mut statics = HashMap::new();
        statics.insert("lumina-fast".to_string(), "a:1b".to_string());
        statics.insert("lumina-deep".to_string(), "b:2b".to_string());
        let store = LuminaAliasStore::from_static(&statics);

        store.set("lumina-fast", "c:3b".to_string()).await;
        assert_eq!(store.resolve("lumina-fast").as_deref(), Some("c:3b"));
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
        let cands = gate_and_build(&raws, None, 0.0, f64::INFINITY);
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
        // metric-v2 is q-heavy, so divergence needs a genuine near-tie in quality
        // that the small responsiveness weight can tip on the fast tier but not the
        // deep tier. `fast_model` is only slightly lower quality but faster (r
        // stays UNDER the 40 tok/s saturation so it still differentiates);
        // `deep_model` is the quality leader but slow; `filler` spreads the
        // normalization so q_norm isn't a degenerate 0/1.
        let raws = vec![
            raw("fast_model", Some(85.0), Some(4.0), Some(40.0), true),
            raw("deep_model", Some(100.0), Some(4.0), Some(10.0), true),
            raw("filler", Some(1.0), Some(4.0), Some(10.0), true),
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

        let fast = &plans[0].repoint;
        let deep = &plans[1].repoint;
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
        // current = "b"; top = "a" but only marginally better. A `filler` spreads
        // the q normalization so "a" and "b" land at 1.0 vs ~0.98 (a genuine
        // near-tie in score) rather than the degenerate 0/1 a 2-candidate min-max
        // would force. Equal a/r keep the difference purely in q.
        let raws = vec![
            raw("a", Some(100.0), Some(4.0), Some(30.0), true),
            raw("b", Some(98.0), Some(4.0), Some(30.0), true),
            raw("filler", Some(1.0), Some(4.0), Some(30.0), true),
        ];
        let cfg = AliasUpdaterConfig {
            switch_margin: 0.50, // large margin → a's tiny edge can't clear it
            ..AliasUpdaterConfig::default()
        };
        let tiers = vec![TierPlan {
            key: "lumina-fast".into(),
            weights: cfg.fast_weights,
            current: Some("b".into()),
        }];
        let plans = plan_repoints(&raws, None, &cfg, &tiers);
        assert!(
            matches!(plans[0].repoint, Repoint::Keep { .. }),
            "within-margin near-tie must NOT switch (got {:?})",
            plans[0].repoint
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
            key: "lumina-fast".into(),
            weights: cfg.fast_weights,
            current: Some("b".into()),
        }];
        let plans = plan_repoints(&raws, None, &cfg, &tiers);
        match &plans[0].repoint {
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
            key: "lumina-fast".into(),
            weights: cfg.fast_weights,
            current: Some("a".into()),
        }];
        let plans = plan_repoints(&raws, None, &cfg, &tiers);
        assert!(matches!(plans[0].repoint, Repoint::Keep { .. }));
    }

    // ── Gates ──────────────────────────────────────────────────────────────────

    #[test]
    fn gate_drops_guard_failing_candidates() {
        let raws = vec![
            raw("guarded_out", Some(5.0), Some(5.0), Some(100.0), false),
            raw("ok", Some(4.0), Some(4.0), Some(50.0), true),
        ];
        let cands = gate_and_build(&raws, None, 0.0, f64::INFINITY);
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].model_id, "ok");
    }

    #[test]
    fn gate_drops_below_quality_floor() {
        let raws = vec![
            raw("too_low", Some(2.5), Some(5.0), Some(100.0), true),
            raw("ok", Some(3.5), Some(4.0), Some(50.0), true),
        ];
        let cands = gate_and_build(&raws, None, 3.0, f64::INFINITY);
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].model_id, "ok");
    }

    #[test]
    fn gate_drops_models_without_quality_data() {
        // No q value at all → cannot clear the floor, dropped.
        let raws = vec![raw("no_q", None, Some(5.0), Some(100.0), true)];
        let cands = gate_and_build(&raws, None, 0.0, f64::INFINITY);
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
        let cands = gate_and_build(&raws, Some(&eligible), 0.0, f64::INFINITY);
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
            key: "lumina-fast".into(),
            weights: cfg.fast_weights,
            current: Some("granite4.1:30b".into()),
        }];
        let plans = plan_repoints(&raws, None, &cfg, &tiers);
        assert!(
            matches!(plans[0].repoint, Repoint::Keep { .. }),
            "no candidates must keep the current target, never blank it"
        );
    }

    #[test]
    fn current_no_longer_a_candidate_adopts_top() {
        // current="stale" is not in the ranked set anymore → adopt the top pick.
        let raws = vec![raw("fresh", Some(4.5), Some(4.5), Some(80.0), true)];
        let cfg = AliasUpdaterConfig::default();
        let tiers = vec![TierPlan {
            key: "lumina-fast".into(),
            weights: cfg.fast_weights,
            current: Some("stale".into()),
        }];
        let plans = plan_repoints(&raws, None, &cfg, &tiers);
        match &plans[0].repoint {
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
        // All three are large enough to clear the size gate (the arch behaviour is
        // what this test exercises).
        let sizes: HashMap<String, u64> = ["mixed", "dead", "unprofiled"]
            .iter()
            .map(|s| (s.to_string(), 30_000_000_000u64))
            .collect();

        let eligible = compute_eligible(&raws, &sizes, 5_000_000_000, &rmap);
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
        // "known" is NOT in the registry sizes map → dropped even with clean routing.
        let sizes: HashMap<String, u64> = HashMap::new();
        let eligible = compute_eligible(&raws, &sizes, 5_000_000_000, &rmap);
        assert!(
            eligible.is_empty(),
            "non-servable models must be filtered out"
        );
    }

    // ── metric-v2 FIX A: size/capability gate ──────────────────────────────────

    #[test]
    fn size_gate_excludes_tiny_and_unsized_models() {
        use crate::serving::profile::RoutingMap;
        let rmap = RoutingMap::empty();
        let raws = vec![
            raw("granite4.1:30b", Some(489.0), Some(4.0), Some(25.0), true),
            raw("qwen2.5:0.5b", Some(344.0), Some(4.0), Some(400.0), true),
            raw("unsized:xb", Some(999.0), Some(5.0), Some(50.0), true),
        ];
        let mut sizes: HashMap<String, u64> = HashMap::new();
        sizes.insert("granite4.1:30b".into(), 18_000_000_000); // ~18 GB
        sizes.insert("qwen2.5:0.5b".into(), 400_000_000); // ~0.4 GB
        sizes.insert("unsized:xb".into(), 0); // unknown/zero size

        let eligible = compute_eligible(&raws, &sizes, 5_000_000_000, &rmap);
        assert!(
            eligible.contains("granite4.1:30b"),
            "a capable ~30B model must survive the size gate"
        );
        assert!(
            !eligible.contains("qwen2.5:0.5b"),
            "a 0.5B model must be excluded by the size gate — it must never win"
        );
        assert!(
            !eligible.contains("unsized:xb"),
            "an unknown/zero-size model must be excluded (fail-safe)"
        );
    }

    #[test]
    fn tiny_high_count_model_excluded_granite_wins_both_tiers() {
        // End-to-end metric-v2 regression: a qwen2.5:0.5b-shaped input (tiny size,
        // deceptively-high q, MAX responsiveness) must be EXCLUDED, and a
        // granite-shaped input (large size, high per-answer q, modest speed) must
        // WIN both the fast and deep tiers.
        use crate::serving::profile::RoutingMap;
        let rmap = RoutingMap::empty();
        let raws = vec![
            raw("granite4.1:30b", Some(489.0), Some(4.5), Some(25.0), true),
            raw("qwen2.5:0.5b", Some(344.0), Some(4.5), Some(400.0), true),
        ];
        let mut sizes: HashMap<String, u64> = HashMap::new();
        sizes.insert("granite4.1:30b".into(), 18_000_000_000);
        sizes.insert("qwen2.5:0.5b".into(), 400_000_000);

        let eligible = compute_eligible(&raws, &sizes, 5_000_000_000, &rmap);
        assert!(
            !eligible.contains("qwen2.5:0.5b"),
            "0.5B excluded pre-scoring"
        );

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
        let plans = plan_repoints(&raws, Some(&eligible), &cfg, &tiers);
        for plan in &plans {
            match &plan.repoint {
                Repoint::Switch { to, .. } => assert_eq!(
                    to, "granite4.1:30b",
                    "the capable model must win tier '{}', never the tiny one",
                    plan.key
                ),
                other => panic!("expected {} to switch to granite, got {other:?}", plan.key),
            }
            // The tiny model must not even appear in the ranked set.
            assert!(
                plan.ranked.iter().all(|s| s.model_id != "qwen2.5:0.5b"),
                "qwen2.5:0.5b must never be a scored candidate"
            );
        }
    }

    // ── metric-v2 FIX C: responsiveness saturation ─────────────────────────────

    #[test]
    fn responsiveness_saturates_so_tiny_fast_gets_no_runaway_bonus() {
        // Two capable models: "solid" is high-q and fast-enough (45 tok/s);
        // "speedy" is lower-q but blazing (400 tok/s). With saturation at 40 tok/s
        // both hit the r ceiling, so r cannot rescue the weaker model.
        let raws = vec![
            raw("solid", Some(480.0), Some(4.5), Some(45.0), true),
            raw("speedy", Some(300.0), Some(4.0), Some(400.0), true),
        ];
        let cfg = AliasUpdaterConfig::default(); // r_saturation 40, fast 0.55/0.30/0.15
        let cands = gate_and_build(&raws, None, 0.0, cfg.r_saturation_tok_s);
        // Both r's are capped at 40 → equal → r_norm degenerate (0.5 each), so the
        // blend is decided by q/a where "solid" dominates.
        let solid = cands.iter().find(|c| c.model_id == "solid").unwrap();
        let speedy = cands.iter().find(|c| c.model_id == "speedy").unwrap();
        assert_eq!(
            solid.r, 40.0,
            "solid's 45 tok/s saturates to the 40 ceiling"
        );
        assert_eq!(
            speedy.r, 40.0,
            "speedy's 400 tok/s saturates to the 40 ceiling"
        );
        let ranked = rank(&cands, cfg.fast_weights);
        assert_eq!(
            ranked[0].model_id, "solid",
            "q wins once speed is saturated"
        );
    }

    // ── metric-v2 FIX D: single-candidate must still pass all gates ─────────────

    #[test]
    fn single_candidate_selected_only_if_it_passes_gates() {
        use crate::serving::profile::RoutingMap;
        let rmap = RoutingMap::empty();
        // Only a tiny model is available → size gate empties the candidate set →
        // keep current (never promote the 0.5B model just because it's the only one).
        let raws = vec![raw(
            "qwen2.5:0.5b",
            Some(344.0),
            Some(4.5),
            Some(400.0),
            true,
        )];
        let mut sizes: HashMap<String, u64> = HashMap::new();
        sizes.insert("qwen2.5:0.5b".into(), 400_000_000);
        let eligible = compute_eligible(&raws, &sizes, 5_000_000_000, &rmap);
        assert!(eligible.is_empty());

        let cfg = AliasUpdaterConfig::default();
        let tiers = vec![TierPlan {
            key: "lumina-fast".into(),
            weights: cfg.fast_weights,
            current: Some("granite4.1:30b".into()),
        }];
        let plans = plan_repoints(&raws, Some(&eligible), &cfg, &tiers);
        assert!(
            matches!(plans[0].repoint, Repoint::Keep { .. }),
            "a lone sub-threshold candidate must NOT be adopted — keep current"
        );
    }

    // ── Defensive round: safety guards can't be silently disabled ──────────────

    #[test]
    #[serial_test::serial]
    fn min_size_env_zero_falls_back_to_default_not_disabled() {
        // FIX 1: MIN_SIZE_BYTES=0 must NOT disable the size floor.
        std::env::remove_var("CHORD_ALIAS_MIN_SIZE_BYTES");
        std::env::set_var("CHORD_ALIAS_MIN_SIZE_BYTES", "0");
        let cfg = AliasUpdaterConfig::from_env();
        assert_eq!(
            cfg.min_size_bytes, DEFAULT_MIN_SIZE_BYTES,
            "MIN_SIZE_BYTES=0 must fall back to the default floor, never disable it"
        );
        std::env::remove_var("CHORD_ALIAS_MIN_SIZE_BYTES");
    }

    #[test]
    fn size_gate_zero_floor_still_excludes_tiny_via_sanitizer() {
        // FIX 1 at the point of use: even if `0` reaches compute_eligible directly
        // (bypassing from_env), the sanitizer restores the default floor so a 0.5B
        // model is STILL excluded — the tiny-model hole cannot be re-opened.
        use crate::serving::profile::RoutingMap;
        let rmap = RoutingMap::empty();
        let raws = vec![
            raw("granite4.1:30b", Some(489.0), Some(4.5), Some(25.0), true),
            raw("qwen2.5:0.5b", Some(344.0), Some(4.5), Some(400.0), true),
        ];
        let mut sizes: HashMap<String, u64> = HashMap::new();
        sizes.insert("granite4.1:30b".into(), 18_000_000_000);
        sizes.insert("qwen2.5:0.5b".into(), 400_000_000);

        // Pass a `0` floor DIRECTLY — must behave like the default, not "disabled".
        let eligible = compute_eligible(&raws, &sizes, 0, &rmap);
        assert!(
            eligible.contains("granite4.1:30b"),
            "the capable model survives"
        );
        assert!(
            !eligible.contains("qwen2.5:0.5b"),
            "a 0 floor must NOT re-enable a 0.5B model — sanitizer restores the default"
        );
    }

    #[test]
    fn direct_config_bad_saturation_sanitized_at_use() {
        // FIX 2: a directly-constructed config with an invalid saturation must be
        // corrected at the point of use, not silently propagated.
        let speedy = vec![raw("speedy", Some(300.0), Some(4.0), Some(400.0), true)];
        for bad in [0.0, -5.0, f64::NAN, f64::INFINITY] {
            let cands = gate_and_build(&speedy, None, 0.0, bad);
            assert_eq!(
                cands[0].r, DEFAULT_R_SATURATION_TOK_S,
                "invalid saturation {bad} must be sanitized to the default at use \
                 (400 tok/s capped to {DEFAULT_R_SATURATION_TOK_S})"
            );
        }
    }

    #[test]
    fn direct_config_bad_weights_do_not_corrupt_scores() {
        // FIX 2 for weights: a NaN/negative weight must not produce NaN/inverted
        // scores — rank sanitizes each weight to a finite, non-negative value.
        let cands = vec![
            AliasCandidate {
                model_id: "a".into(),
                q: 5.0,
                a: 5.0,
                r: 5.0,
            },
            AliasCandidate {
                model_id: "b".into(),
                q: 1.0,
                a: 1.0,
                r: 1.0,
            },
        ];
        let bad_weights = BlendWeights {
            q: f64::NAN,
            a: -1.0,
            r: f64::INFINITY,
        };
        let ranked = rank(&cands, bad_weights);
        for sc in &ranked {
            assert!(
                sc.score.is_finite(),
                "no candidate score may be NaN/inf even with corrupt weights"
            );
            assert!(
                sc.score >= 0.0,
                "scores stay non-negative with sanitized weights"
            );
        }
    }

    // ── FIX 6: invalid config falls back to defaults ───────────────────────────

    #[test]
    #[serial_test::serial]
    fn from_env_rejects_invalid_weights_and_refresh() {
        let keys = [
            "CHORD_ALIAS_REFRESH_SECS",
            "CHORD_ALIAS_MIN_QUALITY",
            "CHORD_ALIAS_MIN_SIZE_BYTES",
            "CHORD_ALIAS_R_SATURATION_TOK_S",
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
        // Invalid values: NaN, negative, zero, and a zero/too-small interval.
        std::env::set_var("CHORD_ALIAS_REFRESH_SECS", "0"); // → floored to MIN_REFRESH_SECS
        std::env::set_var("CHORD_ALIAS_W_FAST_Q", "NaN"); // → default
        std::env::set_var("CHORD_ALIAS_W_FAST_A", "-1.0"); // → default
        std::env::set_var("CHORD_ALIAS_W_DEEP_Q", "inf"); // → default
        std::env::set_var("CHORD_ALIAS_SWITCH_MARGIN", "-0.5"); // → default
        std::env::set_var("CHORD_ALIAS_R_SATURATION_TOK_S", "0"); // → default (must be > 0)

        let cfg = AliasUpdaterConfig::from_env();
        let def = AliasUpdaterConfig::default();

        assert_eq!(
            cfg.refresh_secs, MIN_REFRESH_SECS,
            "0 refresh floored, never panics interval"
        );
        assert_eq!(
            cfg.r_saturation_tok_s, def.r_saturation_tok_s,
            "zero saturation → default (never zero-out responsiveness)"
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

    // ── Per-tier quality floor (CHRD-PTQ-01) ───────────────────────────────────

    /// The per-tier override actually filters ONLY its own tier: with the deep
    /// floor unfloored (0) and the fast floor high, a mediocre-q but capable model
    /// is a candidate for deep but is dropped for fast. Uses distinct weights so
    /// the assertion is about the floor, not the blend.
    #[test]
    fn per_tier_floor_overrides_only_its_tier() {
        // "big" clears the high fast floor; "mid" does not, but is above 0.
        let raws = vec![
            raw("big", Some(200.0), Some(4.0), Some(20.0), true),
            raw("mid", Some(50.0), Some(9.0), Some(20.0), true),
        ];
        let cfg = AliasUpdaterConfig {
            min_quality: 0.0,
            min_quality_fast: 140.0, // fast tier floored above "mid"
            min_quality_deep: 0.0,   // deep tier unfloored
            ..AliasUpdaterConfig::default()
        };
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

        // fast: only "big" survives its 140 floor → it must win.
        let fast = plans.iter().find(|d| d.key == "lumina-fast").unwrap();
        assert_eq!(fast.ranked.len(), 1, "fast floor drops 'mid'");
        assert_eq!(fast.ranked[0].model_id, "big");

        // deep: unfloored → both "big" and "mid" are candidates.
        let deep = plans.iter().find(|d| d.key == "lumina-deep").unwrap();
        assert_eq!(deep.ranked.len(), 2, "deep is unfloored → both candidates");
        let deep_ids: BTreeSet<&str> = deep.ranked.iter().map(|s| s.model_id.as_str()).collect();
        assert!(deep_ids.contains("big") && deep_ids.contains("mid"));
    }

    /// An UNSET per-tier var falls back to the global floor: with only the global
    /// floor set (per-tier vars unset), every tier applies that global floor —
    /// preserving the prior global-only behavior.
    #[test]
    fn unset_per_tier_floor_falls_back_to_global() {
        // from_env with only the global var set → both per-tier floors equal it.
        let keys = [
            "CHORD_ALIAS_MIN_QUALITY",
            "CHORD_ALIAS_MIN_QUALITY_FAST",
            "CHORD_ALIAS_MIN_QUALITY_DEEP",
        ];
        for k in keys {
            std::env::remove_var(k);
        }
        std::env::set_var("CHORD_ALIAS_MIN_QUALITY", "75");
        let cfg = AliasUpdaterConfig::from_env();
        assert_eq!(cfg.min_quality, 75.0);
        assert_eq!(cfg.min_quality_for("lumina-fast"), 75.0, "fast falls back");
        assert_eq!(cfg.min_quality_for("lumina-deep"), 75.0, "deep falls back");
        assert_eq!(cfg.min_quality_for("other"), 75.0, "unknown key → global");
        for k in keys {
            std::env::remove_var(k);
        }
    }

    /// A set per-tier var OVERRIDES the global for its tier only (from_env path).
    #[test]
    fn set_per_tier_floor_overrides_global_from_env() {
        let keys = [
            "CHORD_ALIAS_MIN_QUALITY",
            "CHORD_ALIAS_MIN_QUALITY_FAST",
            "CHORD_ALIAS_MIN_QUALITY_DEEP",
        ];
        for k in keys {
            std::env::remove_var(k);
        }
        std::env::set_var("CHORD_ALIAS_MIN_QUALITY", "140");
        std::env::set_var("CHORD_ALIAS_MIN_QUALITY_DEEP", "0");
        let cfg = AliasUpdaterConfig::from_env();
        assert_eq!(
            cfg.min_quality_for("lumina-fast"),
            140.0,
            "fast = global (unset)"
        );
        assert_eq!(
            cfg.min_quality_for("lumina-deep"),
            0.0,
            "deep overridden to 0"
        );
        for k in keys {
            std::env::remove_var(k);
        }
    }

    /// Global-only behavior is preserved: applying a uniform global floor via the
    /// per-tier path yields the same surviving candidate set every tier would have
    /// gotten from the old single shared `gate_and_build(min_quality)` call.
    #[test]
    fn global_only_behavior_preserved() {
        let raws = vec![
            raw("keep", Some(200.0), Some(4.0), Some(20.0), true),
            raw("drop", Some(10.0), Some(4.0), Some(20.0), true),
        ];
        // Uniform floor of 100 across all tiers (per-tier == global).
        let cfg = AliasUpdaterConfig {
            min_quality: 100.0,
            min_quality_fast: 100.0,
            min_quality_deep: 100.0,
            ..AliasUpdaterConfig::default()
        };
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
        // The OLD path: one shared gate_and_build with the same global floor.
        let shared = gate_and_build(&raws, None, 100.0, cfg.r_saturation_tok_s);
        let shared_ids: BTreeSet<&str> = shared.iter().map(|c| c.model_id.as_str()).collect();
        assert_eq!(shared_ids, BTreeSet::from(["keep"]));
        for plan in &plans {
            let ids: BTreeSet<&str> = plan.ranked.iter().map(|s| s.model_id.as_str()).collect();
            assert_eq!(
                ids, shared_ids,
                "per-tier path with uniform floor == old shared gate for tier {}",
                plan.key
            );
        }
    }
}
