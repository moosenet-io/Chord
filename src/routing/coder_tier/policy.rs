//! RVXR-01 admission POLICY — the pure decision layer.
//!
//! Deliberately free of every crate-internal dependency (only `std` + `serde`),
//! for two reasons:
//!
//! 1. **It is the part that must be mutation-tested**, and a file with no crate
//!    deps can be `#[path]`-included by a standalone harness that compiles in
//!    seconds. The harness includes THIS file, not a copy of it, so the tested
//!    source cannot drift from the shipped source.
//! 2. Every threshold here is a **claim about a live host**, and claims need the
//!    same scrutiny as code. Isolating them makes them reviewable as a set
//!    instead of scattered through control flow where they read as configuration
//!    and get waved through.
//!
//! ## The capacity signal is GTT, and only GTT
//! See `sensors.rs` for the measurements. In one line: on this unified-memory APU
//! the models live in GTT, which is invisible to process RSS (0.52 + 0.40 GB
//! reported by the ollama runners while holding 28.7 GB), and the system "free
//! memory" counters are both wrong in opposite directions — `MemFree` fails low,
//! `MemAvailable` fails high and read 89.4 GB ten minutes before the host hung.
//!
//! ## Two independent vetoes
//! Capacity (is there GTT room for the coder?) and pressure (is the host in
//! trouble?) are separate questions with separate sensors. Either can refuse a
//! load; either can evict a resident coder. They are not combined into one score,
//! because a single number would let a comfortable reading on one axis mask a
//! dangerous one on the other.

use serde::Serialize;

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

/// Env-driven config. No model names, no infrastructure literals — an alias KEY
/// and a set of documented, individually-overridable thresholds.
#[derive(Debug, Clone, PartialEq)]
pub struct CoderTierConfig {
    /// `CHORD_CODER_TIER_ENABLED` — **default FALSE**. Ships dark.
    pub enabled: bool,
    /// `CHORD_CODER_TIER_ALIAS` — the Chord ALIAS KEY the coder resolves through
    /// (never a model name).
    pub alias: String,
    /// `CHORD_CODER_TIER_MIN_IDLE_SECS` (default 900) — how long assistant
    /// traffic must have been absent before the tier will even arm.
    pub min_idle_secs: u64,
    /// `CHORD_CODER_TIER_ARM_CONFIRM_SECS` (default 120) — how long the tier must
    /// stay armed, still idle, before it loads. This is what makes "never load on
    /// the first idle tick" structural rather than incidental.
    pub arm_confirm_secs: u64,
    /// `CHORD_CODER_TIER_COOLDOWN_SECS` (default 600) — after an eviction, no
    /// reload until this has elapsed, however idle the box looks.
    pub cooldown_secs: u64,
    /// `CHORD_CODER_TIER_GTT_MARGIN_GB` (default 8.0) — GTT that must remain free
    /// on top of the coder's estimated runtime footprint before a load, and below
    /// which a resident coder is evicted.
    pub gtt_margin_gb: f64,
    /// `CHORD_CODER_TIER_FOOTPRINT_FACTOR` (default 1.8) — multiplier applied to
    /// the registry's ON-DISK size to estimate RUNTIME footprint (weights + KV
    /// cache + context). Measured on this host: `granite4.1:8b` is 4.98 GiB on
    /// disk and 8.93 GiB resident. Clamped to `>= 1.0` — a factor below one would
    /// UNDER-estimate, the one direction that is unsafe.
    pub footprint_factor: f64,
    /// `CHORD_CODER_TIER_MAX_COMMIT_RATIO` (default 0.85) — refuse to load, and
    /// evict if resident, once `Committed_AS / CommitLimit` exceeds this.
    ///
    /// `Committed_AS` counts memory the kernel has PROMISED, so unlike
    /// `MemAvailable` it rises as danger rises rather than staying reassuring.
    /// 0.85 leaves real room below the limit; it is a claim about this host and
    /// is overridable, not a universal truth.
    pub max_commit_ratio: f64,
    /// `CHORD_CODER_TIER_MAX_SWAP_GROWTH_GB` (default 0.5) — swap USED growing by
    /// more than this between two observations is treated as active pressure.
    ///
    /// The TREND is the signal, not the level: a host with a few GB of long-cold
    /// swap is fine, while a host swapping *right now* is in trouble regardless of
    /// the absolute figure. This moved during the real incident.
    pub max_swap_growth_gb: f64,
    /// `CHORD_CODER_TIER_MAX_LEASES` (default 8) — hard cap on concurrent review
    /// leases.
    ///
    /// The user hot path cancels EVERY live lease, so an unbounded lease set puts
    /// unbounded work on the request that must be fastest. A cap makes that work
    /// O(1) in the worst case. Eight is comfortably above any real review panel.
    pub max_leases: usize,
    /// `CHORD_CODER_TIER_EVICT_BUDGET_SECS` (default 60) — if an eviction task
    /// dies or wedges, the tick force-resolves the phase so the tier cannot stick.
    pub evict_budget_secs: u64,
}

impl Default for CoderTierConfig {
    fn default() -> Self {
        CoderTierConfig {
            enabled: false,
            alias: "lumina-coder".to_string(),
            min_idle_secs: 900,
            arm_confirm_secs: 120,
            cooldown_secs: 600,
            gtt_margin_gb: 8.0,
            footprint_factor: 1.8,
            max_commit_ratio: 0.85,
            max_swap_growth_gb: 0.5,
            max_leases: 8,
            evict_budget_secs: 60,
        }
    }
}

fn env_flag(key: &str, default: bool) -> bool {
    match std::env::var(key) {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            if v.is_empty() {
                default
            } else {
                !matches!(v.as_str(), "0" | "false" | "no" | "off")
            }
        }
        Err(_) => default,
    }
}

pub(crate) fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(default)
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
        .unwrap_or(default)
}

fn env_string(key: &str, default: &str) -> String {
    std::env::var(key)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| default.to_string())
}

impl CoderTierConfig {
    pub fn from_env() -> Self {
        let d = CoderTierConfig::default();
        CoderTierConfig {
            enabled: env_flag("CHORD_CODER_TIER_ENABLED", d.enabled),
            alias: env_string("CHORD_CODER_TIER_ALIAS", &d.alias),
            min_idle_secs: env_u64("CHORD_CODER_TIER_MIN_IDLE_SECS", d.min_idle_secs),
            arm_confirm_secs: env_u64("CHORD_CODER_TIER_ARM_CONFIRM_SECS", d.arm_confirm_secs),
            cooldown_secs: env_u64("CHORD_CODER_TIER_COOLDOWN_SECS", d.cooldown_secs),
            gtt_margin_gb: env_f64("CHORD_CODER_TIER_GTT_MARGIN_GB", d.gtt_margin_gb),
            footprint_factor: env_f64("CHORD_CODER_TIER_FOOTPRINT_FACTOR", d.footprint_factor)
                .max(1.0),
            max_commit_ratio: env_f64("CHORD_CODER_TIER_MAX_COMMIT_RATIO", d.max_commit_ratio),
            max_swap_growth_gb: env_f64(
                "CHORD_CODER_TIER_MAX_SWAP_GROWTH_GB",
                d.max_swap_growth_gb,
            ),
            max_leases: env_u64("CHORD_CODER_TIER_MAX_LEASES", d.max_leases as u64).max(1) as usize,
            evict_budget_secs: env_u64("CHORD_CODER_TIER_EVICT_BUDGET_SECS", d.evict_budget_secs),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Phases, reasons, decisions
// ─────────────────────────────────────────────────────────────────────────────

/// The tier's lifecycle phase. [`Phase::Assistant`] is the resting, always-safe
/// state: no coder loaded, nothing borrowed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "phase", rename_all = "kebab-case")]
pub enum Phase {
    Assistant,
    Arming { armed_at: u64 },
    Loading,
    Ready,
    Evicting { since: u64 },
    Cooldown { until: u64 },
}

impl Phase {
    /// Is the coder loaded or being loaded (i.e. is memory borrowed)?
    pub fn holds_memory(self) -> bool {
        matches!(self, Phase::Loading | Phase::Ready | Phase::Evicting { .. })
    }

    pub fn id(self) -> &'static str {
        match self {
            Phase::Assistant => "assistant",
            Phase::Arming { .. } => "arming",
            Phase::Loading => "loading",
            Phase::Ready => "ready",
            Phase::Evicting { .. } => "evicting",
            Phase::Cooldown { .. } => "cooldown",
        }
    }
}

/// Why an eviction fired. Only [`EvictReason::AssistantArrived`] is the fast
/// path; the rest are safety nets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum EvictReason {
    AssistantArrived,
    ActivityObserved,
    Disabled,
    /// Free GTT fell below the margin — something took the window.
    GttPressure,
    /// `Committed_AS`/`CommitLimit` or the swap trend says the host is in trouble.
    SystemPressure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Hold,
    Rest,
    Arm,
    Load,
    Evict(EvictReason),
    ForceResolveEvict,
}

/// Everything one tick observes. The clock and every sensor value are passed in,
/// never read here — so the whole state machine is exhaustively testable with no
/// sleeping, no wall clock, and no `/sys`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Observation {
    pub now: u64,
    /// Seconds since the last ASSISTANT (non-coder-lease) inference admission.
    pub assistant_idle_secs: u64,
    pub assistant_inflight: bool,
    /// Free GTT in GiB — the ONLY capacity signal. `None` ⇒ unreadable.
    pub gtt_free_gb: Option<f64>,
    /// `Committed_AS / CommitLimit`. `None` ⇒ unreadable.
    pub commit_ratio: Option<f64>,
    /// Growth in swap USED since the previous observation, GiB. `None` ⇒ no prior
    /// observation to compare against (the first tick).
    pub swap_growth_gb: Option<f64>,
    /// Estimated ON-DISK size of the resolved coder, GiB. `None` ⇒ unknown.
    pub coder_gb: Option<f64>,
    pub coder_resolved: bool,
    /// Is a teardown task still running (even one already abandoned by the
    /// force-resolve budget)?
    ///
    /// A NEW load must not begin while one is, and the reason is a side effect the
    /// bookkeeping generation cannot reach: an abandoned teardown that is already
    /// INSIDE `stop_coder` will eventually stop the backend it named. If cooldown
    /// expired and a new cycle loaded the same model in the meantime, that late
    /// stop kills the NEW coder while the tier believes it is `Ready`. Guarding
    /// the bookkeeping is not enough — the load has to wait for the old teardown
    /// to actually finish.
    pub teardown_inflight: bool,
}

/// Estimated RUNTIME footprint: on-disk size inflated by the measured factor.
pub fn runtime_footprint_gb(disk_gb: f64, factor: f64) -> f64 {
    disk_gb * factor.max(1.0)
}

/// Swap growth between two readings. Only GROWTH counts — swap being reclaimed is
/// not pressure, so a negative delta clamps to zero.
pub fn swap_growth_gb(previous_used_gb: Option<f64>, current_used_gb: f64) -> Option<f64> {
    let prev = previous_used_gb?;
    Some((current_used_gb - prev).max(0.0))
}

/// Is there provably GTT room for the coder? Fail-CLOSED: any unknown is "no".
pub fn coder_fits(obs: &Observation, cfg: &CoderTierConfig) -> bool {
    match (obs.coder_resolved, obs.gtt_free_gb, obs.coder_gb) {
        (true, Some(free), Some(disk)) => {
            if !free.is_finite() || !disk.is_finite() {
                return false;
            }
            runtime_footprint_gb(disk, cfg.footprint_factor) + cfg.gtt_margin_gb <= free
        }
        _ => false,
    }
}

/// Is the host under memory pressure right now?
///
/// Two independent signals, either sufficient. An UNREADABLE sensor is not
/// pressure — a sensor gap is not evidence, and treating it as pressure would
/// thrash on precisely what it cannot see. (Unreadability still blocks a LOAD,
/// via [`coder_fits`] and [`can_admit`]; the asymmetry is deliberate — refusing
/// to take the window on no information is cheap, while evicting on no
/// information is a thrash loop.)
pub fn pressure_reason(obs: &Observation, cfg: &CoderTierConfig) -> Option<EvictReason> {
    if let Some(free) = obs.gtt_free_gb {
        if free.is_finite() && free < cfg.gtt_margin_gb {
            return Some(EvictReason::GttPressure);
        }
    }
    if let Some(ratio) = obs.commit_ratio {
        if ratio.is_finite() && ratio > cfg.max_commit_ratio {
            return Some(EvictReason::SystemPressure);
        }
    }
    if let Some(growth) = obs.swap_growth_gb {
        if growth.is_finite() && growth > cfg.max_swap_growth_gb {
            return Some(EvictReason::SystemPressure);
        }
    }
    None
}

/// May we take the window at all? Requires a provable fit AND positively
/// healthy pressure sensors — an unreadable `Committed_AS` blocks a load even
/// though it would not force an eviction.
pub fn can_admit(obs: &Observation, cfg: &CoderTierConfig) -> bool {
    if obs.teardown_inflight {
        // See `Observation::teardown_inflight` — a late stop from an abandoned
        // teardown would land on the coder this load is about to start.
        return false;
    }
    if !coder_fits(obs, cfg) {
        return false;
    }
    if pressure_reason(obs, cfg).is_some() {
        return false;
    }
    // Loading is the risky direction: require the pressure sensor to have
    // actually reported, rather than admitting on its silence.
    // Finite AND non-negative: a negative ratio is a corrupt sensor, and it is
    // never "> max", so a finiteness-only check would admit on it.
    obs.commit_ratio.is_some_and(|r| r.is_finite() && r >= 0.0)
}

/// Is the box quiet enough to consider borrowing the window?
pub fn assistant_quiet(obs: &Observation, min_idle_secs: u64) -> bool {
    !obs.assistant_inflight && obs.assistant_idle_secs >= min_idle_secs
}

/// The pure decision. No clock, no I/O, no globals.
///
/// Ordering is deliberate:
/// 1. a disabled tier that holds memory gives it back before anything else;
/// 2. an eviction already in flight is never second-guessed, only bounded;
/// 3. **assistant activity beats everything** — it evicts from any memory-holding
///    phase and disarms an arming one, before fit or pressure is consulted;
/// 4. memory pressure evicts a resident coder even while the box is quiet;
/// 5. a cooldown is honoured even when the box is perfectly idle;
/// 6. only then can arming, and only after arm-confirm AND admission, loading.
pub fn decide(cfg: &CoderTierConfig, phase: Phase, obs: &Observation) -> Decision {
    if !cfg.enabled {
        return match phase {
            Phase::Evicting { since } => {
                if obs.now.saturating_sub(since) >= cfg.evict_budget_secs {
                    Decision::ForceResolveEvict
                } else {
                    Decision::Hold
                }
            }
            p if p.holds_memory() => Decision::Evict(EvictReason::Disabled),
            Phase::Assistant => Decision::Hold,
            _ => Decision::Rest,
        };
    }

    if let Phase::Evicting { since } = phase {
        return if obs.now.saturating_sub(since) >= cfg.evict_budget_secs {
            Decision::ForceResolveEvict
        } else {
            Decision::Hold
        };
    }

    if !assistant_quiet(obs, cfg.min_idle_secs) {
        return match phase {
            Phase::Loading | Phase::Ready => Decision::Evict(EvictReason::ActivityObserved),
            Phase::Arming { .. } => Decision::Rest,
            _ => Decision::Hold,
        };
    }

    match phase {
        Phase::Cooldown { until } => {
            if obs.now >= until {
                Decision::Rest
            } else {
                Decision::Hold
            }
        }
        Phase::Assistant => Decision::Arm,
        Phase::Arming { armed_at } => {
            if obs.now.saturating_sub(armed_at) < cfg.arm_confirm_secs {
                Decision::Hold
            } else if can_admit(obs, cfg) {
                Decision::Load
            } else {
                Decision::Hold
            }
        }
        // Resident: the box is quiet, but the window can still be taken by
        // something that is not an inference request at all. Re-check every tick;
        // never assume the headroom persisted.
        Phase::Loading | Phase::Ready => match pressure_reason(obs, cfg) {
            Some(reason) => Decision::Evict(reason),
            None => Decision::Hold,
        },
        Phase::Evicting { .. } => Decision::Hold, // handled above
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_on() -> CoderTierConfig {
        CoderTierConfig {
            enabled: true,
            alias: "test-coder-alias".into(),
            ..CoderTierConfig::default()
        }
    }

    /// A quiet, healthy, roomy observation — the fixture every positive case
    /// starts from.
    fn quiet(now: u64) -> Observation {
        Observation {
            now,
            assistant_idle_secs: 5_000,
            assistant_inflight: false,
            gtt_free_gb: Some(50.0),
            commit_ratio: Some(0.40),
            swap_growth_gb: Some(0.0),
            coder_gb: Some(17.3),
            coder_resolved: true,
            teardown_inflight: false,
        }
    }

    /// GUARD AGAINST A LYING FIXTURE: if `quiet()` ever stops being admissible,
    /// every "we refused" test below would pass for the wrong reason.
    #[test]
    fn the_baseline_fixture_is_genuinely_admissible() {
        let cfg = cfg_on();
        let obs = quiet(1_000);
        assert!(coder_fits(&obs, &cfg), "fixture must fit");
        assert!(pressure_reason(&obs, &cfg).is_none(), "fixture must be healthy");
        assert!(can_admit(&obs, &cfg), "fixture must be admissible");
        assert!(assistant_quiet(&obs, cfg.min_idle_secs), "fixture must be quiet");
        // ...and it must actually reach Load from Arming, or the anti-thrash
        // tests below prove nothing.
        assert_eq!(
            decide(&cfg, Phase::Arming { armed_at: 0 }, &obs),
            Decision::Load
        );
    }

    // ── anti-thrash ──────────────────────────────────────────────────────────

    #[test]
    fn a_load_waits_for_an_outstanding_teardown_to_actually_finish() {
        // The bookkeeping generation cannot un-issue a `stop_coder` that is
        // already in flight, so a new cycle must not start underneath one.
        let cfg = cfg_on();
        let mut obs = quiet(1_000);
        obs.teardown_inflight = true;
        assert!(!can_admit(&obs, &cfg));
        assert_eq!(
            decide(&cfg, Phase::Arming { armed_at: 0 }, &obs),
            Decision::Hold
        );
        // CONTROL: the same observation admits once the teardown has finished.
        obs.teardown_inflight = false;
        assert!(can_admit(&obs, &cfg));
    }

    #[test]
    fn never_loads_on_the_first_idle_tick() {
        let cfg = cfg_on();
        assert_eq!(decide(&cfg, Phase::Assistant, &quiet(1_000)), Decision::Arm);
    }

    #[test]
    fn load_requires_both_the_idle_dwell_and_the_arm_confirm() {
        let cfg = cfg_on();
        let armed = Phase::Arming { armed_at: 1_000 };
        assert_eq!(decide(&cfg, armed, &quiet(1_000)), Decision::Hold);
        assert_eq!(decide(&cfg, armed, &quiet(1_119)), Decision::Hold);
        assert_eq!(decide(&cfg, armed, &quiet(1_120)), Decision::Load);
    }

    #[test]
    fn short_idle_never_arms() {
        let cfg = cfg_on();
        let mut obs = quiet(9_999);
        obs.assistant_idle_secs = cfg.min_idle_secs - 1;
        assert_eq!(decide(&cfg, Phase::Assistant, &obs), Decision::Hold);
        assert_eq!(decide(&cfg, Phase::Arming { armed_at: 1 }, &obs), Decision::Rest);
    }

    #[test]
    fn inflight_assistant_beats_a_large_idle_counter() {
        let cfg = cfg_on();
        let mut obs = quiet(9_999);
        obs.assistant_inflight = true;
        assert_eq!(decide(&cfg, Phase::Assistant, &obs), Decision::Hold);
        assert_eq!(
            decide(&cfg, Phase::Ready, &obs),
            Decision::Evict(EvictReason::ActivityObserved)
        );
        assert_eq!(
            decide(&cfg, Phase::Loading, &obs),
            Decision::Evict(EvictReason::ActivityObserved)
        );
        assert_eq!(decide(&cfg, Phase::Arming { armed_at: 1 }, &obs), Decision::Rest);
    }

    #[test]
    fn cooldown_blocks_a_reload_however_idle_the_box_is() {
        let cfg = cfg_on();
        let cd = Phase::Cooldown { until: 5_000 };
        assert_eq!(decide(&cfg, cd, &quiet(4_999)), Decision::Hold);
        assert_eq!(decide(&cfg, cd, &quiet(5_000)), Decision::Rest);
    }

    #[test]
    fn evicting_is_bounded_and_cannot_wedge() {
        let cfg = cfg_on();
        let ev = Phase::Evicting { since: 1_000 };
        assert_eq!(decide(&cfg, ev, &quiet(1_059)), Decision::Hold);
        assert_eq!(decide(&cfg, ev, &quiet(1_060)), Decision::ForceResolveEvict);
        let mut busy = quiet(1_010);
        busy.assistant_inflight = true;
        assert_eq!(decide(&cfg, ev, &busy), Decision::Hold);
    }

    // ── capacity: GTT, and the runtime-inflation factor ──────────────────────

    #[test]
    fn fit_is_fail_closed_on_every_unknown() {
        let cfg = cfg_on();
        let base = quiet(1_000);

        let mut unresolved = base;
        unresolved.coder_resolved = false;
        assert!(!coder_fits(&unresolved, &cfg));

        let mut no_gtt = base;
        no_gtt.gtt_free_gb = None;
        assert!(!coder_fits(&no_gtt, &cfg));
        assert!(!can_admit(&no_gtt, &cfg));

        let mut no_size = base;
        no_size.coder_gb = None;
        assert!(!coder_fits(&no_size, &cfg));

        let mut nan = base;
        nan.gtt_free_gb = Some(f64::NAN);
        assert!(!coder_fits(&nan, &cfg));

        assert_eq!(
            decide(&cfg, Phase::Arming { armed_at: 0 }, &no_gtt),
            Decision::Hold
        );
    }

    #[test]
    fn fit_sizes_against_runtime_footprint_not_on_disk_size() {
        let cfg = cfg_on(); // factor 1.8, margin 8.0
        assert!((runtime_footprint_gb(17.3, 1.8) - 31.14).abs() < 1e-9);

        let mut obs = quiet(1_000);
        obs.coder_gb = Some(17.3);

        obs.gtt_free_gb = Some(40.0);
        assert!(coder_fits(&obs, &cfg), "39.14 needed, 40 free");

        obs.gtt_free_gb = Some(39.0);
        assert!(!coder_fits(&obs, &cfg), "39.14 needed, only 39 free");

        // Naive on-disk sizing would have said yes at 26 GB free. It must not.
        obs.gtt_free_gb = Some(26.0);
        assert!(
            !coder_fits(&obs, &cfg),
            "sizing against the raw on-disk figure under-counts by most of a coder model"
        );
    }

    #[test]
    fn footprint_factor_is_clamped_so_it_can_only_over_estimate() {
        assert_eq!(runtime_footprint_gb(10.0, 0.1), 10.0);
        assert_eq!(runtime_footprint_gb(10.0, 1.0), 10.0);
        assert!((runtime_footprint_gb(10.0, 2.0) - 20.0).abs() < 1e-9);
    }

    // ── pressure: the signals that actually moved ────────────────────────────

    #[test]
    fn gtt_pressure_evicts_a_resident_coder() {
        let cfg = cfg_on();
        let mut obs = quiet(1_000);
        obs.gtt_free_gb = Some(3.0);
        assert_eq!(pressure_reason(&obs, &cfg), Some(EvictReason::GttPressure));
        assert_eq!(
            decide(&cfg, Phase::Ready, &obs),
            Decision::Evict(EvictReason::GttPressure)
        );
    }

    #[test]
    fn a_high_commit_ratio_evicts_even_with_plenty_of_gtt_free() {
        // The incident shape: GTT looks fine, the host is still in trouble.
        let cfg = cfg_on();
        let mut obs = quiet(1_000);
        obs.gtt_free_gb = Some(50.0);
        obs.commit_ratio = Some(0.93);
        assert_eq!(pressure_reason(&obs, &cfg), Some(EvictReason::SystemPressure));
        assert_eq!(
            decide(&cfg, Phase::Ready, &obs),
            Decision::Evict(EvictReason::SystemPressure)
        );
        assert!(!can_admit(&obs, &cfg), "and it must block a load too");
    }

    #[test]
    fn swap_growth_is_pressure_but_a_static_swap_level_is_not() {
        let cfg = cfg_on();
        let mut obs = quiet(1_000);

        // Actively swapping RIGHT NOW: pressure.
        obs.swap_growth_gb = Some(2.0);
        assert_eq!(pressure_reason(&obs, &cfg), Some(EvictReason::SystemPressure));

        // A host sitting on cold swap that is not growing: NOT pressure. The
        // level alone must never trigger, or a box with a few GB of long-cold
        // swap could never host the tier at all.
        obs.swap_growth_gb = Some(0.0);
        assert_eq!(pressure_reason(&obs, &cfg), None);

        // Swap being RECLAIMED is not pressure either.
        assert_eq!(swap_growth_gb(Some(5.0), 3.0), Some(0.0));
        assert_eq!(swap_growth_gb(Some(3.0), 5.0), Some(2.0));
        // No prior reading ⇒ no trend yet.
        assert_eq!(swap_growth_gb(None, 5.0), None);
    }

    #[test]
    fn an_unreadable_sensor_blocks_a_load_but_does_not_force_an_eviction() {
        let cfg = cfg_on();
        let mut obs = quiet(1_000);
        obs.commit_ratio = None;

        // Not evidence of pressure — evicting on a sensor gap would thrash on
        // exactly what it cannot see.
        assert_eq!(pressure_reason(&obs, &cfg), None);
        assert_eq!(decide(&cfg, Phase::Ready, &obs), Decision::Hold);

        // But taking the window on no information is refused.
        assert!(!can_admit(&obs, &cfg));
        assert_eq!(
            decide(&cfg, Phase::Arming { armed_at: 0 }, &obs),
            Decision::Hold
        );

        // NaN is a broken sensor, not a healthy zero.
        obs.commit_ratio = Some(f64::NAN);
        assert_eq!(pressure_reason(&obs, &cfg), None);
        assert!(!can_admit(&obs, &cfg));

        // A NEGATIVE ratio is corrupt too, and it is never "> max" — so a
        // finiteness-only admission check would wave it straight through.
        obs.commit_ratio = Some(-0.5);
        assert_eq!(pressure_reason(&obs, &cfg), None);
        assert!(!can_admit(&obs, &cfg), "a negative ratio must not admit a load");
    }

    #[test]
    fn the_first_tick_has_no_swap_trend_and_that_does_not_block_admission() {
        // `swap_growth_gb: None` means "no prior observation", which is the
        // normal state of the very first tick. It must not be conflated with a
        // broken sensor, or the tier could never start.
        let cfg = cfg_on();
        let mut obs = quiet(1_000);
        obs.swap_growth_gb = None;
        assert_eq!(pressure_reason(&obs, &cfg), None);
        assert!(can_admit(&obs, &cfg));
    }

    #[test]
    fn a_disabled_tier_gives_memory_back_and_then_rests() {
        let mut cfg = cfg_on();
        cfg.enabled = false;
        let obs = quiet(1_000);
        assert_eq!(
            decide(&cfg, Phase::Ready, &obs),
            Decision::Evict(EvictReason::Disabled)
        );
        assert_eq!(
            decide(&cfg, Phase::Loading, &obs),
            Decision::Evict(EvictReason::Disabled)
        );
        assert_eq!(decide(&cfg, Phase::Assistant, &obs), Decision::Hold);
        assert_eq!(decide(&cfg, Phase::Cooldown { until: 9_999 }, &obs), Decision::Rest);
        assert_eq!(
            decide(&cfg, Phase::Evicting { since: 1_000 }, &quiet(1_060)),
            Decision::ForceResolveEvict
        );
    }

    #[test]
    fn the_tier_ships_dark() {
        assert!(
            !CoderTierConfig::default().enabled,
            "this touches a live shared GPU host — it must be opt-in"
        );
    }

    #[test]
    fn every_threshold_default_is_a_sane_claim() {
        let d = CoderTierConfig::default();
        assert!(d.min_idle_secs > 0 && d.arm_confirm_secs > 0 && d.cooldown_secs > 0);
        assert!(d.footprint_factor >= 1.0);
        assert!(d.gtt_margin_gb > 0.0);
        assert!(d.max_commit_ratio > 0.0 && d.max_commit_ratio < 1.0);
        assert!(d.max_swap_growth_gb > 0.0);
        assert!(d.max_leases > 0, "an unbounded lease set puts unbounded work on the user path");
        assert!(d.evict_budget_secs > 0);
    }
}
