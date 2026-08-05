//! RVXR-01 memory sensing — measure the thing that actually holds the memory.
//!
//! ## Why this file exists at all
//! The first cut of this tier sized its admission decision on
//! [`crate::config::read_free_vram_gb`] and, behind it, the system free-memory
//! counters. A fleet measurement (CRB-0, 2026-08-05) showed that is the wrong
//! sensor in three separate ways, each of which independently breaks the
//! decision:
//!
//! **1. GTT is INVISIBLE TO PROCESS RSS.** On this unified-memory APU the model
//! weights live in GTT (`amdgpu`), not in any process's resident set. Measured:
//! the two ollama runner processes reported **0.52 GB and 0.40 GB** in `ps`
//! while holding **28.7 GB** of GTT. Every process in the system summed to
//! 10.2 GB against 28.7 GB of GTT. A fit check that reads process memory does
//! not see the models *at all* — it would happily conclude there is room for a
//! coder next to a fully-resident assistant cohort.
//!
//! **2. `MemFree` fails LOW.** Page cache means a near-zero `MemFree` is the
//! normal, healthy state of a build host. Gating on it would refuse to load
//! essentially always — an inert feature that looks like a working one.
//!
//! **3. `MemAvailable` fails HIGH, and is ANTI-CORRELATED with danger.** It read
//! **89.4 GB while the host was about ten minutes from hanging**, because the
//! reclaimable page cache it counts is exactly what evaporates under pressure.
//! It is at its most reassuring when the situation is at its worst. This is the
//! single most dangerous input that could be wired into an admission gate, and it
//! is the one that looks most obviously correct.
//!
//! So: **GTT for capacity, `Committed_AS` and the swap trend for pressure.** Both
//! of the latter moved during the real incident; neither is reclaim-inflated.
//!
//! ## A threshold is a claim, and claims need the same scrutiny as code
//! Every number below is a documented default with a named env override and a
//! test, not a literal buried in a branch. Thresholds feel like configuration, so
//! they skip review — that is precisely how two wrong ones got adopted here.
//!
//! ## Parsing is pure; only the two readers touch the filesystem
//! Everything that decides anything takes `&str` and is unit-tested against
//! captured real output. The filesystem functions are thin and fail-closed:
//! unreadable or unparseable is always `None`, and `None` always means "refuse",
//! never "assume fine".

use std::path::PathBuf;

const BYTES_PER_GIB: f64 = 1_073_741_824.0;

/// Default glob-free directory pattern for the amdgpu GTT counters. The amdgpu
/// kernel driver exposes `mem_info_gtt_used` / `mem_info_gtt_total` (bytes) under
/// each card's device directory — a stable, documented kernel ABI, not a
/// host-specific path. The card INDEX varies, so it is discovered rather than
/// assumed (see [`discover_gtt_paths`]). Override the whole pair explicitly with
/// [`GTT_USED_ENV`] / [`GTT_TOTAL_ENV`] on a host where discovery is wrong.
const DRM_CLASS_DIR: &str = "/sys/class/drm";

/// Env var naming an explicit `mem_info_gtt_used` path (bytes).
pub const GTT_USED_ENV: &str = "CHORD_CODER_TIER_GTT_USED_PATH";
/// Env var naming an explicit `mem_info_gtt_total` path (bytes).
pub const GTT_TOTAL_ENV: &str = "CHORD_CODER_TIER_GTT_TOTAL_PATH";

/// Is `name` a `/sys/class/drm` CARD DEVICE node (`card0`, `card12`), as opposed
/// to a connector node (`card0-DP-1`) or a render node (`renderD128`)?
///
/// The non-empty index check is not decoration: `"card"[4..]` is empty and
/// `.all()` over an empty iterator is vacuously `true`, so a bare `card` would
/// otherwise be accepted.
pub fn is_card_device_name(name: &str) -> bool {
    let Some(index) = name.strip_prefix("card") else {
        return false;
    };
    !index.is_empty() && index.chars().all(|c| c.is_ascii_digit())
}

/// A GTT capacity reading, in GiB. This is the ONLY capacity signal the tier
/// admits on, because it is the only one that can see model residency.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GttReading {
    pub used_gb: f64,
    pub total_gb: f64,
}

impl GttReading {
    /// Free GTT, never negative (a total/used pair sampled a moment apart can
    /// legitimately cross).
    pub fn free_gb(&self) -> f64 {
        (self.total_gb - self.used_gb).max(0.0)
    }
}

/// Parse one sysfs byte counter. Fail-closed on anything unexpected: empty,
/// non-numeric, negative, or non-finite.
pub fn parse_bytes_gb(raw: &str) -> Option<f64> {
    let bytes: f64 = raw.trim().parse().ok()?;
    if bytes.is_finite() && bytes >= 0.0 {
        Some(bytes / BYTES_PER_GIB)
    } else {
        None
    }
}

/// Build a [`GttReading`] from the two raw counter contents. Fail-closed if
/// either is unreadable, or if `total` is zero (a driver that reports no GTT at
/// all is not a host with infinite room).
pub fn parse_gtt(used_raw: &str, total_raw: &str) -> Option<GttReading> {
    let used_gb = parse_bytes_gb(used_raw)?;
    let total_gb = parse_bytes_gb(total_raw)?;
    if total_gb <= 0.0 {
        return None;
    }
    Some(GttReading { used_gb, total_gb })
}

/// Locate the GTT counter pair: an explicit env override if set, else the first
/// card under `/sys/class/drm` that exposes BOTH counters.
///
/// Discovery rather than a hardcoded `card0`: the index is not stable across
/// boots or across hosts, and a wrong index would silently read a different GPU.
/// Returns `None` when no card exposes them (e.g. a non-amdgpu host) — which
/// makes the tier refuse to load, the correct outcome for "I cannot see the
/// memory".
pub fn discover_gtt_paths() -> Option<(PathBuf, PathBuf)> {
    let explicit_used = std::env::var(GTT_USED_ENV).ok().filter(|s| !s.trim().is_empty());
    let explicit_total = std::env::var(GTT_TOTAL_ENV).ok().filter(|s| !s.trim().is_empty());
    match resolve_override(explicit_used.as_deref(), explicit_total.as_deref()) {
        OverrideOutcome::Use(u, t) => return Some((u, t)),
        OverrideOutcome::Partial => return None,
        OverrideOutcome::None => {}
    }
    let entries = std::fs::read_dir(DRM_CLASS_DIR).ok()?;
    let mut cards: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                // `card0`, `card1`, ... — NOT `card0-DP-1` connector nodes (no
                // `device/mem_info_*` counters), and NOT a bare `card`: an empty
                // index passes `.all()` vacuously, so it must be excluded
                // explicitly. Found by the test below, not by reading this line.
                .map(|n| is_card_device_name(n))
                .unwrap_or(false)
        })
        .collect();
    cards.sort();
    let mut found: Vec<(PathBuf, PathBuf)> = Vec::new();
    for card in cards {
        let used = card.join("device/mem_info_gtt_used");
        let total = card.join("device/mem_info_gtt_total");
        if used.is_file() && total.is_file() {
            found.push((used, total));
        }
    }
    // These counters are PER-DEVICE, not host-wide. With more than one candidate
    // card there is nothing here that ties a card to the device the coder backend
    // will actually load on, so picking the lexicographically first one could
    // authorise a load against a healthy IDLE GPU while the APU that will serve
    // the model is full. Ambiguity fails CLOSED; the operator pins the pair with
    // `CHORD_CODER_TIER_GTT_USED_PATH` / `..._TOTAL_PATH`.
    select_sole_candidate(found)
}

/// What an explicit path override amounts to. Pure so both branches are tested.
#[derive(Debug, PartialEq, Eq)]
pub enum OverrideOutcome {
    /// Both paths given — use them.
    Use(PathBuf, PathBuf),
    /// Exactly ONE path given. Fails CLOSED rather than falling back to
    /// discovery: an operator who set one path meant to PIN the device, and
    /// quietly auto-discovering the other half is how you end up measuring a
    /// different GPU than the one that was configured.
    Partial,
    /// Neither given — discovery is appropriate.
    None,
}

pub fn resolve_override(used: Option<&str>, total: Option<&str>) -> OverrideOutcome {
    match (used, total) {
        (Some(u), Some(t)) => {
            OverrideOutcome::Use(PathBuf::from(u.trim()), PathBuf::from(t.trim()))
        }
        (None, None) => OverrideOutcome::None,
        _ => OverrideOutcome::Partial,
    }
}

/// Pick THE candidate card, or refuse.
///
/// These counters are PER-DEVICE, not host-wide. With more than one candidate
/// there is nothing tying a card to the device the coder will actually load on,
/// so picking the lexicographically first could authorise a load against a
/// healthy IDLE GPU while the APU that serves the model is full. Ambiguity fails
/// CLOSED; the operator pins the pair with the override env vars.
pub fn select_sole_candidate(mut found: Vec<(PathBuf, PathBuf)>) -> Option<(PathBuf, PathBuf)> {
    match found.len() {
        1 => found.pop(),
        0 => None,
        n => {
            tracing::warn!(
                candidates = n,
                "coder tier: multiple GPUs expose GTT counters — refusing to guess which one \
                 serves the coder; set CHORD_CODER_TIER_GTT_USED_PATH/_TOTAL_PATH to pin it"
            );
            None
        }
    }
}

/// Read the live GTT capacity. `None` ⇒ cannot see the memory ⇒ the tier refuses
/// to load (and, while resident, holds rather than thrashing — see the policy).
pub fn read_gtt() -> Option<GttReading> {
    let (used_path, total_path) = discover_gtt_paths()?;
    let used = std::fs::read_to_string(used_path).ok()?;
    let total = std::fs::read_to_string(total_path).ok()?;
    parse_gtt(&used, &total)
}

/// The system-pressure signals that actually moved during the real incident.
/// Deliberately does NOT carry `MemFree` or `MemAvailable` — there is no field to
/// accidentally reach for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CommitReading {
    /// `Committed_AS` — total memory the kernel has PROMISED. Unlike
    /// `MemAvailable` this counts obligations, not reclaimable cache, so it rises
    /// as danger rises instead of falling.
    pub committed_gb: f64,
    /// `CommitLimit`.
    pub commit_limit_gb: f64,
    /// `SwapTotal - SwapFree`. Its TREND is the signal; the level alone is not.
    pub swap_used_gb: f64,
}

impl CommitReading {
    /// Committed as a fraction of the commit limit. `None` when the limit is
    /// zero/absent (overcommit accounting off) — fail-closed at the caller.
    pub fn commit_ratio(&self) -> Option<f64> {
        if self.commit_limit_gb > 0.0 {
            Some(self.committed_gb / self.commit_limit_gb)
        } else {
            None
        }
    }
}

/// Parse `/proc/meminfo`. A universal kernel interface, not infrastructure.
/// Values there are in kB. Fail-closed if any required field is missing.
pub fn parse_meminfo(raw: &str) -> Option<CommitReading> {
    let mut committed_kb: Option<f64> = None;
    let mut limit_kb: Option<f64> = None;
    let mut swap_total_kb: Option<f64> = None;
    let mut swap_free_kb: Option<f64> = None;
    for line in raw.lines() {
        let Some((key, rest)) = line.split_once(':') else {
            continue;
        };
        let value = rest.split_whitespace().next().and_then(|v| v.parse::<f64>().ok());
        let Some(value) = value else { continue };
        match key.trim() {
            "Committed_AS" => committed_kb = Some(value),
            "CommitLimit" => limit_kb = Some(value),
            "SwapTotal" => swap_total_kb = Some(value),
            "SwapFree" => swap_free_kb = Some(value),
            _ => {}
        }
    }
    let kb_to_gb = |kb: f64| kb * 1024.0 / BYTES_PER_GIB;
    let committed_gb = kb_to_gb(committed_kb?);
    let commit_limit_gb = kb_to_gb(limit_kb?);
    // Swap is optional on a host with none configured; absent ⇒ zero used, which
    // is the truth, not a fail-closed case.
    let swap_used_gb = match (swap_total_kb, swap_free_kb) {
        (Some(t), Some(f)) => kb_to_gb((t - f).max(0.0)),
        _ => 0.0,
    };
    Some(CommitReading {
        committed_gb,
        commit_limit_gb,
        swap_used_gb,
    })
}

/// Read the live commit/swap signals. `None` ⇒ unreadable ⇒ fail-closed.
pub fn read_commit() -> Option<CommitReading> {
    parse_meminfo(&std::fs::read_to_string("/proc/meminfo").ok()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gtt_free_is_total_minus_used_and_never_negative() {
        let r = GttReading {
            used_gb: 28.7,
            total_gb: 60.0,
        };
        assert!((r.free_gb() - 31.3).abs() < 1e-9);
        // A used/total pair sampled a moment apart can legitimately cross.
        let crossed = GttReading {
            used_gb: 61.0,
            total_gb: 60.0,
        };
        assert_eq!(crossed.free_gb(), 0.0);
    }

    #[test]
    fn gtt_parsing_is_fail_closed() {
        // Real shape: a bare byte count with a trailing newline.
        let r = parse_gtt("30819254272\n", "64424509440\n").expect("parses");
        assert!((r.used_gb - 28.7).abs() < 0.1, "28.7 GiB used — the measured value");
        assert!((r.total_gb - 60.0).abs() < 0.1);

        assert!(parse_gtt("", "64424509440").is_none(), "empty");
        assert!(parse_gtt("N/A", "64424509440").is_none(), "non-numeric");
        assert!(parse_gtt("-1", "64424509440").is_none(), "negative");
        // A driver reporting zero total GTT is not a host with infinite room.
        assert!(parse_gtt("0", "0").is_none(), "zero total");
    }

    #[test]
    fn meminfo_parses_the_signals_that_actually_move() {
        // Captured shape (kB), including the counters we deliberately IGNORE.
        let raw = "\
MemTotal:       131165244 kB
MemFree:          3878912 kB
MemAvailable:    90337280 kB
Buffers:            51200 kB
Cached:          80000000 kB
SwapCached:          1024 kB
SwapTotal:       25165824 kB
SwapFree:        21706752 kB
CommitLimit:     90748446 kB
Committed_AS:    41943040 kB
";
        let c = parse_meminfo(raw).expect("parses");
        assert!((c.committed_gb - 40.0).abs() < 0.1);
        assert!((c.commit_limit_gb - 86.55).abs() < 0.1);
        // 25165824 - 21706752 = 3459072 kB = 3.3 GiB used.
        assert!((c.swap_used_gb - 3.3).abs() < 0.1);
        let ratio = c.commit_ratio().expect("limit is positive");
        assert!((ratio - 0.462).abs() < 0.01);
    }

    #[test]
    fn meminfo_is_fail_closed_on_missing_commit_fields() {
        // MemAvailable alone is NOT enough — and must never be enough. This is
        // the counter that read 89.4 GB ten minutes before the host hung.
        let only_available = "MemTotal: 131165244 kB\nMemAvailable: 93750000 kB\n";
        assert!(
            parse_meminfo(only_available).is_none(),
            "a reading without Committed_AS/CommitLimit must fail closed, not fall back to MemAvailable"
        );
        assert!(parse_meminfo("").is_none());
        assert!(parse_meminfo("garbage without colons").is_none());
    }

    #[test]
    fn a_host_with_no_swap_reports_zero_used_not_a_failure() {
        let raw = "CommitLimit: 90748446 kB\nCommitted_AS: 41943040 kB\n";
        let c = parse_meminfo(raw).expect("swap is optional");
        assert_eq!(c.swap_used_gb, 0.0);
    }

    #[test]
    fn commit_ratio_is_none_when_overcommit_accounting_is_off() {
        let c = CommitReading {
            committed_gb: 40.0,
            commit_limit_gb: 0.0,
            swap_used_gb: 0.0,
        };
        assert!(c.commit_ratio().is_none());
    }

    #[test]
    fn ambiguous_multi_gpu_discovery_fails_closed() {
        let a = (PathBuf::from("/a/used"), PathBuf::from("/a/total"));
        let b = (PathBuf::from("/b/used"), PathBuf::from("/b/total"));
        // CONTROL: exactly one candidate resolves, else the negatives are hollow.
        assert_eq!(select_sole_candidate(vec![a.clone()]), Some(a.clone()));
        // These counters are PER-DEVICE: with two GPUs, guessing could authorise
        // a load against an idle card while the serving APU is full.
        assert_eq!(select_sole_candidate(vec![a, b]), None);
        assert_eq!(select_sole_candidate(vec![]), None);
    }

    #[test]
    fn a_partial_path_override_fails_closed() {
        assert_eq!(
            resolve_override(Some("/u"), Some("/t")),
            OverrideOutcome::Use(PathBuf::from("/u"), PathBuf::from("/t"))
        );
        assert_eq!(resolve_override(None, None), OverrideOutcome::None);
        // Half-configured must NOT silently auto-discover the other half.
        assert_eq!(resolve_override(Some("/u"), None), OverrideOutcome::Partial);
        assert_eq!(resolve_override(None, Some("/t")), OverrideOutcome::Partial);
    }

    #[test]
    fn discovery_ignores_drm_connector_nodes() {
        // `/sys/class/drm` contains both `cardN` device nodes and `cardN-DP-1`
        // connector nodes; only the former have device/mem_info_* counters.
        // Asserts against the REAL predicate the discovery uses, not a copy of
        // it — a copied predicate would keep passing after the original drifted.
        let is_card = is_card_device_name;
        assert!(is_card("card0"));
        assert!(is_card("card12"));
        assert!(!is_card("card0-DP-1"));
        assert!(!is_card("card0-HDMI-A-1"));
        assert!(!is_card("renderD128"));
        assert!(!is_card("card"), "bare `card` has no index");
    }
}
