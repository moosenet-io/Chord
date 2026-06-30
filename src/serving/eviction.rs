//! Tier-aware eviction policy (S85 SRV-05) — the pure decision core.
//!
//! This module is the *judgment* half of the residency manager and contains NO
//! IO, NO locks, and NO async: given a snapshot of the resident set, the
//! candidate's footprint, and the free VRAM, it computes an [`EvictionPlan`] —
//! the ordered list of residents to evict so the candidate fits, or a clear
//! verdict that it cannot be admitted right now. [`residency`](super::residency)
//! owns the concurrency, IO and state-file; this module owns the policy so the
//! policy is exhaustively unit-testable on its own.
//!
//! ## The v1 conservative policy (never wedges the host, never stalls Lumina)
//! To admit a model `M` needing `need_gb` against `free_gb`:
//!   1. If `M` already fits (`need_gb <= free_gb`) → admit, evict nothing.
//!   2. Otherwise reclaim VRAM in TIER ORDER:
//!      - **transient** residents first (cheap to reload), oldest-use first, until
//!        `M` fits;
//!      - if still short, **keep-warm** residents — EXCEPT the pinned chat-role
//!        model, which is NEVER in the eviction set — LRU first, but only as a
//!        *deferred* action: keep-warm contention is QUEUED first by the caller
//!        (bounded wait) and the LRU keep-warm is evicted only after the wait
//!        threshold expires (see [`EvictionPlan::Queue`]).
//!   3. If even evicting every non-pinned resident cannot make `M` fit → the
//!      pinned-only stall: [`EvictionPlan::CannotAdmit`] (never force-evict chat).
//!
//! The plan is "what to do", computed twice by the caller: once on the fast path
//! (before queueing) and once after the bounded wait, so a resident finishing
//! during the wait is reflected.

use terminus_rs::intake::serving::Runtime;

/// Residency tier of a model held (or to be held) in VRAM. Drives eviction order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// A short-lived (e.g. build/validation) resident — cheap to reload, the
    /// FIRST thing evicted under pressure.
    Transient,
    /// A big, slow-to-cold-load model held resident across requests. Evicted only
    /// after transients are exhausted, and only LRU-after-wait (queue first).
    KeepWarm,
    /// The live chat-role model. NEVER evicted while serving (the pinned-chat
    /// invariant). At most one resident may carry this tier.
    PinnedChat,
}

impl Tier {
    /// Stable lowercase id used in the residency state file and sanitized events.
    pub fn id(self) -> &'static str {
        match self {
            Tier::Transient => "transient",
            Tier::KeepWarm => "keep-warm",
            Tier::PinnedChat => "pinned-chat",
        }
    }

    /// Whether a resident of this tier may EVER be evicted to admit another model.
    /// The pinned chat model never can — that is the load-bearing safety invariant.
    pub fn is_evictable(self) -> bool {
        !matches!(self, Tier::PinnedChat)
    }
}

/// A snapshot view of one resident, as the policy sees it. Cheap to clone; carries
/// only what the decision needs (no endpoint/secret crosses into the policy).
#[derive(Debug, Clone, PartialEq)]
pub struct ResidentView {
    /// The resident model's id (used only to name it in the plan/events).
    pub model_id: String,
    /// The runtime backing it (carried through to the plan for the caller).
    pub runtime: Runtime,
    /// Its VRAM footprint in GB (what evicting it would reclaim).
    pub vram_gb: f64,
    /// Its tier (drives eviction order + the chat-pin exclusion).
    pub tier: Tier,
    /// A monotonic "last used" tick; LOWER = less recently used (LRU target).
    /// The caller stamps this from a monotonic counter, so the policy needs no
    /// clock and stays pure/deterministic.
    pub last_used_tick: u64,
}

/// One unit of an eviction plan: evict this resident (reclaiming `vram_gb`).
#[derive(Debug, Clone, PartialEq)]
pub struct EvictTarget {
    pub model_id: String,
    pub runtime: Runtime,
    pub vram_gb: f64,
    pub tier: Tier,
}

/// The computed admission decision for a candidate model.
#[derive(Debug, Clone, PartialEq)]
pub enum EvictionPlan {
    /// The candidate fits now (possibly after evicting the listed TRANSIENT
    /// residents). An empty list ⇒ it already fit. Eviction here is immediate and
    /// cheap (transients only) — no queueing needed.
    AdmitAfterEvicting(Vec<EvictTarget>),
    /// The candidate cannot fit on transients alone; admitting it requires
    /// evicting keep-warm residents. The caller must QUEUE first (bounded wait)
    /// and only then evict these LRU-ordered keep-warm targets. `transient_first`
    /// are evicted immediately regardless (they are always cheap); `keep_warm_lru`
    /// are the deferred, post-wait evictions.
    Queue {
        /// Transient residents to evict immediately (cheap; do this even while
        /// queued — it may already make room).
        transient_first: Vec<EvictTarget>,
        /// Keep-warm residents to evict, LRU first, ONLY after the wait threshold.
        keep_warm_lru: Vec<EvictTarget>,
    },
    /// The pinned-only stall: even evicting every non-pinned resident cannot make
    /// the candidate fit (or its footprint is unknown and nothing is evictable).
    /// The caller returns `CannotAdmit` and NEVER evicts the pinned chat model.
    CannotAdmit,
}

/// Compute the eviction plan to admit a candidate needing `need_gb` against
/// `free_gb`, given the current `residents`.
///
/// Pure and deterministic. `need_gb == None` means the candidate's footprint is
/// unknown; combined with the fail-safe `free_gb == None` (VRAM unreadable) the
/// only safe verdict is to NOT admit on the fast path — see the caller, which
/// queues. Here, an unknown footprint with readable free VRAM is treated
/// conservatively as "does not fit" unless there is literally nothing resident.
///
/// Invariants enforced here (the safety floor):
///   - the **pinned chat** model is never placed in any eviction list;
///   - **transient** targets are always ordered before keep-warm and chosen
///     LRU-first;
///   - keep-warm targets are LRU-first and only ever appear under
///     [`EvictionPlan::Queue`] (never evicted without the caller's bounded wait);
///   - if the candidate cannot fit even after evicting EVERY evictable resident →
///     [`EvictionPlan::CannotAdmit`] (the pinned-only stall).
pub fn plan_admission(
    need_gb: Option<f64>,
    free_gb: f64,
    residents: &[ResidentView],
) -> EvictionPlan {
    // Unknown footprint: only safe to admit if there is nothing resident at all
    // (an empty host). Otherwise be conservative and require reclamation we can't
    // size → CannotAdmit (the caller already fails safe / queues around this).
    let need = match need_gb {
        Some(n) if n.is_finite() && n >= 0.0 => n,
        _ => {
            return if residents.is_empty() && free_gb >= 0.0 {
                EvictionPlan::AdmitAfterEvicting(Vec::new())
            } else {
                EvictionPlan::CannotAdmit
            };
        }
    };

    // (1) Already fits → admit, evict nothing.
    if need <= free_gb {
        return EvictionPlan::AdmitAfterEvicting(Vec::new());
    }

    // Partition evictable residents by tier (pinned chat is excluded entirely).
    let mut transients: Vec<&ResidentView> = residents
        .iter()
        .filter(|r| r.tier == Tier::Transient)
        .collect();
    let mut keep_warms: Vec<&ResidentView> = residents
        .iter()
        .filter(|r| r.tier == Tier::KeepWarm)
        .collect();
    // LRU first within each tier (lowest last_used_tick first).
    transients.sort_by_key(|r| r.last_used_tick);
    keep_warms.sort_by_key(|r| r.last_used_tick);

    let to_target = |r: &ResidentView| EvictTarget {
        model_id: r.model_id.clone(),
        runtime: r.runtime,
        vram_gb: r.vram_gb,
        tier: r.tier,
    };

    // (2a) Evict transients (cheap) LRU-first until M fits.
    let mut reclaimed = 0.0_f64;
    let mut transient_plan: Vec<EvictTarget> = Vec::new();
    for r in &transients {
        if free_gb + reclaimed >= need {
            break;
        }
        reclaimed += r.vram_gb;
        transient_plan.push(to_target(r));
    }
    if free_gb + reclaimed >= need {
        // Transients alone suffice → immediate admit, keep-warm untouched.
        return EvictionPlan::AdmitAfterEvicting(transient_plan);
    }

    // (2b) Still short → we must consider keep-warm. Build the LRU keep-warm plan,
    // taking only as many as needed on top of all transients.
    let mut keep_warm_plan: Vec<EvictTarget> = Vec::new();
    for r in &keep_warms {
        if free_gb + reclaimed >= need {
            break;
        }
        reclaimed += r.vram_gb;
        keep_warm_plan.push(to_target(r));
    }

    // (3) If even evicting every evictable resident can't make room → the
    // pinned-only stall. NEVER evict the pinned chat model.
    if free_gb + reclaimed < need {
        return EvictionPlan::CannotAdmit;
    }

    // We can fit, but only by touching keep-warm → QUEUE first, evict LRU
    // keep-warm only after the wait threshold. Transients are evicted immediately.
    EvictionPlan::Queue {
        transient_first: transient_plan,
        keep_warm_lru: keep_warm_plan,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn res(id: &str, gb: f64, tier: Tier, tick: u64) -> ResidentView {
        ResidentView {
            model_id: id.into(),
            runtime: Runtime::LlamaCpp,
            vram_gb: gb,
            tier,
            last_used_tick: tick,
        }
    }

    #[test]
    fn fits_immediately_evicts_nothing() {
        let plan = plan_admission(Some(10.0), 20.0, &[res("a", 8.0, Tier::KeepWarm, 1)]);
        assert_eq!(plan, EvictionPlan::AdmitAfterEvicting(vec![]));
    }

    #[test]
    fn transient_evicted_before_keep_warm_when_transient_suffices() {
        // need 12, free 5. A transient (10) alone makes room → keep-warm untouched.
        let residents = vec![
            res("kw", 30.0, Tier::KeepWarm, 1),
            res("t", 10.0, Tier::Transient, 2),
        ];
        let plan = plan_admission(Some(12.0), 5.0, &residents);
        match plan {
            EvictionPlan::AdmitAfterEvicting(targets) => {
                assert_eq!(targets.len(), 1);
                assert_eq!(targets[0].model_id, "t");
                assert_eq!(targets[0].tier, Tier::Transient);
            }
            other => panic!("expected immediate transient-only admit, got {other:?}"),
        }
    }

    #[test]
    fn transient_lru_order() {
        // Two transients; the LRU (lower tick) is taken first.
        let residents = vec![
            res("new", 5.0, Tier::Transient, 9),
            res("old", 5.0, Tier::Transient, 1),
        ];
        // need 8, free 2 → need to reclaim 6, one 5GB transient is not enough,
        // take both — but the FIRST taken must be the LRU "old".
        let plan = plan_admission(Some(8.0), 2.0, &residents);
        match plan {
            EvictionPlan::AdmitAfterEvicting(t) => {
                assert_eq!(t[0].model_id, "old", "LRU transient evicted first");
                assert_eq!(t.len(), 2);
            }
            other => panic!("expected transient admit, got {other:?}"),
        }
    }

    #[test]
    fn keep_warm_contention_queues_then_lru() {
        // need 20, free 0, only keep-warm residents → must queue, LRU keep-warm.
        let residents = vec![
            res("kw_new", 25.0, Tier::KeepWarm, 9),
            res("kw_old", 25.0, Tier::KeepWarm, 1),
        ];
        let plan = plan_admission(Some(20.0), 0.0, &residents);
        match plan {
            EvictionPlan::Queue {
                transient_first,
                keep_warm_lru,
            } => {
                assert!(transient_first.is_empty());
                assert_eq!(keep_warm_lru.len(), 1);
                assert_eq!(keep_warm_lru[0].model_id, "kw_old", "LRU keep-warm chosen");
            }
            other => panic!("expected Queue, got {other:?}"),
        }
    }

    #[test]
    fn transients_then_queue_keep_warm() {
        // need 40, free 0. Transient 10 + keep-warm 35 = 45 ≥ 40. Transient is
        // evicted immediately; the keep-warm is the deferred (queued) eviction.
        let residents = vec![
            res("t", 10.0, Tier::Transient, 1),
            res("kw", 35.0, Tier::KeepWarm, 2),
        ];
        let plan = plan_admission(Some(40.0), 0.0, &residents);
        match plan {
            EvictionPlan::Queue {
                transient_first,
                keep_warm_lru,
            } => {
                assert_eq!(transient_first.len(), 1);
                assert_eq!(transient_first[0].model_id, "t");
                assert_eq!(keep_warm_lru.len(), 1);
                assert_eq!(keep_warm_lru[0].model_id, "kw");
            }
            other => panic!("expected Queue with transient_first, got {other:?}"),
        }
    }

    #[test]
    fn pinned_chat_never_evicted_cannot_admit() {
        // need 50, free 0, only resident is the 40GB pinned chat. It is NEVER
        // evictable → even though evicting it would help, the verdict is
        // CannotAdmit (the pinned-only stall).
        let residents = vec![res("chat", 40.0, Tier::PinnedChat, 1)];
        let plan = plan_admission(Some(50.0), 0.0, &residents);
        assert_eq!(plan, EvictionPlan::CannotAdmit);
    }

    #[test]
    fn pinned_chat_excluded_but_keep_warm_still_used() {
        // need 30, free 0. Resident: pinned chat 40 (off-limits) + keep-warm 35.
        // The keep-warm alone covers it → Queue using only the keep-warm; the
        // pinned chat is NOT in any list.
        let residents = vec![
            res("chat", 40.0, Tier::PinnedChat, 1),
            res("kw", 35.0, Tier::KeepWarm, 2),
        ];
        let plan = plan_admission(Some(30.0), 0.0, &residents);
        match plan {
            EvictionPlan::Queue { keep_warm_lru, .. } => {
                assert_eq!(keep_warm_lru.len(), 1);
                assert_eq!(keep_warm_lru[0].model_id, "kw");
                assert!(
                    keep_warm_lru.iter().all(|t| t.tier != Tier::PinnedChat),
                    "pinned chat must never appear in an eviction list"
                );
            }
            other => panic!("expected Queue, got {other:?}"),
        }
    }

    #[test]
    fn unknown_footprint_with_residents_cannot_admit() {
        let residents = vec![res("kw", 10.0, Tier::KeepWarm, 1)];
        assert_eq!(plan_admission(None, 50.0, &residents), EvictionPlan::CannotAdmit);
    }

    #[test]
    fn unknown_footprint_empty_host_admits() {
        assert_eq!(
            plan_admission(None, 50.0, &[]),
            EvictionPlan::AdmitAfterEvicting(vec![])
        );
    }
}
