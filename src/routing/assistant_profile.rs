//! Assistant-profile → chat-role routing (S84 ASMT-11, chord side).
//!
//! The S84 assistant intake measured every candidate model on six dimensions and
//! ASMT-11's report logic (`terminus_rs::intake::assistant::reporting`) turned
//! those rows into a chat-role selection: the highest prompted-adherence model
//! that ALSO clears a latency/degradation guard. This module is the chord-side
//! consumer — it takes that measured selection and decides what the **Lumina chat
//! alias** should resolve to, WITHOUT ever sending the chat alias to a model the
//! guard rejected.
//!
//! ## Why a guard, structurally
//! Personality fit is necessary but not sufficient for the chat role: a model can
//! sound perfectly like Lumina yet degrade after a handful of turns or respond too
//! slowly to hold a live conversation. The selection we consume has ALREADY applied
//! the latency/degradation guard
//! ([`reporting::select_chat_role`]); here we additionally refuse to act on a
//! selection that isn't backed by a real, registry-known model, and we ALWAYS fall
//! back to the operator's current default alias mapping when no model cleared the
//! guard ("no-model-clears-guard → routing keeps the current default").
//!
//! ## No literals / secrets
//! The assistant scores come from the intake DB via the reporting layer (which
//! reads its URL from `crate::config::intake_database_url` — vault/config, no
//! literal). This module holds only the pure decision logic + a thin async fetch
//! wrapper; the chat alias name and any thresholds are passed in by the caller
//! from config, never hardcoded here.

use terminus_rs::error::ToolError;
use terminus_rs::intake::assistant::reporting::{
    self, AssistantReport, ChatRoleSelection, ModelKey, ReportConfig,
};

/// The chord chat-role routing decision for the Lumina alias.
#[derive(Debug, Clone, PartialEq)]
pub enum ChatRoleDecision {
    /// Route the chat alias to this measured-fit, guard-cleared backend model.
    /// `model_id` is byte-identical to the S83/registry key.
    Route {
        model_id: String,
        backend_tag: String,
        /// The dim-5 behavioral adherence that earned the slot (audit/log only).
        behavioral_mean: f64,
    },
    /// No candidate cleared the latency/degradation guard. Keep the operator's
    /// current default chat alias mapping. `reason` is logged + reported.
    KeepDefault { reason: String },
}

impl ChatRoleDecision {
    /// The backend model the chat alias should resolve to, if a measured pick was
    /// made. `None` ⇒ caller keeps its existing `model_aliases` mapping.
    pub fn routed_model(&self) -> Option<&str> {
        match self {
            ChatRoleDecision::Route { model_id, .. } => Some(model_id),
            ChatRoleDecision::KeepDefault { .. } => None,
        }
    }

    pub fn is_default(&self) -> bool {
        matches!(self, ChatRoleDecision::KeepDefault { .. })
    }
}

/// Decide the chat-role model from an assistant-intake [`ChatRoleSelection`].
///
/// Pure. Honours the guard verdict the reporting layer already computed:
///   - `selected = Some(key)` ⇒ [`ChatRoleDecision::Route`] to that model
///     (optionally constrained to `known_models` so chord never points the alias
///     at a model its registry can't start);
///   - `selected = None` ⇒ [`ChatRoleDecision::KeepDefault`] with the explicit
///     "no model cleared the guard" note.
///
/// `known_models` is the set of registry keys chord can actually serve. When the
/// measured pick isn't in that set, we DON'T invent a route — we keep the default
/// and say why (availability over a stale measurement). Pass an empty slice to
/// skip the registry check (e.g. when the caller has already validated).
pub fn decide_chat_role(
    selection: &ChatRoleSelection,
    known_models: &[String],
) -> ChatRoleDecision {
    match &selection.selected {
        None => ChatRoleDecision::KeepDefault {
            reason: selection
                .no_clearance_note()
                .unwrap_or_else(|| "no chat-role candidate cleared the guard".into()),
        },
        Some(ModelKey {
            model_id,
            backend_tag,
        }) => {
            if !known_models.is_empty() && !known_models.iter().any(|m| m == model_id) {
                return ChatRoleDecision::KeepDefault {
                    reason: format!(
                        "measured chat-role pick '{model_id}' is not a registry-known model — \
                         keeping current default until it is available"
                    ),
                };
            }
            // The behavioral_mean that earned the slot, pulled from the candidate
            // list for the audit trail.
            let behavioral_mean = selection
                .candidates
                .iter()
                .find(|c| &c.key.model_id == model_id && &c.key.backend_tag == backend_tag)
                .map(|c| c.behavioral_mean)
                .unwrap_or(0.0);
            ChatRoleDecision::Route {
                model_id: model_id.clone(),
                backend_tag: backend_tag.clone(),
                behavioral_mean,
            }
        }
    }
}

/// Derive the chat-role decision straight from a built [`AssistantReport`] —
/// convenience for callers that already have the report in hand.
pub fn decide_from_report(report: &AssistantReport, known_models: &[String]) -> ChatRoleDecision {
    decide_chat_role(&report.chat_role, known_models)
}

/// Live path: read the assistant scores, build the ASMT-11 report under `cfg`, and
/// return the chord chat-role decision. `known_models` constrains the pick to what
/// the registry can serve. All DB/secret access is inside `reporting::run_report`
/// (vault/config-sourced URL, no literal here).
///
/// On ANY error fetching/scoring (DB down, no run yet, …) the caller should keep
/// the current default — this returns the error so the caller can log it and fall
/// back, exactly like the per-backend routing's availability-over-strictness rule.
pub async fn fetch_chat_role_decision(
    run_id: Option<uuid::Uuid>,
    cfg: &ReportConfig,
    known_models: &[String],
) -> Result<ChatRoleDecision, ToolError> {
    let (report, _md) = reporting::run_report(run_id, cfg).await?;
    Ok(decide_from_report(&report, known_models))
}

// ─────────────────────────────────────────────────────────────────────────────
// SRV-06: chat-role PIN + residency integration
// ─────────────────────────────────────────────────────────────────────────────
//
// `decide_chat_role` (above) is the pure *which model* decision. This half closes
// the loop to the live host: it registers that model as the residency manager's
// PINNED, never-evicted chat alias — but only after (a) a serving-profile latency
// guard confirms it is responsive enough to be the *interactive* alias, and (b) it
// is brought resident successfully. The transfer is atomic: the new model is loaded
// and pinned BEFORE the old pin is released, so Lumina is never without a resident
// chat model. A failed load keeps the existing pin (a working chat alias is never
// surrendered for a broken one).

use crate::serving::profile::RoutingMap;
use crate::serving::residency::VramResidencyManager;
use crate::serving::ResidencyManager;
use terminus_rs::intake::serving::ModelId;

/// The result of applying a chat-role decision to the residency manager's pin.
#[derive(Debug, Clone, PartialEq)]
pub enum ChatPinOutcome {
    /// The chat alias is pinned to `model_id`. `cold_start_flagged` is true when a
    /// `keep_warm` model was allowed despite a slow cold-load — warm residency
    /// mitigates steady-state latency, but the first cold-start still applies, so
    /// the tradeoff is recorded.
    Pinned {
        model_id: String,
        cold_start_flagged: bool,
    },
    /// The existing pin was left unchanged. `reason` explains why (no measured
    /// switch, latency guard blocked, model unprofiled, or load failed).
    KeptCurrent { reason: String },
}

impl ChatPinOutcome {
    /// The model that ended up pinned, if the pin changed/confirmed to a model.
    pub fn pinned_model(&self) -> Option<&str> {
        match self {
            ChatPinOutcome::Pinned { model_id, .. } => Some(model_id),
            ChatPinOutcome::KeptCurrent { .. } => None,
        }
    }
}

/// Apply a [`ChatRoleDecision`] to the residency manager's chat pin, gated by the
/// serving-profile latency guard and the atomic-transfer rule.
///
/// `max_cold_load_s` is the interactive-latency budget (caller passes it from
/// [`crate::config::chat_pin_max_cold_load_s`] — never hardcoded here). Steps:
///   1. A `KeepDefault` decision changes nothing (no measured pick).
///   2. A `Route` pick with no serving-profile row keeps the current pin (we won't
///      pin a model we can't reason about / size).
///   3. Latency guard: a model whose `cold_load_s` exceeds the budget is rejected
///      as the interactive alias UNLESS it is `keep_warm` (then allowed + flagged).
///   4. Atomic transfer: bring the new model resident (the pinned old model is
///      never evicted to make room — if it doesn't fit, keep the old pin), then
///      pin it; [`set_pinned_chat_model`] releases the old pin in the same locked
///      update. A failed acquire keeps the old pin.
pub async fn apply_chat_pin(
    decision: &ChatRoleDecision,
    routing: &RoutingMap,
    residency: &VramResidencyManager,
    max_cold_load_s: f64,
) -> ChatPinOutcome {
    // (1) No measured pick → leave the operator's current pin untouched.
    let model_id = match decision.routed_model() {
        Some(m) => m.to_string(),
        None => {
            let reason = match decision {
                ChatRoleDecision::KeepDefault { reason } => reason.clone(),
                ChatRoleDecision::Route { .. } => unreachable!("routed_model is Some"),
            };
            return ChatPinOutcome::KeptCurrent { reason };
        }
    };

    let mid = ModelId::from(model_id.as_str());

    // (2) Unprofiled pick → keep current pin (don't pin a model we can't size/guard).
    let route = match routing.get(&mid) {
        Some(r) => r,
        None => {
            return ChatPinOutcome::KeptCurrent {
                reason: format!(
                    "chat-role pick '{model_id}' has no serving profile — keeping current pin"
                ),
            }
        }
    };

    // (3) Latency guard for the INTERACTIVE alias.
    let cold = route.profile.cold_load_s.unwrap_or(0.0);
    let mut cold_start_flagged = false;
    if cold > max_cold_load_s {
        if route.keep_warm() {
            // Held resident → steady-state latency mitigated; flag the first-load
            // tradeoff but allow it.
            cold_start_flagged = true;
        } else {
            // Cold every use AND slow → unresponsive as the live chat alias.
            return ChatPinOutcome::KeptCurrent {
                reason: format!(
                    "chat-role pick '{model_id}' cold-loads too slowly to be the interactive \
                     alias and is not keep-warm — keeping current pin"
                ),
            };
        }
    }

    // Idempotent: already the pinned model → confirm, no reload.
    if residency.pinned_chat_model().await.as_deref() == Some(model_id.as_str()) {
        return ChatPinOutcome::Pinned {
            model_id,
            cold_start_flagged,
        };
    }

    // (4) Atomic transfer: load the new model FIRST (the still-pinned old model is
    // never evicted to make room — SRV-05 keeps it sticky), then pin it.
    match residency.acquire_warm_slot(&mid, route.vram_gb()).await {
        Ok(_slot) => {
            residency.set_pinned_chat_model(Some(&model_id)).await;
            ChatPinOutcome::Pinned {
                model_id,
                cold_start_flagged,
            }
        }
        Err(_e) => ChatPinOutcome::KeptCurrent {
            reason: format!(
                "could not bring chat-role pick '{model_id}' resident — keeping current pin"
            ),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use terminus_rs::intake::assistant::reporting::{
        ChatRoleCandidate, ChatRoleSelection, GuardVerdict, ModelKey,
    };

    fn key(m: &str) -> ModelKey {
        ModelKey {
            model_id: m.into(),
            backend_tag: "gpu".into(),
        }
    }

    fn eligible(m: &str, adherence: f64) -> ChatRoleCandidate {
        ChatRoleCandidate {
            key: key(m),
            behavioral_mean: adherence,
            recall_ceiling_turns: Some(40.0),
            latency_ms: Some(1200.0),
            verdict: GuardVerdict::Eligible,
        }
    }

    fn excluded(m: &str, adherence: f64, reason: &str) -> ChatRoleCandidate {
        ChatRoleCandidate {
            key: key(m),
            behavioral_mean: adherence,
            recall_ceiling_turns: Some(3.0),
            latency_ms: Some(9000.0),
            verdict: GuardVerdict::Excluded {
                reason: reason.into(),
            },
        }
    }

    #[test]
    fn routes_to_guard_cleared_pick() {
        let sel = ChatRoleSelection {
            candidates: vec![eligible("qwen3:8b", 4.5)],
            selected: Some(key("qwen3:8b")),
        };
        let d = decide_chat_role(&sel, &["qwen3:8b".into()]);
        assert_eq!(d.routed_model(), Some("qwen3:8b"));
        match d {
            ChatRoleDecision::Route {
                behavioral_mean, ..
            } => assert!((behavioral_mean - 4.5).abs() < 1e-9),
            _ => panic!("expected Route"),
        }
    }

    #[test]
    fn keeps_default_when_nothing_clears_guard() {
        // Top personality model failed the guard → no selection.
        let sel = ChatRoleSelection {
            candidates: vec![excluded("slowmodel:70b", 5.0, "latency 9000ms > max 4000ms")],
            selected: None,
        };
        let d = decide_chat_role(&sel, &["slowmodel:70b".into()]);
        assert!(d.is_default());
        assert!(d.routed_model().is_none());
        match d {
            ChatRoleDecision::KeepDefault { reason } => {
                assert!(reason.contains("keeps the current default"));
            }
            _ => panic!("expected KeepDefault"),
        }
    }

    #[test]
    fn unknown_registry_pick_keeps_default() {
        // Measured pick that the registry can't serve → keep default, don't route
        // to a model chord can't start.
        let sel = ChatRoleSelection {
            candidates: vec![eligible("ghost:99b", 4.9)],
            selected: Some(key("ghost:99b")),
        };
        let d = decide_chat_role(&sel, &["qwen3:8b".into()]);
        assert!(d.is_default());
        match d {
            ChatRoleDecision::KeepDefault { reason } => {
                assert!(reason.contains("not a registry-known model"));
            }
            _ => panic!("expected KeepDefault"),
        }
    }

    #[test]
    fn empty_known_models_skips_registry_check() {
        let sel = ChatRoleSelection {
            candidates: vec![eligible("anything:1b", 3.0)],
            selected: Some(key("anything:1b")),
        };
        let d = decide_chat_role(&sel, &[]);
        assert_eq!(d.routed_model(), Some("anything:1b"));
    }
}
