//! DOCGEN-04: SLM router evaluation sweep — a test panel that scores candidate
//! routers on ROUTING QUALITY, not just raw model quality.
//!
//! This module answers: "did the router (DOCGEN-03's [`super::slm_router::SlmRouter`]
//! plus a given [`super::policy::RoutingPolicy`]) pick a destination that produced
//! good docs at acceptable cost/latency?" It runs a fixed, representative panel of
//! doc-gen requests ([`fixed_panel`]) through a candidate router, grades each
//! resulting output for doc quality, and combines that with decision
//! appropriateness / cost / latency into one [`DecisionScore`] per request and one
//! [`CandidateEvalSummary`] per candidate.
//!
//! ## Mocked integration (per the spec's APPROACH)
//! The GPU host is held by a permanent production coder-model serve (see
//! project memory), so this module NEVER makes a live inference call.
//! Everything above the [`super::slm_router::Executor`] trait (already mockable —
//! DOCGEN-03 built it that way) is exercised end-to-end against a
//! [`super::slm_router::Executor`] test double and a [`Grader`] test double. The
//! real-GPU integration is future work once a sweep-safe serving slot exists.
//!
//! ## The H4 lesson — sanity-check the grader before trusting the panel
//! A bad grader invalidates the whole sweep (a grader that can't tell a complete,
//! on-topic doc from a truncated, off-topic one just adds noise dressed up as a
//! score). [`sanity_check_grader`] is a REQUIRED pre-flight check: it asserts a
//! grader scores a known-good case meaningfully higher than a known-bad case
//! before any candidate score coming out of that grader is trusted. See
//! [`grader_sanity_check_passes_for_discriminating_grader`] (positive) and
//! [`grader_sanity_check_catches_a_grader_that_cannot_discriminate`] (negative)
//! in the tests below.

use super::policy::RoutingDestination;
use super::slm_router::RoutingDecision;

/// One fixed, representative doc-gen request in the evaluation panel, along
/// with the ground-truth "ideal" destination a well-behaved router should
/// reach for it. `expected_destination` is a panel-authoring judgment call
/// (not derived from any policy under test), so scoring a candidate never
/// circularly reuses the candidate's own policy as its own ground truth.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DocGenEvalRequest {
    pub id: &'static str,
    pub prompt: &'static str,
    pub estimated_tokens: usize,
    pub expected_destination: RoutingDestination,
}

/// The fixed panel: a small, representative spread across the three
/// destinations so every tier of routing decision gets exercised. Kept
/// small deliberately (per the spec's "fixed set" — a panel this is meant to
/// be run against every candidate, so it stays cheap even before the
/// mocked/real-GPU split is resolved).
pub fn fixed_panel() -> Vec<DocGenEvalRequest> {
    vec![
        DocGenEvalRequest {
            id: "simple-readme-blurb",
            prompt: "Summarize a one-function utility module in two sentences.",
            estimated_tokens: 400,
            expected_destination: RoutingDestination::LocalCheap,
        },
        DocGenEvalRequest {
            id: "module-level-docs",
            prompt: "Document a mid-sized module's public API with examples.",
            estimated_tokens: 8_000,
            expected_destination: RoutingDestination::LocalHighContext,
        },
        DocGenEvalRequest {
            id: "whole-repo-architecture",
            prompt: "Write an architecture overview spanning the whole repository's history.",
            estimated_tokens: 40_000,
            expected_destination: RoutingDestination::CloudFrontierFree,
        },
    ]
}

/// Ordinal "tier" of a destination, used to measure how far a routing
/// decision landed from the expected one. Cheapest/smallest first.
fn destination_tier(destination: RoutingDestination) -> i32 {
    match destination {
        RoutingDestination::LocalCheap => 0,
        RoutingDestination::LocalHighContext => 1,
        RoutingDestination::CloudFrontierFree => 2,
    }
}

/// Relative cost weight of actually executing on `destination` — 0.0
/// (free-ish, local cheap) to 1.0 (most expensive, cloud). Deliberately
/// coarse: this sweep measures ROUTING quality, not exact per-token billing.
fn destination_cost_weight(destination: RoutingDestination) -> f64 {
    match destination {
        RoutingDestination::LocalCheap => 0.05,
        RoutingDestination::LocalHighContext => 0.30,
        RoutingDestination::CloudFrontierFree => 1.00,
    }
}

/// A grader: scores generated `output` for doc quality on request `request`,
/// in `[0.0, 1.0]`. Implementations are swappable — production would call a
/// real judge model via the same router this module evaluates; tests use a
/// deterministic heuristic or a fixture double.
pub trait Grader: Send + Sync {
    fn grade(&self, request: &DocGenEvalRequest, output: &str) -> f64;
}

/// A simple, deterministic completeness/relevance heuristic grader — good
/// enough to exercise this module's scoring + persistence end-to-end without
/// a live judge-model call. Rewards output that (a) is reasonably long for
/// the request and (b) ends with an explicit completion marker; an output
/// that was truncated (e.g. because it landed on a destination whose context
/// window couldn't fit the request) neither, so it scores low.
pub struct HeuristicCompletenessGrader;

impl Grader for HeuristicCompletenessGrader {
    fn grade(&self, _request: &DocGenEvalRequest, output: &str) -> f64 {
        let trimmed = output.trim_end();
        let complete = trimmed.ends_with("[COMPLETE]");
        let length_score = (trimmed.len() as f64 / 400.0).min(1.0);
        if complete {
            (0.55 + 0.45 * length_score).min(1.0)
        } else {
            // Truncated/incomplete output: capped well below any complete
            // output regardless of length, so padding can't fake quality.
            (0.35 * length_score).min(0.45)
        }
    }
}

/// Error raised when a grader fails the required pre-flight sanity check —
/// its scores don't discriminate a known-good case from a known-bad one by
/// at least `min_margin`, so the whole sweep would be measuring noise, not
/// routing quality (the H4 lesson).
#[derive(Debug, Clone, thiserror::Error, PartialEq)]
#[error(
    "grader sanity check failed: known-good scored {good_score:.3}, known-bad scored \
     {bad_score:.3} (margin {actual_margin:.3} < required {min_margin:.3}) — this grader \
     cannot be trusted to score the panel"
)]
pub struct GraderSanityError {
    pub good_score: f64,
    pub bad_score: f64,
    pub actual_margin: f64,
    pub min_margin: f64,
}

/// REQUIRED pre-flight check (H4 lesson): assert `grader` scores
/// `known_good_output` at least `min_margin` higher than `known_bad_output`
/// for `request`. Call this once per grader before trusting ANY score it
/// produces for the panel. Returns `Ok(())` only when the grader
/// discriminates meaningfully; a constant/always-approves grader fails this
/// (see the negative test).
pub fn sanity_check_grader(
    grader: &dyn Grader,
    request: &DocGenEvalRequest,
    known_good_output: &str,
    known_bad_output: &str,
    min_margin: f64,
) -> Result<(), GraderSanityError> {
    let good_score = grader.grade(request, known_good_output);
    let bad_score = grader.grade(request, known_bad_output);
    let actual_margin = good_score - bad_score;
    if actual_margin < min_margin {
        return Err(GraderSanityError {
            good_score,
            bad_score,
            actual_margin,
            min_margin,
        });
    }
    Ok(())
}

/// Per-request routing-quality score: decision appropriateness (did this
/// request land on the destination the panel expects?), doc quality (from
/// the grader), cost, and latency, combined into one `composite_score`.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionScore {
    pub request_id: &'static str,
    pub destination: RoutingDestination,
    pub expected_destination: RoutingDestination,
    /// `1.0` exact tier match, `0.5` one tier off, `0.0` two tiers off.
    pub appropriateness_score: f64,
    /// `[0.0, 1.0]` from the grader.
    pub doc_quality_score: f64,
    /// `[0.0, 1.0]`; `1.0` cheapest, penalizes over-provisioned destinations.
    pub cost_score: f64,
    /// `[0.0, 1.0]`; `1.0` fastest of the three destination tiers.
    pub latency_score: f64,
    /// Weighted combination of the four dimensions above.
    pub composite_score: f64,
    /// Set when this decision trips a known anti-pattern (see the spec's
    /// EDGE CASES): always-cloud (cost) or over-context-routed-to-local
    /// (quality). `None` for an unremarkable decision.
    pub flag_reason: Option<String>,
}

/// Weights for the composite score. Doc quality and appropriateness matter
/// most (a router that's fast and cheap but wrong or produces bad docs is
/// not a good router); cost and latency are real but secondary.
const WEIGHT_APPROPRIATENESS: f64 = 0.35;
const WEIGHT_DOC_QUALITY: f64 = 0.35;
const WEIGHT_COST: f64 = 0.20;
const WEIGHT_LATENCY: f64 = 0.10;

/// Latency, in ms, assumed per destination tier for this sweep's scoring
/// (mocked — no live GPU call; see module docs). Local cheap is fastest,
/// cloud slowest, mirroring the real system's expected shape.
fn destination_latency_ms(destination: RoutingDestination) -> u64 {
    match destination {
        RoutingDestination::LocalCheap => 400,
        RoutingDestination::LocalHighContext => 1_500,
        RoutingDestination::CloudFrontierFree => 4_000,
    }
}

fn latency_score(latency_ms: u64) -> f64 {
    // Normalize against the slowest tier (cloud, 4000ms) so the score stays
    // in [0.0, 1.0] for every destination this sweep knows about.
    let slowest = destination_latency_ms(RoutingDestination::CloudFrontierFree) as f64;
    (1.0 - (latency_ms as f64 / slowest)).clamp(0.0, 1.0)
}

/// Score one routing decision against the panel request it was made for and
/// the grader's doc-quality verdict on the resulting output.
///
/// Flags the two anti-patterns called out in the spec's EDGE CASES:
/// - the decision landed on cloud while the panel expected a cheaper local
///   destination (a symptom of "router always picks cloud, ignores local");
/// - the decision landed on a local destination for a request the panel
///   expects to need cloud (a symptom of "over-context routed to local,
///   output truncated/poor" — reflected here via a low `doc_quality_score`
///   too, since a truncated/poor output should already grade low).
pub fn score_decision(
    request: &DocGenEvalRequest,
    decision: &RoutingDecision,
    doc_quality_score: f64,
) -> DecisionScore {
    let actual_tier = destination_tier(decision.destination);
    let expected_tier = destination_tier(request.expected_destination);
    let tier_distance = (actual_tier - expected_tier).abs();
    let appropriateness_score = (1.0 - tier_distance as f64 * 0.5).max(0.0);

    let cost_score = 1.0 - destination_cost_weight(decision.destination);
    let latency_ms = destination_latency_ms(decision.destination);
    let latency_score_v = latency_score(latency_ms);

    let composite_score = WEIGHT_APPROPRIATENESS * appropriateness_score
        + WEIGHT_DOC_QUALITY * doc_quality_score
        + WEIGHT_COST * cost_score
        + WEIGHT_LATENCY * latency_score_v;

    let flag_reason = if actual_tier > expected_tier && decision.destination == RoutingDestination::CloudFrontierFree {
        Some(format!(
            "picked cloud for request '{}' which the panel expects on a cheaper local \
             destination — possible always-cloud router (scored down on cost)",
            request.id
        ))
    } else if actual_tier < expected_tier && request.expected_destination == RoutingDestination::CloudFrontierFree {
        Some(format!(
            "picked a local destination for over-context request '{}' the panel expects on \
             cloud — possible truncated/poor output (scored down on quality)",
            request.id
        ))
    } else {
        None
    };

    DecisionScore {
        request_id: request.id,
        destination: decision.destination,
        expected_destination: request.expected_destination,
        appropriateness_score,
        doc_quality_score,
        cost_score,
        latency_score: latency_score_v,
        composite_score,
        flag_reason,
    }
}

/// Aggregate result for one candidate router evaluated over the whole panel.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateEvalSummary {
    pub candidate_name: String,
    pub scores: Vec<DecisionScore>,
    pub mean_composite_score: f64,
    pub flagged_count: usize,
}

/// Summarize per-decision scores for one candidate into a
/// [`CandidateEvalSummary`]. `scores` must be non-empty (the panel is fixed
/// and never empty — see [`fixed_panel`]); an empty slice yields a summary
/// with `mean_composite_score = 0.0` rather than panicking, since a caller
/// building this incrementally may legitimately pass an empty vec before any
/// requests have been scored.
pub fn summarize_candidate(candidate_name: impl Into<String>, scores: Vec<DecisionScore>) -> CandidateEvalSummary {
    let flagged_count = scores.iter().filter(|s| s.flag_reason.is_some()).count();
    let mean_composite_score = if scores.is_empty() {
        0.0
    } else {
        scores.iter().map(|s| s.composite_score).sum::<f64>() / scores.len() as f64
    };
    CandidateEvalSummary {
        candidate_name: candidate_name.into(),
        scores,
        mean_composite_score,
        flagged_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::router::slm_router::RoutingDecision;

    fn panel_request(id: &'static str, expected: RoutingDestination) -> DocGenEvalRequest {
        DocGenEvalRequest {
            id,
            prompt: "x",
            estimated_tokens: 100,
            expected_destination: expected,
        }
    }

    fn decision(destination: RoutingDestination) -> RoutingDecision {
        RoutingDecision {
            destination,
            model: "test-model".into(),
            reason: "test".into(),
            fallback_from: None,
        }
    }

    // ── fixed_panel: representative, non-empty, spans all three tiers ──────

    #[test]
    fn fixed_panel_spans_all_three_destination_tiers() {
        let panel = fixed_panel();
        assert!(!panel.is_empty());
        assert!(panel.iter().any(|r| r.expected_destination == RoutingDestination::LocalCheap));
        assert!(panel.iter().any(|r| r.expected_destination == RoutingDestination::LocalHighContext));
        assert!(panel.iter().any(|r| r.expected_destination == RoutingDestination::CloudFrontierFree));
    }

    // ── grader sanity check (H4 lesson) ─────────────────────────────────────

    #[test]
    fn grader_sanity_check_passes_for_discriminating_grader() {
        let grader = HeuristicCompletenessGrader;
        let request = panel_request("known-good-case", RoutingDestination::LocalCheap);
        let good = "A thorough, on-topic summary of the module's public API and usage. [COMPLETE]";
        let bad = "A thorough, on-topic summary of the module's public API and usa";
        let result = sanity_check_grader(&grader, &request, good, bad, 0.2);
        assert!(result.is_ok(), "discriminating grader must pass sanity check: {result:?}");
    }

    #[test]
    fn grader_sanity_check_catches_a_grader_that_cannot_discriminate() {
        // Negative test (H4 lesson): a grader that can't tell good from bad
        // must be CAUGHT by the sanity check, not silently trusted.
        struct AlwaysApprovesGrader;
        impl Grader for AlwaysApprovesGrader {
            fn grade(&self, _request: &DocGenEvalRequest, _output: &str) -> f64 {
                0.9
            }
        }
        let grader = AlwaysApprovesGrader;
        let request = panel_request("known-good-case", RoutingDestination::LocalCheap);
        let good = "A thorough, on-topic summary of the module's public API. [COMPLETE]";
        let bad = "asdf asdf incomplete garbage nonsense";
        let result = sanity_check_grader(&grader, &request, good, bad, 0.2);
        assert!(
            matches!(result, Err(GraderSanityError { .. })),
            "a non-discriminating grader must fail the sanity check, invalidating the sweep: {result:?}"
        );
    }

    #[test]
    fn heuristic_grader_scores_truncated_output_below_complete_output() {
        let grader = HeuristicCompletenessGrader;
        let request = panel_request("r", RoutingDestination::LocalHighContext);
        let complete = "A full, detailed treatment of the module's public interface. [COMPLETE]";
        let truncated = "A full, detailed treatment of the module's pub";
        assert!(grader.grade(&request, complete) > grader.grade(&request, truncated));
    }

    // ── score_decision: known routing decision → expected score ────────────

    #[test]
    fn exact_match_decision_scores_full_appropriateness() {
        let request = panel_request("simple", RoutingDestination::LocalCheap);
        let d = decision(RoutingDestination::LocalCheap);
        let score = score_decision(&request, &d, 0.9);
        assert_eq!(score.appropriateness_score, 1.0);
        assert!(score.flag_reason.is_none());
        // Composite is a known, deterministic weighted sum for these inputs.
        let expected_cost = 1.0 - destination_cost_weight(RoutingDestination::LocalCheap);
        let expected_latency = latency_score(destination_latency_ms(RoutingDestination::LocalCheap));
        let expected_composite = WEIGHT_APPROPRIATENESS * 1.0
            + WEIGHT_DOC_QUALITY * 0.9
            + WEIGHT_COST * expected_cost
            + WEIGHT_LATENCY * expected_latency;
        assert!((score.composite_score - expected_composite).abs() < 1e-9);
    }

    #[test]
    fn one_tier_off_decision_scores_half_appropriateness() {
        let request = panel_request("mid", RoutingDestination::LocalCheap);
        let d = decision(RoutingDestination::LocalHighContext);
        let score = score_decision(&request, &d, 0.8);
        assert_eq!(score.appropriateness_score, 0.5);
    }

    #[test]
    fn two_tiers_off_decision_scores_zero_appropriateness() {
        let request = panel_request("mid", RoutingDestination::LocalCheap);
        let d = decision(RoutingDestination::CloudFrontierFree);
        let score = score_decision(&request, &d, 0.8);
        assert_eq!(score.appropriateness_score, 0.0);
    }

    // ── edge case: always-cloud router scored down on cost + flagged ───────

    #[test]
    fn always_cloud_for_a_simple_request_is_flagged_and_scored_down_on_cost() {
        let request = panel_request("simple-readme-blurb", RoutingDestination::LocalCheap);
        let d = decision(RoutingDestination::CloudFrontierFree);
        let score = score_decision(&request, &d, 0.95); // even with great doc quality...
        assert!(score.flag_reason.is_some(), "always-cloud-for-simple must be flagged");
        assert!(
            score.flag_reason.as_ref().unwrap().contains("always-cloud"),
            "flag reason should name the always-cloud anti-pattern: {:?}",
            score.flag_reason
        );
        assert!(score.cost_score < 0.5, "cloud for a simple request must score down on cost");
    }

    // ── edge case: over-context routed to local scored down on quality ─────

    #[test]
    fn over_context_routed_to_local_is_flagged_and_scored_down_on_quality() {
        let request = panel_request("whole-repo-architecture", RoutingDestination::CloudFrontierFree);
        let d = decision(RoutingDestination::LocalCheap);
        // Simulate the truncated/poor output a real over-context-on-local
        // mistake would actually produce: the grader scores it low.
        let grader = HeuristicCompletenessGrader;
        let truncated_output = "A partial architecture note that was cut off mid-sen";
        let doc_quality_score = grader.grade(&request, truncated_output);
        let score = score_decision(&request, &d, doc_quality_score);
        assert!(score.flag_reason.is_some(), "over-context-to-local must be flagged");
        assert!(
            score.flag_reason.as_ref().unwrap().contains("over-context"),
            "flag reason should name the over-context anti-pattern: {:?}",
            score.flag_reason
        );
        assert!(score.doc_quality_score < 0.5, "truncated output must grade low on quality");
    }

    // ── summarize_candidate ──────────────────────────────────────────────────

    #[test]
    fn summarize_candidate_computes_mean_and_flag_count() {
        let request_a = panel_request("a", RoutingDestination::LocalCheap);
        let request_b = panel_request("b", RoutingDestination::LocalCheap);
        let good = score_decision(&request_a, &decision(RoutingDestination::LocalCheap), 0.9);
        let flagged = score_decision(&request_b, &decision(RoutingDestination::CloudFrontierFree), 0.9);
        let summary = summarize_candidate("candidate-x", vec![good.clone(), flagged.clone()]);
        assert_eq!(summary.candidate_name, "candidate-x");
        assert_eq!(summary.flagged_count, 1);
        let expected_mean = (good.composite_score + flagged.composite_score) / 2.0;
        assert!((summary.mean_composite_score - expected_mean).abs() < 1e-9);
    }

    #[test]
    fn summarize_candidate_handles_empty_scores_without_panicking() {
        let summary = summarize_candidate("empty-candidate", vec![]);
        assert_eq!(summary.mean_composite_score, 0.0);
        assert_eq!(summary.flagged_count, 0);
    }

    // ── mocked end-to-end: one candidate scored via the real SlmRouter ─────

    #[tokio::test]
    async fn one_mocked_candidate_scored_end_to_end_via_slm_router() {
        use crate::models::backends::{Backend, BackendKind, Hardware};
        use crate::router::policy::RoutingPolicy;
        use crate::router::slm_router::{Executor, SlmRouter};
        use async_trait::async_trait;
        use std::collections::HashMap;

        // A mocked executor standing in for a live GPU/cloud call — returns a
        // canned, "complete" doc for every destination. No network, no GPU.
        struct MockedDocGenExecutor;
        #[async_trait]
        impl Executor for MockedDocGenExecutor {
            async fn execute(&self, _backend: &Backend, _model: &str, _prompt: &str) -> Result<String, String> {
                Ok("A complete, on-topic generated doc for this request. [COMPLETE]".to_string())
            }
        }

        fn backend(name: &str) -> Backend {
            Backend {
                name: name.into(),
                url: "http://127.0.0.1:0".into(),
                hardware: Hardware::Cpu,
                kind: BackendKind::Ollama,
                unit: None,
                always_on: true,
                idle_stop_secs: 0,
                launch: None,
                api_key_env: None,
            }
        }

        let mut backends = HashMap::new();
        backends.insert("ollama".to_string(), backend("ollama"));
        backends.insert(
            "openrouter".to_string(),
            Backend {
                kind: BackendKind::OpenRouter,
                // Host must match the policy's cloud_egress_allowlist below
                // ("openrouter.ai") — the mocked executor never actually
                // dials this URL, but the egress gate checks the hostname
                // before the mock is ever invoked.
                url: "https://openrouter.ai".into(),
                ..backend("openrouter")
            },
        );

        let policy = RoutingPolicy {
            context_threshold_tokens: 1_000,
            local_high_ctx_max_tokens: 20_000,
            local_high_ctx_model: "local-high-ctx".into(),
            local_cheap_model: "local-cheap".into(),
            cloud_frontier_model: "cloud-frontier".into(),
            allow_cloud: true,
            cloud_egress_allowlist: vec!["openrouter.ai".into()],
        };
        let router = SlmRouter::with_backends(policy, backends);
        let grader = HeuristicCompletenessGrader;
        let executor = MockedDocGenExecutor;

        let mut scores = Vec::new();
        for request in fixed_panel() {
            let routing_request = crate::router::policy::RoutingRequest {
                prompt: request.prompt.to_string(),
                estimated_tokens: request.estimated_tokens,
            };
            let (output, decision) = router
                .route_and_execute(&routing_request, &executor)
                .await
                .expect("mocked executor always succeeds");
            let doc_quality_score = grader.grade(&request, &output);
            scores.push(score_decision(&request, &decision, doc_quality_score));
        }

        let summary = summarize_candidate("mocked-candidate", scores);
        assert_eq!(summary.scores.len(), fixed_panel().len());
        // Every panel request routes exactly per its expected destination
        // under this policy/panel pairing, and the executor always returns a
        // complete doc, so nothing should be flagged and quality is high.
        assert_eq!(summary.flagged_count, 0, "unexpected flags: {:?}", summary.scores);
        assert!(summary.mean_composite_score > 0.7, "mean score too low: {summary:?}");
    }
}
