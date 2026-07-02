//! YARN-03: extended-context validation harness.
//!
//! YARN-01 taught the launcher to emit YaRN flags for a
//! [`crate::serving::profile::RopeScaling`] block, but only when
//! `validated == true` — and YARN-02's GGUF ingestion always prefills
//! `validated: false`. Nothing flips that flag to `true` on say-so: research
//! could not confirm YaRN holds cleanly on gfx1151, so a candidate config only
//! earns trust after THIS harness runs and confirms it end to end.
//!
//! ## What this harness answers
//! Given a candidate [`RopeScaling`] config, does serving at its extended
//! `target_ctx` actually work — both "does it serve at all" (no wedge/hang/
//! crash) AND "does it actually use the context" (recall/coherence hold up
//! relative to the model's own native-context baseline, not just a fixed
//! score)? The most important failure mode this distinguishes is a model that
//! *serves fine* at extended context but silently drops facts planted deep in
//! the prompt — a serving success that is nonetheless a validation failure.
//!
//! ## Injectable seam (why this is unit-testable without real infra)
//! Actually launching `llama-server` with YaRN flags against a real gfx1151
//! GPU, and running real recall/coherence probes against it, is a SEPARATE
//! gated human-action item (YARN-04). This harness only builds the decision
//! machinery: [`YarnLauncher`] and [`ContextProber`] are traits (mirroring the
//! [`crate::serving::launcher::RuntimeSpawner`] /
//! [`crate::serving::launcher::HealthChecker`] seam that module already uses)
//! so production can wire real implementations later while tests inject
//! scripted fakes today. The actual pass/fail decision ([`evaluate`]) is a
//! pure function over already-collected [`LaunchReport`] + [`ProbeResult`]
//! values — deterministic, no I/O, fully exercised by unit tests.
//!
//! ## Never a false positive
//! [`evaluate`] can only recommend `validated: true` when ALL of the
//! following hold: the [`LaunchReport`] served stably, at least one
//! extended-context probe was actually collected, the deepest probe reached
//! the full `target_ctx`, the native baseline score is a FINITE number, the
//! native baseline itself is above [`WEAK_BASELINE_THRESHOLD`] (an absolute
//! floor — a near-zero baseline makes the *relative* collapse check below
//! carry no signal, since its threshold shrinks to near-zero right along with
//! it), and no probe's combined score collapsed (or was itself non-finite —
//! an invalid score is treated as a collapse, never silently ignored)
//! relative to that baseline. There is no code path that flips `validated`
//! without that full evidence trail attached to the returned
//! [`ValidationEvidence`].

use serde::{Deserialize, Serialize};

use crate::serving::profile::RopeScaling;

/// Fractions of `target_ctx` at which extended-context recall/coherence is
/// probed: shallow (30%), mid (60%), and the full extended range (100%). Per
/// YARN-03's design, probing must go up to and INCLUDING the extended range,
/// not stop short of it.
pub const PROBE_DEPTH_FRACTIONS: [f64; 3] = [0.3, 0.6, 1.0];

/// How much of the native-context [`ProbeResult::combined_score`] an
/// extended-context probe must retain to count as "holding" rather than
/// "collapsing". Deliberately conservative (not 100%): some falloff at extreme
/// depth is expected even for a healthy config; the harness is looking for a
/// genuine collapse, not measurement noise.
pub const COLLAPSE_RATIO: f64 = 0.85;

/// Below this native-context baseline score, the model's OWN native behavior
/// is already weak — a low (or zero) extended-context score in that case is
/// not evidence against YaRN, it is inherited from the baseline, AND the
/// relative collapse check below carries no signal against a near-zero
/// baseline (its threshold shrinks to near-zero right along with it, so every
/// extended probe would trivially "hold"). So this is an ABSOLUTE FLOOR, not
/// just an informational flag: [`evaluate`] refuses to validate at all when
/// the native baseline is this weak — see [`ValidationEvidence::native_baseline_weak`].
pub const WEAK_BASELINE_THRESHOLD: f64 = 0.5;

/// Compute the probe depths (in tokens) for a given extended `target_ctx`,
/// per [`PROBE_DEPTH_FRACTIONS`]. Pure arithmetic — no I/O.
pub fn probe_depths_for(target_ctx: u32) -> Vec<u32> {
    PROBE_DEPTH_FRACTIONS
        .iter()
        .map(|f| ((*f) * target_ctx as f64).round() as u32)
        .collect()
}

/// Score a recall transcript: a planted-fact hit/miss vector at some depth. A
/// "transcript" here is deliberately abstract — a `Vec<bool>` of whether each
/// planted fact came back correctly — so the scorer has no dependency on how
/// facts were planted or how the model's completion was graded (that grading
/// happens in the injected [`ContextProber`], not here). Empty input scores
/// `1.0` (vacuously — no facts planted, nothing missed) rather than dividing
/// by zero.
pub fn recall_score(hits: &[bool]) -> f64 {
    if hits.is_empty() {
        return 1.0;
    }
    let correct = hits.iter().filter(|h| **h).count();
    correct as f64 / hits.len() as f64
}

/// One recall/coherence probe result at a given context depth.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProbeResult {
    /// Context depth this probe was run at, in tokens.
    pub depth_tokens: u32,
    /// Fraction of planted facts correctly recalled, `[0.0, 1.0]`. Typically
    /// produced by [`recall_score`] over a hit/miss transcript.
    pub recall_score: f64,
    /// Lightweight coherence score, `[0.0, 1.0]` — does the completion stay
    /// on-topic and well-formed. This is NOT a full quality judgement, just
    /// enough to catch a model that degenerates (repeats, garbles) under
    /// extended context even while still nominally recalling facts.
    pub coherence_score: f64,
}

impl ProbeResult {
    /// Combined score used for collapse comparisons: the MINIMUM of recall and
    /// coherence, not their average. An average would let a fluent-but-wrong
    /// completion (high coherence, low recall) mask a genuine recall
    /// collapse — exactly the failure mode YARN-03 cares most about
    /// distinguishing from a serving failure.
    pub fn combined_score(&self) -> f64 {
        // `f64::min` deliberately ignores NaN (it returns the OTHER operand
        // when one side is NaN) — exactly wrong for this use: a NaN input
        // here means the grading round-trip on that side produced garbage,
        // and that must propagate as an invalid (non-finite) combined score
        // rather than be silently masked by whichever side happened to be a
        // real number. Every non-finite consumer downstream (the `evaluate`
        // collapse/baseline guards) relies on this propagation.
        if self.recall_score.is_nan() || self.coherence_score.is_nan() {
            return f64::NAN;
        }
        self.recall_score.min(self.coherence_score)
    }
}

/// What the injected [`YarnLauncher`] reports about attempting to serve a
/// model with a [`RopeScaling`] block at `rope.target_ctx`. Deliberately
/// narrow: no PID, no raw process output, nothing infra-specific — this is
/// the interface the DECISION needs, not a process handle. Production
/// (YARN-04) fills this in from a real launch attempt; tests construct it by
/// hand.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LaunchReport {
    /// Did the runtime come up and stay up long enough to run probes against —
    /// i.e. no wedge, hang, or crash? `false` means the whole extended-ctx
    /// attempt is a SERVING failure, independent of what any probe would have
    /// shown (probes are never run against an unstable launch).
    pub served_stably: bool,
    /// A short, already-genericized reason when `served_stably` is `false`
    /// (e.g. `"wedge: no response within health-check window"`,
    /// `"oom: kv cache exceeds free vram"`, `"crash: process exited 139"`).
    /// `None` when it served fine.
    pub failure_reason: Option<String>,
    /// VRAM used once resident at the extended context, if the launcher could
    /// read it. `None` when unreadable, or the launch never got that far.
    pub vram_gb_at_extended_ctx: Option<f64>,
    /// Which backend actually served the launch (e.g. `"rocm"`, `"vulkan"`).
    /// Purely descriptive: this harness never CHOOSES a backend, it records
    /// what the injected launcher used so a later read can spot
    /// backend-dependence (research suggests ROCm may beat Vulkan at very
    /// large contexts on gfx1151 — this harness doesn't decide that, it just
    /// preserves the fact).
    pub backend_used: String,
    /// If the launcher had to size DOWN from the requested `target_ctx` (e.g.
    /// a KV-cache OOM at the full extended size), the largest context that
    /// actually fit. `None` when the requested `target_ctx` fit as-is, or the
    /// launch failed before any sizing was attempted. This is informative on
    /// its own — a model that OOMs at 128K but fits at 64K is not a blanket
    /// failure, it just has a smaller proven ceiling.
    pub max_ctx_that_fit: Option<u32>,
}

/// Launches a model with a given [`RopeScaling`] config and reports what
/// happened. Production (YARN-04, gated, NOT built here) wires this to a real
/// `llama-server` launch on gfx1151; tests inject a scripted fake. Mirrors the
/// [`crate::serving::launcher::RuntimeSpawner`] seam already used elsewhere in
/// this codebase to keep tests off real processes.
pub trait YarnLauncher: Send + Sync {
    /// Attempt to bring `model_id` up with `rope` applied at `rope.target_ctx`.
    fn launch(&self, model_id: &str, rope: &RopeScaling) -> LaunchReport;
}

/// Runs a single recall/coherence probe at `depth_tokens` against an already
/// (successfully) launched model. Production (YARN-04) wires this to a real
/// prompt/grade round-trip; tests inject a scripted fake keyed by depth.
pub trait ContextProber: Send + Sync {
    /// Probe recall/coherence at `depth_tokens` into the current context.
    fn probe(&self, depth_tokens: u32) -> ProbeResult;
}

/// The harness's verdict on a candidate [`RopeScaling`] config, with the full
/// evidence trail that produced it. Nothing downstream should ever treat
/// `validated: true` as trustworthy without also inspecting the rest of this
/// struct — that's the point of shipping it as one record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationEvidence {
    /// The harness's recommendation: does this config earn
    /// `RopeScaling::validated = true`? Only `true` when `served_stably` and
    /// every extended-context probe held relative to the native baseline.
    pub validated: bool,
    /// Did the model serve stably at the extended context (no wedge/hang/
    /// crash)? Distinguished from `validated` on purpose — a model can serve
    /// stably (`true`) and still fail validation because recall collapsed.
    pub served_stably: bool,
    /// The context depth (tokens) at which recall/coherence first collapsed
    /// relative to native baseline, if any. `None` when nothing collapsed
    /// (including when the launch never served, so no probes ran).
    pub failure_depth_tokens: Option<u32>,
    /// Human-readable reason validation failed, if it did. Covers both serving
    /// failures (copied from [`LaunchReport::failure_reason`]) and collapse
    /// failures (synthesized here with the collapse depth).
    pub failure_reason: Option<String>,
    /// VRAM used at the extended context, if the launcher reported it.
    pub vram_gb_at_extended_ctx: Option<f64>,
    /// Which backend served the launch (see [`LaunchReport::backend_used`]).
    pub backend_used: String,
    /// The largest context that actually fit, if the launcher had to size
    /// down. See [`LaunchReport::max_ctx_that_fit`].
    pub max_ctx_that_fit: Option<u32>,
    /// The model's native-context baseline combined score, for reference.
    pub native_baseline_score: f64,
    /// `true` when the native baseline itself was weak (below
    /// [`WEAK_BASELINE_THRESHOLD`]) — a low extended score in that case is
    /// inherited from the model's own native behavior, not caused by YaRN.
    /// This IS a gate, not merely informational: [`evaluate`] forces
    /// `validated = false` whenever this is `true` (see
    /// [`WEAK_BASELINE_THRESHOLD`]'s doc for why the relative collapse check
    /// alone can't be trusted against a near-zero baseline).
    pub native_baseline_weak: bool,
    /// Every extended-context probe's combined score, keyed by depth, in the
    /// order probed — the recall/coherence curve a human can read directly.
    pub extended_scores: Vec<(u32, f64)>,
}

/// The pure decision core: given a native-context baseline, the extended-
/// context probes actually collected (empty if the launch never served), the
/// launch report, and the `target_ctx` the config was meant to be validated
/// at, decide whether this config validates. Deterministic — same inputs
/// always produce the same [`ValidationEvidence`] — which is what makes this
/// fully unit-testable without any real launch or probe.
///
/// INVARIANT (enforced HERE, not merely assumed from how a caller happens to
/// invoke this): `validated` can only be `true` when `extended_probes` is
/// non-empty AND its deepest entry reaches `target_ctx` — i.e. there is an
/// actual evidence trail covering the full extended range. A caller (present
/// or future, e.g. the real YARN-04 wiring) that supplies zero probes, or
/// only shallow ones that never exercise the extended range, gets
/// `validated: false` with an explicit reason — never a silent pass on no
/// evidence.
pub fn evaluate(
    native_baseline: ProbeResult,
    extended_probes: &[ProbeResult],
    launch: &LaunchReport,
    target_ctx: u32,
) -> ValidationEvidence {
    let native_baseline_score = native_baseline.combined_score();
    // `native_baseline_score < WEAK_BASELINE_THRESHOLD` is `false` for a NaN
    // score (NaN compares false against everything), so `is_finite()` must be
    // checked separately rather than folded into the weak-baseline predicate
    // — otherwise a NaN baseline would silently skip BOTH the weak-baseline
    // floor and the relative collapse check below. A negative finite score
    // needs no special handling: it naturally compares `< WEAK_BASELINE_THRESHOLD`
    // (a positive constant), so it already trips the weak-baseline floor.
    let native_baseline_weak =
        native_baseline_score.is_finite() && native_baseline_score < WEAK_BASELINE_THRESHOLD;
    let extended_scores: Vec<(u32, f64)> = extended_probes
        .iter()
        .map(|p| (p.depth_tokens, p.combined_score()))
        .collect();

    if !launch.served_stably {
        // Serving failure (wedge/hang/crash/OOM before probing was possible).
        // This is recorded as a validation failure, never a harness panic —
        // and never conflated with a recall collapse (no probes ran).
        return ValidationEvidence {
            validated: false,
            served_stably: false,
            failure_depth_tokens: None,
            failure_reason: Some(
                launch
                    .failure_reason
                    .clone()
                    .unwrap_or_else(|| "launch did not serve stably".to_string()),
            ),
            vram_gb_at_extended_ctx: launch.vram_gb_at_extended_ctx,
            backend_used: launch.backend_used.clone(),
            max_ctx_that_fit: launch.max_ctx_that_fit,
            native_baseline_score,
            native_baseline_weak,
            extended_scores,
        };
    }

    // Served stably, but with no evidence at all: never validate on an empty
    // probe set. This is the "no evidence -> never true" guard — it must live
    // HERE (inside the pure core), not be left to a caller's discipline.
    if extended_probes.is_empty() {
        return ValidationEvidence {
            validated: false,
            served_stably: true,
            failure_depth_tokens: None,
            failure_reason: Some(
                "served but no extended-context probes collected — cannot validate without evidence"
                    .to_string(),
            ),
            vram_gb_at_extended_ctx: launch.vram_gb_at_extended_ctx,
            backend_used: launch.backend_used.clone(),
            max_ctx_that_fit: launch.max_ctx_that_fit,
            native_baseline_score,
            native_baseline_weak,
            extended_scores,
        };
    }

    // Served stably with SOME probes, but they never reached the full
    // extended range (target_ctx): the "runs at extended context" claim was
    // never actually exercised at the depth that matters, so it cannot
    // validate either — a shallow-only probe set is incomplete evidence, not
    // sufficient evidence.
    let deepest_probed = extended_probes
        .iter()
        .map(|p| p.depth_tokens)
        .max()
        .unwrap_or(0);
    if deepest_probed < target_ctx {
        return ValidationEvidence {
            validated: false,
            served_stably: true,
            failure_depth_tokens: None,
            failure_reason: Some(format!(
                "extended-context probes did not reach the full target context (deepest probed {deepest_probed} tokens < target {target_ctx} tokens) — cannot validate without evidence at the extended range"
            )),
            vram_gb_at_extended_ctx: launch.vram_gb_at_extended_ctx,
            backend_used: launch.backend_used.clone(),
            max_ctx_that_fit: launch.max_ctx_that_fit,
            native_baseline_score,
            native_baseline_weak,
            extended_scores,
        };
    }

    // Served stably with full-depth coverage, but the native baseline score
    // itself is not a finite number (NaN or +/-inf — e.g. the native probe's
    // fact-planting/grading round-trip itself failed). This is checked BEFORE
    // (and separately from) the weak-baseline floor: `NaN < WEAK_BASELINE_THRESHOLD`
    // is `false`, so a NaN baseline would otherwise slip past that floor AND
    // the relative collapse check (`x < NaN` is also always `false`), landing
    // on validated=true against garbage evidence. Never trust an invalid score.
    if !native_baseline_score.is_finite() {
        return ValidationEvidence {
            validated: false,
            served_stably: true,
            failure_depth_tokens: None,
            failure_reason: Some(format!(
                "native baseline score is not a finite value ({native_baseline_score:?}) — cannot validate against invalid evidence"
            )),
            vram_gb_at_extended_ctx: launch.vram_gb_at_extended_ctx,
            backend_used: launch.backend_used.clone(),
            max_ctx_that_fit: launch.max_ctx_that_fit,
            native_baseline_score,
            native_baseline_weak,
            extended_scores,
        };
    }

    // Served stably with full-depth coverage, but the native baseline itself
    // is too weak to validate against: the relative collapse check below is a
    // RATIO of native_baseline_score, so a zero (or near-zero) baseline makes
    // collapse_threshold zero (or near-zero) too — every extended probe would
    // trivially clear it regardless of whether the model recalls anything at
    // extended context. That's not "held" evidence, it's "no meaningful
    // reference to hold against". So the weak-baseline flag is an absolute
    // FLOOR, not merely informational: refuse to validate rather than let a
    // degenerate native baseline (failed probe round-trip, or a model that
    // genuinely recalls nothing natively) rubber-stamp the extended scores.
    if native_baseline_weak {
        return ValidationEvidence {
            validated: false,
            served_stably: true,
            failure_depth_tokens: None,
            failure_reason: Some(format!(
                "native baseline too weak to validate against (combined score {native_baseline_score:.3} < {WEAK_BASELINE_THRESHOLD:.3}) — cannot confirm YaRN holds when the native reference itself barely recalls anything"
            )),
            vram_gb_at_extended_ctx: launch.vram_gb_at_extended_ctx,
            backend_used: launch.backend_used.clone(),
            max_ctx_that_fit: launch.max_ctx_that_fit,
            native_baseline_score,
            native_baseline_weak,
            extended_scores,
        };
    }

    // Served stably with full-depth coverage: check for a recall/coherence
    // collapse at any probed depth relative to the native baseline. The FIRST
    // collapsing depth is recorded (not the worst) so the evidence shows
    // where things started going wrong.
    //
    // A non-finite (NaN or +/-inf) probe score is treated as a collapse at
    // its depth rather than compared numerically: `NaN < collapse_threshold`
    // is always `false`, so relying on the plain comparison would silently
    // count a degenerate probe as "holding". Fail-safe instead — invalid
    // evidence counts AGAINST validation, never for it. A negative finite
    // score needs no special case: it naturally compares `< collapse_threshold`
    // (a non-negative value here, since the baseline already passed the
    // weak/finite floors above), so it already trips the collapse path.
    let collapse_threshold = native_baseline_score * COLLAPSE_RATIO;
    let collapse = extended_probes.iter().find(|p| {
        let score = p.combined_score();
        !score.is_finite() || score < collapse_threshold
    });

    let validated = collapse.is_none();
    let failure_depth_tokens = collapse.map(|p| p.depth_tokens);
    let failure_reason = collapse.map(|p| {
        let score = p.combined_score();
        if !score.is_finite() {
            format!(
                "recall/coherence probe at depth {} tokens produced a non-finite score ({:?}) — treated as a collapse (invalid evidence never counts as holding)",
                p.depth_tokens, score
            )
        } else {
            format!(
                "recall/coherence collapsed at depth {} tokens: combined score {:.3} fell below {:.0}% of native baseline ({:.3})",
                p.depth_tokens,
                score,
                COLLAPSE_RATIO * 100.0,
                native_baseline_score
            )
        }
    });

    ValidationEvidence {
        validated,
        served_stably: true,
        failure_depth_tokens,
        failure_reason,
        vram_gb_at_extended_ctx: launch.vram_gb_at_extended_ctx,
        backend_used: launch.backend_used.clone(),
        max_ctx_that_fit: launch.max_ctx_that_fit,
        native_baseline_score,
        native_baseline_weak,
        extended_scores,
    }
}

/// Orchestrates a full validation run: launch via the injected [`YarnLauncher`],
/// then — only if it served stably — probe recall/coherence at each depth from
/// [`probe_depths_for`] via the injected [`ContextProber`], then hand
/// everything to the pure [`evaluate`] core for the decision. `native_baseline`
/// is supplied by the caller (it's the model's own native-context probe,
/// collected once and reused across candidate configs, not re-derived here).
///
/// If the launch does not serve stably, no probes are run at all — a wedged/
/// crashed launch is never probed, matching [`evaluate`]'s serving-failure
/// path.
pub fn run_validation(
    model_id: &str,
    rope: &RopeScaling,
    launcher: &dyn YarnLauncher,
    prober: &dyn ContextProber,
    native_baseline: ProbeResult,
) -> ValidationEvidence {
    let launch = launcher.launch(model_id, rope);
    if !launch.served_stably {
        return evaluate(native_baseline, &[], &launch, rope.target_ctx);
    }
    let extended_probes: Vec<ProbeResult> = probe_depths_for(rope.target_ctx)
        .into_iter()
        .map(|depth| prober.probe(depth))
        .collect();
    evaluate(native_baseline, &extended_probes, &launch, rope.target_ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serving::profile::RopeScalingMethod;
    use std::collections::HashMap;

    fn yarn_rope(target_ctx: u32) -> RopeScaling {
        RopeScaling {
            method: RopeScalingMethod::Yarn,
            rope_scale: 4.0,
            yarn_orig_ctx: 32_768,
            target_ctx,
            ext_factor: 1.0,
            attn_factor: 1.0,
            beta_slow: 1.0,
            beta_fast: 32.0,
            validated: false,
        }
    }

    fn strong_probe(depth: u32) -> ProbeResult {
        ProbeResult {
            depth_tokens: depth,
            recall_score: 0.95,
            coherence_score: 0.95,
        }
    }

    // ---- recall_score -------------------------------------------------

    #[test]
    fn recall_score_all_hits_is_one() {
        assert_eq!(recall_score(&[true, true, true]), 1.0);
    }

    #[test]
    fn recall_score_all_misses_is_zero() {
        assert_eq!(recall_score(&[false, false, false, false]), 0.0);
    }

    #[test]
    fn recall_score_partial_hits_is_correct_fraction() {
        // 3 of 4 planted facts recalled -> 0.75, known transcript, known curve.
        assert_eq!(recall_score(&[true, true, true, false]), 0.75);
    }

    #[test]
    fn recall_score_empty_transcript_is_vacuously_one() {
        assert_eq!(recall_score(&[]), 1.0);
    }

    // ---- probe_depths_for ----------------------------------------------

    #[test]
    fn probe_depths_cover_30_60_100_percent_including_full_extended_range() {
        let depths = probe_depths_for(100_000);
        assert_eq!(depths, vec![30_000, 60_000, 100_000]);
        // Must include the full extended range, not stop short of it.
        assert_eq!(*depths.last().unwrap(), 100_000);
    }

    // ---- combined_score --------------------------------------------------

    #[test]
    fn combined_score_is_the_minimum_not_the_average() {
        let p = ProbeResult {
            depth_tokens: 1000,
            recall_score: 0.2,
            coherence_score: 0.99,
        };
        // A fluent-but-wrong completion must not average away a recall miss.
        assert_eq!(p.combined_score(), 0.2);
    }

    #[test]
    fn combined_score_propagates_nan_instead_of_masking_it() {
        // f64::min ignores NaN and returns the OTHER operand — combined_score
        // must NOT inherit that behavior, or a garbage grading round-trip on
        // one side would silently read as a real, healthy score.
        let recall_nan = ProbeResult {
            depth_tokens: 1000,
            recall_score: f64::NAN,
            coherence_score: 0.9,
        };
        assert!(recall_nan.combined_score().is_nan());

        let coherence_nan = ProbeResult {
            depth_tokens: 1000,
            recall_score: 0.9,
            coherence_score: f64::NAN,
        };
        assert!(coherence_nan.combined_score().is_nan());
    }

    // ---- evaluate: positive case ----------------------------------------

    #[test]
    fn holds_to_extended_context_validates_true_with_evidence() {
        let native = strong_probe(8_000);
        let extended = vec![strong_probe(30_000), strong_probe(60_000), strong_probe(100_000)];
        let launch = LaunchReport {
            served_stably: true,
            failure_reason: None,
            vram_gb_at_extended_ctx: Some(42.0),
            backend_used: "rocm".to_string(),
            max_ctx_that_fit: None,
        };

        let evidence = evaluate(native, &extended, &launch, 100_000);

        assert!(evidence.validated);
        assert!(evidence.served_stably);
        assert!(evidence.failure_depth_tokens.is_none());
        assert!(evidence.failure_reason.is_none());
        assert_eq!(evidence.backend_used, "rocm");
        assert_eq!(evidence.vram_gb_at_extended_ctx, Some(42.0));
        assert_eq!(evidence.extended_scores.len(), 3);
        assert!(!evidence.native_baseline_weak);
    }

    // ---- evaluate: negative case — collapse (MOST IMPORTANT case) -------

    #[test]
    fn collapses_past_native_validates_false_with_failure_depth() {
        let native = strong_probe(8_000); // 0.95 baseline
        let extended = vec![
            strong_probe(30_000),
            ProbeResult {
                depth_tokens: 60_000,
                recall_score: 0.1, // collapse: well below 85% of 0.95
                coherence_score: 0.9,
            },
            ProbeResult {
                depth_tokens: 100_000,
                recall_score: 0.05,
                coherence_score: 0.9,
            },
        ];
        let launch = LaunchReport {
            served_stably: true, // it RUNS fine — this is not a serving failure
            failure_reason: None,
            vram_gb_at_extended_ctx: Some(40.0),
            backend_used: "vulkan".to_string(),
            max_ctx_that_fit: None,
        };

        let evidence = evaluate(native, &extended, &launch, 100_000);

        assert!(!evidence.validated);
        assert!(evidence.served_stably, "must be distinguished from a serving failure");
        assert_eq!(evidence.failure_depth_tokens, Some(60_000));
        assert!(evidence
            .failure_reason
            .as_ref()
            .unwrap()
            .contains("collapsed at depth 60000"));
    }

    // ---- evaluate: negative case — wedge/hang/crash ----------------------

    #[test]
    fn wedge_is_recorded_as_serving_failure_not_a_collapse() {
        let native = strong_probe(8_000);
        let launch = LaunchReport {
            served_stably: false,
            failure_reason: Some("wedge: no response within health-check window".to_string()),
            vram_gb_at_extended_ctx: None,
            backend_used: "rocm".to_string(),
            max_ctx_that_fit: None,
        };

        // No probes ran — the harness must handle this gracefully, not panic.
        let evidence = evaluate(native, &[], &launch, 100_000);

        assert!(!evidence.validated);
        assert!(!evidence.served_stably);
        assert!(evidence.failure_depth_tokens.is_none());
        assert_eq!(
            evidence.failure_reason.as_deref(),
            Some("wedge: no response within health-check window")
        );
        assert!(evidence.extended_scores.is_empty());
    }

    // ---- evaluate: served but zero evidence — THE most important guard ---
    // Both independent reviewers flagged this: `served_stably=true` with an
    // EMPTY probe set must never validate=true. `run_validation` happens to
    // always supply probes, but `evaluate` is `pub` and re-exported — this is
    // exactly the seam YARN-04's real production wiring will call directly,
    // so the "no evidence -> never true" invariant has to be enforced HERE,
    // not merely assumed from today's one caller.

    #[test]
    fn served_stably_with_zero_probes_never_validates_true() {
        let native = strong_probe(8_000);
        let launch = LaunchReport {
            served_stably: true,
            failure_reason: None,
            vram_gb_at_extended_ctx: Some(42.0),
            backend_used: "rocm".to_string(),
            max_ctx_that_fit: None,
        };

        let evidence = evaluate(native, &[], &launch, 100_000);

        assert!(
            !evidence.validated,
            "zero probe evidence must never produce validated=true"
        );
        assert!(evidence.served_stably);
        assert!(evidence.extended_scores.is_empty());
        assert!(evidence
            .failure_reason
            .as_deref()
            .unwrap()
            .contains("no extended-context probes collected"));
    }

    // ---- evaluate: shallow-only probes never reach the extended range ----
    // Secondary reviewer note: probes that never actually exercise the full
    // target_ctx must not validate either — "runs at extended context" has
    // to mean the full extended range was probed, not just a shallow slice.

    #[test]
    fn probes_that_never_reach_target_ctx_never_validate_true() {
        let native = strong_probe(8_000);
        // Only shallow probes (30%/60%) — the 100% depth was never probed.
        let shallow_only = vec![strong_probe(30_000), strong_probe(60_000)];
        let launch = LaunchReport {
            served_stably: true,
            failure_reason: None,
            vram_gb_at_extended_ctx: Some(42.0),
            backend_used: "rocm".to_string(),
            max_ctx_that_fit: None,
        };

        let evidence = evaluate(native, &shallow_only, &launch, 100_000);

        assert!(
            !evidence.validated,
            "probes that stop short of target_ctx must never validate=true"
        );
        assert!(evidence.served_stably);
        assert!(evidence
            .failure_reason
            .as_deref()
            .unwrap()
            .contains("did not reach the full target context"));
    }

    // ---- evaluate: backend-dependence -------------------------------------

    #[test]
    fn wedge_on_one_backend_but_holds_on_another_is_recorded_per_backend() {
        let native = strong_probe(8_000);
        let extended_ok = vec![strong_probe(30_000), strong_probe(60_000), strong_probe(100_000)];

        let vulkan_launch = LaunchReport {
            served_stably: false,
            failure_reason: Some("wedge on vulkan backend".to_string()),
            vram_gb_at_extended_ctx: None,
            backend_used: "vulkan".to_string(),
            max_ctx_that_fit: None,
        };
        let rocm_launch = LaunchReport {
            served_stably: true,
            failure_reason: None,
            vram_gb_at_extended_ctx: Some(41.0),
            backend_used: "rocm".to_string(),
            max_ctx_that_fit: None,
        };

        let vulkan_evidence = evaluate(native, &[], &vulkan_launch, 100_000);
        let rocm_evidence = evaluate(native, &extended_ok, &rocm_launch, 100_000);

        assert!(!vulkan_evidence.validated);
        assert_eq!(vulkan_evidence.backend_used, "vulkan");
        assert!(rocm_evidence.validated);
        assert_eq!(rocm_evidence.backend_used, "rocm");
    }

    // ---- evaluate: OOM sizes down, informative not blanket fail ----------

    #[test]
    fn oom_records_max_ctx_that_fit_as_informative_not_blanket_model_failure() {
        let native = strong_probe(8_000);
        let launch = LaunchReport {
            served_stably: false,
            failure_reason: Some("oom: kv cache exceeds free vram at 128000 ctx".to_string()),
            vram_gb_at_extended_ctx: None,
            backend_used: "rocm".to_string(),
            max_ctx_that_fit: Some(64_000),
        };

        let evidence = evaluate(native, &[], &launch, 100_000);

        assert!(!evidence.validated, "the requested target_ctx did not validate");
        assert_eq!(
            evidence.max_ctx_that_fit,
            Some(64_000),
            "the smaller ceiling that DID fit must still be recorded"
        );
    }

    // ---- evaluate: weak native baseline is flagged, not blamed on YaRN ---

    #[test]
    fn weak_native_baseline_is_flagged_and_gates_validation() {
        let weak_native = ProbeResult {
            depth_tokens: 8_000,
            recall_score: 0.3,
            coherence_score: 0.4,
        }; // combined 0.3, below WEAK_BASELINE_THRESHOLD
        let extended = vec![
            ProbeResult { depth_tokens: 30_000, recall_score: 0.3, coherence_score: 0.4 },
            ProbeResult { depth_tokens: 60_000, recall_score: 0.28, coherence_score: 0.38 },
            ProbeResult { depth_tokens: 100_000, recall_score: 0.29, coherence_score: 0.39 },
        ];
        let launch = LaunchReport {
            served_stably: true,
            failure_reason: None,
            vram_gb_at_extended_ctx: Some(30.0),
            backend_used: "rocm".to_string(),
            max_ctx_that_fit: None,
        };

        let evidence = evaluate(weak_native, &extended, &launch, 100_000);

        assert!(evidence.native_baseline_weak);
        // A weak native baseline is an ABSOLUTE FLOOR, not just an FYI flag:
        // the relative collapse check alone can't be trusted here (its
        // threshold shrinks right along with a weak baseline), so this must
        // never validate regardless of how "steady" the extended scores look
        // relative to that weak reference.
        assert!(!evidence.validated);
        assert!(evidence
            .failure_reason
            .as_deref()
            .unwrap()
            .contains("native baseline too weak to validate against"));
    }

    // ---- evaluate: zero baseline — the reviewer-flagged bypass -----------
    // A native baseline of EXACTLY 0.0 makes collapse_threshold 0.0 too, and
    // combined_score() is always >= 0.0, so the relative-only collapse check
    // would never fire (`0.0 < 0.0` is false) even when every extended probe
    // also scores 0.0 — a validation PASS on zero genuine evidence. The
    // absolute weak-baseline floor above must catch this.

    #[test]
    fn zero_native_baseline_with_all_zero_extended_probes_never_validates_true() {
        let zero_native = ProbeResult {
            depth_tokens: 8_000,
            recall_score: 0.0,
            coherence_score: 0.0,
        };
        let all_zero_extended = vec![
            ProbeResult { depth_tokens: 30_000, recall_score: 0.0, coherence_score: 0.0 },
            ProbeResult { depth_tokens: 60_000, recall_score: 0.0, coherence_score: 0.0 },
            ProbeResult { depth_tokens: 100_000, recall_score: 0.0, coherence_score: 0.0 },
        ];
        let launch = LaunchReport {
            served_stably: true,
            failure_reason: None,
            vram_gb_at_extended_ctx: Some(20.0),
            backend_used: "rocm".to_string(),
            max_ctx_that_fit: None,
        };

        let evidence = evaluate(zero_native, &all_zero_extended, &launch, 100_000);

        assert!(
            !evidence.validated,
            "a zero native baseline must never rubber-stamp all-zero extended probes as validated"
        );
        assert!(evidence.native_baseline_weak);
        // Distinct failure reason from the empty-probes and depth-coverage
        // guards, so the evidence trail shows WHICH gate fired.
        let reason = evidence.failure_reason.as_deref().unwrap();
        assert!(reason.contains("native baseline too weak to validate against"));
        assert!(!reason.contains("no extended-context probes collected"));
        assert!(!reason.contains("did not reach the full target context"));
    }

    // ---- evaluate: NaN native baseline — cycle-3 reviewer-flagged bypass -
    // `f64::min` (used by `combined_score`) ignores NaN, and `NaN < x` /
    // `x < NaN` are always `false`. Without an explicit `is_finite()` check,
    // a NaN baseline would skip BOTH the weak-baseline floor (`NaN <
    // WEAK_BASELINE_THRESHOLD` is false) AND the relative collapse check
    // (`x < NaN` is false for every probe) — a validated=true against a
    // garbage baseline. This must be caught by its OWN distinct reason, not
    // conflated with "weak" or "no probes".

    #[test]
    fn nan_native_baseline_never_validates_true_with_distinct_reason() {
        let nan_native = ProbeResult {
            depth_tokens: 8_000,
            recall_score: f64::NAN,
            coherence_score: f64::NAN,
        };
        let extended = vec![strong_probe(30_000), strong_probe(60_000), strong_probe(100_000)];
        let launch = LaunchReport {
            served_stably: true,
            failure_reason: None,
            vram_gb_at_extended_ctx: Some(30.0),
            backend_used: "rocm".to_string(),
            max_ctx_that_fit: None,
        };

        let evidence = evaluate(nan_native, &extended, &launch, 100_000);

        assert!(
            !evidence.validated,
            "a NaN native baseline must never produce validated=true"
        );
        assert!(evidence.served_stably);
        let reason = evidence.failure_reason.as_deref().unwrap();
        assert!(reason.contains("not a finite value"));
        // Distinct from the weak-baseline and empty/depth-coverage reasons.
        assert!(!reason.contains("too weak to validate against"));
        assert!(!reason.contains("no extended-context probes collected"));
        assert!(!reason.contains("did not reach the full target context"));
    }

    // ---- evaluate: healthy baseline, one NaN extended probe --------------

    #[test]
    fn nan_extended_probe_is_recorded_as_the_collapse_point() {
        let native = strong_probe(8_000); // healthy, finite baseline
        let extended = vec![
            strong_probe(30_000),
            ProbeResult {
                depth_tokens: 60_000,
                recall_score: f64::NAN,
                coherence_score: 0.9,
            },
            strong_probe(100_000),
        ];
        let launch = LaunchReport {
            served_stably: true,
            failure_reason: None,
            vram_gb_at_extended_ctx: Some(30.0),
            backend_used: "rocm".to_string(),
            max_ctx_that_fit: None,
        };

        let evidence = evaluate(native, &extended, &launch, 100_000);

        assert!(
            !evidence.validated,
            "a non-finite probe score must never be silently treated as holding"
        );
        assert!(evidence.served_stably);
        assert_eq!(
            evidence.failure_depth_tokens,
            Some(60_000),
            "the NaN probe's depth must be recorded as the failure/collapse point"
        );
        assert!(evidence
            .failure_reason
            .as_deref()
            .unwrap()
            .contains("non-finite score"));
    }

    // ---- run_validation: end-to-end with mocked launcher + prober -------

    struct FakeLauncher {
        report: LaunchReport,
    }
    impl YarnLauncher for FakeLauncher {
        fn launch(&self, _model_id: &str, _rope: &RopeScaling) -> LaunchReport {
            self.report.clone()
        }
    }

    struct FakeProber {
        by_depth: HashMap<u32, ProbeResult>,
    }
    impl ContextProber for FakeProber {
        fn probe(&self, depth_tokens: u32) -> ProbeResult {
            *self
                .by_depth
                .get(&depth_tokens)
                .expect("test should provide every probed depth")
        }
    }

    #[test]
    fn end_to_end_holds_produces_validated_true_with_full_evidence() {
        let rope = yarn_rope(100_000);
        let launcher = FakeLauncher {
            report: LaunchReport {
                served_stably: true,
                failure_reason: None,
                vram_gb_at_extended_ctx: Some(50.0),
                backend_used: "rocm".to_string(),
                max_ctx_that_fit: None,
            },
        };
        let mut by_depth = HashMap::new();
        for depth in probe_depths_for(rope.target_ctx) {
            by_depth.insert(depth, strong_probe(depth));
        }
        let prober = FakeProber { by_depth };
        let native_baseline = strong_probe(8_000);

        let evidence = run_validation("qwen3-coder:30b", &rope, &launcher, &prober, native_baseline);

        assert!(evidence.validated);
        assert!(evidence.served_stably);
        assert_eq!(evidence.extended_scores.len(), 3);
        assert_eq!(evidence.backend_used, "rocm");
        assert_eq!(evidence.vram_gb_at_extended_ctx, Some(50.0));
    }

    #[test]
    fn end_to_end_wedge_produces_validated_false_without_probing() {
        let rope = yarn_rope(100_000);
        let launcher = FakeLauncher {
            report: LaunchReport {
                served_stably: false,
                failure_reason: Some("wedge: no response".to_string()),
                vram_gb_at_extended_ctx: None,
                backend_used: "vulkan".to_string(),
                max_ctx_that_fit: None,
            },
        };
        // Prober intentionally has NO entries — if run_validation tried to
        // probe after a wedge, this test would panic on `.expect(...)`.
        let prober = FakeProber { by_depth: HashMap::new() };
        let native_baseline = strong_probe(8_000);

        let evidence = run_validation("qwen3-coder:30b", &rope, &launcher, &prober, native_baseline);

        assert!(!evidence.validated);
        assert!(!evidence.served_stably);
        assert!(evidence.extended_scores.is_empty());
    }
}
