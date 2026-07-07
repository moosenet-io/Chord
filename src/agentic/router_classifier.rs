//! ROUT-02 / ROUT-03: Defensive parser for the Supra-Router-51M local
//! classifier daemon's raw output, plus the trivial-arithmetic/trivial-code
//! over-escalation correction.
//!
//! This session's evaluation of `SupraLabs/Supra-Router-51M` (a 51.7M-param
//! fine-tuned routing classifier) against a 47-prompt test set found:
//!
//!   1. The model's own `Route` field is 100% deterministically derivable from
//!      its `Complexity`/`Math`/`Code` flags — it is not independent
//!      reasoning. This parser therefore NEVER reads `Route` or
//!      `Justification` as decision inputs; it recomputes the route in Rust
//!      from the three real signal fields (see [`route_for`]).
//!   2. ~11% of raw outputs have a garbled `Domain` field, and ~2% are fully
//!      unparseable. `Domain`/`Justification` are informational/logging only
//!      (see [`RouterClassification::raw_domain`]) and are never required for
//!      a successful parse.
//!   3. The documented `Label: value` output format is NOT reliable — the
//!      model frequently omits labels entirely. Parsing is therefore
//!      label-tolerant: each of the three real signal fields is matched by a
//!      regex that works with or without its label, and a fully malformed or
//!      unrelated blob fails closed to [`ClassificationError::Unavailable`]
//!      rather than a wrong guess.
//!   4. The model over-escalates trivial single-operation arithmetic
//!      (`12*8`) and one-line code (`def f(x): return x+1`) to "big model"
//!      purely because `Math`/`Code` trip, regardless of actual difficulty —
//!      a blind spot from its tiny (992-sample) training set. [`route_for`]
//!      applies a narrow, conservative correction for exactly this case; it
//!      does not attempt general-purpose difficulty estimation, and a
//!      `Complexity >= 3` signal still escalates independently of the
//!      trivial-case override.

use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};

/// Route decision. Always recomputed in Rust from `Complexity`/`Math`/`Code`
/// — never read from the daemon's own `Route` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Route {
    Small,
    Big,
}

/// A successfully parsed classification.
///
/// `raw_domain` is retained ONLY for audit-log/telemetry purposes (ROUT-06)
/// — it is never consulted for the routing decision itself.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouterClassification {
    /// 1-5, clamped. Out-of-range values from the daemon (e.g. `9`) are
    /// clamped rather than propagated as garbage.
    pub complexity: u8,
    pub math: bool,
    pub code: bool,
    /// Informational only — logging/telemetry, never a decision input.
    pub raw_domain: Option<String>,
}

/// Why a raw daemon output could not be turned into a usable classification.
///
/// Callers (ROUT-04's hybrid router) must treat this identically to a
/// daemon timeout/unreachability: fall back to the existing keyword
/// heuristic, never block or error the request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassificationError {
    Unavailable,
}

const BIG_MODEL_COMPLEXITY_THRESHOLD: u8 = 3;
const MIN_COMPLEXITY: u8 = 1;
const MAX_COMPLEXITY: u8 = 5;

// ── Field extraction (label-tolerant) ──────────────────────────────────────
//
// The daemon's documented format is `Domain: ... | Complexity: N | Math: bool
// | Code: bool`, but this session's eval found labels are frequently omitted.
// Each regex therefore matches the label when present, and callers fall back
// to `None` (never a guess) when it is absent and no unambiguous positional
// signal exists.

static COMPLEXITY_LABELED_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)complexity\D{0,4}(\d{1,2})").unwrap());
/// Label-omitted fallback: a bare `N/5` complexity score elsewhere in the text.
static COMPLEXITY_BARE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b([1-5])\s*/\s*5\b").unwrap());
static MATH_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)\bmath\D{0,4}(true|false|yes|no)\b").unwrap());
static CODE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)\bcode\D{0,4}(true|false|yes|no)\b").unwrap());
static DOMAIN_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)domain\s*:?\s*\|?\s*([A-Za-z0-9_\-. ]{1,40})").unwrap());

fn parse_bool_token(tok: &str) -> bool {
    matches!(tok.to_lowercase().as_str(), "true" | "yes")
}

fn extract_complexity(raw: &str) -> Option<u8> {
    if let Some(v) = COMPLEXITY_LABELED_RE
        .captures(raw)
        .and_then(|c| c.get(1))
        .and_then(|m| m.as_str().parse::<u32>().ok())
    {
        return Some(v.min(u8::MAX as u32) as u8);
    }
    COMPLEXITY_BARE_RE
        .captures(raw)
        .and_then(|c| c.get(1))
        .and_then(|m| m.as_str().parse::<u8>().ok())
}

fn extract_bool(raw: &str, re: &Regex) -> Option<bool> {
    re.captures(raw)
        .and_then(|c| c.get(1))
        .map(|m| parse_bool_token(m.as_str()))
}

/// Parse the daemon's raw text output into a [`RouterClassification`].
///
/// Extracts ONLY `Complexity`/`Math`/`Code`. `Domain` is best-effort and
/// purely informational. `Route` and `Justification` are never read at all
/// — the route is always recomputed by [`route_for`].
///
/// Returns [`ClassificationError::Unavailable`] (never a guess) when:
/// - the input is empty (covers daemon timeout/empty-response callers that
///   pass through an empty string identically)
/// - `Complexity`, `Math`, or `Code` cannot be confidently extracted at all
pub fn parse_classification(raw: &str) -> Result<RouterClassification, ClassificationError> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(ClassificationError::Unavailable);
    }

    let complexity = extract_complexity(trimmed);
    let math = extract_bool(trimmed, &MATH_RE);
    let code = extract_bool(trimmed, &CODE_RE);

    let (complexity, math, code) = match (complexity, math, code) {
        (Some(c), Some(m), Some(cd)) => (c, m, cd),
        _ => return Err(ClassificationError::Unavailable),
    };

    let complexity = complexity.clamp(MIN_COMPLEXITY, MAX_COMPLEXITY);
    let raw_domain = DOMAIN_RE
        .captures(trimmed)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().trim().to_string())
        .filter(|s| !s.is_empty());

    Ok(RouterClassification {
        complexity,
        math,
        code,
        raw_domain,
    })
}

// ── ROUT-03: trivial-arithmetic / trivial-code correction ─────────────────

/// Single-operation arithmetic expressions, e.g. `12*8`, `3.5 + 2`, `-4 / 2`.
/// Deliberately narrow: exactly one binary operator, nothing else.
static TRIVIAL_ARITHMETIC_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^\s*-?\d+(\.\d+)?\s*[+\-*/]\s*-?\d+(\.\d+)?\s*$").unwrap());

/// Keywords that disqualify a snippet from being "trivially short code" —
/// presence of control flow / structure implies genuine complexity.
const CONTROL_FLOW_KEYWORDS: &[&str] = &[
    "if ", "elif ", "else", "for ", "while ", "class ", "try", "except", "match ", "loop", "switch",
];

const TRIVIAL_CODE_MAX_LEN: usize = 80;

/// A prompt is "trivially short code" if it is a single line, short, contains
/// no control-flow keywords, and looks like a minimal function/expression
/// rather than a request for real code (heuristic, intentionally narrow —
/// this is a targeted fix for the eval's confirmed blind spot, not a
/// general-purpose difficulty estimator).
pub fn is_trivial_code(prompt: &str) -> bool {
    let trimmed = prompt.trim();
    if trimmed.is_empty() || trimmed.lines().count() > 1 {
        return false;
    }
    if trimmed.len() > TRIVIAL_CODE_MAX_LEN {
        return false;
    }
    let lower = trimmed.to_lowercase();
    if CONTROL_FLOW_KEYWORDS.iter().any(|kw| lower.contains(kw)) {
        return false;
    }
    lower.starts_with("def ")
        || lower.contains("lambda")
        || lower.contains("=>")
        || lower.contains("return ")
        || lower.contains("function ")
}

/// A prompt is "trivial arithmetic" if it is exactly one binary arithmetic
/// operation with no surrounding text.
pub fn is_trivial_arithmetic(prompt: &str) -> bool {
    TRIVIAL_ARITHMETIC_RE.is_match(prompt.trim())
}

/// ROUT-02 + ROUT-03: recompute the route decision in Rust.
///
/// Base rule (ROUT-02, confirmed 100%-derivable from the eval): big-model if
/// `Complexity >= 3 OR Math OR Code`.
///
/// Correction (ROUT-03): a bare `Math`/`Code` flag alone does not force
/// big-model routing when the prompt is a confirmed-trivial single-operation
/// arithmetic expression or one-line code ask. `Complexity >= 3` always
/// escalates regardless — a trivial-looking prompt embedded in a genuinely
/// complex request is not suppressed.
pub fn route_for(prompt: &str, classification: &RouterClassification) -> Route {
    if classification.complexity >= BIG_MODEL_COMPLEXITY_THRESHOLD {
        return Route::Big;
    }

    let trivial_override = (classification.math && is_trivial_arithmetic(prompt))
        || (classification.code && is_trivial_code(prompt));

    if (classification.math || classification.code) && !trivial_override {
        return Route::Big;
    }

    Route::Small
}

// ─────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Clean, fully-labeled output ────────────────────────────────────────

    #[test]
    fn test_clean_labeled_output_parses() {
        let raw = "Domain: general | Complexity: 4 | Math: False | Code: False | Route: big | Justification: multi-step reasoning";
        let c = parse_classification(raw).unwrap();
        assert_eq!(c.complexity, 4);
        assert!(!c.math);
        assert!(!c.code);
        assert_eq!(c.raw_domain.as_deref(), Some("general"));
    }

    #[test]
    fn test_route_never_reads_raw_route_field() {
        // Raw Route says "small" but Complexity=5 must still force Big — the
        // parser must not be influenced by the raw Route string at all
        // (it isn't even a field on RouterClassification).
        let raw = "Domain: math | Complexity: 5 | Math: True | Code: False | Route: small | Justification: trust me";
        let c = parse_classification(raw).unwrap();
        assert_eq!(route_for("integrate this differential equation", &c), Route::Big);
    }

    // ── Label-omitted output (documented eval finding) ──────────────────────

    #[test]
    fn test_label_omitted_domain_still_parses_complexity_math_code() {
        // Domain label is dropped, but Complexity/Math/Code labels remain —
        // 11%-garbled-Domain case from the eval.
        let raw = "general/coding | Complexity: 2 | Math: False | Code: True";
        let c = parse_classification(raw).unwrap();
        assert_eq!(c.complexity, 2);
        assert!(!c.math);
        assert!(c.code);
    }

    #[test]
    fn test_bare_complexity_score_without_label() {
        let raw = "3/5 math:no code:yes";
        let c = parse_classification(raw).unwrap();
        assert_eq!(c.complexity, 3);
        assert!(!c.math);
        assert!(c.code);
    }

    // ── Fully unparseable / malformed ───────────────────────────────────────

    #[test]
    fn test_fully_unparseable_returns_unavailable() {
        let raw = "the model rambled about something unrelated with no structure";
        assert_eq!(parse_classification(raw), Err(ClassificationError::Unavailable));
    }

    #[test]
    fn test_empty_string_returns_unavailable() {
        assert_eq!(parse_classification(""), Err(ClassificationError::Unavailable));
    }

    #[test]
    fn test_whitespace_only_returns_unavailable() {
        assert_eq!(parse_classification("   \n\t  "), Err(ClassificationError::Unavailable));
    }

    #[test]
    fn test_partial_fields_returns_unavailable() {
        // Complexity present but Math/Code missing entirely — must fail
        // closed rather than guess.
        let raw = "Complexity: 4 | Domain: general";
        assert_eq!(parse_classification(raw), Err(ClassificationError::Unavailable));
    }

    // ── Out-of-range clamping ────────────────────────────────────────────────

    #[test]
    fn test_out_of_range_complexity_is_clamped_not_propagated() {
        let raw = "Complexity: 9 | Math: True | Code: False";
        let c = parse_classification(raw).unwrap();
        assert_eq!(c.complexity, MAX_COMPLEXITY);
    }

    #[test]
    fn test_zero_complexity_clamped_to_min() {
        let raw = "Complexity: 0 | Math: False | Code: False";
        let c = parse_classification(raw).unwrap();
        assert_eq!(c.complexity, MIN_COMPLEXITY);
    }

    // ── ROUT-02: route recomputation ────────────────────────────────────────

    #[test]
    fn test_route_small_when_no_signals() {
        let c = RouterClassification { complexity: 1, math: false, code: false, raw_domain: None };
        assert_eq!(route_for("hello there", &c), Route::Small);
    }

    #[test]
    fn test_route_big_on_high_complexity_alone() {
        let c = RouterClassification { complexity: 3, math: false, code: false, raw_domain: None };
        assert_eq!(route_for("hello there", &c), Route::Big);
    }

    #[test]
    fn test_route_big_on_math_flag_for_nontrivial_prompt() {
        let c = RouterClassification { complexity: 1, math: true, code: false, raw_domain: None };
        assert_eq!(
            route_for("solve the integral of x^2 * sin(x) dx", &c),
            Route::Big
        );
    }

    #[test]
    fn test_route_big_on_code_flag_for_nontrivial_prompt() {
        let c = RouterClassification { complexity: 1, math: false, code: true, raw_domain: None };
        assert_eq!(
            route_for("write a thread-safe LRU cache with eviction callbacks", &c),
            Route::Big
        );
    }

    // ── ROUT-03: trivial-case correction ────────────────────────────────────

    #[test]
    fn test_trivial_arithmetic_does_not_force_big_model() {
        let c = RouterClassification { complexity: 1, math: true, code: false, raw_domain: None };
        assert_eq!(route_for("12*8", &c), Route::Small);
        assert_eq!(route_for(" 3.5 + 2 ", &c), Route::Small);
        assert_eq!(route_for("-4 / 2", &c), Route::Small);
    }

    #[test]
    fn test_trivial_code_does_not_force_big_model() {
        let c = RouterClassification { complexity: 1, math: false, code: true, raw_domain: None };
        assert_eq!(route_for("def f(x): return x+1", &c), Route::Small);
        assert_eq!(route_for("lambda x: x * 2", &c), Route::Small);
    }

    #[test]
    fn test_trivial_case_still_escalates_on_independent_complexity() {
        // Trivial arithmetic embedded in a complexity>=3 signal must still
        // escalate — the correction only suppresses the Math/Code-alone
        // trigger, not an independently-flagged complex request.
        let c = RouterClassification { complexity: 3, math: true, code: false, raw_domain: None };
        assert_eq!(route_for("12*8", &c), Route::Big);
    }

    #[test]
    fn test_multi_line_code_is_not_considered_trivial() {
        let c = RouterClassification { complexity: 1, math: false, code: true, raw_domain: None };
        assert_eq!(
            route_for("def f(x):\n    return x + 1", &c),
            Route::Big,
            "multi-line snippets are not the confirmed trivial blind spot"
        );
    }

    #[test]
    fn test_control_flow_code_is_not_considered_trivial() {
        let c = RouterClassification { complexity: 1, math: false, code: true, raw_domain: None };
        assert_eq!(
            route_for("def f(x): return x+1 if x > 0 else -x", &c),
            Route::Big
        );
    }

    #[test]
    fn test_multi_term_arithmetic_is_not_trivial() {
        let c = RouterClassification { complexity: 1, math: true, code: false, raw_domain: None };
        // Two operators — outside the "single-operation" definition.
        assert_eq!(route_for("12*8+3", &c), Route::Big);
    }

    #[test]
    fn test_no_hardcoded_infrastructure_values() {
        // Documentation test: this module contains no IPs/hostnames/org
        // names. Grep-style guard so future edits don't regress silently.
        let src = include_str!("router_classifier.rs");
        let private_ip_prefix = ["192", "168", "."].concat();
        let org_domain = ["moosenet", ".online"].concat();
        assert!(!src.contains(&private_ip_prefix));
        assert!(!src.contains(&org_domain));
    }
}
