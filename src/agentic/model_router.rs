//! AGENT-05: Agentic mode model routing with mid-loop escalation.
//!
//! `AgenticModelRouter` selects the appropriate model for each step of the
//! agentic loop.  If the context becomes complex (many tool results, large
//! content, or reasoning-oriented query keywords) the router escalates once to a
//! deeper/larger model.  The escalation is capped at one per execution so VRAM
//! is not thrashed.
//!
//! Model names are fully configurable via environment variables:
//!   `CHORD_FAST_MODEL` — the lightweight default model (default: `qwen2.5:20b`)
//!   `CHORD_DEEP_MODEL` — the escalation model     (default: `qwen2.5:120b`)
//!
//! No hardcoded model names appear outside the default-fallback strings.

// ── Complexity heuristic ──────────────────────────────────────────────────────

/// Words in the user query that signal multi-source reasoning, justifying
/// escalation to the deep model.  The list is intentionally small and
/// conservative — false positives waste VRAM; false negatives only delay one
/// escalation.
const REASONING_WORDS: &[&str] = &[
    "analyze",
    "compare",
    "synthesize",
    "evaluate",
    "explain why",
    "reason about",
];

/// Tunable thresholds for the complexity heuristic.
///
/// These are separated into a struct so tests can override them without env vars.
#[derive(Debug, Clone)]
pub struct ComplexityHeuristic {
    /// Escalate when the number of tool results in this turn exceeds this value.
    pub tool_result_count_threshold: usize,
    /// Escalate when the total character count of all results exceeds this value.
    pub total_chars_threshold: usize,
    /// Lowercase reasoning keywords that trigger escalation when found in the query.
    pub reasoning_words: Vec<String>,
}

impl Default for ComplexityHeuristic {
    fn default() -> Self {
        Self {
            tool_result_count_threshold: 2,
            total_chars_threshold: 5_000,
            reasoning_words: REASONING_WORDS.iter().map(|s| s.to_string()).collect(),
        }
    }
}

impl ComplexityHeuristic {
    /// Assess whether the current context warrants escalation to the deep model.
    ///
    /// Returns `true` if ANY of the following hold:
    /// - `tool_result_count > tool_result_count_threshold`
    /// - `total_chars > total_chars_threshold`
    /// - `query` (lowercased) contains one of the reasoning words
    pub fn assess_complexity(
        &self,
        tool_result_count: usize,
        total_chars: usize,
        query: &str,
    ) -> bool {
        if tool_result_count > self.tool_result_count_threshold {
            return true;
        }
        if total_chars > self.total_chars_threshold {
            return true;
        }
        let query_lower = query.to_lowercase();
        for word in &self.reasoning_words {
            if query_lower.contains(word.as_str()) {
                return true;
            }
        }
        false
    }
}

// ── AgenticModelRouter ────────────────────────────────────────────────────────

/// Manages model selection within the agentic loop.
///
/// Create one router per execution (not shared across requests) so escalation
/// state does not leak between users.
///
/// # Example
/// ```rust
/// use chord_proxy::agentic::model_router::AgenticModelRouter;
///
/// let mut router = AgenticModelRouter::new();
/// let model = router.current_model().to_string();
/// // After complex tool results:
/// if let Some(deep) = router.escalate() {
///     println!("Escalated to {}", deep);
/// }
/// // Second escalation attempt is a no-op:
/// assert!(router.escalate().is_none());
/// ```
#[derive(Debug, Clone)]
pub struct AgenticModelRouter {
    /// Model currently in use for inference.
    current: String,
    /// Lightweight fast model (used by default).
    fast_model: String,
    /// Deep / large model used after escalation.
    deep_model: String,
    /// Whether escalation has already occurred for this execution.
    escalated: bool,
    /// Complexity heuristic configuration.
    heuristic: ComplexityHeuristic,
}

impl AgenticModelRouter {
    /// Construct a new router, reading model names from environment variables.
    ///
    /// Falls back to `qwen2.5:20b` / `qwen2.5:120b` when env vars are absent.
    pub fn new() -> Self {
        let fast_model = std::env::var("CHORD_FAST_MODEL")
            .unwrap_or_else(|_| "qwen2.5:20b".to_string());
        let deep_model = std::env::var("CHORD_DEEP_MODEL")
            .unwrap_or_else(|_| "qwen2.5:120b".to_string());
        let current = fast_model.clone();
        Self {
            current,
            fast_model,
            deep_model,
            escalated: false,
            heuristic: ComplexityHeuristic::default(),
        }
    }

    /// Construct a router with explicit model names (useful for testing without
    /// touching environment variables).
    pub fn with_models(fast_model: impl Into<String>, deep_model: impl Into<String>) -> Self {
        let fast = fast_model.into();
        let deep = deep_model.into();
        Self {
            current: fast.clone(),
            fast_model: fast,
            deep_model: deep,
            escalated: false,
            heuristic: ComplexityHeuristic::default(),
        }
    }

    /// Return `true` if the current context is complex enough to warrant
    /// escalation.  Delegates to `ComplexityHeuristic::assess_complexity`.
    pub fn should_escalate(
        &self,
        tool_result_count: usize,
        total_chars: usize,
        query: &str,
    ) -> bool {
        self.heuristic
            .assess_complexity(tool_result_count, total_chars, query)
    }

    /// Attempt to escalate to the deep model.
    ///
    /// Returns `Some(deep_model_name)` on the first call, updating
    /// `current_model` to the deep model.  All subsequent calls return `None`
    /// (max one escalation per execution).
    ///
    /// If the router is already on the deep model (e.g. forced via
    /// `force_deep`), returns `None` immediately without recording an additional
    /// escalation.
    pub fn escalate(&mut self) -> Option<String> {
        if self.escalated {
            return None;
        }
        if self.current == self.deep_model {
            // Already on deep — count this as the escalation so we never try again.
            self.escalated = true;
            return None;
        }
        self.escalated = true;
        self.current = self.deep_model.clone();
        Some(self.deep_model.clone())
    }

    /// Return the model name that should be used for the current inference step.
    pub fn current_model(&self) -> &str {
        &self.current
    }

    /// Force the router directly to the deep model (e.g. for `/deep` prefixed
    /// requests).  Sets `escalated = true` so future `escalate()` calls are
    /// no-ops.
    pub fn force_deep(&mut self) {
        self.current = self.deep_model.clone();
        self.escalated = true;
    }

    /// Detect whether `message` starts with the `/deep` prefix, indicating the
    /// user wants the deep model for the entire execution.
    ///
    /// The check is case-insensitive and trims leading whitespace so that
    /// `/deep` and `/Deep` and `  /deep ` all match.
    pub fn is_deep_request(message: &str) -> bool {
        message.trim_start().to_lowercase().starts_with("/deep")
    }
}

impl Default for AgenticModelRouter {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ComplexityHeuristic tests ─────────────────────────────────────────────

    #[test]
    fn test_simple_context_does_not_escalate() {
        let h = ComplexityHeuristic::default();
        // 1 result, small char count, no reasoning words
        assert!(!h.assess_complexity(1, 100, "what time is it"));
    }

    #[test]
    fn test_high_tool_result_count_triggers_escalation() {
        let h = ComplexityHeuristic::default();
        // threshold is 2, so >2 = 3 should trigger
        assert!(h.assess_complexity(3, 100, "simple query"));
    }

    #[test]
    fn test_exactly_at_threshold_does_not_trigger() {
        let h = ComplexityHeuristic::default();
        // threshold is 2, count == 2 should NOT trigger (strictly greater than)
        assert!(!h.assess_complexity(2, 100, "simple query"));
    }

    #[test]
    fn test_large_char_count_triggers_escalation() {
        let h = ComplexityHeuristic::default();
        // threshold is 5000, so >5000 should trigger
        assert!(h.assess_complexity(1, 5_001, "simple query"));
    }

    #[test]
    fn test_exactly_at_char_threshold_does_not_trigger() {
        let h = ComplexityHeuristic::default();
        assert!(!h.assess_complexity(1, 5_000, "simple query"));
    }

    #[test]
    fn test_reasoning_word_analyze_triggers_escalation() {
        let h = ComplexityHeuristic::default();
        assert!(h.assess_complexity(0, 0, "please analyze this data"));
    }

    #[test]
    fn test_reasoning_word_compare_triggers_escalation() {
        let h = ComplexityHeuristic::default();
        assert!(h.assess_complexity(0, 0, "compare these two approaches"));
    }

    #[test]
    fn test_reasoning_word_synthesize_triggers_escalation() {
        let h = ComplexityHeuristic::default();
        assert!(h.assess_complexity(0, 0, "synthesize the findings"));
    }

    #[test]
    fn test_reasoning_word_evaluate_triggers_escalation() {
        let h = ComplexityHeuristic::default();
        assert!(h.assess_complexity(0, 0, "evaluate the options"));
    }

    #[test]
    fn test_reasoning_word_explain_why_triggers_escalation() {
        let h = ComplexityHeuristic::default();
        assert!(h.assess_complexity(0, 0, "explain why this fails"));
    }

    #[test]
    fn test_reasoning_word_reason_about_triggers_escalation() {
        let h = ComplexityHeuristic::default();
        assert!(h.assess_complexity(0, 0, "reason about the consequences"));
    }

    #[test]
    fn test_reasoning_word_case_insensitive() {
        let h = ComplexityHeuristic::default();
        assert!(h.assess_complexity(0, 0, "ANALYZE the results"));
        assert!(h.assess_complexity(0, 0, "Synthesize all data"));
    }

    #[test]
    fn test_all_six_reasoning_words_covered() {
        let h = ComplexityHeuristic::default();
        let queries = [
            "analyze this",
            "compare those",
            "synthesize the data",
            "evaluate options",
            "explain why it failed",
            "reason about consequences",
        ];
        for q in &queries {
            assert!(
                h.assess_complexity(0, 0, q),
                "Expected escalation for query: {}",
                q
            );
        }
    }

    // ── AgenticModelRouter — model selection ──────────────────────────────────

    #[test]
    fn test_new_router_starts_on_fast_model() {
        let router = AgenticModelRouter::with_models("fast-model", "deep-model");
        assert_eq!(router.current_model(), "fast-model");
    }

    #[test]
    fn test_simple_query_stays_on_fast_model() {
        let mut router = AgenticModelRouter::with_models("fast-20b", "deep-120b");
        // No escalation triggered
        assert!(!router.should_escalate(1, 100, "hello"));
        assert_eq!(router.current_model(), "fast-20b");
    }

    #[test]
    fn test_complex_query_should_escalate_returns_true() {
        let router = AgenticModelRouter::with_models("fast-20b", "deep-120b");
        assert!(router.should_escalate(3, 100, "simple"));
        assert!(router.should_escalate(1, 6000, "simple"));
        assert!(router.should_escalate(0, 0, "analyze this"));
    }

    #[test]
    fn test_escalate_returns_deep_model_first_time() {
        let mut router = AgenticModelRouter::with_models("fast-20b", "deep-120b");
        let escalated_to = router.escalate();
        assert_eq!(escalated_to, Some("deep-120b".to_string()));
        assert_eq!(router.current_model(), "deep-120b");
    }

    #[test]
    fn test_escalate_second_call_returns_none() {
        let mut router = AgenticModelRouter::with_models("fast-20b", "deep-120b");
        // First escalation
        let first = router.escalate();
        assert!(first.is_some());
        // Second escalation must be rejected
        let second = router.escalate();
        assert!(second.is_none(), "max 1 escalation enforced");
    }

    #[test]
    fn test_escalate_many_calls_still_returns_none_after_first() {
        let mut router = AgenticModelRouter::with_models("fast-20b", "deep-120b");
        router.escalate(); // first: ok
        for _ in 0..10 {
            assert!(router.escalate().is_none());
        }
    }

    #[test]
    fn test_model_is_deep_after_escalation() {
        let mut router = AgenticModelRouter::with_models("fast-20b", "deep-120b");
        router.escalate();
        assert_eq!(router.current_model(), "deep-120b");
    }

    // ── force_deep ────────────────────────────────────────────────────────────

    #[test]
    fn test_force_deep_sets_current_to_deep_model() {
        let mut router = AgenticModelRouter::with_models("fast-20b", "deep-120b");
        router.force_deep();
        assert_eq!(router.current_model(), "deep-120b");
    }

    #[test]
    fn test_force_deep_blocks_further_escalation() {
        let mut router = AgenticModelRouter::with_models("fast-20b", "deep-120b");
        router.force_deep();
        // After force_deep the escalation slot is consumed — no swap needed.
        assert!(router.escalate().is_none());
    }

    #[test]
    fn test_force_deep_idempotent() {
        let mut router = AgenticModelRouter::with_models("fast-20b", "deep-120b");
        router.force_deep();
        router.force_deep(); // should not panic
        assert_eq!(router.current_model(), "deep-120b");
    }

    // ── /deep prefix detection ────────────────────────────────────────────────

    #[test]
    fn test_is_deep_request_detects_slash_deep() {
        assert!(AgenticModelRouter::is_deep_request("/deep analyze the data"));
    }

    #[test]
    fn test_is_deep_request_case_insensitive() {
        assert!(AgenticModelRouter::is_deep_request("/Deep analyze"));
        assert!(AgenticModelRouter::is_deep_request("/DEEP analyze"));
    }

    #[test]
    fn test_is_deep_request_with_leading_whitespace() {
        assert!(AgenticModelRouter::is_deep_request("  /deep analyze the data"));
    }

    #[test]
    fn test_is_deep_request_false_for_normal_message() {
        assert!(!AgenticModelRouter::is_deep_request("what is the weather?"));
    }

    #[test]
    fn test_is_deep_request_false_for_slash_other() {
        assert!(!AgenticModelRouter::is_deep_request("/help me with this"));
    }

    #[test]
    fn test_is_deep_request_just_slash_deep() {
        assert!(AgenticModelRouter::is_deep_request("/deep"));
    }

    #[test]
    fn test_is_deep_request_empty_string() {
        assert!(!AgenticModelRouter::is_deep_request(""));
    }

    // ── Env var loading ───────────────────────────────────────────────────────

    #[test]
    fn test_env_var_fast_model_default() {
        // When CHORD_FAST_MODEL is not set, default is "qwen2.5:20b"
        // We use with_models here to avoid touching env (parallel tests may interfere)
        let router = AgenticModelRouter::with_models("qwen2.5:20b", "qwen2.5:120b");
        assert_eq!(router.fast_model, "qwen2.5:20b");
    }

    #[test]
    fn test_env_var_deep_model_default() {
        let router = AgenticModelRouter::with_models("qwen2.5:20b", "qwen2.5:120b");
        assert_eq!(router.deep_model, "qwen2.5:120b");
    }

    #[test]
    fn test_new_reads_env_vars() {
        // Set env vars temporarily (serial-only test — use std::env carefully).
        // We test that AgenticModelRouter::new() picks up the variables.
        // NOTE: Parallel tests can interfere with env vars; use a dedicated
        // serialised test binary if needed. Here we just assert the default path
        // works without env vars being set (most CI environments won't have them).
        let router = AgenticModelRouter::new();
        // Whatever the env says, the router should not panic and both model
        // names must be non-empty.
        assert!(!router.fast_model.is_empty());
        assert!(!router.deep_model.is_empty());
        assert!(!router.current_model().is_empty());
    }

    #[test]
    fn test_no_hardcoded_model_names_except_defaults() {
        // The defaults are explicitly allowed per spec ("configurable defaults").
        // This test documents that the default strings are the only hardcoded values.
        let router = AgenticModelRouter::with_models("custom-fast", "custom-deep");
        assert_eq!(router.fast_model, "custom-fast");
        assert_eq!(router.deep_model, "custom-deep");
        // No mention of qwen in the custom router
        assert!(!router.current_model().contains("qwen"));
    }

    // ── already_on_deep edge case ─────────────────────────────────────────────

    #[test]
    fn test_escalate_when_already_on_deep_returns_none() {
        // Router starts on deep model (e.g. after force_deep)
        let mut router = AgenticModelRouter::with_models("fast-20b", "deep-120b");
        router.force_deep();
        // escalate() must return None — no swap, already there
        assert!(router.escalate().is_none());
    }

    // ── escalation state preserved ────────────────────────────────────────────

    #[test]
    fn test_current_model_stays_deep_after_escalation() {
        let mut router = AgenticModelRouter::with_models("fast-20b", "deep-120b");
        router.escalate();
        // Current must remain deep for all subsequent calls
        assert_eq!(router.current_model(), "deep-120b");
        assert_eq!(router.current_model(), "deep-120b");
    }

    // ── /deep + escalation interaction ───────────────────────────────────────

    #[test]
    fn test_force_deep_then_should_escalate_does_not_change_model() {
        let mut router = AgenticModelRouter::with_models("fast-20b", "deep-120b");
        router.force_deep();
        // Even if heuristic says "escalate", slot is consumed
        assert!(router.should_escalate(5, 10000, "analyze and compare everything"));
        assert!(router.escalate().is_none()); // slot already used
        assert_eq!(router.current_model(), "deep-120b"); // still deep
    }
}
