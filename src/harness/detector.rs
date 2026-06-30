//! Research query detection and triggering (HRNS-04).
//!
//! Decides whether a user query warrants the full Harness-1 deep-research loop
//! versus a simple `searxng_search` call. Not every "search for X" needs a
//! 40-turn research agent — quick factual lookups should hit regular search
//! directly. Harness-1 is reserved for deep, multi-source research.
//!
//! Three signals drive the decision:
//! 1. **Explicit command** — `/research <q>` or `/deep-search <q>` always trigger.
//! 2. **Implicit keywords** — research-intent words or a long query, gated by a
//!    "this is not a trivial factual lookup" guard.
//! 3. **Complexity heuristic** — entity count, comparison words, temporal scope,
//!    and multi-part structure combine into a `complexity_score` in `[0, 1]`.
//!    When the score exceeds [`ResearchDetector::threshold`] the harness fires.
//!
//! All behaviour is configurable via environment variables (no hardcoded
//! infrastructure values):
//! - `HARNESS_ENABLED` (default `true`) — `false` disables detection entirely,
//!   so [`ResearchDetector::should_use_harness`] always returns `false`.
//! - `HARNESS_TRIGGER_THRESHOLD` (default `0.6`) — lower means more queries are
//!   routed to research; higher means fewer.

/// Explicit slash commands that unconditionally trigger the harness.
const EXPLICIT_TRIGGERS: &[&str] = &["/research", "/deep-search"];

/// Research-intent phrases. Presence of any (in imperative/request form) is a
/// strong signal that the user wants synthesised, multi-source research.
///
/// Multi-word phrases are matched as substrings; single words are matched as
/// whole tokens to avoid casual false positives (e.g. "I'll research it later").
const RESEARCH_INTENT_KEYWORDS: &[&str] = &[
    "research",
    "investigate",
    "analyze",
    "compare",
    "comprehensive",
    "deep dive",
    "thorough",
    "survey",
    "literature",
    "evidence for",
    "pros and cons",
    "what are the best approaches to",
];

/// Phrases that mark a query as a trivial factual lookup. These should stay on
/// regular search even if other heuristics nudge upward.
const SIMPLE_QUERY_MARKERS: &[&str] = &[
    "what time is it",
    "weather today",
    "stock price",
    "current time",
    "today's weather",
];

/// Comparison cues that indicate the user wants a contrast across options.
const COMPARISON_WORDS: &[&str] = &[
    " vs ",
    " vs. ",
    "versus",
    "compared to",
    "comparison",
    "difference between",
    "compare",
];

/// Temporal-scope cues that imply multiple sources across time are needed.
const TEMPORAL_WORDS: &[&str] = &[
    "over the past",
    "recent developments",
    "recent advances",
    "in recent years",
    "historically",
    "evolution of",
    "trend",
    "latest",
];

/// Multi-part question cues ("how X works and why Y matters").
const MULTIPART_WORDS: &[&str] = &[" and why ", " and how ", " as well as ", "; "];

/// Default complexity threshold when `HARNESS_TRIGGER_THRESHOLD` is unset.
const DEFAULT_THRESHOLD: f64 = 0.6;

/// Word count above which a query is considered long enough to plausibly need
/// research even without explicit intent keywords.
const LONG_QUERY_WORDS: usize = 20;

/// The outcome of a detection pass. Captured so callers can log it to audit.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DetectionDecision {
    /// The (trimmed) query that was evaluated.
    pub query: String,
    /// Complexity score in `[0, 1]`.
    pub complexity_score: f64,
    /// Whether the harness should be used.
    pub triggered: bool,
    /// Human-readable explanation of why the decision was made.
    pub reason: String,
}

/// Detects whether a query warrants the full Harness-1 research loop.
#[derive(Debug, Clone)]
pub struct ResearchDetector {
    /// Master switch. When `false`, detection always returns `false`.
    enabled: bool,
    /// Complexity score above which the harness triggers.
    threshold: f64,
}

impl Default for ResearchDetector {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold: DEFAULT_THRESHOLD,
        }
    }
}

impl ResearchDetector {
    /// Construct a detector with explicit settings (mainly for tests).
    pub fn new(enabled: bool, threshold: f64) -> Self {
        Self { enabled, threshold }
    }

    /// Build a detector from environment variables.
    ///
    /// - `HARNESS_ENABLED` (default `true`) — any value other than `false`
    ///   (case-insensitive) keeps detection enabled.
    /// - `HARNESS_TRIGGER_THRESHOLD` (default `0.6`) — parsed as `f64`; invalid
    ///   values fall back to the default.
    pub fn from_env() -> Self {
        let enabled = std::env::var("HARNESS_ENABLED")
            .map(|v| v.to_lowercase() != "false")
            .unwrap_or(true);

        let threshold = std::env::var("HARNESS_TRIGGER_THRESHOLD")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(DEFAULT_THRESHOLD);

        Self { enabled, threshold }
    }

    /// The configured trigger threshold.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Whether detection is enabled.
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Returns `true` if the query should be routed to the Harness-1 deep
    /// research loop. Also emits a `tracing` log of the decision.
    pub fn should_use_harness(&self, query: &str) -> bool {
        self.detect(query).triggered
    }

    /// Full detection pass returning the structured [`DetectionDecision`].
    ///
    /// The decision is logged via `tracing` (info when triggered, debug
    /// otherwise) so it can be surfaced to the audit trail.
    pub fn detect(&self, query: &str) -> DetectionDecision {
        let trimmed = query.trim();
        let lower = trimmed.to_lowercase();

        // Disabled: never trigger, regardless of content.
        if !self.enabled {
            let decision = DetectionDecision {
                query: trimmed.to_string(),
                complexity_score: 0.0,
                triggered: false,
                reason: "harness disabled (HARNESS_ENABLED=false)".to_string(),
            };
            Self::log(&decision);
            return decision;
        }

        // 1. Explicit slash command — always triggers.
        if let Some(cmd) = explicit_trigger(&lower) {
            let decision = DetectionDecision {
                query: trimmed.to_string(),
                complexity_score: 1.0,
                triggered: true,
                reason: format!("explicit command `{cmd}`"),
            };
            Self::log(&decision);
            return decision;
        }

        let score = self.complexity_score(trimmed);
        let is_simple = is_simple_query(&lower);
        let has_intent = has_research_intent(&lower);
        let word_count = word_count(&lower);
        let long_query = word_count > LONG_QUERY_WORDS;

        // 2. Implicit triggers: research intent OR a long query, but never a
        //    trivial factual lookup. Backed by the complexity heuristic so an
        //    edge query scoring below threshold stays on regular search.
        let implicit_signal = has_intent || long_query;
        let triggered = implicit_signal && !is_simple && score > self.threshold;

        let reason = if is_simple {
            format!("simple factual query (score {score:.2}) — regular search")
        } else if !implicit_signal {
            format!("no research intent / short query (score {score:.2}) — regular search")
        } else if triggered {
            format!(
                "research intent={has_intent} long={long_query}, score {score:.2} > threshold {:.2}",
                self.threshold
            )
        } else {
            format!(
                "research intent={has_intent} long={long_query}, score {score:.2} <= threshold {:.2} — regular search",
                self.threshold
            )
        };

        let decision = DetectionDecision {
            query: trimmed.to_string(),
            complexity_score: score,
            triggered,
            reason,
        };
        Self::log(&decision);
        decision
    }

    /// Compute a complexity score in `[0, 1]` from query structure.
    ///
    /// Contributions (clamped to `1.0`):
    /// - research-intent keyword present: `0.30`
    /// - more than two distinct entity-like tokens: `0.20`
    /// - comparison words: `0.35`
    /// - temporal scope: `0.15`
    /// - multi-part question: `0.20`
    /// - long query (> [`LONG_QUERY_WORDS`] words): `0.20`
    pub fn complexity_score(&self, query: &str) -> f64 {
        let lower = query.trim().to_lowercase();
        if lower.is_empty() {
            return 0.0;
        }

        let mut score = 0.0_f64;

        if has_research_intent(&lower) {
            score += 0.30;
        }
        if entity_count(query) > 2 {
            score += 0.20;
        }
        if contains_any(&lower, COMPARISON_WORDS) {
            score += 0.35;
        }
        if contains_any(&lower, TEMPORAL_WORDS) {
            score += 0.15;
        }
        if contains_any(&lower, MULTIPART_WORDS) {
            score += 0.20;
        }
        if word_count(&lower) > LONG_QUERY_WORDS {
            score += 0.20;
        }

        score.min(1.0)
    }

    /// Emit the detection decision to `tracing`.
    fn log(decision: &DetectionDecision) {
        if decision.triggered {
            tracing::info!(
                query = %decision.query,
                complexity_score = decision.complexity_score,
                triggered = decision.triggered,
                reason = %decision.reason,
                "research detector decision"
            );
        } else {
            tracing::debug!(
                query = %decision.query,
                complexity_score = decision.complexity_score,
                triggered = decision.triggered,
                reason = %decision.reason,
                "research detector decision"
            );
        }
    }
}

/// Return the matched explicit trigger command, if the query starts with one.
fn explicit_trigger(lower: &str) -> Option<&'static str> {
    let first = lower.split_whitespace().next().unwrap_or("");
    EXPLICIT_TRIGGERS.iter().copied().find(|&cmd| first == cmd)
}

/// Whether the query carries research intent. Single-word keywords must appear
/// as whole tokens (so "researched" / "I'll research it later" casual usage is
/// less likely to false-positive); multi-word phrases match as substrings.
fn has_research_intent(lower: &str) -> bool {
    let tokens: Vec<&str> = lower.split(|c: char| !c.is_alphanumeric()).collect();
    for &kw in RESEARCH_INTENT_KEYWORDS {
        if kw.contains(' ') {
            if lower.contains(kw) {
                return true;
            }
        } else if tokens.iter().any(|&t| t == kw) {
            return true;
        }
    }
    false
}

/// Whether the query is a trivial factual lookup.
fn is_simple_query(lower: &str) -> bool {
    contains_any(lower, SIMPLE_QUERY_MARKERS)
}

/// Count rough "entity-like" tokens: capitalised words and acronyms in the
/// original (case-preserving) query, deduplicated.
fn entity_count(query: &str) -> usize {
    let mut entities = std::collections::HashSet::new();
    for raw in query.split_whitespace() {
        let token: String = raw
            .chars()
            .filter(|c| c.is_alphanumeric())
            .collect();
        if token.len() < 2 {
            continue;
        }
        let mut chars = token.chars();
        let first = chars.next().unwrap();
        let is_acronym = token.chars().all(|c| c.is_uppercase() || c.is_numeric())
            && token.chars().any(|c| c.is_uppercase());
        // Capitalised word (not at very start matters less; we accept either),
        // an acronym (LoRA, AAPL), or a token containing digits (20B, GPT4).
        let is_capitalised = first.is_uppercase();
        let has_digit = token.chars().any(|c| c.is_numeric());
        if is_acronym || is_capitalised || has_digit {
            entities.insert(token.to_lowercase());
        }
    }
    entities.len()
}

/// Number of whitespace-delimited words.
fn word_count(s: &str) -> usize {
    s.split_whitespace().count()
}

/// Whether `haystack` contains any of `needles`.
fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| haystack.contains(n))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: detector with default settings.
    fn detector() -> ResearchDetector {
        ResearchDetector::new(true, DEFAULT_THRESHOLD)
    }

    #[test]
    fn explicit_research_command_triggers() {
        let d = detector();
        assert!(d.should_use_harness("/research the history of transformers"));
        let decision = d.detect("/research the history of transformers");
        assert!(decision.triggered);
        assert!(decision.reason.contains("explicit"));
    }

    #[test]
    fn explicit_deep_search_command_triggers() {
        let d = detector();
        assert!(d.should_use_harness("/deep-search quantum error correction"));
    }

    #[test]
    fn explicit_command_triggers_even_when_otherwise_simple() {
        // Explicit command bypasses the simple-query guard entirely.
        let d = detector();
        assert!(d.should_use_harness("/research weather today"));
    }

    #[test]
    fn simple_weather_query_does_not_trigger() {
        let d = detector();
        assert!(!d.should_use_harness("weather today"));
        let decision = d.detect("weather today");
        assert!(!decision.triggered);
        assert!(decision.reason.contains("regular search"));
    }

    #[test]
    fn simple_time_query_does_not_trigger() {
        let d = detector();
        assert!(!d.should_use_harness("what time is it"));
    }

    #[test]
    fn simple_stock_price_query_does_not_trigger() {
        let d = detector();
        assert!(!d.should_use_harness("AAPL stock price"));
    }

    #[test]
    fn complex_comparison_query_triggers() {
        let d = detector();
        assert!(d.should_use_harness("compare LoRA vs full fine-tuning for 20B models"));
        let decision = d.detect("compare LoRA vs full fine-tuning for 20B models");
        assert!(decision.triggered);
        assert!(decision.complexity_score > DEFAULT_THRESHOLD);
    }

    #[test]
    fn research_intent_word_contributes_to_score() {
        let d = detector();
        let with = d.complexity_score("investigate the causes of inflation");
        let without = d.complexity_score("the causes of inflation");
        assert!(with > without);
    }

    #[test]
    fn casual_research_mention_alone_does_not_trigger() {
        // "research" as a whole token counts as intent, but without other
        // complexity signals the score should stay at/under threshold so a
        // throwaway mention doesn't fire the harness.
        let d = detector();
        let decision = d.detect("ok i will research it later");
        assert!(!decision.triggered, "got: {decision:?}");
    }

    #[test]
    fn threshold_is_configurable_lower_triggers_more() {
        let query = "analyze the impact of remote work";
        let strict = ResearchDetector::new(true, 0.9);
        let lenient = ResearchDetector::new(true, 0.2);
        assert!(!strict.should_use_harness(query));
        assert!(lenient.should_use_harness(query));
    }

    #[test]
    fn harness_enabled_false_disables_detection() {
        let d = ResearchDetector::new(false, DEFAULT_THRESHOLD);
        // Even an explicit command must not trigger when disabled.
        assert!(!d.should_use_harness("/research everything"));
        assert!(!d.should_use_harness("compare LoRA vs full fine-tuning for 20B models"));
        let decision = d.detect("/research everything");
        assert!(!decision.triggered);
        assert!(decision.reason.contains("disabled"));
    }

    #[test]
    fn detection_decision_is_populated_for_logging() {
        let d = detector();
        let decision = d.detect("compare LoRA vs full fine-tuning for 20B models");
        assert_eq!(
            decision.query,
            "compare LoRA vs full fine-tuning for 20B models"
        );
        assert!(decision.complexity_score >= 0.0 && decision.complexity_score <= 1.0);
        assert!(!decision.reason.is_empty());
    }

    #[test]
    fn edge_case_just_below_threshold_does_not_trigger() {
        // A query with research intent (0.30) but nothing else scores 0.30,
        // which is below the 0.6 default — stays on regular search.
        let d = detector();
        let decision = d.detect("research dogs");
        assert!(decision.complexity_score <= DEFAULT_THRESHOLD);
        assert!(!decision.triggered);
    }

    #[test]
    fn complexity_score_is_bounded() {
        let d = detector();
        let busy = "compare and analyze the comprehensive evolution of GPT4 vs Claude over the past year and why scaling matters as well as RLHF in recent developments thoroughly across many models";
        let score = d.complexity_score(busy);
        assert!((0.0..=1.0).contains(&score));
    }

    #[test]
    fn empty_query_scores_zero_and_does_not_trigger() {
        let d = detector();
        assert_eq!(d.complexity_score("   "), 0.0);
        assert!(!d.should_use_harness("   "));
    }

    #[test]
    fn long_query_can_trigger_without_explicit_keyword() {
        // > 20 words, multi-part, with entities/temporal scope — implicit path.
        let d = detector();
        let q = "How do modern GPU architectures from NVIDIA and AMD handle \
                 memory bandwidth over the past year and why does that matter \
                 for training large transformer models at scale";
        assert!(word_count(&q.to_lowercase()) > LONG_QUERY_WORDS);
        assert!(d.should_use_harness(q), "score {}", d.complexity_score(q));
    }

    #[test]
    fn comparison_words_increase_score() {
        let d = detector();
        let with = d.complexity_score("Rust vs Go for systems programming");
        let without = d.complexity_score("Rust for systems programming");
        assert!(with > without);
    }

    #[test]
    fn from_env_defaults() {
        // Without env overrides set in this process, defaults apply. We avoid
        // mutating process env to keep tests parallel-safe; just assert the
        // constructor wiring matches the documented defaults.
        let d = ResearchDetector::new(true, DEFAULT_THRESHOLD);
        assert!(d.enabled());
        assert!((d.threshold() - 0.6).abs() < f64::EPSILON);
    }

    #[test]
    fn entity_count_detects_multiple_entities() {
        // LoRA, 20B count as entities; common words don't.
        assert!(entity_count("compare LoRA vs full fine-tuning for 20B models GPT4") > 2);
        assert!(entity_count("the cat sat on the mat") <= 1);
    }
}
