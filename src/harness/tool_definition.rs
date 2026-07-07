//! HRNS-06: the `deep_research` tool — definition, discovery, and dispatch glue.
//!
//! `deep_research` is a *synthetic* Chord tool (it is not an MCP/fallback tool):
//! the agentic executor advertises it in the always-on tool set so the LLM can
//! always discover it, and recognises a call to it as the trigger for the
//! Harness-1 research flow wired in HRNS-05.
//!
//! This module owns:
//! - the canonical tool name [`DEEP_RESEARCH_TOOL`],
//! - the tool's JSON-Schema parameters and human description,
//! - the [`Depth`] enum (`standard` ⇒ 20 turns, `thorough` ⇒ 40 turns) and the
//!   logic that maps a call's `depth` argument to a harness turn budget.
//!
//! The dispatch itself (running the harness) lives in the agentic loop; this
//! module supplies the pure, well-tested pieces it threads together.

use serde_json::{json, Value};

/// The canonical tool name the LLM calls to request deep research.
///
/// Kept in sync with `agentic::loop_runner::DEEP_RESEARCH_TOOL` (both must name
/// the same tool); a unit test asserts the two constants agree.
pub const DEEP_RESEARCH_TOOL: &str = "deep_research";

/// Max harness turns for `depth = "standard"` (per spec).
pub const STANDARD_MAX_TURNS: usize = 20;
/// Max harness turns for `depth = "thorough"` (per spec).
pub const THOROUGH_MAX_TURNS: usize = 40;

/// How thoroughly the harness should investigate. Controls the harness turn
/// budget threaded into the HRNS-05 research flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Depth {
    /// Up to [`STANDARD_MAX_TURNS`] harness turns. The default.
    Standard,
    /// Up to [`THOROUGH_MAX_TURNS`] harness turns.
    Thorough,
}

impl Default for Depth {
    fn default() -> Self {
        Depth::Standard
    }
}

impl Depth {
    /// The wire token (`"standard"` / `"thorough"`).
    pub fn as_str(self) -> &'static str {
        match self {
            Depth::Standard => "standard",
            Depth::Thorough => "thorough",
        }
    }

    /// The max harness turn budget this depth requests.
    pub fn max_turns(self) -> usize {
        match self {
            Depth::Standard => STANDARD_MAX_TURNS,
            Depth::Thorough => THOROUGH_MAX_TURNS,
        }
    }

    /// Parse a `depth` token. Unknown / missing tokens fall back to the default
    /// (`Standard`) rather than erroring — a bad depth wastes nothing and must
    /// not break the call.
    pub fn from_token(token: &str) -> Depth {
        match token.trim().to_ascii_lowercase().as_str() {
            "thorough" => Depth::Thorough,
            _ => Depth::Standard,
        }
    }
}

/// The tool's human-facing description (steers the LLM toward correct use).
pub const DESCRIPTION: &str = "Conduct deep multi-source research on a topic. \
Uses a specialized search agent to find, curate, and verify evidence across \
multiple sources. Returns a comprehensive, cited analysis. Use for complex \
questions requiring multiple perspectives or thorough investigation. NOT for \
simple lookups — use searxng_search for quick facts.";

/// The tool's JSON-Schema `parameters` object.
pub fn parameters() -> Value {
    json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The research question or topic"
            },
            "depth": {
                "type": "string",
                "enum": ["standard", "thorough"],
                "default": Depth::default().as_str(),
                "description": "standard: up to 20 turns. thorough: up to 40 turns."
            }
        },
        "required": ["query"]
    })
}

/// Extract the `depth` from a tool-call arguments object, defaulting when the
/// field is absent or not a string.
pub fn depth_from_args(args: &Value) -> Depth {
    match args.get("depth").and_then(Value::as_str) {
        Some(tok) => Depth::from_token(tok),
        None => Depth::default(),
    }
}

/// Extract the `query` from a tool-call arguments object (empty if absent).
pub fn query_from_args(args: &Value) -> String {
    args.get("query")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

/// True if `name` is the `deep_research` tool.
pub fn is_deep_research(name: &str) -> bool {
    name == DEEP_RESEARCH_TOOL
}

/// The max harness turns a tool-call with these arguments requests.
///
/// This is the single point that maps the discoverable `depth` parameter onto a
/// harness turn budget; the agentic loop passes the result through to
/// `run_research` as an explicit override.
pub fn max_turns_for_args(args: &Value) -> usize {
    depth_from_args(args).max_turns()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_matches_loop_runner_constant() {
        assert_eq!(
            DEEP_RESEARCH_TOOL,
            crate::agentic::loop_runner::DEEP_RESEARCH_TOOL
        );
    }

    #[test]
    fn definition_is_valid_and_discoverable() {
        // A discoverable tool needs a non-empty name + description and an
        // object-typed parameter schema with the documented properties.
        assert_eq!(DEEP_RESEARCH_TOOL, "deep_research");
        assert!(!DESCRIPTION.is_empty());
        assert!(DESCRIPTION.contains("searxng_search")); // steers away from simple lookups

        let p = parameters();
        assert_eq!(p["type"], "object");
        assert_eq!(p["properties"]["query"]["type"], "string");
        assert_eq!(p["properties"]["depth"]["type"], "string");
        assert_eq!(
            p["properties"]["depth"]["enum"],
            json!(["standard", "thorough"])
        );
        assert_eq!(p["properties"]["depth"]["default"], "standard");
        assert_eq!(p["required"], json!(["query"]));

        // No hardcoded infrastructure values leak into the schema/description.
        let blob = format!("{DESCRIPTION}{p}");
        assert!(!blob.contains("192.168."));
        assert!(!blob.contains("http://"));
    }

    #[test]
    fn depth_controls_max_turns() {
        assert_eq!(Depth::Standard.max_turns(), 20);
        assert_eq!(Depth::Thorough.max_turns(), 40);
        assert_eq!(Depth::default(), Depth::Standard);
    }

    #[test]
    fn depth_parsing_is_lenient() {
        assert_eq!(Depth::from_token("thorough"), Depth::Thorough);
        assert_eq!(Depth::from_token("THOROUGH"), Depth::Thorough);
        assert_eq!(Depth::from_token(" thorough "), Depth::Thorough);
        assert_eq!(Depth::from_token("standard"), Depth::Standard);
        // Unknown / garbage → default (does not break the call).
        assert_eq!(Depth::from_token("turbo"), Depth::Standard);
        assert_eq!(Depth::from_token(""), Depth::Standard);
    }

    #[test]
    fn args_extraction() {
        let a = json!({"query": "why is the sky blue", "depth": "thorough"});
        assert_eq!(query_from_args(&a), "why is the sky blue");
        assert_eq!(depth_from_args(&a), Depth::Thorough);
        assert_eq!(max_turns_for_args(&a), 40);

        // Missing depth → standard / 20.
        let b = json!({"query": "q"});
        assert_eq!(depth_from_args(&b), Depth::Standard);
        assert_eq!(max_turns_for_args(&b), 20);

        // Missing query → empty.
        let c = json!({"depth": "standard"});
        assert_eq!(query_from_args(&c), "");
    }

    #[test]
    fn recognises_tool_name() {
        assert!(is_deep_research("deep_research"));
        assert!(!is_deep_research("searxng_search"));
    }
}
