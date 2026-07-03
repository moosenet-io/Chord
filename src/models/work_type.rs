//! CPROX-01: the "work-type code" request shape.
//!
//! Today Harmony picks a coding model by a hardcoded name. [`WorkTypeCode`] is
//! the tagged request Harmony will send instead — "I need CODE work of this
//! shape" — so Chord (CPROX-02's [`crate::models::coding_selector`]) can pick
//! the best REAL model from measured fleet data rather than a fixed alias.
//!
//! Every field is a closed enum (not a free-form string) so a malformed/unknown
//! request fails deserialization cleanly at the edge — never a panic, never a
//! silent "assume default" guess deep in the matching engine.

use serde::{Deserialize, Serialize};

/// Programming language the work is in. Mirrors the language tag already used
/// by the coder-sweep harness's `code_profile_runs.language` column (see
/// `terminus_rs::intake::code_v2`), so [`to_query_key`](WorkTypeCode::to_query_key)
/// lines up with the sweep data without any translation table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    Bash,
    Python,
    Rust,
    TypeScript,
}

impl Language {
    /// The lowercase tag as stored in `code_profile_runs.language` /
    /// `intake_corpus_v2` case manifests (e.g. `"typescript"`, not `"TypeScript"`).
    pub fn as_str(self) -> &'static str {
        match self {
            Language::Bash => "bash",
            Language::Python => "python",
            Language::Rust => "rust",
            Language::TypeScript => "typescript",
        }
    }
}

/// The shape of the coding task — a rough proxy for how much of the codebase
/// context a model needs to hold at once. Distinct from `reasoning_need`:
/// a `QuickEdit` can still need `Review`-level reasoning (e.g. a one-line
/// security fix), and a `MultiFileBuild` can be pure `Execute` (mechanical
/// scaffolding across files).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskShape {
    /// A small, localized change: one file, a handful of lines.
    QuickEdit,
    /// A change that spans multiple files or needs a whole-workspace view
    /// (matches the coder-sweep's `MultiFileBuild`-style "build_modify" cases).
    MultiFileBuild,
}

/// What kind of thinking the model needs to do, independent of task size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningNeed {
    /// Decompose a goal into steps before writing any code.
    Plan,
    /// Add to / extend existing, working code.
    Enrich,
    /// Read code and judge correctness/quality without necessarily writing any.
    Review,
    /// Mechanically carry out an already-specified change.
    Execute,
}

/// How much context the model must hold to do the work well. Distinct from
/// `task_shape`: a `QuickEdit` in a huge generated file can still need `Long`
/// context just to locate the edit point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ContextDepthNeed {
    Short,
    Long,
}

/// A tagged description of a coding work item, as Harmony sends it to Chord's
/// coding-proxy endpoint (CPROX-03) instead of a hardcoded model name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorkTypeCode {
    pub language: Language,
    pub task_shape: TaskShape,
    pub reasoning_need: ReasoningNeed,
    pub context_depth_need: ContextDepthNeed,
}

impl WorkTypeCode {
    /// Normalize this request into a stable, lowercase, `|`-joined key for the
    /// matching engine (CPROX-02) to use as a cache/log key. NOT a database
    /// query filter by itself — CPROX-02 queries `code_profile_runs` by
    /// `language` alone (the sweep data does not yet have per-task-shape /
    /// per-reasoning-need breakdowns; see `coding_selector` module docs) and
    /// uses the other three fields only to adjust ranking/preferences.
    pub fn to_query_key(&self) -> String {
        format!(
            "{}|{:?}|{:?}|{:?}",
            self.language.as_str(),
            self.task_shape,
            self.reasoning_need,
            self.context_depth_need
        )
        .to_ascii_lowercase()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_LANGUAGES: [Language; 4] =
        [Language::Bash, Language::Python, Language::Rust, Language::TypeScript];
    const ALL_TASK_SHAPES: [TaskShape; 2] = [TaskShape::QuickEdit, TaskShape::MultiFileBuild];
    const ALL_REASONING_NEEDS: [ReasoningNeed; 4] = [
        ReasoningNeed::Plan,
        ReasoningNeed::Enrich,
        ReasoningNeed::Review,
        ReasoningNeed::Execute,
    ];
    const ALL_CONTEXT_DEPTHS: [ContextDepthNeed; 2] =
        [ContextDepthNeed::Short, ContextDepthNeed::Long];

    #[test]
    fn every_variant_combination_round_trips_through_json() {
        for language in ALL_LANGUAGES {
            for task_shape in ALL_TASK_SHAPES {
                for reasoning_need in ALL_REASONING_NEEDS {
                    for context_depth_need in ALL_CONTEXT_DEPTHS {
                        let wtc = WorkTypeCode {
                            language,
                            task_shape,
                            reasoning_need,
                            context_depth_need,
                        };
                        let json = serde_json::to_string(&wtc).expect("serializes");
                        let back: WorkTypeCode =
                            serde_json::from_str(&json).expect("deserializes");
                        assert_eq!(wtc, back, "round trip mismatch for {json}");
                    }
                }
            }
        }
    }

    #[test]
    fn field_wire_shapes_are_stable() {
        let wtc = WorkTypeCode {
            language: Language::Rust,
            task_shape: TaskShape::MultiFileBuild,
            reasoning_need: ReasoningNeed::Enrich,
            context_depth_need: ContextDepthNeed::Long,
        };
        let v: serde_json::Value = serde_json::to_value(&wtc).unwrap();
        assert_eq!(v["language"], "rust");
        assert_eq!(v["task_shape"], "multi_file_build");
        assert_eq!(v["reasoning_need"], "enrich");
        assert_eq!(v["context_depth_need"], "long");
    }

    #[test]
    fn unknown_enum_value_fails_cleanly_not_a_panic() {
        let bad = r#"{"language":"cobol","task_shape":"quick_edit",
            "reasoning_need":"plan","context_depth_need":"short"}"#;
        let result: Result<WorkTypeCode, _> = serde_json::from_str(bad);
        assert!(result.is_err(), "unknown language variant must be rejected, not guessed");
    }

    #[test]
    fn missing_field_fails_cleanly() {
        let bad = r#"{"language":"rust","task_shape":"quick_edit","reasoning_need":"plan"}"#;
        let result: Result<WorkTypeCode, _> = serde_json::from_str(bad);
        assert!(result.is_err());
    }

    #[test]
    fn truncated_json_fails_cleanly() {
        let bad = r#"{"language":"rust","task_shape":"#;
        let result: Result<WorkTypeCode, _> = serde_json::from_str(bad);
        assert!(result.is_err());
    }

    #[test]
    fn wrong_type_for_field_fails_cleanly() {
        let bad = r#"{"language":123,"task_shape":"quick_edit",
            "reasoning_need":"plan","context_depth_need":"short"}"#;
        let result: Result<WorkTypeCode, _> = serde_json::from_str(bad);
        assert!(result.is_err());
    }

    #[test]
    fn to_query_key_is_stable_and_lowercase() {
        let wtc = WorkTypeCode {
            language: Language::TypeScript,
            task_shape: TaskShape::QuickEdit,
            reasoning_need: ReasoningNeed::Review,
            context_depth_need: ContextDepthNeed::Short,
        };
        let key = wtc.to_query_key();
        assert_eq!(key, key.to_ascii_lowercase());
        assert!(key.contains("typescript"));
        assert!(key.contains("quickedit"));
        assert!(key.contains("review"));
        assert!(key.contains("short"));
    }

    #[test]
    fn language_as_str_matches_sweep_language_tags() {
        // These MUST match `code_profile_runs.language` values exactly (see
        // terminus_rs::intake::code_v2 case manifests) — CPROX-02 filters on this.
        assert_eq!(Language::Bash.as_str(), "bash");
        assert_eq!(Language::Python.as_str(), "python");
        assert_eq!(Language::Rust.as_str(), "rust");
        assert_eq!(Language::TypeScript.as_str(), "typescript");
    }
}
