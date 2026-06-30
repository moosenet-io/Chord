//! Harness actions — the structured action vocabulary the model emits, one per
//! turn. The harness executes the action, updates working memory, and renders
//! the next observation.

use crate::harness::state::Importance;
use serde::{Deserialize, Serialize};

/// A single structured action emitted by the model each turn.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum HarnessAction {
    /// Parallel searches across multiple queries; auto-seeds the curated set on
    /// the first search.
    FanOutSearch { queries: Vec<String> },
    /// Single search; results added to the candidate pool.
    SearchCorpus { query: String },
    /// Regex search across the text of documents in the candidate pool.
    GrepCorpus { pattern: String },
    /// Fetch full text for a candidate document and compress it.
    ReadDocument { doc_id: usize },
    /// Present summaries of the given documents for assessment.
    ReviewDocs { doc_ids: Vec<usize> },
    /// Add/update a document in the curated set with an importance tag.
    Curate { doc_id: usize, importance: Importance },
    /// Search for evidence for/against a claim and record a verification.
    Verify { claim: String },
    /// Terminate the search loop.
    EndSearch,
}

impl HarnessAction {
    /// Short name for rendering recent-action history.
    pub fn name(&self) -> &'static str {
        match self {
            HarnessAction::FanOutSearch { .. } => "fan_out_search",
            HarnessAction::SearchCorpus { .. } => "search_corpus",
            HarnessAction::GrepCorpus { .. } => "grep_corpus",
            HarnessAction::ReadDocument { .. } => "read_document",
            HarnessAction::ReviewDocs { .. } => "review_docs",
            HarnessAction::Curate { .. } => "curate",
            HarnessAction::Verify { .. } => "verify",
            HarnessAction::EndSearch => "end_search",
        }
    }

    /// Human-readable list of available actions, shown when the model emits an
    /// invalid action.
    pub fn available() -> &'static str {
        "fan_out_search(queries), search_corpus(query), grep_corpus(pattern), \
         read_document(doc_id), review_docs(doc_ids), curate(doc_id, importance), \
         verify(claim), end_search"
    }

    /// Parse an action from a JSON value. Returns a descriptive error on
    /// failure so the harness can surface available actions to the model.
    pub fn from_json(value: &serde_json::Value) -> Result<Self, String> {
        serde_json::from_value(value.clone())
            .map_err(|e| format!("invalid action: {e}. Available: {}", Self::available()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_json() {
        let actions = vec![
            HarnessAction::FanOutSearch {
                queries: vec!["a".into(), "b".into()],
            },
            HarnessAction::SearchCorpus { query: "x".into() },
            HarnessAction::GrepCorpus {
                pattern: "foo".into(),
            },
            HarnessAction::ReadDocument { doc_id: 3 },
            HarnessAction::ReviewDocs {
                doc_ids: vec![1, 2],
            },
            HarnessAction::Curate {
                doc_id: 1,
                importance: Importance::High,
            },
            HarnessAction::Verify {
                claim: "claim".into(),
            },
            HarnessAction::EndSearch,
        ];
        for a in actions {
            let json = serde_json::to_value(&a).unwrap();
            let back = HarnessAction::from_json(&json).unwrap();
            assert_eq!(a, back);
        }
    }

    #[test]
    fn invalid_action_reports_available() {
        let json = serde_json::json!({"action": "teleport"});
        let err = HarnessAction::from_json(&json).unwrap_err();
        assert!(err.contains("Available"));
        assert!(err.contains("end_search"));
    }

    #[test]
    fn end_search_parses_without_fields() {
        let json = serde_json::json!({"action": "end_search"});
        assert_eq!(
            HarnessAction::from_json(&json).unwrap(),
            HarnessAction::EndSearch
        );
    }
}
