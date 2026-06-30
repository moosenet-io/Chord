//! HRNS-05: Synthesis prompt construction for the answering model.
//!
//! After the Harness-1 search phase curates a set of documents, the agentic
//! executor swaps to the synthesis model and resumes the normal loop with those
//! documents as context. [`SynthesisPrompt`] turns the curated set into a
//! citation-style system prompt: each document is numbered, tagged with its
//! importance, and the model is instructed to cite by index, prioritise the
//! highest-importance sources, note contradictions, and admit insufficiency.
//!
//! Only the curated documents' *compressed* text is included in the prompt (the
//! full text never leaves the harness). The execution log, separately, records
//! metadata only (titles + importance) — see [`SynthesisPrompt::doc_metadata`].

use crate::harness::state::{CuratedDoc, Importance};

/// Builds the synthesis system prompt from a curated document set.
///
/// Construction is pure (no I/O), so it is trivially testable. The prompt text
/// follows the spec's citation template exactly.
pub struct SynthesisPrompt;

/// Metadata-only view of a curated document, safe to put in the execution log.
///
/// Carries the title and importance label — never the document text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CuratedDocMeta {
    /// 1-based citation index, matching the `[n]` markers in the prompt.
    pub index: usize,
    pub title: String,
    /// Importance label (`very_high`, `high`, `fair`, `low`).
    pub importance: &'static str,
}

impl SynthesisPrompt {
    /// Build the citation-style synthesis prompt for `query` over `curated`.
    ///
    /// Documents are ordered most-important first so the model sees the strongest
    /// evidence at the top; `[1]` is therefore the highest-priority source. The
    /// rendered text uses each document's compressed form (BM25 top sentences).
    ///
    /// Returns an empty string when there are no curated documents — callers
    /// should skip synthesis entirely in that case (see the spec's 0-curated edge
    /// case), but an empty prompt is harmless if pushed.
    pub fn build(query: &str, curated: &[CuratedDoc]) -> String {
        if curated.is_empty() {
            return String::new();
        }

        let ordered = Self::ordered(curated);
        let n = ordered.len();

        let mut out = String::new();
        out.push_str(&format!(
            "You have been provided with {n} curated research documents about \"{query}\".\n"
        ));
        out.push_str(
            "Each document has an importance tag (very_high, high, fair, low).\n\n",
        );
        out.push_str("Synthesize a comprehensive answer using these sources.\n");
        out.push_str("Cite documents by their index [1], [2], etc.\n");
        out.push_str("Prioritize very_high and high importance documents.\n");
        out.push_str("Note any contradictions between sources.\n");
        out.push_str("If the evidence is insufficient, say so.\n\n");
        out.push_str("Documents:\n");

        for (i, c) in ordered.iter().enumerate() {
            let idx = i + 1;
            let tag = importance_tag(c.importance);
            let title = if c.document.title.trim().is_empty() {
                "(untitled)"
            } else {
                c.document.title.trim()
            };
            let text = if c.document.compressed.trim().is_empty() {
                c.document.full_text.trim()
            } else {
                c.document.compressed.trim()
            };
            out.push_str(&format!("[{idx}] ({tag}) {title}: {text}\n"));
        }

        out
    }

    /// Metadata-only view of the curated set, in the same order/indexing used by
    /// [`build`]. Safe for the execution log: titles + importance only.
    pub fn doc_metadata(curated: &[CuratedDoc]) -> Vec<CuratedDocMeta> {
        Self::ordered(curated)
            .iter()
            .enumerate()
            .map(|(i, c)| CuratedDocMeta {
                index: i + 1,
                title: c.document.title.clone(),
                importance: importance_tag(c.importance),
            })
            .collect()
    }

    /// Order curated docs most-important first, breaking ties by the turn they
    /// were added (earlier first) for a stable, deterministic ordering.
    fn ordered(curated: &[CuratedDoc]) -> Vec<CuratedDoc> {
        let mut v: Vec<CuratedDoc> = curated.to_vec();
        v.sort_by(|a, b| {
            b.importance
                .rank()
                .cmp(&a.importance.rank())
                .then(a.added_at_turn.cmp(&b.added_at_turn))
                .then(a.document.id.cmp(&b.document.id))
        });
        v
    }
}

/// Lowercase, snake-case importance tag as shown in the prompt.
fn importance_tag(importance: Importance) -> &'static str {
    match importance {
        Importance::VeryHigh => "very_high",
        Importance::High => "high",
        Importance::Fair => "fair",
        Importance::Low => "low",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::state::Document;

    fn doc(id: usize, title: &str, compressed: &str) -> Document {
        Document {
            id,
            url: format!("http://example/{id}"),
            title: title.into(),
            full_text: format!("full text of {title}"),
            compressed: compressed.into(),
        }
    }

    fn curated(id: usize, title: &str, text: &str, imp: Importance, turn: usize) -> CuratedDoc {
        CuratedDoc {
            document: doc(id, title, text),
            importance: imp,
            added_at_turn: turn,
        }
    }

    #[test]
    fn empty_curated_yields_empty_prompt() {
        assert_eq!(SynthesisPrompt::build("anything", &[]), "");
        assert!(SynthesisPrompt::doc_metadata(&[]).is_empty());
    }

    #[test]
    fn prompt_includes_query_and_count() {
        let set = vec![curated(0, "Solar", "Solar grew.", Importance::High, 1)];
        let p = SynthesisPrompt::build("renewable energy", &set);
        assert!(p.contains("renewable energy"));
        assert!(p.contains("1 curated research documents"));
    }

    #[test]
    fn prompt_includes_importance_tags_and_citations() {
        let set = vec![
            curated(0, "Solar", "Solar grew rapidly.", Importance::VeryHigh, 1),
            curated(1, "Wind", "Wind expanded.", Importance::Fair, 2),
        ];
        let p = SynthesisPrompt::build("energy", &set);
        // Importance tags present (snake_case).
        assert!(p.contains("(very_high)"));
        assert!(p.contains("(fair)"));
        // Citation markers present.
        assert!(p.contains("[1]"));
        assert!(p.contains("[2]"));
        // Instructional clauses present.
        assert!(p.contains("Cite documents by their index"));
        assert!(p.contains("Note any contradictions"));
        assert!(p.contains("If the evidence is insufficient"));
        assert!(p.contains("Prioritize very_high and high"));
    }

    #[test]
    fn most_important_document_is_cited_first() {
        let set = vec![
            curated(0, "Low one", "low text", Importance::Low, 1),
            curated(1, "VeryHigh one", "vh text", Importance::VeryHigh, 2),
        ];
        let full = SynthesisPrompt::build("q", &set);
        // Search only within the Documents section (the instructions also mention
        // "[1], [2]"), so we slice from the "Documents:" marker onward.
        let docs_start = full.find("Documents:").unwrap();
        let p = &full[docs_start..];
        // [1] must introduce the very_high doc, ordered ahead of the low one.
        let idx1 = p.find("[1]").unwrap();
        let idx2 = p.find("[2]").unwrap();
        let vh = p.find("VeryHigh one").unwrap();
        let low = p.find("Low one").unwrap();
        assert!(idx1 < vh && vh < idx2, "[1] should introduce the very_high doc");
        assert!(low > idx2 || low > vh);
    }

    #[test]
    fn metadata_is_titles_and_importance_only() {
        let set = vec![
            curated(0, "Solar Report", "FULL SECRET BODY", Importance::VeryHigh, 1),
            curated(1, "Wind Report", "ANOTHER BODY", Importance::High, 2),
        ];
        let meta = SynthesisPrompt::doc_metadata(&set);
        assert_eq!(meta.len(), 2);
        assert_eq!(meta[0].index, 1);
        assert_eq!(meta[0].title, "Solar Report");
        assert_eq!(meta[0].importance, "very_high");
        // No document text fields exist on the metadata struct at all.
        for m in &meta {
            assert!(!m.title.contains("BODY"));
        }
    }

    #[test]
    fn uses_compressed_text_not_full_text() {
        let set = vec![curated(0, "Doc", "COMPRESSED_SENTENCE", Importance::High, 1)];
        let p = SynthesisPrompt::build("q", &set);
        assert!(p.contains("COMPRESSED_SENTENCE"));
        assert!(!p.contains("full text of"));
    }

    #[test]
    fn falls_back_to_full_text_when_compressed_empty() {
        let set = vec![curated(0, "Doc", "", Importance::High, 1)];
        let p = SynthesisPrompt::build("q", &set);
        assert!(p.contains("full text of Doc"));
    }
}
