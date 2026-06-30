//! Working memory for a single search episode.
//!
//! The harness holds all bookkeeping (candidate pool, curated set, evidence
//! graph, verification records, search history, budget) so the model only ever
//! sees a compact rendered observation and decides the next action.

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

/// Maximum number of documents retained in the curated set.
pub const CURATED_CAP: usize = 30;
/// Default maximum number of turns per search episode.
pub const DEFAULT_MAX_TURNS: usize = 40;
/// Environment variable that overrides the per-episode turn budget.
pub const MAX_TURNS_ENV: &str = "HARNESS_MAX_TURNS";

/// A retrieved document. Full text is retained for `ReadDocument`; the
/// `compressed` form (BM25 top sentences) is what the model is shown.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Document {
    pub id: usize,
    pub url: String,
    pub title: String,
    /// Full document text (may be large; stored, not shown).
    pub full_text: String,
    /// Compressed form shown to the model (top sentences).
    pub compressed: String,
}

/// Importance tag applied when a document is curated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Importance {
    VeryHigh,
    High,
    Fair,
    Low,
}

impl Importance {
    /// Ordering weight (higher = more important). Used when the curated set is
    /// full and the least-important document must be demoted.
    pub fn rank(&self) -> u8 {
        match self {
            Importance::VeryHigh => 3,
            Importance::High => 2,
            Importance::Fair => 1,
            Importance::Low => 0,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Importance::VeryHigh => "VeryHigh",
            Importance::High => "High",
            Importance::Fair => "Fair",
            Importance::Low => "Low",
        }
    }
}

/// A document promoted into the curated set with an importance tag.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CuratedDoc {
    pub document: Document,
    pub importance: Importance,
    pub added_at_turn: usize,
}

/// A recorded search query, used to avoid repeats.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchQuery {
    pub query: String,
    pub turn: usize,
    pub result_count: usize,
}

/// A claim checked against retrieved sources.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Verification {
    pub claim: String,
    /// Document ids that support the claim.
    pub supporting: Vec<usize>,
    /// Document ids that contradict the claim.
    pub contradicting: Vec<usize>,
    pub turn: usize,
}

/// Turn / document budget for a search episode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchBudget {
    pub turns_used: usize,
    pub max_turns: usize,
    pub docs_retrieved: usize,
}

impl SearchBudget {
    pub fn new(max_turns: usize) -> Self {
        Self {
            turns_used: 0,
            max_turns,
            docs_retrieved: 0,
        }
    }

    /// Read the configured max turns from the environment, falling back to the
    /// default. Never reads hardcoded infrastructure values.
    pub fn max_turns_from_env() -> usize {
        std::env::var(MAX_TURNS_ENV)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(DEFAULT_MAX_TURNS)
    }

    pub fn turns_remaining(&self) -> usize {
        self.max_turns.saturating_sub(self.turns_used)
    }

    pub fn exhausted(&self) -> bool {
        self.turns_used >= self.max_turns
    }

    /// Consume one turn.
    pub fn spend_turn(&mut self) {
        self.turns_used = self.turns_used.saturating_add(1);
    }
}

/// Evidence graph: entities (proper nouns, years, dates) extracted from
/// documents, with frequency tracking, bridge-document detection (docs that
/// contain 2+ frequent entities) and singleton detection (entities appearing
/// in only one document — follow-up leads).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvidenceGraph {
    /// entity -> set of document ids in which it appears.
    pub entity_docs: HashMap<String, HashSet<usize>>,
    /// Entities that have already been searched (used for "uncovered leads").
    pub searched_entities: HashSet<String>,
}

/// Compiled entity-extraction patterns, initialised once.
struct EntityPatterns {
    proper_noun: Regex,
    year_or_date: Regex,
}

fn entity_patterns() -> &'static EntityPatterns {
    static PATTERNS: OnceLock<EntityPatterns> = OnceLock::new();
    PATTERNS.get_or_init(|| EntityPatterns {
        // One or more capitalised words (proper noun phrases). Captures runs
        // like "United Nations" or "Alan Turing".
        proper_noun: Regex::new(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b")
            .expect("proper_noun regex"),
        // 4-digit years and ISO-ish dates.
        year_or_date: Regex::new(r"\b(\d{4}-\d{2}-\d{2}|(?:1[5-9]|20)\d{2})\b")
            .expect("year_or_date regex"),
    })
}

/// Threshold (inclusive) for an entity to be considered "frequent".
const FREQUENT_THRESHOLD: usize = 2;

impl EvidenceGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Extract entities from a single text blob (deduplicated within the text).
    pub fn extract_entities(text: &str) -> HashSet<String> {
        let patterns = entity_patterns();
        let mut set = HashSet::new();
        for cap in patterns.proper_noun.captures_iter(text) {
            if let Some(m) = cap.get(1) {
                let ent = m.as_str().trim();
                // Drop single very short tokens (e.g. "A", "I") and pure
                // sentence-initial noise of length 1.
                if ent.len() > 1 {
                    set.insert(ent.to_string());
                }
                // Also index the individual capitalised tokens of a multi-word
                // proper-noun phrase, so e.g. "Apollo Program" and "Apollo
                // continued" both contribute the shared entity "Apollo". Skip
                // very short tokens (initials/articles).
                let words: Vec<&str> = ent.split_whitespace().collect();
                if words.len() > 1 {
                    for w in words {
                        if w.len() > 2 {
                            set.insert(w.to_string());
                        }
                    }
                }
            }
        }
        for cap in patterns.year_or_date.captures_iter(text) {
            if let Some(m) = cap.get(1) {
                set.insert(m.as_str().to_string());
            }
        }
        set
    }

    /// Ingest a document's text under its id.
    pub fn ingest(&mut self, doc_id: usize, text: &str) {
        for entity in Self::extract_entities(text) {
            self.entity_docs.entry(entity).or_default().insert(doc_id);
        }
    }

    /// Entities appearing in `FREQUENT_THRESHOLD` or more documents.
    pub fn frequent_entities(&self) -> Vec<(String, usize)> {
        let mut v: Vec<(String, usize)> = self
            .entity_docs
            .iter()
            .filter(|(_, docs)| docs.len() >= FREQUENT_THRESHOLD)
            .map(|(e, docs)| (e.clone(), docs.len()))
            .collect();
        // Most frequent first, then alphabetical for stable rendering.
        v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        v
    }

    /// Document ids that contain 2+ frequent entities (bridge documents).
    pub fn bridge_docs(&self) -> Vec<usize> {
        let frequent: HashSet<&String> = self
            .entity_docs
            .iter()
            .filter(|(_, docs)| docs.len() >= FREQUENT_THRESHOLD)
            .map(|(e, _)| e)
            .collect();

        let mut counts: HashMap<usize, usize> = HashMap::new();
        for (entity, docs) in &self.entity_docs {
            if frequent.contains(entity) {
                for &doc in docs {
                    *counts.entry(doc).or_default() += 1;
                }
            }
        }
        let mut bridges: Vec<usize> = counts
            .into_iter()
            .filter(|(_, c)| *c >= 2)
            .map(|(doc, _)| doc)
            .collect();
        bridges.sort_unstable();
        bridges
    }

    /// Entities appearing in exactly one document (singletons — follow-up leads).
    pub fn singletons(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .entity_docs
            .iter()
            .filter(|(_, docs)| docs.len() == 1)
            .map(|(e, _)| e.clone())
            .collect();
        v.sort();
        v
    }

    /// Frequent entities that have not yet been searched (uncovered leads).
    pub fn uncovered_leads(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .frequent_entities()
            .into_iter()
            .map(|(e, _)| e)
            .filter(|e| !self.searched_entities.contains(e))
            .collect();
        v.sort();
        v
    }

    /// Mark an entity (or query token) as searched.
    pub fn mark_searched(&mut self, entity: &str) {
        self.searched_entities.insert(entity.to_string());
    }
}

/// Per-search-episode working memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkingMemory {
    /// The original query for the episode (used for BM25 relevance).
    pub original_query: String,
    pub candidate_pool: Vec<Document>,
    pub curated_set: Vec<CuratedDoc>,
    pub evidence_graph: EvidenceGraph,
    pub verification_records: Vec<Verification>,
    pub search_history: Vec<SearchQuery>,
    pub budget: SearchBudget,
    /// Document urls that failed to fetch (never retried).
    pub failed_urls: Vec<String>,
    /// Whether any search has been executed yet (controls auto-seed).
    pub first_search_done: bool,
    next_doc_id: usize,
}

impl WorkingMemory {
    pub fn new(original_query: impl Into<String>, max_turns: usize) -> Self {
        Self {
            original_query: original_query.into(),
            candidate_pool: Vec::new(),
            curated_set: Vec::new(),
            evidence_graph: EvidenceGraph::new(),
            verification_records: Vec::new(),
            search_history: Vec::new(),
            budget: SearchBudget::new(max_turns),
            failed_urls: Vec::new(),
            first_search_done: false,
            next_doc_id: 0,
        }
    }

    fn alloc_id(&mut self) -> usize {
        let id = self.next_doc_id;
        self.next_doc_id += 1;
        id
    }

    /// Whether the candidate pool already contains a document with this url.
    pub fn has_url(&self, url: &str) -> bool {
        self.candidate_pool.iter().any(|d| d.url == url)
    }

    /// Add a document to the candidate pool, allocating an id and ingesting its
    /// text into the evidence graph. Returns the new document id. The caller is
    /// responsible for url/content dedup before calling this.
    pub fn add_document(&mut self, url: String, title: String, full_text: String, compressed: String) -> usize {
        let id = self.alloc_id();
        let doc = Document {
            id,
            url,
            title,
            full_text,
            compressed,
        };
        self.evidence_graph.ingest(id, &doc.title);
        self.evidence_graph.ingest(id, &doc.full_text);
        self.candidate_pool.push(doc);
        self.budget.docs_retrieved += 1;
        id
    }

    pub fn get_document(&self, id: usize) -> Option<&Document> {
        self.candidate_pool.iter().find(|d| d.id == id)
    }

    pub fn get_document_mut(&mut self, id: usize) -> Option<&mut Document> {
        self.candidate_pool.iter_mut().find(|d| d.id == id)
    }

    /// Has the (normalised) query already been searched?
    pub fn already_searched(&self, query: &str) -> bool {
        let norm = query.trim().to_lowercase();
        self.search_history
            .iter()
            .any(|q| q.query.trim().to_lowercase() == norm)
    }

    pub fn record_search(&mut self, query: String, result_count: usize) {
        let turn = self.budget.turns_used;
        self.evidence_graph.mark_searched(query.trim());
        self.search_history.push(SearchQuery {
            query,
            turn,
            result_count,
        });
    }

    /// Add or update a document in the curated set. If the document is already
    /// curated, its importance is updated. If the set is at capacity and the
    /// document is new, the least-important existing document is demoted only if
    /// the incoming importance is strictly greater than the weakest; otherwise
    /// the add is rejected (the model must demote explicitly).
    ///
    /// Returns `Ok(())` on success, or `Err(reason)` if the set is full and the
    /// new doc is not important enough to displace the weakest.
    pub fn curate(&mut self, doc_id: usize, importance: Importance) -> Result<(), String> {
        let turn = self.budget.turns_used;

        // Update in place if already curated.
        if let Some(existing) = self.curated_set.iter_mut().find(|c| c.document.id == doc_id) {
            existing.importance = importance;
            existing.added_at_turn = turn;
            return Ok(());
        }

        let document = match self.get_document(doc_id) {
            Some(d) => d.clone(),
            None => return Err(format!("document {doc_id} not in candidate pool")),
        };

        if self.curated_set.len() >= CURATED_CAP {
            // Find weakest curated doc.
            let weakest_idx = self
                .curated_set
                .iter()
                .enumerate()
                .min_by_key(|(_, c)| c.importance.rank())
                .map(|(i, _)| i);

            match weakest_idx {
                Some(idx) if importance.rank() > self.curated_set[idx].importance.rank() => {
                    self.curated_set.remove(idx);
                }
                _ => {
                    return Err(format!(
                        "curated set full ({CURATED_CAP}); demote an existing document to add a stronger one"
                    ));
                }
            }
        }

        self.curated_set.push(CuratedDoc {
            document,
            importance,
            added_at_turn: turn,
        });
        Ok(())
    }

    /// Count of curated docs per importance level: (VeryHigh, High, Fair, Low).
    pub fn curated_counts(&self) -> (usize, usize, usize, usize) {
        let mut counts = (0, 0, 0, 0);
        for c in &self.curated_set {
            match c.importance {
                Importance::VeryHigh => counts.0 += 1,
                Importance::High => counts.1 += 1,
                Importance::Fair => counts.2 += 1,
                Importance::Low => counts.3 += 1,
            }
        }
        counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    fn working_memory_initialises_empty() {
        let wm = WorkingMemory::new("space elevators", 40);
        assert_eq!(wm.original_query, "space elevators");
        assert!(wm.candidate_pool.is_empty());
        assert!(wm.curated_set.is_empty());
        assert!(wm.verification_records.is_empty());
        assert!(wm.search_history.is_empty());
        assert_eq!(wm.budget.max_turns, 40);
        assert_eq!(wm.budget.turns_used, 0);
        assert!(!wm.first_search_done);
    }

    #[test]
    #[serial]
    fn budget_max_turns_from_env_defaults() {
        // Ensure unset env yields default. (Avoid mutating global env in a way
        // that races other tests — only assert default when unset.)
        if std::env::var(MAX_TURNS_ENV).is_err() {
            assert_eq!(SearchBudget::max_turns_from_env(), DEFAULT_MAX_TURNS);
        }
    }

    #[test]
    fn budget_enforcement() {
        let mut b = SearchBudget::new(3);
        assert!(!b.exhausted());
        b.spend_turn();
        b.spend_turn();
        assert_eq!(b.turns_remaining(), 1);
        assert!(!b.exhausted());
        b.spend_turn();
        assert!(b.exhausted());
        assert_eq!(b.turns_remaining(), 0);
    }

    #[test]
    fn add_document_allocates_ids_and_tracks_evidence() {
        let mut wm = WorkingMemory::new("q", 40);
        let id0 = wm.add_document(
            "http://a".into(),
            "Apollo Program".into(),
            "The Apollo Program landed in 1969.".into(),
            "compressed".into(),
        );
        let id1 = wm.add_document(
            "http://b".into(),
            "Other".into(),
            "Apollo continued in 1972.".into(),
            "compressed".into(),
        );
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(wm.budget.docs_retrieved, 2);
        // "Apollo" appears in both docs -> frequent.
        let freq: Vec<String> = wm
            .evidence_graph
            .frequent_entities()
            .into_iter()
            .map(|(e, _)| e)
            .collect();
        assert!(freq.iter().any(|e| e == "Apollo"));
    }

    #[test]
    fn curate_caps_at_thirty() {
        let mut wm = WorkingMemory::new("q", 40);
        // Add 31 docs and curate all as Fair.
        for i in 0..31 {
            let id = wm.add_document(
                format!("http://{i}"),
                format!("Title {i}"),
                format!("Body {i}"),
                "c".into(),
            );
            let res = wm.curate(id, Importance::Fair);
            if i < CURATED_CAP {
                assert!(res.is_ok(), "doc {i} should curate");
            } else {
                // 31st Fair doc cannot displace an equal-importance doc.
                assert!(res.is_err(), "31st equal-importance doc should be rejected");
            }
        }
        assert_eq!(wm.curated_set.len(), CURATED_CAP);
    }

    #[test]
    fn curate_displaces_weaker_when_full() {
        let mut wm = WorkingMemory::new("q", 40);
        for i in 0..CURATED_CAP {
            let id = wm.add_document(format!("http://{i}"), "t".into(), "b".into(), "c".into());
            assert!(wm.curate(id, Importance::Low).is_ok());
        }
        // New high-importance doc should displace a Low one.
        let id = wm.add_document("http://new".into(), "t".into(), "b".into(), "c".into());
        assert!(wm.curate(id, Importance::VeryHigh).is_ok());
        assert_eq!(wm.curated_set.len(), CURATED_CAP);
        assert!(wm.curated_set.iter().any(|c| c.document.id == id));
    }

    #[test]
    fn curate_update_in_place() {
        let mut wm = WorkingMemory::new("q", 40);
        let id = wm.add_document("http://a".into(), "t".into(), "b".into(), "c".into());
        assert!(wm.curate(id, Importance::Low).is_ok());
        assert!(wm.curate(id, Importance::VeryHigh).is_ok());
        assert_eq!(wm.curated_set.len(), 1);
        assert_eq!(wm.curated_set[0].importance, Importance::VeryHigh);
    }

    #[test]
    fn already_searched_is_case_insensitive() {
        let mut wm = WorkingMemory::new("q", 40);
        wm.record_search("Quantum Computing".into(), 5);
        assert!(wm.already_searched("quantum computing"));
        assert!(wm.already_searched("  QUANTUM COMPUTING "));
        assert!(!wm.already_searched("something else"));
    }

    #[test]
    fn evidence_graph_extracts_entities_and_bridges() {
        let mut g = EvidenceGraph::new();
        g.ingest(0, "Marie Curie won in 1903. Pierre Curie also.");
        g.ingest(1, "Marie Curie returned in 1911 with new work.");
        g.ingest(2, "Albert Einstein published in 1905.");
        // Marie Curie appears in docs 0 and 1 -> frequent.
        let freq: Vec<String> = g.frequent_entities().into_iter().map(|(e, _)| e).collect();
        assert!(freq.iter().any(|e| e == "Marie Curie"));
        // Singletons include Einstein and Pierre Curie.
        let singles = g.singletons();
        assert!(singles.iter().any(|e| e == "Albert Einstein"));
    }

    #[test]
    fn evidence_graph_bridge_detection() {
        let mut g = EvidenceGraph::new();
        // Make two entities frequent and have one doc contain both.
        g.ingest(0, "Tokyo and Kyoto are cities.");
        g.ingest(1, "Tokyo is large.");
        g.ingest(2, "Kyoto is historic.");
        g.ingest(3, "Tokyo and Kyoto both featured.");
        // Tokyo in docs 0,1,3 ; Kyoto in 0,2,3 -> both frequent.
        // Docs 0 and 3 contain both frequent entities -> bridges.
        let bridges = g.bridge_docs();
        assert!(bridges.contains(&0));
        assert!(bridges.contains(&3));
    }

    #[test]
    fn uncovered_leads_excludes_searched() {
        let mut g = EvidenceGraph::new();
        g.ingest(0, "Mercury and Venus.");
        g.ingest(1, "Mercury and Venus again.");
        assert!(g.uncovered_leads().iter().any(|e| e == "Mercury"));
        g.mark_searched("Mercury");
        assert!(!g.uncovered_leads().iter().any(|e| e == "Mercury"));
    }

    #[test]
    fn curated_counts_by_importance() {
        let mut wm = WorkingMemory::new("q", 40);
        let a = wm.add_document("http://a".into(), "t".into(), "b".into(), "c".into());
        let b = wm.add_document("http://b".into(), "t".into(), "b".into(), "c".into());
        wm.curate(a, Importance::VeryHigh).unwrap();
        wm.curate(b, Importance::Fair).unwrap();
        let (vh, hi, fa, lo) = wm.curated_counts();
        assert_eq!((vh, hi, fa, lo), (1, 0, 1, 0));
    }
}
