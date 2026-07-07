//! Action execution: maps harness actions to external tools (SearXNG search,
//! web fetch) via an injectable [`SearchBackend`] trait and to internal
//! working-memory operations.
//!
//! No infrastructure values (IPs, URLs, model names) are hardcoded here. The
//! concrete backend reads endpoints from config/env; tests use a mock.

use crate::harness::actions::HarnessAction;
use crate::harness::state::{Importance, WorkingMemory};
use async_trait::async_trait;
use regex::Regex;
use std::collections::HashSet;

/// Number of top documents auto-seeded into the curated set on the first search.
pub const AUTO_SEED_COUNT: usize = 8;
/// Number of sentences kept by BM25 compression.
pub const COMPRESS_SENTENCES: usize = 4;
/// Cosine-similarity threshold above which two documents are considered
/// duplicates and merged.
pub const CONTENT_DUP_THRESHOLD: f64 = 0.9;

/// A raw search result from the backend before it enters working memory.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub url: String,
    pub title: String,
    /// Snippet / preview text returned by the search engine.
    pub snippet: String,
}

/// Fetched document content.
#[derive(Debug, Clone, PartialEq)]
pub struct FetchedDoc {
    pub url: String,
    pub title: String,
    pub text: String,
}

/// Injectable backend for the two external operations the harness needs:
/// searching (SearXNG) and fetching a document (web_fetch). Implementations
/// supply their own endpoints from configuration — never hardcoded here.
#[async_trait]
pub trait SearchBackend: Send + Sync {
    /// Execute a search query, returning ranked results (possibly empty).
    async fn search(&self, query: &str) -> Result<Vec<SearchResult>, String>;

    /// Fetch the full text of a document by URL.
    async fn fetch(&self, url: &str) -> Result<FetchedDoc, String>;
}

/// Outcome of executing one action — rendered into the next observation.
#[derive(Debug, Clone, PartialEq)]
pub struct ActionOutcome {
    pub summary: String,
    /// Whether the loop should terminate (EndSearch).
    pub terminate: bool,
}

impl ActionOutcome {
    fn cont(summary: impl Into<String>) -> Self {
        Self {
            summary: summary.into(),
            terminate: false,
        }
    }
}

/// The executor: stateless wrapper around a backend that applies an action to
/// working memory.
pub struct Executor<B: SearchBackend> {
    backend: B,
}

impl<B: SearchBackend> Executor<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Execute an action against working memory and return its outcome. A turn
    /// is always considered spent by the caller; this method does not mutate
    /// the budget's turn counter (the harness `step` owns that).
    pub async fn execute(
        &self,
        wm: &mut WorkingMemory,
        action: &HarnessAction,
    ) -> ActionOutcome {
        match action {
            HarnessAction::FanOutSearch { queries } => self.fan_out_search(wm, queries).await,
            HarnessAction::SearchCorpus { query } => self.search_corpus(wm, query).await,
            HarnessAction::GrepCorpus { pattern } => self.grep_corpus(wm, pattern),
            HarnessAction::ReadDocument { doc_id } => self.read_document(wm, *doc_id).await,
            HarnessAction::ReviewDocs { doc_ids } => self.review_docs(wm, doc_ids),
            HarnessAction::Curate { doc_id, importance } => self.curate(wm, *doc_id, *importance),
            HarnessAction::Verify { claim } => self.verify(wm, claim).await,
            HarnessAction::EndSearch => ActionOutcome {
                summary: "Search ended.".into(),
                terminate: true,
            },
        }
    }

    async fn fan_out_search(
        &self,
        wm: &mut WorkingMemory,
        queries: &[String],
    ) -> ActionOutcome {
        let mut added = 0usize;
        let mut skipped_dupes = 0usize;
        let mut total_results = 0usize;
        let was_first = !wm.first_search_done;
        let mut seedable: Vec<usize> = Vec::new();

        for query in queries {
            if wm.already_searched(query) {
                continue;
            }
            match self.backend.search(query).await {
                Ok(results) => {
                    total_results += results.len();
                    wm.record_search(query.clone(), results.len());
                    for r in results {
                        match self.ingest_result(wm, &r) {
                            Some(id) => {
                                seedable.push(id);
                                added += 1;
                            }
                            None => skipped_dupes += 1,
                        }
                    }
                }
                Err(e) => {
                    wm.record_search(query.clone(), 0);
                    return ActionOutcome::cont(format!("Search failed for '{query}': {e}"));
                }
            }
        }

        wm.first_search_done = true;

        if total_results == 0 {
            return ActionOutcome::cont("No results found, try different terms.");
        }

        // Auto-seed top-8 of the newly added docs on the first search.
        let mut seeded = 0usize;
        if was_first {
            for &id in seedable.iter().take(AUTO_SEED_COUNT) {
                if wm.curate(id, Importance::Fair).is_ok() {
                    seeded += 1;
                }
            }
        }

        ActionOutcome::cont(format!(
            "Fan-out across {} queries: {added} docs added, {skipped_dupes} duplicates skipped{}.",
            queries.len(),
            if seeded > 0 {
                format!(", auto-seeded top {seeded}")
            } else {
                String::new()
            }
        ))
    }

    async fn search_corpus(&self, wm: &mut WorkingMemory, query: &str) -> ActionOutcome {
        if wm.already_searched(query) {
            return ActionOutcome::cont(format!("Already searched '{query}'."));
        }
        match self.backend.search(query).await {
            Ok(results) => {
                let n = results.len();
                wm.record_search(query.to_string(), n);
                if n == 0 {
                    return ActionOutcome::cont("No results found, try different terms.");
                }
                let mut added = 0;
                let mut dupes = 0;
                let was_first = !wm.first_search_done;
                let mut seedable = Vec::new();
                for r in results {
                    match self.ingest_result(wm, &r) {
                        Some(id) => {
                            seedable.push(id);
                            added += 1;
                        }
                        None => dupes += 1,
                    }
                }
                wm.first_search_done = true;
                let mut seeded = 0;
                if was_first {
                    for &id in seedable.iter().take(AUTO_SEED_COUNT) {
                        if wm.curate(id, Importance::Fair).is_ok() {
                            seeded += 1;
                        }
                    }
                }
                ActionOutcome::cont(format!(
                    "Searched '{query}': {added} docs added, {dupes} duplicates skipped{}.",
                    if seeded > 0 {
                        format!(", auto-seeded top {seeded}")
                    } else {
                        String::new()
                    }
                ))
            }
            Err(e) => {
                wm.record_search(query.to_string(), 0);
                ActionOutcome::cont(format!("Search failed for '{query}': {e}"))
            }
        }
    }

    fn grep_corpus(&self, wm: &mut WorkingMemory, pattern: &str) -> ActionOutcome {
        let re = match Regex::new(pattern) {
            Ok(re) => re,
            Err(e) => return ActionOutcome::cont(format!("Invalid grep pattern: {e}")),
        };
        let mut hits: Vec<usize> = Vec::new();
        for doc in &wm.candidate_pool {
            if re.is_match(&doc.full_text) || re.is_match(&doc.title) {
                hits.push(doc.id);
            }
        }
        if hits.is_empty() {
            ActionOutcome::cont(format!("grep '{pattern}': no matches."))
        } else {
            ActionOutcome::cont(format!(
                "grep '{pattern}': matched docs {hits:?}."
            ))
        }
    }

    async fn read_document(&self, wm: &mut WorkingMemory, doc_id: usize) -> ActionOutcome {
        let url = match wm.get_document(doc_id) {
            Some(d) => d.url.clone(),
            None => return ActionOutcome::cont(format!("No document with id {doc_id}.")),
        };
        match self.backend.fetch(&url).await {
            Ok(fetched) => {
                let query = wm.original_query.clone();
                let compressed = bm25_compress(&fetched.text, &query, COMPRESS_SENTENCES);
                if let Some(doc) = wm.get_document_mut(doc_id) {
                    doc.full_text = fetched.text.clone();
                    doc.compressed = compressed.clone();
                }
                // Re-ingest fuller text into the evidence graph.
                wm.evidence_graph.ingest(doc_id, &fetched.text);
                ActionOutcome::cont(format!(
                    "Read doc {doc_id}. Key sentences: {compressed}"
                ))
            }
            Err(e) => {
                wm.failed_urls.push(url);
                ActionOutcome::cont(format!(
                    "Fetch failed for doc {doc_id} ({e}); skipped, will not retry."
                ))
            }
        }
    }

    fn review_docs(&self, wm: &mut WorkingMemory, doc_ids: &[usize]) -> ActionOutcome {
        let mut parts = Vec::new();
        for &id in doc_ids {
            match wm.get_document(id) {
                Some(d) => parts.push(format!("[{id}] {}: {}", d.title, d.compressed)),
                None => parts.push(format!("[{id}] (missing)")),
            }
        }
        if parts.is_empty() {
            ActionOutcome::cont("No documents to review.")
        } else {
            ActionOutcome::cont(format!("Review:\n{}", parts.join("\n")))
        }
    }

    fn curate(
        &self,
        wm: &mut WorkingMemory,
        doc_id: usize,
        importance: Importance,
    ) -> ActionOutcome {
        match wm.curate(doc_id, importance) {
            Ok(()) => ActionOutcome::cont(format!(
                "Curated doc {doc_id} as {}.",
                importance.label()
            )),
            Err(e) => ActionOutcome::cont(format!("Curate rejected: {e}")),
        }
    }

    async fn verify(&self, wm: &mut WorkingMemory, claim: &str) -> ActionOutcome {
        // Search for evidence supporting/contradicting the claim.
        let results = self.backend.search(claim).await.unwrap_or_default();
        let mut supporting = Vec::new();
        let mut contradicting = Vec::new();
        // Simple lexical check: a result whose snippet shares claim keywords
        // supports; one containing a negation near a keyword contradicts.
        let keywords: HashSet<String> = tokenize(claim).into_iter().collect();
        let negations = ["not", "no", "never", "false", "incorrect", "wrong", "disputed"];
        for r in &results {
            let snip = r.snippet.to_lowercase();
            let shares = tokenize(&snip).iter().any(|t| keywords.contains(t));
            if !shares {
                continue;
            }
            // Match against existing candidate docs by URL to attribute ids.
            let doc_id = wm.candidate_pool.iter().find(|d| d.url == r.url).map(|d| d.id);
            if let Some(id) = doc_id {
                if negations.iter().any(|n| snip.contains(n)) {
                    contradicting.push(id);
                } else {
                    supporting.push(id);
                }
            }
        }
        let turn = wm.budget.turns_used;
        wm.verification_records
            .push(crate::harness::state::Verification {
                claim: claim.to_string(),
                supporting: supporting.clone(),
                contradicting: contradicting.clone(),
                turn,
            });
        ActionOutcome::cont(format!(
            "Verified '{claim}': {} supporting, {} contradicting.",
            supporting.len(),
            contradicting.len()
        ))
    }

    /// Add a search result to the candidate pool with two-level dedup. Returns
    /// the new doc id, or `None` if it was a duplicate (merged/skipped).
    fn ingest_result(&self, wm: &mut WorkingMemory, r: &SearchResult) -> Option<usize> {
        // Level 1: exact URL dedup.
        if wm.has_url(&r.url) {
            return None;
        }
        // Level 2: content similarity dedup against existing snippets.
        for existing in &wm.candidate_pool {
            let sim = cosine_similarity(&r.snippet, &existing.compressed);
            if sim > CONTENT_DUP_THRESHOLD {
                // Keep the more complete version: if the new snippet is longer,
                // we still skip to avoid duplicate ids — pool already has it.
                return None;
            }
        }
        let compressed = bm25_compress(&r.snippet, &wm.original_query, COMPRESS_SENTENCES);
        let id = wm.add_document(
            r.url.clone(),
            r.title.clone(),
            r.snippet.clone(),
            compressed,
        );
        Some(id)
    }
}

/// Split text into sentences on `.`, `!`, `?` boundaries.
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?') {
            let trimmed = current.trim();
            if !trimmed.is_empty() {
                sentences.push(trimmed.to_string());
            }
            current.clear();
        }
    }
    let trimmed = current.trim();
    if !trimmed.is_empty() {
        sentences.push(trimmed.to_string());
    }
    sentences
}

/// Lowercase alphanumeric tokenizer.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect()
}

/// BM25 compression: rank sentences in `text` by BM25 relevance to `query`,
/// return the top `n` sentences re-joined in their original order.
pub fn bm25_compress(text: &str, query: &str, n: usize) -> String {
    let sentences = split_sentences(text);
    if sentences.len() <= n {
        return sentences.join(" ");
    }

    let query_terms: Vec<String> = tokenize(query);
    if query_terms.is_empty() {
        // No query signal — keep the leading n sentences.
        return sentences
            .iter()
            .take(n)
            .cloned()
            .collect::<Vec<_>>()
            .join(" ");
    }

    // BM25 over sentences-as-documents.
    let k1 = 1.5_f64;
    let b = 0.75_f64;
    let tokenized: Vec<Vec<String>> = sentences.iter().map(|s| tokenize(s)).collect();
    let doc_count = tokenized.len() as f64;
    let avg_len: f64 =
        tokenized.iter().map(|d| d.len() as f64).sum::<f64>() / doc_count.max(1.0);

    // Document frequency per query term.
    let mut df: std::collections::HashMap<&String, usize> = std::collections::HashMap::new();
    for term in &query_terms {
        let count = tokenized
            .iter()
            .filter(|doc| doc.iter().any(|t| t == term))
            .count();
        df.insert(term, count);
    }

    let mut scored: Vec<(usize, f64)> = tokenized
        .iter()
        .enumerate()
        .map(|(idx, doc)| {
            let dl = doc.len() as f64;
            let mut score = 0.0;
            for term in &query_terms {
                let f = doc.iter().filter(|t| *t == term).count() as f64;
                if f == 0.0 {
                    continue;
                }
                let n_q = *df.get(term).unwrap_or(&0) as f64;
                // BM25 idf with +1 smoothing (always positive).
                let idf = ((doc_count - n_q + 0.5) / (n_q + 0.5) + 1.0).ln();
                let denom = f + k1 * (1.0 - b + b * dl / avg_len.max(1.0));
                score += idf * (f * (k1 + 1.0)) / denom.max(1e-9);
            }
            (idx, score)
        })
        .collect();

    // Pick top-n by score, then restore original order.
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut chosen: Vec<usize> = scored.iter().take(n).map(|(i, _)| *i).collect();
    chosen.sort_unstable();
    chosen
        .iter()
        .map(|&i| sentences[i].clone())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Cosine similarity over bag-of-words term-frequency vectors.
pub fn cosine_similarity(a: &str, b: &str) -> f64 {
    use std::collections::HashMap;
    let mut va: HashMap<String, f64> = HashMap::new();
    let mut vb: HashMap<String, f64> = HashMap::new();
    for t in tokenize(a) {
        *va.entry(t).or_default() += 1.0;
    }
    for t in tokenize(b) {
        *vb.entry(t).or_default() += 1.0;
    }
    if va.is_empty() || vb.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0;
    for (k, v) in &va {
        if let Some(w) = vb.get(k) {
            dot += v * w;
        }
    }
    let na: f64 = va.values().map(|v| v * v).sum::<f64>().sqrt();
    let nb: f64 = vb.values().map(|v| v * v).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

#[cfg(test)]
pub(crate) mod mock {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;

    /// A scriptable in-memory backend for tests.
    pub struct MockBackend {
        pub responses: HashMap<String, Vec<SearchResult>>,
        pub fetches: HashMap<String, FetchedDoc>,
        pub fail_fetch: HashMap<String, bool>,
        pub search_calls: Mutex<Vec<String>>,
    }

    impl MockBackend {
        pub fn new() -> Self {
            Self {
                responses: HashMap::new(),
                fetches: HashMap::new(),
                fail_fetch: HashMap::new(),
                search_calls: Mutex::new(Vec::new()),
            }
        }

        pub fn with_search(mut self, query: &str, results: Vec<SearchResult>) -> Self {
            self.responses.insert(query.to_string(), results);
            self
        }

        pub fn with_fetch(mut self, url: &str, text: &str) -> Self {
            self.fetches.insert(
                url.to_string(),
                FetchedDoc {
                    url: url.to_string(),
                    title: "fetched".into(),
                    text: text.to_string(),
                },
            );
            self
        }

        pub fn with_fetch_failure(mut self, url: &str) -> Self {
            self.fail_fetch.insert(url.to_string(), true);
            self
        }
    }

    #[async_trait]
    impl SearchBackend for MockBackend {
        async fn search(&self, query: &str) -> Result<Vec<SearchResult>, String> {
            self.search_calls.lock().unwrap().push(query.to_string());
            Ok(self.responses.get(query).cloned().unwrap_or_default())
        }

        async fn fetch(&self, url: &str) -> Result<FetchedDoc, String> {
            if self.fail_fetch.get(url).copied().unwrap_or(false) {
                return Err("404 not found".into());
            }
            self.fetches
                .get(url)
                .cloned()
                .ok_or_else(|| "no such url".to_string())
        }
    }

    pub fn result(url: &str, title: &str, snippet: &str) -> SearchResult {
        SearchResult {
            url: url.into(),
            title: title.into(),
            snippet: snippet.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::mock::*;
    use super::*;
    use crate::harness::state::WorkingMemory;

    fn many_results(n: usize) -> Vec<SearchResult> {
        (0..n)
            .map(|i| result(&format!("http://x/{i}"), &format!("Title {i}"), &format!("Snippet body number {i}.")))
            .collect()
    }

    #[tokio::test]
    async fn search_corpus_adds_documents() {
        let backend = MockBackend::new().with_search("rust", many_results(3));
        let exec = Executor::new(backend);
        let mut wm = WorkingMemory::new("rust", 40);
        let out = exec
            .execute(&mut wm, &HarnessAction::SearchCorpus { query: "rust".into() })
            .await;
        assert!(!out.terminate);
        assert_eq!(wm.candidate_pool.len(), 3);
    }

    #[tokio::test]
    async fn search_zero_results_message() {
        let backend = MockBackend::new().with_search("empty", vec![]);
        let exec = Executor::new(backend);
        let mut wm = WorkingMemory::new("q", 40);
        let out = exec
            .execute(&mut wm, &HarnessAction::SearchCorpus { query: "empty".into() })
            .await;
        assert!(out.summary.contains("No results"));
    }

    #[tokio::test]
    async fn auto_seed_on_first_search() {
        let backend = MockBackend::new().with_search("seed", many_results(12));
        let exec = Executor::new(backend);
        let mut wm = WorkingMemory::new("seed", 40);
        exec.execute(&mut wm, &HarnessAction::SearchCorpus { query: "seed".into() })
            .await;
        // Top 8 auto-seeded into curated set.
        assert_eq!(wm.curated_set.len(), AUTO_SEED_COUNT);
    }

    #[tokio::test]
    async fn no_auto_seed_on_second_search() {
        let backend = MockBackend::new()
            .with_search("first", many_results(12))
            .with_search("second", many_results(5));
        let exec = Executor::new(backend);
        let mut wm = WorkingMemory::new("q", 40);
        exec.execute(&mut wm, &HarnessAction::SearchCorpus { query: "first".into() })
            .await;
        let after_first = wm.curated_set.len();
        exec.execute(&mut wm, &HarnessAction::SearchCorpus { query: "second".into() })
            .await;
        assert_eq!(wm.curated_set.len(), after_first); // no new auto-seed
    }

    #[tokio::test]
    async fn dedup_by_url() {
        let r = result("http://dup", "T", "Some snippet text here.");
        let backend = MockBackend::new()
            .with_search("a", vec![r.clone()])
            .with_search("b", vec![r.clone()]);
        let exec = Executor::new(backend);
        let mut wm = WorkingMemory::new("q", 40);
        exec.execute(&mut wm, &HarnessAction::SearchCorpus { query: "a".into() })
            .await;
        exec.execute(&mut wm, &HarnessAction::SearchCorpus { query: "b".into() })
            .await;
        assert_eq!(wm.candidate_pool.len(), 1);
    }

    #[tokio::test]
    async fn dedup_by_content_similarity() {
        let r1 = result("http://1", "T1", "The quick brown fox jumps over the lazy dog.");
        let r2 = result("http://2", "T2", "The quick brown fox jumps over the lazy dog.");
        let backend = MockBackend::new()
            .with_search("a", vec![r1])
            .with_search("b", vec![r2]);
        let exec = Executor::new(backend);
        let mut wm = WorkingMemory::new("q", 40);
        exec.execute(&mut wm, &HarnessAction::SearchCorpus { query: "a".into() })
            .await;
        exec.execute(&mut wm, &HarnessAction::SearchCorpus { query: "b".into() })
            .await;
        // Same content, different URL -> content dedup keeps one.
        assert_eq!(wm.candidate_pool.len(), 1);
    }

    #[tokio::test]
    async fn fan_out_search_multiple_queries() {
        let backend = MockBackend::new()
            .with_search("q1", many_results(3))
            .with_search("q2", many_results(2));
        // q2's urls collide with q1's (http://x/0..) -> dedup.
        let exec = Executor::new(backend);
        let mut wm = WorkingMemory::new("q", 40);
        exec.execute(
            &mut wm,
            &HarnessAction::FanOutSearch {
                queries: vec!["q1".into(), "q2".into()],
            },
        )
        .await;
        assert_eq!(wm.search_history.len(), 2);
        assert_eq!(wm.candidate_pool.len(), 3); // q2 fully deduped by URL
    }

    #[tokio::test]
    async fn already_searched_query_skipped() {
        let backend = MockBackend::new().with_search("a", many_results(2));
        let exec = Executor::new(backend);
        let mut wm = WorkingMemory::new("q", 40);
        exec.execute(&mut wm, &HarnessAction::SearchCorpus { query: "a".into() })
            .await;
        let out = exec
            .execute(&mut wm, &HarnessAction::SearchCorpus { query: "a".into() })
            .await;
        assert!(out.summary.contains("Already searched"));
    }

    #[tokio::test]
    async fn read_document_fetches_and_compresses() {
        let backend = MockBackend::new()
            .with_search("a", vec![result("http://d", "T", "snippet")])
            .with_fetch(
                "http://d",
                "Rust is fast. Rust is safe. Cats are fluffy. The weather is nice. Rust has zero cost abstractions and Rust is reliable.",
            );
        let exec = Executor::new(backend);
        let mut wm = WorkingMemory::new("rust", 40);
        exec.execute(&mut wm, &HarnessAction::SearchCorpus { query: "a".into() })
            .await;
        let out = exec.execute(&mut wm, &HarnessAction::ReadDocument { doc_id: 0 }).await;
        assert!(out.summary.contains("Rust"));
        let doc = wm.get_document(0).unwrap();
        assert!(doc.compressed.contains("Rust"));
    }

    #[tokio::test]
    async fn read_document_fetch_failure_records_and_skips() {
        let backend = MockBackend::new()
            .with_search("a", vec![result("http://gone", "T", "snippet")])
            .with_fetch_failure("http://gone");
        let exec = Executor::new(backend);
        let mut wm = WorkingMemory::new("q", 40);
        exec.execute(&mut wm, &HarnessAction::SearchCorpus { query: "a".into() })
            .await;
        let out = exec.execute(&mut wm, &HarnessAction::ReadDocument { doc_id: 0 }).await;
        assert!(out.summary.contains("Fetch failed"));
        assert_eq!(wm.failed_urls.len(), 1);
    }

    #[tokio::test]
    async fn grep_corpus_matches() {
        let backend = MockBackend::new()
            .with_search("a", vec![result("http://1", "Quantum", "Quantum entanglement explained.")]);
        let exec = Executor::new(backend);
        let mut wm = WorkingMemory::new("q", 40);
        exec.execute(&mut wm, &HarnessAction::SearchCorpus { query: "a".into() })
            .await;
        let out = exec
            .execute(&mut wm, &HarnessAction::GrepCorpus { pattern: "entangle".into() })
            .await;
        assert!(out.summary.contains("matched"));
    }

    #[tokio::test]
    async fn verify_records_verification() {
        let backend = MockBackend::new()
            .with_search("a", vec![result("http://1", "T", "Water boils at 100 degrees.")])
            .with_search(
                "water boils at 100 degrees",
                vec![result("http://1", "T", "Water boils at 100 degrees confirmed.")],
            );
        let exec = Executor::new(backend);
        let mut wm = WorkingMemory::new("q", 40);
        exec.execute(&mut wm, &HarnessAction::SearchCorpus { query: "a".into() })
            .await;
        exec.execute(
            &mut wm,
            &HarnessAction::Verify {
                claim: "water boils at 100 degrees".into(),
            },
        )
        .await;
        assert_eq!(wm.verification_records.len(), 1);
        assert_eq!(wm.verification_records[0].supporting, vec![0]);
    }

    #[test]
    fn bm25_compress_picks_relevant_sentences() {
        let text = "The cat sat on the mat. Quantum computing uses qubits. \
                    Dogs are loyal animals. Qubits enable quantum parallelism. \
                    The weather today is sunny.";
        let out = bm25_compress(text, "quantum qubits", 2);
        assert!(out.contains("qubits") || out.contains("Quantum") || out.contains("quantum"));
        // Should select 2 sentences.
        assert!(out.matches('.').count() <= 2);
    }

    #[test]
    fn bm25_short_text_returns_all() {
        let text = "One sentence here. Two sentences here.";
        let out = bm25_compress(text, "query", 4);
        assert!(out.contains("One"));
        assert!(out.contains("Two"));
    }

    #[test]
    fn cosine_identical_is_one() {
        let s = "the quick brown fox";
        assert!((cosine_similarity(s, s) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cosine_disjoint_is_zero() {
        assert_eq!(cosine_similarity("apple banana", "car truck"), 0.0);
    }
}
