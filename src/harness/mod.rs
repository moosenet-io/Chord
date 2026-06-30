//! Harness-1 search harness — a state machine inside Chord's agentic executor.
//!
//! The harness maintains per-search [`WorkingMemory`] (candidate pool, curated
//! set, evidence graph, verification records) and presents a compact rendered
//! observation to the model each turn. The model emits one [`HarnessAction`];
//! the harness executes it, updates state, and renders the next observation.
//!
//! This is the core of the stateful cognitive-offloading principle: the harness
//! holds the bookkeeping, the model makes the decisions.

pub mod actions;
pub mod detector;
pub mod executor;
pub mod state;
pub mod tool_definition;
pub mod vram_lifecycle;

use actions::HarnessAction;
use executor::{Executor, SearchBackend};
use state::{SearchBudget, WorkingMemory};

/// What the model sees after each turn.
#[derive(Debug, Clone, PartialEq)]
pub struct Observation {
    /// Compact, parseable rendered state.
    pub rendered: String,
    /// Whether the search loop is complete.
    pub complete: bool,
}

/// The search harness: owns working memory + an injectable backend.
pub struct SearchHarness<B: SearchBackend> {
    memory: WorkingMemory,
    executor: Executor<B>,
    complete: bool,
    /// Last few action names + result summaries for rendering.
    recent: Vec<(String, String)>,
}

const RECENT_HISTORY: usize = 3;

impl<B: SearchBackend> SearchHarness<B> {
    /// Create a harness for a search episode. `max_turns` defaults from the
    /// `HARNESS_MAX_TURNS` env var; callers may pass an explicit override.
    pub fn new(original_query: impl Into<String>, backend: B) -> Self {
        let max_turns = SearchBudget::max_turns_from_env();
        Self {
            memory: WorkingMemory::new(original_query, max_turns),
            executor: Executor::new(backend),
            complete: false,
            recent: Vec::new(),
        }
    }

    /// Create a harness with an explicit turn budget (bypasses env lookup).
    pub fn with_max_turns(original_query: impl Into<String>, backend: B, max_turns: usize) -> Self {
        Self {
            memory: WorkingMemory::new(original_query, max_turns),
            executor: Executor::new(backend),
            complete: false,
            recent: Vec::new(),
        }
    }

    /// Read-only access to working memory (for inspection / synthesis phase).
    pub fn memory(&self) -> &WorkingMemory {
        &self.memory
    }

    /// Whether the search loop has terminated (EndSearch or budget exhausted).
    pub fn is_complete(&self) -> bool {
        self.complete || self.memory.budget.exhausted()
    }

    /// Execute one action. Each call spends a turn (even invalid actions, per
    /// the spec). Returns the observation the model should see next.
    pub async fn step(&mut self, action: HarnessAction) -> Observation {
        if self.is_complete() {
            return Observation {
                rendered: self.render_state_with("Search already complete."),
                complete: true,
            };
        }

        // Spend a turn for this step.
        self.memory.budget.spend_turn();

        let outcome = self.executor.execute(&mut self.memory, &action).await;
        if outcome.terminate {
            self.complete = true;
        }

        self.push_recent(action.name(), &outcome.summary);

        Observation {
            rendered: self.render_state_with(&outcome.summary),
            complete: self.is_complete(),
        }
    }

    /// Execute a raw JSON action (parses, deducting a turn on parse error and
    /// surfacing the available actions).
    pub async fn step_json(&mut self, value: &serde_json::Value) -> Observation {
        match HarnessAction::from_json(value) {
            Ok(action) => self.step(action).await,
            Err(e) => {
                if !self.is_complete() {
                    self.memory.budget.spend_turn();
                }
                self.push_recent("invalid", &e);
                Observation {
                    rendered: self.render_state_with(&e),
                    complete: self.is_complete(),
                }
            }
        }
    }

    fn push_recent(&mut self, name: &str, summary: &str) {
        self.recent.push((name.to_string(), summary.to_string()));
        if self.recent.len() > RECENT_HISTORY {
            let drop = self.recent.len() - RECENT_HISTORY;
            self.recent.drain(0..drop);
        }
    }

    /// Render the current state as a compact observation, no last-action note.
    pub fn render_state(&self) -> String {
        self.render_state_with("")
    }

    fn render_state_with(&self, note: &str) -> String {
        let b = &self.memory.budget;
        let (vh, hi, fa, lo) = self.memory.curated_counts();
        let frequent = self.memory.evidence_graph.frequent_entities();
        let bridges = self.memory.evidence_graph.bridge_docs();
        let singletons = self.memory.evidence_graph.singletons();
        let leads = self.memory.evidence_graph.uncovered_leads();

        let mut out = String::new();
        out.push_str(&format!(
            "SEARCH BUDGET: {}/{} turns, {} docs\n",
            b.turns_used, b.max_turns, b.docs_retrieved
        ));
        out.push_str(&format!(
            "CURATED SET ({} total): VeryHigh={vh} High={hi} Fair={fa} Low={lo}\n",
            self.memory.curated_set.len()
        ));

        // Recent actions (last 3).
        out.push_str("RECENT ACTIONS:\n");
        if self.recent.is_empty() {
            out.push_str("  (none)\n");
        } else {
            for (name, summary) in &self.recent {
                out.push_str(&format!("  - {name}: {}\n", truncate(summary, 160)));
            }
        }

        // Evidence graph.
        let freq_str = frequent
            .iter()
            .take(8)
            .map(|(e, c)| format!("{e}({c})"))
            .collect::<Vec<_>>()
            .join(", ");
        out.push_str(&format!(
            "EVIDENCE GRAPH: {} frequent entities [{}]; {} bridge docs; {} singletons\n",
            frequent.len(),
            freq_str,
            bridges.len(),
            singletons.len()
        ));

        let leads_str = leads
            .iter()
            .take(8)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        out.push_str(&format!("UNCOVERED LEADS: [{leads_str}]\n"));

        out.push_str(&format!(
            "VERIFICATIONS: {}\n",
            self.memory.verification_records.len()
        ));

        if !note.is_empty() {
            out.push_str(&format!("LAST RESULT: {}\n", truncate(note, 400)));
        }

        if self.is_complete() {
            out.push_str("STATUS: COMPLETE\n");
        }
        out
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max).collect();
        format!("{truncated}…")
    }
}

#[cfg(test)]
mod tests {
    use super::executor::mock::{result, MockBackend};
    use super::*;

    fn many_results(n: usize) -> Vec<executor::SearchResult> {
        (0..n)
            .map(|i| {
                result(
                    &format!("http://x/{i}"),
                    &format!("Title {i}"),
                    &format!("Body sentence number {i}."),
                )
            })
            .collect()
    }

    #[tokio::test]
    async fn new_harness_is_not_complete() {
        let h = SearchHarness::new("query", MockBackend::new());
        assert!(!h.is_complete());
        assert_eq!(h.memory().original_query, "query");
    }

    #[tokio::test]
    async fn step_spends_a_turn() {
        let backend = MockBackend::new().with_search("a", many_results(2));
        let mut h = SearchHarness::with_max_turns("q", backend, 40);
        assert_eq!(h.memory().budget.turns_used, 0);
        h.step(HarnessAction::SearchCorpus { query: "a".into() }).await;
        assert_eq!(h.memory().budget.turns_used, 1);
    }

    #[tokio::test]
    async fn end_search_terminates() {
        let mut h = SearchHarness::with_max_turns("q", MockBackend::new(), 40);
        let obs = h.step(HarnessAction::EndSearch).await;
        assert!(obs.complete);
        assert!(h.is_complete());
        assert!(obs.rendered.contains("COMPLETE"));
    }

    #[tokio::test]
    async fn budget_enforcement_completes_at_max_turns() {
        let backend = MockBackend::new()
            .with_search("a", many_results(1))
            .with_search("b", many_results(1));
        let mut h = SearchHarness::with_max_turns("q", backend, 2);
        h.step(HarnessAction::GrepCorpus { pattern: "x".into() }).await;
        assert!(!h.is_complete());
        let obs = h.step(HarnessAction::GrepCorpus { pattern: "y".into() }).await;
        assert!(obs.complete);
        assert!(h.is_complete());
        // Further steps are no-ops.
        let before = h.memory().budget.turns_used;
        h.step(HarnessAction::GrepCorpus { pattern: "z".into() }).await;
        assert_eq!(h.memory().budget.turns_used, before);
    }

    #[tokio::test]
    async fn render_state_is_compact_and_parseable() {
        let backend = MockBackend::new().with_search("a", many_results(3));
        let mut h = SearchHarness::with_max_turns("test query", backend, 40);
        h.step(HarnessAction::SearchCorpus { query: "a".into() }).await;
        let r = h.render_state();
        assert!(r.contains("SEARCH BUDGET:"));
        assert!(r.contains("CURATED SET"));
        assert!(r.contains("EVIDENCE GRAPH:"));
        assert!(r.contains("UNCOVERED LEADS:"));
        assert!(r.contains("VERIFICATIONS:"));
    }

    #[tokio::test]
    async fn invalid_action_deducts_turn_and_shows_available() {
        let mut h = SearchHarness::with_max_turns("q", MockBackend::new(), 40);
        let obs = h.step_json(&serde_json::json!({"action": "fly_to_moon"})).await;
        assert_eq!(h.memory().budget.turns_used, 1);
        assert!(obs.rendered.contains("Available") || obs.rendered.contains("invalid"));
    }

    /// Integration test: a full search episode with a mock SearXNG backend.
    #[tokio::test]
    async fn full_episode_with_mock_backend() {
        let backend = MockBackend::new()
            .with_search(
                "renewable energy",
                vec![
                    result(
                        "http://a",
                        "Solar Power",
                        "Solar power from the Sun grew rapidly in 2020 across Germany.",
                    ),
                    result(
                        "http://b",
                        "Wind Energy",
                        "Wind energy in Germany expanded in 2021 with new turbines.",
                    ),
                    result(
                        "http://c",
                        "Hydro",
                        "Hydropower remains stable in Norway since 1990.",
                    ),
                ],
            )
            .with_fetch(
                "http://a",
                "Solar power is renewable. Germany installed many panels in 2020. \
                 The Sun provides abundant energy. Costs fell sharply. Solar is now competitive.",
            )
            .with_search(
                "solar power Germany",
                vec![result(
                    "http://d",
                    "Germany Solar 2022",
                    "Germany solar capacity in 2022 reached record highs.",
                )],
            );

        let mut h = SearchHarness::with_max_turns("renewable energy", backend, 40);

        // Turn 1: initial search (auto-seeds curated).
        let obs = h
            .step(HarnessAction::SearchCorpus {
                query: "renewable energy".into(),
            })
            .await;
        assert!(!obs.complete);
        assert_eq!(h.memory().candidate_pool.len(), 3);
        assert!(!h.memory().curated_set.is_empty(), "auto-seed should populate curated set");

        // Turn 2: read a document (fetch + compress).
        h.step(HarnessAction::ReadDocument { doc_id: 0 }).await;
        assert!(h.memory().get_document(0).unwrap().compressed.len() > 0);

        // Turn 3: follow a lead with another search.
        h.step(HarnessAction::SearchCorpus {
            query: "solar power Germany".into(),
        })
        .await;
        assert_eq!(h.memory().candidate_pool.len(), 4);

        // Turn 4: curate the new doc highly.
        h.step(HarnessAction::Curate {
            doc_id: 3,
            importance: state::Importance::VeryHigh,
        })
        .await;
        let (vh, _, _, _) = h.memory().curated_counts();
        assert!(vh >= 1);

        // Turn 5: end search.
        let obs = h.step(HarnessAction::EndSearch).await;
        assert!(obs.complete);
        assert!(h.is_complete());

        // Germany appears in multiple docs -> frequent entity.
        let freq: Vec<String> = h
            .memory()
            .evidence_graph
            .frequent_entities()
            .into_iter()
            .map(|(e, _)| e)
            .collect();
        assert!(freq.iter().any(|e| e == "Germany"));
    }
}
