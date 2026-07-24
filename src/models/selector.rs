//! S125 CH-SEL-01: the unified **TaskDescriptor selector**.
//!
//! Clients stop naming raw models. They send a [`TaskDescriptor`] — *what* the work needs
//! (task + modalities + constraints) — and Chord resolves it to a concrete `(model,
//! backend)`. This generalizes the three bespoke selectors that exist today
//! (`models::coding_selector`, `agentic::model_router`, `router::policy`) into ONE
//! capability-aware path, and closes the loop with MINT: candidates carry an operational
//! score (from MINT's profiles) so profiling directly improves routing.
//!
//! The core [`resolve`] is PURE over an injected candidate set, so it is fully unit-testable
//! offline; the runtime adapter that assembles [`ModelCandidate`]s from the model registry +
//! backend catalogue + MINT scores is a thin layer on top (see `candidates_from`).

use crate::models::backends::BackendKind;
use crate::models::capability::{Capability, CapabilityRegistry, Modality};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// What kind of work a request needs. Maps 1:1 to a required [`Capability`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Task {
    Chat,
    Reasoning,
    Code,
    Embed,
    Rerank,
    VisionQa,
    Ocr,
    DocParse,
    ImageGen,
    Tts,
    Stt,
    ToolRoute,
    Diffusion,
}

impl Task {
    /// The capability a model MUST declare to serve this task.
    pub fn required_capability(self) -> Capability {
        match self {
            Task::Chat => Capability::Chat,
            Task::Reasoning => Capability::Reasoning,
            Task::Code => Capability::Code,
            Task::Embed => Capability::Embed,
            Task::Rerank => Capability::Rerank,
            Task::VisionQa => Capability::Vlm,
            Task::Ocr => Capability::Ocr,
            Task::DocParse => Capability::Doc,
            Task::ImageGen => Capability::ImageGen,
            Task::Tts => Capability::Tts,
            Task::Stt => Capability::Stt,
            Task::ToolRoute => Capability::ToolRouter,
            Task::Diffusion => Capability::Diffusion,
        }
    }
}

/// Quality/latency posture. `Fast` biases toward cheaper/local; `Max` toward the
/// highest-scoring model regardless of size; `Balanced` is the default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualityTier {
    Fast,
    Balanced,
    Max,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Constraints {
    pub max_latency_ms: Option<u64>,
    pub quality_tier: Option<QualityTier>,
    /// Sovereignty: exclude remote/cloud (OpenRouter) backends.
    pub local_only: Option<bool>,
    pub max_cost: Option<f64>,
    /// Minimum safe context window the model must support.
    pub context_tokens: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Hints {
    pub language: Option<String>,
    pub domain: Option<String>,
    /// Explicit model override (back-compat: bare-model-name path). When set and the model
    /// is a candidate, it wins outright.
    pub preferred_model: Option<String>,
}

/// A client's request for a model, by CAPABILITY not by name.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskDescriptor {
    pub task: Task,
    #[serde(default)]
    pub input: Vec<Modality>,
    #[serde(default)]
    pub output: Option<Modality>,
    #[serde(default)]
    pub constraints: Constraints,
    #[serde(default)]
    pub hints: Hints,
}

/// A concrete servable model the selector may pick — assembled by the caller from the model
/// registry + backend catalogue + (optionally) a MINT operational score.
#[derive(Debug, Clone)]
pub struct ModelCandidate {
    pub name: String,
    pub backend: String,
    pub backend_kind: BackendKind,
    /// Registered + loadable right now (vs archived/unreachable).
    pub available: bool,
    /// Max safe context window (from the serving profile); `None` ⇒ unknown (not excluded).
    pub max_context: Option<u64>,
    /// MINT operational score for the requested task, higher = better; `None` ⇒ unprofiled.
    pub score: Option<f64>,
    /// False for remote/cloud backends (OpenRouter). Drives `local_only`.
    pub is_local: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Selection {
    pub model: String,
    pub backend: String,
    pub reason: String,
}

/// Resolve a [`TaskDescriptor`] to a `(model, backend)`. PURE over `candidates`.
///
/// Order: (0) explicit `preferred_model` override → (1) capability filter → (2) constraint
/// filters (local_only, context) → (3) prefer currently-available candidates → (4) rank by a
/// tier-weighted key (MINT score, local-first, larger-context tiebreak) and pick the best.
/// Returns `None` when no candidate can serve the task under the constraints.
pub fn resolve(
    desc: &TaskDescriptor,
    caps: &CapabilityRegistry,
    candidates: &[ModelCandidate],
) -> Option<Selection> {
    // (0) Explicit override wins if it is a real candidate (back-compat with bare names).
    if let Some(pref) = desc.hints.preferred_model.as_deref() {
        if let Some(c) = candidates.iter().find(|c| c.name == pref) {
            return Some(Selection {
                model: c.name.clone(),
                backend: c.backend.clone(),
                reason: "preferred_model override".to_string(),
            });
        }
    }

    let need = desc.task.required_capability();

    // (1) Capability filter.
    let mut pool: Vec<&ModelCandidate> =
        candidates.iter().filter(|c| caps.has(&c.name, need)).collect();
    if pool.is_empty() {
        return None;
    }

    // (2) Constraint filters.
    if desc.constraints.local_only.unwrap_or(false) {
        pool.retain(|c| c.is_local);
    }
    if let Some(min_ctx) = desc.constraints.context_tokens {
        // Unknown context is NOT excluded (fail-open) — only a known-too-small one is.
        pool.retain(|c| c.max_context.map(|m| m >= min_ctx).unwrap_or(true));
    }
    if pool.is_empty() {
        return None;
    }

    // (3) Prefer currently-available candidates; fall back to the full pool if none are.
    let available: Vec<&ModelCandidate> = pool.iter().copied().filter(|c| c.available).collect();
    let ranked = if available.is_empty() { pool } else { available };

    // (4) Rank and pick.
    let tier = desc.constraints.quality_tier.unwrap_or(QualityTier::Balanced);
    let best = ranked
        .iter()
        .copied()
        .max_by(|a, b| cmp_candidates(a, b, tier))?;
    Some(Selection {
        model: best.name.clone(),
        backend: best.backend.clone(),
        reason: format!(
            "capability={:?} tier={:?} score={:?} local={}",
            need, tier, best.score, best.is_local
        ),
    })
}

/// Total order for "which candidate is better" (greater = better).
fn cmp_candidates(a: &ModelCandidate, b: &ModelCandidate, tier: QualityTier) -> Ordering {
    // Primary: MINT score (unprofiled = -inf so a profiled model always wins).
    let sa = a.score.unwrap_or(f64::NEG_INFINITY);
    let sb = b.score.unwrap_or(f64::NEG_INFINITY);
    let score_cmp = sa.partial_cmp(&sb).unwrap_or(Ordering::Equal);

    // Tier bias:
    //  - Fast/Balanced prefer LOCAL on a score tie (sovereign + no network latency).
    //  - Max ignores locality on ties and prefers the larger context window.
    match tier {
        QualityTier::Max => score_cmp
            .then_with(|| a.max_context.unwrap_or(0).cmp(&b.max_context.unwrap_or(0)))
            .then_with(|| a.is_local.cmp(&b.is_local)),
        QualityTier::Fast | QualityTier::Balanced => score_cmp
            .then_with(|| a.is_local.cmp(&b.is_local))
            .then_with(|| a.max_context.unwrap_or(0).cmp(&b.max_context.unwrap_or(0))),
    }
    // Final deterministic tiebreak by name (reverse so `max_by` yields the lexicographically
    // SMALLER name — stable across runs).
    .then_with(|| b.name.cmp(&a.name))
}

/// Is this backend kind remote/cloud (i.e. NOT sovereign-local)?
pub fn backend_is_local(kind: BackendKind) -> bool {
    !matches!(kind, BackendKind::OpenRouter)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(name: &str, backend: &str, kind: BackendKind, avail: bool, score: Option<f64>) -> ModelCandidate {
        ModelCandidate {
            name: name.to_string(),
            backend: backend.to_string(),
            backend_kind: kind,
            available: avail,
            max_context: Some(32768),
            score,
            is_local: backend_is_local(kind),
        }
    }

    fn caps_with(entries: &[(&str, &[Capability])]) -> CapabilityRegistry {
        let mut c = CapabilityRegistry::default();
        for (m, cs) in entries {
            c.set(*m, cs.to_vec());
        }
        c
    }

    fn desc(task: Task) -> TaskDescriptor {
        TaskDescriptor {
            task,
            input: vec![],
            output: None,
            constraints: Constraints::default(),
            hints: Hints::default(),
        }
    }

    #[test]
    fn picks_highest_scoring_capable_model() {
        let caps = caps_with(&[
            ("coder-a", &[Capability::Chat, Capability::Code]),
            ("coder-b", &[Capability::Chat, Capability::Code]),
            ("chat-only", &[Capability::Chat]),
        ]);
        let cands = vec![
            cand("coder-a", "ollama", BackendKind::Ollama, true, Some(0.7)),
            cand("coder-b", "lemonade-coder", BackendKind::LlamaServer, true, Some(0.9)),
            cand("chat-only", "ollama", BackendKind::Ollama, true, Some(0.99)),
        ];
        let sel = resolve(&desc(Task::Code), &caps, &cands).unwrap();
        assert_eq!(sel.model, "coder-b"); // highest score AMONG code-capable (chat-only excluded)
    }

    #[test]
    fn local_only_excludes_openrouter() {
        let caps = caps_with(&[
            ("cloud", &[Capability::Chat]),
            ("local", &[Capability::Chat]),
        ]);
        let cands = vec![
            cand("cloud", "openrouter", BackendKind::OpenRouter, true, Some(0.95)),
            cand("local", "ollama", BackendKind::Ollama, true, Some(0.5)),
        ];
        let mut d = desc(Task::Chat);
        d.constraints.local_only = Some(true);
        let sel = resolve(&d, &caps, &cands).unwrap();
        assert_eq!(sel.model, "local"); // cloud excluded despite higher score
    }

    #[test]
    fn no_capable_model_returns_none() {
        let caps = caps_with(&[("chat", &[Capability::Chat])]);
        let cands = vec![cand("chat", "ollama", BackendKind::Ollama, true, Some(0.8))];
        assert!(resolve(&desc(Task::VisionQa), &caps, &cands).is_none());
    }

    #[test]
    fn preferred_model_overrides() {
        let caps = caps_with(&[("a", &[Capability::Chat]), ("b", &[Capability::Chat])]);
        let cands = vec![
            cand("a", "ollama", BackendKind::Ollama, true, Some(0.9)),
            cand("b", "ollama", BackendKind::Ollama, true, Some(0.1)),
        ];
        let mut d = desc(Task::Chat);
        d.hints.preferred_model = Some("b".to_string());
        assert_eq!(resolve(&d, &caps, &cands).unwrap().model, "b");
    }

    #[test]
    fn context_constraint_excludes_too_small() {
        let caps = caps_with(&[("small", &[Capability::Chat]), ("big", &[Capability::Chat])]);
        let mut small = cand("small", "ollama", BackendKind::Ollama, true, Some(0.9));
        small.max_context = Some(8192);
        let mut big = cand("big", "ollama", BackendKind::Ollama, true, Some(0.5));
        big.max_context = Some(131072);
        let mut d = desc(Task::Chat);
        d.constraints.context_tokens = Some(65536);
        let sel = resolve(&d, &caps, &vec![small, big]).unwrap();
        assert_eq!(sel.model, "big"); // small excluded by context need
    }

    #[test]
    fn prefers_available_over_higher_scoring_unavailable() {
        let caps = caps_with(&[("hot", &[Capability::Chat]), ("cold", &[Capability::Chat])]);
        let cands = vec![
            cand("cold", "ollama", BackendKind::Ollama, false, Some(0.99)),
            cand("hot", "ollama", BackendKind::Ollama, true, Some(0.4)),
        ];
        assert_eq!(resolve(&desc(Task::Chat), &caps, &cands).unwrap().model, "hot");
    }

    #[test]
    fn local_tiebreak_on_equal_score() {
        let caps = caps_with(&[("cloud", &[Capability::Chat]), ("local", &[Capability::Chat])]);
        let cands = vec![
            cand("cloud", "openrouter", BackendKind::OpenRouter, true, Some(0.8)),
            cand("local", "ollama", BackendKind::Ollama, true, Some(0.8)),
        ];
        assert_eq!(resolve(&desc(Task::Chat), &caps, &cands).unwrap().model, "local");
    }

    #[test]
    fn task_descriptor_json_roundtrip() {
        let json = r#"{"task":"vision_qa","input":["text","image"],"constraints":{"local_only":true,"quality_tier":"max"}}"#;
        let d: TaskDescriptor = serde_json::from_str(json).unwrap();
        assert_eq!(d.task, Task::VisionQa);
        assert_eq!(d.constraints.local_only, Some(true));
        assert_eq!(d.constraints.quality_tier, Some(QualityTier::Max));
    }
}
