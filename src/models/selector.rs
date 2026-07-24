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
    /// ADVISORY: the input modalities the request carries. The `task` already determines the
    /// required capability (and hence the modalities), so these do not filter selection — they
    /// are surfaced for telemetry/validation and let a client be explicit. Validate with
    /// [`TaskDescriptor::modality_consistent`].
    #[serde(default)]
    pub input: Vec<Modality>,
    /// ADVISORY: the expected output modality (see `input`).
    #[serde(default)]
    pub output: Option<Modality>,
    #[serde(default)]
    pub constraints: Constraints,
    #[serde(default)]
    pub hints: Hints,
}

impl TaskDescriptor {
    /// Validate the ADVISORY `input`/`output` modalities against what the `task` implies (via
    /// its required capability). Returns `Ok(())` when consistent (or when the client left them
    /// unset), `Err(reason)` on a contradiction (e.g. `task: embed` but `output: image`). The
    /// route layer can 400 on this; `resolve` itself does not depend on it.
    pub fn modality_consistent(&self) -> Result<(), String> {
        let cap = self.task.required_capability();
        if let Some(out) = self.output {
            let expected = cap.output_modality();
            if out != expected {
                return Err(format!(
                    "output modality {:?} contradicts task {:?} (expects {:?})",
                    out, self.task, expected
                ));
            }
        }
        for m in &self.input {
            if !cap.input_modalities().contains(m) && *m != Modality::Text {
                return Err(format!(
                    "input modality {:?} not consumed by task {:?}",
                    m, self.task
                ));
            }
        }
        Ok(())
    }
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
    /// A NaN is treated as "no score" for ranking (never poisons the order).
    pub score: Option<f64>,
    /// False for remote/cloud backends (OpenRouter). Drives `local_only`.
    pub is_local: bool,
    /// Estimated latency (ms) for the task, from the serving/MINT profile; `None` ⇒ unknown
    /// (not excluded by `max_latency_ms` — fail-open).
    pub est_latency_ms: Option<u64>,
    /// Estimated per-request cost; `None` ⇒ unknown (not excluded by `max_cost`).
    pub est_cost: Option<f64>,
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
    let need = desc.task.required_capability();

    // (1) Build the servable pool: capability-declared, satisfies ALL constraints, and is
    //     AVAILABLE (registered/loadable now). preferred_model is honored ONLY from within
    //     this validated pool — an override never bypasses capability/constraint/availability.
    let pool: Vec<&ModelCandidate> = candidates
        .iter()
        .filter(|c| c.available)
        .filter(|c| caps.has(&c.name, need))
        .filter(|c| satisfies_constraints(c, &desc.constraints))
        .collect();
    if pool.is_empty() {
        return None;
    }

    let tier = desc.constraints.quality_tier.unwrap_or(QualityTier::Balanced);

    // (0'/preferred) Honor an explicit preference IFF it survived the filters above. When the
    // preferred NAME exists on multiple backends, pick the best-ranked one (deterministic —
    // never "whichever was supplied first").
    if let Some(pref) = desc.hints.preferred_model.as_deref() {
        if let Some(c) = pool
            .iter()
            .copied()
            .filter(|c| c.name == pref)
            .max_by(|a, b| cmp_candidates(a, b, tier))
        {
            return Some(Selection {
                model: c.name.clone(),
                backend: c.backend.clone(),
                reason: "preferred_model override (capability+constraint-valid)".to_string(),
            });
        }
    }

    // (4) Rank and pick (total, deterministic order — NaN-safe).
    let best = pool.iter().copied().max_by(|a, b| cmp_candidates(a, b, tier))?;
    Some(Selection {
        model: best.name.clone(),
        backend: best.backend.clone(),
        reason: format!(
            "capability={:?} tier={:?} score={:?} local={}",
            need, tier, best.score, best.is_local
        ),
    })
}

/// Does a candidate satisfy every hard constraint? Unknown per-candidate data (context/
/// latency/cost `None`) is FAIL-OPEN — only a KNOWN violation excludes.
fn satisfies_constraints(c: &ModelCandidate, k: &Constraints) -> bool {
    if k.local_only.unwrap_or(false) && !c.is_local {
        return false;
    }
    if let Some(min_ctx) = k.context_tokens {
        if let Some(m) = c.max_context {
            if m < min_ctx {
                return false;
            }
        }
    }
    if let Some(max_lat) = k.max_latency_ms {
        if let Some(l) = c.est_latency_ms {
            if l > max_lat {
                return false;
            }
        }
    }
    if let Some(max_cost) = k.max_cost {
        if let Some(cost) = c.est_cost {
            if cost > max_cost {
                return false;
            }
        }
    }
    true
}

/// Sanitize a score to a finite, orderable value: `None`/NaN ⇒ −∞ (ranks last).
fn score_of(c: &ModelCandidate) -> f64 {
    match c.score {
        Some(s) if !s.is_nan() => s,
        _ => f64::NEG_INFINITY,
    }
}

/// TOTAL order for "which candidate is better" (greater = better). NaN-safe via `score_of`
/// + `f64::total_cmp`, and fully deterministic (final tiebreak on name AND backend).
fn cmp_candidates(a: &ModelCandidate, b: &ModelCandidate, tier: QualityTier) -> Ordering {
    let score_cmp = score_of(a).total_cmp(&score_of(b));
    match tier {
        // Max: on a score tie prefer the larger context, then local.
        QualityTier::Max => score_cmp
            .then_with(|| a.max_context.unwrap_or(0).cmp(&b.max_context.unwrap_or(0)))
            .then_with(|| a.is_local.cmp(&b.is_local)),
        // Fast/Balanced: on a score tie prefer LOCAL (sovereign, no network latency), then
        // the lower estimated latency, then larger context.
        QualityTier::Fast | QualityTier::Balanced => score_cmp
            .then_with(|| a.is_local.cmp(&b.is_local))
            .then_with(|| {
                // lower latency is better → reverse (unknown latency sorts as "worst" = u64::MAX).
                b.est_latency_ms
                    .unwrap_or(u64::MAX)
                    .cmp(&a.est_latency_ms.unwrap_or(u64::MAX))
            })
            .then_with(|| a.max_context.unwrap_or(0).cmp(&b.max_context.unwrap_or(0))),
    }
    // Fully deterministic final tiebreak (name, then backend) — reversed so `max_by` yields
    // the lexicographically SMALLEST (name, backend), stable across runs and identical names
    // on different backends.
    .then_with(|| b.name.cmp(&a.name))
    .then_with(|| b.backend.cmp(&a.backend))
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
            est_latency_ms: None,
            est_cost: None,
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

    #[test]
    fn preferred_model_lacking_capability_is_ignored() {
        // preferred_model must NOT bypass the capability filter (review: codex HIGH).
        let caps = caps_with(&[("real-coder", &[Capability::Code]), ("chatty", &[Capability::Chat])]);
        let cands = vec![
            cand("real-coder", "ollama", BackendKind::Ollama, true, Some(0.5)),
            cand("chatty", "ollama", BackendKind::Ollama, true, Some(0.9)),
        ];
        let mut d = desc(Task::Code);
        d.hints.preferred_model = Some("chatty".to_string()); // not code-capable
        // Falls through to the real code-capable model, NOT the preferred non-coder.
        assert_eq!(resolve(&d, &caps, &cands).unwrap().model, "real-coder");
    }

    #[test]
    fn all_unavailable_returns_none() {
        let caps = caps_with(&[("a", &[Capability::Chat])]);
        let cands = vec![cand("a", "ollama", BackendKind::Ollama, false, Some(0.9))];
        assert!(resolve(&desc(Task::Chat), &caps, &cands).is_none());
    }

    #[test]
    fn nan_score_never_poisons_order() {
        // A NaN score must rank LAST (as "no score"), and never make the order non-transitive.
        let caps = caps_with(&[("nan", &[Capability::Chat]), ("good", &[Capability::Chat])]);
        let cands = vec![
            cand("nan", "ollama", BackendKind::Ollama, true, Some(f64::NAN)),
            cand("good", "ollama", BackendKind::Ollama, true, Some(0.3)),
        ];
        assert_eq!(resolve(&desc(Task::Chat), &caps, &cands).unwrap().model, "good");
    }

    #[test]
    fn latency_and_cost_constraints_filter_when_known() {
        let caps = caps_with(&[("slow", &[Capability::Chat]), ("fast", &[Capability::Chat])]);
        let mut slow = cand("slow", "ollama", BackendKind::Ollama, true, Some(0.9));
        slow.est_latency_ms = Some(5000);
        let mut fast = cand("fast", "ollama", BackendKind::Ollama, true, Some(0.4));
        fast.est_latency_ms = Some(200);
        let mut d = desc(Task::Chat);
        d.constraints.max_latency_ms = Some(1000);
        // slow (5s) excluded by the 1s budget despite higher score.
        assert_eq!(resolve(&d, &caps, &vec![slow, fast]).unwrap().model, "fast");
    }

    #[test]
    fn modality_consistency_validation() {
        let mut d = desc(Task::Embed);
        d.output = Some(Modality::Image); // embed outputs a vector, not an image
        assert!(d.modality_consistent().is_err());
        d.output = Some(Modality::Embedding);
        assert!(d.modality_consistent().is_ok());
        // VLM consumes image — consistent.
        let mut v = desc(Task::VisionQa);
        v.input = vec![Modality::Text, Modality::Image];
        assert!(v.modality_consistent().is_ok());
    }

    #[test]
    fn same_name_different_backend_is_deterministic() {
        // Identical name+score on two backends must resolve deterministically (no flapping).
        let caps = caps_with(&[("m", &[Capability::Chat])]);
        let cands = vec![
            cand("m", "vulkan", BackendKind::LlamaServer, true, Some(0.5)),
            cand("m", "ollama", BackendKind::Ollama, true, Some(0.5)),
        ];
        let a = resolve(&desc(Task::Chat), &caps, &cands).unwrap();
        let b = resolve(&desc(Task::Chat), &caps, &cands).unwrap();
        assert_eq!(a.backend, b.backend); // stable across calls
    }
}
