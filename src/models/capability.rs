//! S125 CH-CAP-01: the model **capability / modality** taxonomy.
//!
//! Before S125, a [`ModelRecord`](crate::models::registry::ModelRecord) carried only a
//! backend tag and hardware class — Chord had no notion of *what a model can do*. That is
//! exactly why MINT profiling (and Chord routing) could only ever reason about chat/code
//! on ollama. This module gives every model a set of [`Capability`]s so the unified
//! TaskDescriptor selector (CH-SEL-01) can pick a model *by what the request needs*
//! (vision, embedding, transcription, …) rather than by a raw model name.
//!
//! Capabilities are declared per model (config/registration) and can be refreshed from
//! MINT operational profiles (CH-CAP-02). Modality (what inputs/outputs a capability
//! implies) is *derived* from the capability set so callers never hand-maintain it.

use serde::{Deserialize, Serialize};

/// A single thing a model can do. Mirrors the MINT `FleetCategory` discovery taxonomy plus
/// the finer task splits the selector needs. A model may have several (e.g. a coder model
/// is `[Chat, Code]`; a VLM is `[Chat, Vlm]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    /// General instruct/chat completion.
    Chat,
    /// Deep/multi-step reasoning (a "deep" tier chat model).
    Reasoning,
    /// Code generation / editing.
    Code,
    /// Text embedding (returns a vector, not text).
    Embed,
    /// Cross-encoder reranking of (query, passage) pairs.
    Rerank,
    /// Vision-language: image (+text) -> text.
    Vlm,
    /// OCR / document text extraction.
    Ocr,
    /// Structured document / PDF parsing -> fields/JSON.
    Doc,
    /// Text -> image generation.
    ImageGen,
    /// Text -> speech (audio out).
    Tts,
    /// Speech -> text transcription (audio in).
    Stt,
    /// Small/fast tool-selection or intent-routing classifier.
    ToolRouter,
    /// Block-diffusion language model (DiffusionGemma-class).
    Diffusion,
}

/// An input/output modality a request or model deals in. Derived from [`Capability`]s.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    Text,
    Image,
    Audio,
    Pdf,
    Embedding,
}

impl Capability {
    /// The primary INPUT modalities this capability consumes (beyond the always-present
    /// text prompt). E.g. `Vlm` also consumes `Image`; `Stt` consumes `Audio`.
    pub fn input_modalities(self) -> &'static [Modality] {
        match self {
            Capability::Vlm => &[Modality::Text, Modality::Image],
            Capability::Ocr | Capability::Doc => &[Modality::Pdf, Modality::Image],
            Capability::Stt => &[Modality::Audio],
            Capability::Rerank => &[Modality::Text],
            _ => &[Modality::Text],
        }
    }

    /// The OUTPUT modality this capability produces.
    pub fn output_modality(self) -> Modality {
        match self {
            Capability::Embed => Modality::Embedding,
            Capability::ImageGen => Modality::Image,
            Capability::Tts => Modality::Audio,
            _ => Modality::Text,
        }
    }
}

/// Union of the input modalities across a capability set (deduped, text-first).
pub fn input_modalities_of(caps: &[Capability]) -> Vec<Modality> {
    let mut out: Vec<Modality> = Vec::new();
    for c in caps {
        for m in c.input_modalities() {
            if !out.contains(m) {
                out.push(*m);
            }
        }
    }
    if out.is_empty() {
        out.push(Modality::Text);
    }
    out
}

/// Union of the output modalities across a capability set.
pub fn output_modalities_of(caps: &[Capability]) -> Vec<Modality> {
    let mut out: Vec<Modality> = Vec::new();
    for c in caps {
        let m = c.output_modality();
        if !out.contains(&m) {
            out.push(m);
        }
    }
    if out.is_empty() {
        out.push(Modality::Text);
    }
    out
}

/// S125 CH-CAP-01/02: a config-driven, MINT-refreshable map of `model name -> capabilities`.
/// Kept SEPARATE from [`ModelRegistry`](crate::models::registry::ModelRegistry) on purpose:
/// capabilities are routing *metadata* (declared in config, refreshed from MINT operational
/// profiles) rather than intrinsic model identity, so this avoids bloating every scanned/
/// snapshotted `ModelRecord`. The unified selector (CH-SEL-01) consults this alongside the
/// model registry (for backend/availability).
#[derive(Debug, Clone, Default)]
pub struct CapabilityRegistry {
    map: std::collections::HashMap<String, Vec<Capability>>,
}

impl CapabilityRegistry {
    /// Load from the JSON file at `CHORD_CAPABILITIES_PATH` (shape:
    /// `{ "<model>": ["chat","code"], ... }`). Missing/unset/parse-error ⇒ an empty
    /// registry (never panics — capabilities simply default to unknown/chat-only downstream).
    pub fn from_env() -> Self {
        match std::env::var("CHORD_CAPABILITIES_PATH") {
            Ok(p) if !p.trim().is_empty() => Self::from_path(p.trim()),
            _ => Self::default(),
        }
    }

    /// Load from an explicit path; empty registry on any error.
    pub fn from_path(path: &str) -> Self {
        let map = std::fs::read_to_string(path)
            .ok()
            .and_then(|t| serde_json::from_str::<std::collections::HashMap<String, Vec<Capability>>>(&t).ok())
            .unwrap_or_default();
        Self { map }
    }

    /// The declared capabilities of `model` (empty slice if unknown).
    pub fn of(&self, model: &str) -> &[Capability] {
        self.map.get(model).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Does `model` declare `cap`?
    pub fn has(&self, model: &str, cap: Capability) -> bool {
        self.of(model).contains(&cap)
    }

    /// Every known model that declares `cap` (for the selector's candidate set).
    pub fn models_with(&self, cap: Capability) -> Vec<&str> {
        self.map
            .iter()
            .filter(|(_, caps)| caps.contains(&cap))
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Declare/replace a model's capabilities (CH-CAP-02: MINT operational-profile refresh,
    /// or explicit config). Empty ⇒ removes the entry.
    pub fn set(&mut self, model: impl Into<String>, caps: Vec<Capability>) {
        let model = model.into();
        if caps.is_empty() {
            self.map.remove(&model);
        } else {
            self.map.insert(model, caps);
        }
    }

    /// Number of models with declared capabilities.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_registry_query_and_refresh() {
        let mut reg = CapabilityRegistry::default();
        reg.set("qwen2.5-vl:7b", vec![Capability::Chat, Capability::Vlm]);
        reg.set("qwen3-embedding", vec![Capability::Embed]);
        assert!(reg.has("qwen2.5-vl:7b", Capability::Vlm));
        assert!(!reg.has("qwen2.5-vl:7b", Capability::Embed));
        assert_eq!(reg.models_with(Capability::Vlm), vec!["qwen2.5-vl:7b"]);
        assert_eq!(reg.of("unknown-model"), &[] as &[Capability]);
        // set([]) removes.
        reg.set("qwen3-embedding", vec![]);
        assert!(reg.of("qwen3-embedding").is_empty());
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn capability_registry_parses_json() {
        let json = r#"{ "m-vlm": ["chat","vlm"], "m-embed": ["embed"] }"#;
        let tmp = std::env::temp_dir().join("chord_caps_test_s125.json");
        std::fs::write(&tmp, json).unwrap();
        let reg = CapabilityRegistry::from_path(tmp.to_str().unwrap());
        assert!(reg.has("m-vlm", Capability::Vlm));
        assert!(reg.has("m-embed", Capability::Embed));
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn capability_serde_is_snake_case() {
        assert_eq!(
            serde_json::to_string(&Capability::ImageGen).unwrap(),
            "\"image_gen\""
        );
        assert_eq!(
            serde_json::from_str::<Capability>("\"vlm\"").unwrap(),
            Capability::Vlm
        );
    }

    #[test]
    fn modality_derivation() {
        // A VLM coder consumes text+image, outputs text.
        let caps = [Capability::Chat, Capability::Code, Capability::Vlm];
        assert_eq!(
            input_modalities_of(&caps),
            vec![Modality::Text, Modality::Image]
        );
        assert_eq!(output_modalities_of(&caps), vec![Modality::Text]);

        // Embed outputs a vector; STT consumes audio.
        assert_eq!(
            output_modalities_of(&[Capability::Embed]),
            vec![Modality::Embedding]
        );
        assert_eq!(
            input_modalities_of(&[Capability::Stt]),
            vec![Modality::Audio]
        );
        assert_eq!(output_modalities_of(&[Capability::Tts]), vec![Modality::Audio]);
    }

    #[test]
    fn empty_caps_default_to_text() {
        assert_eq!(input_modalities_of(&[]), vec![Modality::Text]);
        assert_eq!(output_modalities_of(&[]), vec![Modality::Text]);
    }
}
