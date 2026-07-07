//! Local Ollama `/api/ps` polling — which model(s) are currently loaded.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// One currently-loaded Ollama model, as reported by `GET /api/ps`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OllamaLoadedModel {
    pub name: String,
    pub size_vram: Option<i64>,
    pub expires_at: Option<String>,
}

/// Result of a single `/api/ps` poll.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OllamaPsStats {
    pub available: bool,
    pub models: Vec<OllamaLoadedModel>,
    pub error_message: Option<String>,
}

/// Poll `{base_url}/api/ps`. Never panics: any network/parse failure yields
/// `OllamaPsStats { available: false, .. }` with the reason logged at `warn`.
pub async fn query_ollama_ps(client: &reqwest::Client, base_url: &str) -> OllamaPsStats {
    let url = format!("{}/api/ps", base_url.trim_end_matches('/'));
    let resp = match client.get(&url).timeout(Duration::from_secs(5)).send().await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(target: "chord.sweep_status", error = %e, "ollama /api/ps request failed");
            return OllamaPsStats { available: false, models: vec![], error_message: Some(e.to_string()) };
        }
    };
    let body: serde_json::Value = match resp.json().await {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(target: "chord.sweep_status", error = %e, "ollama /api/ps response parse failed");
            return OllamaPsStats { available: false, models: vec![], error_message: Some(e.to_string()) };
        }
    };
    let models = parse_ps_models(&body);
    OllamaPsStats { available: true, models, error_message: None }
}

/// Pure parse of the `/api/ps` JSON body into loaded-model entries. Split out
/// from the HTTP call so the shape-handling logic is directly unit-testable.
pub fn parse_ps_models(body: &serde_json::Value) -> Vec<OllamaLoadedModel> {
    body.get("models")
        .and_then(|m| m.as_array())
        .map(|arr| {
            arr.iter()
                .map(|m| OllamaLoadedModel {
                    name: m
                        .get("name")
                        .or_else(|| m.get("model"))
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string(),
                    size_vram: m.get("size_vram").and_then(|v| v.as_i64()),
                    expires_at: m.get("expires_at").and_then(|v| v.as_str()).map(String::from),
                })
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_typical_ps_response() {
        let body = serde_json::json!({
            "models": [
                {
                    "name": "qwen3-coder:30b",
                    "model": "qwen3-coder:30b",
                    "size_vram": 21474836480i64,
                    "expires_at": "2026-07-02T18:30:00Z"
                }
            ]
        });
        let models = parse_ps_models(&body);
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "qwen3-coder:30b");
        assert_eq!(models[0].size_vram, Some(21474836480));
        assert_eq!(models[0].expires_at.as_deref(), Some("2026-07-02T18:30:00Z"));
    }

    #[test]
    fn empty_models_array_is_empty_vec() {
        let body = serde_json::json!({ "models": [] });
        assert!(parse_ps_models(&body).is_empty());
    }

    #[test]
    fn missing_models_key_is_empty_vec() {
        let body = serde_json::json!({});
        assert!(parse_ps_models(&body).is_empty());
    }

    #[test]
    fn falls_back_to_model_field_when_name_absent() {
        let body = serde_json::json!({ "models": [{ "model": "gpt-oss:20b" }] });
        let models = parse_ps_models(&body);
        assert_eq!(models[0].name, "gpt-oss:20b");
    }
}
