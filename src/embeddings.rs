//! EMBED-01: `POST /v1/embeddings` — OpenAI-compatible embeddings proxy.
//!
//! Serves embeddings **local-first** from the fleet Ollama (Qwen3-Embedding)
//! and falls back to OpenRouter (same Qwen3-Embedding model) when the local
//! backend is unreachable, errors, times out, or returns a vector of the
//! wrong dimensionality. Both sides use the same model family so the
//! resulting vectors are compatible with each other in downstream storage —
//! a caller must never be able to tell (from vector shape) which path served
//! a given request, and Chord must never hand back a vector whose length
//! doesn't match [`EmbeddingsConfig::dim`].
//!
//! ## Config
//! Colocated with the feature (same pattern as `snap::config::SnapConfig` /
//! `sweep_status::config::SweepMonitorConfig`) rather than growing the
//! central `config::Config` — see [`EmbeddingsConfig::from_env`].
//!
//! ## Secrets
//! `OPENROUTER_API_KEY` is never read as an author-time literal and never
//! logged. It is fetched from <secret-manager> at process startup into the shared
//! `DOWNSTREAM_SECRET_KEYS` allowlist (see `secrets_bootstrap.rs`, the same
//! mechanism `CHORD_JWT_SECRET`/`CHORD_API_KEY` use) and read fresh via
//! [`openrouter_api_key`] at dispatch time — this module never touches the
//! env var by any other name or path.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;

// ── Config ───────────────────────────────────────────────────────────────────

/// Runtime configuration for the embeddings proxy. All fields are sourced
/// from env — no infra host/model is ever a compiled-in literal (S1), except
/// the public `openrouter.ai` API host, which the rest of this codebase
/// already treats as a safe, non-PII default (see `models::backends`,
/// `router::policy::DEFAULT_CLOUD_EGRESS_HOST`).
#[derive(Debug, Clone)]
pub struct EmbeddingsConfig {
    /// Full URL of the local (fleet Ollama) embeddings endpoint. Reads
    /// `EMBED_LOCAL_URL`. `None` (unset/blank) means local is not configured
    /// — every request goes straight to the OpenRouter fallback, same
    /// "unreachable" treatment as a live connection failure.
    pub local_url: Option<String>,
    /// Model name requested from the local backend. Reads `EMBED_LOCAL_MODEL`,
    /// default `qwen3-embedding` (Ollama's Qwen3-Embedding tag).
    pub local_model: String,
    /// Model name requested from the OpenRouter fallback. Reads
    /// `EMBED_FALLBACK_MODEL`, default `qwen/qwen3-embedding` — **must** stay
    /// the same model family as `local_model` so both paths emit
    /// dimensionally- and semantically-compatible vectors.
    pub fallback_model: String,
    /// OpenRouter API base (no trailing slash, no `/embeddings` suffix).
    /// Reads `EMBED_FALLBACK_BASE_URL`, default
    /// `https://openrouter.ai/api/v1`.
    pub fallback_base_url: String,
    /// Expected embedding dimensionality. Every vector returned by either
    /// backend is asserted against this — a mismatch is treated as a backend
    /// failure (triggers fallback, or a structured 5xx if both fail). Reads
    /// `EMBED_DIM`, default 1024 (Qwen3-Embedding).
    pub dim: usize,
    /// Maximum number of inputs accepted in one request. Reads
    /// `EMBED_MAX_BATCH_SIZE`, default 256.
    pub max_batch: usize,
    /// Per-backend request timeout. Reads `EMBED_TIMEOUT_SECS`, default 30.
    pub request_timeout: Duration,
}

fn non_empty_env(key: &str) -> Option<String> {
    std::env::var(key).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty())
}

impl EmbeddingsConfig {
    pub fn from_env() -> Self {
        EmbeddingsConfig {
            local_url: non_empty_env("EMBED_LOCAL_URL"),
            local_model: non_empty_env("EMBED_LOCAL_MODEL")
                .unwrap_or_else(|| "qwen3-embedding".to_string()),
            fallback_model: non_empty_env("EMBED_FALLBACK_MODEL")
                .unwrap_or_else(|| "qwen/qwen3-embedding".to_string()),
            fallback_base_url: non_empty_env("EMBED_FALLBACK_BASE_URL")
                .unwrap_or_else(|| "https://openrouter.ai/api/v1".to_string()),
            dim: non_empty_env("EMBED_DIM")
                .and_then(|v| v.parse().ok())
                .unwrap_or(1024),
            max_batch: non_empty_env("EMBED_MAX_BATCH_SIZE")
                .and_then(|v| v.parse().ok())
                .unwrap_or(256),
            request_timeout: Duration::from_secs(
                non_empty_env("EMBED_TIMEOUT_SECS")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(30),
            ),
        }
    }

    /// Test-support constructor that never touches the process environment,
    /// so unit tests (in this crate and in `tests/e2e.rs`, which links the
    /// crate as an external dependency and therefore cannot see `#[cfg(test)]`
    /// items) can point `local_url`/`fallback_base_url` at an `httpmock`
    /// server deterministically (no env races with other `#[serial]` tests).
    /// Deliberately NOT `#[cfg(test)]`-gated for that reason — same tradeoff
    /// already accepted for other always-visible test-support constructors
    /// in this workspace.
    pub fn test_default(local_url: Option<String>, fallback_base_url: String) -> Self {
        EmbeddingsConfig {
            local_url,
            local_model: "qwen3-embedding".to_string(),
            fallback_model: "qwen/qwen3-embedding".to_string(),
            fallback_base_url,
            dim: 1024,
            max_batch: 256,
            request_timeout: Duration::from_secs(5),
        }
    }
}

/// The one sanctioned place this module reads `OPENROUTER_API_KEY`. The
/// value is populated into the process environment at startup by
/// `secrets_bootstrap::fetch_and_apply_downstream_secrets` (<secret-manager>-first,
/// falling back to a static environment) — never a literal, never logged,
/// never returned in any response or error text.
pub fn openrouter_api_key() -> Option<String> {
    non_empty_env("OPENROUTER_API_KEY")
}

// ── Wire types (OpenAI-compatible) ──────────────────────────────────────────

/// `input` may be a single string or an array of strings (OpenAI contract).
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingsInput {
    One(String),
    Many(Vec<String>),
}

impl EmbeddingsInput {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            EmbeddingsInput::One(s) => vec![s],
            EmbeddingsInput::Many(v) => v,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingsRequest {
    pub input: EmbeddingsInput,
    /// Accepted for OpenAI-client compatibility but currently informational
    /// only — Chord always serves `EMBED_LOCAL_MODEL`/`EMBED_FALLBACK_MODEL`
    /// (the pinned, dimensionally-matched pair) rather than honoring an
    /// arbitrary caller-requested model name.
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingObject {
    pub object: &'static str,
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingsUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingsResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    pub usage: EmbeddingsUsage,
}

// ── Errors ───────────────────────────────────────────────────────────────────

/// Structured, safe-to-serialize errors — never includes raw input text.
#[derive(Debug, Clone)]
pub enum EmbeddingsError {
    EmptyInput,
    BatchTooLarge { max: usize, got: usize },
    /// Both the local and fallback backends failed (network/HTTP/parse
    /// error, or a wrong-dimension vector). Carries a short, safe diagnostic
    /// string per side — never a full response body, never input text.
    BothBackendsFailed { local: String, fallback: String },
}

impl EmbeddingsError {
    pub fn status(&self) -> axum::http::StatusCode {
        use axum::http::StatusCode;
        match self {
            EmbeddingsError::EmptyInput | EmbeddingsError::BatchTooLarge { .. } => {
                StatusCode::BAD_REQUEST
            }
            EmbeddingsError::BothBackendsFailed { .. } => StatusCode::BAD_GATEWAY,
        }
    }

    pub fn to_json(&self) -> Value {
        match self {
            EmbeddingsError::EmptyInput => {
                serde_json::json!({ "error": "input must not be empty" })
            }
            EmbeddingsError::BatchTooLarge { max, got } => serde_json::json!({
                "error": format!("batch too large: {got} inputs exceeds max {max}")
            }),
            EmbeddingsError::BothBackendsFailed { local, fallback } => serde_json::json!({
                "error": "embeddings unavailable: both local and fallback backends failed",
                "local": local,
                "fallback": fallback,
            }),
        }
    }
}

// ── Response parsing (tolerant of the OpenAI-style and Ollama-style shapes) ──

/// Parse a backend's JSON body into ordered embedding vectors. Understands
/// three response shapes so the same code path works against an
/// OpenAI-compatible backend (OpenRouter, or an OpenAI-compatible Ollama
/// build) and Ollama's native `/api/embed(dings)` shapes:
///   - OpenAI: `{"data": [{"embedding": [...], "index": n}, ...]}` — sorted
///     by `index` so caller order is preserved regardless of backend order.
///   - Ollama batch: `{"embeddings": [[...], [...]]}` — already in order.
///   - Ollama legacy single: `{"embedding": [...]}` — only valid when exactly
///     one input was sent.
fn parse_embedding_vectors(body: &Value, expected_count: usize) -> Result<Vec<Vec<f32>>, String> {
    if let Some(data) = body.get("data").and_then(Value::as_array) {
        let mut items: Vec<(usize, Vec<f32>)> = Vec::with_capacity(data.len());
        for (i, item) in data.iter().enumerate() {
            let idx = item
                .get("index")
                .and_then(Value::as_u64)
                .map(|v| v as usize)
                .unwrap_or(i);
            let emb = item
                .get("embedding")
                .and_then(Value::as_array)
                .ok_or_else(|| "response 'data' item missing 'embedding' array".to_string())?;
            let vec: Vec<f32> = emb.iter().filter_map(Value::as_f64).map(|x| x as f32).collect();
            items.push((idx, vec));
        }
        items.sort_by_key(|(i, _)| *i);
        return Ok(items.into_iter().map(|(_, v)| v).collect());
    }

    if let Some(embeddings) = body.get("embeddings").and_then(Value::as_array) {
        return embeddings
            .iter()
            .map(|e| {
                e.as_array()
                    .ok_or_else(|| "malformed 'embeddings' entry".to_string())
                    .map(|arr| arr.iter().filter_map(Value::as_f64).map(|x| x as f32).collect())
            })
            .collect();
    }

    if let Some(embedding) = body.get("embedding").and_then(Value::as_array) {
        if expected_count != 1 {
            return Err(
                "backend returned a single 'embedding' field for a multi-input batch".to_string(),
            );
        }
        let vec: Vec<f32> = embedding.iter().filter_map(Value::as_f64).map(|x| x as f32).collect();
        return Ok(vec![vec]);
    }

    Err("unrecognized embeddings response shape (no data/embeddings/embedding field)".to_string())
}

fn assert_dims(vectors: &[Vec<f32>], dim: usize) -> Result<(), String> {
    for (i, v) in vectors.iter().enumerate() {
        if v.len() != dim {
            return Err(format!(
                "embedding at index {i} has dimension {} (expected {dim})",
                v.len()
            ));
        }
    }
    Ok(())
}

/// Full output validation for a backend's parsed vectors: the response must
/// contain **exactly one embedding per input** (OpenAI contract + the
/// ordered-output guarantee — a truncated or padded `data` array silently
/// misaligns every downstream vector-to-input mapping), and every vector must
/// match `dim`. A violation of EITHER is treated as that backend's failure by
/// [`route_embeddings`] (local → fall back; fallback → structured 5xx), so a
/// wrong-count response is never surfaced to the caller.
fn validate_output(vectors: &[Vec<f32>], expected_count: usize, dim: usize) -> Result<(), String> {
    if vectors.len() != expected_count {
        return Err(format!(
            "backend returned {} embeddings for {expected_count} inputs (count mismatch)",
            vectors.len()
        ));
    }
    assert_dims(vectors, dim)
}

// ── Backend calls ────────────────────────────────────────────────────────────

async fn call_local(
    client: &reqwest::Client,
    cfg: &EmbeddingsConfig,
    inputs: &[String],
) -> Result<Vec<Vec<f32>>, String> {
    let url = cfg
        .local_url
        .as_ref()
        .ok_or_else(|| "EMBED_LOCAL_URL not configured".to_string())?;

    let body = serde_json::json!({
        "model": cfg.local_model,
        "input": inputs,
    });

    let resp = client
        .post(url)
        .timeout(cfg.request_timeout)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("local embeddings request failed: {e}"))?;

    let status = resp.status();
    if !status.is_success() {
        // Covers Ollama's 404 "model not pulled" case too — treated as a
        // plain local-backend failure, which triggers fallback.
        return Err(format!("local embeddings backend returned HTTP {status}"));
    }

    let json: Value = resp
        .json()
        .await
        .map_err(|e| format!("local embeddings response not JSON: {e}"))?;

    parse_embedding_vectors(&json, inputs.len())
}

async fn call_openrouter(
    client: &reqwest::Client,
    cfg: &EmbeddingsConfig,
    inputs: &[String],
    api_key: Option<&str>,
) -> Result<Vec<Vec<f32>>, String> {
    let Some(key) = api_key.map(str::trim).filter(|k| !k.is_empty()) else {
        return Err("OPENROUTER_API_KEY not configured".to_string());
    };

    let url = format!("{}/embeddings", cfg.fallback_base_url.trim_end_matches('/'));
    let body = serde_json::json!({
        "model": cfg.fallback_model,
        "input": inputs,
    });

    let resp = client
        .post(&url)
        .timeout(cfg.request_timeout)
        .bearer_auth(key)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("fallback embeddings request failed: {e}"))?;

    let status = resp.status();
    if !status.is_success() {
        // Covers OpenRouter 401/429/etc — never echoes the response body
        // (it could reflect request contents / the key).
        return Err(format!("fallback embeddings backend returned HTTP {status}"));
    }

    let json: Value = resp
        .json()
        .await
        .map_err(|e| format!("fallback embeddings response not JSON: {e}"))?;

    parse_embedding_vectors(&json, inputs.len())
}

// ── Routing (local-first → fallback, with a hard dimension guard) ──────────

/// Which backend actually served the response — logged/metric'd, never
/// exposed as anything sensitive (no host, no key).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedSource {
    Local,
    Fallback,
}

impl EmbedSource {
    pub fn as_str(&self) -> &'static str {
        match self {
            EmbedSource::Local => "local",
            EmbedSource::Fallback => "fallback",
        }
    }
}

#[derive(Debug)]
pub struct EmbedOutcome {
    pub vectors: Vec<Vec<f32>>,
    pub source: EmbedSource,
    pub model: String,
}

/// Local-first, OpenRouter-fallback routing with a hard dimension guard.
/// Never returns a vector whose length != `cfg.dim` — a dimension mismatch
/// from the local backend triggers the fallback exactly like a network
/// error would; a dimension mismatch from the fallback itself is a hard
/// failure (`BothBackendsFailed`).
pub async fn route_embeddings(
    client: &reqwest::Client,
    cfg: &EmbeddingsConfig,
    inputs: &[String],
    openrouter_key: Option<&str>,
) -> Result<EmbedOutcome, EmbeddingsError> {
    let local_err: String = match call_local(client, cfg, inputs).await {
        Ok(vectors) => match validate_output(&vectors, inputs.len(), cfg.dim) {
            Ok(()) => {
                return Ok(EmbedOutcome {
                    vectors,
                    source: EmbedSource::Local,
                    model: cfg.local_model.clone(),
                });
            }
            Err(e) => {
                tracing::warn!("embeddings: local backend returned invalid output, falling back: {e}");
                format!("invalid output: {e}")
            }
        },
        Err(e) => {
            tracing::warn!("embeddings: local backend failed, falling back: {e}");
            e
        }
    };

    match call_openrouter(client, cfg, inputs, openrouter_key).await {
        Ok(vectors) => match validate_output(&vectors, inputs.len(), cfg.dim) {
            Ok(()) => Ok(EmbedOutcome {
                vectors,
                source: EmbedSource::Fallback,
                model: cfg.fallback_model.clone(),
            }),
            Err(e) => Err(EmbeddingsError::BothBackendsFailed {
                local: local_err,
                fallback: format!("invalid output: {e}"),
            }),
        },
        Err(e) => Err(EmbeddingsError::BothBackendsFailed {
            local: local_err,
            fallback: e,
        }),
    }
}

/// Validate + normalize a request's `input` into an ordered `Vec<String>`.
/// `Err` covers the empty-input and oversized-batch 400 cases.
pub fn validate_inputs(
    req: EmbeddingsRequest,
    max_batch: usize,
) -> Result<Vec<String>, EmbeddingsError> {
    let inputs = req.input.into_vec();
    if inputs.is_empty() || inputs.iter().all(|s| s.is_empty()) {
        return Err(EmbeddingsError::EmptyInput);
    }
    if inputs.len() > max_batch {
        return Err(EmbeddingsError::BatchTooLarge {
            max: max_batch,
            got: inputs.len(),
        });
    }
    Ok(inputs)
}

/// Build the final OpenAI-shaped response from a routed outcome. Token
/// counts are a cheap whitespace-split estimate (Chord has no tokenizer for
/// arbitrary embedding models) — good enough for the `usage` block's
/// informational purpose, never used for billing/rate-limit decisions.
pub fn build_response(inputs: &[String], outcome: EmbedOutcome) -> EmbeddingsResponse {
    let prompt_tokens: usize = inputs.iter().map(|s| s.split_whitespace().count()).sum();
    let data = outcome
        .vectors
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| EmbeddingObject {
            object: "embedding",
            embedding,
            index,
        })
        .collect();
    EmbeddingsResponse {
        object: "list",
        data,
        model: outcome.model,
        usage: EmbeddingsUsage {
            prompt_tokens,
            total_tokens: prompt_tokens,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use httpmock::prelude::*;

    fn inputs(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    /// `serde_json::json!` doesn't support Rust's `[x; n]` array-repeat
    /// syntax inside its own array literal parsing, so mock fixture vectors
    /// are built with this helper and interpolated as a plain expression.
    fn vecf(x: f64, n: usize) -> Vec<f64> {
        vec![x; n]
    }

    // ── request shape / order preservation ──────────────────────────────

    #[test]
    fn deserialize_string_input() {
        let req: EmbeddingsRequest =
            serde_json::from_str(r#"{"input":"hello world","model":"whatever"}"#).unwrap();
        assert_eq!(req.input.into_vec(), vec!["hello world".to_string()]);
    }

    #[test]
    fn deserialize_array_input_preserves_order() {
        let req: EmbeddingsRequest =
            serde_json::from_str(r#"{"input":["a","b","c"]}"#).unwrap();
        assert_eq!(req.input.into_vec(), vec!["a", "b", "c"]);
    }

    #[test]
    fn validate_inputs_rejects_empty() {
        let req: EmbeddingsRequest = serde_json::from_str(r#"{"input":[]}"#).unwrap();
        let err = validate_inputs(req, 256).unwrap_err();
        assert!(matches!(err, EmbeddingsError::EmptyInput));
    }

    #[test]
    fn validate_inputs_rejects_oversized_batch() {
        let many: Vec<String> = (0..10).map(|i| format!("item-{i}")).collect();
        let req = EmbeddingsRequest {
            input: EmbeddingsInput::Many(many),
            model: None,
        };
        let err = validate_inputs(req, 5).unwrap_err();
        match err {
            EmbeddingsError::BatchTooLarge { max, got } => {
                assert_eq!(max, 5);
                assert_eq!(got, 10);
            }
            other => panic!("expected BatchTooLarge, got {other:?}"),
        }
    }

    #[test]
    fn validate_inputs_ok_within_bounds() {
        let req: EmbeddingsRequest = serde_json::from_str(r#"{"input":["x","y"]}"#).unwrap();
        let ok = validate_inputs(req, 5).unwrap();
        assert_eq!(ok, vec!["x", "y"]);
    }

    // ── parse_embedding_vectors: all three backend response shapes ─────

    #[test]
    fn parse_openai_style_response_sorts_by_index() {
        let body = serde_json::json!({
            "data": [
                {"object":"embedding","index":1,"embedding":[3.0,4.0]},
                {"object":"embedding","index":0,"embedding":[1.0,2.0]}
            ]
        });
        let v = parse_embedding_vectors(&body, 2).unwrap();
        assert_eq!(v, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    }

    #[test]
    fn parse_ollama_batch_style_response() {
        let body = serde_json::json!({ "embeddings": [[1.0, 2.0], [3.0, 4.0]] });
        let v = parse_embedding_vectors(&body, 2).unwrap();
        assert_eq!(v, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    }

    #[test]
    fn parse_ollama_legacy_single_response() {
        let body = serde_json::json!({ "embedding": [1.0, 2.0, 3.0] });
        let v = parse_embedding_vectors(&body, 1).unwrap();
        assert_eq!(v, vec![vec![1.0, 2.0, 3.0]]);
    }

    #[test]
    fn parse_legacy_single_response_rejected_for_multi_input() {
        let body = serde_json::json!({ "embedding": [1.0, 2.0, 3.0] });
        assert!(parse_embedding_vectors(&body, 2).is_err());
    }

    #[test]
    fn parse_unrecognized_shape_errors() {
        let body = serde_json::json!({ "nonsense": true });
        assert!(parse_embedding_vectors(&body, 1).is_err());
    }

    // ── dimension guard ──────────────────────────────────────────────────

    #[test]
    fn assert_dims_ok() {
        assert!(assert_dims(&[vec![0.0; 1024]], 1024).is_ok());
    }

    #[test]
    fn assert_dims_rejects_mismatch() {
        assert!(assert_dims(&[vec![0.0; 768]], 1024).is_err());
    }

    // ── routing: local-first / fallback / both-fail (mocked HTTP) ──────

    #[tokio::test]
    async fn local_success_returns_local_vectors_no_fallback_call() {
        let local = MockServer::start_async().await;
        let fallback = MockServer::start_async().await;

        let local_mock = local.mock(|when, then| {
            when.method(POST).path("/embed");
            then.status(200)
                .json_body(serde_json::json!({ "embeddings": [vecf(1.0, 1024)] }));
        });
        // Fallback must never be hit on a clean local success.
        let fallback_mock = fallback.mock(|when, then| {
            when.method(POST);
            then.status(200)
                .json_body(serde_json::json!({ "embeddings": [vecf(2.0, 1024)] }));
        });

        let cfg = EmbeddingsConfig::test_default(
            Some(format!("{}/embed", local.base_url())),
            fallback.base_url(),
        );
        let client = reqwest::Client::new();
        let outcome = route_embeddings(&client, &cfg, &inputs(&["hi"]), Some("sk-test"))
            .await
            .unwrap();

        assert_eq!(outcome.source, EmbedSource::Local);
        assert_eq!(outcome.vectors, vec![vec![1.0; 1024]]);
        local_mock.assert();
        fallback_mock.assert_hits(0);
    }

    #[tokio::test]
    async fn local_unreachable_falls_back_to_openrouter() {
        let fallback = MockServer::start_async().await;
        let fallback_mock = fallback.mock(|when, then| {
            when.method(POST).path("/embeddings");
            then.status(200)
                .json_body(serde_json::json!({ "data": [{"embedding": vecf(9.0, 1024), "index": 0}] }));
        });

        // Port 1 on loopback refuses connections — a reliable "unreachable".
        let cfg = EmbeddingsConfig::test_default(
            Some("http://127.0.0.1:1/embed".to_string()),
            fallback.base_url(),
        );
        let client = reqwest::Client::new();
        let outcome = route_embeddings(&client, &cfg, &inputs(&["hi"]), Some("sk-test"))
            .await
            .unwrap();

        assert_eq!(outcome.source, EmbedSource::Fallback);
        assert_eq!(outcome.vectors, vec![vec![9.0; 1024]]);
        fallback_mock.assert();
    }

    #[tokio::test]
    async fn local_404_model_not_pulled_falls_back() {
        let local = MockServer::start_async().await;
        let fallback = MockServer::start_async().await;

        local.mock(|when, then| {
            when.method(POST);
            then.status(404).json_body(serde_json::json!({"error":"model not found"}));
        });
        let fallback_mock = fallback.mock(|when, then| {
            when.method(POST).path("/embeddings");
            then.status(200)
                .json_body(serde_json::json!({ "embeddings": [vecf(5.0, 1024)] }));
        });

        let cfg = EmbeddingsConfig::test_default(
            Some(format!("{}/embed", local.base_url())),
            fallback.base_url(),
        );
        let client = reqwest::Client::new();
        let outcome = route_embeddings(&client, &cfg, &inputs(&["hi"]), Some("sk-test"))
            .await
            .unwrap();

        assert_eq!(outcome.source, EmbedSource::Fallback);
        fallback_mock.assert();
    }

    #[tokio::test]
    async fn openrouter_401_surfaces_as_structured_error_when_local_also_down() {
        let fallback = MockServer::start_async().await;
        fallback.mock(|when, then| {
            when.method(POST).path("/embeddings");
            then.status(401).json_body(serde_json::json!({"error":"unauthorized"}));
        });

        let cfg = EmbeddingsConfig::test_default(None, fallback.base_url());
        let client = reqwest::Client::new();
        let err = route_embeddings(&client, &cfg, &inputs(&["hi"]), Some("bad-key"))
            .await
            .unwrap_err();

        match err {
            EmbeddingsError::BothBackendsFailed { local, fallback } => {
                assert!(local.contains("not configured"));
                assert!(fallback.contains("401"));
            }
            other => panic!("expected BothBackendsFailed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn both_backends_unreachable_yields_structured_error() {
        let cfg = EmbeddingsConfig::test_default(
            Some("http://127.0.0.1:1/embed".to_string()),
            "http://127.0.0.1:1".to_string(),
        );
        let client = reqwest::Client::new();
        let err = route_embeddings(&client, &cfg, &inputs(&["hi"]), Some("sk-test"))
            .await
            .unwrap_err();
        assert!(matches!(err, EmbeddingsError::BothBackendsFailed { .. }));
    }

    #[tokio::test]
    async fn wrong_dim_vector_from_local_is_rejected_and_falls_back() {
        let local = MockServer::start_async().await;
        let fallback = MockServer::start_async().await;

        local.mock(|when, then| {
            when.method(POST);
            // Wrong dimension (768 instead of the configured 1024).
            then.status(200)
                .json_body(serde_json::json!({ "embeddings": [vecf(0.0, 768)] }));
        });
        let fallback_mock = fallback.mock(|when, then| {
            when.method(POST).path("/embeddings");
            then.status(200)
                .json_body(serde_json::json!({ "embeddings": [vecf(0.0, 1024)] }));
        });

        let cfg = EmbeddingsConfig::test_default(
            Some(format!("{}/embed", local.base_url())),
            fallback.base_url(),
        );
        let client = reqwest::Client::new();
        let outcome = route_embeddings(&client, &cfg, &inputs(&["hi"]), Some("sk-test"))
            .await
            .unwrap();

        assert_eq!(outcome.source, EmbedSource::Fallback);
        assert_eq!(outcome.vectors[0].len(), 1024);
        fallback_mock.assert();
    }

    #[tokio::test]
    async fn wrong_dim_vector_from_fallback_is_a_hard_failure_never_returned() {
        let fallback = MockServer::start_async().await;
        fallback.mock(|when, then| {
            when.method(POST).path("/embeddings");
            then.status(200)
                .json_body(serde_json::json!({ "embeddings": [vecf(0.0, 42)] }));
        });

        let cfg = EmbeddingsConfig::test_default(None, fallback.base_url());
        let client = reqwest::Client::new();
        let err = route_embeddings(&client, &cfg, &inputs(&["hi"]), Some("sk-test"))
            .await
            .unwrap_err();
        match err {
            EmbeddingsError::BothBackendsFailed { fallback, .. } => {
                assert!(fallback.contains("invalid output"));
                assert!(fallback.contains("dimension"));
            }
            other => panic!("expected BothBackendsFailed, got {other:?}"),
        }
    }

    // ── output cardinality (one embedding per input, ordered) ──────────

    #[test]
    fn validate_output_rejects_too_few_embeddings() {
        let vecs = vec![vec![0.0; 1024]];
        assert!(validate_output(&vecs, 2, 1024).is_err());
    }

    #[test]
    fn validate_output_rejects_too_many_embeddings() {
        let vecs = vec![vec![0.0; 1024], vec![0.0; 1024], vec![0.0; 1024]];
        assert!(validate_output(&vecs, 2, 1024).is_err());
    }

    #[test]
    fn validate_output_accepts_exact_count_and_dim() {
        let vecs = vec![vec![0.0; 1024], vec![0.0; 1024]];
        assert!(validate_output(&vecs, 2, 1024).is_ok());
    }

    #[tokio::test]
    async fn local_wrong_count_falls_back_to_openrouter() {
        let local = MockServer::start_async().await;
        let fallback = MockServer::start_async().await;

        // Local returns ONE embedding for a TWO-input request → count mismatch.
        local.mock(|when, then| {
            when.method(POST);
            then.status(200)
                .json_body(serde_json::json!({ "embeddings": [vecf(1.0, 1024)] }));
        });
        let fallback_mock = fallback.mock(|when, then| {
            when.method(POST).path("/embeddings");
            then.status(200).json_body(serde_json::json!({
                "data": [
                    {"embedding": vecf(2.0, 1024), "index": 0},
                    {"embedding": vecf(3.0, 1024), "index": 1}
                ]
            }));
        });

        let cfg = EmbeddingsConfig::test_default(
            Some(format!("{}/embed", local.base_url())),
            fallback.base_url(),
        );
        let client = reqwest::Client::new();
        let outcome = route_embeddings(&client, &cfg, &inputs(&["a", "b"]), Some("sk-test"))
            .await
            .unwrap();

        assert_eq!(outcome.source, EmbedSource::Fallback);
        assert_eq!(outcome.vectors.len(), 2);
        fallback_mock.assert();
    }

    #[tokio::test]
    async fn both_wrong_count_yields_structured_5xx() {
        let local = MockServer::start_async().await;
        let fallback = MockServer::start_async().await;

        // Both backends return one embedding for a two-input request.
        local.mock(|when, then| {
            when.method(POST);
            then.status(200)
                .json_body(serde_json::json!({ "embeddings": [vecf(1.0, 1024)] }));
        });
        fallback.mock(|when, then| {
            when.method(POST).path("/embeddings");
            then.status(200)
                .json_body(serde_json::json!({ "embeddings": [vecf(2.0, 1024)] }));
        });

        let cfg = EmbeddingsConfig::test_default(
            Some(format!("{}/embed", local.base_url())),
            fallback.base_url(),
        );
        let client = reqwest::Client::new();
        let err = route_embeddings(&client, &cfg, &inputs(&["a", "b"]), Some("sk-test"))
            .await
            .unwrap_err();
        match err {
            EmbeddingsError::BothBackendsFailed { local, fallback } => {
                assert!(local.contains("count mismatch"));
                assert!(fallback.contains("count mismatch"));
            }
            other => panic!("expected BothBackendsFailed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn no_openrouter_key_configured_is_a_clean_failure_not_a_panic() {
        let cfg = EmbeddingsConfig::test_default(None, "https://openrouter.ai/api/v1".to_string());
        let client = reqwest::Client::new();
        let err = route_embeddings(&client, &cfg, &inputs(&["hi"]), None)
            .await
            .unwrap_err();
        match err {
            EmbeddingsError::BothBackendsFailed { fallback, .. } => {
                assert!(fallback.contains("OPENROUTER_API_KEY"));
            }
            other => panic!("expected BothBackendsFailed, got {other:?}"),
        }
    }

    #[test]
    fn error_json_never_leaks_the_api_key() {
        let err = EmbeddingsError::BothBackendsFailed {
            local: "local down".to_string(),
            fallback: "fallback returned HTTP 401 Unauthorized".to_string(),
        };
        let json = err.to_json();
        let s = json.to_string();
        assert!(!s.contains("sk-"));
        assert!(!s.contains("Bearer"));
    }

    #[test]
    fn build_response_preserves_order_and_shape() {
        let outcome = EmbedOutcome {
            vectors: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            source: EmbedSource::Local,
            model: "qwen3-embedding".to_string(),
        };
        let resp = build_response(&inputs(&["hello world", "hi"]), outcome);
        assert_eq!(resp.object, "list");
        assert_eq!(resp.data.len(), 2);
        assert_eq!(resp.data[0].index, 0);
        assert_eq!(resp.data[1].index, 1);
        assert_eq!(resp.data[0].embedding, vec![1.0, 2.0]);
        assert_eq!(resp.usage.prompt_tokens, 3); // "hello world" (2) + "hi" (1)
    }
}
