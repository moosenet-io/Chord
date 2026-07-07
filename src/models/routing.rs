//! Tag-aware backend routing + on-demand lifecycle glue (P5 step 6).
//!
//! Bridges chord's in-memory [`ModelRegistry`] (which owns the backend
//! definitions + model tags) to the `terminus-rs` lifecycle/inference helpers
//! (which run the actual systemctl/llama-server work). `chat_completions` calls
//! [`resolve_and_ensure`] to pick + start the right backend before forwarding;
//! a background [`idle_stop_sweep`] stops on-demand GPU backends that have gone
//! idle so no backend perpetually holds the GPU.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex;

use crate::models::backends::{Backend, BackendKind, Hardware};
use crate::models::registry::ModelRegistry;

use terminus_rs::intake::infer::{LaunchSpec as TLaunch, ResolvedBackend};
use terminus_rs::intake::lifecycle;

/// Convert a chord [`Backend`] (+ the requesting model's local path) into the
/// `terminus-rs` [`ResolvedBackend`] the lifecycle helpers consume.
fn to_resolved(
    b: &Backend,
    model_local_path: Option<String>,
    model_gguf_path: Option<String>,
) -> ResolvedBackend {
    ResolvedBackend {
        name: b.name.clone(),
        url: b.url.trim_end_matches('/').to_string(),
        kind: match b.kind {
            BackendKind::Ollama => "ollama",
            BackendKind::LlamaServer => "llama-server",
            BackendKind::Daemon => "daemon",
            // terminus-rs's lifecycle helpers only actively manage the
            // "llama-server" kind (start/stop a local process); every other
            // kind is treated as already up. OpenRouter has no local process
            // either — map it to "daemon" so `lifecycle::ensure_up` no-ops for
            // it exactly like DiffusionGemma, without needing a terminus-rs
            // change for a kind that crate doesn't need to know about.
            BackendKind::OpenRouter => "daemon",
        }
        .to_string(),
        hardware: match b.hardware {
            Hardware::Gpu => "gpu",
            Hardware::Cpu => "cpu",
        }
        .to_string(),
        always_on: b.always_on,
        unit: b.unit.clone(),
        launch: b.launch.as_ref().map(|l| TLaunch {
            bin: l.bin.clone(),
            args: l.args.clone(),
            model_arg: l.model_arg.clone(),
        }),
        model_local_path,
        model_gguf_path,
    }
}

/// Resolve `model`'s backend, start it on demand if needed, and return the
/// OpenAI chat-completions URL to forward to, plus the bearer API key to send
/// with it (`Some(key)` only for backends with `api_key_env` set, e.g.
/// OpenRouter — `None` for every local/unauthenticated backend). Returns
/// `None` (caller falls back to `CHORD_LLM_URL`) when no backend is defined or
/// the tagged backend could not be started — availability over strictness for
/// live chat.
///
/// The key is read fresh from the backend's `api_key_env`-named environment
/// variable on every call (never cached, never persisted) — see
/// `Backend::api_key_env` docs for why. A backend that names an env var whose
/// value is unset/empty at call time resolves to `None` (request goes out
/// unauthenticated and will fail upstream with the provider's own auth error,
/// same "availability over strictness, fail at the edge" posture as the rest
/// of this function).
pub async fn resolve_and_ensure(
    registry: &Arc<Mutex<ModelRegistry>>,
    registry_key: &str,
    model: &str,
) -> Option<(String, Option<String>)> {
    // Brief lock: snapshot the backend + the model's local path, then release so
    // a (possibly long) on-demand start does not block other requests.
    let (resolved, bearer_key) = {
        let reg = registry.lock().await;
        let b = reg.backend_for(registry_key)?.clone();
        let local = reg.get(registry_key).and_then(|r| r.local_path.clone());
        let gguf = reg.get(registry_key).and_then(|r| r.gguf_path.clone());
        let bearer_key = b
            .api_key_env
            .as_ref()
            .and_then(|env_name| std::env::var(env_name).ok())
            .filter(|v| !v.trim().is_empty());
        (to_resolved(&b, local, gguf), bearer_key)
    };

    if let Err(e) = lifecycle::ensure_up(&resolved, model).await {
        tracing::warn!(
            "routing: backend '{}' ensure_up failed for {model}: {e}; falling back to default",
            resolved.name
        );
        return None;
    }
    // ensure_up already touched the shared last-used file (read by the sweep).
    Some((format!("{}/v1/chat/completions", resolved.url), bearer_key))
}

/// Background task: every `interval`, stop each on-demand backend whose
/// `idle_stop_secs` has elapsed since its last request. Keeps the GPU free —
/// "no perpetual holds". Always-on / Ollama / daemon backends are never stopped.
pub async fn idle_stop_sweep(registry: Arc<Mutex<ModelRegistry>>, interval: Duration) {
    loop {
        tokio::time::sleep(interval).await;
        // Snapshot on-demand backends + their idle thresholds.
        let candidates: Vec<(ResolvedBackend, u64)> = {
            let reg = registry.lock().await;
            reg.backends()
                .values()
                .filter(|b| b.on_demand() && b.idle_stop_secs > 0)
                .map(|b| (to_resolved(b, None, None), b.idle_stop_secs))
                .collect()
        };
        if candidates.is_empty() {
            continue;
        }
        for (backend, idle_threshold) in candidates {
            // Idle time comes from the shared file that lifecycle::ensure_up
            // touches on EVERY use (harness in-process AND chat path), so a
            // backend under active load is never stopped. `None` (never used
            // this boot) is NOT eligible — only stop backends we've seen used
            // and then go quiet.
            let Some(idle) = lifecycle::idle_secs(&backend.name) else {
                continue;
            };
            if idle >= idle_threshold {
                tracing::info!(
                    "routing: stopping idle backend '{}' (idle {idle}s ≥ {idle_threshold}s)",
                    backend.name,
                );
                lifecycle::stop(&backend);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::backends::LaunchSpec;

    #[test]
    fn to_resolved_maps_enums_and_launch() {
        let b = Backend {
            name: "llama-gpu".into(),
            url: "http://localhost:8082/".into(),
            hardware: Hardware::Gpu,
            kind: BackendKind::LlamaServer,
            unit: None,
            always_on: false,
            idle_stop_secs: 600,
            launch: Some(LaunchSpec {
                bin: "/x/llama-server".into(),
                args: vec!["-ngl".into(), "999".into()],
                model_arg: "-m".into(),
                model_from: "ollama-blob".into(),
            }),
            api_key_env: None,
        };
        let r = to_resolved(&b, Some("/opt/ollama-models".into()), None);
        assert_eq!(r.kind, "llama-server");
        assert_eq!(r.hardware, "gpu");
        assert_eq!(r.url, "http://localhost:8082"); // trailing slash trimmed
        assert!(!r.always_on);
        assert_eq!(r.model_local_path.as_deref(), Some("/opt/ollama-models"));
        let l = r.launch.unwrap();
        assert_eq!(l.bin, "/x/llama-server");
        assert_eq!(l.model_arg, "-m");
    }

    #[test]
    fn to_resolved_maps_openrouter_kind_to_daemon() {
        // OpenRouter has no local process for terminus-rs's lifecycle helpers
        // to manage — it maps to "daemon" (assumed always up), same as any
        // other externally-managed backend.
        let b = Backend {
            name: "openrouter".into(),
            url: "https://openrouter.ai/api".into(),
            hardware: Hardware::Cpu,
            kind: BackendKind::OpenRouter,
            unit: None,
            always_on: true,
            idle_stop_secs: 0,
            launch: None,
            api_key_env: Some("OPENROUTER_API_KEY_CHORDHARMONY".into()),
        };
        let r = to_resolved(&b, None, None);
        assert_eq!(r.kind, "daemon");
        assert!(r.always_on);
        assert!(r.launch.is_none());
    }

    #[tokio::test]
    async fn resolve_and_ensure_returns_bearer_key_for_openrouter_backend() {
        // Env var is read fresh inside resolve_and_ensure, keyed off the
        // backend's api_key_env — never stored in the Backend/ModelRecord.
        std::env::set_var("TEST_OWL_ALPHA_KEY_VAR", "sk-or-v1-test-value-not-real");

        let mut reg = ModelRegistry::new(
            std::path::PathBuf::from("/nonexistent/chord-test-registry.json"),
            std::path::PathBuf::from("/nonexistent/local"),
            std::path::PathBuf::from("/nonexistent/archive"),
            vec![],
        );
        reg.upsert_backend(Backend {
            name: "openrouter".into(),
            url: "http://127.0.0.1:0".into(), // unreachable on purpose; ensure_up no-ops for "daemon"
            hardware: Hardware::Cpu,
            kind: BackendKind::OpenRouter,
            unit: None,
            always_on: true,
            idle_stop_secs: 0,
            launch: None,
            api_key_env: Some("TEST_OWL_ALPHA_KEY_VAR".into()),
        });
        assert!(reg.register_remote_api_model("openrouter/owl-alpha", "openrouter-api", "openrouter"));
        let registry = Arc::new(Mutex::new(reg));

        let result = resolve_and_ensure(&registry, "openrouter/owl-alpha", "openrouter/owl-alpha").await;
        let (url, bearer) = result.expect("openrouter backend should resolve");
        assert_eq!(url, "http://127.0.0.1:0/v1/chat/completions");
        assert_eq!(bearer.as_deref(), Some("sk-or-v1-test-value-not-real"));

        std::env::remove_var("TEST_OWL_ALPHA_KEY_VAR");
    }
}
