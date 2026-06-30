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
/// OpenAI chat-completions URL to forward to. Returns `None` (caller falls back
/// to `CHORD_LLM_URL`) when no backend is defined or the tagged backend could
/// not be started — availability over strictness for live chat.
pub async fn resolve_and_ensure(
    registry: &Arc<Mutex<ModelRegistry>>,
    registry_key: &str,
    model: &str,
) -> Option<String> {
    // Brief lock: snapshot the backend + the model's local path, then release so
    // a (possibly long) on-demand start does not block other requests.
    let resolved = {
        let reg = registry.lock().await;
        let b = reg.backend_for(registry_key)?.clone();
        let local = reg.get(registry_key).and_then(|r| r.local_path.clone());
        let gguf = reg.get(registry_key).and_then(|r| r.gguf_path.clone());
        to_resolved(&b, local, gguf)
    };

    if let Err(e) = lifecycle::ensure_up(&resolved, model).await {
        tracing::warn!(
            "routing: backend '{}' ensure_up failed for {model}: {e}; falling back to default",
            resolved.name
        );
        return None;
    }
    // ensure_up already touched the shared last-used file (read by the sweep).
    Some(format!("{}/v1/chat/completions", resolved.url))
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
}
