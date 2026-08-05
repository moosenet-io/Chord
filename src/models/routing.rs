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
use crate::serving::profile::RoutingMap;

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
        // terminus-rs >= 1.3.2: `ResolvedBackend` carries the backend's
        // api-key ENV VAR NAME (never a value), so the lower crate can read the
        // key fresh at its own call boundary exactly as `resolve_and_ensure`
        // does below. Threading Chord's own `api_key_env` through keeps the two
        // sides naming the same variable instead of silently disagreeing.
        api_key_env: b.api_key_env.clone(),
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
    routing_map: &Arc<Mutex<RoutingMap>>,
    registry_key: &str,
    model: &str,
) -> Option<(String, Option<String>)> {
    // ARCH-AWARE resolution (CHRD arch-aware fix): consult the serving
    // `RoutingMap` so a model whose registry-tagged backend is arch-EXCLUDED
    // (e.g. a `gpt-oss` model still tagged `llama-gpu`, which llama-server
    // cannot load) is steered to the map's chosen usable backend instead of
    // being dispatched to a backend that will crash-loop.
    //
    // The routing_map guard is taken FIRST and dropped BEFORE the registry lock
    // — we extract only small OWNED values (the excluded tiers + chosen tier for
    // THIS model). `list_models`/`get_model` lock the registry THEN the
    // routing_map; holding both here in the opposite order would be a lock-order
    // inversion (deadlock). Extracting-then-dropping removes the nested hold
    // entirely, so there is no ordering to invert.
    let (excluded_tiers, chosen_tier) = {
        let routing = routing_map.lock().await;
        let model_id = terminus_rs::intake::serving::ModelId::from(registry_key);
        (
            routing.excluded_tiers(&model_id),
            routing.chosen_backend(&model_id),
        )
    };

    // Brief lock: snapshot the backend + the model's local path, then release so
    // a (possibly long) on-demand start does not block other requests.
    let (resolved, bearer_key, arch_known) = {
        let reg = registry.lock().await;
        let b = reg
            .backend_for_arch_aware(registry_key, &excluded_tiers, chosen_tier)?
            .clone();
        let local = reg.get(registry_key).and_then(|r| r.local_path.clone());
        let gguf = reg.get(registry_key).and_then(|r| r.gguf_path.clone());
        // CHRD phase3: is this model's arch already recorded? (Drives the
        // serve-time profiler backfill below.)
        // "Known" means NONBLANK — an empty-string arch (`Some("")`) is treated
        // as unresolved (re-derive), consistent with `set_arch_if_absent` and the
        // routing guard, which both treat blank/empty arch as absent/unknown.
        let arch_known = crate::models::registry::arch_is_known(
            reg.get(registry_key).and_then(|r| r.arch.as_deref()),
        );
        let bearer_key = b
            .api_key_env
            .as_ref()
            .and_then(|env_name| std::env::var(env_name).ok())
            .filter(|v| !v.trim().is_empty());
        (to_resolved(&b, local, gguf), bearer_key, arch_known)
    };

    if let Err(e) = lifecycle::ensure_up(&resolved, model).await {
        tracing::warn!(
            "routing: backend '{}' ensure_up failed for {model}: {e}; falling back to default",
            resolved.name
        );
        return None;
    }

    // CHRD phase3 (profiler / task-26): the MINT sweep drives REAL serves through
    // this path. The first time a model actually serves and its architecture is
    // not yet known, derive it off-lock from disk and record it — so arch-aware
    // routing becomes data-driven through the normal serve/sweep flow, not only
    // via reconcile. Fully additive and non-blocking: it runs in a detached task
    // (never delays this response), reads the GGUF on a blocking thread (never on
    // the async runtime), and any failure simply leaves arch unset for now
    // (reconcile still populates it). Only fires for LOCAL models (a resolvable
    // local root) and only until arch is known — not a per-request cost.
    if !arch_known {
        if let Some(local_root) = resolved.model_local_path.clone() {
            let registry = registry.clone();
            let key = registry_key.to_string();
            tokio::spawn(async move {
                let key_for_read = key.clone();
                let derived = tokio::task::spawn_blocking(move || {
                    crate::models::registry::derive_arch_for_local_model(
                        std::path::Path::new(&local_root),
                        &key_for_read,
                    )
                })
                .await
                .ok()
                .flatten();
                if let Some(arch) = derived {
                    let mut reg = registry.lock().await;
                    if reg.record_served_arch(&key, &arch) {
                        tracing::debug!(
                            model = %key,
                            arch = %arch,
                            "CHRD phase3: recorded served arch (profiler backfill)"
                        );
                    }
                }
            });
        }
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

/// BLD-09 idle-mode provider-stop hook: UNCONDITIONALLY stop every on-demand
/// backend (regardless of its per-backend `idle_stop_secs`) so the heavy host's
/// GPU/RAM is freed for the compiler. This is the "stop/park a provider" primitive
/// the idle-mode admin API drives — the same `to_resolved` + `lifecycle::stop`
/// path as [`idle_stop_sweep`], but immediate and total rather than idle-gated.
///
/// Always-on / Ollama / daemon backends are NOT process-managed here (same as
/// `idle_stop_sweep`); the resident MODELS those hold are unloaded separately by
/// the idle handler (VRAM eviction via `gpu_exclusive::evict_resident_models`).
/// Returns the number of on-demand backends stopped (for the freed-RAM report).
/// Best-effort: a `lifecycle::stop` that no-ops (backend already down) still
/// counts as "stopped" from the caller's contract — the goal state is "not running".
pub async fn stop_all_on_demand_backends(registry: &Arc<Mutex<ModelRegistry>>) -> usize {
    let candidates: Vec<ResolvedBackend> = {
        let reg = registry.lock().await;
        reg.backends()
            .values()
            .filter(|b| b.on_demand())
            .map(|b| to_resolved(b, None, None))
            .collect()
    };
    let mut stopped = 0usize;
    for backend in candidates {
        tracing::info!(backend = %backend.name, "idle-mode: stopping on-demand backend");
        lifecycle::stop(&backend);
        stopped += 1;
    }
    stopped
}

/// RVXR-01: the SAFETY GATE for [`stop_on_demand_backend_for_model`], separated
/// from the side effect so it is unit-testable without stopping anything.
///
/// Returns the backend to stop, or `None` when there is nothing this function is
/// permitted to stop. **An `always_on` backend always yields `None`** — the
/// primary Ollama serve is the assistant's own engine, and stopping it to reclaim
/// a coder's memory would take live Lumina down to make room for a review. That
/// is the single most damaging thing this whole feature could do, so the gate is
/// its own named, tested function rather than a condition inside an effect.
pub fn on_demand_backend_to_stop(
    reg: &ModelRegistry,
    model: &str,
    excluded_tiers: &[terminus_rs::intake::serving::ServingBackend],
    chosen_tier: Option<terminus_rs::intake::serving::ServingBackend>,
) -> Option<ResolvedBackend> {
    // ARCH-AWARE, and that matters: [`resolve_and_ensure`] STARTS the coder
    // through the routing map's arch-aware selection, so a plain
    // `backend_for` here could resolve a DIFFERENT backend than the one that was
    // actually started — stopping something unrelated while the real coder stays
    // resident. Start and stop must agree on which backend they mean.
    match reg.backend_for_arch_aware(model, excluded_tiers, chosen_tier) {
        Some(b) if b.on_demand() => Some(to_resolved(b, None, None)),
        _ => None,
    }
}

/// RVXR-01: stop the ON-DEMAND backend serving exactly ONE model, via the same
/// `to_resolved` + `lifecycle::stop` path as [`idle_stop_sweep`] and
/// [`stop_all_on_demand_backends`] — a narrower address on the existing
/// primitive, not a second lifecycle.
///
/// **An always-on backend is NEVER stopped**, whatever the model claims: the
/// primary Ollama serve is the assistant's own engine, and stopping it to reclaim
/// a coder's memory would take the live assistant down to make room for a review.
/// [`Backend::on_demand`] is the single gate, and it is the same one the idle
/// sweep uses.
///
/// Returns `true` when an on-demand backend was addressed (stopping an already-
/// stopped backend is a successful no-op — the contract is the goal state "not
/// running", exactly as in [`stop_all_on_demand_backends`]), `false` when the
/// model has no on-demand backend to stop.
/// RVXR-01: would `model` be served by an ON-DEMAND backend?
///
/// The precondition for the coder tier LOADING anything: a coder that resolves to
/// the always-on serve is running on the assistant's own engine, and the stop gate
/// will (correctly) refuse to stop it — so it could never be evicted. Uses the
/// SAME arch-aware resolution as [`stop_on_demand_backend_for_model`], so start
/// and stop can never disagree about which backend they mean.
///
/// ## CHRD #112: the `free_gpu` refusal that used to live here is GONE, deliberately
/// RVXR-01 added a second gate here: refuse to load AT ALL while any always-on GPU
/// unit was registered, because `terminus-rs`'s `free_gpu` would `systemctl stop`
/// it on a cold start. That was a shield over one caller, and on the live GPU host
/// it refused every load — the coder tier could never run.
///
/// terminus-rs 1.3.2 fixes it at the point of action instead: `free_gpu` can only
/// iterate `gpu_stop_guard::StoppableGpuBackend` values, and the only constructor
/// of that type excludes always-on backends. There is no longer a caller-side check
/// to keep, for this caller or any other, so keeping this one would misinform the
/// next reader about a live hazard while disabling a working feature.
///
/// The removal is not taken on trust: `upstream_free_gpu_cannot_stop_the_always_on_engine`
/// (below) asserts the contract against the terminus-rs actually linked into this
/// build, and fails to COMPILE against a version that predates the fix.
pub async fn model_has_on_demand_backend(
    registry: &Arc<Mutex<ModelRegistry>>,
    routing_map: &Arc<Mutex<RoutingMap>>,
    model: &str,
) -> bool {
    let (excluded_tiers, chosen_tier) = {
        let routing = routing_map.lock().await;
        let model_id = terminus_rs::intake::serving::ModelId::from(model);
        (
            routing.excluded_tiers(&model_id),
            routing.chosen_backend(&model_id),
        )
    };
    let reg = registry.lock().await;
    on_demand_backend_to_stop(&reg, model, &excluded_tiers, chosen_tier).is_some()
}

pub async fn stop_on_demand_backend_for_model(
    registry: &Arc<Mutex<ModelRegistry>>,
    routing_map: &Arc<Mutex<RoutingMap>>,
    model: &str,
) -> bool {
    // Same lock ORDER as `resolve_and_ensure`: take the routing map first,
    // extract small owned values, drop it, then take the registry. Holding both
    // in the opposite order would be a lock-order inversion.
    let (excluded_tiers, chosen_tier) = {
        let routing = routing_map.lock().await;
        let model_id = terminus_rs::intake::serving::ModelId::from(model);
        (
            routing.excluded_tiers(&model_id),
            routing.chosen_backend(&model_id),
        )
    };
    let resolved = {
        let reg = registry.lock().await;
        on_demand_backend_to_stop(&reg, model, &excluded_tiers, chosen_tier)
    };
    match resolved {
        Some(backend) => {
            tracing::info!(
                backend = %backend.name,
                model = %model,
                "coder tier: stopping on-demand backend"
            );
            // `lifecycle::stop` shells out SYNCHRONOUSLY (`systemctl stop` via
            // `std::process::Command`). This runs on a detached teardown task
            // spawned from the inference hot path, so leaving it on a runtime
            // worker lets a process wait occupy a thread that an incoming user
            // request may need. Move it to the blocking pool.
            let b = backend.clone();
            let _ = tokio::task::spawn_blocking(move || lifecycle::stop(&b)).await;
            true
        }
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use crate::models::backends::LaunchSpec;

    /// RVXR-01: the coder tier must NEVER be able to stop the assistant's own
    /// always-on engine. Both directions asserted — the control matters as much
    /// as the guard: a gate that returns `None` for everything would also pass
    /// the negative case while making the feature inert.
    #[test]
    fn the_stop_gate_refuses_always_on_backends_and_permits_on_demand_ones() {
        use crate::models::backends::Hardware;
        use std::collections::HashMap;

        let on_demand = Backend {
            name: "llama-gpu".into(),
            url: "http://localhost:8082".into(),
            hardware: Hardware::Gpu,
            kind: BackendKind::LlamaServer,
            unit: None,
            always_on: false,
            idle_stop_secs: 600,
            launch: None,
            api_key_env: None,
        };
        let always_on = Backend {
            name: "ollama".into(),
            url: "http://localhost:11434".into(),
            hardware: Hardware::Gpu,
            kind: BackendKind::Ollama,
            unit: Some("ollama.service".into()),
            always_on: true,
            idle_stop_secs: 0,
            launch: None,
            api_key_env: None,
        };
        let mut catalogue = HashMap::new();
        catalogue.insert(on_demand.name.clone(), on_demand.clone());
        catalogue.insert(always_on.name.clone(), always_on.clone());

        let dir = std::env::temp_dir().join(format!("rvxr01-stopgate-{}", std::process::id()));
        let mut reg = ModelRegistry::new_with_backends(
            dir.join("registry.json"),
            dir.join("local"),
            dir.join("archive"),
            vec![],
            catalogue,
        );
        assert!(reg.register_remote_api_model("a-coder", "test", "llama-gpu"));
        assert!(reg.register_remote_api_model("the-assistant", "test", "ollama"));

        // CONTROL: an on-demand backend IS stoppable, else this test proves nothing.
        let target =
            on_demand_backend_to_stop(&reg, "a-coder", &[], None).expect("on-demand is stoppable");
        assert_eq!(target.name, "llama-gpu");

        // THE GUARD: the always-on assistant engine is never a stop target.
        assert!(
            on_demand_backend_to_stop(&reg, "the-assistant", &[], None).is_none(),
            "stopping the always-on serve would take live Lumina down to make room for a review"
        );
    }

    /// RVXR-01: STOP must resolve the SAME backend START did.
    ///
    /// `resolve_and_ensure` starts the coder through arch-aware selection, so a
    /// plain `backend_for` here would resolve a DIFFERENT backend whenever the
    /// registry tag is arch-excluded — stopping something unrelated while the
    /// real coder stays resident and holding memory. Raised by the review panel;
    /// this is the fixture where the two resolutions actually diverge (an empty
    /// exclusion set makes them agree, which is why the first test missed it).
    /// CHRD #112: the fix now lives UPSTREAM, so assert the upstream contract
    /// against the `terminus-rs` actually linked into THIS build.
    ///
    /// This is what replaces the deleted Chord-side refusal, and it is a stronger
    /// thing than the refusal was. The refusal was a check racing the file re-read
    /// that `free_gpu` itself performs; this asserts that `free_gpu` cannot be
    /// handed an always-on backend at all, because the only constructor of the
    /// value it iterates excludes them.
    ///
    /// It is also the version ratchet: against a `terminus-rs` predating 1.3.2
    /// this test does not COMPILE (`gpu_stop_guard` does not exist there), so a
    /// dependency downgrade cannot quietly restore the hazard the refusal used to
    /// cover.
    #[test]
    fn upstream_free_gpu_cannot_stop_the_always_on_engine() {
        use terminus_rs::intake::gpu_stop_guard::stoppable_gpu_backends_from_json;

        // Exactly the shape Chord seeds on the live GPU host.
        let live = r#"{"backends":{
            "ollama":{"name":"ollama","url":"http://x","hardware":"gpu","kind":"ollama",
                      "unit":"ollama.service","always_on":true},
            "llama-gpu":{"name":"llama-gpu","url":"http://y","hardware":"gpu",
                         "kind":"llama-server","always_on":false},
            "lemonade-coder":{"name":"lemonade-coder","url":"http://z","hardware":"gpu",
                              "kind":"llama-server","unit":"lemonade-coder.service",
                              "always_on":false}}}"#;

        let stoppable = stoppable_gpu_backends_from_json(live, "llama-gpu");
        assert!(
            !stoppable.iter().any(|b| b.name() == "ollama"),
            "starting a coder must never yield `systemctl stop ollama.service`: {stoppable:?}"
        );

        // CONTROL: on-demand GPU backends ARE still stoppable. Without this, a
        // guard that refused everything would pass the assertion above while
        // making GPU arbitration inert — every backend perpetually holding the GPU.
        let names: Vec<&str> = stoppable.iter().map(|b| b.name()).collect();
        assert_eq!(
            names,
            vec!["lemonade-coder"],
            "an on-demand GPU backend with a unit must stay stoppable — that is the design"
        );
    }

    /// CHRD #112 reconciliation: the LOAD precondition no longer refuses just
    /// because an always-on GPU engine is registered.
    ///
    /// This test is the inverse of the one it replaces, and that inversion IS the
    /// decision. RVXR-01 asserted that adding `ollama` (gpu + always_on + unit) to
    /// the registry made the coder unloadable — which on the live GPU host meant
    /// always. With the upstream fix, starting the coder can no longer stop that
    /// engine, so refusing would disable a working feature to guard a hazard that
    /// no longer exists.
    ///
    /// Both directions are still asserted: the precondition must answer the
    /// question it is actually for — "is there an on-demand backend to evict?" —
    /// and still say NO when there is not.
    #[tokio::test]
    async fn the_load_precondition_ignores_a_registered_always_on_engine() {
        use crate::models::backends::Hardware;
        use std::collections::HashMap;

        let coder_backend = Backend {
            name: "llama-gpu".into(),
            url: "http://localhost:8082".into(),
            hardware: Hardware::Gpu,
            kind: BackendKind::LlamaServer,
            unit: None,
            always_on: false,
            idle_stop_secs: 600,
            launch: None,
            api_key_env: None,
        };
        let assistant_engine = Backend {
            name: "ollama".into(),
            url: "http://localhost:11434".into(),
            hardware: Hardware::Gpu,
            kind: BackendKind::Ollama,
            unit: Some("ollama.service".into()),
            always_on: true,
            idle_stop_secs: 0,
            launch: None,
            api_key_env: None,
        };

        let dir = std::env::temp_dir().join(format!("chrd112-precond-{}", std::process::id()));
        let build = |cat: HashMap<String, Backend>, tag: &str, model_backend: &str| {
            let d = dir.join(tag);
            std::fs::create_dir_all(&d).unwrap();
            let mut reg = ModelRegistry::new_with_backends(
                d.join("registry.json"),
                d.join("local"),
                d.join("archive"),
                vec![],
                cat,
            );
            assert!(reg.register_remote_api_model("a-coder", "test", model_backend));
            reg.save().expect("registry file written");
            Arc::new(Mutex::new(reg))
        };
        let routing = Arc::new(Mutex::new(crate::serving::profile::RoutingMap::empty()));

        // THE RECONCILIATION: the always-on engine is registered, and the coder is
        // loadable anyway. Under RVXR-01 this asserted the opposite.
        let mut hazardous = HashMap::new();
        hazardous.insert(coder_backend.name.clone(), coder_backend.clone());
        hazardous.insert(assistant_engine.name.clone(), assistant_engine.clone());
        assert!(
            model_has_on_demand_backend(&build(hazardous, "with-engine", "llama-gpu"), &routing, "a-coder")
                .await,
            "starting the coder can no longer stop ollama.service (terminus-rs >= 1.3.2), \
             so the presence of the always-on engine must not veto the load"
        );

        // CONTROL: the precondition still answers its own question. A model served
        // BY the always-on engine has nothing on-demand to evict, so it is false —
        // otherwise this test would pass for a precondition hardwired to `true`.
        let mut only_engine = HashMap::new();
        only_engine.insert(assistant_engine.name.clone(), assistant_engine.clone());
        assert!(
            !model_has_on_demand_backend(&build(only_engine, "engine-only", "ollama"), &routing, "a-coder")
                .await,
            "a model on the always-on engine has no on-demand backend to evict"
        );
    }

    #[test]
    fn the_stop_gate_resolves_the_same_backend_arch_aware_start_would() {
        use crate::models::backends::Hardware;
        use std::collections::HashMap;

        let llama = Backend {
            name: "llama-gpu".into(),
            url: "http://localhost:8082".into(),
            hardware: Hardware::Gpu,
            kind: BackendKind::LlamaServer,
            unit: None,
            always_on: false,
            idle_stop_secs: 600,
            launch: None,
            api_key_env: None,
        };
        let ollama = Backend {
            name: "ollama".into(),
            url: "http://localhost:11434".into(),
            hardware: Hardware::Gpu,
            kind: BackendKind::Ollama,
            unit: Some("ollama.service".into()),
            always_on: true,
            idle_stop_secs: 0,
            launch: None,
            api_key_env: None,
        };
        let mut catalogue = HashMap::new();
        catalogue.insert(llama.name.clone(), llama);
        catalogue.insert(ollama.name.clone(), ollama);

        let dir = std::env::temp_dir().join(format!("rvxr01-archstop-{}", std::process::id()));
        let mut reg = ModelRegistry::new_with_backends(
            dir.join("registry.json"),
            dir.join("local"),
            dir.join("archive"),
            vec![],
            catalogue,
        );
        // Tagged to the llama-server backend, but of an architecture llama.cpp
        // cannot load — so arch-aware selection steers AWAY from that tag while a
        // plain tag lookup would still name it.
        assert!(reg.register_remote_api_model("odd-arch-model", "test", "llama-gpu"));
        assert!(reg.record_served_arch("odd-arch-model", "gptoss"));

        // CONTROL: the naive lookup really does name the on-demand backend, so
        // the assertion below is testing a difference that exists.
        assert_eq!(
            reg.backend_for("odd-arch-model").map(|b| b.name.as_str()),
            Some("llama-gpu"),
            "fixture must be one where the two resolutions DIVERGE"
        );

        // Arch-aware selection sends this model to the always-on ollama serve, so
        // there is no on-demand backend to stop. Stopping `llama-gpu` here would
        // be stopping a backend this model is not being served on.
        assert!(
            on_demand_backend_to_stop(&reg, "odd-arch-model", &[], None).is_none(),
            "stop must not target a backend arch-aware START would never have used"
        );
    }

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
    #[serial] // CHRD-94: mutates PROCESS-GLOBAL env; must not overlap another test that reads it
    async fn resolve_and_ensure_returns_bearer_key_for_openrouter_backend() {
        // Env var is read fresh inside resolve_and_ensure, keyed off the
        // backend's api_key_env — never stored in the Backend/ModelRecord.
        std::env::set_var("TEST_OWL_ALPHA_KEY_VAR", "<REDACTED-SECRET>");

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
        let routing_map = Arc::new(Mutex::new(RoutingMap::empty()));

        let result = resolve_and_ensure(
            &registry,
            &routing_map,
            "openrouter/owl-alpha",
            "openrouter/owl-alpha",
        )
        .await;
        let (url, bearer) = result.expect("openrouter backend should resolve");
        assert_eq!(url, "http://127.0.0.1:0/v1/chat/completions");
        assert_eq!(bearer.as_deref(), Some("<REDACTED-SECRET>"));

        std::env::remove_var("TEST_OWL_ALPHA_KEY_VAR");
    }
}
