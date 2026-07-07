//! SRV-04 integration tests: serving-profile reader → runtime launcher.
//!
//! End-to-end (with mocked launcher + health-checker, no real process/network):
//! a caller asks for a model → the routing map resolves its profile → the
//! launcher constructs the correct runtime command, launches, and health-checks;
//! fallback on failure; keep_warm delegated to the residency manager; unprofiled
//! model → clear error; no infra strings leak.
//!
//! Tests that read runtime-endpoint config helpers are `#[serial]` (shared env).

use async_trait::async_trait;
use serial_test::serial;
use std::sync::Mutex;

use chord_proxy::serving::{
    build_launch_command, FailureRecorder, HealthChecker, LaunchCommand, LaunchError, Launcher,
    ProfileSource, ResidencyError, ResidencyManager, RoutingMap, RuntimeSpawner, Slot,
    StaticProfileSource,
};
use terminus_rs::intake::serving::{
    ExclusionReason, ModelId, RecheckTrigger, Runtime, ServingBackend, ServingProfile,
};

fn profile_row(
    model: &str,
    backend: ServingBackend,
    best: Runtime,
    env_json: &str,
    keep_warm: bool,
    fallback: Option<Runtime>,
) -> ServingProfile {
    ServingProfile {
        model_id: ModelId::from(model),
        backend_tag: backend,
        best_runtime: best,
        env_json: env_json.into(),
        tok_s: Some(25.0),
        vram_or_ram_peak_gb: Some(9.0),
        cold_load_s: Some(15.0),
        keep_warm,
        fallback_runtime: fallback,
        exclusion_reason: ExclusionReason::None,
        recheck_trigger: RecheckTrigger::None,
        provenance: None,
    }
}

fn set_runtime_endpoints() {
    std::env::set_var("LLAMA_SERVER_URL", "http://llama.invalid/health");
    std::env::set_var("OLLAMA_URL", "http://ollama.invalid/health");
    std::env::set_var("OLLAMA_CPU_URL", "http://ollama-cpu.invalid/health");
}

// ── scripted collaborators ───────────────────────────────────────────────────

struct Spawner {
    fail: Vec<Runtime>,
    calls: Mutex<Vec<Runtime>>,
}
#[async_trait]
impl RuntimeSpawner for Spawner {
    async fn spawn(&self, cmd: &LaunchCommand) -> Result<(), String> {
        self.calls.lock().unwrap().push(cmd.runtime);
        if self.fail.contains(&cmd.runtime) {
            Err("scripted".into())
        } else {
            Ok(())
        }
    }
}

struct Health(bool);
#[async_trait]
impl HealthChecker for Health {
    async fn check(&self, _e: &str) -> bool {
        self.0
    }
}

struct Rec(Mutex<Vec<(String, Runtime, String)>>);
impl FailureRecorder for Rec {
    fn record_failure(&self, m: &str, r: Runtime, reason: &str) {
        self.0.lock().unwrap().push((m.into(), r, reason.into()));
    }
}

struct Residency {
    called: Mutex<bool>,
}
#[async_trait]
impl ResidencyManager for Residency {
    async fn acquire_warm_slot(
        &self,
        model_id: &ModelId,
        _vram_gb: Option<f64>,
    ) -> Result<Slot, ResidencyError> {
        *self.called.lock().unwrap() = true;
        Ok(Slot {
            model_id: model_id.as_str().to_string(),
            runtime: Runtime::LlamaCpp,
            endpoint: "http://warm.invalid/health".into(),
            netns: None,
        })
    }
}

struct PanicResidency;
#[async_trait]
impl ResidencyManager for PanicResidency {
    async fn acquire_warm_slot(
        &self,
        _m: &ModelId,
        _v: Option<f64>,
    ) -> Result<Slot, ResidencyError> {
        panic!("residency must not be consulted for a non-keep-warm model");
    }
}

// ── tests ────────────────────────────────────────────────────────────────────

#[tokio::test]
#[serial]
async fn request_routes_to_correct_runtime_and_health_checks() {
    set_runtime_endpoints();
    let src = StaticProfileSource::new(vec![profile_row(
        "qwen3:8b",
        ServingBackend::LlamaGpu,
        Runtime::LlamaCpp,
        r#"{"mmap":false,"gfx_override":"11.0.0"}"#,
        false,
        Some(Runtime::Ollama),
    )]);
    let map = RoutingMap::load(&src).await.unwrap();
    let entry = map.get(&ModelId::from("qwen3:8b")).expect("routed");

    let spawner = Spawner {
        fail: vec![],
        calls: Mutex::new(vec![]),
    };
    let health = Health(true);
    let rec = Rec(Mutex::new(vec![]));
    let resi = PanicResidency;
    let launcher = Launcher::new(&spawner, &health, &rec, &resi);

    let handle = launcher
        .serve_model(&ModelId::from("qwen3:8b"), entry, "/w/q.gguf")
        .await
        .unwrap();
    assert_eq!(handle.runtime, Runtime::LlamaCpp);
    assert!(!handle.from_warm_slot);
    assert_eq!(*spawner.calls.lock().unwrap(), vec![Runtime::LlamaCpp]);

    // The constructed command carried the --no-mmap flag (verified independently).
    let cmd = build_launch_command(entry, Runtime::LlamaCpp, "/w/q.gguf").unwrap();
    assert!(cmd.args.iter().any(|a| a == "--no-mmap"));
}

#[tokio::test]
#[serial]
async fn launch_failure_falls_back_then_genericizes() {
    set_runtime_endpoints();
    let src = StaticProfileSource::new(vec![profile_row(
        "m",
        ServingBackend::LlamaGpu,
        Runtime::LlamaCpp,
        "{}",
        false,
        Some(Runtime::Ollama),
    )]);
    let map = RoutingMap::load(&src).await.unwrap();
    let entry = map.get(&ModelId::from("m")).unwrap();

    // Both runtimes fail → genericized terminal error.
    let spawner = Spawner {
        fail: vec![Runtime::LlamaCpp, Runtime::Ollama],
        calls: Mutex::new(vec![]),
    };
    let rec = Rec(Mutex::new(vec![]));
    let launcher = Launcher::new(&spawner, &Health(true), &rec, &PanicResidency);
    let err = launcher
        .serve_model(&ModelId::from("m"), entry, "/w/m.gguf")
        .await
        .unwrap_err();
    assert!(matches!(err, LaunchError::AllRuntimesFailed(_)));
    // Both attempted.
    assert_eq!(
        *spawner.calls.lock().unwrap(),
        vec![Runtime::LlamaCpp, Runtime::Ollama]
    );
    // No infra leak in the surfaced error.
    let s = err.to_string();
    assert!(!s.contains("://"));
    assert!(!s.contains(".gguf"));
    assert!(!s.contains("invalid"));
    assert!(!s.contains("192.168."));
}

#[tokio::test]
#[serial]
async fn keep_warm_is_delegated_not_cold_launched() {
    set_runtime_endpoints();
    let src = StaticProfileSource::new(vec![profile_row(
        "big:120b",
        ServingBackend::LlamaGpu,
        Runtime::LlamaCpp,
        "{}",
        true, // keep_warm
        Some(Runtime::Ollama),
    )]);
    let map = RoutingMap::load(&src).await.unwrap();
    let entry = map.get(&ModelId::from("big:120b")).unwrap();

    let spawner = Spawner {
        fail: vec![],
        calls: Mutex::new(vec![]),
    };
    let resi = Residency {
        called: Mutex::new(false),
    };
    let rec = Rec(Mutex::new(vec![]));
    let launcher = Launcher::new(&spawner, &Health(true), &rec, &resi);

    let handle = launcher
        .serve_model(&ModelId::from("big:120b"), entry, "/w/big.gguf")
        .await
        .unwrap();
    assert!(handle.from_warm_slot);
    // Residency consulted.
    assert!(*resi.called.lock().unwrap());
    // NEGATIVE: the cold-launch spawner was NEVER touched.
    assert!(
        spawner.calls.lock().unwrap().is_empty(),
        "keep_warm model must route through ResidencyManager, not cold-launch"
    );
}

#[tokio::test]
#[serial]
async fn unprofiled_model_is_a_clear_error_no_guess() {
    set_runtime_endpoints();
    let src = StaticProfileSource::new(vec![profile_row(
        "qwen3:8b",
        ServingBackend::LlamaGpu,
        Runtime::LlamaCpp,
        "{}",
        false,
        None,
    )]);
    let map = RoutingMap::load(&src).await.unwrap();
    // A model with NO row → no entry → caller turns the None into an UnprofiledModel
    // error rather than guessing a runtime.
    let missing = ModelId::from("never-profiled:1b");
    let entry = map.get(&missing);
    assert!(entry.is_none());

    // Build the error a caller would surface and assert it is clear + leak-free.
    let err = LaunchError::UnprofiledModel(missing.as_str().to_string());
    let s = err.to_string();
    assert!(s.contains("never-profiled:1b"));
    assert!(s.contains("no serving profile"));
    assert!(!s.contains("://") && !s.contains("192.168."));
}

#[tokio::test]
async fn routing_map_loads_all_rows_keyed_by_model() {
    let src = StaticProfileSource::new(vec![
        profile_row("a:1b", ServingBackend::LlamaGpu, Runtime::LlamaCpp, "{}", false, None),
        profile_row("b:2b", ServingBackend::OllamaGpu, Runtime::Ollama, "{}", false, None),
    ]);
    let rows = src.load_all().await.unwrap();
    assert_eq!(rows.len(), 2);
    let map = RoutingMap::load_from(rows);
    assert_eq!(map.len(), 2);
    assert!(map.get(&ModelId::from("a:1b")).is_some());
    assert!(map.get(&ModelId::from("b:2b")).is_some());
}
