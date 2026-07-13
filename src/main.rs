use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, warn};

use chord_proxy::{
    agentic::AgenticExecutor,
    audit::AuditLogger,
    config::Config,
    fallback::build_fallback_registry,
    mcp_proxy::McpProxy,
    models::eviction::{new_disk_op_lock, run_eviction_sweep, FsLocalEvictor},
    models::gc,
    models::registry::ModelRegistry,
    models::transfer::{PullCoordinator, StatvfsProbe},
    rate_limiter::ProxyRateLimiter,
    routes::{build_router, AppState},
    serving::profile::{DbProfileSource, RoutingMap},
};

#[tokio::main]
async fn main() {
    // Version flag — handled before any logging or service startup.
    if std::env::args().any(|a| a == "--version" || a == "-V") {
        println!("{}", chord_proxy::version::version_line());
        return;
    }

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("chord_proxy=info".parse().unwrap()),
        )
        .init();

    // CSEC-02: attempt to fetch CHORD_JWT_SECRET/CHORD_API_KEY fresh from
    // <secret-manager> and set them into the process environment BEFORE
    // Config::from_env() (below) or HarnessVramManager::from_env() (called
    // later, from harness_integration) read them — see src/secrets_bootstrap.rs
    // for the full fallback contract. Never a hard startup failure: falls back
    // to the static environment when <secret-manager> isn't configured or the fetch
    // fails.
    let secret_outcome = chord_proxy::secrets_bootstrap::fetch_and_apply_downstream_secrets().await;
    chord_proxy::secrets_bootstrap::log_secret_fetch_outcome(&secret_outcome);

    let config = Config::from_env().unwrap_or_else(|e| {
        eprintln!("Configuration error: {e}");
        std::process::exit(1);
    });

    let port = config.listen_port;
    let jwt_secret = config.jwt_secret.clone();
    let llm_backend_url = config.llm_backend_url.clone();
    let model_aliases = config.model_aliases.clone();
    if !model_aliases.is_empty() {
        info!("model aliases loaded: {} mapping(s)", model_aliases.len());
    }
    match &llm_backend_url {
        Some(url) => info!("LLM proxy enabled → {url}"),
        None => info!("LLM proxy disabled (CHORD_LLM_URL unset) — /v1/chat/completions returns 503"),
    }

    // Build terminus-rs registry with all compiled-in Rust tools
    let mut terminus = terminus_rs::ToolRegistry::new();
    terminus_rs::register_all(&mut terminus);
    info!("terminus-rs: {} tools registered", terminus.len());

    let fallback = Arc::new(build_fallback_registry(terminus));
    let proxy = McpProxy::new(&config, fallback);
    let proxy_arc = Arc::new(McpProxy::new(
        &config,
        Arc::new(chord_proxy::fallback::build_fallback_registry({
            let mut t = terminus_rs::ToolRegistry::new();
            terminus_rs::register_all(&mut t);
            t
        })),
    ));
    let agentic_executor = Arc::new(AgenticExecutor::new(proxy_arc));

    // ── Task 2 (federation): optional second McpProxy for terminus_personal ──
    // Only constructed when PERSONAL_BACKEND_URL is configured — Chord runs fine
    // with this unset (no hard dependency). Deliberately unfiltered (no
    // tool_allowlist::is_core_tool scoping) and reachable only via
    // /v1/personal/tools/{list,call}, never merged into the default catalog.
    let personal_proxy = config.personal_backend_url.clone().map(|url| {
        info!("personal backend federation enabled -> {url}");
        let mut personal_config = config.clone();
        personal_config.mcp_backend_url = url;
        personal_config.mcp_backend_token = config.personal_backend_token.clone();
        // No Rust fallback registry here (deliberately empty): this proxy's
        // whole purpose is a pure passthrough to terminus_personal's own
        // 147-tool catalog — it must never silently serve Chord's own
        // in-process Rust tools under the personal-catalog routes.
        Arc::new(McpProxy::new_unfiltered(
            &personal_config,
            Arc::new(chord_proxy::mcp_proxy::FallbackRegistry::new()),
        ))
    });
    if personal_proxy.is_none() {
        info!(
            "personal backend federation disabled (PERSONAL_BACKEND_URL unset) — /v1/personal/* returns 503"
        );
    }
    let audit_logger = Arc::new(AuditLogger::from_env());
    let rate_limiter = Arc::new(Mutex::new(ProxyRateLimiter::new(config.rate_limits.clone())));
    let http_client = reqwest::Client::builder()
        .build()
        .unwrap_or_else(|e| {
            eprintln!("Failed to build HTTP client: {e}");
            std::process::exit(1);
        });

    // ── Model registry + pull coordinator (TIER-01/02) ──
    // load_or_new never fails (corrupt JSON rebuilds empty); reconcile()/save()
    // are best-effort and must NOT crash startup.
    let mut model_registry = ModelRegistry::load_or_new(
        std::path::PathBuf::from(&config.model_registry_path),
        std::path::PathBuf::from(&config.model_local_path),
        std::path::PathBuf::from(&config.model_archive_path),
        config.model_protected.clone(),
    );
    model_registry.reconcile();
    // S80 DGEM-03: register DiffusionGemma (non-Ollama, llama-diffusion-daemon) after the Ollama-driven
    // reconcile, so it survives re-tiering and shows up in the control API / counts.
    model_registry.register_diffusiongemma_from_env();
    // Owl Alpha (OpenRouter) — opt-in, see `register_openrouter_owl_alpha_from_env` docs for why
    // this is gated behind OPENROUTER_OWL_ALPHA_ENABLED=1 rather than on-by-default.
    model_registry.register_openrouter_owl_alpha_from_env();
    let (hot, warm, cold) = model_registry.tier_counts();
    info!("model registry: {warm} warm, {cold} cold, {hot} hot");
    if let Err(e) = model_registry.save() {
        warn!("model registry: failed to persist after reconcile: {e}");
    }
    let model_registry = Arc::new(Mutex::new(model_registry));

    // ── TIER-03 eviction wiring ──
    // A shared disk-operation lock serialises the background sweep with pre-pull
    // eviction so their destructive filesystem ops never interleave.
    let disk_op_lock = new_disk_op_lock();
    let local_evictor: Arc<dyn chord_proxy::models::eviction::LocalEvictor> = Arc::new(
        FsLocalEvictor::new(std::path::PathBuf::from(&config.model_local_path)),
    );

    let archive_copy_timeout =
        std::time::Duration::from_secs(config.model_archive_copy_timeout_secs);

    let pull_coordinator = Arc::new(
        PullCoordinator::new(
            model_registry.clone(),
            std::time::Duration::from_secs(config.model_pull_timeout_secs),
        )
        .with_eviction(local_evictor.clone(), disk_op_lock.clone())
        .with_archive_copy_timeout(archive_copy_timeout),
    );

    // Background disk-pressure eviction sweep (non-fatal; logs and continues).
    //
    // MSM-01: every tick starts with reconcile() + an atomic persist so the
    // on-disk registry never lags in-memory reality by more than one sweep
    // interval, self-healing drift (e.g. a stale tier, a lost protected flag —
    // MSM-05) without waiting for a restart. Both reconcile and persist are
    // best-effort: a failure is logged and the sweep continues, it never aborts
    // startup or the loop.
    {
        let registry = model_registry.clone();
        let evictor = local_evictor.clone();
        let lock = disk_op_lock.clone();
        let threshold = config.model_disk_pressure_percent;
        let interval = config.model_sweep_interval_secs;
        let cooldown_hours = config.model_warm_cooldown_hours;
        let copy_timeout = archive_copy_timeout;
        let gc_min_age_secs = config.model_gc_min_age_secs;
        let local_path = std::path::PathBuf::from(&config.model_local_path);
        let archive_path = std::path::PathBuf::from(&config.model_archive_path);
        if cooldown_hours == 0 {
            warn!("MODEL_WARM_COOLDOWN_HOURS=0; cooldown eviction (warm→cold after inactivity) is DISABLED");
        }
        info!("eviction sweep task started, interval={interval}s, cooldown_hours={cooldown_hours}");
        tokio::spawn(async move {
            let probe = StatvfsProbe;
            let mut ticker = tokio::time::interval(std::time::Duration::from_secs(interval.max(1)));
            loop {
                ticker.tick().await;

                // MSM-01 + S111 fix: reconcile + persist before this tick's
                // eviction passes, but NEVER hold the registry lock across the
                // (multi-second, NFS) manifest scan. Do the slow scan in
                // spawn_blocking OFF the lock, then take the lock only for the
                // fast in-memory apply + persist (milliseconds) so concurrent
                // `chat_completions`/`update_last_requested` are never blocked
                // for more than that. Canonical lock order is disk_op_lock →
                // registry (see run_eviction_sweep); this reconcile block takes
                // ONLY the registry lock and never disk_op_lock, so it can't
                // invert that order.
                {
                    let scan = tokio::task::spawn_blocking({
                        let local_path = local_path.clone();
                        let archive_path = archive_path.clone();
                        move || ModelRegistry::scan_disk(local_path, archive_path)
                    })
                    .await;
                    match scan {
                        Ok(scan) => {
                            let mut reg = registry.lock().await;
                            reg.apply_reconcile(scan);
                            if let Err(e) = reg.save() {
                                warn!("eviction sweep: failed to persist registry after reconcile: {e}");
                            }
                        }
                        Err(e) => {
                            warn!("eviction sweep: reconcile scan task failed: {e}");
                        }
                    }
                }

                run_eviction_sweep(
                    &registry,
                    threshold,
                    cooldown_hours,
                    &probe,
                    evictor.as_ref(),
                    &lock,
                    copy_timeout,
                )
                .await;

                // MSM-01: persist again — eviction may have changed tiers since the
                // reconcile-time snapshot above (evict_to_archive already saves per
                // model, but this covers any change made outside that path).
                {
                    let reg = registry.lock().await;
                    if let Err(e) = reg.save() {
                        warn!("eviction sweep: failed to persist registry after eviction: {e}");
                    }
                }

                // MSM-03: orphan-blob GC, after eviction so newly-orphaned blobs
                // (from this sweep's evictions) are considered too. Best-effort;
                // failures are logged and never abort the sweep loop.
                let gc_result =
                    gc::run_gc(&registry, &local_path, &archive_path, &lock, gc_min_age_secs).await;
                if gc_result.orphans_deleted > 0 || !gc_result.errors.is_empty() {
                    info!(
                        orphans_deleted = gc_result.orphans_deleted,
                        freed_bytes = gc_result.freed_bytes,
                        errors = gc_result.errors.len(),
                        "eviction sweep: orphan-blob GC pass complete"
                    );
                }
            }
        });
    }

    // ── P5 idle-stop sweep: stop on-demand GPU backends when idle (no
    // perpetual holds). Lightweight (a registry snapshot every 60s).
    {
        let registry = Arc::clone(&model_registry);
        info!("backend idle-stop sweep started, interval=60s");
        tokio::spawn(async move {
            chord_proxy::models::routing::idle_stop_sweep(
                registry,
                std::time::Duration::from_secs(60),
            )
            .await;
        });
    }

    // ── YARN-06: SRV-04 serving-profile routing map ──
    // The source of a model's ThinkingConfig — capability advertisement
    // (`GET /api/models`) and per-request thinking honoring
    // (`/v1/chat/completions`) both read this. Best-effort, same fail-open
    // discipline as the model registry/eviction sweep above: an unconfigured
    // or unreachable intake DB yields an empty map (every lookup misses,
    // `thinking_available` reports `false`) rather than blocking startup —
    // Chord's core proxy/tooling function is independent of this feature.
    let routing_map = Arc::new(Mutex::new(RoutingMap::empty()));
    {
        let routing_map = routing_map.clone();
        tokio::spawn(async move {
            let Some(db_url) = terminus_rs::config::intake_database_url() else {
                info!(
                    "serving profile DB not configured — thinking capability/routing disabled \
                     (GET /api/models reports supports_thinking=false for all models)"
                );
                return;
            };
            let pool = match sqlx::PgPool::connect(&db_url).await {
                Ok(p) => p,
                Err(e) => {
                    warn!("serving profile DB connect failed: {e}");
                    return;
                }
            };
            let source = DbProfileSource::new(pool);
            match RoutingMap::load(&source).await {
                Ok(map) => {
                    let count = map.len();
                    *routing_map.lock().await = map;
                    info!("serving profile routing map loaded: {count} model(s)");
                }
                Err(e) => warn!("serving profile routing map load failed: {e}"),
            }
        });
    }

    // ── CPROX-02/03: fleet-driven coding-model selection ──
    // Same fail-open discipline as the serving-profile routing map above: an
    // unconfigured/unreachable intake DB yields `None` (POST /v1/coding/select
    // returns a clear 503, never blocks startup) rather than making Chord's
    // core proxy function depend on this feature.
    let coding_profile_source: chord_proxy::coding_proxy::SharedCodingProfileSource =
        Arc::new(Mutex::new(None));
    {
        let coding_profile_source = coding_profile_source.clone();
        tokio::spawn(async move {
            let Some(db_url) = terminus_rs::config::intake_database_url() else {
                info!(
                    "coding-model intake DB not configured — POST /v1/coding/select disabled \
                     (503 NotConfigured)"
                );
                return;
            };
            let pool = match sqlx::PgPool::connect(&db_url).await {
                Ok(p) => p,
                Err(e) => {
                    warn!("coding-model intake DB connect failed: {e}");
                    return;
                }
            };
            let source: Arc<dyn chord_proxy::models::coding_selector::CodeProfileSource> =
                Arc::new(chord_proxy::models::coding_selector::DbCodeProfileSource::new(pool));
            *coding_profile_source.lock().await = Some(source);
            info!("coding-model selection data source connected");
        });
    }

    let state = Arc::new(AppState {
        proxy,
        jwt_secret,
        audit_logger,
        rate_limiter,
        agentic_executor,
        llm_backend_url,
        model_aliases,
        http_client,
        model_registry,
        pull_coordinator,
        local_evictor,
        disk_op_lock,
        disk_probe: std::sync::Arc::new(StatvfsProbe),
        disk_pressure_percent: config.model_disk_pressure_percent,
        model_warm_cooldown_hours: config.model_warm_cooldown_hours,
        model_archive_copy_timeout_secs: config.model_archive_copy_timeout_secs,
        model_gc_min_age_secs: config.model_gc_min_age_secs,
        routing_map,
        coding_profile_source,
        personal_proxy,
        embeddings_config: chord_proxy::embeddings::EmbeddingsConfig::from_env(),
    });
    // TIER-05: the model-tier control API runs on a SECOND listener (control port,
    // default 8090), sharing the same AppState. Build it before `state` is moved
    // into the proxy router.
    // ── SNAP observability subsystem (additive) ──
    // Background health/VRAM poller populates the process-global shared
    // inference state read by the SNAP control-API routes. Best-effort: with no
    // engine URLs configured it simply records empty snapshots.
    let snap_cfg = std::sync::Arc::new(chord_proxy::snap::config::SnapConfig::from_env());
    chord_proxy::snap::spawn_health_monitor(snap_cfg);
    info!("SNAP observability subsystem started (vram/activity/inventory/analytics)");

    // ── Sweep-status monitor (additive) ──
    // Background poller (every CHORD_SWEEP_POLL_INTERVAL_SECS, default 30s)
    // watching intake-coder-sweep.service / intake-assistant-sweep.service for
    // the gfx1151 GPU-MoE-wedge failure signature (GPU pegged + no fresh DB
    // rows + service still active). Best-effort, same fail-open discipline as
    // SNAP/the eviction sweep: an unconfigured intake DB degrades to
    // `db_configured: false` snapshots rather than blocking startup.
    let sweep_status_cfg = chord_proxy::sweep_status::config::SweepMonitorConfig::from_env();
    chord_proxy::sweep_status::poll::spawn(sweep_status_cfg);
    info!("sweep-status monitor started (GET /v1/sweep/status, /v1/sweep/status/history)");

    // ── BLD-09 idle-mode watchdog ──
    // Fail-safe: if the proxy is left idle past the watchdog deadline with no
    // active compiler GPU-exclusive lease (a crashed/forgotten compiler, or a
    // stale idle state reloaded after a Chord restart), auto-activate so Chord is
    // never left silently dead. Cheap (a snapshot every 60s); no-op while active.
    {
        let watchdog_state = state.clone();
        tokio::spawn(async move {
            chord_proxy::admin::idle::watchdog_loop(
                watchdog_state,
                std::time::Duration::from_secs(60),
            )
            .await;
        });
    }

    let control_port = config.control_port;
    let control_router = chord_proxy::control::build_control_router(state.clone());

    let router = build_router(state);

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}"))
        .await
        .unwrap_or_else(|e| {
            eprintln!("Failed to bind port {port}: {e}");
            std::process::exit(1);
        });

    // Control API server: a bind/serve failure here must NOT take down the proxy.
    tokio::spawn(async move {
        match tokio::net::TcpListener::bind(format!("0.0.0.0:{control_port}")).await {
            Ok(l) => {
                info!("chord-proxy control API listening on port {control_port}");
                if let Err(e) = axum::serve(l, control_router).await {
                    warn!("control API server error: {e}");
                }
            }
            Err(e) => warn!("failed to bind control API on port {control_port}: {e}"),
        }
    });

    info!("chord-proxy listening on port {port}");
    axum::serve(listener, router).await.unwrap();
}
