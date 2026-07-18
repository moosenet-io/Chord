//! CHRD-DIFF-01 issue-1 (codex HIGH) end-to-end forwarding-URL guard.
//!
//! Proves the URL the `chat_completions` router forwards a diffusion request to
//! carries EXACTLY ONE `/v1/chat/completions` — never the doubled
//! `/v1/chat/completions/v1/chat/completions` (→ 404) codex flagged.
//!
//! This lives in its own integration-test binary (separate process) rather than
//! the `chord_proxy` lib unit tests on purpose: its httpmock + reqwest timing
//! jitter, running concurrently inside the lib test binary, widened a
//! PRE-EXISTING global-lock race in `routes::` (`test_embeddings_not_gpu_exclusive_gated`
//! holds the process-global `GPU_EXCLUSIVE` during a request, so a concurrent
//! `chat_completions` test then sees the gate held → 503). A dedicated binary
//! keeps that jitter out of the lib suite.

use chord_proxy::diffusion::{DiffusionConfig, DiffusionManager};

#[tokio::test]
#[ignore = "needs live/mock diffusion daemon + network (binds an httpmock port); \
            run explicitly with --ignored in a real env. The URL single-path \
            contract is ALSO covered by the non-ignored lib unit test \
            diffusion::tests::chat_completions_url_has_exactly_one_path_segment."]
async fn ensure_running_adopts_ambient_daemon_and_forwards_single_path_url() {
    // Stand up a mock "ambient daemon" (as if the standalone dgem.service were
    // still listening): it answers /health, and records exactly which path a
    // forwarded chat request lands on. `ensure_running_gated` must adopt it (no
    // spawn) and return a URL that, POSTed verbatim, hits `/v1/chat/completions`
    // ONCE — never a doubled path.
    let server = httpmock::MockServer::start_async().await;
    let health = server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/health");
        then.status(200);
    });
    let chat = server.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/v1/chat/completions");
        then.status(200).json_body(serde_json::json!({"ok": true}));
    });
    // A request to the DOUBLED path must NOT be what we send.
    let doubled = server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/chat/completions/v1/chat/completions");
        then.status(404);
    });

    let addr = server.address();
    let mgr = DiffusionManager::new(DiffusionConfig {
        bind: addr.ip().to_string(),
        port: addr.port(),
        // Bogus bin: if adoption FAILED and we tried to spawn, it'd error —
        // proving the returned URL came from the ambient-adopt path.
        bin: "/nonexistent/llama-diffusion-daemon".into(),
        ..DiffusionConfig::default()
    });

    // gpu_held = false; the mock is already healthy on the port ⇒ adopt it.
    let url = mgr
        .ensure_running_gated(false)
        .await
        .expect("must adopt the ambient healthy daemon without spawning");
    assert_eq!(
        url.matches("/v1/chat/completions").count(),
        1,
        "forwarded URL must carry exactly one chat-completions path, got: {url}"
    );
    // Adopted, not owned: is_running stays false (no child to idle-evict).
    assert!(!mgr.is_running().await);
    health.assert();

    // Forward a request to exactly that URL (what chat_completions does).
    let resp = reqwest::Client::new()
        .post(&url)
        .json(&serde_json::json!({"model": "diffusion-gemma"}))
        .send()
        .await
        .expect("forward to adopted daemon");
    assert_eq!(resp.status(), 200);
    chat.assert(); // the single-path endpoint was hit
    doubled.assert_hits(0); // the doubled path was never requested
}
