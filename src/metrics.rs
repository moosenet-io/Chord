//! PROMEX-02: application-level Prometheus metrics exporter for Chord.
//!
//! A process-global [`prometheus::Registry`] plus a small, fixed set of
//! application metrics — LLM inference request volume and latency — exposed
//! as `GET /metrics` in the standard Prometheus text exposition format (see
//! `crate::control::build_control_router`, mounted alongside `/health`).
//!
//! This mirrors the REFERENCE PATTERN merged in the `Terminus` repo
//! (`terminus-rs`'s `src/metrics/mod.rs`, PROMEX-01) — same
//! process-global-`OnceLock` idiom, same shape of metrics, same cardinality
//! discipline — adapted to Chord's domain: Chord is an LLM proxy, so its
//! meaningful application metric is INFERENCE requests (`/v1/chat/completions`
//! et al.), not MCP tool calls.
//!
//! ## Design
//! - **One registry, lazily built once per process** (`OnceLock`), matching
//!   the Terminus pattern rather than pulling in a separate `lazy_static`/
//!   `once_cell` dependency.
//! - **Two metrics, deliberately minimal**:
//!   - `chord_inference_requests_total{model, result}` — a `CounterVec`,
//!     `result` is always `"ok"` or `"error"` (never a raw error message or
//!     upstream status text, so cardinality stays bounded by model count).
//!   - `chord_inference_duration_seconds{model}` — a `HistogramVec` of
//!     end-to-end proxy latency (request start to upstream response headers),
//!     default bucket boundaries.
//! - **No secrets, no caller-controlled label values.** The `model` label is
//!   passed through [`bounded_model_label`] at the call site, which maps a
//!   caller-supplied `model` string onto a BOUNDED set — {names present in
//!   Chord's own `ModelRegistry`} ∪ {`<unknown>`} — so an arbitrary client-
//!   supplied model string (which a caller fully controls in the request
//!   body) can never inflate cardinality or leak a secret/PII-shaped string
//!   into a label. `result` is likewise a closed `"ok"`/`"error"` set.
//! - **Read-only, unauthenticated, always-on.** The control router's existing
//!   `/health` route is likewise unauthenticated (see `control.rs`'s module
//!   doc, "Auth choice" section — JWT auth for `/api/*`/`/admin/*` is checked
//!   INSIDE those handlers, not by a router-wide layer, and `/health` simply
//!   never calls it) — metrics are equally non-sensitive (counts and timings
//!   only, bounded model names), so `/metrics` is mounted the same way, no
//!   separate env gate.
//!
//! ## Usage
//! Call [`record_inference`] from `crate::routes::chat_completions` at each
//! point it already calls `state.audit_logger.log_llm_call` (the existing
//! central audit points for this handler) — the config-missing,
//! archive-pull-failed, upstream-unreachable, and final upstream-status
//! outcomes. Call [`gather_text`] from the `/metrics` HTTP handler.

use std::borrow::Cow;
use std::sync::OnceLock;
use std::time::Duration;

use prometheus::{CounterVec, HistogramVec, Registry, TextEncoder};

/// The result label recorded on `chord_inference_requests_total`. Deliberately
/// a closed two-value set (never the raw upstream error/status text) so the
/// metric's cardinality is bounded by `model count * 2`, not by arbitrary
/// error strings.
const RESULT_OK: &str = "ok";
const RESULT_ERROR: &str = "error";

/// The bounded sentinel used in place of any model name that is not present
/// in Chord's own model registry at record time — see [`bounded_model_label`].
const UNKNOWN_MODEL: &str = "<unknown>";

struct Metrics {
    registry: Registry,
    inference_requests_total: CounterVec,
    inference_duration_seconds: HistogramVec,
}

static METRICS: OnceLock<Metrics> = OnceLock::new();

fn metrics() -> &'static Metrics {
    METRICS.get_or_init(|| {
        let registry = Registry::new();

        let inference_requests_total = CounterVec::new(
            prometheus::Opts::new(
                "chord_inference_requests_total",
                "Total number of Chord LLM inference proxy requests, by model and outcome.",
            ),
            &["model", "result"],
        )
        .expect("chord_inference_requests_total: static metric definition is well-formed");

        let inference_duration_seconds = HistogramVec::new(
            prometheus::HistogramOpts::new(
                "chord_inference_duration_seconds",
                "Chord LLM inference proxy request latency in seconds, by model.",
            ),
            &["model"],
        )
        .expect("chord_inference_duration_seconds: static metric definition is well-formed");

        registry
            .register(Box::new(inference_requests_total.clone()))
            .expect("chord_inference_requests_total: single registration at process startup");
        registry
            .register(Box::new(inference_duration_seconds.clone()))
            .expect("chord_inference_duration_seconds: single registration at process startup");

        Metrics {
            registry,
            inference_requests_total,
            inference_duration_seconds,
        }
    })
}

/// Record one completed inference proxy request: increments
/// `chord_inference_requests_total{model, result}` and observes `duration`
/// into `chord_inference_duration_seconds{model}`.
///
/// `model` MUST already be a bounded label value — pass it through
/// [`bounded_model_label`] at the call site, never the raw client-supplied
/// model string. See this module's doc for why label values must come from a
/// bounded set.
pub fn record_inference(model: &str, is_ok: bool, duration: Duration) {
    let m = metrics();
    let result = if is_ok { RESULT_OK } else { RESULT_ERROR };
    m.inference_requests_total
        .with_label_values(&[model, result])
        .inc();
    m.inference_duration_seconds
        .with_label_values(&[model])
        .observe(duration.as_secs_f64());
}

/// Map a caller-supplied inference `model` name onto a BOUNDED metric label
/// value, so the `model` label can never be inflated by an arbitrary or
/// unknown client-supplied string.
///
/// `is_known_served` comes from the CALLER (kept out of this fn so it stays
/// pure/testable) and MUST be derived from validated state — Chord's own
/// `ModelRegistry` (`state.model_registry.lock().await.get(registry_key)`,
/// the same lookup `chat_completions` already performs for TIER-02
/// pull-on-miss), never from parsing the raw request body. Only a model the
/// registry actually knows about (i.e. is currently tracked/served) passes
/// through as itself; anything else — a typo, a probe, an arbitrary or
/// secret-shaped string — collapses to the fixed `<unknown>` sentinel.
pub fn bounded_model_label<'a>(model: &'a str, is_known_served: bool) -> Cow<'a, str> {
    if is_known_served {
        Cow::Borrowed(model)
    } else {
        Cow::Borrowed(UNKNOWN_MODEL)
    }
}

/// Encode every registered metric in the Prometheus text exposition format
/// (the `GET /metrics` response body).
pub fn gather_text() -> String {
    let m = metrics();
    let families = m.registry.gather();
    let encoder = TextEncoder::new();
    encoder
        .encode_to_string(&families)
        .unwrap_or_else(|e| format!("# error encoding metrics: {e}\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_inference_appears_in_gathered_text() {
        record_inference("promex02_test_model", true, Duration::from_millis(42));

        let text = gather_text();
        assert!(
            text.contains("chord_inference_requests_total"),
            "expected counter family name in output:\n{text}"
        );
        assert!(
            text.contains("chord_inference_duration_seconds"),
            "expected histogram family name in output:\n{text}"
        );
        assert!(
            text.contains("model=\"promex02_test_model\""),
            "expected the recorded model label in output:\n{text}"
        );
        assert!(
            text.contains("result=\"ok\""),
            "expected the ok result label in output:\n{text}"
        );
    }

    #[test]
    fn record_inference_error_uses_error_result_label() {
        record_inference("promex02_test_model_err", false, Duration::from_millis(1));

        let text = gather_text();
        assert!(
            text.contains("model=\"promex02_test_model_err\",result=\"error\"")
                || text.contains("result=\"error\",model=\"promex02_test_model_err\""),
            "expected an error-result sample for the model in output:\n{text}"
        );
    }

    #[test]
    fn bounded_model_label_known_served_passes_through() {
        assert_eq!(
            bounded_model_label("qwen3-coder:30b", true),
            "qwen3-coder:30b"
        );
    }

    #[test]
    fn bounded_model_label_unknown_is_sentinel() {
        // Not in the registry (e.g. a typo, a probe, or an arbitrary/secret-shaped
        // client-supplied string) — never passed through as a label value.
        assert_eq!(
            bounded_model_label("totally-made-up-xyz", false),
            "<unknown>"
        );
        assert_eq!(
            bounded_model_label("customer-secret-anything", false),
            "<unknown>"
        );
    }
}
