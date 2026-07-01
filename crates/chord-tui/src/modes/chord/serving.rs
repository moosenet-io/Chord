//! S85 serving-control SEAM (S91 CTUI-03) — STUBBED, pending S85.
//!
//! The S85 serving / coordinator / clean-swap surfaces are NOT built yet. This
//! module defines the [`ServingControl`] trait that the real client will
//! implement later, plus a clearly-named [`MockServingControl`] returning
//! placeholder data so the panels render + navigate NOW. Swapping in the real
//! client is a single localized change (construct the real impl instead of the
//! mock where the app wires it).
//!
//! SAFETY: coordinator + clean-swap operations are DESTRUCTIVE. They require a
//! typed confirmation (enforced in `confirm.rs`) AND are gated INERT by
//! [`crate::config::Settings::enable_stubbed_mutations`] (off by default) until
//! S85 is verified. A stubbed clean-swap performs NO real operation — see
//! [`MockServingControl::clean_swap`].

use async_trait::async_trait;

use crate::modes::chord::coordinator::CoordinatorView;

/// Placeholder banner every stubbed panel shows.
pub const PENDING_S85_BANNER: &str = "pending S85 integration — panel is a stub (no live data / inert mutations)";

/// Serving-profile view (mock placeholder shape mirroring the eventual S85 API).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServingView {
    /// e.g. "assistant-live" / "batch-coder" (mirrors OperatingMode ids).
    pub operating_mode: String,
    /// Pinned chat-role model, if any (SRV-06 concept).
    pub pinned_assistant: Option<String>,
    /// Whether this is placeholder data (always true for the mock).
    pub is_stub: bool,
}

/// Result of a (stubbed) clean-swap request. `performed_real_op` MUST be false
/// for the mock — proving inertness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CleanSwapResult {
    pub performed_real_op: bool,
    pub note: String,
}

/// The seam the real S85 client will implement. Kept deliberately small +
/// isolated so the mock→real swap is one localized change.
#[async_trait]
pub trait ServingControl: Send + Sync {
    /// Read the serving profile (read-only; safe).
    async fn serving_view(&self) -> ServingView;

    /// Read the coordinator view (read-only; safe).
    async fn coordinator_view(&self) -> CoordinatorView;

    /// DESTRUCTIVE: request a clean-swap of the live model. The stub performs NO
    /// real operation and returns `performed_real_op: false`.
    async fn clean_swap(&self, target_model: &str) -> CleanSwapResult;
}

/// MOCK implementation — clearly named + isolated. Returns placeholder data and
/// performs no real operations. Replace with the real S85 client in one place.
pub struct MockServingControl;

#[async_trait]
impl ServingControl for MockServingControl {
    async fn serving_view(&self) -> ServingView {
        ServingView {
            operating_mode: "assistant-live".into(),
            pinned_assistant: Some("<placeholder-assistant>".into()),
            is_stub: true,
        }
    }

    async fn coordinator_view(&self) -> CoordinatorView {
        CoordinatorView::placeholder()
    }

    async fn clean_swap(&self, _target_model: &str) -> CleanSwapResult {
        // INERT: never touches any real serving process. The real op lands with
        // S85 wiring behind the enable-flag + typed confirm.
        CleanSwapResult {
            performed_real_op: false,
            note: "stub: clean-swap not wired (pending S85); no operation performed".into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_serving_view_is_flagged_stub() {
        let v = MockServingControl.serving_view().await;
        assert!(v.is_stub);
        assert_eq!(v.operating_mode, "assistant-live");
    }

    /// NEGATIVE TEST: a stubbed clean-swap performs NO real operation.
    #[tokio::test]
    async fn stubbed_clean_swap_performs_no_real_op() {
        let r = MockServingControl.clean_swap("some-model").await;
        assert!(!r.performed_real_op, "mock clean-swap must be inert");
    }

    #[test]
    fn banner_marks_pending_s85() {
        assert!(PENDING_S85_BANNER.contains("pending S85"));
    }
}
