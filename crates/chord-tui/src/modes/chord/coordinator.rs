//! S85 coordinator panel (S91 CTUI-03) — STUBBED, pending S85.
//!
//! Renders a placeholder coordinator view behind the [`super::serving::ServingControl`]
//! seam. Coordinator mutations are DESTRUCTIVE → typed confirmation + inert
//! unless the enable flag is on (see `confirm.rs` / `config.rs`).

use crate::modes::chord::serving::PENDING_S85_BANNER;

/// Placeholder coordinator view (mirrors the eventual S85 coordinator state
/// file shape at a high level: mode + resident set + headroom).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoordinatorView {
    pub operating_mode: String,
    /// Models the coordinator considers resident (placeholder).
    pub resident: Vec<String>,
    /// Free VRAM headroom in bytes, if known (placeholder → None).
    pub headroom_bytes: Option<u64>,
    pub banner: String,
    pub is_stub: bool,
}

impl CoordinatorView {
    pub fn placeholder() -> Self {
        CoordinatorView {
            operating_mode: "assistant-live".into(),
            resident: vec!["<placeholder-resident>".into()],
            headroom_bytes: None,
            banner: PENDING_S85_BANNER.to_string(),
            is_stub: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_is_flagged_and_bannered() {
        let v = CoordinatorView::placeholder();
        assert!(v.is_stub);
        assert!(v.banner.contains("pending S85"));
        assert_eq!(v.headroom_bytes, None, "headroom unknown in stub");
    }
}
