//! Chord mode (S91 CTUI-02 live + CTUI-03 stubbed S85 seam).
//!
//! Sub-panels:
//!   - [`models`]      — live registry table (CTUI-02, read-first)
//!   - [`backends`]    — live backend status (CTUI-02, read-only)
//!   - [`serving`]     — STUBBED S85 serving profile (CTUI-03)
//!   - [`coordinator`] — STUBBED S85 coordinator (CTUI-03)
//!   - [`cleanswap`]   — STUBBED S85 clean-swap action (CTUI-03, destructive+inert)
//!
//! [`chord_client`] wraps ONLY Chord's stable control endpoints. The S85 panels
//! go through the [`serving::ServingControl`] seam with a mock impl.

pub mod backends;
pub mod chord_client;
pub mod cleanswap;
pub mod coordinator;
pub mod models;
pub mod serving;

/// Which Chord-mode panel is focused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChordPanel {
    /// CTUI-02 live.
    Models,
    /// CTUI-02 live.
    Backends,
    /// CTUI-03 stub.
    Serving,
    /// CTUI-03 stub.
    Coordinator,
    /// CTUI-03 stub.
    CleanSwap,
}

impl ChordPanel {
    pub const ALL: [ChordPanel; 5] = [
        ChordPanel::Models,
        ChordPanel::Backends,
        ChordPanel::Serving,
        ChordPanel::Coordinator,
        ChordPanel::CleanSwap,
    ];

    pub fn title(self) -> &'static str {
        match self {
            ChordPanel::Models => "Models",
            ChordPanel::Backends => "Backends",
            ChordPanel::Serving => "Serving (S85)",
            ChordPanel::Coordinator => "Coordinator (S85)",
            ChordPanel::CleanSwap => "Clean-Swap (S85)",
        }
    }

    /// True for the CTUI-03 not-yet-wired panels (render the pending banner).
    pub fn is_stub(self) -> bool {
        matches!(self, ChordPanel::Serving | ChordPanel::Coordinator | ChordPanel::CleanSwap)
    }

    pub fn next(self) -> ChordPanel {
        let all = Self::ALL;
        let i = all.iter().position(|p| *p == self).unwrap_or(0);
        all[(i + 1) % all.len()]
    }

    pub fn prev(self) -> ChordPanel {
        let all = Self::ALL;
        let i = all.iter().position(|p| *p == self).unwrap_or(0);
        all[(i + all.len() - 1) % all.len()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_panels_are_flagged() {
        assert!(!ChordPanel::Models.is_stub());
        assert!(!ChordPanel::Backends.is_stub());
        assert!(ChordPanel::Serving.is_stub());
        assert!(ChordPanel::Coordinator.is_stub());
        assert!(ChordPanel::CleanSwap.is_stub());
    }

    #[test]
    fn panel_cycling_wraps() {
        assert_eq!(ChordPanel::Models.next(), ChordPanel::Backends);
        assert_eq!(ChordPanel::Models.prev(), ChordPanel::CleanSwap);
        assert_eq!(ChordPanel::CleanSwap.next(), ChordPanel::Models);
    }
}
