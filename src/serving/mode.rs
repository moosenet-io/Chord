//! Operating-mode controller (S85 SRV-13).
//!
//! Two EXPLICIT modes let live Lumina and batch coder work coexist on one
//! memory-tight GPU. Mode is deliberate, persisted state — never inferred:
//!
//!   - [`OperatingMode::AssistantLive`] — Lumina's chat-role model is pinned +
//!     always resident (SRV-06). Coder/transient models clean-swap (SRV-12) only in
//!     the GPU headroom remaining *around* the pinned assistant; the assistant is
//!     never torn down for a coder swap. A coder that fits the full GPU but not the
//!     leftover-after-pin is rejected with "run in batch-coder", not by evicting
//!     the assistant.
//!   - [`OperatingMode::BatchCoder`] — no live assistant pinned; the full GPU is
//!     available for coder models cycling one-at-a-time through the SRV-12 verified
//!     clean-swap barrier (the overnight Harmony-build case).
//!
//! Switching OFF assistant-live requires an explicit confirm + a graceful unpin
//! (the assistant is demoted to evictable keep-warm, not hard-dropped), so live
//! Lumina is never silently torn down. Switching back loads + pins the chat model
//! before coder work is accepted.

use serde::{Deserialize, Serialize};

/// The explicit operating mode. Persisted in the coordinator state file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OperatingMode {
    /// Live Lumina: chat model pinned + resident; coders swap in the leftover.
    /// The default — the host serves live Lumina unless deliberately switched.
    #[default]
    AssistantLive,
    /// Batch coder: no pinned assistant; full GPU for one-at-a-time coder swaps.
    BatchCoder,
}

impl OperatingMode {
    /// Stable lowercase id used in the state file + status (`assistant-live` /
    /// `batch-coder`).
    pub fn id(self) -> &'static str {
        match self {
            OperatingMode::AssistantLive => "assistant-live",
            OperatingMode::BatchCoder => "batch-coder",
        }
    }

    /// Parse the stable id back (for restoring persisted mode across restart).
    pub fn from_id(s: &str) -> Option<OperatingMode> {
        match s {
            "assistant-live" => Some(OperatingMode::AssistantLive),
            "batch-coder" => Some(OperatingMode::BatchCoder),
            _ => None,
        }
    }
}

/// The action a mode switch requires of the residency manager.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModeAction {
    /// Already in the target mode — nothing to do.
    NoChange,
    /// Leaving assistant-live: gracefully UNPIN the chat model (demote to
    /// evictable keep-warm; do NOT hard-drop it).
    GracefulUnpin,
    /// Entering assistant-live: LOAD + PIN the chat model before accepting coder
    /// work (so Lumina is resident before the mode goes live).
    LoadAndPin,
}

/// Why a mode switch was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModeError {
    /// Switching OFF assistant-live without `confirm` — refused so live Lumina is
    /// never silently dropped.
    NeedsConfirm,
}

/// The mode + the per-pool ceilings it reasons about. Pure decision logic; the
/// residency manager owns the persisted copy + applies [`ModeAction`]s.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModeController {
    pub mode: OperatingMode,
    /// The GPU pool ceiling (GB) — full ceiling in batch-coder; the assistant's
    /// footprint is subtracted from it in assistant-live.
    pub gpu_ceiling_gb: f64,
    /// The CPU/system pool ceiling (GB).
    pub cpu_ceiling_gb: f64,
}

impl ModeController {
    /// The GPU headroom a coder model may use under the current mode:
    ///   - assistant-live → `gpu_ceiling − pinned_assistant_gb` (the leftover),
    ///   - batch-coder → the full `gpu_ceiling` (no assistant pinned).
    pub fn coder_headroom_gb(&self, pinned_assistant_gb: f64) -> f64 {
        match self.mode {
            OperatingMode::AssistantLive => (self.gpu_ceiling_gb - pinned_assistant_gb).max(0.0),
            OperatingMode::BatchCoder => self.gpu_ceiling_gb,
        }
    }

    /// Whether a coder model of `footprint_gb` fits the current mode's headroom.
    pub fn coder_fits(&self, footprint_gb: f64, pinned_assistant_gb: f64) -> bool {
        footprint_gb <= self.coder_headroom_gb(pinned_assistant_gb)
    }

    /// A clear rejection reason when a coder is too big for assistant-live (fits the
    /// full GPU but not the leftover-after-pin), else `None`. In batch-coder a model
    /// that doesn't fit the full ceiling is a genuine OOM, not a mode mismatch, so
    /// this returns `None` there (the admission path handles it).
    pub fn oversize_reason(&self, footprint_gb: f64, pinned_assistant_gb: f64) -> Option<String> {
        if self.mode == OperatingMode::AssistantLive
            && !self.coder_fits(footprint_gb, pinned_assistant_gb)
            && footprint_gb <= self.gpu_ceiling_gb
        {
            Some(format!(
                "model needs {footprint_gb:.0}GB but only {:.0}GB is free around the pinned \
                 assistant — exceeds assistant-live headroom; run in batch-coder",
                self.coder_headroom_gb(pinned_assistant_gb)
            ))
        } else {
            None
        }
    }

    /// Decide the action for a deliberate switch to `target`. Switching OFF
    /// assistant-live requires `confirm` (else [`ModeError::NeedsConfirm`]) so live
    /// Lumina is never silently dropped.
    pub fn request_switch(
        &self,
        target: OperatingMode,
        confirm: bool,
    ) -> Result<ModeAction, ModeError> {
        if target == self.mode {
            return Ok(ModeAction::NoChange);
        }
        match (self.mode, target) {
            (OperatingMode::AssistantLive, OperatingMode::BatchCoder) => {
                if confirm {
                    Ok(ModeAction::GracefulUnpin)
                } else {
                    Err(ModeError::NeedsConfirm)
                }
            }
            (OperatingMode::BatchCoder, OperatingMode::AssistantLive) => Ok(ModeAction::LoadAndPin),
            // The same-mode case is handled above; this arm is unreachable.
            _ => Ok(ModeAction::NoChange),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctl(mode: OperatingMode) -> ModeController {
        ModeController {
            mode,
            gpu_ceiling_gb: 96.0,
            cpu_ceiling_gb: 31.0,
        }
    }

    #[test]
    fn assistant_live_headroom_is_leftover_after_pin() {
        let c = ctl(OperatingMode::AssistantLive);
        // 96 ceiling - 20 pinned assistant = 76 leftover for coders.
        assert_eq!(c.coder_headroom_gb(20.0), 76.0);
        assert!(c.coder_fits(70.0, 20.0));
        assert!(!c.coder_fits(80.0, 20.0));
    }

    #[test]
    fn batch_coder_uses_full_ceiling() {
        let c = ctl(OperatingMode::BatchCoder);
        // No assistant pin matters; full 96 available.
        assert_eq!(c.coder_headroom_gb(20.0), 96.0);
        assert!(c.coder_fits(90.0, 20.0));
    }

    #[test]
    fn oversize_in_assistant_live_explains_batch_coder() {
        let c = ctl(OperatingMode::AssistantLive);
        // 80GB coder fits the full 96 GPU but not the 76 leftover → clear reason.
        let reason = c.oversize_reason(80.0, 20.0).expect("oversize reason");
        assert!(reason.contains("batch-coder"));
        // A 70GB coder fits the leftover → no reason.
        assert!(c.oversize_reason(70.0, 20.0).is_none());
        // A 200GB coder doesn't fit even the full GPU → genuine OOM, not a mode
        // mismatch → no "run in batch-coder" reason.
        assert!(c.oversize_reason(200.0, 20.0).is_none());
    }

    #[test]
    fn batch_coder_never_emits_mode_oversize_reason() {
        let c = ctl(OperatingMode::BatchCoder);
        assert!(c.oversize_reason(200.0, 0.0).is_none());
    }

    #[test]
    fn switch_off_assistant_live_requires_confirm() {
        let c = ctl(OperatingMode::AssistantLive);
        // No confirm → refused (never silently drop live Lumina).
        assert_eq!(
            c.request_switch(OperatingMode::BatchCoder, false),
            Err(ModeError::NeedsConfirm)
        );
        // With confirm → graceful unpin.
        assert_eq!(
            c.request_switch(OperatingMode::BatchCoder, true),
            Ok(ModeAction::GracefulUnpin)
        );
    }

    #[test]
    fn switch_into_assistant_live_loads_and_pins() {
        let c = ctl(OperatingMode::BatchCoder);
        assert_eq!(
            c.request_switch(OperatingMode::AssistantLive, false),
            Ok(ModeAction::LoadAndPin)
        );
    }

    #[test]
    fn same_mode_is_no_change() {
        let c = ctl(OperatingMode::AssistantLive);
        assert_eq!(
            c.request_switch(OperatingMode::AssistantLive, false),
            Ok(ModeAction::NoChange)
        );
    }

    #[test]
    fn mode_id_round_trips() {
        for m in [OperatingMode::AssistantLive, OperatingMode::BatchCoder] {
            assert_eq!(OperatingMode::from_id(m.id()), Some(m));
        }
        assert_eq!(OperatingMode::from_id("nonsense"), None);
    }
}
