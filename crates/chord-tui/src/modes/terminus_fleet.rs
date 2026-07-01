//! Terminus-fleet mode (S91 CTUI-01 shell scaffold).
//!
//! The second mode. It shares all CTUI-01 plumbing (connection manager, config,
//! secrets, event loop) with Chord mode but is a SEPARATE view — the two are
//! never blended. Fleet-specific panels (tool inventory, per-agent health) are a
//! later phase; here we provide the mode identity + a scaffold pane so the mode
//! switch is real and the shared plumbing is exercised now.

/// Fleet-mode panels (scaffold; expanded in a later phase).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FleetPanel {
    /// Overview of configured Terminus instances (health from shared manager).
    Instances,
}

impl FleetPanel {
    pub fn title(self) -> &'static str {
        match self {
            FleetPanel::Instances => "Instances",
        }
    }
}

/// Short scaffold description shown in the fleet pane.
pub const FLEET_SCAFFOLD_NOTE: &str =
    "Terminus-fleet mode — shares plumbing with Chord mode; fleet panels land in a later phase.";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fleet_panel_has_title() {
        assert_eq!(FleetPanel::Instances.title(), "Instances");
    }
}
