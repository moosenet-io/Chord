//! Chord "Models" panel (S91 CTUI-02).
//!
//! Read-first table over the stable `/api/models` registry. Row formatting is
//! pure so it is unit-testable without a terminal. The one available mutation
//! (pull a cold model / archive a warm one) is a SIMPLE mutation requiring an
//! explicit confirm keystroke — nothing here is destructive.

use crate::confirm::PendingMutation;
use crate::modes::chord::chord_client::ModelRow;

/// Human-formatted columns for one model row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelDisplay {
    pub name: String,
    pub tier: String,
    pub loaded: String,
    pub backend: String,
    pub size: String,
}

/// Render bytes as a compact human string; "-" when unavailable (drift).
fn human_size(bytes: Option<u64>) -> String {
    match bytes {
        None => "-".into(),
        Some(b) => {
            const U: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
            let mut v = b as f64;
            let mut i = 0;
            while v >= 1024.0 && i < U.len() - 1 {
                v /= 1024.0;
                i += 1;
            }
            format!("{v:.1}{}", U[i])
        }
    }
}

/// Format a [`ModelRow`] for the table; every missing field shows a stable
/// "field unavailable" placeholder rather than blanking the row.
pub fn display_row(m: &ModelRow) -> ModelDisplay {
    ModelDisplay {
        name: m.name.clone(),
        tier: m.tier.clone().unwrap_or_else(|| "-".into()),
        loaded: match m.loaded {
            Some(true) => "loaded".into(),
            Some(false) => "cold".into(),
            None => "-".into(),
        },
        backend: m.backend.clone().unwrap_or_else(|| "-".into()),
        size: human_size(m.size_bytes),
    }
}

/// Build the (simple, keystroke-confirmed) pull mutation for a cold model.
pub fn pull_mutation(model: &str) -> PendingMutation {
    PendingMutation::simple("chord.model.pull", format!("Pull cold model '{model}' to warm"))
}

/// Build the (simple, keystroke-confirmed) archive mutation for a warm model.
pub fn archive_mutation(model: &str) -> PendingMutation {
    PendingMutation::simple("chord.model.archive", format!("Archive warm model '{model}'"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_full_row() {
        let m = ModelRow {
            name: "llama3.3:70b".into(),
            tier: Some("warm".into()),
            loaded: Some(true),
            backend: Some("vulkan-radv".into()),
            size_bytes: Some(42_000_000_000),
            protected: Some(false),
        };
        let d = display_row(&m);
        assert_eq!(d.tier, "warm");
        assert_eq!(d.loaded, "loaded");
        assert_eq!(d.backend, "vulkan-radv");
        assert!(d.size.ends_with("GB"));
    }

    #[test]
    fn drifted_row_shows_placeholders_not_blanks() {
        let m = ModelRow { name: "x".into(), ..Default::default() };
        let d = display_row(&m);
        assert_eq!(d.tier, "-");
        assert_eq!(d.loaded, "-");
        assert_eq!(d.backend, "-");
        assert_eq!(d.size, "-");
    }

    #[test]
    fn model_mutations_are_simple_not_destructive() {
        assert!(pull_mutation("m").satisfied_by_keystroke('y'));
        assert!(archive_mutation("m").satisfied_by_keystroke('y'));
        // Not destructive → not typed.
        assert!(!pull_mutation("m").satisfied_by_typed("m"));
    }
}
