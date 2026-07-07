//! Top-level application state + mode machinery (S91 CTUI-01).
//!
//! Holds the current [`Mode`] (Chord | TerminusFleet), the focused panel per
//! mode, a toast/status line, and the pending-confirmation state machine that
//! enforces control-plane safety (explicit keystroke for simple mutations,
//! TYPED confirmation for destructive ones; stubbed mutations inert).
//!
//! `App` is pure state + transitions — no terminal or socket I/O — so the whole
//! interaction model is unit-testable headlessly.

use crate::config::{Config, Settings};
use crate::confirm::{resolve_execution, ExecOutcome, PendingMutation, Severity};
use crate::modes::chord::ChordPanel;
use crate::modes::terminus_fleet::FleetPanel;

/// The two top-level modes. One binary; NOT blended into one view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Chord,
    TerminusFleet,
}

impl Mode {
    pub fn label(self) -> &'static str {
        match self {
            Mode::Chord => "CHORD",
            Mode::TerminusFleet => "TERMINUS-FLEET",
        }
    }
    /// Toggle between the two modes (bound to a switch key in the shell).
    pub fn toggled(self) -> Mode {
        match self {
            Mode::Chord => Mode::TerminusFleet,
            Mode::TerminusFleet => Mode::Chord,
        }
    }
}

/// The confirm overlay state. `None` means no mutation is pending.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfirmState {
    pub mutation: PendingMutation,
    /// Buffer of typed input for destructive confirmations.
    pub typed: String,
}

/// A transient status/toast message with a severity for coloring.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Toast {
    pub text: String,
    pub level: ToastLevel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToastLevel {
    Info,
    Warn,
    Error,
}

/// Top-level app state.
pub struct App {
    pub mode: Mode,
    pub chord_panel: ChordPanel,
    pub fleet_panel: FleetPanel,
    pub settings: Settings,
    pub confirm: Option<ConfirmState>,
    pub toast: Option<Toast>,
    /// Set when the user requests quit.
    pub should_quit: bool,
    /// Last known terminal size, updated on resize (reflow, never panic).
    pub term_size: (u16, u16),
    /// True when the fleet is empty and we should show the add-instance prompt.
    pub show_add_instance_prompt: bool,
}

impl App {
    pub fn new(config: &Config) -> Self {
        App {
            mode: Mode::Chord,
            chord_panel: ChordPanel::Models,
            fleet_panel: FleetPanel::Instances,
            settings: config.settings.clone(),
            confirm: None,
            toast: None,
            should_quit: false,
            term_size: (80, 24),
            show_add_instance_prompt: config.instances.is_empty(),
        }
    }

    /// Switch top-level mode. Views are separate; cancels any pending confirm.
    pub fn switch_mode(&mut self) {
        self.mode = self.mode.toggled();
        self.confirm = None; // safety: never carry a pending mutation across modes
        self.set_toast(format!("mode → {}", self.mode.label()), ToastLevel::Info);
    }

    pub fn set_toast(&mut self, text: impl Into<String>, level: ToastLevel) {
        self.toast = Some(Toast { text: text.into(), level });
    }

    /// Handle a terminal resize. Only records the new size; ratatui reflows the
    /// layout on the next draw. Never panics on tiny sizes.
    pub fn on_resize(&mut self, w: u16, h: u16) {
        self.term_size = (w, h);
    }

    // ── Confirmation state machine ────────────────────────────────────────────

    /// Begin a mutation: opens the confirm overlay. No side effects yet.
    pub fn request_mutation(&mut self, m: PendingMutation) {
        self.confirm = Some(ConfirmState { mutation: m, typed: String::new() });
    }

    /// Cancel a pending confirmation.
    pub fn cancel_confirm(&mut self) {
        if self.confirm.is_some() {
            self.confirm = None;
            self.set_toast("cancelled", ToastLevel::Info);
        }
    }

    /// Append a typed char to a destructive confirmation buffer (ignored for
    /// simple mutations, which use the keystroke path).
    pub fn confirm_type_char(&mut self, c: char) {
        if let Some(cs) = &mut self.confirm {
            if matches!(cs.mutation.severity, Severity::Destructive { .. }) {
                cs.typed.push(c);
            }
        }
    }

    pub fn confirm_backspace(&mut self) {
        if let Some(cs) = &mut self.confirm {
            cs.typed.pop();
        }
    }

    /// Attempt to satisfy a SIMPLE mutation via the explicit confirm keystroke.
    /// Returns the [`ExecOutcome`] and clears the overlay if handled. A
    /// destructive mutation is NOT satisfiable this way.
    pub fn confirm_keystroke(&mut self, key: char) -> Option<ExecOutcome> {
        let cs = self.confirm.as_ref()?;
        if !cs.mutation.satisfied_by_keystroke(key) {
            return None; // wrong key or destructive → stays pending
        }
        let m = cs.mutation.clone();
        let outcome = resolve_execution(&m, self.settings.enable_stubbed_mutations);
        self.finish_confirm(&m, &outcome);
        Some(outcome)
    }

    /// Attempt to satisfy a DESTRUCTIVE mutation via the typed buffer. Returns
    /// the outcome iff the typed phrase matches exactly.
    pub fn confirm_submit_typed(&mut self) -> Option<ExecOutcome> {
        let cs = self.confirm.as_ref()?;
        if !cs.mutation.satisfied_by_typed(&cs.typed) {
            // Not yet matched — keep overlay, hint the operator.
            return None;
        }
        let m = cs.mutation.clone();
        let outcome = resolve_execution(&m, self.settings.enable_stubbed_mutations);
        self.finish_confirm(&m, &outcome);
        Some(outcome)
    }

    fn finish_confirm(&mut self, m: &PendingMutation, outcome: &ExecOutcome) {
        self.confirm = None;
        let msg = match outcome {
            ExecOutcome::Executed => format!("{}: executed", m.action),
            ExecOutcome::InertStub => {
                format!("{}: INERT stub (pending S85; no real op)", m.action)
            }
            ExecOutcome::Rejected => format!("{}: rejected", m.action),
        };
        let level = match outcome {
            ExecOutcome::Executed => ToastLevel::Info,
            ExecOutcome::InertStub => ToastLevel::Warn,
            ExecOutcome::Rejected => ToastLevel::Error,
        };
        self.set_toast(msg, level);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modes::chord::cleanswap::{clean_swap_mutation, CLEAN_SWAP_CHALLENGE};
    use crate::modes::chord::models::pull_mutation;

    fn app() -> App {
        App::new(&Config::default())
    }

    #[test]
    fn starts_in_chord_mode_with_add_prompt_when_empty() {
        let a = app();
        assert_eq!(a.mode, Mode::Chord);
        assert!(a.show_add_instance_prompt, "empty fleet → add-instance prompt");
    }

    #[test]
    fn mode_switch_toggles_and_is_not_blended() {
        let mut a = app();
        assert_eq!(a.mode, Mode::Chord);
        a.switch_mode();
        assert_eq!(a.mode, Mode::TerminusFleet);
        a.switch_mode();
        assert_eq!(a.mode, Mode::Chord);
    }

    #[test]
    fn resize_records_size_without_panic() {
        let mut a = app();
        a.on_resize(1, 1); // absurdly small — must not panic
        assert_eq!(a.term_size, (1, 1));
        a.on_resize(200, 60);
        assert_eq!(a.term_size, (200, 60));
    }

    /// NEGATIVE TEST: a simple mutation requires the explicit confirm keystroke.
    /// Without pressing 'y', nothing executes.
    #[test]
    fn simple_mutation_requires_confirm_keystroke() {
        let mut a = app();
        a.request_mutation(pull_mutation("m"));
        // A non-confirm key does not execute and keeps the overlay.
        assert!(a.confirm_keystroke('n').is_none());
        assert!(a.confirm.is_some(), "still pending until confirmed");
        // The confirm keystroke executes.
        assert_eq!(a.confirm_keystroke('y'), Some(ExecOutcome::Executed));
        assert!(a.confirm.is_none());
    }

    /// NEGATIVE TEST: a destructive (clean-swap) mutation cannot be confirmed by
    /// a keystroke and requires the exact typed phrase; being a stub, it stays
    /// INERT with the flag off.
    #[test]
    fn destructive_stub_requires_typed_and_is_inert() {
        let mut a = app();
        assert!(!a.settings.enable_stubbed_mutations, "flag off by default");
        a.request_mutation(clean_swap_mutation("qwen3-coder:30b"));

        // Keystroke 'y' must NOT satisfy a destructive mutation.
        assert!(a.confirm_keystroke('y').is_none());
        assert!(a.confirm.is_some());

        // Wrong typed phrase → not satisfied.
        for c in "clean".chars() {
            a.confirm_type_char(c);
        }
        assert!(a.confirm_submit_typed().is_none());

        // Reset buffer and type the exact challenge.
        a.confirm = Some(ConfirmState {
            mutation: clean_swap_mutation("qwen3-coder:30b"),
            typed: String::new(),
        });
        for c in CLEAN_SWAP_CHALLENGE.chars() {
            a.confirm_type_char(c);
        }
        // Correct typed phrase → resolves, but INERT because it's a stub w/ flag off.
        assert_eq!(a.confirm_submit_typed(), Some(ExecOutcome::InertStub));
        assert!(a.confirm.is_none());
    }

    #[test]
    fn typed_chars_ignored_for_simple_mutation() {
        let mut a = app();
        a.request_mutation(pull_mutation("m"));
        a.confirm_type_char('x'); // simple mutation ignores typed input
        if let Some(cs) = &a.confirm {
            assert!(cs.typed.is_empty(), "simple mutations don't accumulate typed input");
        } else {
            panic!("should still be pending");
        }
    }

    #[test]
    fn switching_mode_clears_pending_confirm() {
        let mut a = app();
        a.request_mutation(pull_mutation("m"));
        assert!(a.confirm.is_some());
        a.switch_mode();
        assert!(a.confirm.is_none(), "pending mutation dropped on mode switch (safety)");
    }
}
