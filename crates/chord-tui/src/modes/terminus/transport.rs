//! Per-instance transport config (S91 CTUI-05).
//!
//! Lets the operator set an instance's transport to stdio (a launcher command)
//! or HTTP (an endpoint URL). All values come from operator input / config —
//! there are NO hardcoded infrastructure endpoints here. Changing transport is a
//! confirm-gated (simple) mutation; the new transport is validated before it can
//! be applied so a malformed endpoint never silently breaks the instance.

use crate::confirm::PendingMutation;
use crate::modes::terminus::fleet::Transport;

/// A proposed transport edit, validated before apply.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TransportEdit {
    Stdio { command: String, args: Vec<String> },
    Http { endpoint: String },
}

impl TransportEdit {
    /// Validate the proposed transport. Returns the concrete [`Transport`] or a
    /// human reason. No infra literal is ever injected — an empty endpoint is a
    /// validation error, not a silent default.
    pub fn validate(&self) -> Result<Transport, String> {
        match self {
            TransportEdit::Stdio { command, args } => {
                if command.trim().is_empty() {
                    return Err("stdio command must not be empty".into());
                }
                Ok(Transport::Stdio { command: command.clone(), args: args.clone() })
            }
            TransportEdit::Http { endpoint } => {
                let e = endpoint.trim();
                if e.is_empty() {
                    return Err("http endpoint must not be empty".into());
                }
                if !(e.starts_with("http://") || e.starts_with("https://")) {
                    return Err("http endpoint must start with http:// or https://".into());
                }
                Ok(Transport::Http { endpoint: e.to_string() })
            }
        }
    }
}

/// Build the confirm-gated (simple) transport-change mutation.
pub fn transport_change_mutation(instance: &str) -> PendingMutation {
    PendingMutation::simple(
        "terminus.transport.change",
        format!("Change transport for instance '{instance}'"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_http_endpoint_accepted() {
        let e = TransportEdit::Http { endpoint: "http://host.invalid/mcp".into() };
        let t = e.validate().unwrap();
        assert_eq!(t.label(), "http");
    }

    #[test]
    fn valid_stdio_command_accepted() {
        let e = TransportEdit::Stdio { command: "/opt/terminus/stdio.sh".into(), args: vec![] };
        assert_eq!(e.validate().unwrap().label(), "stdio");
    }

    /// NEGATIVE: an empty endpoint is rejected — never silently filled with a
    /// hardcoded default.
    #[test]
    fn empty_endpoint_rejected_no_default() {
        let e = TransportEdit::Http { endpoint: "   ".into() };
        assert!(e.validate().unwrap_err().contains("must not be empty"));
        let s = TransportEdit::Stdio { command: "".into(), args: vec![] };
        assert!(s.validate().unwrap_err().contains("must not be empty"));
    }

    #[test]
    fn non_http_scheme_rejected() {
        let e = TransportEdit::Http { endpoint: "ftp://x.invalid".into() };
        assert!(e.validate().unwrap_err().contains("http://"));
    }

    #[test]
    fn transport_change_is_confirm_gated() {
        assert!(transport_change_mutation("i").satisfied_by_keystroke('y'));
    }
}
