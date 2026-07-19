# crates: chord-tui + chord-secrets

The workspace sub-crates (339 KG nodes, `crates/`). Both are clients of the
root proxy's stable surfaces — neither links into, restarts, or reconfigures
the live `chord-proxy` process.

## chord-tui

A ratatui control TUI (S91 CTUI-01) for operating Chord and Terminus from a
terminal. The async event loop `select!`s over terminal events and a redraw
tick, with each instance polled on its own task — a slow or dead instance can
never freeze input or rendering.

| Symbol | Kind | File | What it does |
|---|---|---|---|
| `main` (event loop) | function | `crates/chord-tui/src/main.rs` | Wires config → secrets → connection manager into the ratatui shell |
| `app` / `ui` | modules | `crates/chord-tui/src/app.rs`, `ui.rs` | Application state and rendering |
| `connection` | module | `crates/chord-tui/src/connection.rs` | The connection manager over Chord/Terminus control endpoints |
| `secret::SecretManager::status` | function | `crates/chord-tui/src/secret.rs` | Secret-loading status surfaced in the UI (a top KG hotspot) |
| `secret::SecretValue::is_empty` | function | `crates/chord-tui/src/secret.rs` | Guarded secret handling — values are wrapped, not passed as strings |
| `confirm::PendingMutation::simple` | function | `crates/chord-tui/src/confirm.rs` | Explicit confirmation step before any mutating control call |
| `modes::terminus::fleet::Transport` | enum | `crates/chord-tui/src/modes/terminus/fleet.rs` | Fleet-mode transport descriptor (`endpoint_display` / `label` render it); the highest-ranked symbol in the crates subsystem |
| `modes::terminus::fleet::Fleet` | struct | `crates/chord-tui/src/modes/terminus/fleet.rs` | The fleet view's data model |

Usage details and screenshots: [../chord-tui.md](../chord-tui.md).

## chord-secrets

A dependency-light <secret-manager> Universal Auth client (CSEC-01), shared by
`chord-proxy` and `chord-tui` so the auth flow is written once.

| Symbol | Kind | File | What it does |
|---|---|---|---|
| `InfisicalConfig` | struct | `crates/chord-secrets/src/lib.rs` | Client configuration; `base_url()` derives the API root |
| Universal Auth login + raw-secret fetch | functions | `crates/chord-secrets/src/lib.rs` | POST client credentials for a bearer token, then GET the scoped raw secrets |

The standing architectural decision (from the crate docs): Chord authenticates
to <secret-manager> **directly** with its own bootstrap identity — not brokered
through another fleet service over internal HTTP — because several internal
hops are not TLS-terminated and secrets never travel un-terminated paths. The
proxy's startup consumer is `src/secrets_bootstrap.rs`
(`fetch_and_apply_downstream_secrets`), which fetches `CHORD_JWT_SECRET` /
`CHORD_API_KEY` fresh before `Config::from_env` and falls back to the static
environment on any failure — never a hard startup error.

## How it connects

`chord-tui` speaks HTTP to the proxy/control ports (model tiering, status,
fleet views) — it is outside the `chord-proxy` process entirely.
`chord-secrets` is a build dependency of both binaries; at runtime it only
talks to the <secret-manager> endpoint.

## Configuration

`INFISICAL_URL`, `INFISICAL_CLIENT_ID`, `INFISICAL_CLIENT_SECRET`,
`CHORD_INFISICAL_PROJECT_ID`, `CHORD_INFISICAL_ENVIRONMENT`,
`CHORD_INFISICAL_SECRET_PATH` (key names only; values live in the vault).

## Notes and gaps

- The TUI requires the control endpoints to be reachable and a valid JWT; it
  has no offline mode beyond displaying connection state.
- This page does not document every TUI mode — see
  [../chord-tui.md](../chord-tui.md) for the operator-facing walkthrough.
