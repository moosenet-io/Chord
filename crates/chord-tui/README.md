# chord-tui (S91 CTUI-01..06)

> Operator guide & full reference: [`docs/chord-tui.md`](../../docs/chord-tui.md).

A read-mostly terminal control UI (ratatui + crossterm + tokio) for the **Chord**
and **Terminus-fleet** control planes. It is a **client** — it connects to the
existing stable control endpoints and **never** links, restarts, or reconfigures
the live proxy binary.

## Two modes, one binary

- **Chord mode** — Chord's control plane.
- **Terminus-fleet mode** — the Terminus fleet control plane.

They share all plumbing (async connection manager, config, secrets, event loop)
but are **separate views**, never blended. `Tab` switches modes.

## Panels

### Chord mode
- **Models / Backends** (CTUI-02, *live now*) — wrap Chord's stable endpoints
  (`/health`, `/api/models`, `/api/storage`). Read-first; missing API fields
  degrade to "field unavailable"; a mid-sweep/busy flag suppresses disruptive
  actions.
- **Serving / Coordinator / Clean-Swap** (CTUI-03, *stubbed, pending S85*) — built
  against the `ServingControl` trait with a clearly-named `MockServingControl`.
  Panels render + navigate now and show a **"pending S85 integration"** banner.
  Swapping in the real S85 client is a single localized change.

### Terminus-fleet mode (CTUI-04 connect+status, CTUI-05 per-instance config)
- **Instances** — add/remove/select instances. Each declares a transport
  (stdio | HTTP), an endpoint (from config, never a literal), and a kind
  (local | remote | chord-embedded). A chord-embedded Terminus is just another
  instance. Same-name instances are disambiguated by name + endpoint.
- **Detail** — per-instance connection status, tool count, and tool inventory
  (name, enabled, domain). Connects over stdio OR HTTP via `mcp_client`.
  Unexpected MCP version → shown *incompatible* (no crash, tools untrusted);
  stdio process death → *disconnected + retriable*; remote auth fail → per-instance
  `auth-failed` only. **One unreachable instance never breaks the others.**
- **Tools** — enable/disable a tool + view/edit scopes, **only if the server
  exposes the control**; otherwise read-only "not supported" (no faked mutation).
  Confirm-gated; the UI reflects the server's true state (no optimistic lie) and
  reverts + explains on rejection.
- **Secrets** — per-instance **vault-backed** management: names + status only,
  values **never** shown. Change = **typed** confirmation, written to the vault
  only (never file/screen), atomic-or-nothing.
- **Transport** — set stdio/HTTP endpoint from config; a blank endpoint is
  rejected (never defaulted); no hardcoded infra.

Every fleet mutation is **audit-logged sanitized** (action + instance + target
NAME + outcome — never a secret value).

## Safety model

- Read-mostly by default.
- **Simple** mutations (model pull/archive) require an explicit confirm keystroke.
- **Destructive** mutations (unload live model, clean-swap, secret change) require
  a **typed** confirmation of an exact challenge phrase.
- Stubbed / not-yet-wired S85 mutations are **INERT** — gated OFF by
  `settings.enable_stubbed_mutations` (default `false`). A stubbed clean-swap
  performs **no real operation**.

## Secrets

Secrets come from a vault-backed `SecretManager`. The config file stores only a
`SecretRef` (a vault key **name**) — never a value. `SecretValue` is not
`Serialize`, so persisting one is a compile error. Secret values are never
displayed or logged.

## Config

Persisted at the platform config dir (`chord-tui/config.toml`). Missing config →
empty fleet + add-instance prompt. Corrupt config → backed up to
`config.toml.corrupt-<ts>`, a fresh default starts, and a warning is surfaced —
the fleet is never silently lost. No infrastructure endpoints are baked in; the
operator supplies the instance list.
