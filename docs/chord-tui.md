# chord-tui — Operator Guide & Reference (S91)

`chord-tui` is a terminal control panel (a TUI) for two things at once:

1. **Chord** — the model-serving proxy (which models are loaded, on which
   backend, disk usage).
2. **Terminus fleet** — one or more MCP tool servers (their health, their tool
   inventory, and per-instance configuration).

It is a **read-mostly client**. It connects over the network (or a local
process) to Chord and Terminus and shows you their state. It **never restarts,
reconfigures, or replaces** a running Chord or Terminus process. You can watch a
live system safely, and any change you make is deliberately gated behind a
confirmation.

> Nothing in `chord-tui` touches the live serving process on its own. It is a
> separate program you run in your terminal.

---

## Two modes, one program

There is a single binary with **two modes** that share the same plumbing
(connection handling, config, secrets) but are shown as **separate screens** —
they are never blended into one cluttered view:

| Mode | What it shows |
|------|---------------|
| **CHORD** | The Chord serving control plane: models, backends, and (once S85 lands) serving/coordinator/clean-swap panels. |
| **TERMINUS-FLEET** | Your fleet of Terminus MCP servers: health, tool inventory, and per-instance tool/secret/transport config. |

You switch between them with a single key. Switching modes also cancels any
half-entered confirmation, so a pending change can never "leak" across modes.

---

## Keybindings & navigation

Global:

| Key | Action |
|-----|--------|
| `Tab` | Switch mode (CHORD ↔ TERMINUS-FLEET) |
| `←` / `→` | Move between panels in the current mode |
| `q` | Quit |

In **CHORD** mode the panels are: Models · Backends · Serving (S85) ·
Coordinator (S85) · Clean-Swap (S85). On the Models panel, `p` requests a
(confirm-gated) model pull.

In **TERMINUS-FLEET** mode the panels are: Instances · Detail · Tools · Secrets ·
Transport. `a` adds a new instance.

Confirmation overlay (when a change is pending):

| Key | Action |
|-----|--------|
| `y` | Confirm a **simple** change |
| type the phrase, then `Enter` | Confirm a **destructive** change |
| `Esc` | Cancel |
| `Backspace` | Edit the typed phrase |

---

## CHORD mode — what works now vs. pending S85

`chord-tui` is honest about what is live today and what is a placeholder.

**Live now (reads Chord's existing, stable endpoints):**

- **Models** — the registry table: model name, tier (hot/warm/cold), loaded
  state, backend tag (e.g. `vulkan-radv`, `llama.cpp-rocm`, `cpu`), and size.
  Missing fields show `-` rather than breaking the row.
- **Backends** — how many models are loaded per backend.
- A **busy / mid-sweep** indicator: if Chord reports it is busy, disruptive
  actions are suppressed so you don't disturb a running sweep.

**Pending S85 (stubbed panels — clearly marked):**

- **Serving**, **Coordinator**, and **Clean-Swap** panels render a
  `pending S85 integration` banner. They exist so navigation and the safety flow
  are real, but they carry **placeholder data** and their mutations are **inert**
  (they perform no real operation). These become live in a later, localized
  change when the S85 serving surfaces are built — you will not need to relearn
  the UI.

---

## TERMINUS-FLEET mode

### Instances panel — your fleet

Each instance you add declares three things, all from your config (never
hardcoded):

- **Transport** — how the TUI reaches it:
  - **stdio** — a launcher command the TUI runs, speaking MCP over its
    stdin/stdout (good for a local Terminus).
  - **HTTP** — an endpoint URL (good for a remote Terminus).
- **Endpoint** — the URL (for HTTP) or command (for stdio). Supplied by you.
- **Kind** — `local`, `remote`, or `chord-embedded`. A Terminus embedded inside
  a Chord process is just another fleet instance.

**Adding an instance:** press `a`, then provide a name, transport, endpoint, and
(optionally) the vault key name for its auth token. The token **value** is never
typed into the config — only the name of the vault key that holds it.

**Same-name instances** are allowed and are disambiguated by their endpoint, so
two instances both called `terminus` pointing at different hosts are kept
distinct.

**One unreachable instance never breaks the others.** Every instance is polled
and operated on independently. A dead HTTP endpoint or a crashed stdio process
shows as a per-instance error; the rest of your fleet keeps working.

### Status states

| State | Meaning |
|-------|---------|
| `connected` | Handshake OK and the MCP version is supported. |
| `incompatible` | Reachable, but the server speaks an unsupported MCP version. Shown, not crashed; its tools are **not** trusted for changes. |
| `disconnected` | The stdio process died (or the HTTP connection dropped). Retriable. |
| `auth-failed` | A remote rejected the token (this instance only). |
| `error` | Any other transport error, isolated to this instance. |
| `idle` | Not connected yet. |

### Detail panel

Per-instance: name, endpoint, connection status, reported version, and the
**tool inventory** — each tool's name, whether it is enabled (if the server
reports it), and its domain/module.

### Tools panel — enable/disable + scopes

You can enable or disable an individual tool, and view/edit its scopes, **only if
that Terminus exposes the control**. If it doesn't, the panel shows the control
**read-only** as *"not supported by this instance"* and no change is attempted —
the TUI will **never fake** a mutation.

- Every toggle or scope edit needs an explicit **confirm** (`y`).
- The displayed state updates **only from the server's confirmed response**. If
  the server refuses a toggle, the UI keeps showing the true, unchanged state
  (no optimistic lie) and tells you why. A rejected scope edit reverts the
  display and explains the rejection.
- If the instance goes unreachable mid-change, the action fails cleanly and the
  panel re-syncs when the instance reconnects.

### Secrets panel — vault-backed, names only

Per-instance secrets (like an auth token) are managed through the **vault**.

- The panel shows only the secret's **name** and a **status**
  (`present` / `empty` / `missing`). **Secret values are never displayed,
  logged, or written to a file or the screen.**
- Changing a secret is a **destructive** action: you must **type** the
  confirmation phrase (not just press a key).
- A confirmed change writes **only to the vault** — never to the config file or
  the terminal. The write is **atomic-or-nothing**: if it is interrupted, the
  previous value is preserved, never half-written.

### Transport panel

Change an instance's transport (stdio command, or HTTP endpoint). Values come
from your input/config — there are **no hardcoded infrastructure endpoints**. A
blank endpoint is **rejected**, never silently replaced with a default, and an
HTTP endpoint must start with `http://` or `https://`.

---

## The safety model

`chord-tui` is read-mostly by design. Changes are gated by severity:

| Severity | How you confirm | Examples |
|----------|-----------------|----------|
| **Simple** | Press `y` | Pull/archive a model; toggle a tool; edit a scope; change a transport. |
| **Destructive** | **Type** the exact challenge phrase, then `Enter` | Clean-swap a live model (`CLEAN-SWAP`); change a secret (`CHANGE-SECRET`). |

Additional guardrails:

- **Stubbed (pending-S85) mutations are inert and off by default.** They run the
  confirmation UX but perform no real operation until the S85 wiring is verified
  and explicitly enabled.
- **Capability-honest.** If an instance doesn't expose a control, it is shown
  read-only; a mutation is never faked.
- **No optimistic lies.** The UI reflects the server/vault's true state; a failed
  change is shown as failed, with the real, unchanged state.
- **Secrets are never shown.** Only names and presence status. Changes go to the
  vault only.
- **Everything is audit-logged, sanitized.** Each attempted or applied change is
  recorded with the action, the instance, the target **name** (tool or secret
  reference), and the outcome — **never a secret value**.

---

## Configuration & environment

- The fleet + settings live in a TOML config file under your config directory.
  It stores instance **names, kinds, transports, endpoints, and vault key
  names** — **never secret values** (a secret value is structurally impossible to
  persist there).
- Secret values are resolved at connect time from the **vault** (e.g. an
  Infisical-injected environment). Provide them via your vault, referenced by
  name. Placeholder examples:

  ```
  TERMINUS_LOCAL_TOKEN=<vault-injected>
  TERMINUS_REMOTE_TOKEN=<vault-injected>
  CHORD_CONTROL_TOKEN=<vault-injected>
  ```

- A missing config starts an empty fleet (you'll be prompted to add an
  instance). A corrupt config is backed up and a fresh one is started, so your
  fleet is never silently lost.

There are **no baked-in endpoints or tokens** anywhere in `chord-tui`. Every
address is something you configure; every secret is a vault reference.
