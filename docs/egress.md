# Chord — Launch-Env Scrub & Egress Policy (S88 ISO-01)

> **Honest scope.** ISO-01 ships the **launch-environment scrub** and the
> **egress-policy config surface** only. It is **ADVISORY**: it sets documented
> telemetry-off / offline opt-out variables and strips proxy variables, and it
> *declares* a per-launch egress posture — but it relies on the runtimes HONOURING
> those opt-outs. A binary that ignores them is **not** stopped by this layer.
>
> The **kernel guarantee** — a network namespace that physically blocks egress so a
> misbehaving runtime cannot reach the internet — is **ISO-02 and is NOT built
> yet**. Do not treat ISO-01 as a security boundary. It is defense-in-depth plus
> the policy plumbing that ISO-02 will enforce.

As of chord-proxy **1.3.0**.

## Modules

| Box | Module | Responsibility |
|-----|--------|----------------|
| Launch-env scrub | [`supervisor::launch_env`](../src/supervisor/launch_env.rs) | Build the minimal, telemetry-off, proxy-stripped env a runtime child is spawned with. |
| Egress policy | [`supervisor::egress_policy`](../src/supervisor/egress_policy.rs) | Decide the per-launch egress posture (`Serve` → Denied, `Pull` → allow-list or Denied). |
| Config surface | [`config`](../src/config.rs) | `model_source_allowlist()`, `CHORD_OUTBOUND_PROXY`, `CHORD_RUNTIME_TELEMETRY_OFF`. |

## What the scrub does

`build_runtime_env(class, &Config)` ([`launch_env`](../src/supervisor/launch_env.rs))
returns the COMPLETE env for a runtime launch:

1. **Minimal passthrough** — only `PATH`, `HOME`, `LANG`, `LC_ALL`, `TZ`, `TMPDIR`
   are carried from the supervisor. The full environment is NOT inherited; nothing
   else (secrets, ambient proxy vars, inherited telemetry toggles) leaks in.
2. **Telemetry-off / offline opt-outs** (always set):
   - `DO_NOT_TRACK=1` (cross-tool do-not-track convention)
   - `OLLAMA_NO_ANALYTICS=1`, `OLLAMA_NOPRUNE=1`, `OLLAMA_NO_UPDATE_CHECK=1`
   - `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`
3. **Proxy strip** — `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY` (and lowercase) are
   NEVER inherited. They are only (re)introduced when Chord is explicitly
   configured with `CHORD_OUTBOUND_PROXY` **and** the launch is an allow-listed
   `Pull`. A `Serve` launch never receives a proxy (it has no egress).

The serving launcher
([`serving::launcher::scrub_launch_env`](../src/serving/launcher.rs)) layers the
runtime-specific env (the gfx override / cpu-lib pairs) ON TOP of this scrubbed
base, so a per-launch override always wins. The integration is **additive**: the
only new variables on an existing cold-launch are the telemetry-off/offline ones;
nothing the launcher previously set is dropped. A `Launcher` built with
`Launcher::with_scrub(.., &cfg)` applies the scrub on every cold-launch.

## Egress posture

`posture_for(class, &Config)` ([`egress_policy`](../src/supervisor/egress_policy.rs)):

| `RuntimeClass` | Posture |
|----------------|---------|
| `Serve` | `Denied` — a serving runtime answers local requests; it never needs the net. |
| `Pull`  | `AllowList(model_source_allowlist)` — egress only to the configured sources. |
| `Pull` with an **unset/empty** allow-list | `Denied` — **FAIL CLOSED**. Never allow-all. |

### The allow-list is CONFIG, fail-closed

`MODEL_SOURCE_ALLOWLIST` is the list of model-source hosts/domains a `Pull` may
reach (comma/space separated). It is **never a baked-in host**. When it is **unset
or empty**, `model_source_allowlist()` logs a loud `tracing::warn!` and every pull
is **Denied** — Chord fails closed rather than defaulting to allow-all. The
production value for a given host comes from that host's **egress audit**; the
examples in [`.env.example`](../.env.example) (`registry.ollama.ai`,
`huggingface.co`) are public placeholders only.

## What ISO-02 adds (not in 1.3.0)

ISO-02 will consume `EgressPosture` in the launcher's spawn path to build a
**network namespace** for the runtime child, so a `Denied` posture is enforced by
the kernel (the runtime physically cannot open an outbound socket) and an
`AllowList` posture is enforced by firewall rules scoped to the listed sources.
Only then does the policy become a guarantee rather than advice.
