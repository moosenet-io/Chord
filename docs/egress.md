# Chord — Runtime Egress: Launch-Env Scrub (ISO-01) + Netns Isolation (ISO-02)

Two layers govern what a runtime Chord launches can reach on the network:

| Layer | What it is | Guarantee |
|-------|-----------|-----------|
| **ISO-01** (1.3.0) | Launch-environment **scrub** + egress-policy **config surface** | **ADVISORY** — relies on the runtime honouring opt-outs |
| **ISO-02** (1.4.0) | Per-runtime **network namespace** (kernel-enforced) | **KERNEL GUARANTEE** for runtimes Chord LAUNCHES |

> **Honest scope of ISO-02.** ISO-02 is the load-bearing, kernel-enforced egress
> layer **for the runtimes Chord launches**. It does **NOT** firewall Chord's own
> process (that concern was ISO-03, addressed by dependency review) and it does
> **NOT** replace the host firewall for non-Chord processes on the box. It isolates
> the *launched runtime*, full stop.

As of chord-proxy **1.4.0**.

## ISO-01 — the policy surface (advisory)

ISO-01 ships the **launch-environment scrub** and the **egress-policy config
surface**. It sets documented telemetry-off / offline opt-out variables and strips
proxy variables, and it *declares* a per-launch egress posture — but it relies on
the runtimes HONOURING those opt-outs. A binary that ignores them is not stopped by
ISO-01 alone; that is what ISO-02 enforces.

### Modules (ISO-01)

| Box | Module | Responsibility |
|-----|--------|----------------|
| Launch-env scrub | [`supervisor::launch_env`](../src/supervisor/launch_env.rs) | Build the minimal, telemetry-off, proxy-stripped env a runtime child is spawned with. |
| Egress policy | [`supervisor::egress_policy`](../src/supervisor/egress_policy.rs) | Decide the per-launch egress posture (`Serve` → Denied, `Pull` → allow-list or Denied). |
| Config surface | [`config`](../src/config.rs) | `model_source_allowlist()`, `CHORD_OUTBOUND_PROXY`, `CHORD_RUNTIME_TELEMETRY_OFF`. |

### What the scrub does

`build_runtime_env(class, &Config)` returns the COMPLETE env for a runtime launch:

1. **Minimal passthrough** — only `PATH`, `HOME`, `LANG`, `LC_ALL`, `TZ`, `TMPDIR`
   are carried from the supervisor. The full environment is NOT inherited.
2. **Telemetry-off / offline opt-outs** (always set): `DO_NOT_TRACK=1`,
   `OLLAMA_NO_ANALYTICS=1`, `OLLAMA_NOPRUNE=1`, `OLLAMA_NO_UPDATE_CHECK=1`,
   `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`.
3. **Proxy strip** — `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY` (and lowercase) are
   NEVER inherited; only (re)introduced for an allow-listed `Pull` when
   `CHORD_OUTBOUND_PROXY` is set.

### Egress posture

`posture_for(class, &Config)`:

| `RuntimeClass` | Posture |
|----------------|---------|
| `Serve` | `Denied` — a serving runtime answers local requests; it never needs the net. |
| `Pull`  | `AllowList(model_source_allowlist)` — egress only to the configured sources. |
| `Pull` with an **unset/empty** allow-list | `Denied` — **FAIL CLOSED**. Never allow-all. |

`MODEL_SOURCE_ALLOWLIST` is CONFIG, never a baked-in host. Unset/empty → every pull
is `Denied`. The examples in [`.env.example`](../.env.example) (`registry.ollama.ai`,
`huggingface.co`) are public placeholders; a deployment's real list comes from its
egress audit.

## ISO-02 — the kernel guarantee (netns)

ISO-02 ENFORCES the ISO-01 posture by spawning each launched runtime **inside its
own Linux network namespace**. A misbehaving binary that ignores every ISO-01
opt-out still cannot reach a network it is not allowed to — the kernel denies it.

### Modules (ISO-02)

| Box | Module | Responsibility |
|-----|--------|----------------|
| Namespace lifecycle | [`supervisor::netns`](../src/supervisor/netns.rs) | Create / configure / teardown a per-runtime netns; posture→config mapping; fail-closed capability probe. |
| Pull egress filter | [`supervisor::egress_filter`](../src/supervisor/egress_filter.rs) | Build + apply the in-namespace nftables default-drop ruleset that restricts a Pull namespace to the allow-list. |
| Isolation decision | [`supervisor::launch_isolation`](../src/supervisor/launch_isolation.rs) | The fail-closed policy: isolate / disabled-by-config / unisolated-override / refused. |
| Launcher integration | [`serving::launcher`](../src/serving/launcher.rs) | Spawn the runtime INSIDE its namespace (with the ISO-01 scrubbed env); reap on launch failure. |
| Teardown tie-in | [`serving::swap`](../src/serving/swap.rs) | `NetnsReapingTeardown` reaps the outgoing runtime's namespace in the SRV-12 verified clean swap. |

### Serve (no route) vs. Pull (allow-list filtered)

* **`Serve` → `Denied`** → a namespace with **loopback up but NO default route and
  NO veth to the host**. There is no path off the box, so any external `connect()`
  fails at the kernel (`ENETUNREACH`). This is the load-bearing guarantee — a
  serving runtime physically cannot phone home or exfiltrate. No firewall is needed
  because there is no route to filter.
* **`Pull` → `AllowList`** → a namespace WITH a constrained egress path (a veth
  pair + default route through the host) AND an **nftables default-drop ruleset
  applied inside the namespace** that permits only DNS (to resolve the names) plus
  the configured allow-list hosts, dropping everything else. Nothing is
  blanket-allowed; an empty allow-list is `Denied` (no path at all).

### Mechanism choice

ISO-02 uses **`nix` (`unshare`/`CLONE_NEWNET`) for the namespace primitive and the
`ip`/`nft` userspace binaries for link/route/filter setup**, rather than the pure
`rtnetlink` crate. Rationale: a named netns (`/run/netns/<name>`) + a veth pair + a
default route + an in-namespace nftables ruleset is exactly what `iproute2`/`nft`
express concisely and what an operator can reproduce by hand under ISO-04; a named
netns also makes teardown a single idempotent unlink. `nix` is pinned at an explicit
version and is a **Linux-only** target dependency; the `ip`/`nft` binaries come from
`CHORD_IP_BIN`/`CHORD_NFT_BIN` config helpers (never a hardcoded path). On any
non-Linux platform `netns::prepare` returns a clear `Unsupported` error — never a
silent no-op that would masquerade as isolation.

### Fail-closed + the override

Creating/configuring a namespace needs `CAP_NET_ADMIN` (and the `ip`/`nft`
userspace). ISO-02 is **fail-closed**:

* **Isolation is default-ON** (`CHORD_NETNS_ISOLATION`, disabled only by an explicit
  `0` — the dev/CI opt-out that takes the legacy non-isolated path).
* When isolation is ON but the namespace **cannot** be created (missing capability,
  non-Linux, missing tooling), the launch is **REFUSED** — the runtime is NOT
  spawned with full host egress. `serve_model` returns `IsolationRefused` / records
  `isolation-refused`.
* An operator may set **`CHORD_ALLOW_UNISOLATED=1`** (exactly `1`) to launch WITHOUT
  isolation anyway. This is the only sanctioned bypass: it is **loud** (logged at
  `warn`), **explicit**, and **off by default**.

### Teardown (ties into the SRV-12 clean swap)

Namespace teardown is **idempotent**: deleting an already-gone (or never-created,
or half-built) namespace is a success, so a crashed runtime never leaks a netns.
The SRV-12 verified clean swap reaps the outgoing runtime's namespace via
`NetnsReapingTeardown` — it runs the backend teardown first (a failed teardown
aborts the swap and does NOT reap, since the runtime may still be alive), then reaps
the namespace by its derived name. A launch that fails AFTER the namespace was
created (spawn error / health-check failure) reaps the namespace immediately.

## Tests + ISO-04 verification

The pure cores are unit-tested without privilege: posture→netns-config mapping, the
fail-closed decision (a NEGATIVE test asserting a missing capability yields *refused*
and never a full-egress launch), the nft allow-list ruleset construction (including
empty-list = deny-all and malformed-host rejection), and teardown idempotency.

The end-to-end kernel behaviour — a Serve namespace's external connect failing, a
Pull namespace's allow-listed reachable / non-allow-listed denied — needs
`CAP_NET_ADMIN` and so lives in `#[ignore]`d integration tests
([`tests/netns_integration.rs`](../tests/netns_integration.rs)). They compile in
every build and run under `cargo test --test netns_integration -- --ignored` on a
privileged host. That on-host verification is **ISO-04** (a separate operator task);
in an unprivileged build these tests are skipped, which is expected.

> **What ISO-02 could not fully exercise in an unprivileged build.** The actual
> `ip`/`nft`-in-namespace *apply* (veth + route + the nftables default-drop) requires
> runtime privilege. That code path is built and unit-tested at the construction
> boundary, but its live enforcement is asserted by the gated integration tests
> under ISO-04 — not by this CI run.
