# supervisor

The runtime supervisor (70 KG nodes, `src/supervisor/`): launch-environment
scrubbing plus network egress policy for every runtime Chord launches (S88
ISO-01/ISO-02). ISO-01 is *advisory* — a scrubbed env with telemetry-off /
offline opt-outs the runtimes are expected to honor. ISO-02 is the *kernel
guarantee* — a per-runtime network namespace that physically blocks the egress
ISO-01 only declared, fail-closed: without `CAP_NET_ADMIN` the launch is
refused, never run with full host egress.

## Key types and functions

| Symbol | Kind | File | What it does |
|---|---|---|---|
| `supervisor::egress_policy::posture_for` | function | `src/supervisor/egress_policy.rs` | Maps a `RuntimeClass` to its `EgressPosture` (`Serve`/`Denied` vs `Pull`/`AllowList`) |
| `supervisor::egress_policy::EgressPosture` / `RuntimeClass` | enums | `src/supervisor/egress_policy.rs` | The policy vocabulary: what kind of launch, what egress it deserves |
| `supervisor::launch_env::build_runtime_env` | function | `src/supervisor/launch_env.rs` | The scrubbed child environment: minimal vars, telemetry-off/offline opt-outs |
| `supervisor::launch_isolation::decide_isolation` | function | `src/supervisor/launch_isolation.rs` | The launch-time decision: isolate, refuse, or (explicit override only) run unisolated |
| `supervisor::launch_isolation::IsolationDecision` | enum | `src/supervisor/launch_isolation.rs` | The three-way outcome consumed by the serving launcher |
| `supervisor::netns::NetnsConfig::from_posture` | function | `src/supervisor/netns.rs` | Namespace parameters derived from the egress posture |
| `supervisor::netns::prepare` | function | `src/supervisor/netns.rs` | Creates the namespace (`unshare` + `CLONE_NEWNET` via the `nix` crate, Linux-only); probes `CAP_NET_ADMIN` fail-closed |
| `supervisor::netns::namespace_name` / `NetnsHandle` | function / struct | `src/supervisor/netns.rs` | Deterministic naming + the handle used for teardown |
| `supervisor::egress_filter::build_nft_ruleset` | function | `src/supervisor/egress_filter.rs` | Generates the nftables ruleset for `Pull`/`AllowList` launches (model sources only) |
| `supervisor::egress_filter::sanitize_host` | function | `src/supervisor/egress_filter.rs` | Validates allowlist hosts before they reach a ruleset |

## How it connects

**serving** is the consumer: `serving::launcher` calls `build_runtime_env` for
the child env and `launch_isolation`/`netns` to spawn the runtime inside its
namespace; the SRV-12 clean swap (`serving::swap`, `NetnsReapingTeardown`)
tears the outgoing runtime's namespace down. A `Serve`/`Denied` runtime gets a
namespace with **no route** — every external `connect()` fails at the kernel; a
`Pull`/`AllowList` runtime gets a constrained, nftables-filtered path to the
configured model sources (`MODEL_SOURCE_ALLOWLIST`) only. The netns integration
test lives at `tests/netns_integration.rs`.

## Configuration

`CHORD_RUNTIME_TELEMETRY_OFF`, `CHORD_OUTBOUND_PROXY`, `CHORD_IP_BIN`,
`CHORD_NFT_BIN`, `CHORD_ALLOW_UNISOLATED` (the loud, off-by-default override),
`MODEL_SOURCE_ALLOWLIST`.

## Notes and gaps

- Honest scope (from the module docs and [../egress.md](../egress.md)): ISO-02
  isolates the runtimes Chord **launches**. It does not firewall Chord's own
  process (that would be ISO-03) and does not replace the host firewall for
  non-Chord processes.
- The privileged netns path is `#[cfg(target_os = "linux")]`; on other
  platforms `netns::prepare` returns `Unsupported`.
- ISO-01's opt-outs depend on runtime cooperation; only the namespace is a
  guarantee.
