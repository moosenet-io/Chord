//! Per-runtime network-namespace isolation — S88 ISO-02 (the KERNEL guarantee).
//!
//! ISO-01 ([`super::egress_policy`]) declares the *posture* a runtime launch
//! should have; this module ENFORCES it with a Linux network namespace so a
//! misbehaving binary physically cannot reach the network it is not allowed to.
//!
//! ## What the kernel guarantees here
//!   * **`Serve` ([`EgressPosture::Denied`])** → a netns with **loopback up but no
//!     default route and no veth to the host** → there is no path off the box, so
//!     any external `connect()` fails at the kernel (`ENETUNREACH`). This is the
//!     load-bearing guarantee: a serving runtime cannot exfiltrate or phone home
//!     even if it ignores every ISO-01 opt-out.
//!   * **`Pull` ([`EgressPosture::AllowList`])** → a netns WITH a constrained
//!     egress path (a veth pair + default route through the host), and the
//!     [`super::egress_filter`] nftables ruleset applied *inside* the namespace so
//!     only the configured `model_source_allowlist()` hosts are reachable. Nothing
//!     is blanket-allowed: an empty allow-list is `Denied` (no route at all).
//!
//! ## Honest scope
//! This isolates the runtimes Chord LAUNCHES. It does **not** firewall Chord's own
//! process (that was ISO-03, handled by dependency review) and it does **not**
//! replace the host firewall for non-Chord processes. See `docs/egress.md`.
//!
//! ## Privilege + fail-closed
//! Creating/configuring a netns needs `CAP_NET_ADMIN` (and the `ip`/`nft`
//! userspace, or an equivalent privileged path). When that capability is absent we
//! **FAIL CLOSED**: [`prepare`] returns [`NetnsError::CapabilityUnavailable`] and
//! the launcher must NOT fall back to a full-host-egress launch. An operator may
//! set `CHORD_ALLOW_UNISOLATED=1` to deliberately run without isolation — that path
//! is loud (logged at `warn`), explicit, and off by default.
//!
//! ## Linux-only
//! Network namespaces are a Linux primitive. The whole privileged path is
//! `#[cfg(target_os = "linux")]`; on any other OS [`prepare`] returns
//! [`NetnsError::Unsupported`] (a clear runtime error, never a silent no-op that
//! would look like isolation).
//!
//! ## Mechanism choice (documented)
//! We use **`nix` (`unshare`/`setns` + `CLONE_NEWNET`) for the namespace primitive
//! and the `ip`/`nft` userspace binaries for link/route/filter setup**, rather than
//! the pure-`rtnetlink` crate. Rationale: the configuration we need (named netns,
//! a veth pair straddling host↔netns, a default route, and an nftables ruleset
//! *inside* the namespace) is exactly what `iproute2` expresses concisely and what
//! every operator can reproduce by hand under ISO-04. `nix` is added at an explicit
//! version; the `ip`/`nft` binaries are resolved from config helpers (never a
//! hardcoded path). A named netns persisted under `/run/netns/<name>` also makes
//! teardown a single idempotent unlink, which is what the SRV-12 clean-swap needs.

use super::egress_policy::EgressPosture;

/// A prepared, configured network namespace ready to host one runtime launch.
///
/// Construct via [`prepare`]. Spawn the runtime *into* this namespace by applying
/// [`NetnsHandle::enter_arg`]/[`NetnsHandle::wrap_command`] to the launch command,
/// and tear it down with [`NetnsHandle::teardown`] (or rely on the SRV-12 clean
/// swap, which calls teardown for the outgoing runtime). Teardown is idempotent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetnsHandle {
    /// The kernel/iproute2 name of the namespace (`/run/netns/<name>`). Derived
    /// from a Chord-internal token + the model id hash; never infra-leaking.
    name: String,
    /// The posture this namespace was built to enforce (recorded for teardown +
    /// observability). `Denied` → no route; `AllowList` → filtered egress path.
    posture: EgressPosture,
}

impl NetnsHandle {
    /// Reconstruct a handle from a namespace NAME alone, purely to call its
    /// idempotent [`teardown`](NetnsHandle::teardown). Used by the launcher /
    /// clean-swap teardown hook, which records only the name on a [`ServeHandle`].
    /// The posture is irrelevant to teardown (deletion needs only the name), so a
    /// placeholder `Denied` is used.
    pub fn for_teardown(name: &str) -> Self {
        NetnsHandle {
            name: name.to_string(),
            posture: EgressPosture::Denied,
        }
    }

    /// The namespace name (`/run/netns/<name>` key).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The posture this namespace enforces.
    pub fn posture(&self) -> &EgressPosture {
        &self.posture
    }

    /// The `ip netns exec <name>` prefix tokens to run a command INSIDE this
    /// namespace. The launcher prepends these to the runtime's `bin`+`args` so the
    /// spawned process lands in the isolated namespace at exec time.
    ///
    /// `ip_bin` is the resolved `ip` binary (from a config helper — never a literal).
    pub fn enter_argv(&self, ip_bin: &str) -> Vec<String> {
        vec![
            ip_bin.to_string(),
            "netns".to_string(),
            "exec".to_string(),
            self.name.clone(),
        ]
    }

    /// Rewrite a launch (bin + args) so it executes inside this namespace:
    /// `["llama-server", "--model", x]` → `["ip","netns","exec","<ns>","llama-server","--model",x]`.
    /// Pure data transform — does not spawn anything, so it is unit-testable.
    pub fn wrap_command(&self, ip_bin: &str, bin: &str, args: &[String]) -> (String, Vec<String>) {
        let mut wrapped = self.enter_argv(ip_bin);
        wrapped.push(bin.to_string());
        wrapped.extend_from_slice(args);
        // The new program to exec is `ip`; the original bin becomes its argument.
        let prog = wrapped.remove(0);
        (prog, wrapped)
    }

    /// Tear down the namespace (delete `/run/netns/<name>` + its veth, if any).
    /// **Idempotent**: deleting an already-gone namespace is `Ok(())`, so a crashed
    /// runtime never leaks a namespace and the SRV-12 clean swap can always call it.
    pub fn teardown(&self) -> Result<(), NetnsError> {
        teardown_named(&self.name)
    }
}

/// Why preparing/tearing down a network namespace failed. Genericized (S77): a
/// stable reason, no host/path/cap detail beyond the category.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetnsError {
    /// The privilege needed to create/configure a netns (`CAP_NET_ADMIN` / a
    /// usable user namespace) is unavailable. **FAIL CLOSED** — the caller must NOT
    /// launch with full host egress; it either refuses or honours an explicit
    /// `CHORD_ALLOW_UNISOLATED=1` operator override.
    CapabilityUnavailable,
    /// Network namespaces are not supported on this platform (non-Linux). A clear
    /// error, never a silent no-op.
    Unsupported,
    /// The userspace tool (`ip`/`nft`) needed to configure the namespace is not
    /// configured/available.
    ToolUnavailable,
    /// Configuring the namespace (link/route/filter) failed after creation. The
    /// partially-created namespace is torn down before this is returned (no leak).
    ConfigureFailed,
}

impl std::fmt::Display for NetnsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NetnsError::CapabilityUnavailable => f.write_str(
                "network-namespace isolation requires CAP_NET_ADMIN; refusing to launch \
                 without isolation (set CHORD_ALLOW_UNISOLATED=1 to override, not recommended)",
            ),
            NetnsError::Unsupported => {
                f.write_str("network-namespace isolation is only supported on Linux")
            }
            NetnsError::ToolUnavailable => {
                f.write_str("the userspace tool needed to configure network isolation is unavailable")
            }
            NetnsError::ConfigureFailed => {
                f.write_str("failed to configure the runtime's isolated network namespace")
            }
        }
    }
}

impl std::error::Error for NetnsError {}

/// Whether the operator has explicitly opted OUT of isolation. Loud + off by
/// default: only an exact `CHORD_ALLOW_UNISOLATED=1` counts. When this is set and
/// the capability is missing, the launcher may proceed unisolated — but it logs a
/// `warn` first (see [`super::launch_isolation`]).
pub fn unisolated_override() -> bool {
    matches!(
        std::env::var("CHORD_ALLOW_UNISOLATED").ok().as_deref(),
        Some("1")
    )
}

/// Whether netns isolation is the default-ON path. Recommended-on per the S88
/// spec; gated behind `CHORD_NETNS_ISOLATION` so unprivileged dev/CI (and the
/// existing tests) still run without a privileged host. **Defaults to ON** — an
/// unset or any value other than `0` enables isolation; only `CHORD_NETNS_ISOLATION=0`
/// disables it (and that disable is itself an unisolated path, so it is gated the
/// same loud way as the override).
pub fn isolation_enabled() -> bool {
    !matches!(
        std::env::var("CHORD_NETNS_ISOLATION").ok().as_deref(),
        Some("0")
    )
}

/// Map an [`EgressPosture`] to the netns CONFIG that enforces it. Pure mapping,
/// no privilege — the unit-testable core of the posture→namespace decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetnsConfig {
    /// `true` → an egress path (veth + default route) is created; `false` → the
    /// namespace gets loopback only and NO route (the `Serve`/`Denied` guarantee).
    pub egress_path: bool,
    /// When `egress_path` is true, the allow-list of hosts the
    /// [`super::egress_filter`] restricts egress to. Empty ⇒ there is no egress
    /// path at all (the `Denied` case never reaches here with a path).
    pub allow_list: Vec<String>,
}

impl NetnsConfig {
    /// Derive the namespace configuration from the enforced posture.
    ///   * [`EgressPosture::Denied`] → **no egress path** (loopback only, no route).
    ///   * [`EgressPosture::AllowList`] → an egress path filtered to the listed
    ///     hosts. An *empty* allow-list can never appear in an `AllowList` (ISO-01
    ///     `posture_for` collapses it to `Denied`); defensively we still treat an
    ///     empty list as no-egress-path (fail closed), never allow-all.
    pub fn from_posture(posture: &EgressPosture) -> Self {
        match posture {
            EgressPosture::Denied => NetnsConfig {
                egress_path: false,
                allow_list: Vec::new(),
            },
            EgressPosture::AllowList(hosts) if !hosts.is_empty() => NetnsConfig {
                egress_path: true,
                allow_list: hosts.clone(),
            },
            // Defensive fail-closed: an (impossible) empty AllowList → no egress.
            EgressPosture::AllowList(_) => NetnsConfig {
                egress_path: false,
                allow_list: Vec::new(),
            },
        }
    }
}

/// Prepare (create + configure) a network namespace that enforces `posture` for a
/// runtime identified by `slot_token` (a Chord-internal, non-infra token used to
/// derive a stable, unique namespace name).
///
/// FAIL CLOSED: on a host without the privilege/platform/tooling this returns the
/// matching [`NetnsError`] and creates NOTHING. The launcher treats any `Err` as
/// "do not launch unisolated" (unless the explicit operator override is set).
///
/// The actual privileged create/configure runs only under `#[cfg(target_os =
/// "linux")]` AND when the capability probe passes. The integration tests that
/// exercise the real namespace are `#[ignore]`d so they only run on a privileged
/// host under ISO-04.
///
/// ## CHRD-77: why the capability probe is injectable
/// The fail-closed DECISION ("no capability => [`NetnsError::CapabilityUnavailable`],
/// and NOTHING is created") used to be testable only by running the unit tests on a
/// host that happened to LACK `CAP_NET_ADMIN`. Chord's build/test host runs as root,
/// so that precondition was silently false: the unit test drove the REAL privileged
/// path, created a live namespace on the build host, leaked it, and then failed —
/// meaning the guard it claims to cover was never actually exercised anywhere. The
/// probe is now injected through [`prepare_with_probe`] so the decision is provable
/// on ANY host with zero side effects; [`prepare`] itself is behaviorally unchanged.
///
/// ## CHRD-97: why a unit-test build can never reach the privileged path
/// Chord's build/test host runs as ROOT, so on that host the real probe passes and
/// this function used to actually create `/run/netns/chord-<hash>` whenever a unit
/// test reached it — `supervisor::launch_isolation::tests` and
/// `serving::launcher::tests` both drive it through `decide_isolation`. That leaked
/// live namespaces onto the build host (three were found after one run), and a test
/// suite that mutates host network state is a hazard well past the leak: it can
/// collide with a concurrent run, and this build host also runs live services.
///
/// The fix reuses the seam that already existed: under `cfg(test)` `prepare` calls
/// [`prepare_with_probe`] with a probe pinned to `false` and a create operation that
/// is `unreachable!()`, so the privileged branch is not merely unlikely — it cannot
/// be entered, and entering it would panic loudly rather than silently touch the
/// host. `cfg(test)` applies ONLY to the crate's own unit-test build; a production
/// build (and the `tests/` integration build, where the real path is deliberately
/// exercised) compiles the unchanged privileged branch.
pub fn prepare(slot_token: &str, posture: &EgressPosture) -> Result<NetnsHandle, NetnsError> {
    #[cfg(all(target_os = "linux", not(test)))]
    {
        prepare_with_probe(slot_token, posture, has_net_admin, configure_namespace)
    }

    #[cfg(all(target_os = "linux", test))]
    {
        // TEST BUILD ONLY. Same seam, side-effect-free arguments: fail closed at the
        // capability gate, and make the create path a hard panic so a future refactor
        // that reorders the guard is caught here instead of on the host's `ip netns`.
        prepare_with_probe(slot_token, posture, || false, |_name, _cfg| {
            unreachable!(
                "CHRD-97: the privileged namespace-create path must be unreachable in a \
                 unit-test build; the real path is covered by the #[ignore]d \
                 tests/netns_integration.rs suite on a privileged host"
            )
        })
    }

    #[cfg(not(target_os = "linux"))]
    {
        let _ = (slot_token, posture);
        Err(NetnsError::Unsupported)
    }
}

/// [`prepare`] with BOTH privileged-path seams passed in (CHRD-77).
///
/// Two things are injected, and they serve different purposes:
///
/// * `has_cap` — the `CAP_NET_ADMIN` probe. Injecting it makes the fail-closed
///   DECISION provable without privilege, on any host.
/// * `create_ns` — the namespace-CREATING operation. Injecting it makes the
///   *structural* guarantee testable: a test can assert the create path was never
///   ENTERED, which is the actual invariant, instead of inspecting `/run/netns`
///   afterwards and inferring it from filesystem state (a proxy that reads live host
///   state and can be confounded by an unrelated concurrent run).
///
/// When `has_cap()` is `false` this returns [`NetnsError::CapabilityUnavailable`]
/// **before touching the kernel or the `ip` binary**, so no namespace can be created
/// and no handle can escape. Production always passes [`has_net_admin`] and
/// [`configure_namespace`], so `prepare`'s behavior is unchanged.
///
/// CHRD-97: this is `pub(crate)` rather than module-private so
/// [`super::launch_isolation`]'s tests can build a side-effect-free fake `prepare`
/// out of the seam that already exists here, instead of inventing a second one.
#[cfg(target_os = "linux")]
pub(crate) fn prepare_with_probe<F, G>(
    slot_token: &str,
    posture: &EgressPosture,
    has_cap: F,
    create_ns: G,
) -> Result<NetnsHandle, NetnsError>
where
    F: FnOnce() -> bool,
    G: FnOnce(&str, &NetnsConfig) -> Result<(), NetnsError>,
{
    let cfg = NetnsConfig::from_posture(posture);
    let name = namespace_name(slot_token);

    // FAIL CLOSED, before any privileged syscall or userspace invocation.
    if !has_cap() {
        return Err(NetnsError::CapabilityUnavailable);
    }
    create_ns(&name, &cfg)?;
    tracing::info!(
        target: "chord.supervisor.netns",
        ns = %name,
        egress_path = cfg.egress_path,
        allow_hosts = cfg.allow_list.len(),
        "prepared isolated network namespace for runtime launch (ISO-02)"
    );
    Ok(NetnsHandle {
        name,
        posture: posture.clone(),
    })
}

/// Derive a stable, unique, non-infra namespace name from a slot token. Hashed so
/// the name carries no model id / path; prefixed `chord-` so an operator can spot +
/// reap Chord-owned namespaces under ISO-04.
pub fn namespace_name(slot_token: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    slot_token.hash(&mut h);
    format!("chord-{:016x}", h.finish())
}

// ─────────────────────────────────────────────────────────────────────────────
// Privileged path (Linux only). In an unprivileged build the capability probe
// fails first, so these are exercised by the #[ignore]d integration tests on a
// privileged host (ISO-04), not by this CI run.
// ─────────────────────────────────────────────────────────────────────────────

/// Probe for `CAP_NET_ADMIN` (the capability netns create/configure needs). We
/// check effective capabilities via the kernel rather than assume root==cap, so a
/// dropped-capability service is correctly detected as unprivileged → fail closed.
#[cfg(target_os = "linux")]
fn has_net_admin() -> bool {
    // unshare(CLONE_NEWNET) is the minimal privileged op; if we cannot even create
    // a throwaway net namespace we definitively lack the capability. We do this in
    // a child so the probe never mutates the supervisor's own namespace.
    //
    // SAFETY/PRIVILEGE: this is the single capability gate. A false here is the
    // fail-closed signal the launcher depends on.
    use nix::sched::{unshare, CloneFlags};
    use nix::unistd::{fork, ForkResult};

    match unsafe { fork() } {
        Ok(ForkResult::Child) => {
            // In the child: try the unshare; exit 0 on success, 1 on failure.
            let ok = unshare(CloneFlags::CLONE_NEWNET).is_ok();
            // Use _exit to avoid running atexit handlers in the forked child.
            unsafe { nix::libc::_exit(if ok { 0 } else { 1 }) };
        }
        Ok(ForkResult::Parent { child }) => {
            use nix::sys::wait::{waitpid, WaitStatus};
            matches!(waitpid(child, None), Ok(WaitStatus::Exited(_, 0)))
        }
        Err(_) => false,
    }
}

/// Create + configure the named namespace per `cfg`. On ANY failure the partially
/// created namespace is torn down before returning `ConfigureFailed` (no leak).
#[cfg(target_os = "linux")]
// CHRD-97: in a unit-test build `prepare` no longer references this (the create seam
// is a panicking stub), so it is legitimately unreachable there. It is still the
// production create path and is exercised by tests/netns_integration.rs.
#[cfg_attr(test, allow(dead_code))]
fn configure_namespace(name: &str, cfg: &NetnsConfig) -> Result<(), NetnsError> {
    let ip = crate::config::ip_bin().ok_or(NetnsError::ToolUnavailable)?;

    // (1) Create the named netns: `ip netns add <name>`.
    if run_priv(&ip, &["netns", "add", name]).is_err() {
        return Err(NetnsError::ConfigureFailed);
    }

    // (2) Bring loopback up INSIDE the namespace (always — a serving runtime binds
    // its local HTTP socket on loopback): `ip netns exec <name> ip link set lo up`.
    if run_priv(&ip, &["netns", "exec", name, &ip, "link", "set", "lo", "up"]).is_err() {
        let _ = teardown_named(name);
        return Err(NetnsError::ConfigureFailed);
    }

    if cfg.egress_path {
        // (3) Pull posture: build the constrained egress path (veth + default route)
        // and apply the allow-list filter INSIDE the namespace. Any step failing
        // tears the namespace down (fail closed) — we never leave a half-open path.
        if super::egress_filter::configure_pull_egress(name, &ip, &cfg.allow_list).is_err() {
            let _ = teardown_named(name);
            return Err(NetnsError::ConfigureFailed);
        }
    }
    // Serve posture: nothing else. Loopback only, NO default route → no path off
    // the box → the kernel denies every external connect. That IS the guarantee.

    Ok(())
}

/// Tear down a named namespace idempotently: `ip netns del <name>` (deleting an
/// absent namespace is treated as success). Linux-only; on other platforms this is
/// a no-op success (nothing was ever created).
fn teardown_named(name: &str) -> Result<(), NetnsError> {
    #[cfg(target_os = "linux")]
    {
        let ip = match crate::config::ip_bin() {
            Some(b) => b,
            // No tool ⇒ nothing could have been created by us ⇒ idempotent success.
            None => return Ok(()),
        };
        // `ip netns del` on a missing namespace returns non-zero; we treat that as
        // success (idempotency) by checking existence first.
        if !named_netns_exists(name) {
            return Ok(());
        }
        let _ = run_priv(&ip, &["netns", "del", name]);
        // The veth host-side peer is auto-removed with the namespace; nft rules
        // lived inside the namespace and die with it. Nothing else to reap.
        Ok(())
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = name;
        Ok(())
    }
}

/// Whether a named netns currently exists (`/run/netns/<name>`). Used to make
/// teardown idempotent without shelling out to a delete that errors on absence.
#[cfg(target_os = "linux")]
fn named_netns_exists(name: &str) -> bool {
    std::path::Path::new("/run/netns").join(name).exists()
}

/// Run a privileged `ip`/`nft` invocation, returning `Ok(())` on a zero exit. The
/// command + args are logged at `debug`; output is captured (not surfaced) per S77.
#[cfg(target_os = "linux")]
pub(crate) fn run_priv(bin: &str, args: &[&str]) -> Result<(), NetnsError> {
    let out = std::process::Command::new(bin)
        .args(args)
        .output()
        .map_err(|_| NetnsError::ToolUnavailable)?;
    if out.status.success() {
        Ok(())
    } else {
        tracing::debug!(
            target: "chord.supervisor.netns",
            "privileged netns configuration step failed"
        );
        Err(NetnsError::ConfigureFailed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    // ── posture → config mapping (the pure, always-runnable core) ──────────────

    #[test]
    fn serve_denied_maps_to_no_egress_path() {
        let cfg = NetnsConfig::from_posture(&EgressPosture::Denied);
        assert!(!cfg.egress_path, "a Denied (Serve) netns must have NO egress path");
        assert!(cfg.allow_list.is_empty());
    }

    #[test]
    fn pull_allowlist_maps_to_filtered_egress_path() {
        let posture =
            EgressPosture::AllowList(vec!["registry.ollama.ai".into(), "huggingface.co".into()]);
        let cfg = NetnsConfig::from_posture(&posture);
        assert!(cfg.egress_path, "a Pull netns must have an egress path");
        assert_eq!(cfg.allow_list, vec!["registry.ollama.ai", "huggingface.co"]);
    }

    #[test]
    fn empty_allowlist_fails_closed_to_no_egress_even_defensively() {
        // posture_for never produces an empty AllowList, but if one ever reached
        // here it must collapse to no-egress, NEVER an allow-all path.
        let cfg = NetnsConfig::from_posture(&EgressPosture::AllowList(vec![]));
        assert!(!cfg.egress_path, "empty allow-list must fail closed (no egress path)");
        assert!(cfg.allow_list.is_empty());
    }

    #[test]
    fn namespace_name_is_stable_unique_and_non_infra() {
        let a = namespace_name("slot-A");
        let b = namespace_name("slot-B");
        assert_eq!(a, namespace_name("slot-A"), "name must be stable per token");
        assert_ne!(a, b, "different slots get different namespaces");
        assert!(a.starts_with("chord-"), "namespaces are Chord-prefixed for reaping");
        // No infra leakage in the derived name.
        assert!(!a.contains("192.168."));
        assert!(!a.contains('/'));
    }

    // ── teardown idempotency ──────────────────────────────────────────────────

    #[test]
    fn teardown_of_absent_namespace_is_idempotent_ok() {
        // On this unprivileged/CI host nothing was created; teardown of a
        // never-created namespace must be Ok (no leak, no error). This exercises
        // the idempotency contract the SRV-12 clean swap relies on.
        let h = NetnsHandle {
            name: namespace_name("never-created"),
            posture: EgressPosture::Denied,
        };
        assert!(h.teardown().is_ok(), "tearing down an absent namespace must be idempotent");
        // A second teardown is also Ok.
        assert!(h.teardown().is_ok());
    }

    // ── fail-closed: prepare on a host without the capability ──────────────────

    /// NEGATIVE TEST — the load-bearing fail-closed guard (CHRD-77).
    ///
    /// It asserts that when the `CAP_NET_ADMIN` probe reports NO capability,
    /// `prepare` returns `CapabilityUnavailable` and **never** hands back a handle
    /// the launcher could use to spawn a runtime with full host egress.
    ///
    /// **Why the probe AND the create operation are injected.**
    /// Until CHRD-77 this test called the real `prepare()` and asserted an `Err`,
    /// relying on the build host lacking `CAP_NET_ADMIN`. Chord's build/test host
    /// runs as ROOT, so that precondition was false: `prepare()` took the privileged
    /// branch, actually created `/run/netns/chord-<hash>` (and leaked it), returned
    /// `Ok` on a clean host — and on every subsequent run `ip netns add` failed
    /// EEXIST, so the test failed with `ConfigureFailed`. Either way the guard was
    /// NEVER exercised: a genuine fail-open regression would have looked identical
    /// to that red. Injecting the probe makes the decision provable on every host,
    /// with no privileged syscall and no host-state mutation.
    ///
    /// The create operation is injected for a second, separate reason: the guarantee
    /// "nothing was created" is asserted STRUCTURALLY — the namespace-creating
    /// operation is never INVOKED — rather than by comparing `/run/netns/<name>`
    /// before and after. The earlier filesystem comparison was honest but it read
    /// live host state, so its result depended on the host and could be confounded by
    /// an unrelated concurrent run; "the create path was not entered" is the actual
    /// invariant, and it is fully deterministic.
    ///
    /// The real privileged create/configure path stays covered by the `#[ignore]`d
    /// `tests/netns_integration.rs` suite, run under ISO-04 on a privileged host.
    #[cfg(target_os = "linux")]
    #[test]
    fn prepare_fails_closed_without_capability_never_returns_unisolated_handle() {
        use std::cell::Cell;

        let probe_calls = Cell::new(0u32);
        let create_calls = Cell::new(0u32);
        let slot = "chrd77-failclosed-slot";
        let res = prepare_with_probe(
            slot,
            &EgressPosture::Denied,
            || {
                probe_calls.set(probe_calls.get() + 1);
                false
            },
            |_name, _cfg| {
                // Deliberately reports SUCCESS: if the guard ever fails open, the
                // call reaches here and `prepare` goes on to return Ok(handle) —
                // so the fail-open regression shows up as a handle escaping, not
                // as an incidental error from a stubbed-out failure.
                create_calls.set(create_calls.get() + 1);
                Ok(())
            },
        );

        // CONDITION-DETECTION SELF-CHECK: the guard must actually CONSULT the
        // capability probe. If a future refactor stops calling it, this test would
        // otherwise keep passing for the wrong reason.
        assert_eq!(
            probe_calls.get(),
            1,
            "the capability probe MUST be consulted exactly once by prepare — a guard \
             that never asks is not a guard"
        );

        assert!(
            res.is_err(),
            "prepare must FAIL CLOSED without CAP_NET_ADMIN, never return a handle"
        );
        let err = res.unwrap_err();
        assert!(
            matches!(err, NetnsError::CapabilityUnavailable),
            "fail-closed error must be the capability category, got {err:?}"
        );

        // And NOTHING was created — asserted structurally: the namespace-creating
        // operation was never entered. This is the invariant itself, not a
        // filesystem proxy for it, and it holds identically on every host.
        assert_eq!(
            create_calls.get(),
            0,
            "a fail-closed prepare must never enter the namespace-creating path"
        );

        // And the error string carries no infra.
        let s = err.to_string();
        assert!(!s.contains("192.168.") && !s.contains("/run/netns"));
    }

    /// CHRD-97 — the TEST-SUITE SAFETY invariant itself.
    ///
    /// Every unit test that reaches `prepare` (directly, or via `decide_isolation`
    /// from `launch_isolation` / `serving::launcher`) must be structurally incapable
    /// of creating a namespace on the host. `prepare`'s `cfg(test)` arm makes the
    /// create operation an `unreachable!()`, so if the fail-closed guard were ever
    /// reordered this test PANICS rather than quietly creating `/run/netns/<name>`.
    ///
    /// This is asserted on the real `prepare` on purpose: injecting a fake here
    /// would test the fake. The point is that the entry point production and the
    /// other test modules call is inert in a unit-test build.
    #[cfg(target_os = "linux")]
    #[test]
    fn prepare_cannot_create_a_namespace_in_a_unit_test_build_even_as_root() {
        let slot = "chrd97-no-host-mutation-slot";
        let name = namespace_name(slot);
        let res = prepare(slot, &EgressPosture::Denied);
        assert!(
            matches!(res, Err(NetnsError::CapabilityUnavailable)),
            "in a unit-test build prepare must fail closed at the capability gate \
             regardless of the host's real privilege, got {res:?}"
        );
        // Belt-and-braces on the observable side effect: nothing landed on the host.
        // (The `unreachable!()` create seam is the primary, deterministic guarantee;
        // this only confirms the seam is wired to the path that would have created.)
        assert!(
            !named_netns_exists(&name),
            "a unit test must never leave a network namespace on the build host"
        );
    }

    /// Non-Linux counterpart: netns is a Linux primitive, so `prepare` must return
    /// a clear `Unsupported` error rather than a silent no-op that looks isolated.
    #[cfg(not(target_os = "linux"))]
    #[test]
    fn prepare_fails_closed_without_capability_never_returns_unisolated_handle() {
        let res = prepare("chrd77-failclosed-slot", &EgressPosture::Denied);
        let err = res.expect_err("prepare must FAIL CLOSED on a non-Linux host");
        assert!(matches!(err, NetnsError::Unsupported), "got {err:?}");
    }

    /// CHRD-77 condition-detection guard for the *real* probe.
    ///
    /// `has_net_admin()` is what production feeds into the fail-closed decision, so
    /// a probe stuck at a constant is a silent failure mode. This cross-checks it
    /// against an INDEPENDENT observation — `CapEff` in `/proc/self/status`, bit 12
    /// (`CAP_NET_ADMIN`) — which does not go through `fork`/`unshare` at all.
    ///
    /// The check is deliberately one-directional: *having* the capability
    /// definitively implies `unshare(CLONE_NEWNET)` works, so the probe MUST report
    /// true. The converse is not asserted because an unprivileged user namespace can
    /// also permit `CLONE_NEWNET`. When we cannot make a definitive statement the
    /// test prints a greppable marker rather than passing silently.
    #[cfg(target_os = "linux")]
    #[test]
    fn capability_probe_agrees_with_the_kernels_effective_capability_set() {
        const CAP_NET_ADMIN_BIT: u64 = 12;

        let status = std::fs::read_to_string("/proc/self/status")
            .expect("/proc/self/status must be readable on Linux");
        let cap_eff = status
            .lines()
            .find_map(|l| l.strip_prefix("CapEff:"))
            .and_then(|v| u64::from_str_radix(v.trim(), 16).ok())
            .expect("CapEff must be present and hex-parseable in /proc/self/status");

        if cap_eff & (1 << CAP_NET_ADMIN_BIT) != 0 {
            eprintln!("CHRD77-NETNS-CAPABILITY-DETECTED=present");
            assert!(
                has_net_admin(),
                "this process holds CAP_NET_ADMIN (CapEff={cap_eff:#x}) but the probe \
                 reported false — the probe is broken and prepare() would refuse every \
                 launch"
            );
        } else {
            // Cannot prove the negative direction here (user namespaces), so say so
            // loudly instead of pretending this run verified the probe.
            eprintln!(
                "CHRD77-NETNS-CAPABILITY-DETECTED=absent (probe reported {}); the \
                 probe-vs-kernel agreement check is NOT verified on this host",
                has_net_admin()
            );
        }
    }

    // ── command wrapping (pure data transform) ─────────────────────────────────

    #[test]
    fn wrap_command_prefixes_ip_netns_exec() {
        let h = NetnsHandle {
            name: "chord-deadbeef".into(),
            posture: EgressPosture::Denied,
        };
        let (prog, args) = h.wrap_command(
            "/usr/sbin/ip",
            "llama-server",
            &["--model".to_string(), "/w/m.gguf".to_string()],
        );
        assert_eq!(prog, "/usr/sbin/ip");
        assert_eq!(
            args,
            vec![
                "netns",
                "exec",
                "chord-deadbeef",
                "llama-server",
                "--model",
                "/w/m.gguf"
            ]
        );
    }

    // ── override / enable flags (off-by-default loudness) ──────────────────────

    // CHRD-94: `#[serial]` is required, not decorative. `unisolated_override()`
    // and `isolation_enabled()` read `CHORD_ALLOW_UNISOLATED` /
    // `CHORD_NETNS_ISOLATION` straight out of the PROCESS environment — there is
    // no injectable seam — so these two tests must mutate process-global state to
    // exercise them at all. `supervisor::launch_isolation::tests` and
    // `serving::launcher::tests` drive `decide_isolation` through the same two
    // variables and are already `#[serial]`; without this attribute these tests
    // sit outside that group and their `remove_var` at the end of the body can
    // land between another test's `set_var` and its `decide_isolation()` call,
    // flipping that test's outcome. Keep the group UNNAMED so it is the same lock
    // those tests use.
    #[test]
    #[serial]
    #[cfg_attr(miri, ignore)]
    fn unisolated_override_is_off_unless_exactly_1() {
        // Default off.
        std::env::remove_var("CHORD_ALLOW_UNISOLATED");
        assert!(!unisolated_override());
        // Any value other than exactly "1" stays off.
        std::env::set_var("CHORD_ALLOW_UNISOLATED", "true");
        assert!(!unisolated_override(), "only exact '1' may enable the override");
        std::env::set_var("CHORD_ALLOW_UNISOLATED", "1");
        assert!(unisolated_override());
        std::env::remove_var("CHORD_ALLOW_UNISOLATED");
    }

    // CHRD-94: `#[serial]` for the same reason as the test above — process-global
    // `CHORD_NETNS_ISOLATION`, shared with the `#[serial]` isolation-decision tests.
    #[test]
    #[serial]
    #[cfg_attr(miri, ignore)]
    fn isolation_is_default_on_disabled_only_by_explicit_0() {
        std::env::remove_var("CHORD_NETNS_ISOLATION");
        assert!(isolation_enabled(), "isolation is default-ON");
        std::env::set_var("CHORD_NETNS_ISOLATION", "0");
        assert!(!isolation_enabled(), "explicit 0 disables isolation");
        std::env::set_var("CHORD_NETNS_ISOLATION", "1");
        assert!(isolation_enabled());
        std::env::remove_var("CHORD_NETNS_ISOLATION");
    }
}
