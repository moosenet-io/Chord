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
/// "linux")]` AND when the capability probe passes; the non-Linux / unprivileged
/// builds (including this CI) take the fail-closed branch. The integration tests
/// that exercise the real namespace are `#[ignore]`d so they only run on a
/// privileged host under ISO-04.
pub fn prepare(slot_token: &str, posture: &EgressPosture) -> Result<NetnsHandle, NetnsError> {
    let cfg = NetnsConfig::from_posture(posture);
    let name = namespace_name(slot_token);

    #[cfg(target_os = "linux")]
    {
        if !has_net_admin() {
            return Err(NetnsError::CapabilityUnavailable);
        }
        configure_namespace(&name, &cfg)?;
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

    #[cfg(not(target_os = "linux"))]
    {
        let _ = (&cfg, &name);
        Err(NetnsError::Unsupported)
    }
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

    #[test]
    fn prepare_fails_closed_without_capability_never_returns_unisolated_handle() {
        // NEGATIVE TEST. On this unprivileged build prepare() must return an Err
        // (CapabilityUnavailable on Linux, Unsupported elsewhere) — it must NEVER
        // return Ok with a handle that would let the launcher spawn unisolated.
        let res = prepare("test-slot", &EgressPosture::Denied);
        assert!(
            res.is_err(),
            "prepare must FAIL CLOSED on a host without CAP_NET_ADMIN, never return a handle"
        );
        let err = res.unwrap_err();
        assert!(
            matches!(err, NetnsError::CapabilityUnavailable | NetnsError::Unsupported),
            "fail-closed error must be the capability/unsupported category, got {err:?}"
        );
        // And the error string carries no infra.
        let s = err.to_string();
        assert!(!s.contains("192.168.") && !s.contains("/run/netns"));
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

    #[test]
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

    #[test]
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
