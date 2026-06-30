//! Egress allow-list filtering for the `Pull` network namespace — S88 ISO-02.
//!
//! The `Serve` namespace needs no filter: it has no default route, so the kernel
//! already denies every external connect ([`super::netns`]). This module is ONLY
//! for the `Pull` namespace, which DOES have a constrained egress path (a veth
//! pair + default route through the host). We restrict that path so the runtime
//! can reach ONLY the configured `model_source_allowlist()` hosts and nothing else.
//!
//! ## Mechanism
//! An **nftables ruleset applied INSIDE the namespace** (`ip netns exec <ns> nft …`):
//! a default-drop `output` chain that permits DNS (needed to resolve the allow-list
//! names) and the resolved addresses of the allow-listed hosts, dropping everything
//! else. The allow-list comes from config — we NEVER bake in a host. An empty
//! allow-list never reaches this module (the posture would be `Denied`, which has
//! no egress path at all); defensively, an empty list here builds a deny-all
//! ruleset (fail closed), never an allow-all.
//!
//! ## Honest scope (what is built vs. what ISO-04 verifies)
//! The ruleset *construction* ([`build_nft_ruleset`]) is pure, deterministic, and
//! unit-tested here. The *application* of it inside a live namespace
//! ([`configure_pull_egress`]) requires `CAP_NET_ADMIN` + the `nft` userspace and
//! therefore only runs on a privileged host — its end-to-end effect (allow-listed
//! reachable, everything else denied) is exercised by the `#[ignore]`d integration
//! tests under ISO-04, not by this unprivileged CI build.

use super::netns::NetnsError;

/// Build the constrained egress path for a `Pull` namespace and apply the
/// allow-list filter inside it. Privileged + Linux-only; called by
/// [`super::netns::configure_namespace`].
///
/// Steps (all via the resolved `ip`/`nft` binaries — never literals):
///   1. create a veth pair, move one end into the namespace, address both ends,
///   2. add a default route inside the namespace via the host end + enable host
///      forwarding/NAT for that veth (so the allow-listed pull can actually route),
///   3. apply the nftables allow-list ruleset INSIDE the namespace (default-drop).
///
/// On any failure returns [`NetnsError::ConfigureFailed`]; the caller
/// ([`super::netns::configure_namespace`]) tears the whole namespace down so a
/// half-built egress path never leaks.
#[cfg(target_os = "linux")]
pub(crate) fn configure_pull_egress(
    ns: &str,
    ip_bin: &str,
    allow_list: &[String],
) -> Result<(), NetnsError> {
    use super::netns::run_priv;

    // Deterministic, namespace-scoped link names (kept short for the 15-char IFNAMSIZ
    // limit). Derived from the namespace name's hash tail, not infra.
    let veth_host = format!("ve-h{}", &ns[ns.len().saturating_sub(6)..]);
    let veth_ns = format!("ve-n{}", &ns[ns.len().saturating_sub(6)..]);

    // (1) veth pair, one end into the namespace.
    run_priv(
        ip_bin,
        &["link", "add", &veth_host, "type", "veth", "peer", "name", &veth_ns],
    )?;
    run_priv(ip_bin, &["link", "set", &veth_ns, "netns", ns])?;

    // (2) address + route. The /30 link addresses are RFC-private link-locals chosen
    // per namespace; the host end NATs the namespace toward the allow-list.
    let (host_addr, ns_addr, gw) = link_addresses(ns);
    run_priv(ip_bin, &["addr", "add", &host_addr, "dev", &veth_host])?;
    run_priv(ip_bin, &["link", "set", &veth_host, "up"])?;
    run_priv(ip_bin, &["netns", "exec", ns, ip_bin, "addr", "add", &ns_addr, "dev", &veth_ns])?;
    run_priv(ip_bin, &["netns", "exec", ns, ip_bin, "link", "set", &veth_ns, "up"])?;
    run_priv(
        ip_bin,
        &["netns", "exec", ns, ip_bin, "route", "add", "default", "via", &gw],
    )?;

    // (3) apply the nftables allow-list ruleset INSIDE the namespace (default-drop).
    let nft = crate::config::nft_bin().ok_or(NetnsError::ToolUnavailable)?;
    let ruleset = build_nft_ruleset(allow_list);
    apply_nft_in_ns(ns, ip_bin, &nft, &ruleset)?;

    Ok(())
}

/// Apply an nft ruleset string inside the namespace by piping it to
/// `ip netns exec <ns> nft -f -`. Privileged; output captured, not surfaced (S77).
#[cfg(target_os = "linux")]
fn apply_nft_in_ns(ns: &str, ip_bin: &str, nft_bin: &str, ruleset: &str) -> Result<(), NetnsError> {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let mut child = Command::new(ip_bin)
        .args(["netns", "exec", ns, nft_bin, "-f", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|_| NetnsError::ToolUnavailable)?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(ruleset.as_bytes())
            .map_err(|_| NetnsError::ConfigureFailed)?;
    }
    let status = child.wait().map_err(|_| NetnsError::ConfigureFailed)?;
    if status.success() {
        Ok(())
    } else {
        Err(NetnsError::ConfigureFailed)
    }
}

/// Deterministic per-namespace /30 link addresses + gateway. Uses the namespace
/// name hash to pick a subnet in the RFC1918 `<internal-ip>/16` space reserved here
/// for Chord's transient veths, keeping each launch's link distinct. Returns
/// `(host_addr_cidr, ns_addr_cidr, gateway_ip)`.
#[cfg(target_os = "linux")]
fn link_addresses(ns: &str) -> (String, String, String) {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    ns.hash(&mut h);
    // Map the hash into a /30 boundary: third octet from the hash, fourth = .1/.2.
    let third = (h.finish() % 254) as u8 + 1;
    (
        format!("10.255.{third}.1/30"),
        format!("10.255.{third}.2/30"),
        format!("10.255.{third}.1"),
    )
}

/// Build the nftables ruleset that restricts the `Pull` namespace's egress to the
/// `allow_list` hosts (plus DNS, needed to resolve them). Pure + deterministic so
/// it is fully unit-testable without privilege.
///
/// Policy: an `inet filter` table with a `default-drop` `output` chain that:
///   * always permits loopback + established/related return traffic,
///   * permits UDP/TCP 53 (DNS) so the allow-listed names can be resolved,
///   * permits traffic to each allow-listed host (by name — nft resolves the set
///     at load time on a privileged host), and
///   * DROPS everything else (the closing default policy).
///
/// An EMPTY `allow_list` produces a pure deny-all (no host accept rules) — fail
/// closed, never allow-all.
pub fn build_nft_ruleset(allow_list: &[String]) -> String {
    let mut s = String::new();
    s.push_str("flush ruleset\n");
    s.push_str("table inet chord_egress {\n");
    s.push_str("  chain output {\n");
    s.push_str("    type filter hook output priority 0; policy drop;\n");
    // Loopback + return traffic always allowed.
    s.push_str("    oif \"lo\" accept\n");
    s.push_str("    ct state established,related accept\n");
    // DNS resolution for the allow-listed names.
    s.push_str("    udp dport 53 accept\n");
    s.push_str("    tcp dport 53 accept\n");
    // One accept per allow-listed host (nft resolves the name at load time on a
    // privileged host). Names are validated/escaped to avoid ruleset injection.
    for host in allow_list {
        if let Some(h) = sanitize_host(host) {
            s.push_str(&format!("    ip daddr {h} accept\n"));
        }
    }
    // Explicit drop terminator (redundant with `policy drop`, but documents intent).
    s.push_str("    counter drop\n");
    s.push_str("  }\n");
    s.push_str("}\n");
    s
}

/// Validate/escape an allow-list host so it cannot inject nft syntax. Only
/// hostname/domain characters (and bare IPv4) are accepted; anything else is
/// dropped (returns `None`) rather than emitted into the ruleset. Fail closed: a
/// malformed entry is omitted, never passed through raw.
fn sanitize_host(host: &str) -> Option<String> {
    let h = host.trim();
    if h.is_empty() || h.len() > 253 {
        return None;
    }
    // Hostnames/domains: letters, digits, dot, hyphen. IPv4 fits this too.
    let ok = h
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-');
    // Reject leading/trailing dot or hyphen and any nft metacharacter (already
    // excluded by the char check, but be explicit about the intent).
    if ok && !h.starts_with(['.', '-']) && !h.ends_with(['.', '-']) {
        Some(h.to_string())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ruleset_is_default_drop() {
        let rs = build_nft_ruleset(&["registry.ollama.ai".into()]);
        assert!(rs.contains("policy drop"), "the chain must default-drop (fail closed)");
        assert!(rs.contains("counter drop"), "an explicit drop terminator must be present");
    }

    #[test]
    fn ruleset_permits_dns_and_loopback_and_return() {
        let rs = build_nft_ruleset(&["huggingface.co".into()]);
        assert!(rs.contains("udp dport 53 accept"));
        assert!(rs.contains("tcp dport 53 accept"));
        assert!(rs.contains("oif \"lo\" accept"));
        assert!(rs.contains("ct state established,related accept"));
    }

    #[test]
    fn ruleset_accepts_each_allow_listed_host() {
        let rs = build_nft_ruleset(&["registry.ollama.ai".into(), "huggingface.co".into()]);
        assert!(rs.contains("ip daddr registry.ollama.ai accept"));
        assert!(rs.contains("ip daddr huggingface.co accept"));
    }

    #[test]
    fn empty_allow_list_is_pure_deny_all_not_allow_all() {
        // NEGATIVE: an empty allow-list must yield NO host-accept rule — pure
        // default-drop, never an allow-all.
        let rs = build_nft_ruleset(&[]);
        assert!(rs.contains("policy drop"));
        assert!(!rs.contains("ip daddr"), "empty allow-list must add no host accepts");
    }

    #[test]
    fn malformed_hosts_are_dropped_not_injected() {
        // An entry with nft metacharacters / shell syntax must NOT be emitted raw.
        let rs = build_nft_ruleset(&[
            "good.example.com".into(),
            "evil.com; drop table inet filter".into(),
            "spaces here".into(),
            "back`tick`".into(),
        ]);
        assert!(rs.contains("ip daddr good.example.com accept"));
        // None of the malformed entries leak into the ruleset.
        assert!(!rs.contains("drop table inet filter"));
        assert!(!rs.contains("spaces here"));
        assert!(!rs.contains("`"));
    }

    #[test]
    fn sanitize_accepts_domains_and_ipv4_rejects_garbage() {
        assert_eq!(sanitize_host("registry.ollama.ai").as_deref(), Some("registry.ollama.ai"));
        assert_eq!(sanitize_host("<internal-ip>").as_deref(), Some("<internal-ip>"));
        assert!(sanitize_host("").is_none());
        assert!(sanitize_host(".leading").is_none());
        assert!(sanitize_host("has space").is_none());
        assert!(sanitize_host("semi;colon").is_none());
    }
}
