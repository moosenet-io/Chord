//! CHRD-100: the **logical route catalog** — what a caller may target, as
//! opposed to what models happen to be on disk.
//!
//! ## Why this exists
//!
//! Chord already publishes a model INVENTORY (`GET /api/models`: names, tiers,
//! sizes) and a health COUNT (`GET /health`). Neither answers the question a
//! client-side model picker actually has, which is *"which named things may I
//! send a conversation to, and what is true about each of them right now?"*.
//! The alias table that answers it was pure process state with no read surface
//! at all, so Harmony's SCOUT console (spec S132, SCOUT-02) correctly returned
//! an honest `503 unsupported` rather than inventing a list.
//!
//! A route catalog is deliberately NOT the model list:
//!
//! * A route is a **stable name for a purpose** (`lumina-fast`). Its target is
//!   runtime-mutable — the assistant-fit updater repoints the lumina tiers with
//!   no restart (see [`super::lumina_alias`]) — so a caller that pinned a model
//!   name would be pinned to a decision Chord makes and re-makes.
//! * The catalog is the layer where **Chord answers "local or cloud"**, because
//!   Chord is the only process that knows which backend a route resolves to. A
//!   consumer that guessed this from a model name would be guessing; a
//!   consumer that was TOLD it by Chord is not.
//!
//! ## The invariant this module keeps: no model or provider name leaves here
//!
//! A [`RouteView`] carries no target model, backend name, backend URL, engine
//! tier or provider. That is a property of the STRUCT, not of a filter applied
//! to it: the resolution inputs ([`RouteFacts`]) are reduced to booleans and
//! enums *before* a view is built, so there is no field a name could travel in.
//! In particular `label` never falls back to the route id — a route Chord has
//! no purpose-label for publishes no label at all, and the consumer decides
//! what to do about that. An id rendered as a purpose is exactly how a model
//! reference ends up on a screen.
//!
//! ## `locality` is derived, never declared and never inferred from a name
//!
//! [`Locality`] comes from ONE place: the [`BackendKind`] of the backend the
//! route currently resolves to. A remote, bearer-authenticated backend
//! ([`BackendKind::OpenRouter`]) is `cloud`; every local serving process
//! (Ollama, `llama-server`, a managed daemon) is `local`. It is not an operator
//! declaration, because an operator's declaration can be wrong about where
//! traffic actually goes, and it is not pattern-matched off a model name,
//! because that is a guess dressed as a fact.
//!
//! The corollary is stated rather than hidden: a route whose target cannot be
//! resolved has **no** locality, and this module publishes none. An absent
//! locality is honest; a defaulted one would be a fabrication about where the
//! user's tokens go.
//!
//! ## `available` distinguishes three different facts
//!
//! 1. **No such route** — the id is not in the catalog at all. That is the
//!    absence of an entry (and a `404` on the single-route path), never an
//!    entry with `available: false`.
//! 2. **The route exists and its backend is reachable** — `available: true`.
//! 3. **The route exists and something is wrong with it** — `available: false`
//!    plus an [`UnavailableReason`], which is a **closed enumeration**, never
//!    free text. The consumer writes the human sentence; upstream prose that
//!    could name a model or a host never reaches a browser. A code is a
//!    contract; a sentence is a leak waiting for a bad day.
//!
//! An on-demand backend that is not currently running is NOT unavailable.
//! Chord lazy-starts those on the first request and idle-stops them again
//! (`lemonade-coder`, `llama-gpu`, `vulkan`), so reporting "down" for a stopped
//! one would report Chord's normal resting state as a fault. Only an
//! `always_on` local backend is probed, because only for that one is "not
//! answering" actually wrong.

use std::collections::{BTreeMap, HashMap};

use serde::Serialize;

use crate::models::backends::{Backend, BackendKind};

// ── Vocabulary ───────────────────────────────────────────────────────────────

/// Where a route executes. Derived from the resolved backend's kind — see the
/// module docs on why this is never declared and never inferred from a name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Locality {
    /// Runs on fleet hardware.
    Local,
    /// Leaves the fleet for a remote, bearer-authenticated API.
    Cloud,
}

impl Locality {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Cloud => "cloud",
        }
    }

    /// The one derivation rule. Remote, credentialed backends are cloud;
    /// everything Chord starts and stops itself is local.
    pub fn of_backend_kind(kind: BackendKind) -> Self {
        match kind {
            BackendKind::OpenRouter => Self::Cloud,
            BackendKind::Ollama | BackendKind::LlamaServer | BackendKind::Daemon => Self::Local,
        }
    }
}

/// Why a route that EXISTS cannot currently be used.
///
/// Closed on purpose. Each variant is a distinct thing an operator would fix
/// differently, and a consumer can switch on it exhaustively. Adding free text
/// here would hand a browser an upstream string that can name a model, a host,
/// or a provider — the whole point of the catalog is that it cannot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UnavailableReason {
    /// The route is configured but resolves to no target at all. A naming
    /// entry with nothing behind it — an alias-table configuration fault.
    NoTarget,
    /// The route resolves to a target the model registry has never heard of.
    /// Distinct from [`NoTarget`](Self::NoTarget): the route points somewhere,
    /// and that somewhere is not here.
    UnknownModel,
    /// The target is known but no defined backend can serve it.
    NoBackend,
    /// The route's backend is one that is supposed to be up, and it did not
    /// answer. This is the "backend is down" fault, and it is deliberately NOT
    /// reported for an on-demand backend that is merely idle-stopped.
    Unreachable,
    /// The route is switched off: either declared disabled, or a remote
    /// backend whose credential env var is not provisioned (a route that would
    /// fail on first use is not an available route).
    Disabled,
}

impl UnavailableReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NoTarget => "no_target",
            Self::UnknownModel => "unknown_model",
            Self::NoBackend => "no_backend",
            Self::Unreachable => "unreachable",
            Self::Disabled => "disabled",
        }
    }
}

/// Liveness of the backend a route resolves to, as far as Chord can tell
/// CHEAPLY and truthfully.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendLiveness {
    /// An always-on backend answered a probe.
    Up,
    /// An always-on backend did not answer a probe.
    Down,
    /// Not probed, and correctly so: an on-demand backend Chord starts when a
    /// request arrives. "Currently stopped" is its resting state, not a fault.
    OnDemand,
    /// A remote backend whose API-key env var is not set. Never probed — a
    /// catalog read must not spend a request against a paid API, and a missing
    /// credential is decidable without one.
    CredentialMissing,
}

// ── Declared, non-derivable metadata ─────────────────────────────────────────

/// Is `id` shaped like a ROUTE NAME rather than a model reference?
///
/// Fail-closed, and it exists because of a real gap a reviewer found: the
/// catalog's ids are alias-table KEYS, and nothing stops an operator writing
/// `qwen2.5:7b` as an alias key. Publishing that as a "route id" would put a
/// model reference in a browser through the one string field resolution cannot
/// reach — the struct-shape guarantee says no field carries a name, and this is
/// what keeps `id` honest.
///
/// The rule is deliberately the same one the consumer (Harmony SCOUT-02's
/// `validate_route`) enforces, so the producer and the consumer agree instead of
/// the consumer silently dropping entries the producer thought it had published:
/// lowercase ASCII alphanumerics, `-` and `_`, starting alphanumeric, ≤ 64
/// chars. That excludes the `family:size` and `provider/model` shapes.
///
/// Stated plainly rather than implied: this cannot catch a bare `llama3`, which
/// is shaped exactly like a route name. No check can. What it does is close
/// every shape that is UNAMBIGUOUSLY a model reference, and leave the rest to
/// the operator who wrote the alias table.
pub fn is_route_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
        && id
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
}

/// The closed vocabulary a declared `cost_tier` may use.
///
/// A reviewer was right that `cost_tier` was the one place an engine tier could
/// walk in as free text — "runs on the a100 pool" is a cost tier only if you
/// squint. Everything else this endpoint publishes as a category is an
/// enumeration, so this is too. A declaration outside the vocabulary is dropped
/// with a warning rather than passed through: an unrecognised tier is a config
/// error to fix, not a string to forward.
pub const COST_TIERS: &[&str] = &["free", "metered", "paid"];

/// The parts of a route that Chord genuinely cannot derive and will not invent.
///
/// A route's PURPOSE, its cost tier, its usable context window and whether it
/// supports tool calling are product facts about the route, not observable
/// properties of a serving process — Chord's registry stores none of them. So
/// they are declared, and a route with nothing declared publishes nothing for
/// them rather than a plausible-looking default. `context_window: 0` would read
/// as "this route holds nothing", which is a false claim, not a missing one.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RouteDeclaration {
    /// What the route is FOR, in the operator's words. Never an id.
    ///
    /// This is the ONE deliberately-free string this endpoint publishes, and it
    /// is not policed. A purpose is a sentence a human writes, and there is no
    /// filter that can tell "Quick conversational answers" from a sentence that
    /// names an engine — a blocklist over prose is a blocklist pretending to be
    /// an allowlist (the same reasoning that made the consumer stop scrubbing
    /// upstream reason text). What makes it safe is its PROVENANCE: it comes
    /// from operator-authored local config, never from an upstream response, a
    /// model record, or a backend. Chord will not invent one, and will not
    /// substitute an id for a missing one.
    pub label: Option<String>,
    /// Constrained to [`COST_TIERS`]. See that constant for why.
    pub cost_tier: Option<String>,
    pub context_window: Option<u64>,
    pub supports_tools: Option<bool>,
    /// `false` ⇒ the route is published but reported unavailable/disabled, so a
    /// consumer can see that it exists and is deliberately off. Removing a
    /// route from the alias table is how you make it not exist.
    pub enabled: bool,
}

impl RouteDeclaration {
    fn enabled_default() -> Self {
        RouteDeclaration {
            enabled: true,
            ..Default::default()
        }
    }
}

/// Non-secret env var holding a JSON object of `route id → declaration`.
///
/// Shape (every field optional except that an unknown field is ignored):
/// ```json
/// { "lumina-fast": { "label": "Quick conversational answers",
///                    "cost_tier": "free", "context_window": 32768,
///                    "supports_tools": true, "enabled": true } }
/// ```
pub const ROUTE_CATALOG_ENV: &str = "CHORD_ROUTE_CATALOG";

/// The labels Chord itself is entitled to state, because Chord defines these
/// routes and their purpose is documented in [`super::lumina_alias`]: the two
/// Lumina chat tiers are a fast conversational tier and a deeper-reasoning
/// tier. They name a purpose, never a model — and any of them can be
/// overridden or extended by [`ROUTE_CATALOG_ENV`].
///
/// Nothing else gets a built-in label. A route Chord did not define has a
/// purpose only its operator knows.
const BUILTIN_LABELS: &[(&str, &str)] = &[
    ("lumina-fast", "Quick conversational answers"),
    ("lumina-deep", "Deeper reasoning for harder questions"),
];

/// Parse [`ROUTE_CATALOG_ENV`] over the built-in labels.
///
/// Fails OPEN and per-entry, matching how `CHORD_MODEL_ALIASES` treats a
/// malformed value (see `crate::config`): a bad entry is dropped with a warning
/// and every good one survives, because a JSON typo must not black out a
/// catalog. A wholly-unparseable value leaves the built-ins in place — that is
/// strictly better than publishing an unlabelled catalog.
pub fn declarations_from_env() -> HashMap<String, RouteDeclaration> {
    let raw = std::env::var(ROUTE_CATALOG_ENV).unwrap_or_default();
    parse_declarations(&raw)
}

/// The testable core of [`declarations_from_env`].
pub fn parse_declarations(raw: &str) -> HashMap<String, RouteDeclaration> {
    let mut out: HashMap<String, RouteDeclaration> = BUILTIN_LABELS
        .iter()
        .map(|(id, label)| {
            (
                (*id).to_string(),
                RouteDeclaration {
                    label: Some((*label).to_string()),
                    ..RouteDeclaration::enabled_default()
                },
            )
        })
        .collect();

    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return out;
    }
    let parsed: serde_json::Value = match serde_json::from_str(trimmed) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(
                error = %e,
                "{ROUTE_CATALOG_ENV} is not valid JSON — route declarations fall back to \
                 Chord's built-in labels; declared labels/cost tiers/context windows are NOT \
                 applied for this process"
            );
            return out;
        }
    };
    let Some(obj) = parsed.as_object() else {
        tracing::warn!(
            "{ROUTE_CATALOG_ENV} is not a JSON object of route-id → declaration — route \
             declarations fall back to Chord's built-in labels"
        );
        return out;
    };

    for (id, v) in obj {
        let Some(entry) = v.as_object() else {
            tracing::warn!(route = %id, "{ROUTE_CATALOG_ENV} entry is not an object — dropped");
            continue;
        };
        let str_field = |k: &str| {
            entry
                .get(k)
                .and_then(serde_json::Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
        };
        // A declared entry REPLACES the built-in rather than merging into it:
        // half-overriding would make the effective label depend on which fields
        // an operator happened to write, which is the sort of surprise that
        // gets discovered in production.
        out.insert(
            id.clone(),
            RouteDeclaration {
                label: str_field("label"),
                cost_tier: str_field("cost_tier").and_then(|t| {
                    let t = t.to_ascii_lowercase();
                    if COST_TIERS.contains(&t.as_str()) {
                        Some(t)
                    } else {
                        tracing::warn!(
                            route = %id,
                            "{ROUTE_CATALOG_ENV}: cost_tier is not one of {COST_TIERS:?} — \
                             dropped rather than published as free text"
                        );
                        None
                    }
                }),
                context_window: entry.get("context_window").and_then(serde_json::Value::as_u64),
                supports_tools: entry
                    .get("supports_tools")
                    .and_then(serde_json::Value::as_bool),
                enabled: entry
                    .get("enabled")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(true),
            },
        );
    }
    out
}

// ── Facts in, view out ───────────────────────────────────────────────────────

/// What resolution found for one route. Reduced to booleans and enums *before*
/// a [`RouteView`] is built — see the module docs: this is where model and
/// backend names stop.
#[derive(Debug, Clone, PartialEq)]
pub struct RouteFacts {
    /// The route id (an alias key). The only string that crosses into the view.
    pub id: String,
    /// Did the alias resolve to any target at all?
    pub has_target: bool,
    /// Is that target a model the registry knows?
    pub target_known: bool,
    /// The kind of backend the target resolves to, if one does.
    pub backend_kind: Option<BackendKind>,
    /// Liveness of that backend. `None` iff `backend_kind` is `None`.
    pub liveness: Option<BackendLiveness>,
}

/// One route, exactly as the control API publishes it.
///
/// Every optional field is OMITTED when unknown rather than defaulted. Absent
/// is a different claim from zero/false, and the consumer is entitled to tell
/// them apart.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RouteView {
    pub id: String,
    /// The route's PURPOSE. Absent when undeclared — never the id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Absent when the route resolves to no backend, so locality is genuinely
    /// unknown. Never guessed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub locality: Option<Locality>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_tools: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost_tier: Option<String>,
    pub available: bool,
    /// Present exactly when `available` is false. An unavailable route with no
    /// reason is not a warning, it is a shrug.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unavailable_reason: Option<UnavailableReason>,
}

/// Build the published view of one route from resolved facts plus whatever was
/// declared about it.
///
/// The ordering of the checks is the ordering of the faults' PRECEDENCE, from
/// most fundamental outward: a route that is switched off is reported as
/// switched off even if its backend also happens to be down, because "turn it
/// on" is the action either way and reporting the deeper fault would send an
/// operator chasing a backend they were never using.
pub fn build_view(facts: &RouteFacts, decl: Option<&RouteDeclaration>) -> RouteView {
    // Locality is derived from the backend and nothing else — including on the
    // unavailable paths. A disabled cloud route is still a cloud route, and
    // saying so is what lets a consumer warn about it honestly.
    let locality = facts.backend_kind.map(Locality::of_backend_kind);

    let unavailable_reason = if decl.is_some_and(|d| !d.enabled) {
        Some(UnavailableReason::Disabled)
    } else if !facts.has_target {
        Some(UnavailableReason::NoTarget)
    } else if !facts.target_known {
        Some(UnavailableReason::UnknownModel)
    } else {
        match facts.liveness {
            None => Some(UnavailableReason::NoBackend),
            Some(BackendLiveness::Down) => Some(UnavailableReason::Unreachable),
            Some(BackendLiveness::CredentialMissing) => Some(UnavailableReason::Disabled),
            Some(BackendLiveness::Up) | Some(BackendLiveness::OnDemand) => None,
        }
    };

    RouteView {
        id: facts.id.clone(),
        label: decl.and_then(|d| d.label.clone()),
        locality,
        context_window: decl.and_then(|d| d.context_window),
        supports_tools: decl.and_then(|d| d.supports_tools),
        cost_tier: decl.and_then(|d| d.cost_tier.clone()),
        available: unavailable_reason.is_none(),
        unavailable_reason,
    }
}

/// Classify a backend's liveness given whether an always-on probe succeeded.
///
/// `probe` is consulted ONLY for a backend that is supposed to be up. An
/// on-demand backend is never probed (a stopped one is normal), and a remote
/// backend is decided by whether its credential is provisioned rather than by
/// spending a request against a paid API.
pub fn classify_liveness(
    backend: &Backend,
    credential_present: impl FnOnce(&str) -> bool,
    probe: impl FnOnce(&str) -> bool,
) -> BackendLiveness {
    if backend.kind == BackendKind::OpenRouter {
        let ok = match backend.api_key_env.as_deref() {
            // A remote backend with no credential env var NAMED at all cannot
            // be authenticated, so it cannot be used. Same answer, and it is
            // reached without inventing a variable name to look up.
            None => false,
            Some(env_key) => credential_present(env_key),
        };
        return if ok {
            BackendLiveness::Up
        } else {
            BackendLiveness::CredentialMissing
        };
    }
    if backend.on_demand() {
        return BackendLiveness::OnDemand;
    }
    // Always-on (or an externally-managed daemon): "not answering" is a fault.
    if probe(&backend.url) {
        BackendLiveness::Up
    } else {
        BackendLiveness::Down
    }
}

/// Assemble the whole catalog, sorted by id so the output is stable.
///
/// `BTreeMap` in, sorted `Vec` out: a catalog whose order changes between two
/// identical reads makes a UI reshuffle for no reason and makes a diff of two
/// captures unreadable.
pub fn build_catalog(
    facts: &BTreeMap<String, RouteFacts>,
    decls: &HashMap<String, RouteDeclaration>,
) -> Vec<RouteView> {
    facts
        .values()
        .map(|f| build_view(f, decls.get(&f.id)))
        .collect()
}

// ── Liveness probing ─────────────────────────────────────────────────────────
//
// The only network I/O this module does, and it does as little as it can:
// one request per DISTINCT always-on backend URL, short-timeout, briefly cached.
// A catalog read is a UI refresh — it must never turn into a fan-out of probes
// against every backend on every keystroke, and it must never touch a paid API
// at all (see `classify_liveness`).

use std::collections::BTreeSet;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Non-secret knob: how long a probe result is reused.
pub const LIVENESS_TTL_ENV: &str = "CHORD_ROUTE_LIVENESS_TTL_SECS";
/// Short on purpose. `available` is a liveness claim; a long cache turns "this
/// route is down" into stale reassurance, which is worse than not answering.
pub const DEFAULT_LIVENESS_TTL_SECS: u64 = 10;
/// A backend that has not answered in this long is, for a catalog read's
/// purposes, not answering. The caller is a page waiting to render.
const PROBE_TIMEOUT: Duration = Duration::from_secs(2);

type ProbeCache = Mutex<HashMap<String, (Instant, bool)>>;

fn probe_cache() -> &'static ProbeCache {
    static CACHE: OnceLock<ProbeCache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn liveness_ttl() -> Duration {
    let secs = std::env::var(LIVENESS_TTL_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_LIVENESS_TTL_SECS);
    Duration::from_secs(secs)
}

/// Probe each URL that is not already cached fresh, and return reachability for
/// all of them.
///
/// "Reachable" means the socket answered — ANY HTTP status counts, including a
/// 404. We are asking whether the serving process is there, not whether it
/// likes the path we happened to pick; a backend that 404s its root is up.
pub async fn probe_always_on(
    client: &reqwest::Client,
    urls: &BTreeSet<String>,
) -> HashMap<String, bool> {
    let ttl = liveness_ttl();
    let now = Instant::now();
    let mut out: HashMap<String, bool> = HashMap::new();
    let mut to_probe: Vec<String> = Vec::new();

    // Never hold this lock across an await.
    {
        if let Ok(cache) = probe_cache().lock() {
            for u in urls {
                match cache.get(u) {
                    Some((at, ok)) if now.duration_since(*at) < ttl => {
                        out.insert(u.clone(), *ok);
                    }
                    _ => to_probe.push(u.clone()),
                }
            }
        } else {
            // A poisoned cache must not black out the catalog: probe everything.
            to_probe.extend(urls.iter().cloned());
        }
    }

    let results = futures_util::future::join_all(to_probe.into_iter().map(|u| {
        let client = client.clone();
        async move {
            let ok = client
                .get(&u)
                .timeout(PROBE_TIMEOUT)
                .send()
                .await
                .is_ok();
            (u, ok)
        }
    }))
    .await;

    if let Ok(mut cache) = probe_cache().lock() {
        let at = Instant::now();
        for (u, ok) in &results {
            cache.insert(u.clone(), (at, *ok));
        }
        // The cache is keyed by URL and backends are reconfigured, not created
        // per request, so it cannot grow without bound in practice — but an
        // unbounded map that nobody ever prunes is how "in practice" stops
        // being true. Drop entries no longer worth remembering.
        cache.retain(|_, (a, _)| at.duration_since(*a) < ttl * 6);
    }
    out.extend(results);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backend(name: &str, kind: BackendKind, always_on: bool) -> Backend {
        Backend {
            name: name.to_string(),
            url: format!("http://127.0.0.1/{name}"),
            hardware: crate::models::backends::Hardware::Gpu,
            kind,
            unit: None,
            always_on,
            idle_stop_secs: 0,
            launch: None,
            api_key_env: None,
        }
    }

    fn facts(id: &str) -> RouteFacts {
        RouteFacts {
            id: id.to_string(),
            has_target: true,
            target_known: true,
            backend_kind: Some(BackendKind::Ollama),
            liveness: Some(BackendLiveness::Up),
        }
    }

    // ── locality is derived from the backend, never from anything else ───────

    #[test]
    fn locality_comes_from_the_backend_kind_not_from_a_name() {
        assert_eq!(
            Locality::of_backend_kind(BackendKind::OpenRouter),
            Locality::Cloud
        );
        for kind in [
            BackendKind::Ollama,
            BackendKind::LlamaServer,
            BackendKind::Daemon,
        ] {
            assert_eq!(
                Locality::of_backend_kind(kind),
                Locality::Local,
                "{kind:?} is a serving process Chord runs — it is local"
            );
        }
    }

    #[test]
    fn a_route_with_no_resolvable_backend_publishes_no_locality() {
        let f = RouteFacts {
            backend_kind: None,
            liveness: None,
            ..facts("orphan")
        };
        let v = build_view(&f, None);
        assert_eq!(
            v.locality, None,
            "locality must be ABSENT, not defaulted to local — a default here would be a \
             fabrication about where the user's tokens go"
        );
        assert!(!v.available);
        assert_eq!(v.unavailable_reason, Some(UnavailableReason::NoBackend));
    }

    #[test]
    fn a_disabled_cloud_route_still_reports_that_it_is_cloud() {
        let f = RouteFacts {
            backend_kind: Some(BackendKind::OpenRouter),
            liveness: Some(BackendLiveness::Up),
            ..facts("paid")
        };
        let decl = RouteDeclaration {
            enabled: false,
            ..RouteDeclaration::default()
        };
        let v = build_view(&f, Some(&decl));
        assert_eq!(v.locality, Some(Locality::Cloud));
        assert_eq!(v.unavailable_reason, Some(UnavailableReason::Disabled));
    }

    // ── the three availability facts are three different answers ─────────────

    #[test]
    fn a_reachable_route_is_available_with_no_reason() {
        let v = build_view(&facts("ok"), None);
        assert!(v.available);
        assert_eq!(v.unavailable_reason, None);
    }

    #[test]
    fn a_down_always_on_backend_is_a_fault_not_an_absent_route() {
        let f = RouteFacts {
            liveness: Some(BackendLiveness::Down),
            ..facts("down")
        };
        let v = build_view(&f, None);
        assert!(!v.available);
        assert_eq!(v.unavailable_reason, Some(UnavailableReason::Unreachable));
        // The route is still PRESENT. "Down" and "not a route" must never
        // collapse into the same observation.
        assert_eq!(v.id, "down");
    }

    #[test]
    fn an_idle_on_demand_backend_is_available_because_chord_starts_it() {
        let f = RouteFacts {
            liveness: Some(BackendLiveness::OnDemand),
            ..facts("lazy")
        };
        let v = build_view(&f, None);
        assert!(
            v.available,
            "an idle-stopped on-demand backend is Chord's resting state, not a fault"
        );
    }

    #[test]
    fn each_resolution_failure_gets_its_own_code() {
        let no_target = build_view(
            &RouteFacts {
                has_target: false,
                target_known: false,
                backend_kind: None,
                liveness: None,
                ..facts("empty")
            },
            None,
        );
        assert_eq!(
            no_target.unavailable_reason,
            Some(UnavailableReason::NoTarget)
        );

        let unknown = build_view(
            &RouteFacts {
                target_known: false,
                ..facts("ghost")
            },
            None,
        );
        assert_eq!(
            unknown.unavailable_reason,
            Some(UnavailableReason::UnknownModel),
            "a route pointing at a model the registry never heard of is a DIFFERENT fault \
             from a route pointing at nothing"
        );
    }

    #[test]
    fn disabled_takes_precedence_over_a_backend_fault() {
        let f = RouteFacts {
            liveness: Some(BackendLiveness::Down),
            ..facts("off")
        };
        let decl = RouteDeclaration {
            enabled: false,
            ..RouteDeclaration::default()
        };
        let v = build_view(&f, Some(&decl));
        assert_eq!(v.unavailable_reason, Some(UnavailableReason::Disabled));
    }

    // ── the label is a purpose, and never an id ──────────────────────────────

    #[test]
    fn an_undeclared_route_publishes_no_label_rather_than_its_id() {
        let v = build_view(&facts("some-route"), None);
        assert_eq!(
            v.label, None,
            "falling back to the id is how a model reference reaches a screen as a purpose"
        );
        let json = serde_json::to_value(&v).unwrap();
        assert!(
            json.get("label").is_none(),
            "an unknown label is OMITTED, not null-and-present"
        );
    }

    #[test]
    fn unknown_optional_facts_are_omitted_not_zeroed() {
        let json = serde_json::to_value(build_view(&facts("bare"), None)).unwrap();
        for k in ["context_window", "supports_tools", "cost_tier"] {
            assert!(
                json.get(k).is_none(),
                "{k} must be omitted when unknown — 0/false is a claim, absence is not"
            );
        }
    }

    #[test]
    fn the_view_carries_no_field_a_model_or_backend_name_could_travel_in() {
        let json = serde_json::to_value(build_view(&facts("r"), None)).unwrap();
        let keys: Vec<&str> = json.as_object().unwrap().keys().map(String::as_str).collect();
        for k in &keys {
            assert!(
                matches!(
                    *k,
                    "id" | "label"
                        | "locality"
                        | "context_window"
                        | "supports_tools"
                        | "cost_tier"
                        | "available"
                        | "unavailable_reason"
                ),
                "unexpected field {k} on a published route — the catalog's no-names guarantee \
                 is enforced by the shape of RouteView, so a new field needs a deliberate \
                 review, not a silent pass"
            );
        }
    }

    // ── reason codes are a closed, stable wire vocabulary ────────────────────

    #[test]
    fn reason_codes_serialize_to_their_documented_strings() {
        for (r, s) in [
            (UnavailableReason::NoTarget, "no_target"),
            (UnavailableReason::UnknownModel, "unknown_model"),
            (UnavailableReason::NoBackend, "no_backend"),
            (UnavailableReason::Unreachable, "unreachable"),
            (UnavailableReason::Disabled, "disabled"),
        ] {
            assert_eq!(r.as_str(), s);
            assert_eq!(serde_json::to_value(r).unwrap(), serde_json::json!(s));
        }
        assert_eq!(
            serde_json::to_value(Locality::Local).unwrap(),
            serde_json::json!("local")
        );
        assert_eq!(
            serde_json::to_value(Locality::Cloud).unwrap(),
            serde_json::json!("cloud")
        );
    }

    // ── liveness classification ──────────────────────────────────────────────

    #[test]
    fn an_on_demand_backend_is_never_probed() {
        let b = backend("llama-gpu", BackendKind::LlamaServer, false);
        let live = classify_liveness(
            &b,
            |_| panic!("a local backend must not be asked for a credential"),
            |_| panic!("an on-demand backend must never be probed — being stopped is normal"),
        );
        assert_eq!(live, BackendLiveness::OnDemand);
    }

    #[test]
    fn an_always_on_backend_is_probed_and_a_refusal_is_down() {
        let b = backend("ollama", BackendKind::Ollama, true);
        assert_eq!(
            classify_liveness(&b, |_| unreachable!(), |_| true),
            BackendLiveness::Up
        );
        assert_eq!(
            classify_liveness(&b, |_| unreachable!(), |_| false),
            BackendLiveness::Down
        );
    }

    #[test]
    fn a_remote_backend_is_decided_by_its_credential_and_never_probed() {
        let mut b = backend("openrouter", BackendKind::OpenRouter, true);
        b.api_key_env = Some("SOME_KEY_ENV".to_string());
        assert_eq!(
            classify_liveness(
                &b,
                |k| {
                    assert_eq!(k, "SOME_KEY_ENV");
                    true
                },
                |_| panic!("a catalog read must never spend a request against a paid API")
            ),
            BackendLiveness::Up
        );
        assert_eq!(
            classify_liveness(&b, |_| false, |_| panic!("must not probe")),
            BackendLiveness::CredentialMissing
        );
        // No credential env var named at all: unusable, and decided without
        // inventing a variable name to go looking for.
        b.api_key_env = None;
        assert_eq!(
            classify_liveness(&b, |_| panic!("nothing to look up"), |_| panic!("must not probe")),
            BackendLiveness::CredentialMissing
        );
    }

    // ── declarations ─────────────────────────────────────────────────────────

    #[test]
    fn chords_own_route_purposes_are_labelled_out_of_the_box() {
        let d = parse_declarations("");
        assert_eq!(
            d.get("lumina-fast").and_then(|d| d.label.as_deref()),
            Some("Quick conversational answers")
        );
        assert_eq!(
            d.get("lumina-deep").and_then(|d| d.label.as_deref()),
            Some("Deeper reasoning for harder questions")
        );
        for (_, decl) in d.iter() {
            assert!(decl.enabled);
            // A built-in label states a PURPOSE. It must not smuggle a model
            // family in through the label field.
            let l = decl.label.clone().unwrap_or_default().to_lowercase();
            for banned in ["llama", "qwen", "granite", "gpt", "gemma", "mistral", "ollama"] {
                assert!(!l.contains(banned), "built-in label names an engine: {l}");
            }
        }
    }

    #[test]
    fn a_declaration_overrides_a_builtin_and_adds_new_routes() {
        let d = parse_declarations(
            r#"{"lumina-fast":{"label":"Scoping and short answers","cost_tier":"free",
                "context_window":32768,"supports_tools":true},
                "scout-cloud":{"label":"Frontier review pass","cost_tier":"paid","enabled":false}}"#,
        );
        let fast = d.get("lumina-fast").unwrap();
        assert_eq!(fast.label.as_deref(), Some("Scoping and short answers"));
        assert_eq!(fast.context_window, Some(32768));
        assert_eq!(fast.supports_tools, Some(true));
        assert_eq!(fast.cost_tier.as_deref(), Some("free"));
        assert!(fast.enabled);
        let cloud = d.get("scout-cloud").unwrap();
        assert!(!cloud.enabled);
        // Untouched built-in survives.
        assert!(d.get("lumina-deep").unwrap().label.is_some());
    }

    #[test]
    fn a_malformed_declaration_value_never_blacks_out_the_catalog() {
        // Not JSON at all.
        assert!(parse_declarations("{not json").get("lumina-fast").is_some());
        // JSON, but not an object.
        assert!(parse_declarations("[1,2,3]").get("lumina-fast").is_some());
        // One bad entry, one good one: the good one survives.
        let d = parse_declarations(r#"{"bad": 7, "good": {"label":"Fine"}}"#);
        assert!(d.get("bad").is_none());
        assert_eq!(d.get("good").and_then(|x| x.label.as_deref()), Some("Fine"));
        assert!(d.get("lumina-fast").is_some());
    }

    #[test]
    fn a_blank_declared_label_is_absent_not_blank() {
        let d = parse_declarations(r#"{"r":{"label":"   ","cost_tier":""}}"#);
        let r = d.get("r").unwrap();
        assert_eq!(r.label, None);
        assert_eq!(r.cost_tier, None);
    }

    // ── the catalog as a whole ───────────────────────────────────────────────

    // ── the id shape gate (reviewer finding: `id` was the one unguarded string) ─

    #[test]
    fn a_model_shaped_alias_key_is_not_a_route_id() {
        for bad in [
            "qwen2.5:7b",
            "meta-llama/Llama-3-8B",
            "granite4.1:small",
            "Lumina-Fast",
            "-leading-dash",
            "",
            "has space",
            "trailing:",
        ] {
            assert!(
                !is_route_id(bad),
                "{bad:?} is not a route name and must never be published as a route id"
            );
        }
        for good in ["lumina-fast", "scout_deep", "r2", "a"] {
            assert!(is_route_id(good), "{good:?} is a route name");
        }
        assert!(!is_route_id(&"a".repeat(65)));
        assert!(is_route_id(&"a".repeat(64)));
    }

    #[test]
    fn the_id_rule_is_the_same_one_the_consumer_enforces() {
        // Harmony SCOUT-02 `validate_route`: non-empty, <= 64, first char
        // lowercase-alnum, body lowercase-alnum/-/_. If these two rules drift,
        // the producer publishes routes the consumer silently drops.
        fn consumer_rule(r: &str) -> bool {
            !r.is_empty()
                && r.len() <= 64
                && r.chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
                && r.chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
        }
        for s in [
            "lumina-fast", "qwen2.5:7b", "A", "", "x_y-1", "a/b", "a".repeat(65).as_str(),
        ] {
            assert_eq!(is_route_id(s), consumer_rule(s), "producer/consumer disagree on {s:?}");
        }
    }

    // ── cost_tier is a closed vocabulary, not free text ──────────────────────

    #[test]
    fn an_unrecognised_cost_tier_is_dropped_rather_than_forwarded() {
        let d = parse_declarations(
            r#"{"a":{"cost_tier":"free"},"b":{"cost_tier":"PAID"},
                "c":{"cost_tier":"runs on the big gpu pool"}}"#,
        );
        assert_eq!(d["a"].cost_tier.as_deref(), Some("free"));
        assert_eq!(
            d["b"].cost_tier.as_deref(),
            Some("paid"),
            "the vocabulary is case-insensitive on input and normalised on output"
        );
        assert_eq!(
            d["c"].cost_tier, None,
            "an out-of-vocabulary tier is a config error to fix, not a string to forward —              this was the one field an engine tier could have walked in through"
        );
    }

    #[test]
    fn the_catalog_is_sorted_by_id_so_two_identical_reads_match() {
        let mut f = BTreeMap::new();
        for id in ["zulu", "alpha", "mike"] {
            f.insert(id.to_string(), facts(id));
        }
        let out = build_catalog(&f, &HashMap::new());
        let ids: Vec<&str> = out.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(ids, vec!["alpha", "mike", "zulu"]);
    }
}
