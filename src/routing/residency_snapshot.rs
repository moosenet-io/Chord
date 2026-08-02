//! CHRD-95: the residency SNAPSHOT FILE — Chord's only outward-facing record of
//! what is actually resident on the GPU.
//!
//! ## Why this module exists
//! Terminus's `serving_residency_status` tool reads a JSON snapshot from
//! `CHORD_RESIDENCY_STATE_PATH`. Chord never wrote one: the writer lived on
//! `serving::residency::VramResidencyManager`, which has never been constructed
//! anywhere in this repo's history (a bulk-vendored SRV-13 import). The tool
//! therefore reported a plausible falsehood — "IDLE, nothing resident" — on a box
//! that was actively serving models, on every single call.
//!
//! ## Why the writer moved HERE and not onto that manager
//! CHRD-PIN-01 (`f22251d`) made [`crate::routing::resident_set::ResidentSet`] the
//! **single owner of VRAM residency**, precisely to stop two mechanisms fighting
//! over one GPU. Constructing the dead manager to get its writer back would
//! re-create exactly that bug. So the writer is SALVAGED — the atomic
//! temp-in-same-dir + fsync + rename sequence below is lifted from
//! `serving::residency::write_state_file`, which was already correct — and driven
//! by the live owner instead.
//!
//! ## THE LOAD-BEARING SAFETY PROPERTY
//! **A failed snapshot write MUST NEVER break a warm or a release.** This file is
//! observability, and observability that can take down the thing it observes is
//! strictly worse than no observability: an unwritable path, a full disk, or a
//! read-only mount would turn a diagnostic nicety into a GPU-residency outage.
//! Every entry point here is therefore infallible to the caller — errors are
//! logged and swallowed, and no IO error can propagate into the serving path.
//! [`SnapshotWriter::publish`] returns `()` for that reason; it is not an
//! oversight and must not "helpfully" be given a `Result`.
//!
//! ## What is NOT in the file
//! Four fields of the old SRV-13 shape — `mode`, `assumed_memory_model`,
//! `gpu_ceiling_gb`, `cpu_ceiling_gb` — have **no live source** in Chord today:
//! nothing tracks an operating mode or a ceiling since CHRD-PIN-01. They are
//! emitted as explicit JSON `null` via [`Unsourced`], never as a zero or a
//! default. A fabricated `gpu_ceiling_gb: 0.0` reads as a MEASUREMENT, and a
//! defaulted `mode` reads as a confident claim about something nothing tracks;
//! both are worse than absence, because a reader cannot tell them from truth.
//! [`Unsourced`] is a type with no constructor arguments, so there is nothing to
//! fabricate them WITH.

use std::sync::Mutex;

use serde::{Serialize, Serializer};
use tracing::{debug, warn};

/// A field of the snapshot contract that has **no live source in Chord**.
///
/// Always serializes as JSON `null`. It carries no value and cannot be given one:
/// that is the point. See the module docs — a zeroed ceiling or a defaulted mode
/// is indistinguishable from a measured one to whoever reads this file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Unsourced;

impl Serialize for Unsourced {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_none()
    }
}

/// One resident model, as the reader sees it.
///
/// Field names match what Terminus's `serving_residency_status` deserializes
/// (`role`, `model_id`, `vram_gb`), so this is a wire contract — rename with care.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ResidentEntry {
    /// The resident-set ROLE id (`personality` | `router` | `embedding`).
    pub role: &'static str,
    /// The concrete model the role's alias currently resolves to.
    pub model_id: String,
    /// **An ESTIMATE, and a LOW one.** This is the registry's ON-DISK model size,
    /// not a measured VRAM footprint: it is what the resident set budgets
    /// against, and it is the only per-role size the live owner has under its own
    /// lock. Real resident VRAM is larger (KV cache, context, runtime overhead),
    /// so this figure reads LOW against `nvidia-smi` / Ollama's `size_vram`.
    /// Deliberately not sourced from Ollama `/api/ps` here: that would mean an
    /// extra network round trip inside the warm path for a diagnostic field, and
    /// the safety property above says this file must never be able to slow or
    /// break a warm. The top-level `vram_gb_source` marker says the same thing in
    /// the file itself, where a reader will actually see it.
    ///
    /// Omitted entirely when the registry has no size (never emitted as `0.0` —
    /// the reader defaults absent numerics to zero, but a zero we WROTE would be
    /// a claim).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vram_gb: Option<f64>,
}

/// The snapshot document itself.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ResidencySnapshot {
    /// Every role slot currently HELD. Empty means genuinely nothing resident —
    /// which is now distinguishable from "no file", because a file exists.
    pub residents: Vec<ResidentEntry>,
    /// Free VRAM in GB at the moment of the state change. Omitted when the
    /// counter is unreadable (the reader defaults it to zero; we do not assert
    /// zero).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub free_vram_gb: Option<f64>,
    /// The model held by the `personality` role — post-CHRD-PIN-01 that IS the
    /// pinned chat model. `null` when the role is not currently held.
    pub pinned_chat_model: Option<String>,
    /// Says, in the file, what [`ResidentEntry::vram_gb`] actually measures.
    pub vram_gb_source: &'static str,
    /// RFC 3339 UTC. Without this a stale file left behind by a dead Chord reads
    /// as live state; a reader can now age it out.
    pub written_at: String,

    // ── No live source: see [`Unsourced`] and the module docs ───────────────
    pub mode: Unsourced,
    pub assumed_memory_model: Unsourced,
    pub gpu_ceiling_gb: Unsourced,
    pub cpu_ceiling_gb: Unsourced,
}

/// What [`ResidentEntry::vram_gb`] is, stated in the file.
const VRAM_GB_SOURCE: &str = "registry-on-disk-model-size (estimate; reads LOW vs resident VRAM)";

impl ResidencySnapshot {
    /// Build a snapshot from the live owner's state. The four unsourced fields
    /// are not parameters — there is deliberately no way to supply them.
    pub fn new(
        residents: Vec<ResidentEntry>,
        free_vram_gb: Option<f64>,
        pinned_chat_model: Option<String>,
    ) -> Self {
        Self {
            residents,
            free_vram_gb,
            pinned_chat_model,
            vram_gb_source: VRAM_GB_SOURCE,
            written_at: chrono::Utc::now().to_rfc3339(),
            mode: Unsourced,
            assumed_memory_model: Unsourced,
            gpu_ceiling_gb: Unsourced,
            cpu_ceiling_gb: Unsourced,
        }
    }
}

/// Publishes the snapshot on a STATE CHANGE, never on every reconcile tick.
///
/// The resident set re-asserts on a background loop; writing the same bytes every
/// tick is pointless disk churn on a shared host. So the writer remembers the
/// last body it successfully wrote (everything except `written_at`, which changes
/// every call by construction) and skips an identical one. A failed write is NOT
/// remembered, so the next state change retries rather than latching a stale file
/// as current.
#[derive(Debug, Default)]
pub struct SnapshotWriter {
    /// Last body successfully written, `written_at` excluded.
    last: Mutex<Option<serde_json::Value>>,
}

impl SnapshotWriter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Write the snapshot if the state actually changed.
    ///
    /// **Infallible by contract** — see the module docs. Every failure mode
    /// (no path configured, serialization, IO) logs and returns. Nothing here can
    /// propagate into a warm or a release.
    ///
    /// `reason` is only for the log line.
    pub fn publish(&self, snapshot: &ResidencySnapshot, reason: &str) {
        // No path configured ⇒ snapshot persistence is off. Not an error: the
        // path is never guessed (pii_gate).
        let Some(path) = crate::config::residency_state_path() else {
            return;
        };

        let body = match serde_json::to_value(snapshot) {
            Ok(v) => v,
            Err(e) => {
                warn!(
                    reason,
                    error = %e,
                    "residency snapshot: could not serialize — the file keeps its previous contents"
                );
                return;
            }
        };
        // The change key: identical state must not rewrite, and `written_at`
        // would defeat that on its own.
        let mut key = body.clone();
        if let Some(obj) = key.as_object_mut() {
            obj.remove("written_at");
        }

        // Held across the write: it is a short synchronous fs op with no await
        // inside, and holding it makes "compare, write, remember" atomic against
        // a concurrent publisher. Poison is irrelevant here (there is no
        // invariant to corrupt), so it is recovered rather than propagated —
        // a panic here would violate the safety property.
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        if last.as_ref() == Some(&key) {
            debug!(
                reason,
                "residency snapshot: unchanged since the last write — not rewriting"
            );
            return;
        }

        match write_state_file(&path, &body) {
            Ok(()) => {
                *last = Some(key);
                debug!(
                    reason,
                    residents = snapshot.residents.len(),
                    "residency snapshot: written"
                );
            }
            // THE SAFETY PROPERTY, at its one enforcement point: log, do not
            // remember, RETURN. The warm/release that called us carries on.
            Err(e) => warn!(
                reason,
                error = %e,
                "residency snapshot: could not be written — residency itself is unaffected, only the \
                 Terminus-visible snapshot is stale (it carries `written_at` so a reader can tell)"
            ),
        }
    }
}

/// Monotonic nonce so concurrent atomic writes never collide on a temp name.
static STATE_WRITE_NONCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Atomically write the state file: serialize to JSON, write to a uniquely-named
/// temp file in the SAME directory (so the rename is atomic on the same
/// filesystem), fsync it, then `rename` over the target. A reader either sees the
/// old complete file or the new complete file — never a torn/partial write. On any
/// error the temp file is removed so no junk accumulates.
///
/// SALVAGED verbatim in behaviour from `serving::residency::write_state_file`,
/// which was already correct; only the payload type changed. Errors are returned
/// here and swallowed by the single caller — keeping the IO fallible at this level
/// is what lets the caller's log line say something useful.
fn write_state_file(path: &str, body: &serde_json::Value) -> std::io::Result<()> {
    use std::io::Write;
    use std::sync::atomic::Ordering;

    let json = serde_json::to_vec_pretty(body).map_err(std::io::Error::other)?;

    let target = std::path::Path::new(path);
    let dir = target.parent().unwrap_or_else(|| std::path::Path::new("."));

    // Unique sibling temp name (pid + a monotonic nonce) in the SAME directory so
    // the final rename is atomic (same mount). Not a security temp — just a
    // crash-safe staging file we control and rename immediately.
    let nonce = STATE_WRITE_NONCE.fetch_add(1, Ordering::Relaxed);
    let tmp_path = dir.join(format!(".residency.{}.{}.tmp", std::process::id(), nonce));

    let write_result = (|| -> std::io::Result<()> {
        let mut f = std::fs::File::create(&tmp_path)?;
        f.write_all(&json)?;
        f.sync_all()?;
        Ok(())
    })();

    if let Err(e) = write_result {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(e);
    }

    // Atomic replace.
    if let Err(e) = std::fs::rename(&tmp_path, target) {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(e);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap() -> ResidencySnapshot {
        ResidencySnapshot::new(
            vec![ResidentEntry {
                role: "personality",
                model_id: "voice:1".into(),
                vram_gb: Some(8.0),
            }],
            Some(42.0),
            Some("voice:1".into()),
        )
    }

    /// The four fields with no live source are `null` — never `0.0`, never a
    /// defaulted mode string.
    #[test]
    fn unsourced_fields_serialize_as_null_not_zero() {
        let v = serde_json::to_value(snap()).unwrap();
        for f in [
            "mode",
            "assumed_memory_model",
            "gpu_ceiling_gb",
            "cpu_ceiling_gb",
        ] {
            assert!(
                v.get(f).is_some_and(|x| x.is_null()),
                "{f} must be JSON null, got {:?}",
                v.get(f)
            );
        }
    }

    /// An unknown numeric is OMITTED rather than written as a zero we did not
    /// measure.
    #[test]
    fn unknown_numerics_are_omitted_not_zeroed() {
        let s = ResidencySnapshot::new(
            vec![ResidentEntry {
                role: "router",
                model_id: "router:1".into(),
                vram_gb: None,
            }],
            None,
            None,
        );
        let v = serde_json::to_value(&s).unwrap();
        assert!(v.get("free_vram_gb").is_none(), "free_vram_gb must be absent");
        assert!(
            v["residents"][0].get("vram_gb").is_none(),
            "vram_gb must be absent"
        );
        // pinned_chat_model IS a nullable field in the reader's contract.
        assert!(v.get("pinned_chat_model").is_some_and(|x| x.is_null()));
    }

    #[test]
    fn written_at_is_present_and_parseable() {
        let v = serde_json::to_value(snap()).unwrap();
        let ts = v["written_at"].as_str().expect("written_at");
        chrono::DateTime::parse_from_rfc3339(ts).expect("RFC 3339");
    }

    /// The salvaged writer replaces the target atomically and leaves no temp junk.
    #[test]
    fn write_state_file_replaces_atomically_and_leaves_no_temp() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("residency.json");
        std::fs::write(&path, "stale").unwrap();

        write_state_file(
            path.to_str().unwrap(),
            &serde_json::to_value(snap()).unwrap(),
        )
        .unwrap();

        let back: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(back["residents"][0]["model_id"], "voice:1");

        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
            .filter(|n| n != "residency.json")
            .collect();
        assert!(leftovers.is_empty(), "temp junk left behind: {leftovers:?}");
    }

    /// A write to an unwritable path returns an error rather than panicking —
    /// the caller's swallow only works if the failure is a value.
    #[test]
    fn write_state_file_errors_on_an_unwritable_path() {
        let err = write_state_file(
            "/nonexistent-dir-chrd95/residency.json",
            &serde_json::to_value(snap()).unwrap(),
        );
        assert!(err.is_err());
    }
}
