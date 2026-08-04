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
//! **A failed OR SLOW snapshot write must never break, block, or slow a warm or a
//! release.** This file is observability, and observability that can take down —
//! or merely stall — the thing it observes is strictly worse than no
//! observability.
//!
//! Note the "or slow". Swallowing an IO error only helps once the IO *returns*; a
//! degraded disk, a full filesystem, an NFS stall, or an `fsync` queued behind
//! other writers does not return an error, it just takes a long time. Doing that
//! work on the caller's thread — even synchronously, even with every error
//! swallowed — would stall the warm and, while the resident-set mutex is held,
//! every other residency operation queued behind it. That is why this is not a
//! function call that happens to be careful; it is a **structural** separation:
//!
//! 1. [`SnapshotWriter::publish`] does NO filesystem IO. It serializes a small
//!    document and hands it to a dedicated OS thread. It cannot block on a disk.
//! 2. The handoff is a **single slot** — a bounded queue of depth one. A snapshot
//!    is a point-in-time diagnostic, so when a newer one arrives while the writer
//!    is still busy, the older one is DROPPED. That is the correct semantic (the
//!    superseded state is no longer true) and it means a stalled writer cannot
//!    grow memory either.
//! 3. The writer thread is an OS thread, not a tokio task, so its blocking IO
//!    cannot occupy a runtime worker.
//! 4. Both call sites additionally publish with the resident-set lock ALREADY
//!    DROPPED, so even the (tiny, IO-free) publish call is outside the critical
//!    section.
//!
//! An earlier revision of this change read the residents back through
//! `ResidentSet::status()` inside the commit and deadlocked the warm path on lock
//! re-entry; the revision after that did the IO synchronously under the lock. Both
//! were the same mistake — letting the observability path take the serving path's
//! resources, once its mutex and once its thread — and both are why the separation
//! above is structural rather than a comment asking future callers to be careful.
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

use std::sync::{Arc, Condvar, Mutex, Once};
use std::time::{Duration, Instant};

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
    /// extra network round trip to build a diagnostic field. The top-level
    /// `vram_gb_source` marker says the same thing in the file itself, where a
    /// reader will actually see it.
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

/// How long an identical payload that FAILED to write is left alone before it is
/// attempted again. See [`Worker::handle`] — without this, a persistently
/// unwritable path turns every state change into another doomed write attempt and
/// another warn line, forever.
const RETRY_BACKOFF: Duration = Duration::from_secs(60);

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

/// Where a snapshot actually goes. Injectable so tests can substitute a SLOW sink
/// and prove the warm path does not wait on it — the property is about latency as
/// well as failure, so it has to be testable with latency, not merely with errors.
type Sink = Box<dyn Fn(&str, &serde_json::Value) -> std::io::Result<()> + Send + Sync>;

/// One queued snapshot.
struct Job {
    path: String,
    body: serde_json::Value,
    /// The resident-set lifecycle generation this snapshot describes, captured
    /// under the SAME lock that committed the state change it reports.
    ///
    /// Publication happens after that lock is dropped — deliberately, so a slow
    /// disk cannot stall a warm — but dropping the lock also drops the ordering
    /// it used to give us for free. Without this, a warm that captures a
    /// non-empty set, is descheduled, and publishes after a release has already
    /// published an empty one leaves stale residents on disk as the final word.
    /// `written_at` cannot detect that: the stale write carries the LATER
    /// wall-clock stamp and so looks fresher than the truth.
    generation: u64,
    /// `body` minus `written_at` — the change key. `written_at` moves on every
    /// call by construction and would defeat change detection on its own.
    key: serde_json::Value,
    reason: String,
}

/// The single-slot handoff. Depth ONE, drop-oldest: see the module docs.
#[derive(Default)]
struct Queue {
    next: Option<Job>,
    /// Monotonic count of publishes accepted. With `done`, lets a test wait for
    /// quiescence without sleeping.
    queued: u64,
    done: u64,
    shutdown: bool,
}

struct Shared {
    queue: Mutex<Queue>,
    cv: Condvar,
    sink: Sink,
    /// The destination, supplied when the writer is built and never re-read.
    ///
    /// Deliberately not resolved from the environment here — not per publish, and
    /// not per construction either. In production the value cannot change
    /// (`CHORD_RESIDENCY_STATE_PATH` is process-lifetime config), so an ambient
    /// read bought nothing and cost correctness: EVERY writer in the process
    /// silently targets whatever path is configured at the instant it reads.
    ///
    /// That is not a hypothetical. It was measured: with the resolution ambient,
    /// one test's tempdir path was written by the writers of a dozen unrelated
    /// tests running in parallel, because the env var it set was visible to all of
    /// them. Moving the read from publish-time to construction-time narrowed the
    /// window but did not close it — a writer merely had to be CONSTRUCTED inside
    /// the window instead of publishing inside it. Only making the destination a
    /// parameter closes it, because then there is no instant at which a writer
    /// consults shared state at all.
    ///
    /// `None` ⇒ snapshot persistence is off for this writer. Not an error: the
    /// path is never guessed (pii_gate).
    path: Option<String>,
}

/// Publishes the snapshot on a STATE CHANGE, never on every reconcile tick, and
/// never on the caller's thread.
///
/// The resident set re-asserts on a background loop; writing the same bytes every
/// tick is pointless disk churn on a shared host. Change detection lives in the
/// writer thread (so even the comparison is off the serving path): it remembers
/// the last body it successfully wrote and skips an identical one.
pub struct SnapshotWriter {
    shared: Arc<Shared>,
    /// The writer thread is spawned on FIRST publish, not at construction: the
    /// resident set is built in a great many tests that never configure a
    /// snapshot path, and none of them should get a thread.
    spawn: Once,
}

// Deliberately NO `Default`. A default writer would have to invent a
// destination, and the only one available to invent is the process-global env
// var — the exact ambient this type no longer consults. Callers say where.

impl SnapshotWriter {
    /// A writer bound to an EXPLICIT destination.
    ///
    /// The path is a parameter, not something this type goes and looks up. The
    /// ambient `CHORD_RESIDENCY_STATE_PATH` is read in exactly one place — the
    /// owner's config, once, at construction — and travels from there. That is
    /// what makes two writers in one process independent of each other; see the
    /// `path` field for the failure this prevents.
    ///
    /// `None` ⇒ snapshot persistence is off for this writer.
    pub fn to_path(path: Option<String>) -> Self {
        Self::with_path_and_sink(path, Box::new(write_state_file))
    }

    /// A writer with an explicit destination and a substituted sink. Test seam —
    /// see [`Sink`].
    pub fn with_path_and_sink(path: Option<String>, sink: Sink) -> Self {
        Self {
            shared: Arc::new(Shared {
                path,
                queue: Mutex::new(Queue::default()),
                cv: Condvar::new(),
                sink,
            }),
            spawn: Once::new(),
        }
    }

    /// Hand the snapshot to the writer thread if the state actually changed.
    ///
    /// **Infallible AND non-blocking by contract** — see the module docs. It
    /// performs no filesystem IO whatsoever: the only lock it takes is the
    /// handoff's own, held for the duration of an `Option` replace. Every failure
    /// mode (no path configured, serialization, IO) is logged and swallowed, here
    /// or on the writer thread. Nothing in this call can propagate into, or wait
    /// on behalf of, a warm or a release.
    ///
    /// `reason` is only for the log line.
    pub fn publish(&self, snapshot: &ResidencySnapshot, generation: u64, reason: &str) {
        // No path bound at construction ⇒ persistence is off for this writer.
        // Checked BEFORE the thread is spawned, so a set with no snapshot path
        // never starts one.
        let Some(path) = self.shared.path.clone() else {
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
        let mut key = body.clone();
        if let Some(obj) = key.as_object_mut() {
            obj.remove("written_at");
        }

        let shared = self.shared.clone();
        self.spawn.call_once(move || {
            if let Err(e) = std::thread::Builder::new()
                .name("residency-snapshot".into())
                .spawn(move || Worker::new(shared).run())
            {
                // Even this is soft: no writer thread means no snapshot, which is
                // exactly where we were before CHRD-95 — never a failed warm.
                warn!(
                    error = %e,
                    "residency snapshot: writer thread could not be started — snapshots are off for this process"
                );
            }
        });

        // Depth-one, drop-oldest. Superseding an unwritten snapshot is CORRECT
        // rather than merely acceptable: the dropped one is BY DEFINITION
        // superseded by a newer one that does get written, and newer is more true.
        // There is no ordering in which dropping the older loses information the
        // reader wanted.
        //
        // The one residual case, recorded here so it is not rediscovered: a
        // snapshot dropped while the writer is in its failure BACKOFF leaves the
        // on-disk file slightly stale. That is exactly what `written_at` is for —
        // the reader can see how old the state is and age it out — and it is
        // strictly better than the pre-CHRD-95 behaviour, where there was no file
        // and the tool asserted "IDLE".
        let mut q = self.shared.queue.lock().unwrap_or_else(|e| e.into_inner());
        q.queued += 1;
        if q.next.is_some() {
            debug!(
                reason,
                "residency snapshot: superseded an unwritten snapshot — only the newest state is published"
            );
        }
        q.next = Some(Job {
            path,
            body,
            generation,
            key,
            reason: reason.to_string(),
        });
        drop(q);
        self.shared.cv.notify_one();
    }

    /// Wait until everything published so far has been processed (written,
    /// skipped, or failed). Returns `false` on timeout.
    ///
    /// Test support, and deliberately not called from any serving path — the whole
    /// point of this module is that nothing on that path ever waits for the disk.
    pub fn flush(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let mut q = self.shared.queue.lock().unwrap_or_else(|e| e.into_inner());
        let target = q.queued;
        while q.done < target {
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                return false;
            };
            let (guard, res) = self
                .shared
                .cv
                .wait_timeout(q, remaining)
                .unwrap_or_else(|e| e.into_inner());
            q = guard;
            if res.timed_out() && q.done < target {
                return false;
            }
        }
        true
    }
}

impl Drop for SnapshotWriter {
    fn drop(&mut self) {
        if let Ok(mut q) = self.shared.queue.lock() {
            q.shutdown = true;
        }
        // Signal and go. **Deliberately NOT joined, and this must stay that way.**
        // A join would block the dropping thread for however long an in-flight
        // write takes — i.e. it would reintroduce, at shutdown, the exact coupling
        // to a hung disk that the whole module is structured to avoid, turning a
        // stalled NFS mount into a process that will not exit. The cost of not
        // joining is that a snapshot in flight at shutdown may not land, which is
        // the correct trade: it describes a process that is going away, and the
        // next Chord to start republishes on its first warm.
        self.shared.cv.notify_all();
    }
}

/// The writer thread. Owns change detection and every filesystem call.
struct Worker {
    shared: Arc<Shared>,
    /// Last key CONFIRMED on disk. An identical payload is skipped.
    ///
    /// ASSUMPTION: the key is the BODY only, not the destination path. That is
    /// safe exactly as long as the path is fixed for the process lifetime — it
    /// comes from `CHORD_RESIDENCY_STATE_PATH`, read per publish but never
    /// changed at runtime. If the path ever becomes dynamic, this key must
    /// include it, or an unchanged body would skip the first write to a NEW file.
    last_written: Option<serde_json::Value>,
    /// Highest generation actually written. A job older than this is stale and is
    /// dropped without a trace — see `handle`.
    high_generation: Option<u64>,
    /// Last key that FAILED, and when. Kept separate from `last_written` so a
    /// repeated identical payload does not re-attempt doomed IO on every state
    /// change, while a retry still happens once [`RETRY_BACKOFF`] has elapsed —
    /// a transient failure (a full disk that is later cleared) still heals.
    last_failed: Option<(serde_json::Value, Instant)>,
}

impl Worker {
    fn new(shared: Arc<Shared>) -> Self {
        Self {
            shared,
            last_written: None,
            high_generation: None,
            last_failed: None,
        }
    }

    fn run(mut self) {
        loop {
            let (job, seq) = 'wait: loop {
                let mut q = self.shared.queue.lock().unwrap_or_else(|e| e.into_inner());
                loop {
                    if let Some(job) = q.next.take() {
                        break 'wait (job, q.queued);
                    }
                    if q.shutdown {
                        return;
                    }
                    q = self.shared.cv.wait(q).unwrap_or_else(|e| e.into_inner());
                }
            };

            self.handle(job);

            let mut q = self.shared.queue.lock().unwrap_or_else(|e| e.into_inner());
            q.done = seq;
            drop(q);
            self.shared.cv.notify_all();
        }
    }

    fn handle(&mut self, job: Job) {
        // ORDERING. Publication is off the serving path, so two commits can reach
        // this thread out of order: a warm captures a non-empty set, is
        // descheduled, and arrives AFTER a release that already published an empty
        // one. Writing it would leave residents on disk as the last word after the
        // GPU was emptied — a durable wrong answer, and the exact class of lie
        // CHRD-95 exists to remove.
        //
        // The generation is captured under the same lock that commits the state,
        // so it orders commits even though the writes race. A stale job is dropped
        // WITHOUT A TRACE: it must not touch `last_written` or `last_failed`
        // either, or it would suppress or mis-attribute the write that supersedes
        // it.
        if let Some(high) = self.high_generation {
            if job.generation < high {
                debug!(
                    reason = %job.reason,
                    job_generation = job.generation,
                    written_generation = high,
                    "residency snapshot: superseded by a newer commit — dropping the stale write"
                );
                return;
            }
        }
        if self.last_written.as_ref() == Some(&job.key) {
            debug!(
                reason = %job.reason,
                "residency snapshot: unchanged since the last write — not rewriting"
            );
            return;
        }
        if let Some((failed_key, at)) = &self.last_failed {
            if failed_key == &job.key && at.elapsed() < RETRY_BACKOFF {
                debug!(
                    reason = %job.reason,
                    "residency snapshot: identical payload already failed recently — backing off rather than retrying"
                );
                return;
            }
        }

        // PANIC ISOLATION. The writer thread is spawned exactly once (`Once`), so
        // a panic escaping the sink would unwind it and detach the whole mechanism
        // permanently — and the only symptom would be that snapshots quietly stop
        // appearing. That is precisely the failure class CHRD-95 exists to remove:
        // a component that has stopped working while looking fine. (Terminus would
        // at least say `state=UNKNOWN` rather than lie, but the cause would be
        // invisible.) So a panicking sink is caught, logged loudly ONCE, and the
        // thread carries on serving subsequent publishes.
        //
        // `AssertUnwindSafe` is sound here: the closure owns `job` outright, the
        // sink is `Fn` (no interior state we mutate across the boundary), and the
        // only state that survives a panic is `last_written`/`last_failed` below —
        // which are updated AFTER the boundary, from the result, never during it.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            (self.shared.sink)(&job.path, &job.body)
        }));
        let result = match outcome {
            Ok(r) => r,
            Err(_) => {
                // Treated as a failed write: it takes the same backoff, so a sink
                // that panics deterministically on one payload does not spin.
                self.last_failed = Some((job.key, Instant::now()));
                tracing::error!(
                    reason = %job.reason,
                    "residency snapshot: the writer PANICKED — snapshot skipped, writer kept alive. \
                     Residency itself is unaffected; this is a bug in the snapshot path, not the serving path"
                );
                return;
            }
        };

        match result {
            Ok(()) => {
                self.last_written = Some(job.key);
                // Only a SUCCESSFUL write advances the ordering high-water mark. A
                // failed or panicking write must not raise it, or the retry that
                // supersedes it would itself be judged stale and dropped.
                self.high_generation = Some(job.generation);
                self.last_failed = None;
                debug!(reason = %job.reason, "residency snapshot: written");
            }
            // Log, remember the failure for the backoff, carry on. Nobody is
            // waiting on this.
            Err(e) => {
                self.last_failed = Some((job.key, Instant::now()));
                warn!(
                    reason = %job.reason,
                    error = %e,
                    "residency snapshot: could not be written — residency itself is unaffected, only the \
                     Terminus-visible snapshot is stale (it carries `written_at` so a reader can tell)"
                );
            }
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
/// which was already correct; only the payload type changed. This is the only
/// blocking code in the module and it runs exclusively on the writer thread.
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
    use std::sync::atomic::{AtomicUsize, Ordering};

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

    /// A destination inside a tempdir owned by the calling test.
    ///
    /// There used to be a `PathGuard` here that set `CHORD_RESIDENCY_STATE_PATH`
    /// instead, and it was the wrong shape: a process-global that every OTHER
    /// concurrently-running test's writer could read. The destination is a
    /// parameter now, so a test's file is reachable by exactly one writer — its
    /// own — and these tests need no serialization to stay isolated from each
    /// other.
    fn dest(dir: &tempfile::TempDir) -> Option<String> {
        Some(dir.path().join("residency.json").to_string_lossy().into_owned())
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
    /// the swallow only works if the failure is a value.
    #[test]
    fn write_state_file_errors_on_an_unwritable_path() {
        let err = write_state_file(
            "/nonexistent-dir-chrd95/residency.json",
            &serde_json::to_value(snap()).unwrap(),
        );
        assert!(err.is_err());
    }

    /// `publish` performs no filesystem IO on the calling thread: with a sink that
    /// takes seconds, publishing still returns immediately. This is the latency
    /// half of the safety property — a slow disk must not be able to slow a warm.
    #[test]
    fn publish_does_not_wait_for_the_sink() {
        let dir = tempfile::tempdir().unwrap();

        let writer = SnapshotWriter::with_path_and_sink(dest(&dir), Box::new(|_, _| {
            std::thread::sleep(Duration::from_secs(2));
            Ok(())
        }));

        let t0 = Instant::now();
        writer.publish(&snap(), 1, "test");
        let elapsed = t0.elapsed();
        assert!(
            elapsed < Duration::from_millis(250),
            "publish waited on the sink ({elapsed:?}) — the write is back on the caller's thread"
        );
    }

    /// The handoff is depth-one and drop-oldest, so a stalled writer cannot grow
    /// memory: 200 publishes behind a slow sink reach the disk a handful of times,
    /// not 200.
    #[test]
    fn a_stalled_writer_drops_superseded_snapshots_instead_of_queueing_them() {
        let dir = tempfile::tempdir().unwrap();

        let seen = Arc::new(AtomicUsize::new(0));
        let s = seen.clone();
        let writer = SnapshotWriter::with_path_and_sink(dest(&dir), Box::new(move |_, _| {
            s.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(200));
            Ok(())
        }));

        for i in 0..200 {
            writer.publish(
                &ResidencySnapshot::new(
                    vec![ResidentEntry {
                        role: "router",
                        model_id: format!("m:{i}"),
                        vram_gb: None,
                    }],
                    None,
                    None,
                ),
                i as u64 + 1,
                "test",
            );
        }
        assert!(writer.flush(Duration::from_secs(20)), "writer never drained");
        assert!(
            seen.load(Ordering::SeqCst) <= 5,
            "superseded snapshots must be dropped, not queued: {} writes for 200 publishes",
            seen.load(Ordering::SeqCst)
        );
    }

    /// Change detection: an identical payload is not rewritten.
    #[test]
    fn an_identical_payload_is_not_rewritten() {
        let dir = tempfile::tempdir().unwrap();

        let writes = Arc::new(AtomicUsize::new(0));
        let w = writes.clone();
        let writer = SnapshotWriter::with_path_and_sink(dest(&dir), Box::new(move |_, _| {
            w.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }));

        for _ in 0..3 {
            writer.publish(&snap(), 1, "test");
            assert!(writer.flush(Duration::from_secs(5)));
        }
        assert_eq!(
            writes.load(Ordering::SeqCst),
            1,
            "only the first of three identical snapshots should reach the disk"
        );
    }

    /// ORDERING. Publication is off the serving path, so commits can reach the
    /// writer out of order. A stale one must be DROPPED, not written.
    ///
    /// The case that motivated this: a warm captures a non-empty resident set, is
    /// descheduled, and arrives after a release has already published an empty
    /// one. Writing it leaves residents on disk as the final word after the GPU
    /// was emptied. `written_at` cannot catch it — the stale write carries the
    /// LATER wall-clock stamp, so it looks fresher than the truth.
    #[test]
    fn a_superseded_commit_is_dropped_not_written() {
        // The residency path is a process-global env var, so these need the
        // same guard + serialization every other test in this module uses.
        // Without it `publish` returns early (no path configured) and the sink
        // never runs — which reads as "nothing was written", i.e. exactly the
        // assertion these make, passing or failing for the wrong reason.
        let dir = tempfile::tempdir().unwrap();
        let seen = Arc::new(std::sync::Mutex::new(Vec::<serde_json::Value>::new()));
        let s2 = seen.clone();
        let writer = SnapshotWriter::with_path_and_sink(dest(&dir), Box::new(move |_p: &str, body: &serde_json::Value| {
            s2.lock().unwrap().push(body.clone());
            Ok(())
        }));

        // Generation 5 lands first (the release), then generation 3 arrives late
        // (the descheduled warm). The late one is older and must not be written.
        writer.publish(&ResidencySnapshot::new(Vec::new(), None, None), 5, "release");
        assert!(writer.flush(Duration::from_secs(20)), "writer never drained");
        writer.publish(&snap(), 3, "late-warm");
        assert!(writer.flush(Duration::from_secs(20)), "writer never drained");

        let writes = seen.lock().unwrap();
        assert_eq!(writes.len(), 1, "the superseded commit must not have been written");
        assert_eq!(
            writes[0]["residents"].as_array().map(|a| a.len()),
            Some(0),
            "the file must still hold the RELEASE's empty set, not the stale warm's"
        );
    }

    /// POSITIVE CONTROL for the above. In the RIGHT order the later commit is
    /// written — so a build that rejected everything, or that dropped every
    /// second job, could not pass the pair.
    #[test]
    fn a_newer_commit_is_written() {
        // The residency path is a process-global env var, so these need the
        // same guard + serialization every other test in this module uses.
        // Without it `publish` returns early (no path configured) and the sink
        // never runs — which reads as "nothing was written", i.e. exactly the
        // assertion these make, passing or failing for the wrong reason.
        let dir = tempfile::tempdir().unwrap();
        let seen = Arc::new(std::sync::Mutex::new(Vec::<serde_json::Value>::new()));
        let s2 = seen.clone();
        let writer = SnapshotWriter::with_path_and_sink(dest(&dir), Box::new(move |_p: &str, body: &serde_json::Value| {
            s2.lock().unwrap().push(body.clone());
            Ok(())
        }));

        writer.publish(&snap(), 3, "warm");
        assert!(writer.flush(Duration::from_secs(20)), "writer never drained");
        writer.publish(&ResidencySnapshot::new(Vec::new(), None, None), 5, "release");
        assert!(writer.flush(Duration::from_secs(20)), "writer never drained");

        let writes = seen.lock().unwrap();
        assert_eq!(writes.len(), 2, "a newer commit must be written");
        assert_eq!(
            writes[1]["residents"].as_array().map(|a| a.len()),
            Some(0),
            "and the LAST word must be the release"
        );
    }

    /// An EQUAL generation is not stale — a warm and its own follow-up report the
    /// same commit, and the second must not be silently swallowed by the ordering
    /// guard (change detection, not ordering, is what suppresses a true duplicate).
    #[test]
    fn an_equal_generation_is_not_treated_as_stale() {
        // The residency path is a process-global env var, so these need the
        // same guard + serialization every other test in this module uses.
        // Without it `publish` returns early (no path configured) and the sink
        // never runs — which reads as "nothing was written", i.e. exactly the
        // assertion these make, passing or failing for the wrong reason.
        let dir = tempfile::tempdir().unwrap();
        let seen = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let s2 = seen.clone();
        let writer = SnapshotWriter::with_path_and_sink(dest(&dir), Box::new(move |_p: &str, _b: &serde_json::Value| {
            s2.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }));

        writer.publish(&snap(), 7, "warm");
        assert!(writer.flush(Duration::from_secs(20)), "writer never drained");
        // Same generation, DIFFERENT body — must still be written.
        writer.publish(&ResidencySnapshot::new(Vec::new(), None, None), 7, "warm-followup");
        assert!(writer.flush(Duration::from_secs(20)), "writer never drained");
        assert_eq!(seen.load(Ordering::SeqCst), 2, "an equal generation must not be dropped");
    }

    /// A FAILED write must not advance the high-water mark, or the retry that
    /// supersedes it would itself be judged stale and dropped — leaving the file
    /// permanently behind with no way to catch up.
    #[test]
    fn a_failed_write_does_not_advance_the_ordering_mark() {
        // The residency path is a process-global env var, so these need the
        // same guard + serialization every other test in this module uses.
        // Without it `publish` returns early (no path configured) and the sink
        // never runs — which reads as "nothing was written", i.e. exactly the
        // assertion these make, passing or failing for the wrong reason.
        let dir = tempfile::tempdir().unwrap();
        let fail = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let seen = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (f2, s2) = (fail.clone(), seen.clone());
        let writer = SnapshotWriter::with_path_and_sink(dest(&dir), Box::new(move |_p: &str, _b: &serde_json::Value| {
            if f2.load(Ordering::SeqCst) {
                Err(std::io::Error::other("disk full"))
            } else {
                s2.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
        }));

        writer.publish(&snap(), 9, "warm-fails");
        assert!(writer.flush(Duration::from_secs(20)), "writer never drained");
        fail.store(false, Ordering::SeqCst);
        // A LOWER generation than the failed one would still be stale, but an equal
        // or newer one must go through — the failure must not have raised the mark.
        writer.publish(&ResidencySnapshot::new(Vec::new(), None, None), 9, "retry");
        assert!(writer.flush(Duration::from_secs(20)), "writer never drained");
        assert_eq!(seen.load(Ordering::SeqCst), 1, "the retry after a failure must be written");
    }

    /// A PANICKING sink must not latch the writer dead. The thread is spawned
    /// once, so an escaping panic would detach the mechanism permanently and the
    /// only symptom would be snapshots quietly ceasing — the same "stopped working
    /// while looking fine" failure this whole change exists to remove.
    #[test]
    fn a_panicking_sink_does_not_kill_the_writer() {
        let dir = tempfile::tempdir().unwrap();

        let calls = Arc::new(AtomicUsize::new(0));
        let c = calls.clone();
        let landed = Arc::new(AtomicUsize::new(0));
        let l = landed.clone();
        let writer = SnapshotWriter::with_path_and_sink(dest(&dir), Box::new(move |_, _| {
            if c.fetch_add(1, Ordering::SeqCst) == 0 {
                panic!("sink panicked on its first call");
            }
            l.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }));

        writer.publish(&snap(), 1, "first");
        assert!(writer.flush(Duration::from_secs(5)), "a panic must not wedge the queue");

        // A DIFFERENT payload, so the failure backoff does not apply.
        writer.publish(&ResidencySnapshot::new(Vec::new(), None, None), 2, "second");
        assert!(writer.flush(Duration::from_secs(5)));

        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "the writer thread must survive the panic and take the next job"
        );
        assert_eq!(
            landed.load(Ordering::SeqCst),
            1,
            "and the subsequent write must actually land"
        );
    }

    /// A FAILED write must not turn every subsequent state change into another
    /// doomed attempt: an identical payload backs off, while a genuinely new one
    /// is still attempted (so a cleared disk heals on the next real change).
    #[test]
    fn a_failed_identical_payload_backs_off_but_a_new_one_is_still_attempted() {
        let dir = tempfile::tempdir().unwrap();

        let attempts = Arc::new(AtomicUsize::new(0));
        let a = attempts.clone();
        let writer = SnapshotWriter::with_path_and_sink(dest(&dir), Box::new(move |_, _| {
            a.fetch_add(1, Ordering::SeqCst);
            Err(std::io::Error::other("disk full"))
        }));

        for _ in 0..5 {
            writer.publish(&snap(), 1, "test");
            assert!(writer.flush(Duration::from_secs(5)));
        }
        assert_eq!(
            attempts.load(Ordering::SeqCst),
            1,
            "an identical payload that just failed must back off, not retry on every state change"
        );

        writer.publish(&ResidencySnapshot::new(Vec::new(), None, None), 2, "mode-swap");
        assert!(writer.flush(Duration::from_secs(5)));
        assert_eq!(
            attempts.load(Ordering::SeqCst),
            2,
            "a genuinely different payload must still be attempted after a failure"
        );
    }
}
