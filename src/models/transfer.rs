//! TIER-02: transparent archive pull (cold → warm), the copy half of the
//! cold → warm → hot promotion path.
//!
//! When an inference request names a model that lives only in the archive (cold
//! tier), [`archive_pull`] copies that model's Ollama **manifest** leaf plus
//! every **blob** it references from the archive root to the local Ollama root,
//! preserving the relative `manifests/.../<tag>` + `blobs/sha256-…` layout.
//! Ollama recognises a model the moment those files exist locally, so after a
//! successful pull the model is loadable into VRAM by the existing lifecycle
//! ([`crate::harness::vram_lifecycle`]) — that warm → hot step is *not* done here.
//!
//! ## Robustness guarantees (from the TIER-02 spec)
//! - **Disk precheck (fail fast):** the model's on-disk size is summed from its
//!   manifest blobs and checked against free space on the local filesystem
//!   *before* any byte is copied. Insufficient space → error, nothing written.
//! - **Concurrent-pull dedup:** a [`PullCoordinator`] holds a per-model async
//!   lock so two requests for the same cold model never double-copy — the second
//!   awaits the first, then sees the model already warm and returns early.
//! - **Timeout:** the copy runs as an owned, cancellable task. On expiry it is
//!   *cancelled and awaited* before anything is removed, so cleanup is ordered
//!   strictly after the copy's last write rather than merely after the timer
//!   fired. Only files this pull published — never a blob that was already there
//!   — are removed.
//! - **Mid-copy failure cleanup:** any error partway through copying goes through
//!   that same single cleanup route, with the same pre-existing filter.
//! - **No in-place writes:** every file is staged in a temp file in the
//!   destination's own directory and published with `rename`, so a local blob
//!   Ollama may have loaded is never truncated or partially overwritten.
//! - **Progress events:** an optional [`PullEvent`] channel surfaces
//!   "retrieving from archive" / "loading into VRAM" so a long NFS copy doesn't
//!   look stuck. Tests pass `None`.
//!
//! Nothing here hardcodes infrastructure — all paths come from the registry /
//! config, model names from the request.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, Mutex};

use super::registry::{
    manifest_rel_path, parse_manifest_blobs, ManifestBlobs, ModelRegistry, StorageTier,
};

// ── Progress events ────────────────────────────────────────────────────────────

/// Progress events emitted during a pull, mirroring the
/// [`crate::agentic::streaming::ProgressEvent`] tagged-enum style so they can be
/// forwarded onto an SSE stream by the caller.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PullEvent {
    /// The model's blobs are being copied from the archive to local disk.
    RetrievingFromArchive {
        /// Model name being pulled.
        model: String,
        /// Approximate size in GiB (for a human-readable "fetching N GB" message).
        size_gb: f64,
    },
    /// The local copy is complete; the model is being loaded into VRAM (the
    /// warm → hot step performed by the existing lifecycle). Emitted by the
    /// higher-level `ensure_local` flow, not by the raw copy.
    LoadingIntoVram {
        /// Model name being loaded.
        model: String,
    },
}

// ── Errors ───────────────────────────────────────────────────────────────────

/// Errors an [`archive_pull`] / [`PullCoordinator::ensure_local`] may surface.
#[derive(Debug, thiserror::Error)]
pub enum PullError {
    /// The model is not known to the registry at all.
    #[error("model not found in registry: {0}")]
    UnknownModel(String),
    /// The model has no archive path / its manifest is missing from the archive.
    #[error("model not present in archive: {0}")]
    MissingArchive(String),
    /// Not enough free space on the local filesystem to hold the model.
    #[error("insufficient disk space: need {need_gb:.2} GB, have {have_gb:.2} GB")]
    InsufficientDiskSpace { need_gb: f64, have_gb: f64 },
    /// The pull exceeded its configured timeout (partial files cleaned up).
    #[error("archive pull timed out after {0:?}")]
    Timeout(Duration),
    /// An I/O error during copy (partial files cleaned up).
    #[error("archive pull I/O error: {0}")]
    Io(String),
    /// `stash_existing_manifest` could not clear a pre-existing manifest
    /// backup and therefore never reached a known-safe state: no blob was
    /// copied and this attempt created no backup of its own. The generic
    /// undo/restore route must NOT run for this error — `restore_manifest_backup`
    /// has no way to tell this untouched, possibly-stale backup apart from one
    /// this attempt legitimately created, so restoring it would resurrect
    /// content that predates this attempt (see round-9/10 history in
    /// `stash_existing_manifest`'s doc comment). Every caller checks for this
    /// variant and skips the restore step, leaving the local tree exactly as
    /// found.
    #[error("archive pull could not verify the previous manifest backup was safely cleared: {0}")]
    StashUnsafe(String),
}

const BYTES_PER_GB: f64 = 1_073_741_824.0; // 1 GiB

fn bytes_to_gb(bytes: u64) -> f64 {
    bytes as f64 / BYTES_PER_GB
}

// ── Disk-space probe (injectable for tests) ────────────────────────────────────

/// Abstracts "how many free bytes are on the filesystem holding this path".
/// Production uses [`StatvfsProbe`] (a pure `statvfs(2)` FFI call, no shelling
/// out); tests inject a fake to exercise the insufficient-space path
/// deterministically.
pub trait DiskSpaceProbe: Send + Sync {
    /// Free bytes available to an unprivileged process on the filesystem
    /// containing `path`. `None` if it can't be determined (caller treats an
    /// unknown probe as "assume enough" so a probe failure never blocks a pull).
    fn available_bytes(&self, path: &Path) -> Option<u64>;

    /// Total bytes of the filesystem containing `path`. `None` if it can't be
    /// determined. Used by the TIER-03 disk-pressure check (used% =
    /// (total − free) / total). Has a default of `None` so existing test probes
    /// that only implement `available_bytes` keep compiling; the production
    /// [`StatvfsProbe`] overrides it.
    fn total_bytes(&self, _path: &Path) -> Option<u64> {
        None
    }
}

/// Real disk-space probe backed by `statvfs(2)` via a tiny FFI binding (no
/// `libc` crate dependency, no `df` subprocess).
pub struct StatvfsProbe;

impl DiskSpaceProbe for StatvfsProbe {
    fn available_bytes(&self, path: &Path) -> Option<u64> {
        statvfs_available_bytes(path)
    }

    fn total_bytes(&self, path: &Path) -> Option<u64> {
        statvfs_total_bytes(path)
    }
}

/// `statvfs(2)` binding. We only need `f_bavail` (blocks free to unprivileged
/// users) × `f_frsize` (fragment size). The struct layout below matches the
/// Linux `struct statvfs`; fields we don't use are padding-correct `c_ulong`s.
#[cfg(target_os = "linux")]
fn statvfs_available_bytes(path: &Path) -> Option<u64> {
    use std::ffi::CString;
    use std::os::raw::{c_int, c_ulong};

    #[repr(C)]
    struct Statvfs {
        f_bsize: c_ulong,
        f_frsize: c_ulong,
        f_blocks: c_ulong,
        f_bfree: c_ulong,
        f_bavail: c_ulong,
        f_files: c_ulong,
        f_ffree: c_ulong,
        f_favail: c_ulong,
        f_fsid: c_ulong,
        f_flag: c_ulong,
        f_namemax: c_ulong,
        // glibc reserves trailing ints; oversize the buffer to be safe.
        __reserved: [c_int; 6],
    }

    extern "C" {
        fn statvfs(path: *const std::os::raw::c_char, buf: *mut Statvfs) -> c_int;
    }

    let cpath = CString::new(path.as_os_str().to_string_lossy().as_bytes()).ok()?;
    // Safety: `buf` is a valid, sized, writable struct; `cpath` is NUL-terminated.
    let mut buf: Statvfs = unsafe { std::mem::zeroed() };
    let rc = unsafe { statvfs(cpath.as_ptr(), &mut buf as *mut Statvfs) };
    if rc != 0 {
        return None;
    }
    let frsize = if buf.f_frsize != 0 { buf.f_frsize } else { buf.f_bsize };
    Some((buf.f_bavail as u64).saturating_mul(frsize as u64))
}

#[cfg(not(target_os = "linux"))]
fn statvfs_available_bytes(_path: &Path) -> Option<u64> {
    None
}

/// `statvfs(2)`-derived total size (bytes) of the filesystem containing `path`:
/// `f_blocks` (total data blocks) × `f_frsize`. Used by the TIER-03
/// disk-pressure calculation. Mirrors [`statvfs_available_bytes`] but reads
/// `f_blocks` instead of `f_bavail`.
#[cfg(target_os = "linux")]
fn statvfs_total_bytes(path: &Path) -> Option<u64> {
    use std::ffi::CString;
    use std::os::raw::{c_int, c_ulong};

    #[repr(C)]
    struct Statvfs {
        f_bsize: c_ulong,
        f_frsize: c_ulong,
        f_blocks: c_ulong,
        f_bfree: c_ulong,
        f_bavail: c_ulong,
        f_files: c_ulong,
        f_ffree: c_ulong,
        f_favail: c_ulong,
        f_fsid: c_ulong,
        f_flag: c_ulong,
        f_namemax: c_ulong,
        __reserved: [c_int; 6],
    }

    extern "C" {
        fn statvfs(path: *const std::os::raw::c_char, buf: *mut Statvfs) -> c_int;
    }

    let cpath = CString::new(path.as_os_str().to_string_lossy().as_bytes()).ok()?;
    // Safety: `buf` is a valid, sized, writable struct; `cpath` is NUL-terminated.
    let mut buf: Statvfs = unsafe { std::mem::zeroed() };
    let rc = unsafe { statvfs(cpath.as_ptr(), &mut buf as *mut Statvfs) };
    if rc != 0 {
        return None;
    }
    let frsize = if buf.f_frsize != 0 { buf.f_frsize } else { buf.f_bsize };
    Some((buf.f_blocks as u64).saturating_mul(frsize as u64))
}

#[cfg(not(target_os = "linux"))]
fn statvfs_total_bytes(_path: &Path) -> Option<u64> {
    None
}

// ── Manifest / blob location ───────────────────────────────────────────────────

/// Locate the actual manifest leaf for `name` under `<root>/manifests`. We
/// *discover* it (rather than blindly reconstructing the registry-host segment)
/// so the on-disk host (`registry.ollama.ai`, `hf.co`, …) doesn't have to match
/// any assumption. Falls back to the canonical [`manifest_rel_path`] layout if a
/// direct walk finds nothing (keeps tests that use the canonical layout simple).
pub(crate) fn find_manifest_leaf(root: &Path, name: &str) -> Option<PathBuf> {
    let manifests = root.join("manifests");
    // Canonical layout first (cheap, exact).
    if let Some(rel) = manifest_rel_path(name) {
        let candidate = manifests.join(&rel);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    // Otherwise walk: match on the trailing `<namespace>/<model>/<tag>` components.
    let (body, tag) = name.rsplit_once(':')?;
    let model = body.rsplit('/').next()?;
    let mut found = None;
    walk_for_leaf(&manifests, model, tag, &mut found);
    found
}

/// Recursively search for a file leaf named `tag` whose parent dir is `model`.
fn walk_for_leaf(dir: &Path, model: &str, tag: &str, out: &mut Option<PathBuf>) {
    if out.is_some() {
        return;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        if out.is_some() {
            return;
        }
        let path = entry.path();
        match entry.file_type() {
            Ok(ft) if ft.is_dir() => walk_for_leaf(&path, model, tag, out),
            Ok(ft) if ft.is_file() => {
                let is_tag = path.file_name().map(|n| n == tag).unwrap_or(false);
                let parent_is_model = path
                    .parent()
                    .and_then(|p| p.file_name())
                    .map(|n| n == model)
                    .unwrap_or(false);
                if is_tag && parent_is_model {
                    *out = Some(path);
                }
            }
            _ => {}
        }
    }
}

/// Convert a blob digest (`sha256:HEX`) to its on-disk filename (`sha256-HEX`).
pub(crate) fn blob_filename(digest: &str) -> String {
    digest.replacen(':', "-", 1)
}

// ── Core pull ──────────────────────────────────────────────────────────────────

/// Inputs describing one model's archive location, resolved from the registry.
/// `Clone` so the copy can be handed to an owned `tokio::spawn`ed task while the
/// caller keeps its own copy for cleanup.
#[derive(Clone)]
struct PullPlan {
    name: String,
    archive_root: PathBuf,
    local_root: PathBuf,
    archive_manifest: PathBuf,
    blobs: ManifestBlobs,
}

/// Resolve where the model lives in the archive and what blobs it needs, or an
/// error explaining why it can't be pulled.
fn plan_pull(name: &str, archive_root: &Path, local_root: &Path) -> Result<PullPlan, PullError> {
    let archive_manifest = find_manifest_leaf(archive_root, name)
        .ok_or_else(|| PullError::MissingArchive(name.to_string()))?;
    let blobs = parse_manifest_blobs(&archive_manifest);
    Ok(PullPlan {
        name: name.to_string(),
        archive_root: archive_root.to_path_buf(),
        local_root: local_root.to_path_buf(),
        archive_manifest,
        blobs,
    })
}

/// Copy a model's manifest + blobs from `archive_root` to `local_root`.
///
/// Performs the disk-space precheck, the timeout-wrapped copy, and partial-file
/// cleanup on any failure. Emits [`PullEvent::RetrievingFromArchive`] (when a
/// channel is provided) before copying. Does **not** touch the registry or VRAM —
/// callers ([`PullCoordinator::ensure_local`]) own those side effects.
///
/// `disk_probe` is injectable so tests can force the insufficient-space path;
/// production passes [`StatvfsProbe`].
#[allow(clippy::too_many_arguments)]
pub async fn archive_pull(
    name: &str,
    archive_root: &Path,
    local_root: &Path,
    timeout: Duration,
    disk_probe: &dyn DiskSpaceProbe,
    progress: Option<&mpsc::UnboundedSender<PullEvent>>,
) -> Result<(), PullError> {
    let plan = plan_pull(name, archive_root, local_root)?;

    // ── Disk-space precheck (fail fast, copy nothing) ──
    // Use the local manifests root's nearest existing ancestor as the probe
    // target (the local tree may not exist yet on first pull).
    let probe_target = nearest_existing_ancestor(&plan.local_root);
    if let Some(free) = disk_probe.available_bytes(&probe_target) {
        if free < plan.blobs.total_size {
            return Err(PullError::InsufficientDiskSpace {
                need_gb: bytes_to_gb(plan.blobs.total_size),
                have_gb: bytes_to_gb(free),
            });
        }
    }

    if let Some(tx) = progress {
        let _ = tx.send(PullEvent::RetrievingFromArchive {
            model: plan.name.clone(),
            size_gb: bytes_to_gb(plan.blobs.total_size),
        });
    }

    // Snapshot which of the planned LOCAL paths already exist before this pull
    // writes anything. This is the pull side's answer to the same question the
    // eviction side asks: a path that was already here is not ours to delete. A
    // local blob is content-addressed and freely shared between local manifests,
    // so `blobs/sha256-<d>` may be the only copy of a blob a *different* warm
    // model depends on — and Ollama may have it open right now. See
    // [`cleanup_attempt`] for the two preconditions this snapshot needs.
    let planned = planned_local_paths(&plan);
    let pre_existing: HashSet<PathBuf> = planned.iter().filter(|p| p.exists()).cloned().collect();
    // The manifest's path relative to the manifests root, which is all the undo
    // route needs to find the parked previous manifest (the backup name is
    // deterministic, so no state has to survive the copy task to reach cleanup).
    let manifest_rel = plan
        .archive_manifest
        .strip_prefix(plan.archive_root.join("manifests"))
        .map(|r| r.to_path_buf())
        .unwrap_or_default();

    // ── Copy archive → local as an owned, cancellable task ──
    //
    // Not `timeout(copy_future)`: timing out on a future only *drops* it, and a
    // dropped `tokio::fs::copy` keeps running on the blocking pool. The old shape
    // therefore ran its cleanup while the copy was still live and still able to
    // create the very file being cleaned up — into the local Ollama root, which
    // Ollama itself watches. See [`CopyCancel`].
    let cancel: CopyCancel = Arc::new(AtomicBool::new(false));
    // Proves quiescence on the one path where awaiting the task does not: see
    // [`CopyActivity`] and [`cleanup_after_join_error`].
    let activity = CopyActivity::default();
    let mut copy_task = tokio::spawn(copy_model_files(
        plan.clone(),
        cancel.clone(),
        activity.clone(),
    ));

    let copy_res = match tokio::time::timeout(timeout, &mut copy_task).await {
        Ok(joined) => joined,
        Err(_elapsed) => {
            // Ask the copy to stop, then WAIT until it genuinely has before
            // touching the local tree.
            cancel.store(true, Ordering::Relaxed);
            match tokio::time::timeout(COPY_CANCEL_GRACE, &mut copy_task).await {
                // Stopped early: we have the exact list of what it published.
                Ok(Ok(Err((e, published)))) => {
                    let restore_is_safe = !matches!(e, PullError::StashUnsafe(_));
                    undo_pull_attempt(
                        &plan,
                        &manifest_rel,
                        &published,
                        &pre_existing,
                        restore_is_safe,
                    )
                }
                // Finished (or panicked) before it saw the cancel: no exact list,
                // so fall back to every planned local path — minus the ones that
                // were already there.
                Ok(Ok(Ok(()))) | Ok(Err(_)) => {
                    undo_pull_attempt(&plan, &manifest_rel, &planned, &pre_existing, true)
                }
                Err(_) => {
                    // The copy is wedged inside a single chunk (only reachable on
                    // a stalled mount). We must not block the caller on it, so we
                    // return while it is still alive — and therefore **delete
                    // nothing, now or later**. Once we return, the disk-op lock
                    // `ensure_local` holds is released and another pull, an
                    // eviction, or the orphan GC may legitimately create or come to
                    // depend on any of these paths; a deferred cleanup carrying
                    // this attempt's stale snapshot would delete a file that is by
                    // then someone else's. An orphan blob costs disk and is
                    // reconciled by the next pull of this model under the lock; a
                    // wrongly deleted local blob can break a model that is loading.
                    //
                    // Temp-file staging makes the abandoned writer nearly free: it
                    // is confined to its own scratch path, cannot mutate any local
                    // blob, and will not publish once it observes the cancel flag.
                    tracing::warn!(
                        model = %plan.name,
                        grace_secs = COPY_CANCEL_GRACE.as_secs(),
                        temp_infix = COPY_TEMP_INFIX,
                        "archive pull copy did not stop within the cancellation grace period; \
                         NOT cleaning up — ownership of these local paths can no longer be \
                         proven once the disk-op lock is released. The abandoned copy cannot \
                         corrupt or publish anything; it may leave one scratch file behind."
                    );
                }
            }
            return Err(PullError::Timeout(timeout));
        }
    };

    match copy_res {
        Ok(Ok(())) => {
            // The attempt stands: drop the previous manifest we parked.
            commit_manifest_backup(&plan.local_root, &manifest_rel);
            Ok(())
        }
        Ok(Err((e, published))) => {
            // Mid-copy failure: reclaim what this attempt published, through the
            // SAME route (and the same pre-existing filter) as the timeout path.
            // Two cleanup routes with different safety rules is how one of them
            // ends up wrong — there is exactly one route.
            //
            // `StashUnsafe` is the one exception to "always restore": it means
            // `stash_existing_manifest` never reached a known-safe state (a
            // pre-existing backup could not be cleared), so `published` is
            // guaranteed empty and any content at the backup path is unverified
            // — restoring it would resurrect content this attempt never
            // touched. See `PullError::StashUnsafe`'s doc comment.
            let restore_is_safe = !matches!(e, PullError::StashUnsafe(_));
            undo_pull_attempt(
                &plan,
                &manifest_rel,
                &published,
                &pre_existing,
                restore_is_safe,
            );
            Err(e)
        }
        Err(join_err) => {
            // The copy task panicked (or was aborted). Its inner `spawn_blocking`
            // closure is NOT stopped by that, so the flag alone is not enough:
            // cancel, then WAIT for the writer to actually stop before deleting
            // anything, and delete nothing at all if it does not. A live writer
            // that reaches its `rename` after cleanup would republish the manifest
            // over blobs cleanup just removed.
            cleanup_after_join_error(
                PULL_OP,
                &plan.name,
                &cancel,
                &activity,
                COPY_CANCEL_GRACE,
                &planned,
                &pre_existing,
                &plan.local_root,
                &manifest_rel,
            )
            .await;
            Err(PullError::Io(format!(
                "archive pull copy task failed to join: {join_err}"
            )))
        }
    }
}

/// Label used in [`cleanup_attempt`] logs for the cold→warm direction.
const PULL_OP: &str = "archive pull (cold→warm)";

/// The **one** route by which a failed pull undoes its filesystem effects.
///
/// Two things, in this order, and both are needed because they answer different
/// questions:
/// 1. **Restore** the manifest this attempt replaced. Cleanup's `pre_existing`
///    filter reasons about PATHS — "was something here when we started" — and by
///    that test the manifest destination is someone else's and must be kept. But
///    the CONTENT at that path is now this attempt's, so keeping it is precisely
///    wrong: it leaves a manifest Ollama will read naming blobs that step 2 is
///    about to delete. Deleting it instead would discard a model entry this pull
///    never owned. Restoring the previous one is the only outcome that is neither.
/// 2. **Reclaim** the blobs this attempt published, filtered as always.
///
/// When there was no previous manifest the restore is a no-op and the manifest
/// falls out of `pre_existing`, so step 2 removes it as this attempt's own — which
/// is correct for that case and needs no special branch.
///
/// `restore_is_safe` gates step 1 (round 10): a caller passes `false` when the
/// failure that led here is [`PullError::StashUnsafe`] — `stash_existing_manifest`
/// never reached a known-safe state (a pre-existing backup could not be cleared),
/// so any content sitting at the backup path is unverified, and this attempt
/// never published a blob (`candidates` is guaranteed empty in that case). Both
/// steps are skipped in that case: `restore_manifest_backup` cannot tell this
/// untouched, possibly-stale backup apart from one this attempt legitimately
/// created, so trusting `backup.exists()` there would resurrect content this
/// attempt never touched. This is the same "leave everything in place" outcome
/// the not-quiescent JoinError branch already accepts, reached through a
/// different signal.
fn undo_pull_attempt(
    plan: &PullPlan,
    manifest_rel: &Path,
    candidates: &[PathBuf],
    pre_existing: &HashSet<PathBuf>,
    restore_is_safe: bool,
) {
    if !restore_is_safe {
        return;
    }
    let dst_manifest = plan.local_root.join("manifests").join(manifest_rel);
    // If there was a backup to restore and the restore itself failed, we cannot
    // tell whether `dst_manifest` still names this attempt's blobs or the
    // previous ones — reclaiming `candidates` in that state risks deleting data
    // a still-published manifest points at. Skip cleanup entirely rather than
    // guess; the next transfer of this model, under the same lock, reconciles
    // it, exactly as the not-quiescent JoinError branch already accepts for the
    // same reason.
    if !restore_manifest_backup(&plan.local_root, manifest_rel, &dst_manifest) {
        return;
    }
    cleanup_attempt(PULL_OP, &plan.name, candidates, pre_existing);
}

/// Copy the manifest + every referenced blob into the local root.
///
/// On error returns the error plus every local path this call successfully
/// **published** — an exact list, not a guess: each file is staged in a temp file
/// and `rename`d into place, so a path absent from this list was not modified at
/// all and no partial file exists anywhere. Blobs are copied first and the
/// manifest last, so Ollama never sees a manifest whose blobs are missing.
///
/// `cancel` makes the copy stoppable: it is checked between blobs, between 4 MiB
/// chunks within a blob, and once more immediately before publishing — so a
/// cancelled pull never surfaces a blob into the root Ollama watches. Observing
/// cancellation is reported as an error carrying the `published` list, like any
/// other failure.
///
/// Takes an owned [`PullPlan`] so it can be driven as a `tokio::spawn`ed task the
/// caller can await after cancelling.
async fn copy_model_files(
    plan: PullPlan,
    cancel: CopyCancel,
    activity: CopyActivity,
) -> Result<(), (PullError, Vec<PathBuf>)> {
    let mut published: Vec<PathBuf> = Vec::new();

    // Manifest leaf — mirror its path relative to the archive manifests root.
    // Resolved and STASHED before a single blob is copied (see the comment on
    // the stash call below) — everything here through `stash_existing_manifest`
    // depends only on `plan`, never on the blob loop, so there is no ordering
    // cost to moving it first.
    let archive_manifests = plan.archive_root.join("manifests");
    let rel = match plan.archive_manifest.strip_prefix(&archive_manifests) {
        Ok(r) => r.to_path_buf(),
        Err(_) => {
            return Err((
                PullError::Io("archive manifest path outside manifests root".into()),
                published,
            ))
        }
    };
    let dst_manifest = plan.local_root.join("manifests").join(&rel);
    // Park the manifest that is already there, if any, BEFORE replacing it —
    // and before touching a single blob. Publishing with `rename` guarantees
    // the destination is never left partial; it does nothing for the previous
    // CONTENT, which this attempt is about to replace wholesale. Without the
    // stash, an attempt that publishes the manifest and is then abandoned
    // leaves a manifest naming blobs that cleanup deletes — the exact state
    // blobs-first-manifest-last exists to prevent, reached through the
    // ownership filter (the PATH pre-existed, so cleanup keeps it) rather than
    // through truncation. The filter reasons about paths; this is about
    // content.
    //
    // Running this BEFORE the blob loop (not after, as originally written) is
    // itself a correctness requirement, not just an optimization: every
    // failure return in this function funnels into the caller's undo route
    // (`undo_pull_attempt` / `cleanup_after_join_error`), which unconditionally
    // calls `restore_manifest_backup` for `rel` — it has no way to know
    // whether THIS attempt ever reached the stash step. If a blob copy failed
    // before the stash ran, that restore would act on a backup this attempt
    // never created (any pre-existing stale backup, e.g. one orphaned by
    // eviction — see the round-6/7 fixes), resurrecting unrelated content at
    // `dst_manifest` that this attempt never touched. Stashing first means
    // that by the time any failure can be returned, this attempt has always
    // either (a) staged a real backup of `dst`'s actual prior content, which a
    // later restore correctly undoes, or (b) confirmed there was nothing at
    // `dst` to stash, in which case any backup restore finds is stale by the
    // same eviction argument and `stash_existing_manifest` has already
    // cleared it — either way, restore_manifest_backup can no longer observe
    // a backup this attempt did not itself just create.
    //
    // (Round 10 correction to the claim above: (a) and (b) are the only two
    // outcomes of `stash_existing_manifest` returning `Ok`. It can also
    // return `Err` — when it fails to CLEAR a pre-existing stale backup in
    // the first place (round-9 fix; see its doc comment) — and in that one
    // case `backup` can still exist, unverified, when this function returns.
    // `PullError::StashUnsafe` exists specifically to mark that case: every
    // caller (`archive_pull`'s two undo-dispatch sites) checks for it and
    // skips `undo_pull_attempt`/`restore_manifest_backup` entirely rather
    // than let the generic route trust `backup.exists()` the way it safely
    // can for (a)/(b). `published` is guaranteed empty here (no blob has
    // been touched yet), so skipping cleanup as well as skipping restore
    // costs nothing.
    //
    // (Round 11 correction: this call must run BEFORE
    // `create_dir_all(dst_manifest.parent())` below, not after — an ordering
    // bug both round-10 reviewers independently found. `create_dir_all` can
    // itself fail (EACCES, disk quota, a filesystem error) and return
    // `PullError::Io`, a variant `archive_pull` treats as `restore_is_safe =
    // true`. If that create_dir_all failure happened before
    // `stash_existing_manifest` ever ran, any pre-existing stale backup would
    // still be uncleared, and the generic undo route would resurrect it at
    // `dst_manifest` anyway — the exact bug `StashUnsafe` was built to
    // prevent, reached through a sibling error path `StashUnsafe` doesn't
    // cover. `stash_existing_manifest` does not need `dst_manifest`'s parent
    // to exist first: it only calls `dst.exists()` (never creates anything
    // under it) and creates `backup`'s own parent directory independently
    // (see `backup.parent()` handling inside it) — so there is no ordering
    // dependency the other way, and running it first closes this gap.
    if let Err(e) = stash_existing_manifest(&plan.local_root, &rel, &dst_manifest) {
        return Err((
            PullError::StashUnsafe(format!("stash existing manifest: {e}")),
            published,
        ));
    }
    if let Some(parent) = dst_manifest.parent() {
        if let Err(e) = tokio::fs::create_dir_all(parent).await {
            return Err((PullError::Io(e.to_string()), published));
        }
    }

    // Blobs.
    let archive_blobs = plan.archive_root.join("blobs");
    let local_blobs = plan.local_root.join("blobs");
    if let Err(e) = tokio::fs::create_dir_all(&local_blobs).await {
        return Err((PullError::Io(e.to_string()), published));
    }
    for digest in &plan.blobs.digests {
        let fname = blob_filename(digest);
        let src = archive_blobs.join(&fname);
        let dst = local_blobs.join(&fname);
        // Skip blobs already present locally (content-addressed → identical).
        // NOT added to `published`: a pre-existing local blob may be shared with
        // another local manifest and must never be removed by this pull's
        // cleanup. (`cleanup_attempt` enforces that independently; this skip
        // additionally means we never even open it.)
        if dst.exists() {
            continue;
        }
        match copy_file_cancellable(&src, &dst, &cancel, &activity).await {
            Ok(true) => published.push(dst),
            // Cancelled → `dst` was never touched (the copy only ever wrote its
            // own temp file, which it has already removed). The manifest WAS
            // already stashed above; the caller's undo route restores it.
            Ok(false) => return Err((PullError::Io(CANCELLED.to_string()), published)),
            Err(e) => {
                return Err((PullError::Io(format!("copy blob {fname}: {e}")), published));
            }
        }
    }

    match copy_file_cancellable(&plan.archive_manifest, &dst_manifest, &cancel, &activity).await {
        Ok(true) => published.push(dst_manifest),
        // The manifest never got published, so put the previous one straight back
        // and leave the tree exactly as found. The caller's undo route would do
        // this too; doing it here keeps the returned `published` list honest —
        // it never mentions a path this call did not change.
        Ok(false) => {
            restore_manifest_backup(&plan.local_root, &rel, &dst_manifest);
            return Err((PullError::Io(CANCELLED.to_string()), published));
        }
        Err(e) => {
            restore_manifest_backup(&plan.local_root, &rel, &dst_manifest);
            return Err((PullError::Io(format!("copy manifest: {e}")), published));
        }
    }

    Ok(())
}

/// Every local path this pull *would* create (every referenced blob + the
/// manifest). Used as the cleanup candidate list when the precise `published`
/// list is unavailable (the copy task finished or panicked rather than reporting
/// back).
///
/// This is a *superset* fallback, not evidence of what was written — which is
/// exactly why it must never be handed to [`cleanup_partial`] directly. It
/// contains blobs that were already on disk before this pull started and are
/// shared with other local manifests; [`cleanup_attempt`] is the only sanctioned
/// consumer, and it drops those. Because every copy is published atomically, none
/// of these paths can be half-written; the only question cleanup answers is which
/// complete blobs to reclaim.
fn planned_local_paths(plan: &PullPlan) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    let local_blobs = plan.local_root.join("blobs");
    for digest in &plan.blobs.digests {
        paths.push(local_blobs.join(blob_filename(digest)));
    }
    let archive_manifests = plan.archive_root.join("manifests");
    if let Ok(rel) = plan.archive_manifest.strip_prefix(&archive_manifests) {
        paths.push(plan.local_root.join("manifests").join(rel));
    }
    paths
}

/// How long a timed-out transfer waits for its cancelled copy to actually stop
/// before giving up on cleaning up inline.
///
/// Cancellation is checked between 4 MiB chunks, so this is only ever reached
/// when the filesystem itself is wedged (a stalled NFS read/write inside a single
/// chunk). Shared by both directions — it is a property of the cancellation
/// protocol, not of either transfer.
pub(crate) const COPY_CANCEL_GRACE: Duration = Duration::from_secs(30);

/// Error message a copy returns when it stopped because its [`CopyCancel`] flag
/// was set. Only a timeout path sets that flag, and both timeout paths return
/// their own typed `Timeout` error, so this never surfaces to a caller.
pub(crate) const CANCELLED: &str = "archive copy cancelled by timeout";

/// The **one** route by which a failed transfer attempt removes files it may
/// have published — in either direction (`op` names it, for logs only).
///
/// Every path handed here is a *complete* file: [`copy_file_cancellable`] stages
/// into a temp file and publishes with `rename`, so a partial or corrupt file
/// cannot exist. This is therefore purely about reclaiming disk from blobs an
/// abandoned transfer published, never about scrubbing damage.
///
/// Removes every candidate that did **not** exist in `pre_existing` — the
/// snapshot taken, under the caller's `DiskOpLock`, before this attempt published
/// anything. A path that was already there is data this attempt did not create.
/// On the archive side that is a blob shared with another archived model; on the
/// local side it is a blob shared with another *warm* model, which Ollama may
/// have loaded or may load next. Either way it is never removed.
///
/// Two preconditions, both required for the snapshot to be accurate:
/// 1. the copy has genuinely stopped (cancelled and awaited, or finished), and
/// 2. the caller still holds `DiskOpLock`, so nothing else can have created a
///    file since the snapshot.
///
/// Every failure path — mid-copy I/O error, timeout, copy-task panic — goes
/// through here. The one path that cannot satisfy precondition 2 (the
/// cancellation-grace expiry, which returns while the copy is still alive)
/// deletes nothing at all and logs instead.
pub(crate) fn cleanup_attempt(
    op: &str,
    model: &str,
    candidates: &[PathBuf],
    pre_existing: &HashSet<PathBuf>,
) {
    let (to_clean, kept): (Vec<PathBuf>, Vec<PathBuf>) = candidates
        .iter()
        .cloned()
        .partition(|p| !pre_existing.contains(p));
    if !kept.is_empty() {
        tracing::info!(
            op = %op,
            model = %model,
            kept = ?kept.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
            "leaving pre-existing files in place during cleanup of a failed transfer"
        );
    }
    cleanup_partial(&to_clean);
}

/// Directory under a transfer's destination root where a manifest that is about
/// to be replaced is parked, so a failed attempt can put the original back.
///
/// **Deliberately not the manifest's own directory**, even though that is where
/// [`copy_file_cancellable`] stages its temp files. `registry::scan_manifest_tree`
/// treats EVERY file leaf under `manifests/` as a model, with the filename as the
/// tag and no dotfile filter — so a backup parked next to the manifest would be
/// discovered as a phantom model (`park:.1.chord-mfbak`). A copy temp survives
/// only for the duration of one copy; a backup survives a whole failed attempt,
/// which is far too long a window for that.
pub(crate) const MANIFEST_BACKUP_DIR: &str = ".chord-manifest-backup";

/// Where the manifest at `rel` (relative to `<root>/manifests`) is parked while an
/// attempt replaces it.
///
/// The name is **deterministic** — no pid, no sequence — so any cleanup route can
/// find the backup without the attempt having to thread its state through. That is
/// safe because every writer of a given root holds `DiskOpLock`, and pulls for one
/// model are additionally serialised by the per-model lock, so two attempts can
/// never be staging the same manifest at once.
/// Returns `None` for an empty `rel`, which is not a manifest and must never be
/// treated as one: the path would collapse to the backup DIRECTORY itself, and a
/// restore would then `rename` that directory over `<root>/manifests` — clobbering
/// the whole tree. Only reachable if a caller passes a manifest outside its
/// manifests root, which the copy already rejects, so this is a guard against a
/// future caller rather than a live path.
fn manifest_backup_path(root: &Path, rel: &Path) -> Option<PathBuf> {
    if rel.as_os_str().is_empty() {
        return None;
    }
    let flat = rel.to_string_lossy().replace(['/', '\\'], "%");
    Some(root.join(MANIFEST_BACKUP_DIR).join(flat))
}

/// Park a COPY of the manifest currently at `dst` aside, so it can be restored if
/// this attempt fails. No-op (returning `false`) when there is nothing there.
///
/// Deliberately a copy, not a move: [`copy_file_cancellable`] never requires `dst`
/// to be absent — it always stages into its own temp path and only ever touches
/// `dst` with the single atomic `rename` that publishes. So `dst` does not need to
/// be vacated for that to work, and vacating it anyway would open a window, for
/// the whole duration of the manifest copy, where the model has NO manifest on
/// disk at all — any concurrent reader (Ollama, `registry::scan_manifest_tree`)
/// would see it as missing rather than as its previous, still-valid version. A
/// copy keeps `dst` serving the previous content right up to the moment the real
/// publish atomically replaces it, matching the atomic-replacement guarantee the
/// rest of this module relies on everywhere else.
fn stash_existing_manifest(root: &Path, rel: &Path, dst: &Path) -> std::io::Result<bool> {
    let backup = match manifest_backup_path(root, rel) {
        Some(b) => b,
        None => return Ok(false),
    };
    // A backup already sitting at this path cannot belong to THIS attempt: the
    // only way to reach here is at the very start of a fresh manifest copy,
    // after any earlier attempt's own backup was already consumed by its
    // commit or restore. Its presence means an earlier attempt was abandoned
    // without being reconciled (the not-quiescent JoinError branch
    // deliberately leaves everything in place). That content must never be
    // mistaken for THIS attempt's stash — including when `dst` is currently
    // ABSENT: the destination can vanish out from under a stale backup
    // (eviction removes the manifest leaf via `FsLocalEvictor` but has no
    // reason to know about, or touch, this module's backup directory), so
    // "nothing to stash right now" is not evidence the backup is fresh. This
    // clear runs unconditionally, before the `dst.exists()` check below, so
    // that case is covered too. Clearing it first also means: if the rename
    // below fails, it leaves NOTHING at `backup` — the one state
    // `restore_manifest_backup` already treats as safe — rather than stale
    // content a later restore could mistake for this attempt's own.
    // The removal's own result is NOT ignored: if it fails (permissions, a
    // read-only filesystem, another fs error) `backup` may still exist
    // afterward, and this function must not silently report "cleared" —
    // that would let a later `restore_manifest_backup` call trust
    // `backup.exists()` and resurrect content this attempt never verified
    // as stale. Propagating the error here means this attempt's own
    // failure return is what stops it: `copy_model_files` bails before
    // touching a single blob, and this specific model is left exactly as
    // it was found — untouched — rather than proceeding on an unverified
    // clear.
    if backup.exists() {
        if backup.is_dir() {
            std::fs::remove_dir_all(&backup)?;
        } else {
            std::fs::remove_file(&backup)?;
        }
    }
    if !dst.exists() {
        return Ok(false);
    }
    if let Some(parent) = backup.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // Stage-then-rename, same as every publish elsewhere in this module: a copy
    // that fails partway (disk full, permission revoked mid-write) must never
    // leave a PARTIAL file sitting at `backup`. `restore_manifest_backup`'s
    // presence check is the only thing standing between "there is nothing to
    // restore" and "clobber the still-untouched destination with a truncated
    // backup" on the failure path this call's own `Err` returns into — so that
    // check must never be able to see a half-written file.
    let tmp = copy_temp_path(&backup);
    if let Err(e) = std::fs::copy(dst, &tmp) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    if let Err(e) = std::fs::rename(&tmp, &backup) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    Ok(true)
}

/// Put the previous manifest back, undoing a replacement this attempt made.
/// A no-op that reports success when this attempt never replaced one.
///
/// Returns `false` only when a real backup existed and the restore rename
/// itself failed — the one case in which the caller must NOT proceed to
/// reclaim this attempt's blobs, since `dst` may still name them and we have
/// no way to make it stop. Every other outcome (nothing to restore, or a
/// restore that succeeded) returns `true`: it is safe to reclaim.
fn restore_manifest_backup(root: &Path, rel: &Path, dst: &Path) -> bool {
    let backup = match manifest_backup_path(root, rel) {
        Some(b) => b,
        None => return true,
    };
    if !backup.exists() {
        return true;
    }
    if let Some(parent) = dst.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match std::fs::rename(&backup, dst) {
        Ok(()) => true,
        Err(e) => {
            tracing::warn!(
                backup = %backup.display(),
                dst = %dst.display(),
                error = %e,
                "failed to restore the previous manifest after a failed transfer; \
                 leaving it in place and skipping blob cleanup for this attempt — \
                 deleting blobs a manifest may still name is worse than an orphan"
            );
            false
        }
    }
}

/// Accept this attempt's manifest: drop the previous one. Called only once the
/// attempt has actually succeeded.
fn commit_manifest_backup(root: &Path, rel: &Path) {
    let backup = match manifest_backup_path(root, rel) {
        Some(b) => b,
        None => return,
    };
    if backup.exists() {
        if let Err(e) = std::fs::remove_file(&backup) {
            tracing::warn!(
                backup = %backup.display(),
                error = %e,
                "failed to remove the superseded manifest backup"
            );
        }
    }
}

/// Handle a copy task that ended in a [`tokio::task::JoinError`] — a panic in the
/// outer copy task, or an abort.
///
/// A `JoinError` is the one outcome that carries **no evidence about the
/// filesystem**. The outer task unwinding drops the `JoinHandle` of the nested
/// `spawn_blocking` closure, which detaches it: the blocking writer may still be
/// live, mid-file, and able to reach its `rename` *after* cleanup has run. Every
/// other outcome is safe because the outer task returned normally, which means
/// each `copy_file_cancellable` inside it completed its own await.
///
/// Why that matters more on the pull side than the eviction side, and why this is
/// not merely tidy: a late publish into the archive restores a complete,
/// content-addressed blob — harmless. A late publish into the LOCAL root can
/// restore the **manifest**, and the manifest is copied last precisely so it never
/// exists without its blobs. A manifest resurrected after cleanup removed those
/// blobs is a model entry Ollama will read and act on, pointing at files that are
/// gone. Blobs-first-manifest-last is defeated by exactly one late `rename`.
///
/// So: cancel, then *prove* the writer stopped before deleting anything.
/// - Quiescent within `grace`, and any stashed manifest restores cleanly (or
///   there was nothing to restore) → cleanup is provably safe; run the normal
///   filtered [`cleanup_attempt`] and return `true`.
/// - Not quiescent, OR quiescent but the manifest restore itself failed → we
///   cannot prove ownership of a single path (or cannot prove `dst_manifest` no
///   longer names this attempt's blobs), so **delete nothing** and log the
///   candidates, exactly as the cancellation-grace expiry does. An orphan blob
///   or a stale scratch file costs disk and is reclaimed by the next transfer of
///   this model under the lock; a resurrected or still-published manifest is a
///   broken model entry that something else acts on.
///
/// `grace` is a parameter rather than [`COPY_CANCEL_GRACE`] so the not-quiescent
/// branch is reachable in a test without a 30 s wait.
///
/// `root`/`manifest_rel` identify the manifest this attempt may have stashed
/// (see [`stash_existing_manifest`]) — once the writer is proven quiescent we are
/// in exactly the same position as [`undo_pull_attempt`]'s other two callers, and
/// must restore it for the same reason: `pre_existing` reasons about paths, not
/// content, so without an explicit restore a manifest this attempt stashed would
/// stay orphaned in the backup dir while `cleanup_attempt` deletes the blobs a
/// resurrected copy would still name.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn cleanup_after_join_error(
    op: &str,
    model: &str,
    cancel: &CopyCancel,
    activity: &CopyActivity,
    grace: Duration,
    candidates: &[PathBuf],
    pre_existing: &HashSet<PathBuf>,
    root: &Path,
    manifest_rel: &Path,
) -> bool {
    // Ask first, so the writer is already stopping while we wait — and so that if
    // it is between its final cancel check and its `rename`, that is the only
    // window left rather than the whole remainder of the file.
    cancel.store(true, Ordering::Relaxed);
    if activity.wait_quiescent(grace).await {
        // Same order as `undo_pull_attempt`: restore before reclaim, so a manifest
        // this attempt stashed is never left orphaned in the backup dir while its
        // blobs are removed out from under it. And same rule as `undo_pull_attempt`:
        // if a real backup existed and the restore rename itself failed, we cannot
        // tell whether `dst_manifest` still names this attempt's blobs — skip
        // cleanup rather than guess. The writer being quiescent only tells us it is
        // safe to look; it says nothing about whether the restore itself succeeded.
        if !manifest_rel.as_os_str().is_empty() {
            let dst_manifest = root.join("manifests").join(manifest_rel);
            if !restore_manifest_backup(root, manifest_rel, &dst_manifest) {
                tracing::warn!(
                    op = %op,
                    model = %model,
                    "manifest restore failed after a JoinError; skipping blob cleanup \
                     rather than risk deleting data a still-published manifest names"
                );
                return false;
            }
        }
        cleanup_attempt(op, model, candidates, pre_existing);
        return true;
    }
    tracing::warn!(
        op = %op,
        model = %model,
        grace_secs = grace.as_secs(),
        temp_infix = COPY_TEMP_INFIX,
        candidates = ?candidates.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
        "copy task ended in a JoinError and its blocking writer did not stop within \
         the grace period; NOT cleaning up — a still-live writer can publish after \
         cleanup, and a manifest republished over removed blobs is worse than an \
         orphan blob. These paths are reconciled by the next transfer of this model \
         under the disk-op lock."
    );
    false
}

/// Remove files left by a failed/timed-out transfer (best-effort).
///
/// **Not a public cleanup entry point**: it applies no ownership filter, so
/// handing it a planned-path list would delete files this transfer never created.
/// Call [`cleanup_attempt`] instead — it is the single filtered route both
/// directions use.
pub(crate) fn cleanup_partial(paths: &[PathBuf]) {
    for p in paths {
        if p.exists() {
            if let Err(e) = std::fs::remove_file(p) {
                tracing::warn!(path = %p.display(), error = %e, "failed to clean up partial pull file");
            }
        }
    }
}

/// Cooperative cancellation flag for an in-flight [`copy_file_cancellable`].
///
/// `tokio::fs::*` operations are **not cancellable**: they run `std::fs` calls on
/// the blocking pool via `spawn_blocking`, and dropping the future (which is all
/// `tokio::time::timeout` does on expiry) detaches the `JoinHandle` without
/// stopping the blocking work. The copy keeps running — and keeps *creating and
/// writing its destination file* — after the timeout branch has already returned
/// and cleaned up. Cleanup then races the still-live writer, and when the
/// blocking task is dispatched late (a busy blocking pool / a loaded box) the
/// writer wins and leaves the very file cleanup was supposed to remove.
///
/// A copy driven through this flag is stoppable: the caller sets it, then
/// **awaits the copy task** so any cleanup is ordered strictly after the last
/// write.
pub(crate) type CopyCancel = Arc<std::sync::atomic::AtomicBool>;

/// Tracks how many blocking copy closures are still touching the filesystem.
///
/// A [`CopyCancel`] flag asks a copy to stop; this counter is what proves it
/// *has*. The two are not interchangeable, and the difference is the whole reason
/// this type exists: a `JoinError` from the outer copy task — a panic, or an
/// abort — tells you only that the OUTER future is finished. The nested
/// `spawn_blocking` closure is not stopped by the outer task unwinding; unwinding
/// drops its `JoinHandle`, which *detaches* the closure and leaves it running.
/// So "the task I awaited is done" is not evidence that nothing is writing.
///
/// Awaiting the copy task to a normal `Ok`/`Err` return IS such evidence, because
/// every `copy_file_cancellable` call inside it completed its own `.await` on the
/// blocking handle. Only the `JoinError` path lacks it, and that is the one place
/// this counter is consulted.
///
/// The counter is incremented before the closure is spawned (never after, so
/// there is no window where a closure is pending but uncounted) and decremented
/// by a guard the closure owns — so it drops on the normal path, on a panic
/// inside the closure, and even if the runtime discards the closure unrun.
#[derive(Clone, Default)]
pub(crate) struct CopyActivity(Arc<std::sync::atomic::AtomicUsize>);

/// Decrements a [`CopyActivity`] when the blocking closure holding it ends, by
/// any route.
struct ActivityGuard(CopyActivity);

impl Drop for ActivityGuard {
    fn drop(&mut self) {
        self.0 .0.fetch_sub(1, Ordering::SeqCst);
    }
}

impl CopyActivity {
    /// How many blocking copy closures are still live. Non-zero means at least
    /// one writer may still touch the filesystem.
    pub(crate) fn in_flight(&self) -> usize {
        self.0.load(Ordering::SeqCst)
    }

    /// Test-only: take the same count a live blocking closure holds, released
    /// when the returned guard drops. Lets a test put the counter in the
    /// "a writer is live" state without depending on where a real copy has got
    /// to — which is not observable without a hook, and not race-free to guess.
    #[cfg(test)]
    fn hold_for_test(&self) -> ActivityGuard {
        self.0.fetch_add(1, Ordering::SeqCst);
        ActivityGuard(self.clone())
    }

    /// Wait until no blocking copy closure is live, bounded by `grace`.
    ///
    /// Returns `true` only when quiescence was actually observed. A `false`
    /// return is not "probably fine" — it is the caller's signal that it cannot
    /// prove anything about the filesystem and must not delete.
    pub(crate) async fn wait_quiescent(&self, grace: Duration) -> bool {
        const POLL: Duration = Duration::from_millis(10);
        let deadline = tokio::time::Instant::now() + grace;
        loop {
            if self.in_flight() == 0 {
                return true;
            }
            if tokio::time::Instant::now() >= deadline {
                return false;
            }
            tokio::time::sleep(POLL).await;
        }
    }
}

/// Chunk size for [`copy_file_cancellable`]. Cancellation is observed at most one
/// chunk late, which bounds how long a cancelled copy can keep touching the
/// filesystem (rather than "until this multi-GB blob finishes", which is what an
/// uncancellable `std::fs::copy` costs).
const COPY_CHUNK_BYTES: usize = 4 * 1024 * 1024;

/// Filename infix marking a [`copy_file_cancellable`] scratch file. Temp files
/// are `.<final-name>.<COPY_TEMP_INFIX>.<pid>-<seq>` in the destination's own
/// directory, so an abandoned one (only possible if the process dies mid-copy)
/// is identifiable by name and reapable, and can never be mistaken for a blob.
pub(crate) const COPY_TEMP_INFIX: &str = "chord-copytmp";

/// Per-process counter making temp names unique across concurrent copies.
static COPY_TEMP_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Scratch path used to stage a copy of `dst` before publishing it.
fn copy_temp_path(dst: &Path) -> PathBuf {
    let seq = COPY_TEMP_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let name = dst
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unnamed".to_string());
    let tmp = format!(".{name}.{COPY_TEMP_INFIX}.{}-{seq}", std::process::id());
    dst.parent().unwrap_or(Path::new(".")).join(tmp)
}

/// Copy `src` → `dst` on the blocking pool in chunks, checking `cancel` between
/// chunks, and **publish atomically**.
///
/// The copy is staged into a per-attempt temp file in `dst`'s own directory and
/// `rename`d onto `dst` only after it is complete and fsynced. This is what makes
/// the copy safe to abandon:
///
/// - **`dst` is never mutated in place.** An existing `dst` — which may be a blob
///   shared with another archived model — keeps its exact bytes unless this call
///   fully succeeds. Truncate-then-overwrite (`File::create` on `dst`) could
///   corrupt a shared blob on any cancel/error path, and a corrupted file is
///   worse than a missing one: it looks intact until something reads it. No
///   filter over "which paths may I delete" can see that class of damage,
///   because the damage is a mutation, not a deletion.
/// - **An abandoned copy is harmless.** A writer that outlives its caller (a
///   cancellation grace expiry, or a panic in the outer task, both of which
///   detach this `spawn_blocking` closure without stopping it) writes only into
///   its own temp path. Nothing else stats or trusts that path, so no other
///   operation can size-match it, read it, or come to depend on it.
/// - **Cleanup is trivially ownable.** The only file this call leaves behind on a
///   non-success path is its own temp file, which it removes itself.
///
/// Returns `Ok(true)` when `dst` was published, `Ok(false)` when the copy observed
/// cancellation and stopped early. In every non-`Ok(true)` case `dst` is exactly
/// as it was and no temp file remains.
pub(crate) async fn copy_file_cancellable(
    src: &Path,
    dst: &Path,
    cancel: &CopyCancel,
    activity: &CopyActivity,
) -> std::io::Result<bool> {
    use std::io::{Read, Write};

    let (src, dst, cancel) = (src.to_path_buf(), dst.to_path_buf(), cancel.clone());
    // Count the closure BEFORE spawning it: a caller that observes zero must be
    // able to conclude nothing is pending, and an increment inside the closure
    // would leave a window where it is queued but invisible.
    activity.0.fetch_add(1, Ordering::SeqCst);
    let guard = ActivityGuard(activity.clone());
    tokio::task::spawn_blocking(move || {
        // Dropped when this closure ends by ANY route — normal return, panic, or
        // the runtime discarding it unrun.
        let _guard = guard;
        if cancel.load(Ordering::Relaxed) {
            return Ok(false);
        }
        let mut reader = std::fs::File::open(&src)?;
        let perms = reader.metadata().ok().map(|m| m.permissions());
        let tmp = copy_temp_path(&dst);

        // Everything from here on must leave `tmp` removed unless we publish it.
        let staged = (|| -> std::io::Result<bool> {
            let mut writer = std::fs::File::create(&tmp)?;
            let mut buf = vec![0u8; COPY_CHUNK_BYTES];
            loop {
                if cancel.load(Ordering::Relaxed) {
                    return Ok(false);
                }
                let n = reader.read(&mut buf)?;
                if n == 0 {
                    break;
                }
                writer.write_all(&buf[..n])?;
            }
            if let Some(perms) = perms {
                // Best-effort: match `std::fs::copy`, which carries source perms.
                let _ = writer.set_permissions(perms);
            }
            // Durable before publish: a reader that sees `dst` must see its bytes.
            writer.sync_all()?;
            // Last cancel check — a cancelled copy must never publish, even if it
            // finished staging, so an abandoned writer cannot surface a blob the
            // caller has already reported as timed out.
            if cancel.load(Ordering::Relaxed) {
                return Ok(false);
            }
            std::fs::rename(&tmp, &dst)?;
            Ok(true)
        })();

        if !matches!(staged, Ok(true)) {
            let _ = std::fs::remove_file(&tmp);
        }
        staged
    })
    .await
    .unwrap_or_else(|e| {
        Err(std::io::Error::other(format!(
            "copy task failed to join: {e}"
        )))
    })
}

/// Walk up from `path` until an existing directory is found (for the disk probe,
/// since the local tree may not exist before the first pull). Falls back to `/`.
/// Shared with the eviction module so the sweep and pre-pull path anchor disk
/// pressure on the same directory.
pub(crate) fn nearest_existing_ancestor(path: &Path) -> PathBuf {
    let mut cur = path;
    loop {
        if cur.exists() {
            return cur.to_path_buf();
        }
        match cur.parent() {
            Some(p) => cur = p,
            None => return PathBuf::from("/"),
        }
    }
}

// ── PullCoordinator: per-model dedup + registry integration ────────────────────

/// Coordinates archive pulls so concurrent requests for the same cold model
/// don't double-copy, and exposes the [`ensure_local`](PullCoordinator::ensure_local)
/// entry point used at the model-load boundary.
///
/// Cloneable and cheap to share (everything is behind `Arc`).
#[derive(Clone)]
pub struct PullCoordinator {
    /// Shared registry (read for tier/paths, written for promote + timestamps).
    registry: Arc<Mutex<ModelRegistry>>,
    /// Per-model pull locks. The outer mutex guards the map; each inner mutex
    /// serialises pulls for one model name.
    locks: Arc<Mutex<HashMap<String, Arc<Mutex<()>>>>>,
    /// Max copy duration before abort + cleanup (`MODEL_PULL_TIMEOUT_SECS`).
    timeout: Duration,
    /// Disk-space probe (real `statvfs` in prod, injectable in tests).
    disk_probe: Arc<dyn DiskSpaceProbe>,
    /// TIER-03 pre-pull eviction hooks (optional). When present, a cold pull that
    /// would fail the disk precheck first evicts LRU warm models to make room.
    /// `None` disables pre-pull eviction (the pull just fails on insufficient
    /// space, preserving the original TIER-02 behaviour — used by TIER-02 tests).
    evictor: Option<Arc<dyn super::eviction::LocalEvictor>>,
    /// Shared disk-operation lock so a pre-pull eviction and a background sweep
    /// never interleave destructive filesystem ops. Set alongside `evictor`.
    disk_op_lock: Option<super::eviction::DiskOpLock>,
    /// MSM-02: max duration for a single warm→cold copy during pre-pull
    /// eviction (`MODEL_ARCHIVE_COPY_TIMEOUT_SECS`). Set alongside `evictor`;
    /// unused when pre-pull eviction is disabled.
    archive_copy_timeout: Duration,
}

impl PullCoordinator {
    /// Build a coordinator over a shared registry with the configured timeout
    /// and the real [`StatvfsProbe`].
    pub fn new(registry: Arc<Mutex<ModelRegistry>>, timeout: Duration) -> Self {
        Self::with_probe(registry, timeout, Arc::new(StatvfsProbe))
    }

    /// Build with an injected disk-space probe (tests).
    pub fn with_probe(
        registry: Arc<Mutex<ModelRegistry>>,
        timeout: Duration,
        disk_probe: Arc<dyn DiskSpaceProbe>,
    ) -> Self {
        Self {
            registry,
            locks: Arc::new(Mutex::new(HashMap::new())),
            timeout,
            disk_probe,
            evictor: None,
            disk_op_lock: None,
            archive_copy_timeout: Duration::from_secs(1800),
        }
    }

    /// Enable TIER-03 pre-pull eviction: before copying a cold model, if the disk
    /// precheck shows insufficient space, evict LRU warm non-protected models
    /// (sharing `disk_op_lock` with the background sweep) until there's room.
    /// Returns `self` for builder-style wiring in `main.rs`.
    pub fn with_eviction(
        mut self,
        evictor: Arc<dyn super::eviction::LocalEvictor>,
        disk_op_lock: super::eviction::DiskOpLock,
    ) -> Self {
        self.evictor = Some(evictor);
        self.disk_op_lock = Some(disk_op_lock);
        self
    }

    /// MSM-02: override the warm→cold copy timeout used by pre-pull eviction
    /// (default 1800s if never called). Reads `MODEL_ARCHIVE_COPY_TIMEOUT_SECS`
    /// via `Config` in `main.rs`.
    pub fn with_archive_copy_timeout(mut self, timeout: Duration) -> Self {
        self.archive_copy_timeout = timeout;
        self
    }

    /// Get-or-insert the per-model lock.
    async fn model_lock(&self, name: &str) -> Arc<Mutex<()>> {
        let mut map = self.locks.lock().await;
        map.entry(name.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }

    /// Ensure `model` is present on local disk (at least warm), pulling it from
    /// the archive if it is cold. This is the single integration point wired into
    /// the model-load path (see `lib.rs` / the load boundary).
    ///
    /// Behaviour by tier (records `last_requested` for every known model):
    /// - **Hot**  → no-op (already resident).
    /// - **Warm** → no-op (the VRAM load is the existing lifecycle's job).
    /// - **Cold** → archive-pull, then promote to warm.
    /// - **unknown** → [`PullError::UnknownModel`].
    ///
    /// Concurrent calls for the same cold model are deduped via the per-model
    /// lock: the second caller awaits the first, then sees the model warm and
    /// returns without a second copy.
    pub async fn ensure_local(
        &self,
        model: &str,
        progress: Option<&mpsc::UnboundedSender<PullEvent>>,
    ) -> Result<(), PullError> {
        // Snapshot tier + paths under a short-lived lock; record the request.
        let (tier, archive_root, local_root) = {
            let mut reg = self.registry.lock().await;
            reg.update_last_requested(model);
            match reg.get(model) {
                None => return Err(PullError::UnknownModel(model.to_string())),
                Some(rec) => (
                    rec.tier.clone(),
                    rec.archive_path
                        .clone()
                        .map(PathBuf::from)
                        .unwrap_or_else(|| reg.archive_path().to_path_buf()),
                    reg.local_path().to_path_buf(),
                ),
            }
        };

        match tier {
            StorageTier::Hot | StorageTier::Warm => Ok(()),
            StorageTier::Cold => {
                // Dedup: serialise pulls for this model name.
                let lock = self.model_lock(model).await;
                let _guard = lock.lock().await;

                // Re-check under the lock: a concurrent pull may have warmed it.
                {
                    let reg = self.registry.lock().await;
                    if let Some(rec) = reg.get(model) {
                        if rec.tier != StorageTier::Cold {
                            return Ok(());
                        }
                    }
                }

                // ── TIER-03 pre-pull eviction ──
                // If eviction is wired and the incoming model won't fit, evict LRU
                // warm models first to make room (sharing the disk-op lock with the
                // background sweep). We then fall through to archive_pull, whose own
                // precheck still surfaces InsufficientDiskSpace if eviction couldn't
                // free enough — keeping the existing error path intact.
                if let (Some(evictor), Some(lock)) = (&self.evictor, &self.disk_op_lock) {
                    // Size the incoming model from its archive manifest.
                    if let Some(leaf) = find_manifest_leaf(&archive_root, model) {
                        let need = parse_manifest_blobs(&leaf).total_size;
                        let probe_target = nearest_existing_ancestor(&local_root);
                        let short = self
                            .disk_probe
                            .available_bytes(&probe_target)
                            .map(|free| free < need)
                            .unwrap_or(false);
                        if short {
                            super::eviction::evict_for_space(
                                &self.registry,
                                need,
                                &local_root,
                                self.disk_probe.as_ref(),
                                evictor.as_ref(),
                                lock,
                                self.archive_copy_timeout,
                            )
                            .await;
                        }
                    }
                }

                // ── B1 fix: hold the shared disk-op lock across the copy ──
                // `copy_model_files` writes each blob to its FINAL
                // `local/blobs/sha256-<d>` path and the referencing manifest
                // LAST, so mid-pull a just-copied blob is briefly on disk with
                // no local manifest referencing it — exactly what the orphan-GC
                // (and eviction) treat as a deletable orphan. Acquire
                // `disk_op_lock` here so the pull-copy phase is serialised with
                // GC/eviction (the lock's own contract), preventing GC from
                // deleting a blob that is being pulled for a live on-demand load.
                // Acquired AFTER the pre-pull `evict_for_space` above (which
                // takes the same lock internally) so we never self-deadlock the
                // non-reentrant async mutex.
                let _disk_guard = match &self.disk_op_lock {
                    Some(l) => Some(l.lock().await),
                    None => None,
                };

                archive_pull(
                    model,
                    &archive_root,
                    &local_root,
                    self.timeout,
                    self.disk_probe.as_ref(),
                    progress,
                )
                .await?;

                // Promote cold → warm (the warm → hot VRAM load is handled by
                // the existing lifecycle, which calls set_tier(Hot)). Still under
                // `_disk_guard` so the promote+persist can't race a GC either.
                let local_str = local_root.to_string_lossy().to_string();
                let mut reg = self.registry.lock().await;
                reg.promote_to_warm(model, &local_str);
                // Persist the cold→warm transition so the on-disk registry and the
                // control API reflect reality without waiting for the next restart.
                if let Err(e) = reg.save() {
                    tracing::warn!("failed to persist registry after pull of {model}: {e}");
                }
                Ok(())
            }
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::registry::ModelRegistry;
    use std::fs;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use tempfile::tempdir;

    #[tokio::test]
    async fn cancellable_copy_copies_in_full_when_not_cancelled() {
        let tmp = tempdir().unwrap();
        let src = tmp.path().join("src.bin");
        let dst = tmp.path().join("dst.bin");
        // Larger than one chunk, so the cancel check runs mid-copy on the happy path.
        let payload: Vec<u8> = (0..(COPY_CHUNK_BYTES * 2 + 12345)).map(|i| i as u8).collect();
        fs::write(&src, &payload).unwrap();

        // A pre-existing destination is replaced atomically, not overwritten.
        fs::write(&dst, vec![b'Z'; 128]).unwrap();

        let cancel: CopyCancel = Arc::new(AtomicBool::new(false));
        let completed = copy_file_cancellable(&src, &dst, &cancel, &CopyActivity::default())
            .await
            .unwrap();

        assert!(completed, "an uncancelled copy must report completion");
        assert_eq!(fs::read(&dst).unwrap(), payload, "copy must be byte-identical");
        assert!(
            leftover_temps(tmp.path()).is_empty(),
            "a published copy must leave no scratch file behind"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancelled_midcopy_never_mutates_an_existing_destination() {
        // The finding this guards: writing straight into `dst` with a truncating
        // `File::create` destroys a pre-existing blob the instant the copy starts,
        // and every later safety rule then *preserves* the damage — the
        // `pre_existing` filter deliberately declines to delete it, because it is
        // not ours. Deletion filters cannot see mutation.
        //
        // Unlike the 1 ns eviction timeout (which cancels the copy before it ever
        // opens a file), this cancels the copy while it is genuinely mid-flight.
        //
        // The source is a FIFO, so the TEST controls exactly how far the copy gets
        // rather than racing it: the copy cannot advance past what we feed it.
        // (A wall-clock/observer race here is precisely the class of flake this
        // whole change exists to remove — it must not be reintroduced in a test.)
        use std::io::Write as _;
        let tmp = tempdir().unwrap();
        let src = tmp.path().join("src.fifo");
        let dst = tmp.path().join("dst.bin");
        match std::process::Command::new("mkfifo").arg(&src).status() {
            Ok(s) if s.success() => {}
            _ => return, // no mkfifo on this platform — nothing to assert
        }
        let victim: Vec<u8> = vec![b'Z'; 4096];
        fs::write(&dst, &victim).unwrap();

        let cancel: CopyCancel = Arc::new(AtomicBool::new(false));
        let copy = tokio::spawn({
            let (src, dst, cancel) = (src.clone(), dst.clone(), cancel.clone());
            async move { copy_file_cancellable(&src, &dst, &cancel, &CopyActivity::default()).await }
        });

        // Opening the write end releases the copy's blocking `File::open`.
        let mut w = std::fs::OpenOptions::new().write(true).open(&src).unwrap();
        // A full chunk. The pipe buffer is far smaller than this, so `write_all`
        // cannot return until the copy has consumed most of it — i.e. it has
        // provably created its scratch file and written into it. THIS is the
        // mid-flight moment, established by the pipe rather than by timing.
        w.write_all(&vec![9u8; COPY_CHUNK_BYTES]).unwrap();
        w.flush().unwrap();
        cancel.store(true, Ordering::Relaxed);
        // Unblock a reader that may be parked in `read`, then EOF. Whichever
        // interleaving occurs, the copy stops without publishing: either the
        // loop's cancel check fires, or it reaches EOF and the final pre-publish
        // cancel check does.
        let _ = w.write_all(&[9u8]);
        drop(w);

        let completed = copy.await.unwrap().unwrap();

        assert!(!completed, "a copy cancelled mid-flight must not report completion");
        assert_eq!(
            fs::read(&dst).unwrap(),
            victim,
            "an existing destination must be byte-identical after a mid-flight cancel \
             — it must never be truncated or partially overwritten in place"
        );
        assert!(
            leftover_temps(tmp.path()).is_empty(),
            "a cancelled copy must remove its own scratch file"
        );
    }

    /// Any `COPY_TEMP_INFIX` scratch files directly under `dir`.
    fn leftover_temps(dir: &Path) -> Vec<String> {
        fs::read_dir(dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains(COPY_TEMP_INFIX))
            .collect()
    }

    #[tokio::test]
    async fn cancelled_copy_stops_and_has_finished_when_it_resolves() {
        // The whole point of `CopyCancel`: unlike a dropped `tokio::fs::copy`
        // future (which leaves an uncancellable `std::fs::copy` running on the
        // blocking pool), a cancelled copy has *genuinely stopped* by the time
        // its future resolves — and, staging into a temp file, it leaves the
        // destination exactly as it found it.
        let tmp = tempdir().unwrap();
        let src = tmp.path().join("src.bin");
        let dst = tmp.path().join("dst.bin");
        let absent = tmp.path().join("absent.bin");
        fs::write(&src, vec![7u8; COPY_CHUNK_BYTES * 2]).unwrap();
        let existing: Vec<u8> = vec![b'Z'; 4096];
        fs::write(&dst, &existing).unwrap();

        let cancel: CopyCancel = Arc::new(AtomicBool::new(true));

        // Over an existing destination: untouched, not truncated.
        let completed = copy_file_cancellable(&src, &dst, &cancel, &CopyActivity::default())
            .await
            .unwrap();
        assert!(!completed, "a cancelled copy must not report completion");
        assert_eq!(
            fs::read(&dst).unwrap(),
            existing,
            "a cancelled copy must leave an existing destination byte-identical"
        );

        // Over an absent destination: still absent — it never publishes.
        let completed = copy_file_cancellable(&src, &absent, &cancel, &CopyActivity::default())
            .await
            .unwrap();
        assert!(!completed);
        assert!(!absent.exists(), "a cancelled copy must not publish anything");

        assert!(
            leftover_temps(tmp.path()).is_empty(),
            "a cancelled copy must remove its own scratch file"
        );
    }

    /// Write a manifest + its referenced blob files under `root`, returning the
    /// model name. Blob digests are derived from `model`+index so distinct models
    /// don't collide. Each blob file is `size` bytes of filler.
    fn make_model(root: &Path, model: &str, tag: &str, blob_sizes: &[u64]) -> String {
        let manifests = root
            .join("manifests")
            .join("registry.ollama.ai")
            .join("library")
            .join(model);
        fs::create_dir_all(&manifests).unwrap();
        let blobs_dir = root.join("blobs");
        fs::create_dir_all(&blobs_dir).unwrap();

        let mut layers = Vec::new();
        for (i, size) in blob_sizes.iter().enumerate() {
            let digest = format!("sha256:{model}{i}");
            let fname = digest.replacen(':', "-", 1);
            fs::write(blobs_dir.join(&fname), vec![b'x'; *size as usize]).unwrap();
            layers.push(serde_json::json!({ "size": size, "digest": digest }));
        }
        // config blob
        let cfg_digest = format!("sha256:{model}cfg");
        fs::write(blobs_dir.join(cfg_digest.replacen(':', "-", 1)), b"cfg").unwrap();
        let body = serde_json::json!({
            "config": { "size": 3, "digest": cfg_digest },
            "layers": layers,
        });
        fs::write(manifests.join(tag), serde_json::to_string(&body).unwrap()).unwrap();
        format!("{model}:{tag}")
    }

    /// Probe that always reports a fixed number of free bytes.
    struct FixedProbe(u64);
    impl DiskSpaceProbe for FixedProbe {
        fn available_bytes(&self, _: &Path) -> Option<u64> {
            Some(self.0)
        }
    }

    /// Probe that returns `None` (unknown → "assume enough").
    struct UnknownProbe;
    impl DiskSpaceProbe for UnknownProbe {
        fn available_bytes(&self, _: &Path) -> Option<u64> {
            None
        }
    }

    /// Probe that delays before reporting, to widen the race window in the
    /// concurrent-pull test, and counts invocations.
    struct CountingSlowProbe(Arc<AtomicUsize>);
    impl DiskSpaceProbe for CountingSlowProbe {
        fn available_bytes(&self, _: &Path) -> Option<u64> {
            self.0.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(50));
            Some(u64::MAX)
        }
    }

    fn reg_with(base: &Path, protected: Vec<String>) -> ModelRegistry {
        ModelRegistry::new(
            base.join("registry.json"),
            base.join("local"),
            base.join("archive"),
            protected,
        )
    }

    #[tokio::test]
    async fn cold_model_with_valid_archive_pulls_and_promotes_to_warm() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        let model = make_model(&base.join("archive"), "cold", "1", &[100, 200]);

        let mut reg = reg_with(base, vec![]);
        reg.reconcile();
        assert_eq!(reg.get(&model).unwrap().tier, StorageTier::Cold);

        let registry = Arc::new(Mutex::new(reg));
        let coord = PullCoordinator::with_probe(
            registry.clone(),
            Duration::from_secs(30),
            Arc::new(FixedProbe(u64::MAX)),
        );
        coord.ensure_local(&model, None).await.unwrap();

        // Files copied to local.
        let local = base.join("local");
        assert!(local
            .join("manifests/registry.ollama.ai/library/cold/1")
            .is_file());
        assert!(local.join("blobs/sha256-cold0").is_file());
        assert!(local.join("blobs/sha256-cold1").is_file());
        assert!(local.join("blobs/sha256-coldcfg").is_file());

        // Registry promoted to warm.
        let reg = registry.lock().await;
        assert_eq!(reg.get(&model).unwrap().tier, StorageTier::Warm);
        assert!(reg.get(&model).unwrap().last_requested.is_some());
    }

    #[tokio::test]
    async fn cold_model_missing_archive_errors_clearly() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        // Register a cold model whose archive manifest does not actually exist.
        let mut reg = reg_with(base, vec![]);
        reg.reconcile();
        // Manually inject a cold record with no real archive file.
        let registry = Arc::new(Mutex::new(reg));
        {
            // Inject via a fresh pull plan path: create archive dir but no manifest.
            fs::create_dir_all(base.join("archive").join("manifests")).unwrap();
        }
        let coord = PullCoordinator::with_probe(
            registry.clone(),
            Duration::from_secs(5),
            Arc::new(FixedProbe(u64::MAX)),
        );
        // Unknown to registry → UnknownModel; this verifies the unknown path too.
        let err = coord.ensure_local("ghost:1", None).await.unwrap_err();
        assert!(matches!(err, PullError::UnknownModel(_)), "got {err:?}");

        // Now a registered-cold-but-archive-missing case via direct archive_pull.
        let err = archive_pull(
            "ghost:1",
            &base.join("archive"),
            &base.join("local"),
            Duration::from_secs(5),
            &FixedProbe(u64::MAX),
            None,
        )
        .await
        .unwrap_err();
        assert!(matches!(err, PullError::MissingArchive(_)), "got {err:?}");
    }

    #[tokio::test]
    async fn warm_model_does_not_copy_from_archive() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        // Present locally → warm.
        make_model(&base.join("local"), "warm", "1", &[10]);
        let mut reg = reg_with(base, vec![]);
        reg.reconcile();
        assert_eq!(reg.get("warm:1").unwrap().tier, StorageTier::Warm);

        let registry = Arc::new(Mutex::new(reg));
        let coord = PullCoordinator::with_probe(
            registry.clone(),
            Duration::from_secs(5),
            Arc::new(FixedProbe(0)), // would fail the disk check IF a pull happened
        );
        // No archive copy → no disk check → succeeds even with 0 free bytes.
        coord.ensure_local("warm:1", None).await.unwrap();
        assert_eq!(registry.lock().await.get("warm:1").unwrap().tier, StorageTier::Warm);
    }

    #[tokio::test]
    async fn hot_model_unchanged() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        make_model(&base.join("local"), "hot", "1", &[10]);
        let mut reg = reg_with(base, vec![]);
        reg.reconcile();
        reg.set_tier("hot:1", StorageTier::Hot);

        let registry = Arc::new(Mutex::new(reg));
        let coord = PullCoordinator::with_probe(
            registry.clone(),
            Duration::from_secs(5),
            Arc::new(FixedProbe(0)),
        );
        coord.ensure_local("hot:1", None).await.unwrap();
        assert_eq!(registry.lock().await.get("hot:1").unwrap().tier, StorageTier::Hot);
    }

    #[tokio::test]
    async fn insufficient_disk_space_errors_and_copies_nothing() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        make_model(&base.join("archive"), "big", "1", &[1000, 2000]);

        // Need = 1000+2000+3(cfg) = 3003 bytes; have only 10.
        let err = archive_pull(
            "big:1",
            &base.join("archive"),
            &base.join("local"),
            Duration::from_secs(5),
            &FixedProbe(10),
            None,
        )
        .await
        .unwrap_err();
        assert!(matches!(err, PullError::InsufficientDiskSpace { .. }), "got {err:?}");
        // Nothing copied.
        assert!(!base.join("local").join("blobs").exists());
    }

    #[tokio::test]
    async fn timeout_errors_and_cleans_up_partial_files() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        // A large blob so the copy can't finish within a ~1ms timeout.
        make_model(&base.join("archive"), "slow", "1", &[8 * 1024 * 1024]);

        let err = archive_pull(
            "slow:1",
            &base.join("archive"),
            &base.join("local"),
            Duration::from_nanos(1), // expire immediately
            &FixedProbe(u64::MAX),
            None,
        )
        .await
        .unwrap_err();
        assert!(matches!(err, PullError::Timeout(_)), "got {err:?}");

        // No partial blob or manifest left behind.
        let local = base.join("local");
        let blob = local.join("blobs/sha256-slow0");
        let manifest = local.join("manifests/registry.ollama.ai/library/slow/1");
        assert!(!blob.exists(), "partial blob must be cleaned up");
        assert!(!manifest.exists(), "manifest must not exist after timeout");
    }

    /// One non-blocking attempt to open the write end of `fifo`.
    ///
    /// `O_WRONLY|O_NONBLOCK` on a FIFO succeeds only when a reader is already
    /// present and fails with `ENXIO` otherwise, so this doubles as a probe for
    /// "has the copy reached `File::open` yet".
    fn try_open_fifo_writer(fifo: &Path) -> Option<std::fs::File> {
        use std::os::unix::fs::OpenOptionsExt;
        const O_NONBLOCK: i32 = 0o4000;
        std::fs::OpenOptions::new()
            .write(true)
            .custom_flags(O_NONBLOCK)
            .open(fifo)
            .ok()
    }

    /// Open the write end of `fifo` without ever blocking on a reader that may
    /// never arrive, returning `None` if none appears within `deadline`.
    ///
    /// A plain blocking open here is a test deadlock, and not a theoretical one:
    /// the copy closure can legitimately exit at its first cancel check WITHOUT
    /// opening its source, which happens whenever the blocking pool is saturated
    /// enough that the closure starts after the cancel is set. The 100x stress run
    /// hung a test on exactly that for 15 minutes. So probe non-blockingly, and
    /// only once a reader exists take a normal blocking handle — which cannot
    /// block, because the probe handle is holding the reader — so that `write_all`
    /// still blocks until the copy consumes it, which is what makes these tests
    /// structural rather than timed.
    async fn open_fifo_writer(fifo: &Path, deadline: Duration) -> Option<std::fs::File> {
        let start = std::time::Instant::now();
        loop {
            if let Some(probe) = try_open_fifo_writer(fifo) {
                let w = std::fs::OpenOptions::new().write(true).open(fifo).ok();
                drop(probe);
                return w;
            }
            if start.elapsed() >= deadline {
                return None;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    }

    /// Replace `path` with a FIFO of the same name, so a copy reading it advances
    /// exactly as far as the test feeds it. Returns false when `mkfifo` isn't
    /// available (the caller then skips).
    fn fifoize(path: &Path) -> bool {
        let _ = fs::remove_file(path);
        matches!(
            std::process::Command::new("mkfifo").arg(path).status(),
            Ok(s) if s.success()
        )
    }

    #[test]
    fn cleanup_attempt_never_removes_a_preexisting_local_blob() {
        // The defect this whole change exists to close on the pull side: the old
        // timeout path handed `planned_local_paths` straight to `cleanup_partial`
        // with NO ownership filter, so a timed-out pull could delete a local blob
        // that was already on disk and is shared with another local manifest —
        // one Ollama may have loaded, or may load next.
        //
        // The filter is unit-tested here rather than end-to-end on purpose: the
        // branch that consumes the planned-path *superset* is only reachable when
        // the copy task happens to finish between the timer firing and the cancel
        // store. That window is not deterministically constructible, and building
        // a test that tries to hit it by timing would reintroduce exactly the kind
        // of wall-clock race this change removes.
        let tmp = tempdir().unwrap();
        let shared = tmp.path().join("sha256-shared");
        let mine = tmp.path().join("sha256-mine");
        let never_written = tmp.path().join("sha256-absent");
        fs::write(&shared, b"another manifest depends on this").unwrap();
        fs::write(&mine, b"published by this attempt").unwrap();

        let planned = vec![shared.clone(), mine.clone(), never_written.clone()];
        // Snapshot as `archive_pull` does: taken before this attempt writes.
        let pre_existing: HashSet<PathBuf> = vec![shared.clone()].into_iter().collect();
        cleanup_attempt(PULL_OP, "m:1", &planned, &pre_existing);

        assert!(
            shared.is_file(),
            "a blob that already existed before the pull started is shared data this \
             pull did not create — cleanup must never remove it"
        );
        assert_eq!(
            fs::read(&shared).unwrap(),
            b"another manifest depends on this",
            "a kept blob must also be byte-identical"
        );
        assert!(!mine.exists(), "a blob this attempt published must be reclaimed");
        assert!(!never_written.exists());
    }

    #[tokio::test]
    async fn timeout_leaves_a_preexisting_local_blob_intact() {
        // End-to-end companion to the unit test above, against the *old* shape:
        // `cleanup_partial(&planned_local_paths(&plan))` deleted every planned
        // path that existed — including the pre-seeded shared blob below.
        //
        // Honest note on why it passes now: at a 1 ns timeout the copy is
        // cancelled before it opens anything, so cleanup runs over an EMPTY
        // published list and the pre-existing filter is not what saves the blob.
        // It is still a real regression guard (it fails against the old code),
        // but the filter itself is covered by
        // `cleanup_attempt_never_removes_a_preexisting_local_blob`.
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        make_model(&base.join("archive"), "shared", "1", &[8 * 1024 * 1024]);

        // A local blob with the same digest already on disk, referenced by some
        // other warm model, with content we can prove was not touched.
        let local_blobs = base.join("local").join("blobs");
        fs::create_dir_all(&local_blobs).unwrap();
        let victim = local_blobs.join("sha256-sharedcfg");
        fs::write(&victim, b"belongs to another local manifest").unwrap();

        let err = archive_pull(
            "shared:1",
            &base.join("archive"),
            &base.join("local"),
            Duration::from_nanos(1),
            &FixedProbe(u64::MAX),
            None,
        )
        .await
        .unwrap_err();
        assert!(matches!(err, PullError::Timeout(_)), "got {err:?}");

        assert!(
            victim.is_file(),
            "a timed-out pull must not delete a local blob it did not create"
        );
        assert_eq!(
            fs::read(&victim).unwrap(),
            b"belongs to another local manifest",
            "and must not have mutated it either"
        );
        assert!(
            leftover_temps(&local_blobs).is_empty(),
            "no scratch file may be left in the local blobs dir"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn pull_copy_cancelled_midflight_publishes_nothing() {
        // "Mid-flight" is established STRUCTURALLY by a FIFO source: the copy can
        // only advance as far as the test feeds it, so there is no observer
        // thread, no scratch-file polling, and nothing for CPU load to starve.
        use std::io::Write as _;
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        let (archive, local) = (base.join("archive"), base.join("local"));
        make_model(&archive, "fif", "1", &[64]);
        let fifo = archive.join("blobs/sha256-fif0");
        if !fifoize(&fifo) {
            return; // no mkfifo on this platform — nothing to assert
        }

        let plan = plan_pull("fif:1", &archive, &local).unwrap();
        let cancel: CopyCancel = Arc::new(AtomicBool::new(false));
        let copy = tokio::spawn(copy_model_files(plan, cancel.clone(), CopyActivity::default()));

        // Opening the write end releases the copy's blocking `File::open`. The
        // pipe buffer is far smaller than a chunk, so `write_all` cannot return
        // until the copy has consumed most of it — it has provably created its
        // scratch file and written into it. That is the mid-flight moment.
        let mut w = open_fifo_writer(&fifo, Duration::from_secs(60))
            .await
            .expect("cancel is false here, so the copy cannot exit without opening its source");
        w.write_all(&vec![3u8; COPY_CHUNK_BYTES]).unwrap();
        w.flush().unwrap();
        cancel.store(true, Ordering::Relaxed);
        let _ = w.write_all(&[3u8]);
        drop(w); // EOF, so a reader parked in `read` is released either way

        let (err, published) = copy.await.unwrap().unwrap_err();
        assert!(
            matches!(&err, PullError::Io(m) if m == CANCELLED),
            "got {err:?}"
        );
        let blob_dst = local.join("blobs/sha256-fif0");
        assert!(
            !blob_dst.exists(),
            "a cancelled pull must not publish a blob into the root Ollama watches"
        );
        assert!(
            !published.contains(&blob_dst),
            "the published list must be exact — it must not claim a blob that was never renamed into place"
        );
        assert!(
            !local.join("manifests/registry.ollama.ai/library/fif/1").exists(),
            "a cancelled pull must never publish the manifest"
        );
        assert!(
            leftover_temps(&local.join("blobs")).is_empty(),
            "a cancelled copy must remove its own scratch file"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_join_error_does_not_stop_the_blocking_writer() {
        // Half one of the hazard, proven on its own: a `JoinError` — a panic, or
        // an abort — finishes the OUTER task and tells you NOTHING about the
        // nested `spawn_blocking` closure. Unwinding drops that closure's
        // `JoinHandle`, which detaches it and leaves it live and writing.
        //
        // Deterministic because the cancel flag is never set here: the copy's
        // source is a FIFO that never gets a writer, so the closure either has not
        // started or is blocked in `File::open`, and in neither state can it
        // finish. There is no interleaving in which the count reaches zero.
        let tmp = tempdir().unwrap();
        let src = tmp.path().join("src.fifo");
        let dst = tmp.path().join("manifest-dst");
        if !fifoize(&src) {
            return;
        }

        let cancel: CopyCancel = Arc::new(AtomicBool::new(false));
        let activity = CopyActivity::default();
        // The task signals before calling the copy, and there is no await point
        // between that signal and the activity increment + `spawn_blocking` — an
        // abort can only land at an await, so receiving this guarantees the count
        // was taken. (Aborting a task that has never been polled would cancel it
        // before it counted anything.)
        let (started_tx, started_rx) = tokio::sync::oneshot::channel::<()>();
        let task = tokio::spawn({
            let (src, dst, cancel, activity) =
                (src.clone(), dst.clone(), cancel.clone(), activity.clone());
            async move {
                let _ = started_tx.send(());
                copy_file_cancellable(&src, &dst, &cancel, &activity).await
            }
        });
        started_rx.await.unwrap();

        task.abort();
        assert!(
            task.await.is_err(),
            "the scenario under test is a JoinError, so we must actually have one"
        );
        assert_eq!(
            activity.in_flight(),
            1,
            "the outer task is finished while its blocking writer is not — this gap \
             IS the hazard, and a JoinError is the only outcome that has it"
        );
        assert!(
            !activity.wait_quiescent(Duration::from_millis(200)).await,
            "quiescence must not be reported while a detached writer is live"
        );

        // Cancel FIRST, then release the FIFO. The writer stops instead of
        // publishing, which is why skipping cleanup costs at most an orphan.
        //
        // The release must not block: the closure is either parked in `File::open`
        // (a reader — the probe succeeds, and dropping the handle gives it EOF) or
        // has not started and will exit at its first cancel check without ever
        // opening (no reader ever appears). A blocking open would hang forever in
        // the second case, which is what the stress run caught.
        cancel.store(true, Ordering::Relaxed);
        let start = std::time::Instant::now();
        while activity.in_flight() != 0 && start.elapsed() < Duration::from_secs(10) {
            drop(try_open_fifo_writer(&src));
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert!(
            activity.wait_quiescent(Duration::from_secs(10)).await,
            "the writer must stop, and be seen to stop, once released"
        );
        assert!(
            !dst.exists(),
            "an abandoned writer must never publish, even after we stopped waiting"
        );
        assert!(leftover_temps(tmp.path()).is_empty());
    }

    #[tokio::test]
    async fn join_error_with_a_live_writer_deletes_nothing() {
        // Half two: given a live writer, the decision must be to delete NOTHING.
        //
        // The writer is synthetic — a held activity guard, exactly what a live
        // blocking closure holds. That is deliberate. Driving this with a real
        // parked copy is not race-free: an earlier version fed a FIFO chunk and
        // relied on the copy sitting in `read`, which flaked 1-in-~12 in the full
        // suite. `write_all` returning means the PIPE took the bytes, not that the
        // reader drained them, and under a saturated blocking pool the closure may
        // not even start until after the cancel is set — in which case it exits at
        // its first check and IS quiescent. "Parked mid-file" is not observable
        // without a hook, so the test must not assert on it. The linkage this
        // stands in for — that a detached writer really does hold the count — is
        // proven independently by `a_join_error_does_not_stop_the_blocking_writer`.
        let tmp = tempdir().unwrap();
        let published = tmp.path().join("sha256-mine");
        let shared = tmp.path().join("sha256-shared");
        fs::write(&published, b"published by this attempt").unwrap();
        fs::write(&shared, b"another manifest depends on this").unwrap();
        let pre_existing: HashSet<PathBuf> = vec![shared.clone()].into_iter().collect();

        let cancel: CopyCancel = Arc::new(AtomicBool::new(false));
        let activity = CopyActivity::default();
        let live_writer = activity.hold_for_test();

        let cleaned = cleanup_after_join_error(
            PULL_OP,
            "m:1",
            &cancel,
            &activity,
            Duration::from_millis(300),
            &[published.clone(), shared.clone()],
            &pre_existing,
            tmp.path(),
            Path::new(""),
        )
        .await;

        // Filesystem first: the damage is the claim, the return value is bookkeeping.
        assert!(
            published.is_file(),
            "when the writer cannot be proven stopped, cleanup must delete NOTHING \
             — a live writer can republish over whatever we remove, and on the pull \
             side the file it republishes is the manifest"
        );
        assert!(shared.is_file());
        assert!(
            !cleaned,
            "quiescence cannot be proven while a writer is live"
        );
        assert!(
            cancel.load(Ordering::Relaxed),
            "the flag must still be set, so the live writer stops rather than publishes"
        );
        drop(live_writer);
    }

    #[tokio::test]
    async fn stash_leaves_the_manifest_readable_at_its_destination() {
        // The TOCTOU this closes: `stash_existing_manifest` used to `rename` the
        // manifest away, leaving `dst` completely ABSENT from disk for the whole
        // duration of the manifest copy that follows — a real window in which a
        // concurrent reader (Ollama listing/loading models, or
        // `registry::scan_manifest_tree`) sees this model as missing rather than
        // as its previous, still-valid version. `copy_file_cancellable` never
        // needs `dst` absent — it always stages into its own temp path and only
        // ever touches `dst` with the one atomic `rename` that publishes — so
        // there is no reason for the stash to vacate it first.
        let tmp = tempdir().unwrap();
        let root = tmp.path();
        let rel = Path::new("registry.ollama.ai/library/toctou/1");
        let dst = root.join("manifests").join(rel);
        fs::create_dir_all(dst.parent().unwrap()).unwrap();
        let original = b"a reader must be able to see this the whole time".to_vec();
        fs::write(&dst, &original).unwrap();

        assert!(stash_existing_manifest(root, rel, &dst).unwrap());

        assert!(
            dst.is_file(),
            "the manifest must still be present at its destination immediately \
             after stashing — a concurrent reader must never see it missing"
        );
        assert_eq!(
            fs::read(&dst).unwrap(),
            original,
            "and its content must be untouched until the real publish replaces it"
        );
        assert_eq!(
            fs::read(manifest_backup_path(root, rel).unwrap()).unwrap(),
            original,
            "the backup must also hold a full copy, so a restore is still possible"
        );
    }

    #[tokio::test]
    async fn a_failed_stash_never_leaves_a_partial_backup() {
        // The gap this closes (found by the T2 panel, round 4): a stash-by-copy
        // that fails partway used to still risk leaving a file sitting at the
        // exact `backup` path, because `std::fs::copy` creates its destination
        // up front and can fail mid-write (disk full, permission revoked). The
        // ENTIRE safety of the restore path rests on `backup.exists()` meaning
        // "there is something complete to restore" — a caller whose stash
        // failed (`copy_model_files` never even attempts the manifest copy in
        // that case) still unconditionally runs `undo_pull_attempt`, which would
        // rename a half-written backup straight over a destination manifest
        // this attempt never touched, corrupting a manifest that needed no
        // restoring at all.
        //
        // Forced here by handing the stash a DIRECTORY as the "existing
        // manifest" — `std::fs::copy` refuses a non-regular-file source, which
        // exercises the exact copy-fails-before-completion path without needing
        // root or a real disk-full condition.
        let tmp = tempdir().unwrap();
        let root = tmp.path();
        let rel = Path::new("registry.ollama.ai/library/faildir/1");
        let dst = root.join("manifests").join(rel);
        fs::create_dir_all(&dst).unwrap(); // exists, but is not a manifest file

        stash_existing_manifest(root, rel, &dst).unwrap_err();

        assert!(
            !manifest_backup_path(root, rel).unwrap().exists(),
            "a failed stash must leave NOTHING at the final backup path — that \
             presence check is the only thing standing between \"nothing to \
             restore\" and clobbering an untouched destination on the very next \
             failure path"
        );
    }

    #[tokio::test]
    async fn a_blob_failure_before_the_old_stash_point_no_longer_resurrects_a_stale_backup() {
        // The gap this closes (found by the T2 panel, round 7): `stash_existing_manifest`
        // used to run only AFTER the entire blob-copy loop completed. Every
        // failure return from `copy_model_files` funnels into the caller's undo
        // route (`undo_pull_attempt`), which unconditionally calls
        // `restore_manifest_backup` for this model's `rel` — it has no way to
        // know whether THIS attempt ever reached the stash step. So a blob
        // failure that returned BEFORE the old stash call left any pre-existing
        // stale backup (e.g. one orphaned by eviction, per the round-6/7 fixes)
        // completely untouched by this attempt, and the undo route would then
        // resurrect it at `dst_manifest` — content this attempt never created
        // and never touched, restored over a destination that may have been
        // legitimately absent (evicted) or held something else entirely.
        //
        // Fixed by moving the stash call ahead of the blob loop, so by the time
        // ANY failure can be returned, this attempt has always either staged a
        // real backup of what was actually at `dst`, or (as here) confirmed
        // there was nothing there and cleared any stale leftover — either way
        // `restore_manifest_backup` can no longer observe a backup this
        // attempt did not itself just create or clear.
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        let (archive, local) = (base.join("archive"), base.join("local"));
        let model = make_model(&archive, "blobfail", "1", &[64]);
        let rel = Path::new("registry.ollama.ai/library/blobfail/1");

        // Force the very first blob copy to fail deterministically: delete the
        // archive's config blob (the first digest `parse_manifest_blobs`
        // collects) so `copy_file_cancellable` hits a real ENOENT before any
        // blob is copied and long before the OLD stash call site would have
        // run.
        let cfg_blob = archive.join("blobs").join("sha256-blobfailcfg");
        assert!(cfg_blob.exists(), "precondition: make_model wrote the config blob");
        fs::remove_file(&cfg_blob).unwrap();

        // `dst_manifest` is absent, as if this model had already been evicted —
        // but a stale backup from that eviction's predecessor attempt persists.
        let dst_manifest = local.join("manifests").join(rel);
        let backup = manifest_backup_path(&local, rel).unwrap();
        fs::create_dir_all(backup.parent().unwrap()).unwrap();
        fs::write(&backup, b"stale content from an attempt this pull has nothing to do with")
            .unwrap();

        let plan = plan_pull(&model, &archive, &local).unwrap();
        let planned = planned_local_paths(&plan);
        let pre_existing: HashSet<PathBuf> =
            planned.iter().filter(|p| p.exists()).cloned().collect();

        let (err, published) = copy_model_files(
            plan.clone(),
            Arc::new(AtomicBool::new(false)),
            CopyActivity::default(),
        )
        .await
        .unwrap_err();
        assert!(
            matches!(err, PullError::Io(ref msg) if msg.contains("copy blob")),
            "precondition: the forced failure must be the blob copy, not \
             something else: {err:?}"
        );

        // The caller's real undo route.
        undo_pull_attempt(&plan, rel, &published, &pre_existing, true);

        assert!(
            !dst_manifest.exists(),
            "a blob failure must never resurrect unrelated stale content at \
             dst_manifest — this attempt never touched it and dst was \
             legitimately absent before this pull began"
        );
        assert!(
            !backup.exists(),
            "the stale backup must be cleared, not preserved for a future \
             attempt to mistakenly restore"
        );
    }

    #[tokio::test]
    async fn a_stash_with_no_destination_still_clears_a_stale_backup() {
        // The gap this closes (found by the T2 panel, round 6): the stale-backup
        // clear used to sit AFTER the `if !dst.exists() { return Ok(false); }`
        // early return, so it was unreachable whenever the destination happened
        // to be absent. That is not a rare case: a manifest this module
        // previously stashed a backup for can vanish out from under that backup
        // via local eviction (`FsLocalEvictor` removes the manifest leaf but has
        // no reason to know about, or clean up, `.chord-manifest-backup`) — so
        // "no current destination" is not evidence the backup is fresh.
        //
        // Left uncleared, a later attempt at the same model whose OWN manifest
        // copy then failed would have its `copy_model_files`/`undo_pull_attempt`
        // restore route find this stale backup, treat `backup.exists()` as
        // proof of validity, and resurrect an unrelated, possibly-dangling
        // manifest at `dst` — the exact hazard class this whole fix chain
        // exists to prevent, just reached through eviction instead of a failed
        // rename.
        let tmp = tempdir().unwrap();
        let root = tmp.path();
        let rel = Path::new("registry.ollama.ai/library/evicted/1");
        let dst = root.join("manifests").join(rel); // deliberately never created

        let backup = manifest_backup_path(root, rel).unwrap();
        fs::create_dir_all(backup.parent().unwrap()).unwrap();
        fs::write(&backup, b"orphaned by an eviction that removed dst but not this").unwrap();

        let staged = stash_existing_manifest(root, rel, &dst).unwrap();

        assert!(
            !staged,
            "there was nothing at `dst` to stash, so this call published no \
             new backup of its own"
        );
        assert!(
            !backup.exists(),
            "a stale backup left behind by eviction must be cleared even when \
             `dst` is currently absent — otherwise a LATER attempt's failed \
             manifest copy would restore this orphan as if it were valid"
        );
    }

    #[tokio::test]
    async fn a_failed_stale_backup_removal_is_reported_not_swallowed() {
        // The gap this closes (found by the T2 panel, round 8): the stale-backup
        // clear discarded the removal's own `Result` via `let _ = ...`. If the
        // removal itself failed (permissions, a read-only filesystem, another
        // fs error) `backup` could still exist afterward, yet the function fell
        // straight through to the `dst.exists()` check and, on an absent `dst`,
        // returned `Ok(false)` — reporting "nothing to stash" while a stale,
        // unverified backup was still sitting there for a later
        // `restore_manifest_backup` call to trust and resurrect. That is the
        // exact hazard class this whole fix chain exists to prevent, reached
        // through a swallowed error instead of an unreachable code path.
        //
        // Forced here by making the backup's parent directory read-only, so the
        // `remove_file` call fails with a real `EACCES` instead of a simulated
        // error. Asserts the failure now propagates as `Err` — a swallowed
        // error would have returned `Ok(false)` here instead.
        use std::os::unix::fs::PermissionsExt;

        // Root bypasses the DAC permission check this test relies on to force
        // a genuine removal failure — under root the `remove_file` below would
        // silently succeed anyway, and the test would prove nothing. Skip
        // rather than assert something this environment cannot actually show.
        // A bare `geteuid()` FFI call avoids pulling in a dependency-tree
        // change (the `nix` crate is already a dependency but its `Uid` type
        // is gated behind a "user" feature this workspace doesn't enable) for
        // what a single libc symbol answers directly.
        extern "C" {
            fn geteuid() -> u32;
        }
        if unsafe { geteuid() } == 0 {
            eprintln!("skipping a_failed_stale_backup_removal_is_reported_not_swallowed: running as root");
            return;
        }

        let tmp = tempdir().unwrap();
        let root = tmp.path();
        let rel = Path::new("registry.ollama.ai/library/unremovable/1");
        let dst = root.join("manifests").join(rel); // absent, as after eviction

        let backup = manifest_backup_path(root, rel).unwrap();
        let backup_parent = backup.parent().unwrap().to_path_buf();
        fs::create_dir_all(&backup_parent).unwrap();
        fs::write(&backup, b"stale content this attempt must not silently accept").unwrap();

        // Read-only parent dir: unlinking an entry from it fails regardless of
        // the entry's own permissions.
        let mut perms = fs::metadata(&backup_parent).unwrap().permissions();
        perms.set_mode(0o555);
        fs::set_permissions(&backup_parent, perms).unwrap();

        let result = stash_existing_manifest(root, rel, &dst);

        // Restore write permission before the tempdir's own cleanup runs, no
        // matter which assertion below fires.
        let mut restore_perms = fs::metadata(&backup_parent).unwrap().permissions();
        restore_perms.set_mode(0o755);
        fs::set_permissions(&backup_parent, restore_perms).unwrap();

        assert!(
            result.is_err(),
            "a removal failure must propagate as an error, not be silently \
             reported as `Ok(false)` (\"nothing to stash\") while the \
             unverified stale backup is still on disk"
        );
        assert!(
            backup.exists(),
            "precondition check: the backup must genuinely still be present \
             (removal genuinely failed) for this test to mean anything"
        );
    }

    #[tokio::test]
    async fn a_stash_unsafe_error_skips_restore_through_the_real_undo_route() {
        // The gap this closes (found by the T2 panel, round 9): round 9 made
        // `stash_existing_manifest` propagate a removal failure as `Err`
        // instead of swallowing it, and claimed that was enough to stop the
        // resurrection round-8's fix already covered. Both reviewers (codex
        // AND agy, independently) traced the actual caller/undo sequence and
        // showed it was NOT enough: `copy_model_files` converts the `Err` into
        // a normal `PullError`, which `archive_pull` still hands to
        // `undo_pull_attempt` UNCONDITIONALLY — and `undo_pull_attempt` calls
        // `restore_manifest_backup`, which sees `backup.exists() == true`
        // (the removal that failed left it there) and restores it to `dst`
        // anyway. Propagating the error only moved WHERE the resurrection
        // happened; it did not stop it.
        //
        // This test is the missing coverage both reviewers pointed out: the
        // prior regression test (`a_failed_stale_backup_removal_is_reported_
        // not_swallowed`) only called `stash_existing_manifest` directly, never
        // exercising `undo_pull_attempt`/`restore_manifest_backup` at all — so
        // it could not have caught this. This one drives the REAL sequence:
        // `copy_model_files` (which now returns `PullError::StashUnsafe` for
        // this failure) → the real `undo_pull_attempt`, gated on
        // `!matches!(err, PullError::StashUnsafe(_))` exactly as
        // `archive_pull` itself now does.
        use std::os::unix::fs::PermissionsExt;

        extern "C" {
            fn geteuid() -> u32;
        }
        if unsafe { geteuid() } == 0 {
            eprintln!(
                "skipping a_stash_unsafe_error_skips_restore_through_the_real_undo_route: \
                 running as root"
            );
            return;
        }

        let tmp = tempdir().unwrap();
        let base = tmp.path();
        let (archive, local) = (base.join("archive"), base.join("local"));
        let model = make_model(&archive, "unremovable2", "1", &[64]);
        let rel = Path::new("registry.ollama.ai/library/unremovable2/1");

        // `dst_manifest` is absent, as if this model had already been evicted —
        // but a stale, unrelated backup persists, and its parent directory is
        // read-only, so THIS attempt's own stash step cannot clear it.
        let dst_manifest = local.join("manifests").join(rel);
        let backup = manifest_backup_path(&local, rel).unwrap();
        let backup_parent = backup.parent().unwrap().to_path_buf();
        fs::create_dir_all(&backup_parent).unwrap();
        fs::write(&backup, b"stale content this attempt never created or verified").unwrap();
        let mut perms = fs::metadata(&backup_parent).unwrap().permissions();
        perms.set_mode(0o555);
        fs::set_permissions(&backup_parent, perms).unwrap();

        let plan = plan_pull(&model, &archive, &local).unwrap();
        let planned = planned_local_paths(&plan);
        let pre_existing: HashSet<PathBuf> =
            planned.iter().filter(|p| p.exists()).cloned().collect();

        let copy_result = copy_model_files(
            plan.clone(),
            Arc::new(AtomicBool::new(false)),
            CopyActivity::default(),
        )
        .await;

        // Restore write permission before any further filesystem work (the
        // real undo route, and the tempdir's own cleanup) runs, no matter what
        // the assertions below find.
        let mut restore_perms = fs::metadata(&backup_parent).unwrap().permissions();
        restore_perms.set_mode(0o755);
        fs::set_permissions(&backup_parent, restore_perms).unwrap();

        let (err, published) = copy_result.unwrap_err();
        assert!(
            matches!(err, PullError::StashUnsafe(_)),
            "precondition: the forced failure must be the stash-clear step, \
             surfaced as StashUnsafe, not something else: {err:?}"
        );
        assert!(
            published.is_empty(),
            "no blob is ever copied before the stash step runs"
        );

        // Exactly what `archive_pull` itself now does: gate the restore on
        // whether this specific error is StashUnsafe.
        let restore_is_safe = !matches!(err, PullError::StashUnsafe(_));
        undo_pull_attempt(&plan, rel, &published, &pre_existing, restore_is_safe);

        assert!(
            !dst_manifest.exists(),
            "a StashUnsafe failure must never resurrect unrelated stale \
             content at dst_manifest — this attempt never verified the \
             backup and dst was legitimately absent before this pull began"
        );
        assert!(
            backup.exists(),
            "the stale backup must be left exactly as found (untouched, not \
             restored, not deleted) when this attempt could not verify it — \
             the same 'when you can't prove it's safe, do nothing' outcome \
             the not-quiescent JoinError branch already accepts"
        );
    }

    #[tokio::test]
    async fn a_create_dir_all_failure_before_stash_no_longer_resurrects_a_stale_backup() {
        // The gap this closes (found by the T2 panel, round 10 — BOTH codex and
        // agy voted, but only agy caught this one as CRITICAL): round 10 added
        // `PullError::StashUnsafe` to stop `stash_existing_manifest`'s OWN
        // failures from routing through the generic undo/restore path. But
        // `copy_model_files` ran `tokio::fs::create_dir_all(dst_manifest.parent())`
        // BEFORE calling `stash_existing_manifest` at all — so a `create_dir_all`
        // failure (permissions, disk quota, a filesystem error, or here: a path
        // component that already exists as a non-directory) returned
        // `PullError::Io` having never reached the stash step. `archive_pull`
        // computes `restore_is_safe = !matches!(e, PullError::StashUnsafe(_))`,
        // which is `true` for `PullError::Io` — so any pre-existing stale
        // backup (e.g. one orphaned by eviction, per rounds 6/7) was left
        // completely uncleared, and the generic undo route resurrected it at
        // `dst_manifest` anyway. `StashUnsafe` covered the case where the stash
        // step itself fails to clear a backup; it did nothing for the sibling
        // case where a step BEFORE the stash step fails and the stash step
        // never runs at all — the exact same resurrection hazard, reached one
        // step earlier.
        //
        // Fixed by reordering `copy_model_files` so `stash_existing_manifest`
        // runs BEFORE `create_dir_all(dst_manifest.parent())`, not after.
        // `stash_existing_manifest` has no dependency on that parent directory
        // existing (it only calls `dst.exists()`, and creates `backup`'s own
        // parent independently), so there is no ordering requirement the other
        // way — moving it first closes this gap.
        //
        // This test forces a REAL, deterministic `create_dir_all` failure that
        // is not permission-based (this gate environment runs as root, where a
        // read-only-directory technique is a no-op): it places a plain FILE at
        // the exact path `dst_manifest`'s parent needs to become, so
        // `create_dir_all` hits `ENOTDIR`/`AlreadyExists` regardless of
        // privilege level. No root-skip guard is needed for that reason.
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        let (archive, local) = (base.join("archive"), base.join("local"));
        let model = make_model(&archive, "mkdirfail", "1", &[64]);
        let rel = Path::new("registry.ollama.ai/library/mkdirfail/1");

        // `dst_manifest` is absent, as if this model had already been evicted —
        // but a stale, unrelated backup persists.
        let dst_manifest = local.join("manifests").join(rel);
        let backup = manifest_backup_path(&local, rel).unwrap();
        fs::create_dir_all(backup.parent().unwrap()).unwrap();
        fs::write(&backup, b"stale content this attempt never created or verified").unwrap();

        // `dst_manifest`'s parent is `.../manifests/registry.ollama.ai/library/mkdirfail`.
        // Create everything up to (not including) that leaf as real directories,
        // then occupy the leaf itself with a plain file, so
        // `create_dir_all(dst_manifest.parent())` cannot create it as a
        // directory no matter who runs the test.
        let parent = dst_manifest.parent().unwrap();
        fs::create_dir_all(parent.parent().unwrap()).unwrap();
        fs::write(parent, b"occupying the directory slot with a plain file").unwrap();

        let plan = plan_pull(&model, &archive, &local).unwrap();
        let planned = planned_local_paths(&plan);
        let pre_existing: HashSet<PathBuf> =
            planned.iter().filter(|p| p.exists()).cloned().collect();

        let (err, published) = copy_model_files(
            plan.clone(),
            Arc::new(AtomicBool::new(false)),
            CopyActivity::default(),
        )
        .await
        .unwrap_err();
        assert!(
            matches!(err, PullError::Io(_)) && !matches!(err, PullError::StashUnsafe(_)),
            "precondition: the forced failure must be create_dir_all's Io error, \
             not StashUnsafe — this test exercises the sibling path that ran \
             before stash_existing_manifest was ever reached in round 10's \
             code: {err:?}"
        );
        assert!(
            published.is_empty(),
            "no blob is ever copied before create_dir_all/stash run"
        );

        // Exactly what `archive_pull` itself now does: gate the restore on
        // whether this specific error is StashUnsafe. It is not, so
        // `restore_is_safe` is `true` here — this test's whole point is that
        // being `true` must no longer matter, because `stash_existing_manifest`
        // (now running first) has already cleared the stale backup by the time
        // this Io error is ever returned.
        let restore_is_safe = !matches!(err, PullError::StashUnsafe(_));
        assert!(
            restore_is_safe,
            "precondition: PullError::Io must compute restore_is_safe = true — \
             this test is only meaningful if the generic undo route is actually \
             exercised, not skipped the same way StashUnsafe skips it"
        );
        undo_pull_attempt(&plan, rel, &published, &pre_existing, restore_is_safe);

        assert!(
            !dst_manifest.exists(),
            "a create_dir_all failure before the stash step must never \
             resurrect unrelated stale content at dst_manifest — this attempt \
             never touched it and dst was legitimately absent before this \
             pull began"
        );
        assert!(
            !backup.exists(),
            "the stale backup must have been cleared by stash_existing_manifest \
             running BEFORE create_dir_all, not preserved for the generic undo \
             route to mistakenly restore"
        );
    }

    #[tokio::test]
    async fn abandoned_pull_restores_the_previous_manifest() {
        // The defect this closes: a pull that COMPLETES and is then abandoned
        // (the timer fires just as the copy finishes, or a quiescent JoinError)
        // cleans up over the planned superset. The manifest destination is in
        // `pre_existing`, so the filter keeps it — but the content at that path is
        // now this attempt's, naming blobs the same cleanup is deleting. That is
        // exactly the state blobs-first-manifest-last exists to prevent, reached
        // through the ownership filter instead of through truncation.
        //
        // Driven through the real functions in the real order rather than through
        // the timer: `copy_model_files` to completion, then the same undo route
        // `archive_pull`'s abandon branches call. The race that gets you here is
        // not constructible; what it does once there is.
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        let (archive, local) = (base.join("archive"), base.join("local"));
        make_model(&archive, "restore", "1", &[128]);

        // A previous, DIFFERENT model entry already at the manifest destination,
        // with its own blob already on disk.
        let rel = Path::new("registry.ollama.ai/library/restore/1");
        let dst_manifest = local.join("manifests").join(rel);
        fs::create_dir_all(dst_manifest.parent().unwrap()).unwrap();
        let original = b"{\"config\":{\"size\":3,\"digest\":\"sha256:old\"},\"layers\":[]}".to_vec();
        fs::write(&dst_manifest, &original).unwrap();
        let local_blobs = local.join("blobs");
        fs::create_dir_all(&local_blobs).unwrap();
        let old_blob = local_blobs.join("sha256-old");
        fs::write(&old_blob, b"the previous entry's blob").unwrap();

        let plan = plan_pull("restore:1", &archive, &local).unwrap();
        let planned = planned_local_paths(&plan);
        let pre_existing: HashSet<PathBuf> =
            planned.iter().filter(|p| p.exists()).cloned().collect();
        assert!(
            pre_existing.contains(&dst_manifest),
            "the manifest PATH must be in the snapshot — that is what made the \
             filter keep this attempt's content"
        );

        // The copy runs to completion: blobs published, manifest replaced.
        copy_model_files(
            plan.clone(),
            Arc::new(AtomicBool::new(false)),
            CopyActivity::default(),
        )
        .await
        .unwrap();
        assert_ne!(
            fs::read(&dst_manifest).unwrap(),
            original,
            "precondition: the attempt really did replace the manifest"
        );

        // ...and is then abandoned.
        undo_pull_attempt(&plan, rel, &planned, &pre_existing, true);

        assert_eq!(
            fs::read(&dst_manifest).unwrap(),
            original,
            "an abandoned pull must leave the PREVIOUS manifest on disk — not its \
             own (which names blobs that were just deleted), and not nothing \
             (which would discard a model entry this pull never owned)"
        );
        assert!(
            !local_blobs.join("sha256-restore0").exists(),
            "this attempt's blobs must still be reclaimed"
        );
        assert!(old_blob.is_file(), "the previous entry's blob must survive");

        // No debris, and in particular nothing left in the manifests tree: every
        // file leaf under `manifests/` is a MODEL to `scan_manifest_tree`, which
        // applies no dotfile filter, so a stray backup there would be discovered
        // as a phantom model named after the file.
        let leaves: Vec<String> = fs::read_dir(dst_manifest.parent().unwrap())
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(leaves, vec!["1".to_string()], "stray file in the manifests tree");
        assert!(
            !manifest_backup_path(&local, rel).unwrap().exists(),
            "the backup must be consumed by the restore, not left behind"
        );
    }

    #[tokio::test]
    async fn a_failed_restore_skips_cleanup_instead_of_guessing() {
        // The gap this closes (found by the T2 panel, round 5b): a failed
        // restore was only logged — `undo_pull_attempt` proceeded to
        // `cleanup_attempt` regardless, on the assumption that "restore was
        // attempted" is as good as "restore succeeded". It is not: a failed
        // `rename(&backup, dst)` leaves `dst` completely unchanged (rename is
        // atomic — it never partially applies), so `dst` may still hold THIS
        // attempt's manifest, still naming the very blobs cleanup is about to
        // delete. Skipping cleanup on a failed restore is the same "don't
        // delete when you can't prove it's safe" rule the not-quiescent
        // JoinError branch already applies.
        //
        // Forced here by making `dst_manifest` a NON-EMPTY DIRECTORY: renaming
        // a regular file onto it fails deterministically (ENOTEMPTY/EEXIST on
        // Linux) without needing root or real disk pressure.
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        let (archive, local) = (base.join("archive"), base.join("local"));
        make_model(&archive, "restorefail", "1", &[64]);

        let rel = Path::new("registry.ollama.ai/library/restorefail/1");
        let dst_manifest = local.join("manifests").join(rel);
        fs::create_dir_all(&dst_manifest).unwrap(); // occupied, non-empty below
        fs::write(dst_manifest.join("occupant"), b"blocks the rename").unwrap();

        // A real backup exists at the deterministic path, as if an earlier
        // stash had succeeded before this attempt's undo route ran.
        let backup = manifest_backup_path(&local, rel).unwrap();
        fs::create_dir_all(backup.parent().unwrap()).unwrap();
        fs::write(&backup, b"the manifest that should have been restored").unwrap();

        let plan = plan_pull("restorefail:1", &archive, &local).unwrap();
        let candidates = planned_local_paths(&plan);
        let local_blobs = local.join("blobs");
        fs::create_dir_all(&local_blobs).unwrap();
        let this_attempts_blob = local_blobs.join("sha256-restorefail0");
        fs::write(&this_attempts_blob, b"published by this attempt").unwrap();
        let pre_existing: HashSet<PathBuf> = HashSet::new();

        undo_pull_attempt(&plan, rel, &candidates, &pre_existing, true);

        assert!(
            this_attempts_blob.exists(),
            "cleanup must be SKIPPED when the restore itself failed — deleting \
             this attempt's blobs while dst_manifest may still name them (the \
             rename never applied, so dst is unchanged) is worse than an orphan"
        );
        assert!(
            backup.is_file(),
            "the backup must be left in place too, since it was never consumed \
             — a future retry or operator can still recover it"
        );
    }

    #[tokio::test]
    async fn successful_pull_keeps_its_own_manifest_and_drops_the_backup() {
        // The other half: the stash must not resurrect anything on the happy path.
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        let (archive, local) = (base.join("archive"), base.join("local"));
        let model = make_model(&archive, "keep", "1", &[64]);

        let rel = Path::new("registry.ollama.ai/library/keep/1");
        let dst_manifest = local.join("manifests").join(rel);
        fs::create_dir_all(dst_manifest.parent().unwrap()).unwrap();
        fs::write(&dst_manifest, b"the stale previous entry").unwrap();

        archive_pull(
            &model,
            &archive,
            &local,
            Duration::from_secs(30),
            &FixedProbe(u64::MAX),
            None,
        )
        .await
        .unwrap();

        assert_eq!(
            fs::read(&dst_manifest).unwrap(),
            fs::read(archive.join("manifests").join(rel)).unwrap(),
            "a successful pull must leave ITS manifest in place"
        );
        assert!(
            !manifest_backup_path(&local, rel).unwrap().exists(),
            "a successful pull must drop the previous manifest it parked"
        );
    }

    #[tokio::test]
    async fn join_error_with_a_quiescent_writer_still_cleans_up() {
        // The other half: "delete nothing" applies only when quiescence cannot be
        // PROVEN. Once it can, a JoinError must still reclaim what the attempt
        // published — otherwise the fix would just be a disabled cleanup path.
        let tmp = tempdir().unwrap();
        let published = tmp.path().join("sha256-mine");
        let shared = tmp.path().join("sha256-shared");
        fs::write(&published, b"published by this attempt").unwrap();
        fs::write(&shared, b"another manifest depends on this").unwrap();
        let pre_existing: HashSet<PathBuf> = vec![shared.clone()].into_iter().collect();

        let cancel: CopyCancel = Arc::new(AtomicBool::new(false));
        let activity = CopyActivity::default(); // nothing in flight
        let cleaned = cleanup_after_join_error(
            PULL_OP,
            "m:1",
            &cancel,
            &activity,
            Duration::from_millis(300),
            &[published.clone(), shared.clone()],
            &pre_existing,
            tmp.path(),
            Path::new(""),
        )
        .await;

        assert!(cleaned, "a quiescent writer makes cleanup provably safe");
        assert!(!published.exists(), "this attempt's file must be reclaimed");
        assert!(
            shared.is_file(),
            "the pre-existing filter still applies on this route"
        );
        assert!(
            cancel.load(Ordering::Relaxed),
            "the cancel flag must be set on every join-error path, quiescent or not"
        );
    }

    #[tokio::test]
    async fn join_error_with_a_quiescent_writer_restores_the_stashed_manifest() {
        // The gap this closes: the quiescent branch of `cleanup_after_join_error`
        // used to call only `cleanup_attempt`, never `restore_manifest_backup` —
        // the one failure route that bypassed `undo_pull_attempt` entirely. A
        // manifest this attempt stashed would stay orphaned in the backup dir
        // forever: either the model entry goes missing (if the new manifest never
        // published) or points at blobs this same cleanup call just deleted (if
        // it did) — exactly the hazard blobs-first-manifest-last exists to
        // prevent, reached through the one path that never restored.
        let tmp = tempdir().unwrap();
        let root = tmp.path();
        let rel = Path::new("registry.ollama.ai/library/joinrestore/1");
        let dst_manifest = root.join("manifests").join(rel);
        fs::create_dir_all(dst_manifest.parent().unwrap()).unwrap();
        let original = b"the previous entry, parked before this attempt".to_vec();
        fs::write(&dst_manifest, &original).unwrap();

        // Stash it exactly as `copy_model_files` does before it starts the
        // manifest copy, then simulate this attempt having already replaced the
        // destination — whether the writer reached its publish `rename` before
        // the panic is exactly what a `JoinError` cannot tell us, so the restore
        // must be unconditional once quiescence is proven, not dependent on
        // which side of that race we landed on.
        assert!(stash_existing_manifest(root, rel, &dst_manifest).unwrap());
        fs::write(&dst_manifest, b"this attempt's own, now-orphaned manifest").unwrap();

        let published = root.join("blobs").join("sha256-thisattempt");
        fs::create_dir_all(published.parent().unwrap()).unwrap();
        fs::write(&published, b"a blob this attempt published").unwrap();
        let pre_existing: HashSet<PathBuf> = HashSet::new();

        let cancel: CopyCancel = Arc::new(AtomicBool::new(false));
        let activity = CopyActivity::default(); // nothing in flight: quiescent immediately

        let cleaned = cleanup_after_join_error(
            PULL_OP,
            "m:1",
            &cancel,
            &activity,
            Duration::from_millis(300),
            &[published.clone()],
            &pre_existing,
            root,
            rel,
        )
        .await;

        assert!(cleaned, "a quiescent writer makes cleanup provably safe");
        assert_eq!(
            fs::read(&dst_manifest).unwrap(),
            original,
            "a JoinError must restore the manifest this attempt stashed, not leave \
             its own (orphaned) content and not leave nothing at all"
        );
        assert!(
            !published.exists(),
            "this attempt's blob must still be reclaimed"
        );
        assert!(
            !manifest_backup_path(root, rel).unwrap().exists(),
            "the backup must be consumed by the restore, not left behind"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn pull_never_truncates_an_existing_local_manifest() {
        // The pull side's at-risk file is the MANIFEST, not a blob: blobs that
        // already exist locally are skipped outright (content-addressed), so they
        // are never opened for writing — but the manifest is copied
        // unconditionally, over whatever is already at that path.
        //
        // What protects it is now TWO different mechanisms answering two different
        // questions, and this test passes if either holds, so read the controls
        // rather than this test to know which is which:
        //   - temp-staging stops a PARTIAL write from ever reaching the
        //     destination. Its control is
        //     `cancelled_midcopy_never_mutates_an_existing_destination`, which
        //     fails reliably when staging is removed. This test does NOT fail then
        //     — the stash below already moved the original out of harm's way.
        //   - the stash/restore stops a COMPLETE replacement from surviving an
        //     abandoned attempt. Its control is
        //     `abandoned_pull_restores_the_previous_manifest`.
        // Removing both together fails this test, which is what "neither is
        // redundant" means here.
        //
        // The archive manifest is swapped for a FIFO *after* planning (planning is
        // what reads it), so the copy parks inside the manifest write and the test
        // — not the clock — decides when it is mid-flight.
        use std::io::Write as _;
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        let (archive, local) = (base.join("archive"), base.join("local"));
        make_model(&archive, "mani", "1", &[64]);
        let plan = plan_pull("mani:1", &archive, &local).unwrap();

        // Every blob already local → the copy goes straight to the manifest.
        let local_blobs = local.join("blobs");
        fs::create_dir_all(&local_blobs).unwrap();
        fs::write(local_blobs.join("sha256-manicfg"), b"cfg").unwrap();
        fs::write(local_blobs.join("sha256-mani0"), vec![b'x'; 64]).unwrap();

        // A local manifest already at the destination path.
        let dst_manifest = local.join("manifests/registry.ollama.ai/library/mani/1");
        fs::create_dir_all(dst_manifest.parent().unwrap()).unwrap();
        let victim = b"{\"config\":{},\"layers\":[]} the manifest Ollama is reading".to_vec();
        fs::write(&dst_manifest, &victim).unwrap();

        let src_manifest = plan.archive_manifest.clone();
        if !fifoize(&src_manifest) {
            return;
        }

        let cancel: CopyCancel = Arc::new(AtomicBool::new(false));
        let copy = tokio::spawn(copy_model_files(plan, cancel.clone(), CopyActivity::default()));

        let mut w = open_fifo_writer(&src_manifest, Duration::from_secs(60))
            .await
            .expect("cancel is false here, so the copy cannot exit without opening its source");
        w.write_all(&vec![7u8; COPY_CHUNK_BYTES]).unwrap();
        w.flush().unwrap();
        cancel.store(true, Ordering::Relaxed);
        let _ = w.write_all(&[7u8]);
        drop(w);

        let (_err, published) = copy.await.unwrap().unwrap_err();
        assert!(published.is_empty(), "nothing was published: {published:?}");
        assert_eq!(
            fs::read(&dst_manifest).unwrap(),
            victim,
            "an existing local manifest must be byte-identical after a cancelled pull \
             — it must never be truncated or partially overwritten in place"
        );
        assert!(
            leftover_temps(dst_manifest.parent().unwrap()).is_empty(),
            "a cancelled copy must remove its own scratch file"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn timed_out_pull_stops_the_copy_before_cleaning_up() {
        // The end-to-end shape of the fix: on timeout `archive_pull` sets the
        // cancel flag and AWAITS the copy, so by the time it returns the copy has
        // genuinely stopped and cannot resurrect a file cleanup just removed.
        //
        // The FIFO parks the copy inside its read loop, so the timeout provably
        // fires while a copy is in flight — the one thing a 1 ns timeout cannot
        // demonstrate, because there the copy stops before opening anything.
        use std::io::Write as _;
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        let (archive, local) = (base.join("archive"), base.join("local"));
        make_model(&archive, "park", "1", &[64]);
        let fifo = archive.join("blobs/sha256-park0");
        if !fifoize(&fifo) {
            return;
        }

        // Pre-seed the cfg blob locally (shared with another local manifest): the
        // copy skips it, and cleanup must leave it alone.
        let local_blobs = local.join("blobs");
        fs::create_dir_all(&local_blobs).unwrap();
        let victim = local_blobs.join("sha256-parkcfg");
        fs::write(&victim, b"shared with another warm model").unwrap();

        let (a, l) = (archive.clone(), local.clone());
        let pull = tokio::spawn(async move {
            archive_pull(
                "park:1",
                &a,
                &l,
                Duration::from_secs(2),
                &FixedProbe(u64::MAX),
                None,
            )
            .await
        });

        // May be None: if the blocking pool is saturated the pull's own 1 s timeout
        // can set the cancel flag before the closure ever opens its source, and the
        // closure then exits without becoming a reader. Both interleavings must
        // reach the same assertions — in neither may anything be published — so the
        // test never blocks waiting for a reader that will not arrive.
        // Feed one chunk once the copy reaches its source. `open_fifo_writer`
        // probes non-blockingly rather than blocking on a reader, because the copy
        // is not guaranteed to become one: if the cancel flag beats it to its first
        // check it exits without ever opening. Both interleavings must reach the
        // same assertions — in neither may anything be published.
        if let Some(mut w) = open_fifo_writer(&fifo, Duration::from_secs(2)).await {
            // One chunk consumed → the copy is staging, then parks awaiting more.
            w.write_all(&vec![5u8; COPY_CHUNK_BYTES]).unwrap();
            w.flush().unwrap();
            // Outlast the 2 s pull timeout, so the cancel is set while the copy is
            // mid-file, then release it. It must stop rather than publish.
            tokio::time::sleep(Duration::from_millis(3000)).await;
            let _ = w.write_all(&vec![5u8; 4096]);
            drop(w);
        }
        // Safety net: release a reader that only reached `open` after we gave up
        // probing, so no blocking thread is ever left parked on this FIFO.
        drop(try_open_fifo_writer(&fifo));

        let err = pull.await.unwrap().unwrap_err();
        assert!(matches!(err, PullError::Timeout(_)), "got {err:?}");

        // `archive_pull` only returns after the copy has stopped, so the
        // assertions below are stable facts, not a snapshot of a still-moving
        // filesystem. The settle window is therefore redundant for the correct
        // implementation — it exists so this test also fails loudly against a
        // shape that merely DROPS the copy future (where the abandoned writer
        // would go on to publish into the local root shortly after this point).
        tokio::time::sleep(Duration::from_millis(500)).await;
        assert!(
            !local.join("blobs/sha256-park0").exists(),
            "a timed-out pull must not leave (or later publish) its blob locally"
        );
        assert!(
            !local.join("manifests/registry.ollama.ai/library/park/1").exists(),
            "a timed-out pull must not publish a manifest"
        );
        assert!(victim.is_file(), "the pre-existing shared blob must survive");
        assert_eq!(
            fs::read(&victim).unwrap(),
            b"shared with another warm model",
            "and must be byte-identical — nothing is written in place"
        );
        assert!(
            leftover_temps(&local_blobs).is_empty(),
            "a stopped copy must remove its own scratch file"
        );
    }

    #[tokio::test]
    async fn concurrent_pulls_same_model_copy_once() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        let model = make_model(&base.join("archive"), "dup", "1", &[256]);
        let mut reg = reg_with(base, vec![]);
        reg.reconcile();
        assert_eq!(reg.get(&model).unwrap().tier, StorageTier::Cold);

        let registry = Arc::new(Mutex::new(reg));
        let probe_calls = Arc::new(AtomicUsize::new(0));
        let coord = PullCoordinator::with_probe(
            registry.clone(),
            Duration::from_secs(30),
            Arc::new(CountingSlowProbe(probe_calls.clone())),
        );

        let c1 = coord.clone();
        let c2 = coord.clone();
        let m1 = model.clone();
        let m2 = model.clone();
        let h1 = tokio::spawn(async move { c1.ensure_local(&m1, None).await });
        let h2 = tokio::spawn(async move { c2.ensure_local(&m2, None).await });
        h1.await.unwrap().unwrap();
        h2.await.unwrap().unwrap();

        // The disk probe (run once per actual copy) fired exactly once → single copy.
        assert_eq!(probe_calls.load(Ordering::SeqCst), 1, "model copied exactly once");
        assert_eq!(registry.lock().await.get(&model).unwrap().tier, StorageTier::Warm);
    }

    #[tokio::test]
    async fn ensure_local_updates_last_requested() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        make_model(&base.join("local"), "req", "1", &[10]);
        let mut reg = reg_with(base, vec![]);
        reg.reconcile();
        // Clear last_requested to prove ensure_local sets it.
        let registry = Arc::new(Mutex::new(reg));
        let coord = PullCoordinator::with_probe(
            registry.clone(),
            Duration::from_secs(5),
            Arc::new(UnknownProbe),
        );
        coord.ensure_local("req:1", None).await.unwrap();
        assert!(registry.lock().await.get("req:1").unwrap().last_requested.unwrap() > 0);
    }

    #[tokio::test]
    async fn progress_event_emitted_on_pull() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        make_model(&base.join("archive"), "ev", "1", &[1024]);
        let (tx, mut rx) = mpsc::unbounded_channel();
        archive_pull(
            "ev:1",
            &base.join("archive"),
            &base.join("local"),
            Duration::from_secs(30),
            &FixedProbe(u64::MAX),
            Some(&tx),
        )
        .await
        .unwrap();
        let ev = rx.try_recv().unwrap();
        match ev {
            PullEvent::RetrievingFromArchive { model, .. } => assert_eq!(model, "ev:1"),
            other => panic!("unexpected event {other:?}"),
        }
    }

    #[test]
    fn pull_event_serializes_tagged() {
        let ev = PullEvent::RetrievingFromArchive { model: "m:1".into(), size_gb: 1.5 };
        let json = serde_json::to_string(&ev).unwrap();
        assert!(json.contains("\"type\":\"retrieving_from_archive\""), "{json}");
        let ev2 = PullEvent::LoadingIntoVram { model: "m:1".into() };
        let json2 = serde_json::to_string(&ev2).unwrap();
        assert!(json2.contains("\"type\":\"loading_into_vram\""), "{json2}");
    }
}
