//! TIER-03: disk-pressure eviction (warm → cold).
//!
//! When the local Ollama disk crosses a used-percentage threshold
//! (`MODEL_DISK_PRESSURE_PERCENT`, default 80%), the least-recently-requested
//! **warm** models are archived back to the cold tier (e.g. NFS) until usage
//! drops below the threshold. This is the reverse of TIER-02's archive pull:
//! the model's manifest + referenced blobs are copied from the local root to the
//! archive root, the archive copy is **verified**, and only then is the local
//! copy removed.
//!
//! ## Safety invariants
//! - **Never evict** Hot (VRAM-resident), protected, or non-Warm models.
//! - **Archive-first, delete-after:** local files are deleted only after every
//!   referenced blob + the manifest are confirmed present in the archive with a
//!   matching size. A failed/partial archive copy leaves the model warm.
//! - **No archive ⇒ no eviction:** if the archive root isn't mounted/present we
//!   skip the sweep entirely (evicting with nowhere to put the data would lose
//!   it).
//! - **GC-aware local removal:** a blob is deleted locally only if no *other*
//!   local manifest still references it (content-addressed blobs are shared).
//! - **Disk-op lock:** a sweep and an archive pull share a global async mutex so
//!   their destructive filesystem operations never interleave.
//!
//! Nothing here hardcodes infrastructure — all paths come from the registry /
//! config; model names come from the registry records.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::Mutex;
use tracing::{info, warn};

use super::registry::{
    collect_manifest_leaves, parse_manifest_blobs, ManifestBlobs, ModelRegistry, StorageTier,
};
use super::transfer::{
    blob_filename, cleanup_after_join_error, cleanup_attempt, copy_file_cancellable,
    find_manifest_leaf, CopyActivity, CopyCancel, DiskSpaceProbe, CANCELLED, COPY_CANCEL_GRACE,
    COPY_TEMP_INFIX,
};

/// Label used in `cleanup_attempt` logs for the warm→cold direction.
const EVICT_OP: &str = "archive eviction (warm→cold)";

const BYTES_PER_GB: f64 = 1_073_741_824.0; // 1 GiB
const SECS_PER_HOUR: i64 = 3_600;

fn bytes_to_gb(bytes: u64) -> f64 {
    bytes as f64 / BYTES_PER_GB
}

/// Current wall-clock time in epoch seconds. Isolated in one place so the
/// cooldown decision can be exercised with an injected `now` in tests (the
/// production sweep passes this value through).
fn now_epoch_secs() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Shared global lock serialising destructive disk operations (eviction sweeps,
/// archive pulls, orphan GC). Held for the duration of a single model's eviction
/// copy + local removal (and, since the S111 B1 fix, across an archive pull's
/// copy phase) so those never race the same blobs.
///
/// ## Canonical lock-acquisition order (S111 — deadlock avoidance)
///
/// There are two async mutexes that some paths need together:
///   1. `DiskOpLock` (this lock)
///   2. the registry `Arc<Mutex<ModelRegistry>>`
///
/// **The canonical order is: `DiskOpLock` FIRST, then the registry lock.** Every
/// path that holds both acquires them in this order:
///   - `evict_to_archive` — caller holds `DiskOpLock`; it takes the registry
///     lock only briefly (snapshot, then mark-evicted+save), never across the
///     NFS blob copy.
///   - `cooldown_pass` / `run_eviction_sweep_at` disk-pressure pass — take
///     `DiskOpLock`, then call `evict_to_archive` (registry-brief).
///   - `evict_for_space` (pre-pull) — takes `DiskOpLock`, then registry-brief.
///   - `gc::run_gc` — takes `DiskOpLock`, no registry lock at all.
///   - `PullCoordinator::ensure_local` — takes `DiskOpLock` across the pull copy,
///     then the registry lock briefly to promote (disk_op → registry).
///
/// Paths that take ONLY the registry lock (never `DiskOpLock`) — and therefore
/// cannot invert the order — include the sweep's reconcile-apply block and the
/// `/api/models/reconcile` handler (both scan the filesystem OFF any lock, then
/// apply under a brief registry lock), plus `chat_completions` /
/// `update_last_requested`. No path acquires the registry lock and THEN
/// `DiskOpLock`; keep it that way.
pub type DiskOpLock = Arc<Mutex<()>>;

/// Create a fresh disk-operation lock to share between the eviction sweep and the
/// pull coordinator.
pub fn new_disk_op_lock() -> DiskOpLock {
    Arc::new(Mutex::new(()))
}

// ── Errors ───────────────────────────────────────────────────────────────────

/// Why an [`evict_to_archive`] call did not archive + remove a model.
#[derive(Debug, thiserror::Error)]
pub enum EvictError {
    /// The model is unknown to the registry.
    #[error("model not found in registry: {0}")]
    UnknownModel(String),
    /// The model is not in the Warm tier (Hot or Cold) and cannot be evicted.
    #[error("model {0} is not warm; refusing to evict")]
    NotWarm(String),
    /// The model is protected and is never auto-archived.
    #[error("model {0} is protected; refusing to evict")]
    Protected(String),
    /// The model's local manifest could not be located.
    #[error("local manifest not found for {0}")]
    MissingLocalManifest(String),
    /// Copying the model to the archive failed (I/O error). Local copy untouched.
    #[error("archive copy failed for {0}: {1}")]
    ArchiveCopy(String, String),
    /// MSM-02: the archive copy did not finish within `MODEL_ARCHIVE_COPY_TIMEOUT_SECS`
    /// (e.g. a stalled NFS write). Any partial archive files this copy wrote are
    /// cleaned up; the local copy is untouched and the model stays Warm for retry on
    /// the next sweep.
    #[error("archive copy for {0} timed out after {1:?}")]
    Timeout(String, Duration),
    /// The archive copy did not verify (a blob/manifest missing or size
    /// mismatch). Local copy is intentionally left in place.
    #[error("archive copy verification failed for {0}: {1}")]
    VerifyFailed(String, String),
    /// Removing the local copy failed after a verified archive copy.
    #[error("local removal failed for {0}: {1}")]
    LocalRemove(String, String),
}

// ── Local removal (injectable for tests) ───────────────────────────────────────

/// Removes a model's files from the local Ollama root. Injectable so tests can
/// assert eviction ordering / verify-before-delete without touching real files,
/// and so the production filesystem removal is swappable.
#[async_trait]
pub trait LocalEvictor: Send + Sync {
    /// Remove the named model's local manifest and any blobs it *exclusively*
    /// owns (GC-aware). Must not delete blobs still referenced by other local
    /// manifests.
    async fn remove(&self, model: &str) -> Result<(), String>;
}

/// Production [`LocalEvictor`]: GC-aware filesystem removal under a local Ollama
/// root. Deletes the model's manifest leaf, then each referenced blob *iff* no
/// other local manifest references it. Never shells out to `ollama`.
pub struct FsLocalEvictor {
    local_root: PathBuf,
}

impl FsLocalEvictor {
    /// Build an evictor rooted at the local Ollama models directory.
    pub fn new(local_root: PathBuf) -> Self {
        Self { local_root }
    }
}

#[async_trait]
impl LocalEvictor for FsLocalEvictor {
    async fn remove(&self, model: &str) -> Result<(), String> {
        let root = self.local_root.clone();
        let model = model.to_string();
        // Filesystem walk + removals are blocking; run off the async reactor.
        tokio::task::spawn_blocking(move || fs_remove_model(&root, &model))
            .await
            .map_err(|e| format!("join error: {e}"))?
    }
}

/// GC-aware removal of `model` under `root` (a local OR an archive Ollama root).
///
/// Deletes the model's manifest leaf, then each referenced blob *iff* no other
/// manifest under the same `root` references it (content-addressed blobs are
/// shared). Pointed at the local root this is warm-eviction's local delete;
/// pointed at the archive root it is TIER-05 cold-quota's GC-aware archive
/// delete — the SAME primitive, so cold pruning never re-invents blob-sharing
/// logic and Chord stays the single deletion authority for both tiers.
pub(crate) fn fs_remove_model(local_root: &Path, model: &str) -> Result<(), String> {
    let manifest = find_manifest_leaf(local_root, model)
        .ok_or_else(|| format!("local manifest not found for {model}"))?;
    let blobs = parse_manifest_blobs(&manifest);

    // Blobs referenced by EVERY OTHER local manifest (so we never delete a shared
    // blob). Computed before we delete this model's manifest.
    let others = referenced_by_other_manifests(local_root, &manifest);

    // Delete this model's manifest first so it no longer references the blobs.
    std::fs::remove_file(&manifest)
        .map_err(|e| format!("remove manifest {}: {e}", manifest.display()))?;

    let blobs_dir = local_root.join("blobs");
    for digest in &blobs.digests {
        if others.contains(digest) {
            // Shared with another local model → keep.
            continue;
        }
        let path = blobs_dir.join(blob_filename(digest));
        if path.exists() {
            if let Err(e) = std::fs::remove_file(&path) {
                // Non-fatal: the manifest is already gone (model is unloadable);
                // a leftover blob is just wasted space, not corruption.
                warn!(path = %path.display(), error = %e, "failed to remove local blob during eviction");
            }
        }
    }
    Ok(())
}

/// Set of blob digests referenced by every manifest under `local_root` *except*
/// `exclude`. Shared by warm eviction (local root) and TIER-05 cold-quota
/// pruning (archive root) so both compute blob-sharing identically.
pub(crate) fn referenced_by_other_manifests(local_root: &Path, exclude: &Path) -> HashSet<String> {
    let mut referenced = HashSet::new();
    for leaf in collect_manifest_leaves(local_root) {
        if leaf == exclude {
            continue;
        }
        for d in parse_manifest_blobs(&leaf).digests {
            referenced.insert(d);
        }
    }
    referenced
}

// ── Disk-pressure check ─────────────────────────────────────────────────────────

/// Whether local disk usage exceeds `threshold_pct` percent.
///
/// Uses the injected [`DiskSpaceProbe`]: `used% = (total − free) / total * 100`.
/// If total or free can't be determined the probe is treated as "no pressure"
/// (returns `false`) so a probe failure never triggers destructive eviction.
pub fn check_disk_pressure(local_path: &Path, threshold_pct: u8, probe: &dyn DiskSpaceProbe) -> bool {
    let target = crate::models::transfer::nearest_existing_ancestor(local_path);
    let (Some(total), Some(free)) = (probe.total_bytes(&target), probe.available_bytes(&target))
    else {
        return false;
    };
    if total == 0 {
        return false;
    }
    let used = total.saturating_sub(free);
    // used% > threshold (strictly above, matching "exceeds threshold").
    (used as u128) * 100 > (threshold_pct as u128) * (total as u128)
}

// `nearest_existing_ancestor` is shared from the transfer module (used via its
// fully-qualified path below) so the sweep and the pre-pull path stay in sync.

// ── Single-model eviction (warm → cold) ─────────────────────────────────────────

/// Result of a successful eviction: the freed size in bytes.
#[derive(Debug)]
pub struct Evicted {
    /// Bytes the model occupied locally (manifest blob total).
    pub freed_bytes: u64,
}

/// Evict one warm model to the archive (warm → cold).
///
/// Steps:
/// 1. Validate the model is Warm, non-protected (else a typed error/skip).
/// 2. Copy its manifest + referenced blobs local → archive (skipping blobs that
///    already exist in the archive with a matching size), bounded by
///    `copy_timeout` (MSM-02). Each file is staged in a temp file and published
///    with `rename`, so the archive only ever contains complete files and an
///    existing archive blob is never mutated in place. On timeout or a mid-copy
///    I/O error the blobs this attempt published are reclaimed (best-effort) and
///    the model is left Warm — never Cold, and the local copy is never touched.
///    On timeout the copy is **cancelled and awaited** before cleanup runs, so
///    cleanup can never be undone by a copy that is still in flight.
/// 3. **Verify** every referenced blob + the manifest exist in the archive with
///    matching sizes. On failure: do NOT delete locally; return [`EvictError`].
/// 4. Remove the local copy via the injected [`LocalEvictor`] (GC-aware).
/// 5. Update the registry (tier → Cold, local_path = None, archive_path set) and
///    `save()` (non-fatal on error).
///
/// The registry is locked only for short snapshots, never across the copy.
pub async fn evict_to_archive(
    registry: &Arc<Mutex<ModelRegistry>>,
    model: &str,
    evictor: &dyn LocalEvictor,
    copy_timeout: Duration,
) -> Result<Evicted, EvictError> {
    // ── Snapshot tier/paths + validate under a short lock ──
    let (local_root, archive_root) = {
        let reg = registry.lock().await;
        let rec = reg
            .get(model)
            .ok_or_else(|| EvictError::UnknownModel(model.to_string()))?;
        if reg.is_protected(model) {
            return Err(EvictError::Protected(model.to_string()));
        }
        match rec.tier {
            StorageTier::Warm => {}
            _ => return Err(EvictError::NotWarm(model.to_string())),
        }
        let local_root = rec
            .local_path
            .clone()
            .map(PathBuf::from)
            .unwrap_or_else(|| reg.local_path().to_path_buf());
        let archive_root = rec
            .archive_path
            .clone()
            .map(PathBuf::from)
            .unwrap_or_else(|| reg.archive_path().to_path_buf());
        (local_root, archive_root)
    };

    let local_manifest = find_manifest_leaf(&local_root, model)
        .ok_or_else(|| EvictError::MissingLocalManifest(model.to_string()))?;
    let blobs = parse_manifest_blobs(&local_manifest);
    let planned = planned_archive_paths(&blobs, &local_manifest, &local_root, &archive_root);
    // Snapshot which of the planned archive paths ALREADY exist before we start
    // copying — e.g. a blob shared with another already-archived model.
    //
    // This snapshot governs DELETION only. It is not, and no longer needs to be,
    // ownership evidence for writing: `copy_file_cancellable` stages every copy
    // in a per-attempt temp file and publishes it with `rename`, so an existing
    // archive file is never mutated in place and a failed attempt cannot corrupt
    // one. What the snapshot still buys is the "never delete a blob referenced
    // elsewhere" invariant (mirroring the local side): if a path was already
    // there, it is someone else's data — possibly the only copy of a blob another
    // archived model depends on — and this attempt must not remove it even though
    // it may have atomically replaced it with a correct copy of the same digest.
    //
    // For that deletion rule the snapshot is only accurate while nothing else can
    // write to the archive. Every production caller of `evict_to_archive`
    // (`cooldown_pass`, the `run_eviction_sweep_at` disk-pressure pass,
    // `evict_for_space`) holds `DiskOpLock` across this whole function, as do
    // `gc::run_gc` and `PullCoordinator::ensure_local` — so nothing can create an
    // archive file between this snapshot and the inline cleanup below, which runs
    // before we return and therefore before the lock is released. If evictions
    // are ever parallelised, or the lock dropped around a copy, this snapshot
    // stops being accurate and `cleanup_attempt` must stop being used.
    //
    // Cleanup that would run AFTER the lock is released cannot rely on it at all,
    // and therefore never deletes — see the cancellation-grace branch below.
    let pre_existing: HashSet<PathBuf> = planned.iter().filter(|p| p.exists()).cloned().collect();

    // ── Copy local → archive (reverse pull), bounded by copy_timeout (MSM-02) ──
    //
    // The copy runs as an owned task with a cancellation flag rather than as a
    // bare future handed to `tokio::time::timeout`. Timing out on a future only
    // *drops* it, and a dropped `tokio::fs::copy` keeps running on the blocking
    // pool — so the old shape ran its cleanup while the copy was still live and
    // could still create the file being cleaned up. See `CopyCancel`.
    let cancel: CopyCancel = Arc::new(AtomicBool::new(false));
    // Proves the blocking writer has stopped on the one path where awaiting the
    // task does not — see `transfer::cleanup_after_join_error`.
    let activity = CopyActivity::default();
    let mut copy_task = tokio::spawn(copy_model_to_archive(
        local_root.clone(),
        local_manifest.clone(),
        archive_root.clone(),
        cancel.clone(),
        activity.clone(),
    ));

    let copy_res = match tokio::time::timeout(copy_timeout, &mut copy_task).await {
        Ok(joined) => joined,
        Err(_elapsed) => {
            // Timed out (e.g. a stalled NFS write). Ask the copy to stop, then
            // WAIT until it has genuinely stopped before touching the archive —
            // cleanup must happen-after the last write, not merely after the
            // timer fired. The disk-op lock held by the caller
            // (cooldown_pass/sweep loop) is released when this function returns,
            // so the sweep is never wedged.
            cancel.store(true, Ordering::Relaxed);
            match tokio::time::timeout(COPY_CANCEL_GRACE, &mut copy_task).await {
                // Stopped early: we have the precise list of what it published.
                // Still inside the caller's DiskOpLock, so the snapshot holds.
                Ok(Ok(Err((_, published)))) => {
                    cleanup_attempt(EVICT_OP, model, &published, &pre_existing)
                }
                // Finished (or panicked) before it saw the cancel: we don't have
                // an exact list, so fall back to every planned archive path that
                // did not exist before this attempt started.
                Ok(Ok(Ok(()))) | Ok(Err(_)) => {
                    cleanup_attempt(EVICT_OP, model, &planned, &pre_existing)
                }
                Err(_) => {
                    // The copy is wedged inside a single chunk write (only
                    // reachable on a stalled mount). We must not block the sweep
                    // on it — so we return, and the caller's DiskOpLock is
                    // released while this copy is still alive.
                    //
                    // **Therefore we delete nothing, now or later.** Once the lock
                    // is gone a subsequent eviction may legitimately create — or
                    // come to depend on — any of our planned paths: it could write
                    // one itself, or skip-match one this attempt published and then
                    // delete its local copy trusting it. A deferred cleanup
                    // carrying this attempt's stale `pre_existing` snapshot would
                    // happily delete that file. Ownership is not provable from
                    // here, and the asymmetry is not close: an orphan blob costs
                    // disk and is reconciled by the next eviction of this model
                    // under the lock (it becomes `pre_existing`, and is either
                    // size-skipped as valid content-addressed data or replaced),
                    // whereas a wrongly deleted archive blob is unrecoverable.
                    //
                    // Thanks to temp-file staging this now costs very little: the
                    // abandoned writer is confined to its own temp path, cannot
                    // mutate any archive file, and will not publish at all once it
                    // observes the cancel flag. Worst case it leaves one
                    // `COPY_TEMP_INFIX` scratch file, which is reapable by name.
                    // The task is left detached rather than aborted — it stops
                    // itself at its next chunk check and removes its own temp.
                    warn!(
                        model = %model,
                        grace_secs = COPY_CANCEL_GRACE.as_secs(),
                        temp_infix = COPY_TEMP_INFIX,
                        "archive copy did not stop within the cancellation grace period; \
                         NOT cleaning up — ownership of these archive paths can no longer be \
                         proven once the disk-op lock is released. The abandoned copy cannot \
                         corrupt or publish anything; it may leave one scratch file behind."
                    );
                }
            }
            return Err(EvictError::Timeout(model.to_string(), copy_timeout));
        }
    };

    match copy_res {
        Ok(Ok(())) => {}
        Ok(Err((e, published))) => {
            // Mid-copy failure: reclaim the blobs this attempt published, through
            // the SAME route as the timeout path. `published` can contain a path
            // that already existed (a shared archive blob whose size did not
            // match, which this attempt atomically replaced); removing that would
            // destroy data this attempt did not create, so `cleanup_attempt`
            // filters it out. Two cleanup routes with different safety rules is
            // how one of them ends up wrong — there is exactly one route.
            cleanup_attempt(EVICT_OP, model, &published, &pre_existing);
            return Err(EvictError::ArchiveCopy(model.to_string(), e));
        }
        Err(join_err) => {
            // The copy task panicked (or was aborted), which stops the OUTER task
            // but not its inner `spawn_blocking` closure. Cancel, then prove the
            // writer stopped before deleting anything — and delete nothing if it
            // cannot be proven. A late publish here is more benign than on the
            // pull side (it restores a complete, content-addressed archive blob
            // rather than a manifest over removed blobs), but the rule is the
            // same one, through the same shared route.
            cleanup_after_join_error(
                EVICT_OP,
                model,
                &cancel,
                &activity,
                COPY_CANCEL_GRACE,
                &planned,
                &pre_existing,
                // Eviction (warm→cold) never stashes a manifest — that machinery
                // is pull-side only (see the doc comment on
                // `cleanup_after_join_error`) — so an empty `manifest_rel` makes
                // the restore step a guaranteed no-op here.
                &archive_root,
                Path::new(""),
            )
            .await;
            return Err(EvictError::ArchiveCopy(
                model.to_string(),
                format!("archive copy task failed to join: {join_err}"),
            ));
        }
    }

    // ── Verify the archive copy BEFORE any local deletion ──
    let freed_bytes = verify_archive_copy(model, &local_manifest, &archive_root)
        .map_err(|e| EvictError::VerifyFailed(model.to_string(), e))?;

    // ── Remove local copy (GC-aware, injectable) ──
    evictor
        .remove(model)
        .await
        .map_err(|e| EvictError::LocalRemove(model.to_string(), e))?;

    // ── Update registry (non-fatal save) ──
    {
        let mut reg = registry.lock().await;
        let archive_str = archive_root.to_string_lossy().to_string();
        reg.mark_evicted_to_archive(model, &archive_str);
        if let Err(e) = reg.save() {
            warn!("failed to persist registry after evicting {model}: {e}");
        }
    }

    Ok(Evicted { freed_bytes })
}

/// Copy a model's manifest + referenced blobs from the local root to the archive
/// root, preserving the `manifests/.../<tag>` + `blobs/sha256-…` layout. Blobs
/// already present in the archive with a matching size are skipped (and never
/// added to the returned `published` list, so a failure never causes a
/// pre-existing/shared archive blob to be cleaned up). Blobs are copied before
/// the manifest so the archive never has a manifest whose blobs are missing.
///
/// On error, returns `(message, published)` where `published` is every archive
/// path this call successfully **published** — used by [`evict_to_archive`] to
/// reclaim them (MSM-02). Each copy is staged in a temp file and `rename`d into
/// place, so a path that is not in `published` was not modified at all: the list
/// is exact, not a best guess, and there is no partial file anywhere to find.
///
/// `cancel` makes the copy stoppable: it is checked between blobs, between 4 MiB
/// chunks within a blob, and once more immediately before publishing — so a
/// cancelled copy never surfaces a blob into the archive. Observing cancellation
/// is reported as an error carrying the `published` list, like any other failure.
///
/// Takes owned paths so it can be driven as a `tokio::spawn`ed task the caller
/// can await after cancelling.
async fn copy_model_to_archive(
    local_root: PathBuf,
    local_manifest: PathBuf,
    archive_root: PathBuf,
    cancel: CopyCancel,
    activity: CopyActivity,
) -> Result<(), (String, Vec<PathBuf>)> {
    let mut published: Vec<PathBuf> = Vec::new();
    let blobs = parse_manifest_blobs(&local_manifest);
    let local_blobs = local_root.join("blobs");
    let archive_blobs = archive_root.join("blobs");
    if let Err(e) = tokio::fs::create_dir_all(&archive_blobs).await {
        return Err((format!("create archive blobs dir: {e}"), published));
    }

    for digest in &blobs.digests {
        let fname = blob_filename(digest);
        let src = local_blobs.join(&fname);
        let dst = archive_blobs.join(&fname);
        // Skip if already in archive with matching size (content-addressed →
        // identical content for the same digest + size). NOT added to
        // `published` — a pre-existing archive blob (possibly shared with another
        // already archived model) must never be deleted by this copy's cleanup.
        if let (Ok(s), Ok(d)) = (tokio::fs::metadata(&src).await, tokio::fs::metadata(&dst).await) {
            if s.len() == d.len() {
                continue;
            }
        }
        match copy_file_cancellable(&src, &dst, &cancel, &activity).await {
            Ok(true) => published.push(dst),
            // Cancelled or failed → `dst` was never touched (the copy only ever
            // wrote its own temp file, which it has already removed). Nothing to
            // record and nothing to clean up for this path.
            Ok(false) => return Err((CANCELLED.to_string(), published)),
            Err(e) => {
                return Err((format!("copy blob {fname}: {e}"), published));
            }
        }
    }

    // Manifest leaf — mirror its path relative to the local manifests root.
    let local_manifests = local_root.join("manifests");
    let rel = match local_manifest.strip_prefix(&local_manifests) {
        Ok(r) => r,
        Err(_) => {
            return Err((
                "local manifest path outside manifests root".to_string(),
                published,
            ))
        }
    };
    let dst_manifest = archive_root.join("manifests").join(rel);
    if let Some(parent) = dst_manifest.parent() {
        if let Err(e) = tokio::fs::create_dir_all(parent).await {
            return Err((format!("create archive manifest dir: {e}"), published));
        }
    }
    match copy_file_cancellable(&local_manifest, &dst_manifest, &cancel, &activity).await {
        Ok(true) => published.push(dst_manifest),
        Ok(false) => return Err((CANCELLED.to_string(), published)),
        Err(e) => {
            return Err((format!("copy manifest: {e}"), published));
        }
    }
    Ok(())
}

/// Every archive path a warm→cold copy of `local_manifest`'s model *would*
/// create (every referenced blob + the manifest), used for best-effort cleanup
/// when the precise `published` list is unavailable (the copy task finished or
/// panicked rather than reporting back). It is only ever consumed by
/// [`cleanup_attempt`], which drops any path that already existed before this
/// attempt started, so a pre-existing/shared archive blob is never removed via
/// this list.
///
/// Note this is a *superset* fallback, not evidence of what was written. Because
/// every copy is published atomically, none of these paths can be half-written;
/// the only question cleanup answers is which complete blobs to reclaim.
fn planned_archive_paths(
    blobs: &ManifestBlobs,
    local_manifest: &Path,
    local_root: &Path,
    archive_root: &Path,
) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    let archive_blobs = archive_root.join("blobs");
    for digest in &blobs.digests {
        paths.push(archive_blobs.join(blob_filename(digest)));
    }
    let local_manifests = local_root.join("manifests");
    if let Ok(rel) = local_manifest.strip_prefix(&local_manifests) {
        paths.push(archive_root.join("manifests").join(rel));
    }
    paths
}

/// Verify the archive holds a complete copy of the model: the manifest exists,
/// and every referenced blob exists in the archive with a size matching the
/// local source blob. Returns the total verified size (bytes) on success.
pub(crate) fn verify_archive_copy(
    _model: &str,
    local_manifest: &Path,
    archive_root: &Path,
) -> Result<u64, String> {
    let blobs = parse_manifest_blobs(local_manifest);
    let archive_blobs = archive_root.join("blobs");

    // Local blobs dir: the nearest ancestor of the manifest that contains a
    // `blobs/` directory (the local Ollama root). Used only for size comparison.
    let local_blobs_dir = local_manifest
        .ancestors()
        .find(|a| a.join("blobs").is_dir())
        .map(|a| a.join("blobs"));

    let mut total = 0u64;
    for digest in &blobs.digests {
        let fname = blob_filename(digest);
        let archive_path = archive_blobs.join(&fname);
        let amd = std::fs::metadata(&archive_path)
            .map_err(|_| format!("archive blob missing: {fname}"))?;
        // Compare against the local source size when available.
        if let Some(ref lbd) = local_blobs_dir {
            if let Ok(lmd) = std::fs::metadata(lbd.join(&fname)) {
                if lmd.len() != amd.len() {
                    return Err(format!(
                        "archive blob size mismatch for {fname}: local {} != archive {}",
                        lmd.len(),
                        amd.len()
                    ));
                }
            }
        }
        total += amd.len();
    }

    // Manifest present in archive (mirror the relative path).
    // Find the local manifests root by walking up to the dir literally named
    // "manifests".
    let mut manifests_root: Option<&Path> = None;
    for anc in local_manifest.ancestors() {
        if anc.file_name().map(|n| n == "manifests").unwrap_or(false) {
            manifests_root = Some(anc);
            break;
        }
    }
    let manifests_root =
        manifests_root.ok_or_else(|| "local manifest not under a manifests/ dir".to_string())?;
    let rel = local_manifest
        .strip_prefix(manifests_root)
        .map_err(|_| "manifest rel-path error".to_string())?;
    let archive_manifest = archive_root.join("manifests").join(rel);
    if !archive_manifest.is_file() {
        return Err(format!(
            "archive manifest missing: {}",
            archive_manifest.display()
        ));
    }

    Ok(total)
}

// ── Sweep ────────────────────────────────────────────────────────────────────

/// TIER-04 cooldown pass: archive every warm, non-protected model that has been
/// idle longer than `cooldown_hours`, regardless of disk pressure.
///
/// - `cooldown_hours == 0` → cooldown eviction disabled (returns immediately;
///   the startup warning covers the operator-facing notice).
/// - `last_requested == None` (legacy / never requested) → treated as
///   infinitely idle → eligible.
/// - Protected / Hot models are already excluded by `warm_eviction_candidates`
///   and re-checked by `evict_to_archive`.
///
/// Holds the shared disk-op lock for the duration so it can't race a pull or the
/// disk-pressure pass. Failed candidates are logged and skipped (the model stays
/// warm); they are simply retried on the next sweep.
async fn cooldown_pass(
    registry: &Arc<Mutex<ModelRegistry>>,
    cooldown_hours: u64,
    now_secs: i64,
    evictor: &dyn LocalEvictor,
    disk_op_lock: &DiskOpLock,
    copy_timeout: Duration,
) {
    if cooldown_hours == 0 {
        return; // cooldown eviction disabled
    }
    let cooldown_secs = (cooldown_hours as i64).saturating_mul(SECS_PER_HOUR);

    // Snapshot the warm candidates and decide eligibility up front (the set only
    // shrinks as we evict, so a single snapshot is sufficient and avoids holding
    // the registry lock across the copy).
    let candidates: Vec<(String, i64)> = {
        let reg = registry.lock().await;
        reg.warm_eviction_candidates()
            .into_iter()
            .filter_map(|(name, last_requested, _)| {
                // None ⇒ infinitely idle. Otherwise idle = now - last_requested.
                let idle_secs = match last_requested {
                    Some(ts) => now_secs.saturating_sub(ts),
                    None => i64::MAX,
                };
                (idle_secs > cooldown_secs).then_some((name, idle_secs))
            })
            .collect()
    };
    if candidates.is_empty() {
        return;
    }

    // Serialise destructive ops with pulls / the disk-pressure pass.
    let _guard = disk_op_lock.lock().await;
    for (name, idle_secs) in candidates {
        let hours_idle = if idle_secs == i64::MAX {
            // Never requested — report the configured cooldown as a floor rather
            // than an absurd MAX/3600 value.
            cooldown_hours as i64
        } else {
            idle_secs / SECS_PER_HOUR
        };
        match evict_to_archive(registry, &name, evictor, copy_timeout).await {
            Ok(ev) => {
                info!("cooldown_eviction model={} idle_hours={}", name, hours_idle);
                let _ = ev; // freed bytes not surfaced for cooldown evictions
            }
            Err(e) => {
                warn!(model = %name, error = %e, "cooldown eviction candidate failed; leaving warm");
            }
        }
    }
}

/// Run one eviction sweep: a TIER-04 cooldown pass (always) followed by a
/// TIER-03 disk-pressure pass (only if still over threshold).
///
/// - If the archive root is not present/mounted → warn + skip the whole sweep
///   (data safety — evicting with nowhere to put the data would lose it).
/// - **Cooldown pass (always):** every warm, non-protected model whose
///   `last_requested` is older than `cooldown_hours` (None ⇒ treated as
///   infinitely idle) is archived (warm → cold), regardless of disk pressure.
///   `cooldown_hours == 0` disables this pass entirely.
/// - **Disk-pressure pass:** if disk usage is still above `threshold` after the
///   cooldown pass, evict warm, non-protected, non-Hot models LRU-first,
///   re-checking pressure after each, until below threshold or no candidates
///   remain. If still over pressure with no candidates → warn (disk alert).
///
/// Both passes reuse the same verify-before-delete / GC-aware / disk-op-lock
/// safety as [`evict_to_archive`].
pub async fn run_eviction_sweep(
    registry: &Arc<Mutex<ModelRegistry>>,
    threshold_pct: u8,
    cooldown_hours: u64,
    probe: &dyn DiskSpaceProbe,
    evictor: &dyn LocalEvictor,
    disk_op_lock: &DiskOpLock,
    copy_timeout: Duration,
) {
    run_eviction_sweep_at(
        registry,
        threshold_pct,
        cooldown_hours,
        now_epoch_secs(),
        probe,
        evictor,
        disk_op_lock,
        copy_timeout,
    )
    .await
}

/// Sweep with an injected `now_secs` so the cooldown decision is deterministic
/// in tests. Production calls go through [`run_eviction_sweep`] which supplies
/// the real wall-clock time.
#[allow(clippy::too_many_arguments)]
pub async fn run_eviction_sweep_at(
    registry: &Arc<Mutex<ModelRegistry>>,
    threshold_pct: u8,
    cooldown_hours: u64,
    now_secs: i64,
    probe: &dyn DiskSpaceProbe,
    evictor: &dyn LocalEvictor,
    disk_op_lock: &DiskOpLock,
    copy_timeout: Duration,
) {
    // Snapshot paths under a short lock.
    let (local_root, archive_root) = {
        let reg = registry.lock().await;
        (
            reg.local_path().to_path_buf(),
            reg.archive_path().to_path_buf(),
        )
    };

    // Data safety: never evict if we can't reach the archive.
    if !archive_root.exists() {
        warn!(
            archive_path = %archive_root.display(),
            "archive path not present / not mounted; skipping eviction sweep"
        );
        return;
    }

    // ── Cooldown pass (always runs, independent of disk pressure) ──
    cooldown_pass(registry, cooldown_hours, now_secs, evictor, disk_op_lock, copy_timeout).await;

    // ── Disk-pressure pass (only if still over threshold) ──
    if !check_disk_pressure(&local_root, threshold_pct, probe) {
        return;
    }

    // Serialise destructive ops with archive pulls.
    let _guard = disk_op_lock.lock().await;

    // Candidates whose eviction failed this sweep — skipped on later iterations so
    // a persistently-failing model isn't retried every round (a successful
    // eviction shrinks the candidate set, but a failing one would otherwise keep
    // reappearing at the head of the LRU list and waste work each pass).
    let mut failed: HashSet<String> = HashSet::new();

    loop {
        // Re-check candidates each iteration (the set shrinks as we evict),
        // excluding any that already failed this sweep.
        let candidates: Vec<String> = {
            let reg = registry.lock().await;
            reg.warm_eviction_candidates()
                .into_iter()
                .map(|(name, _, _)| name)
                .filter(|name| !failed.contains(name))
                .collect()
        };
        if candidates.is_empty() {
            warn!(
                threshold_pct,
                "disk pressure above threshold but no evictable warm models remain (all hot/protected/failed); disk pressure alert"
            );
            return;
        }

        // Evict the LRU candidate. On failure (e.g. verify failed) record it so we
        // don't retry it, and try the next.
        let mut evicted_any = false;
        for name in candidates {
            match evict_to_archive(registry, &name, evictor, copy_timeout).await {
                Ok(ev) => {
                    info!(
                        "disk_pressure_eviction model={} freed_gb={:.2}",
                        name,
                        bytes_to_gb(ev.freed_bytes)
                    );
                    evicted_any = true;
                    break;
                }
                Err(e) => {
                    warn!(model = %name, error = %e, "eviction candidate failed; skipping for this sweep");
                    failed.insert(name);
                    continue;
                }
            }
        }

        if !evicted_any {
            warn!(
                threshold_pct,
                "disk pressure above threshold but every warm candidate failed to evict; disk pressure alert"
            );
            return;
        }

        // Re-check pressure; stop when relieved.
        if !check_disk_pressure(&local_root, threshold_pct, probe) {
            return;
        }
    }
}

/// Targeted pre-pull eviction: free at least `needed_bytes` of local space by
/// evicting LRU warm, non-protected models, stopping as soon as the probe reports
/// enough free space (or no candidates remain). Holds the shared disk-op lock so
/// it can't race a concurrent sweep. Returns the number of models evicted.
///
/// Caller (the pull path) re-checks free space afterwards and surfaces the
/// existing insufficient-space error if still short.
#[allow(clippy::too_many_arguments)]
pub async fn evict_for_space(
    registry: &Arc<Mutex<ModelRegistry>>,
    needed_bytes: u64,
    local_root: &Path,
    probe: &dyn DiskSpaceProbe,
    evictor: &dyn LocalEvictor,
    disk_op_lock: &DiskOpLock,
    copy_timeout: Duration,
) -> usize {
    let _guard = disk_op_lock.lock().await;
    let target = crate::models::transfer::nearest_existing_ancestor(local_root);
    let mut evicted = 0usize;

    loop {
        // Enough free already?
        if let Some(free) = probe.available_bytes(&target) {
            if free >= needed_bytes {
                return evicted;
            }
        }
        let next = {
            let reg = registry.lock().await;
            reg.warm_eviction_candidates()
                .into_iter()
                .map(|(name, _, _)| name)
                .next()
        };
        let Some(name) = next else {
            return evicted; // no more candidates; caller surfaces the error
        };
        match evict_to_archive(registry, &name, evictor, copy_timeout).await {
            Ok(ev) => {
                info!(
                    "pre_pull_eviction model={} freed_gb={:.2}",
                    name,
                    bytes_to_gb(ev.freed_bytes)
                );
                evicted += 1;
            }
            Err(e) => {
                // Skip a failed candidate; without removing it from the candidate
                // set we'd loop forever, so bail if it's still first next round.
                warn!(model = %name, error = %e, "pre-pull eviction candidate failed");
                return evicted;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::registry::ModelRegistry;
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tempfile::tempdir;

    /// Generous default copy timeout for tests that aren't exercising MSM-02's
    /// timeout path itself (fast in-memory/tempdir copies never approach this).
    const TEST_COPY_TIMEOUT: Duration = Duration::from_secs(30);

    /// Write a manifest + its referenced blob files under `root`, returning the
    /// model name. Blob digests derive from `model`+index. Config blob included.
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
        let cfg_digest = format!("sha256:{model}cfg");
        fs::write(blobs_dir.join(cfg_digest.replacen(':', "-", 1)), b"cfg").unwrap();
        let body = serde_json::json!({
            "config": { "size": 3, "digest": cfg_digest },
            "layers": layers,
        });
        fs::write(manifests.join(tag), serde_json::to_string(&body).unwrap()).unwrap();
        format!("{model}:{tag}")
    }

    /// Write a manifest that references a *shared* blob digest (so two models can
    /// reference the same physical blob file).
    fn make_model_sharing(
        root: &Path,
        model: &str,
        tag: &str,
        shared_digest: &str,
        shared_size: u64,
    ) -> String {
        let manifests = root
            .join("manifests")
            .join("registry.ollama.ai")
            .join("library")
            .join(model);
        fs::create_dir_all(&manifests).unwrap();
        let blobs_dir = root.join("blobs");
        fs::create_dir_all(&blobs_dir).unwrap();
        let fname = shared_digest.replacen(':', "-", 1);
        fs::write(blobs_dir.join(&fname), vec![b'x'; shared_size as usize]).unwrap();
        let cfg_digest = format!("sha256:{model}cfg");
        fs::write(blobs_dir.join(cfg_digest.replacen(':', "-", 1)), b"cfg").unwrap();
        let body = serde_json::json!({
            "config": { "size": 3, "digest": cfg_digest },
            "layers": [ { "size": shared_size, "digest": shared_digest } ],
        });
        fs::write(manifests.join(tag), serde_json::to_string(&body).unwrap()).unwrap();
        format!("{model}:{tag}")
    }

    fn reg_with(base: &Path, protected: Vec<String>) -> ModelRegistry {
        ModelRegistry::new(
            base.join("registry.json"),
            base.join("local"),
            base.join("archive"),
            protected,
        )
    }

    /// Probe with configurable total/free, optionally mutated over time so a
    /// sweep "sees" pressure relieved after evictions.
    struct ScriptedProbe {
        total: u64,
        free: Arc<std::sync::atomic::AtomicU64>,
    }
    impl DiskSpaceProbe for ScriptedProbe {
        fn available_bytes(&self, _: &Path) -> Option<u64> {
            Some(self.free.load(Ordering::SeqCst))
        }
        fn total_bytes(&self, _: &Path) -> Option<u64> {
            Some(self.total)
        }
    }

    /// Real-fs evictor over the test's local root.
    fn fs_evictor(base: &Path) -> FsLocalEvictor {
        FsLocalEvictor::new(base.join("local"))
    }

    /// Evictor that records calls but does NOT touch the filesystem (to prove
    /// verify-before-delete: if verification fails it must never be called).
    struct SpyEvictor(Arc<AtomicUsize>);
    #[async_trait]
    impl LocalEvictor for SpyEvictor {
        async fn remove(&self, _model: &str) -> Result<(), String> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    #[test]
    fn disk_pressure_true_above_threshold_false_below() {
        struct P(u64, u64);
        impl DiskSpaceProbe for P {
            fn available_bytes(&self, _: &Path) -> Option<u64> {
                Some(self.1)
            }
            fn total_bytes(&self, _: &Path) -> Option<u64> {
                Some(self.0)
            }
        }
        let tmp = tempdir().unwrap();
        // total 100, free 10 → used 90% > 80% → pressure.
        assert!(check_disk_pressure(tmp.path(), 80, &P(100, 10)));
        // total 100, free 50 → used 50% < 80% → no pressure.
        assert!(!check_disk_pressure(tmp.path(), 80, &P(100, 50)));
        // Unknown total → no pressure (never evict on probe failure).
        struct U;
        impl DiskSpaceProbe for U {
            fn available_bytes(&self, _: &Path) -> Option<u64> {
                Some(0)
            }
        }
        assert!(!check_disk_pressure(tmp.path(), 80, &U));
    }

    #[tokio::test]
    async fn evicts_one_warm_model_to_archive_and_marks_cold() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        fs::create_dir_all(base.join("archive")).unwrap();
        let model = make_model(&base.join("local"), "warm", "1", &[100, 200]);
        let mut reg = reg_with(base, vec![]);
        reg.reconcile();
        assert_eq!(reg.get(&model).unwrap().tier, StorageTier::Warm);

        let registry = Arc::new(Mutex::new(reg));
        let evictor = fs_evictor(base);

        evict_to_archive(&registry, &model, &evictor, TEST_COPY_TIMEOUT).await.unwrap();

        // Registry now cold, local cleared, archive set.
        let reg = registry.lock().await;
        let rec = reg.get(&model).unwrap();
        assert_eq!(rec.tier, StorageTier::Cold);
        assert!(rec.local_path.is_none());
        assert!(rec.archive_path.is_some());
        drop(reg);

        // Archive has the copy; local is gone.
        assert!(base
            .join("archive/manifests/registry.ollama.ai/library/warm/1")
            .is_file());
        assert!(base.join("archive/blobs/sha256-warm0").is_file());
        assert!(base.join("archive/blobs/sha256-warm1").is_file());
        assert!(!base
            .join("local/manifests/registry.ollama.ai/library/warm/1")
            .is_file());
        assert!(!base.join("local/blobs/sha256-warm0").is_file());
    }

    // ── MSM-02: timeout-wrapped, crash-safe eviction copy ────────────────────

    #[tokio::test]
    async fn eviction_copy_timeout_cleans_partial_archive_and_leaves_model_warm() {
        // A stalled/slow archive copy (e.g. an NFS write during a volume resize)
        // must not wedge the sweep: it times out, cleans up whatever partial
        // archive files this attempt wrote, and leaves the model Warm so it is
        // retried on the next sweep — never marked Cold, local never touched.
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        fs::create_dir_all(base.join("archive")).unwrap();
        // A large blob so the copy can't finish within a ~1ns timeout (mirrors the
        // TIER-02 `archive_pull` timeout test in transfer.rs).
        let model = make_model(&base.join("local"), "slow", "1", &[8 * 1024 * 1024]);
        let mut reg = reg_with(base, vec![]);
        reg.reconcile();
        let registry = Arc::new(Mutex::new(reg));
        let evictor = fs_evictor(base);

        let err = evict_to_archive(&registry, &model, &evictor, Duration::from_nanos(1))
            .await
            .unwrap_err();
        assert!(matches!(err, EvictError::Timeout(..)), "got {err:?}");

        // Model stays Warm; local copy untouched.
        {
            let reg = registry.lock().await;
            assert_eq!(reg.get(&model).unwrap().tier, StorageTier::Warm);
        }
        assert!(base
            .join("local/manifests/registry.ollama.ai/library/slow/1")
            .is_file());

        // No partial archive blob left behind.
        assert!(
            !base.join("archive/blobs/sha256-slow0").is_file(),
            "partial archive blob must be cleaned up on timeout"
        );

        // The disk-op lock/registry are not wedged: a retry with a real timeout
        // succeeds (proving the failed attempt released everything it held).
        evict_to_archive(&registry, &model, &evictor, TEST_COPY_TIMEOUT)
            .await
            .unwrap();
        assert_eq!(registry.lock().await.get(&model).unwrap().tier, StorageTier::Cold);
    }

    #[tokio::test]
    async fn eviction_timeout_never_deletes_preexisting_archive_blob() {
        // A blob already present in the archive (e.g. shared with a model archived
        // in an earlier sweep) must never be removed by a LATER eviction's timeout
        // cleanup, even though that later copy also references it.
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        fs::create_dir_all(base.join("archive/blobs")).unwrap();
        let shared = "sha256:sharedblob";
        let large = 8 * 1024 * 1024u64;
        // Pre-seed the archive as if `beta` (not modeled here) already evicted it.
        fs::write(base.join("archive/blobs/sha256-sharedblob"), vec![b'x'; large as usize]).unwrap();

        let a = make_model_sharing(&base.join("local"), "alpha", "1", shared, large);
        let mut reg = reg_with(base, vec![]);
        reg.reconcile();
        let registry = Arc::new(Mutex::new(reg));
        let evictor = fs_evictor(base);

        // Copy times out immediately; the shared blob is skip-matched (same
        // digest+size) so it's never re-copied, but MUST also never be swept up
        // by the timeout's best-effort cleanup of "planned" archive paths.
        let _ = evict_to_archive(&registry, &a, &evictor, Duration::from_nanos(1)).await;

        assert!(
            base.join("archive/blobs/sha256-sharedblob").is_file(),
            "pre-existing archive blob must survive another model's copy timeout"
        );
    }

    /// Any `COPY_TEMP_INFIX` scratch files left under `dir` (recursively).
    fn leftover_temps(dir: &Path) -> Vec<String> {
        let mut found = Vec::new();
        let mut stack = vec![dir.to_path_buf()];
        while let Some(d) = stack.pop() {
            let Ok(rd) = fs::read_dir(&d) else { continue };
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p
                    .file_name()
                    .map(|n| n.to_string_lossy().contains(COPY_TEMP_INFIX))
                    .unwrap_or(false)
                {
                    found.push(p.display().to_string());
                }
            }
        }
        found
    }

    #[tokio::test]
    async fn timed_out_copy_leaves_a_preexisting_blob_byte_identical() {
        // End-to-end: a timed-out eviction must leave a pre-existing archive blob
        // (here at a MISMATCHED size, so it is NOT skip-matched and this attempt
        // would re-copy it) exactly as it found it, and leave no scratch file.
        //
        // Note on what this does and does not prove: at a 1 ns timeout the cancel
        // flag is set while the copy task is still creating the archive directory,
        // so the copy stops at its first cancel check without opening any file —
        // this test would also pass against an in-place-writing copy. The
        // mid-flight mutation case is the one that distinguishes them, and it is
        // pinned deterministically by
        // `transfer::tests::cancelled_midcopy_never_mutates_an_existing_destination`.
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        fs::create_dir_all(base.join("archive/blobs")).unwrap();
        // Local blob: 8 MiB of 'x' (make_model's filler) so the copy can't finish
        // within a ~1ns timeout.
        let model = make_model(&base.join("local"), "slow", "1", &[8 * 1024 * 1024]);
        // Archive already holds a DIFFERENT, smaller file at that path — distinct
        // content so any partial overwrite is detectable byte-for-byte.
        let victim: Vec<u8> = vec![b'Z'; 4 * 1024 * 1024];
        let victim_path = base.join("archive/blobs/sha256-slow0");
        fs::write(&victim_path, &victim).unwrap();

        let mut reg = reg_with(base, vec![]);
        reg.reconcile();
        let registry = Arc::new(Mutex::new(reg));
        let evictor = fs_evictor(base);

        let err = evict_to_archive(&registry, &model, &evictor, Duration::from_nanos(1))
            .await
            .unwrap_err();
        assert!(matches!(err, EvictError::Timeout(..)), "got {err:?}");

        assert_eq!(
            fs::read(&victim_path).unwrap(),
            victim,
            "a pre-existing archive blob must be byte-identical after a timed-out copy \
             — it must never be truncated or partially overwritten in place"
        );
        assert!(
            leftover_temps(&base.join("archive")).is_empty(),
            "a stopped copy must remove its own scratch file: {:?}",
            leftover_temps(&base.join("archive"))
        );
    }

    #[tokio::test]
    async fn midcopy_failure_never_deletes_a_preexisting_archive_blob() {
        // The mid-copy-error path must obey the same ownership rule as the
        // timeout path. A blob already in the archive (shared with a model
        // archived earlier) whose size does NOT match is re-copied by this
        // attempt — so it lands in the copy's `written` list. If a LATER blob
        // then fails, cleaning up `written` verbatim would delete a file this
        // attempt did not create, and which may be the only remaining copy.
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        fs::create_dir_all(base.join("archive/blobs")).unwrap();
        let model = make_model(&base.join("local"), "mix", "1", &[1024, 2048]);

        // `sha256-mix0` is already in the archive at a different size → not
        // skip-matched, so this attempt overwrites it and records it as written.
        fs::write(base.join("archive/blobs/sha256-mix0"), b"older").unwrap();
        // Make the SECOND blob's copy fail: its local source is gone.
        fs::remove_file(base.join("local/blobs/sha256-mix1")).unwrap();

        let mut reg = reg_with(base, vec![]);
        reg.reconcile();
        let registry = Arc::new(Mutex::new(reg));
        let evictor = fs_evictor(base);

        let err = evict_to_archive(&registry, &model, &evictor, TEST_COPY_TIMEOUT)
            .await
            .unwrap_err();
        assert!(matches!(err, EvictError::ArchiveCopy(..)), "got {err:?}");

        assert!(
            base.join("archive/blobs/sha256-mix0").is_file(),
            "a pre-existing archive blob must survive this attempt's mid-copy failure"
        );
        assert!(
            !base.join("archive/blobs/sha256-mix1").is_file(),
            "the failed blob this attempt created must be cleaned up"
        );
        // Model stays Warm; local copy untouched.
        assert_eq!(
            registry.lock().await.get(&model).unwrap().tier,
            StorageTier::Warm
        );
    }

    #[tokio::test]
    async fn sweep_skips_persistently_failing_candidate_tries_it_once() {
        // A candidate whose local removal always fails must be tried exactly once
        // per sweep (recorded in the skip-set), not re-attempted every iteration,
        // while other candidates still evict and the sweep terminates.
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        fs::create_dir_all(base.join("archive")).unwrap();
        let bad = make_model(&base.join("local"), "bad", "1", &[100]);
        let good = make_model(&base.join("local"), "good", "1", &[100]);
        let mut reg = reg_with(base, vec![]);
        reg.reconcile();
        // Deterministic order: `bad` older → sorts LRU-first; `good` newer.
        reg.set_last_requested_for_test(&bad, 1000);
        reg.set_last_requested_for_test(&good, 2000);
        let registry = Arc::new(Mutex::new(reg));

        struct Selective {
            bad: String,
            local_root: PathBuf,
            bad_calls: Arc<AtomicUsize>,
        }
        #[async_trait]
        impl LocalEvictor for Selective {
            async fn remove(&self, model: &str) -> Result<(), String> {
                if model == self.bad {
                    self.bad_calls.fetch_add(1, Ordering::SeqCst);
                    return Err("simulated local removal failure".into());
                }
                fs_remove_model(&self.local_root, model)
            }
        }
        let bad_calls = Arc::new(AtomicUsize::new(0));
        let evictor = Selective {
            bad: bad.clone(),
            local_root: base.join("local"),
            bad_calls: bad_calls.clone(),
        };

        // Probe reports permanent pressure (used 99%), so the sweep loops until the
        // candidate set is exhausted rather than until pressure relieves.
        struct Full;
        impl DiskSpaceProbe for Full {
            fn available_bytes(&self, _: &Path) -> Option<u64> {
                Some(1)
            }
            fn total_bytes(&self, _: &Path) -> Option<u64> {
                Some(100)
            }
        }
        let lock = new_disk_op_lock();
        run_eviction_sweep(&registry, 80, 0, &Full, &evictor, &lock, TEST_COPY_TIMEOUT).await;

        let reg = registry.lock().await;
        assert_eq!(reg.get(&good).unwrap().tier, StorageTier::Cold, "good evicted");
        assert_eq!(
            reg.get(&bad).unwrap().tier,
            StorageTier::Warm,
            "bad stays warm (removal failed)"
        );
        assert_eq!(
            bad_calls.load(Ordering::SeqCst),
            1,
            "failing candidate must be tried exactly once, not retried each loop"
        );
    }

    #[tokio::test]
    async fn sweep_only_protected_warm_logs_warning_no_eviction() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        fs::create_dir_all(base.join("archive")).unwrap();
        let model = make_model(&base.join("local"), "keepme", "1", &[100]);
        let mut reg = reg_with(base, vec![model.clone()]);
        reg.reconcile();
        assert!(reg.is_protected(&model));

        let registry = Arc::new(Mutex::new(reg));
        // 90% used, stays high (protected model can't be evicted).
        let free = Arc::new(std::sync::atomic::AtomicU64::new(10));
        let probe = ScriptedProbe { total: 100, free };
        let evictor = fs_evictor(base);
        let lock = new_disk_op_lock();

        run_eviction_sweep(&registry, 80, 0, &probe, &evictor, &lock, TEST_COPY_TIMEOUT).await;

        // Model untouched: still warm, still local.
        let reg = registry.lock().await;
        assert_eq!(reg.get(&model).unwrap().tier, StorageTier::Warm);
        assert!(base
            .join("local/manifests/registry.ollama.ai/library/keepme/1")
            .is_file());
    }

    #[tokio::test]
    async fn verify_failure_keeps_model_warm_and_no_local_removal() {
        // Make the archive blobs dir a FILE so create_dir_all/copy fails partway,
        // OR force a size mismatch. Here we pre-seed the archive with a wrong-size
        // blob that the copy will skip-by-size? No — copy skips only on MATCH.
        // Instead: pre-create the archive manifest dir as read-only is fragile.
        // Simpler: corrupt verification by making one referenced blob absent from
        // local so verify finds an archive blob but we delete the local source
        // first? We can't. So we force ArchiveCopy failure by removing a local
        // source blob the manifest references → copy errors → model stays warm.
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        fs::create_dir_all(base.join("archive")).unwrap();
        let model = make_model(&base.join("local"), "bad", "1", &[100, 200]);
        // Remove a referenced local blob so the archive copy fails.
        fs::remove_file(base.join("local/blobs/sha256-bad1")).unwrap();

        let mut reg = reg_with(base, vec![]);
        reg.reconcile();
        let registry = Arc::new(Mutex::new(reg));
        let spy_calls = Arc::new(AtomicUsize::new(0));
        let spy = SpyEvictor(spy_calls.clone());

        let err = evict_to_archive(&registry, &model, &spy, TEST_COPY_TIMEOUT).await.unwrap_err();
        assert!(matches!(err, EvictError::ArchiveCopy(..)), "got {err:?}");
        // Local removal never invoked; model still warm + local present.
        assert_eq!(spy_calls.load(Ordering::SeqCst), 0, "must not remove before a good copy");
        assert_eq!(registry.lock().await.get(&model).unwrap().tier, StorageTier::Warm);
        assert!(base
            .join("local/manifests/registry.ollama.ai/library/bad/1")
            .is_file());
    }

    #[tokio::test]
    async fn verify_failure_blocks_local_removal_model_stays_warm() {
        // Directly exercise the verify-before-delete guard: a half-copied archive
        // (manifest present but a referenced blob MISSING) must fail verification,
        // and evict_to_archive must therefore NOT invoke the local evictor and
        // must leave the model warm.
        //
        // We inject the bad archive state with a SpyEvictor whose `remove` would
        // record a call (proving deletion happened). Because copy_model_to_archive
        // always writes a complete copy, we instead assert the verify function
        // itself rejects an incomplete archive, then assert the end-to-end path
        // never deletes on that rejection.
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        let _model = make_model(&base.join("local"), "mm", "1", &[100]);
        let local_manifest =
            base.join("local/manifests/registry.ollama.ai/library/mm/1");
        let archive = base.join("archive");

        // 1) Hand-build an INCOMPLETE archive: manifest present, blob missing.
        let arch_manifest = archive.join("manifests/registry.ollama.ai/library/mm/1");
        fs::create_dir_all(arch_manifest.parent().unwrap()).unwrap();
        fs::copy(&local_manifest, &arch_manifest).unwrap();
        fs::create_dir_all(archive.join("blobs")).unwrap();
        // Only the cfg blob present; the layer blob `sha256-mm0` is missing.
        fs::write(archive.join("blobs/sha256-mmcfg"), b"cfg").unwrap();

        // verify must reject the incomplete archive.
        let err = verify_archive_copy("mm:1", &local_manifest, &archive).unwrap_err();
        assert!(err.contains("archive blob missing"), "got {err}");

        // 2) Now force the SAME verify failure through evict_to_archive by making
        // the archive blobs dir non-writable so the copy "succeeds" trivially for
        // an already-present blob but the missing one can't be written → the copy
        // errors first (ArchiveCopy) OR verify fails. Either way: NO local delete.
        // Simplest deterministic injection: make the local SOURCE layer blob
        // unreadable-as-file by removing it so copy fails, proving no removal.
        fs::remove_file(base.join("local/blobs/sha256-mm0")).unwrap();
        let mut reg = reg_with(base, vec![]);
        reg.reconcile();
        let registry = Arc::new(Mutex::new(reg));
        let spy_calls = Arc::new(AtomicUsize::new(0));
        let spy = SpyEvictor(spy_calls.clone());

        let err = evict_to_archive(&registry, "mm:1", &spy, TEST_COPY_TIMEOUT).await.unwrap_err();
        assert!(
            matches!(err, EvictError::ArchiveCopy(..) | EvictError::VerifyFailed(..)),
            "got {err:?}"
        );
        assert_eq!(spy_calls.load(Ordering::SeqCst), 0, "no local removal on bad copy/verify");
        assert_eq!(registry.lock().await.get("mm:1").unwrap().tier, StorageTier::Warm);
        assert!(local_manifest.is_file(), "local manifest must remain");
    }

    #[tokio::test]
    async fn lru_orders_evictions_oldest_first() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        fs::create_dir_all(base.join("archive")).unwrap();
        let old = make_model(&base.join("local"), "old", "1", &[100]);
        let new = make_model(&base.join("local"), "new", "1", &[100]);
        let mut reg = reg_with(base, vec![]);
        reg.reconcile();
        // Force timestamps: old < new.
        // (reconcile set last_requested = mtime; override deterministically.)
        // Deterministic timestamps: old < new (epoch-second resolution would
        // otherwise tie models created in the same wall-clock second).
        reg.set_last_requested_for_test(&old, 1000);
        reg.set_last_requested_for_test(&new, 2000);
        let registry = Arc::new(Mutex::new(reg));
        let cands: Vec<String> = {
            let r = registry.lock().await;
            r.warm_eviction_candidates().into_iter().map(|(n, _, _)| n).collect()
        };
        assert_eq!(cands.first().unwrap(), &old, "LRU: oldest (old) evicts first");
        assert_eq!(cands.last().unwrap(), &new);
    }

    #[tokio::test]
    async fn shared_blob_not_deleted_on_eviction() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        fs::create_dir_all(base.join("archive")).unwrap();
        let shared = "sha256:sharedblob";
        let a = make_model_sharing(&base.join("local"), "alpha", "1", shared, 100);
        let _b = make_model_sharing(&base.join("local"), "beta", "1", shared, 100);
        let mut reg = reg_with(base, vec![]);
        reg.reconcile();
        let registry = Arc::new(Mutex::new(reg));
        let evictor = fs_evictor(base);

        evict_to_archive(&registry, &a, &evictor, TEST_COPY_TIMEOUT).await.unwrap();

        // alpha's manifest gone; shared blob KEPT (beta still references it).
        assert!(!base
            .join("local/manifests/registry.ollama.ai/library/alpha/1")
            .is_file());
        assert!(
            base.join("local/blobs/sha256-sharedblob").is_file(),
            "blob shared with beta must NOT be deleted"
        );
        // alpha's own cfg blob (unshared) is gone.
        assert!(!base.join("local/blobs/sha256-alphacfg").is_file());
    }

    #[tokio::test]
    async fn pre_pull_eviction_frees_space() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        fs::create_dir_all(base.join("archive")).unwrap();
        let warm = make_model(&base.join("local"), "evictme", "1", &[1000]);
        let mut reg = reg_with(base, vec![]);
        reg.reconcile();
        assert_eq!(reg.get(&warm).unwrap().tier, StorageTier::Warm);
        let registry = Arc::new(Mutex::new(reg));

        // Free starts at 100 (< needed 500), bumps to 2000 after one eviction.
        let free = Arc::new(std::sync::atomic::AtomicU64::new(100));
        struct BumpProbe(Arc<std::sync::atomic::AtomicU64>);
        impl DiskSpaceProbe for BumpProbe {
            fn available_bytes(&self, _: &Path) -> Option<u64> {
                // Read, then simulate freed space for the NEXT read.
                let cur = self.0.load(Ordering::SeqCst);
                if cur < 500 {
                    self.0.store(2000, Ordering::SeqCst);
                }
                Some(cur)
            }
        }
        let probe = BumpProbe(free.clone());
        let evictor = fs_evictor(base);
        let lock = new_disk_op_lock();

        let n = evict_for_space(&registry, 500, &base.join("local"), &probe, &evictor, &lock, TEST_COPY_TIMEOUT).await;
        assert_eq!(n, 1, "one model evicted to free space");
        assert_eq!(registry.lock().await.get(&warm).unwrap().tier, StorageTier::Cold);
    }

    #[tokio::test]
    async fn sweep_skips_when_archive_unmounted() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        // archive dir intentionally NOT created → simulates unmounted NFS.
        let model = make_model(&base.join("local"), "x", "1", &[100]);
        let mut reg = reg_with(base, vec![]);
        reg.reconcile();
        let registry = Arc::new(Mutex::new(reg));
        let free = Arc::new(std::sync::atomic::AtomicU64::new(0)); // 100% used
        let probe = ScriptedProbe { total: 100, free };
        let evictor = fs_evictor(base);
        let lock = new_disk_op_lock();

        run_eviction_sweep(&registry, 80, 0, &probe, &evictor, &lock, TEST_COPY_TIMEOUT).await;
        // No archive → no eviction even under max pressure.
        assert_eq!(registry.lock().await.get(&model).unwrap().tier, StorageTier::Warm);
    }

    // ── TIER-04: cooldown eviction ────────────────────────────────────────────

    /// Probe that always reports plenty of free space → NO disk pressure. Lets
    /// the cooldown tests prove cooldown eviction is independent of disk pressure.
    struct NoPressureProbe;
    impl DiskSpaceProbe for NoPressureProbe {
        fn available_bytes(&self, _: &Path) -> Option<u64> {
            Some(1000) // 1000/1000 free → 0% used
        }
        fn total_bytes(&self, _: &Path) -> Option<u64> {
            Some(1000)
        }
    }

    const HOUR: i64 = 3_600;
    const DAY: i64 = 24 * HOUR;
    // Arbitrary fixed "now" (epoch seconds) so tests never touch wall-clock.
    const NOW: i64 = 1_700_000_000;

    #[tokio::test]
    async fn cooldown_evicts_idle_model_regardless_of_disk_pressure() {
        // Model last requested 8 days ago, cooldown 168h (7 days), and the probe
        // reports NO pressure → cooldown alone must archive it.
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        fs::create_dir_all(base.join("archive")).unwrap();
        let model = make_model(&base.join("local"), "stale", "1", &[100]);
        let mut reg = reg_with(base, vec![]);
        reg.reconcile();
        reg.set_last_requested_for_test(&model, NOW - 8 * DAY);
        let registry = Arc::new(Mutex::new(reg));
        let evictor = fs_evictor(base);
        let lock = new_disk_op_lock();

        run_eviction_sweep_at(&registry, 80, 168, NOW, &NoPressureProbe, &evictor, &lock, TEST_COPY_TIMEOUT).await;

        let reg = registry.lock().await;
        let rec = reg.get(&model).unwrap();
        assert_eq!(rec.tier, StorageTier::Cold, "idle model archived by cooldown");
        assert!(rec.local_path.is_none());
        assert!(base
            .join("archive/manifests/registry.ollama.ai/library/stale/1")
            .is_file());
    }

    #[tokio::test]
    async fn cooldown_keeps_recently_used_model() {
        // Idle only 6 days < 7-day cooldown, no pressure → must stay warm.
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        fs::create_dir_all(base.join("archive")).unwrap();
        let model = make_model(&base.join("local"), "fresh", "1", &[100]);
        let mut reg = reg_with(base, vec![]);
        reg.reconcile();
        reg.set_last_requested_for_test(&model, NOW - 6 * DAY);
        let registry = Arc::new(Mutex::new(reg));
        let evictor = fs_evictor(base);
        let lock = new_disk_op_lock();

        run_eviction_sweep_at(&registry, 80, 168, NOW, &NoPressureProbe, &evictor, &lock, TEST_COPY_TIMEOUT).await;

        assert_eq!(
            registry.lock().await.get(&model).unwrap().tier,
            StorageTier::Warm,
            "model used within cooldown stays warm"
        );
    }

    #[tokio::test]
    async fn cooldown_exempts_protected_model() {
        // Protected model idle 30 days → never evicted by cooldown.
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        fs::create_dir_all(base.join("archive")).unwrap();
        let model = make_model(&base.join("local"), "keepme", "1", &[100]);
        let mut reg = reg_with(base, vec![model.clone()]);
        reg.reconcile();
        reg.set_last_requested_for_test(&model, NOW - 30 * DAY);
        assert!(reg.is_protected(&model));
        let registry = Arc::new(Mutex::new(reg));
        let evictor = fs_evictor(base);
        let lock = new_disk_op_lock();

        run_eviction_sweep_at(&registry, 80, 168, NOW, &NoPressureProbe, &evictor, &lock, TEST_COPY_TIMEOUT).await;

        assert_eq!(
            registry.lock().await.get(&model).unwrap().tier,
            StorageTier::Warm,
            "protected model exempt from cooldown eviction"
        );
    }

    #[tokio::test]
    async fn cooldown_disabled_when_zero_hours() {
        // cooldown_hours == 0 → cooldown eviction never triggers, even for a model
        // idle for years, with no disk pressure.
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        fs::create_dir_all(base.join("archive")).unwrap();
        let model = make_model(&base.join("local"), "ancient", "1", &[100]);
        let mut reg = reg_with(base, vec![]);
        reg.reconcile();
        reg.set_last_requested_for_test(&model, NOW - 1000 * DAY);
        let registry = Arc::new(Mutex::new(reg));
        let evictor = fs_evictor(base);
        let lock = new_disk_op_lock();

        run_eviction_sweep_at(&registry, 80, 0, NOW, &NoPressureProbe, &evictor, &lock, TEST_COPY_TIMEOUT).await;

        assert_eq!(
            registry.lock().await.get(&model).unwrap().tier,
            StorageTier::Warm,
            "cooldown==0 disables cooldown eviction"
        );
    }

    #[tokio::test]
    async fn cooldown_evicts_never_requested_legacy_model() {
        // last_requested == None (legacy entry) → treated as infinitely idle →
        // eligible for cooldown eviction.
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        fs::create_dir_all(base.join("archive")).unwrap();
        let model = make_model(&base.join("local"), "legacy", "1", &[100]);
        let mut reg = reg_with(base, vec![]);
        reg.reconcile();
        reg.clear_last_requested_for_test(&model);
        let registry = Arc::new(Mutex::new(reg));
        let evictor = fs_evictor(base);
        let lock = new_disk_op_lock();

        run_eviction_sweep_at(&registry, 80, 168, NOW, &NoPressureProbe, &evictor, &lock, TEST_COPY_TIMEOUT).await;

        assert_eq!(
            registry.lock().await.get(&model).unwrap().tier,
            StorageTier::Cold,
            "never-requested model is infinitely idle → cooldown-evicted"
        );
    }

    #[tokio::test]
    async fn cooldown_keeps_freshly_pulled_model_with_ancient_manifest_mtime() {
        // CHRD-75 regression. A freshly pulled/warmed model that has NEVER been
        // served must read ~0h idle and must NOT be cooldown-evicted — even when
        // its manifest file carries an ancient on-disk mtime (blob dedup /
        // copy-preserving timestamps). reconcile used to seed `last_requested`
        // from that mtime, so with MODEL_WARM_COOLDOWN_HOURS=1 a model pulled
        // minutes ago looked ~918h idle and was deleted mid-use. Discovery (the
        // warm/pull moment) is now the last-access time, so idle is ~0h.
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        fs::create_dir_all(base.join("archive")).unwrap();
        let model = make_model(&base.join("local"), "freshpull", "latest", &[100]);

        // Backdate the manifest leaf mtime to ~918h ago (the exact bug symptom).
        let manifest =
            base.join("local/manifests/registry.ollama.ai/library/freshpull/latest");
        let ancient = std::time::SystemTime::now() - Duration::from_secs(918 * 3600);
        fs::File::options()
            .write(true)
            .open(&manifest)
            .unwrap()
            .set_modified(ancient)
            .unwrap();

        let mut reg = reg_with(base, vec![]);
        reg.reconcile(); // seeds last_requested — must be ~now, not the ancient mtime

        // Freshly discovered → ~0h idle, not ~918h.
        let now = now_epoch_secs();
        let lr = reg
            .get(&model)
            .unwrap()
            .last_requested
            .expect("newly discovered model has last_requested");
        let idle_hours = (now - lr) / 3_600;
        assert!(
            idle_hours < 1,
            "freshly pulled model idle {idle_hours}h; must read ~0h, not ~918h"
        );

        // …and the cooldown pass with a 1h cooldown must leave it warm.
        let registry = Arc::new(Mutex::new(reg));
        let evictor = fs_evictor(base);
        let lock = new_disk_op_lock();
        run_eviction_sweep_at(
            &registry, 80, 1, now, &NoPressureProbe, &evictor, &lock, TEST_COPY_TIMEOUT,
        )
        .await;

        assert_eq!(
            registry.lock().await.get(&model).unwrap().tier,
            StorageTier::Warm,
            "freshly pulled, never-served model must survive the cooldown pass"
        );
    }
}
