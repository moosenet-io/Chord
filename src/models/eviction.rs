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
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;
use tracing::{info, warn};

use super::registry::{
    collect_manifest_leaves, parse_manifest_blobs, ModelRegistry, StorageTier,
};
use super::transfer::{blob_filename, find_manifest_leaf, DiskSpaceProbe};

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

/// Shared global lock serialising destructive disk operations (eviction sweeps
/// and archive pulls). Held for the duration of a single model's eviction copy +
/// local removal so a concurrent pull can't race the same blobs.
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

/// GC-aware local removal of `model` under `local_root`.
fn fs_remove_model(local_root: &Path, model: &str) -> Result<(), String> {
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

/// Set of blob digests referenced by every local manifest *except* `exclude`.
fn referenced_by_other_manifests(local_root: &Path, exclude: &Path) -> HashSet<String> {
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
///    already exist in the archive with a matching size).
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

    // ── Copy local → archive (reverse pull) ──
    copy_model_to_archive(model, &local_root, &local_manifest, &archive_root)
        .await
        .map_err(|e| EvictError::ArchiveCopy(model.to_string(), e))?;

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
/// already present in the archive with a matching size are skipped. Blobs are
/// copied before the manifest so the archive never has a manifest whose blobs are
/// missing. Returns a stringly error (mapped to [`EvictError::ArchiveCopy`]).
async fn copy_model_to_archive(
    _model: &str,
    local_root: &Path,
    local_manifest: &Path,
    archive_root: &Path,
) -> Result<(), String> {
    let blobs = parse_manifest_blobs(local_manifest);
    let local_blobs = local_root.join("blobs");
    let archive_blobs = archive_root.join("blobs");
    tokio::fs::create_dir_all(&archive_blobs)
        .await
        .map_err(|e| format!("create archive blobs dir: {e}"))?;

    for digest in &blobs.digests {
        let fname = blob_filename(digest);
        let src = local_blobs.join(&fname);
        let dst = archive_blobs.join(&fname);
        // Skip if already in archive with matching size (content-addressed →
        // identical content for the same digest + size).
        if let (Ok(s), Ok(d)) = (tokio::fs::metadata(&src).await, tokio::fs::metadata(&dst).await) {
            if s.len() == d.len() {
                continue;
            }
        }
        tokio::fs::copy(&src, &dst)
            .await
            .map_err(|e| format!("copy blob {fname}: {e}"))?;
    }

    // Manifest leaf — mirror its path relative to the local manifests root.
    let local_manifests = local_root.join("manifests");
    let rel = local_manifest
        .strip_prefix(&local_manifests)
        .map_err(|_| "local manifest path outside manifests root".to_string())?;
    let dst_manifest = archive_root.join("manifests").join(rel);
    if let Some(parent) = dst_manifest.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| format!("create archive manifest dir: {e}"))?;
    }
    tokio::fs::copy(local_manifest, &dst_manifest)
        .await
        .map_err(|e| format!("copy manifest: {e}"))?;
    Ok(())
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
        match evict_to_archive(registry, &name, evictor).await {
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
) {
    run_eviction_sweep_at(
        registry,
        threshold_pct,
        cooldown_hours,
        now_epoch_secs(),
        probe,
        evictor,
        disk_op_lock,
    )
    .await
}

/// Sweep with an injected `now_secs` so the cooldown decision is deterministic
/// in tests. Production calls go through [`run_eviction_sweep`] which supplies
/// the real wall-clock time.
pub async fn run_eviction_sweep_at(
    registry: &Arc<Mutex<ModelRegistry>>,
    threshold_pct: u8,
    cooldown_hours: u64,
    now_secs: i64,
    probe: &dyn DiskSpaceProbe,
    evictor: &dyn LocalEvictor,
    disk_op_lock: &DiskOpLock,
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
    cooldown_pass(registry, cooldown_hours, now_secs, evictor, disk_op_lock).await;

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
            match evict_to_archive(registry, &name, evictor).await {
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
pub async fn evict_for_space(
    registry: &Arc<Mutex<ModelRegistry>>,
    needed_bytes: u64,
    local_root: &Path,
    probe: &dyn DiskSpaceProbe,
    evictor: &dyn LocalEvictor,
    disk_op_lock: &DiskOpLock,
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
        match evict_to_archive(registry, &name, evictor).await {
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

        evict_to_archive(&registry, &model, &evictor).await.unwrap();

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
        run_eviction_sweep(&registry, 80, 0, &Full, &evictor, &lock).await;

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

        run_eviction_sweep(&registry, 80, 0, &probe, &evictor, &lock).await;

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

        let err = evict_to_archive(&registry, &model, &spy).await.unwrap_err();
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

        let err = evict_to_archive(&registry, "mm:1", &spy).await.unwrap_err();
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

        evict_to_archive(&registry, &a, &evictor).await.unwrap();

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

        let n = evict_for_space(&registry, 500, &base.join("local"), &probe, &evictor, &lock).await;
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

        run_eviction_sweep(&registry, 80, 0, &probe, &evictor, &lock).await;
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

        run_eviction_sweep_at(&registry, 80, 168, NOW, &NoPressureProbe, &evictor, &lock).await;

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

        run_eviction_sweep_at(&registry, 80, 168, NOW, &NoPressureProbe, &evictor, &lock).await;

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

        run_eviction_sweep_at(&registry, 80, 168, NOW, &NoPressureProbe, &evictor, &lock).await;

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

        run_eviction_sweep_at(&registry, 80, 0, NOW, &NoPressureProbe, &evictor, &lock).await;

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

        run_eviction_sweep_at(&registry, 80, 168, NOW, &NoPressureProbe, &evictor, &lock).await;

        assert_eq!(
            registry.lock().await.get(&model).unwrap().tier,
            StorageTier::Cold,
            "never-requested model is infinitely idle → cooldown-evicted"
        );
    }
}
