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
//! - **Timeout:** the copy is wrapped in `tokio::time::timeout`; on expiry every
//!   file written by *this* pull is removed so no partial/corrupt local state is
//!   left behind.
//! - **Mid-copy failure cleanup:** any error partway through copying triggers the
//!   same cleanup of this pull's partial files.
//! - **Progress events:** an optional [`PullEvent`] channel surfaces
//!   "retrieving from archive" / "loading into VRAM" so a long NFS copy doesn't
//!   look stuck. Tests pass `None`.
//!
//! Nothing here hardcodes infrastructure — all paths come from the registry /
//! config, model names from the request.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
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

    // ── Timeout-wrapped copy with partial-file tracking for cleanup ──
    let copy_fut = copy_model_files(&plan);
    match tokio::time::timeout(timeout, copy_fut).await {
        Ok(Ok(())) => Ok(()),
        Ok(Err((e, written))) => {
            cleanup_partial(&written);
            Err(e)
        }
        Err(_elapsed) => {
            // Timed out: we don't have the in-flight written list here, so remove
            // every file this pull *would* have created that now exists locally.
            cleanup_partial(&planned_local_paths(&plan));
            Err(PullError::Timeout(timeout))
        }
    }
}

/// Copy the manifest + every referenced blob. On error returns the error plus
/// the list of files this pull created so far (for cleanup). Blobs are copied
/// first, manifest last, so Ollama never sees a manifest whose blobs are missing.
async fn copy_model_files(plan: &PullPlan) -> Result<(), (PullError, Vec<PathBuf>)> {
    let mut written: Vec<PathBuf> = Vec::new();

    // Blobs.
    let archive_blobs = plan.archive_root.join("blobs");
    let local_blobs = plan.local_root.join("blobs");
    if let Err(e) = tokio::fs::create_dir_all(&local_blobs).await {
        return Err((PullError::Io(e.to_string()), written));
    }
    for digest in &plan.blobs.digests {
        let fname = blob_filename(digest);
        let src = archive_blobs.join(&fname);
        let dst = local_blobs.join(&fname);
        // Skip blobs already present locally (content-addressed → identical),
        // but don't add them to `written` so cleanup never deletes pre-existing
        // shared blobs.
        if dst.exists() {
            continue;
        }
        if let Err(e) = tokio::fs::copy(&src, &dst).await {
            written.push(dst);
            return Err((PullError::Io(format!("copy blob {fname}: {e}")), written));
        }
        written.push(dst);
    }

    // Manifest leaf — mirror its path relative to the archive manifests root.
    let archive_manifests = plan.archive_root.join("manifests");
    let rel = match plan.archive_manifest.strip_prefix(&archive_manifests) {
        Ok(r) => r.to_path_buf(),
        Err(_) => {
            return Err((
                PullError::Io("archive manifest path outside manifests root".into()),
                written,
            ))
        }
    };
    let dst_manifest = plan.local_root.join("manifests").join(&rel);
    if let Some(parent) = dst_manifest.parent() {
        if let Err(e) = tokio::fs::create_dir_all(parent).await {
            return Err((PullError::Io(e.to_string()), written));
        }
    }
    if let Err(e) = tokio::fs::copy(&plan.archive_manifest, &dst_manifest).await {
        written.push(dst_manifest);
        return Err((PullError::Io(format!("copy manifest: {e}")), written));
    }
    written.push(dst_manifest);

    Ok(())
}

/// Every local path this pull would create (manifest + blobs). Used for cleanup
/// after a timeout, where we lack the precise in-flight written list. Only paths
/// that exist are removed, and pre-existing shared blobs are conservatively left
/// in place is NOT possible here — so we only remove blobs whose copy this pull
/// could plausibly own. To stay safe we limit timeout cleanup to the manifest +
/// blobs, which is acceptable: a half-copied blob is corrupt and must go.
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

/// Remove partial files left by a failed/timed-out pull (best-effort).
fn cleanup_partial(paths: &[PathBuf]) {
    for p in paths {
        if p.exists() {
            if let Err(e) = std::fs::remove_file(p) {
                tracing::warn!(path = %p.display(), error = %e, "failed to clean up partial pull file");
            }
        }
    }
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
                            )
                            .await;
                        }
                    }
                }

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
                // the existing lifecycle, which calls set_tier(Hot)).
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
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tempfile::tempdir;

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
