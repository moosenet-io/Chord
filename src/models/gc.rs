//! MSM-03: orphan-blob GC.
//!
//! An interrupted eviction (manifest removed but blob delete skipped, or the
//! reverse — blob copied/removed but the manifest write didn't land) can leave
//! local blobs referenced by NO local manifest. This pass finds them and safely
//! deletes the ones that are either:
//! - also present in the archive (`<archive>/blobs`) — a cold copy already
//!   exists, so the local copy is redundant, or
//! - unreferenced **anywhere** — no local manifest AND no archive manifest
//!   references the digest, and no archive blob file exists either — dead
//!   weight with nothing to protect.
//!
//! A blob referenced by ANY local manifest is NEVER deleted (GC-aware, mirrors
//! `eviction::fs_remove_model`'s shared-blob protection — content-addressed
//! blobs are shared across models). An orphan blob that IS referenced by an
//! archive manifest but has no archive blob file (an inconsistent archive) is
//! also left alone — deleting the only remaining copy would be unrecoverable.
//!
//! Runs as part of the background sweep (after eviction, see `main.rs`) and is
//! exposed as a control endpoint (`POST /api/storage/gc`, MSM-04). Holds the
//! shared disk-op lock for the duration so it can't race a concurrent sweep or
//! pull.
//!
//! Nothing here hardcodes infrastructure — paths are passed in by the caller
//! (config-derived), never a literal.

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use tokio::sync::Mutex;
use tracing::{info, warn};

use super::eviction::DiskOpLock;
use super::registry::{collect_manifest_leaves, parse_manifest_blobs, ModelRegistry};

/// Result of one [`run_gc`] pass.
#[derive(Debug, Default, Clone)]
pub struct GcResult {
    /// Number of orphan blob files deleted.
    pub orphans_deleted: usize,
    /// Total bytes freed by deleted orphans.
    pub freed_bytes: u64,
    /// Non-fatal errors encountered while deleting individual orphans
    /// (best-effort; a single failed delete never aborts the rest of the pass).
    pub errors: Vec<String>,
    /// Whether the pass ran at all (`false` when the local blobs dir doesn't
    /// exist yet — nothing to scan, not an error).
    pub ran: bool,
}

/// Run one orphan-blob GC pass, deleting local blobs referenced by no local
/// manifest that are also present in the archive (with a **matching size**)
/// OR unreferenced anywhere. Never deletes a blob referenced by any local
/// manifest, nor one younger than `min_age_secs` (the B1 defense-in-depth
/// grace window — an in-flight pull writes blobs before its manifest, so a
/// too-young "orphan" may be mid-copy). Holds `disk_op_lock` for the duration
/// so it can't race a concurrent sweep or (lock-honouring) pull.
///
/// `registry` is accepted but not required for the filesystem scan itself
/// (the pass is purely filesystem-driven, matching
/// `eviction::fs_remove_model`'s approach) — kept for API symmetry with the
/// other MSM-01..04 entry points and so a future revision can restrict the
/// scan to known/Ollama-managed models without changing the call signature.
pub async fn run_gc(
    _registry: &Arc<Mutex<ModelRegistry>>,
    local_root: &Path,
    archive_root: &Path,
    disk_op_lock: &DiskOpLock,
    min_age_secs: u64,
) -> GcResult {
    let _guard = disk_op_lock.lock().await;
    let local_root = local_root.to_path_buf();
    let archive_root = archive_root.to_path_buf();
    // Filesystem walk + removals are blocking; run off the async reactor
    // (mirrors `FsLocalEvictor::remove`'s `spawn_blocking` pattern).
    tokio::task::spawn_blocking(move || run_gc_sync(&local_root, &archive_root, min_age_secs))
        .await
        .unwrap_or_else(|e| GcResult {
            errors: vec![format!("gc task join error: {e}")],
            ..Default::default()
        })
}

/// Current wall-clock time in epoch seconds (best-effort; 0 on failure).
fn now_epoch_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Epoch seconds of a file's mtime (best-effort; 0 on failure — a 0 mtime is
/// treated as "very old", i.e. GC-eligible, which is safe: a real file always
/// has a real mtime, and an unreadable one is not a live-copy candidate).
fn mtime_epoch_secs(path: &Path) -> u64 {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Blocking filesystem implementation of the GC pass.
fn run_gc_sync(local_root: &Path, archive_root: &Path, min_age_secs: u64) -> GcResult {
    let mut result = GcResult::default();
    let now = now_epoch_secs();
    let local_blobs_dir = local_root.join("blobs");
    if !local_blobs_dir.is_dir() {
        return result; // nothing local to scan; `ran` stays false
    }
    result.ran = true;

    // Every blob digest referenced by ANY local manifest — never delete these.
    let local_referenced: HashSet<String> = collect_manifest_leaves(local_root)
        .iter()
        .flat_map(|leaf| parse_manifest_blobs(leaf).digests)
        .collect();

    // Archive-side knowledge, best-effort: an unmounted/missing archive simply
    // means "nothing archived" (empty sets) rather than an error — an orphan
    // is then only deleted if it's ALSO unreferenced by any archive manifest,
    // which trivially holds when there are no archive manifests to reference
    // it. This is the conservative posture the MSM-03 edge cases call for.
    let archive_blobs_dir = archive_root.join("blobs");
    let archive_referenced: HashSet<String> = if archive_root.exists() {
        collect_manifest_leaves(archive_root)
            .iter()
            .flat_map(|leaf| parse_manifest_blobs(leaf).digests)
            .collect()
    } else {
        HashSet::new()
    };

    let entries = match std::fs::read_dir(&local_blobs_dir) {
        Ok(e) => e,
        Err(e) => {
            result
                .errors
                .push(format!("read local blobs dir {}: {e}", local_blobs_dir.display()));
            return result;
        }
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }
        let Some(fname) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        // On-disk filename is "sha256-XXXX"; the digest form manifests
        // reference is "sha256:XXXX" (see `transfer::blob_filename`, the
        // inverse of this).
        let digest = fname.replacen('-', ":", 1);
        if local_referenced.contains(&digest) {
            continue; // referenced by a local manifest → never delete
        }

        // B1 defense-in-depth: never delete a blob younger than the grace
        // window. An in-flight archive pull writes each blob to its final path
        // BEFORE the referencing manifest lands, so a freshly-written blob can
        // momentarily look like an "orphan with a cold copy". The disk-op lock
        // is the primary guard (the pull-copy phase holds it), but this age
        // check protects against any future path that forgets the lock.
        let local_size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        let age = now.saturating_sub(mtime_epoch_secs(&path));
        if age < min_age_secs {
            continue;
        }

        let archive_blob_path = archive_blobs_dir.join(fname);
        // S1: a cold copy only counts if the archive blob EXISTS **and its size
        // matches** the local blob — a truncated/partial archive copy (e.g. an
        // interrupted eviction) is NOT a safe cold copy, so we must not delete
        // the local original on the strength of it.
        let archived_blob_valid = std::fs::metadata(&archive_blob_path)
            .map(|m| m.is_file() && m.len() == local_size)
            .unwrap_or(false);
        let archive_blob_present = archive_blob_path.is_file();
        let archive_manifest_references_it = archive_referenced.contains(&digest);

        if archived_blob_valid {
            delete_orphan(&path, fname, "archived (cold copy exists, size-verified)", &mut result);
        } else if archive_blob_present {
            // The archive has a blob file for this digest but its size does NOT
            // match the local copy (truncated/partial cold copy) — refuse to
            // delete the local original; the archive copy can't be trusted.
            warn!(
                blob = %fname,
                local_size,
                "gc: archive blob size does not match local; NOT deleting (partial/truncated cold copy)"
            );
        } else if !archive_manifest_references_it {
            // No local manifest, no archive manifest, no archive blob file —
            // dead weight with nothing anywhere depending on it.
            delete_orphan(&path, fname, "unreferenced anywhere", &mut result);
        } else {
            // An archive manifest references this digest but the archive has no
            // blob file for it (an inconsistent/incomplete archive) — leaving
            // the local copy in place is the only way to avoid losing the last
            // remaining copy of that data.
            warn!(
                blob = %fname,
                "gc: orphan blob is referenced by an archive manifest with no matching archive \
                 blob file; NOT deleting (would lose the only remaining copy)"
            );
        }
    }

    result
}

/// Delete one confirmed-safe orphan blob, updating `result` (freed bytes /
/// count on success, a logged non-fatal error on failure).
fn delete_orphan(path: &Path, fname: &str, reason: &str, result: &mut GcResult) {
    let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    match std::fs::remove_file(path) {
        Ok(()) => {
            info!(blob = %fname, reason, size_bytes = size, "gc: deleted orphan blob");
            result.orphans_deleted += 1;
            result.freed_bytes += size;
        }
        Err(e) => {
            warn!(blob = %fname, error = %e, "gc: failed to delete orphan blob");
            result.errors.push(format!("delete {fname}: {e}"));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use tempfile::tempdir;

    /// Write a manifest referencing the given blob digests (with a filler
    /// "size" of 1 each) under `<root>/manifests/registry.ollama.ai/library/<model>/<tag>`.
    fn write_manifest(root: &Path, model: &str, tag: &str, digests: &[&str]) {
        let dir = root
            .join("manifests")
            .join("registry.ollama.ai")
            .join("library")
            .join(model);
        fs::create_dir_all(&dir).unwrap();
        let layers: Vec<serde_json::Value> = digests
            .iter()
            .map(|d| serde_json::json!({ "size": 1, "digest": d }))
            .collect();
        let body = serde_json::json!({ "config": null, "layers": layers });
        fs::write(dir.join(tag), serde_json::to_string(&body).unwrap()).unwrap();
    }

    /// Write a blob file at `<root>/blobs/<digest-as-filename>` with `size` bytes.
    fn write_blob(root: &Path, digest: &str, size: usize) -> PathBuf {
        let blobs_dir = root.join("blobs");
        fs::create_dir_all(&blobs_dir).unwrap();
        let fname = digest.replacen(':', "-", 1);
        let path = blobs_dir.join(&fname);
        fs::write(&path, vec![b'x'; size]).unwrap();
        path
    }

    fn lock() -> DiskOpLock {
        super::super::eviction::new_disk_op_lock()
    }

    fn empty_registry(base: &Path) -> Arc<Mutex<ModelRegistry>> {
        Arc::new(Mutex::new(ModelRegistry::new(
            base.join("registry.json"),
            base.join("local"),
            base.join("archive"),
            vec![],
        )))
    }

    #[tokio::test]
    async fn orphan_with_archive_copy_is_deleted() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        let local = base.join("local");
        let archive = base.join("archive");
        // Local blob with NO referencing local manifest, but present in archive.
        write_blob(&local, "sha256:orphan1", 100);
        write_blob(&archive, "sha256:orphan1", 100);

        let reg = empty_registry(base);
        let result = run_gc(&reg, &local, &archive, &lock(), 0).await;

        assert_eq!(result.orphans_deleted, 1);
        assert_eq!(result.freed_bytes, 100);
        assert!(result.errors.is_empty());
        assert!(!local.join("blobs/sha256-orphan1").exists());
    }

    #[tokio::test]
    async fn referenced_blob_is_never_deleted() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        let local = base.join("local");
        let archive = base.join("archive");
        write_blob(&local, "sha256:keep1", 50);
        write_manifest(&local, "kept", "1", &["sha256:keep1"]);
        // Also present in archive — would be a valid orphan-delete target if it
        // weren't referenced; proves the local-reference check wins.
        write_blob(&archive, "sha256:keep1", 50);

        let reg = empty_registry(base);
        let result = run_gc(&reg, &local, &archive, &lock(), 0).await;

        assert_eq!(result.orphans_deleted, 0);
        assert!(local.join("blobs/sha256-keep1").is_file(), "referenced blob must survive GC");
    }

    #[tokio::test]
    async fn shared_blob_referenced_by_another_local_manifest_is_kept() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        let local = base.join("local");
        let archive = base.join("archive");
        write_blob(&local, "sha256:shared1", 10);
        // Two local manifests both reference it.
        write_manifest(&local, "alpha", "1", &["sha256:shared1"]);
        write_manifest(&local, "beta", "1", &["sha256:shared1"]);

        let reg = empty_registry(base);
        let result = run_gc(&reg, &local, &archive, &lock(), 0).await;

        assert_eq!(result.orphans_deleted, 0);
        assert!(local.join("blobs/sha256-shared1").is_file());
    }

    #[tokio::test]
    async fn orphan_unreferenced_anywhere_is_deleted() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        let local = base.join("local");
        let archive = base.join("archive");
        fs::create_dir_all(&archive).unwrap(); // mounted, but empty
        write_blob(&local, "sha256:deadweight", 30);

        let reg = empty_registry(base);
        let result = run_gc(&reg, &local, &archive, &lock(), 0).await;

        assert_eq!(result.orphans_deleted, 1);
        assert_eq!(result.freed_bytes, 30);
        assert!(!local.join("blobs/sha256-deadweight").exists());
    }

    #[tokio::test]
    async fn orphan_referenced_only_by_archive_manifest_without_blob_file_is_kept() {
        // An inconsistent archive (manifest present, blob file missing) must
        // never cause the LAST remaining copy (the local one) to be deleted.
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        let local = base.join("local");
        let archive = base.join("archive");
        write_blob(&local, "sha256:onlycopy", 20);
        write_manifest(&archive, "archived-only", "1", &["sha256:onlycopy"]);
        // No archive blob file written — archive is inconsistent for this digest.

        let reg = empty_registry(base);
        let result = run_gc(&reg, &local, &archive, &lock(), 0).await;

        assert_eq!(result.orphans_deleted, 0);
        assert!(
            local.join("blobs/sha256-onlycopy").is_file(),
            "must not delete the only remaining copy of a blob an archive manifest expects"
        );
    }

    #[tokio::test]
    async fn no_local_blobs_dir_is_a_no_op() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        let local = base.join("local"); // never created
        let archive = base.join("archive");
        let reg = empty_registry(base);

        let result = run_gc(&reg, &local, &archive, &lock(), 0).await;
        assert!(!result.ran);
        assert_eq!(result.orphans_deleted, 0);
        assert!(result.errors.is_empty());
    }

    #[tokio::test]
    async fn gc_is_idempotent() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        let local = base.join("local");
        let archive = base.join("archive");
        write_blob(&local, "sha256:orphan1", 100);
        write_blob(&archive, "sha256:orphan1", 100);
        let reg = empty_registry(base);
        let l = lock();

        let first = run_gc(&reg, &local, &archive, &l, 0).await;
        assert_eq!(first.orphans_deleted, 1);

        // Second pass: nothing left to delete, no errors.
        let second = run_gc(&reg, &local, &archive, &l, 0).await;
        assert_eq!(second.orphans_deleted, 0);
        assert!(second.errors.is_empty());
    }

    // ── B1 defense-in-depth + S1 ──────────────────────────────────────────────

    #[tokio::test]
    async fn gc_skips_blob_written_within_grace_window() {
        // A just-written orphan blob (with a valid archive copy) must NOT be
        // deleted when its mtime is within the grace window — it could be an
        // in-flight pull's blob whose manifest hasn't landed yet.
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        let local = base.join("local");
        let archive = base.join("archive");
        write_blob(&local, "sha256:fresh1", 100); // just created → age ~0s
        write_blob(&archive, "sha256:fresh1", 100);
        let reg = empty_registry(base);

        // A large grace window makes the freshly-written blob ineligible.
        let result = run_gc(&reg, &local, &archive, &lock(), 3600).await;
        assert_eq!(result.orphans_deleted, 0, "blob within grace window must be skipped");
        assert!(
            local.join("blobs/sha256-fresh1").is_file(),
            "a too-young orphan must survive GC (may be an in-flight pull)"
        );

        // With no grace window (0s) the same blob IS collected → proves the age
        // check, not something else, is what protected it above.
        let result = run_gc(&reg, &local, &archive, &lock(), 0).await;
        assert_eq!(result.orphans_deleted, 1);
        assert!(!local.join("blobs/sha256-fresh1").exists());
    }

    #[tokio::test]
    async fn gc_rejects_size_mismatched_archive_copy() {
        // S1: an archive blob that exists but whose size does NOT match the
        // local blob (a truncated/partial cold copy) is not a safe cold copy —
        // the local original must be kept.
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        let local = base.join("local");
        let archive = base.join("archive");
        write_blob(&local, "sha256:partial1", 100);
        write_blob(&archive, "sha256:partial1", 40); // truncated archive copy
        let reg = empty_registry(base);

        let result = run_gc(&reg, &local, &archive, &lock(), 0).await;
        assert_eq!(result.orphans_deleted, 0, "must not delete on a size-mismatched cold copy");
        assert!(
            local.join("blobs/sha256-partial1").is_file(),
            "local original kept when the archive copy is truncated/partial"
        );

        // Sanity: a size-MATCHING archive copy of the same digest IS accepted.
        write_blob(&archive, "sha256:partial1", 100);
        let result = run_gc(&reg, &local, &archive, &lock(), 0).await;
        assert_eq!(result.orphans_deleted, 1);
        assert!(!local.join("blobs/sha256-partial1").exists());
    }

    #[tokio::test]
    async fn pull_holding_disk_op_lock_excludes_concurrent_gc() {
        // B1: the disk-op lock serialises a pull-copy phase with GC. Model the
        // race directly: hold the shared lock (as the pull-copy phase now does),
        // write a blob that has no local manifest yet (as mid-pull), and fire a
        // GC on the SAME lock — the GC must block until the "pull" releases the
        // lock, and by then the manifest exists so the blob is no longer an
        // orphan. This proves GC can't delete a blob mid-pull.
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        let local = base.join("local");
        let archive = base.join("archive");
        write_blob(&archive, "sha256:pulling1", 100);
        let reg = empty_registry(base);
        let l = lock();

        // Acquire the lock to simulate the in-flight pull-copy phase.
        let guard = l.lock().await;
        // Blob copied to local (final path) but manifest not yet written.
        write_blob(&local, "sha256:pulling1", 100);

        // Kick off a concurrent GC on the same lock; it must NOT proceed while
        // we hold the guard. min_age 0 so the age window isn't what blocks it —
        // the LOCK is.
        let reg2 = reg.clone();
        let l2 = l.clone();
        let local2 = local.clone();
        let archive2 = archive.clone();
        let gc_handle = tokio::spawn(async move {
            run_gc(&reg2, &local2, &archive2, &l2, 0).await
        });

        // Give the spawned GC a chance to run; it should be parked on the lock.
        tokio::task::yield_now().await;
        assert!(
            local.join("blobs/sha256-pulling1").is_file(),
            "GC must not have deleted the blob while the pull holds the lock"
        );

        // The pull finishes: manifest lands, then the lock is released.
        write_manifest(&local, "pulling", "1", &["sha256:pulling1"]);
        drop(guard);

        let result = gc_handle.await.unwrap();
        assert_eq!(result.orphans_deleted, 0, "blob now referenced by a manifest → not an orphan");
        assert!(
            local.join("blobs/sha256-pulling1").is_file(),
            "the pulled blob survives — GC saw it referenced once the lock was free"
        );
    }
}
