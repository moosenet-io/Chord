//! Model storage registry: persistent record of which models exist at which
//! storage tier, their sizes, and when they were last loaded / requested.
//!
//! The registry is a JSON file on local disk (survives Chord restarts; not a
//! database dependency). At startup `reconcile()` scans the local and archive
//! Ollama manifest trees and updates the records to match reality.
//!
//! ## Timestamp representation (deliberate deviation from the spec)
//!
//! The TIER-01 spec sketches `last_loaded` / `last_requested` as
//! `Option<DateTime<Utc>>`. This crate does **not** depend on `chrono` — it
//! represents time as Unix epoch **seconds** (`i64`) everywhere (see the
//! `SystemTime → UNIX_EPOCH` helpers in `audit.rs`). To avoid pulling in a new
//! dependency for a single struct field, timestamps here are `Option<i64>`
//! holding epoch seconds. This is a conscious, documented deviation.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::models::backends::{self, Backend};

/// Storage tier a model currently lives at.
///
/// - `Hot`  — loaded in VRAM (or marked loaded); fastest.
/// - `Warm` — present on local disk, not loaded.
/// - `Cold` — only present in the archive (e.g. NFS), must be pulled before use.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum StorageTier {
    Hot,
    Warm,
    Cold,
}

/// A single tracked model.
///
/// Timestamps are Unix epoch **seconds** (`i64`), not `chrono::DateTime` — see
/// the module-level docs for why this deviates from the spec.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelRecord {
    /// Model name/tag, e.g. `qwen3-coder:30b` or `hf.co/org/model:tag`.
    pub name: String,
    /// Current storage tier.
    pub tier: StorageTier,
    /// Path to the local Ollama root that holds this model, if present locally.
    pub local_path: Option<String>,
    /// Path to the archive root that holds this model, if present in archive.
    pub archive_path: Option<String>,
    /// Total size in bytes (sum of config + layer sizes from the manifest).
    pub size_bytes: u64,
    /// Epoch seconds of the last time the model was loaded into VRAM.
    pub last_loaded: Option<i64>,
    /// Epoch seconds of the last time any request referenced this model.
    pub last_requested: Option<i64>,
    /// Protected models are never auto-archived (tier never auto-demoted).
    pub protected: bool,
    /// What manages this model's lifecycle. `"ollama"` (the default) means it is discovered and
    /// re-tiered from Ollama manifest trees by `reconcile()`. Non-Ollama models (e.g.
    /// `"llama-diffusion"` for DiffusionGemma) are registered explicitly via `register_external` and
    /// are NOT touched by `reconcile()` — their tier/paths are managed out of band.
    #[serde(default = "default_managed_by")]
    pub managed_by: String,
    /// Name of the inference [`Backend`](crate::models::backends::Backend) this
    /// model is tagged to run on (hardware-aware routing/testing). `None` ⇒ use
    /// the registry's default backend. Defaults to `None` so registries written
    /// before this field existed still deserialize.
    #[serde(default)]
    pub backend: Option<String>,
    /// Direct GGUF path for a non-Ollama model served on a llama-server backend
    /// (`-m <path>`), bypassing Ollama-blob resolution. For a sharded GGUF this
    /// is the FIRST shard (`…-00001-of-000NN.gguf`); llama.cpp loads the rest.
    /// `None` ⇒ the model is Ollama-managed and its blob is resolved normally.
    #[serde(default)]
    pub gguf_path: Option<String>,
}

/// Default for `ModelRecord::managed_by` so registries written before this field existed still
/// deserialize (every legacy record is an Ollama-managed model).
fn default_managed_by() -> String {
    "ollama".to_string()
}

/// `managed_by` value for an Ollama-managed model (the reconcile default).
pub const MANAGED_BY_OLLAMA: &str = "ollama";

/// File-backed registry of all known models.
#[derive(Debug, Clone)]
pub struct ModelRegistry {
    records: HashMap<String, ModelRecord>,
    backends: HashMap<String, Backend>,
    path: PathBuf,
    protected: Vec<String>,
    local_path: PathBuf,
    archive_path: PathBuf,
}

/// On-disk registry file shape (P5). The legacy format was a bare
/// `HashMap<String, ModelRecord>`; the new format wraps it so backend
/// definitions live alongside the models. `load_or_new` accepts BOTH.
#[derive(Serialize, Deserialize, Default)]
struct RegistryFile {
    #[serde(default)]
    models: HashMap<String, ModelRecord>,
    #[serde(default)]
    backends: HashMap<String, Backend>,
}

/// Current time as Unix epoch seconds. Mirrors the `audit.rs` approach
/// (`SystemTime::now()` → `UNIX_EPOCH`) to avoid a chrono dependency.
fn now_epoch_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Epoch seconds of a file's modification time (best-effort; 0 on failure).
fn mtime_epoch_secs(path: &Path) -> i64 {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Subset of an Ollama manifest we care about: config blob + layer blobs each
/// carry a `size` field whose sum is the model's on-disk size.
#[derive(Deserialize)]
struct OllamaManifest {
    #[serde(default)]
    config: Option<OllamaBlob>,
    #[serde(default)]
    layers: Vec<OllamaBlob>,
}

#[derive(Deserialize)]
struct OllamaBlob {
    #[serde(default)]
    size: u64,
    /// Content-addressed digest, e.g. `sha256:abc…`. Used by TIER-02's transfer
    /// to locate the blob file (`blobs/sha256-abc…`). May be absent in some
    /// manifests; size accounting tolerates that.
    #[serde(default)]
    digest: String,
}

/// The digests + total size referenced by a single Ollama manifest. Used by the
/// TIER-02 archive-pull (`transfer.rs`) to know which blob files to copy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestBlobs {
    /// All blob digests (`sha256:…`) the manifest references (config + layers).
    pub digests: Vec<String>,
    /// Sum of all referenced blob sizes (bytes).
    pub total_size: u64,
}

/// Parse an Ollama manifest file and return every blob digest it references
/// (config + layers) plus their total size. Empty/missing digests are skipped.
/// A missing/unreadable/non-JSON file yields an empty result (best-effort) so a
/// caller can surface a clear "manifest unreadable" error of its own.
///
/// Shared with `transfer.rs` so the archive-pull reuses the exact same manifest
/// parsing the registry's reconcile relies on (single source of truth for the
/// Ollama layout).
pub fn parse_manifest_blobs(path: &Path) -> ManifestBlobs {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(_) => return ManifestBlobs { digests: Vec::new(), total_size: 0 },
    };
    let manifest: OllamaManifest = match serde_json::from_str(&text) {
        Ok(m) => m,
        Err(_) => return ManifestBlobs { digests: Vec::new(), total_size: 0 },
    };
    let mut digests = Vec::new();
    let mut total_size = 0u64;
    if let Some(cfg) = manifest.config {
        if !cfg.digest.is_empty() {
            digests.push(cfg.digest);
        }
        total_size += cfg.size;
    }
    for layer in manifest.layers {
        if !layer.digest.is_empty() {
            digests.push(layer.digest);
        }
        total_size += layer.size;
    }
    ManifestBlobs { digests, total_size }
}

/// Map a model name back to its manifest path *relative to the manifests root*,
/// i.e. `<registry>/<namespace>/<model>/<tag>`. Returns `None` when the name has
/// no tag (every Ollama model name carries a `:tag`).
///
/// This is the inverse of `scan_manifest_tree`'s naming. For a `library`-style
/// name (`model:tag`) the registry host defaults to `registry.ollama.ai` and the
/// namespace to `library`. For a namespaced name (`ns/model:tag`) the host is
/// still `registry.ollama.ai`. Hugging-Face style names (`hf.co/org/model:tag`)
/// carry their own host as the first segment.
///
/// Because the on-disk registry host can vary, `transfer.rs` does **not** rely on
/// reconstructing the path blindly; it instead *discovers* the actual manifest
/// leaf under the archive tree (see `find_manifest_leaf`). This helper is kept
/// for callers/tests that want the canonical relative layout.
pub fn manifest_rel_path(name: &str) -> Option<PathBuf> {
    let (body, tag) = name.rsplit_once(':')?;
    let parts: Vec<&str> = body.split('/').collect();
    let rel = match parts.as_slice() {
        // model
        [model] => PathBuf::from("registry.ollama.ai").join("library").join(model).join(tag),
        // namespace/model
        [ns, model] => PathBuf::from("registry.ollama.ai").join(ns).join(model).join(tag),
        // host/namespace/model (e.g. hf.co/org/model)
        [host, ns, model] => PathBuf::from(host).join(ns).join(model).join(tag),
        _ => return None,
    };
    Some(rel)
}

/// A model discovered by walking a manifest tree.
struct DiscoveredModel {
    /// Canonical model name (`<model>:<tag>` for the `library` namespace,
    /// otherwise `<namespace>/<model>:<tag>`).
    name: String,
    /// Total size in bytes from the manifest (config + layers).
    size_bytes: u64,
    /// Epoch seconds of the manifest file's mtime.
    mtime: i64,
}

/// Parse the registry file, accepting BOTH the new `{models, backends}` wrapper
/// and the legacy bare `{name: ModelRecord}` map. New format is detected by a
/// top-level `models`/`backends` key — every Ollama model name carries a `:tag`,
/// so neither word can be a legacy model key.
fn parse_registry_file(
    text: &str,
) -> Result<(HashMap<String, ModelRecord>, HashMap<String, Backend>), serde_json::Error> {
    let value: serde_json::Value = serde_json::from_str(text)?;
    let is_new = value
        .as_object()
        .map(|o| o.contains_key("models") || o.contains_key("backends"))
        .unwrap_or(false);
    if is_new {
        let file: RegistryFile = serde_json::from_value(value)?;
        Ok((file.models, file.backends))
    } else {
        let records: HashMap<String, ModelRecord> = serde_json::from_value(value)?;
        Ok((records, HashMap::new()))
    }
}

impl ModelRegistry {
    /// Create an empty registry bound to the given paths.
    pub fn new(
        path: PathBuf,
        local_path: PathBuf,
        archive_path: PathBuf,
        protected: Vec<String>,
    ) -> Self {
        ModelRegistry {
            records: HashMap::new(),
            backends: backends::seed_from_env(),
            path,
            protected,
            local_path,
            archive_path,
        }
    }

    /// Load the registry from `path` if it exists, otherwise start empty.
    ///
    /// If the on-disk JSON is corrupt the registry is rebuilt from empty (a
    /// `reconcile()` call will repopulate it from the filesystem) and a warning
    /// is logged — it never panics.
    pub fn load_or_new(
        path: PathBuf,
        local_path: PathBuf,
        archive_path: PathBuf,
        protected: Vec<String>,
    ) -> Self {
        let (records, mut backends) = match std::fs::read_to_string(&path) {
            Ok(text) => parse_registry_file(&text).unwrap_or_else(|e| {
                tracing::warn!(
                    path = %path.display(),
                    error = %e,
                    "model registry JSON is corrupt; rebuilding from empty"
                );
                (HashMap::new(), HashMap::new())
            }),
            Err(_) => (HashMap::new(), HashMap::new()),
        };
        // Seed the backend catalogue from env the first time a registry without a
        // `backends` section is loaded (legacy flat files, or a fresh install).
        if backends.is_empty() {
            backends = backends::seed_from_env();
        }
        ModelRegistry {
            records,
            backends,
            path,
            protected,
            local_path,
            archive_path,
        }
    }

    /// Atomically persist the registry to `path`: serialize to a temp file in
    /// the same directory, then `rename` over the target so a crash mid-write
    /// can never leave a half-written registry.
    pub fn save(&self) -> std::io::Result<()> {
        let file = RegistryFile {
            models: self.records.clone(),
            backends: self.backends.clone(),
        };
        let json = serde_json::to_string_pretty(&file)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        let dir = self.path.parent().unwrap_or_else(|| Path::new("."));
        if !dir.exists() {
            std::fs::create_dir_all(dir)?;
        }
        // Temp file in the SAME dir so the rename is atomic (same filesystem).
        let file_name = self
            .path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "model-registry.json".to_string());
        let tmp = dir.join(format!(".{}.tmp.{}", file_name, std::process::id()));
        std::fs::write(&tmp, json.as_bytes())?;
        std::fs::rename(&tmp, &self.path)?;
        Ok(())
    }

    // ── Backends (P5) ────────────────────────────────────────────────────────

    /// All defined backends (name → backend).
    pub fn backends(&self) -> &HashMap<String, Backend> {
        &self.backends
    }

    /// Look up a backend by name.
    pub fn backend(&self, name: &str) -> Option<&Backend> {
        self.backends.get(name)
    }

    /// The default backend for untagged models: the primary `ollama`, else any
    /// `always_on` backend, else any backend.
    pub fn default_backend(&self) -> Option<&Backend> {
        self.backends
            .get("ollama")
            .or_else(|| self.backends.values().find(|b| b.always_on))
            .or_else(|| self.backends.values().next())
    }

    /// Resolve the backend a model should run on: its tagged `backend` if set and
    /// known, else the default backend.
    pub fn backend_for(&self, model: &str) -> Option<&Backend> {
        if let Some(name) = self.records.get(model).and_then(|r| r.backend.as_deref()) {
            if let Some(b) = self.backends.get(name) {
                return Some(b);
            }
        }
        self.default_backend()
    }

    /// Set (or clear) a model's backend tag. Returns false if the model is
    /// unknown or the named backend is not defined.
    pub fn set_model_backend(&mut self, model: &str, backend: Option<String>) -> bool {
        if let Some(name) = &backend {
            if !self.backends.contains_key(name) {
                return false;
            }
        }
        match self.records.get_mut(model) {
            Some(rec) => {
                rec.backend = backend;
                true
            }
            None => false,
        }
    }

    /// Insert or replace a backend definition.
    pub fn upsert_backend(&mut self, b: Backend) {
        self.backends.insert(b.name.clone(), b);
    }

    /// Tag every **dense large** model (see [`backends::is_vulkan_candidate`]) to
    /// the driver-stable [`backends::VULKAN_BACKEND`] serving backend, *only* when
    /// that backend is defined and the model is not already explicitly tagged to a
    /// different backend. Returns the names of the models newly tagged.
    ///
    /// This is additive and conservative: it never overrides an operator's
    /// existing per-model backend tag, and it is a no-op if the `vulkan` backend
    /// is not present in the catalogue. It lets Chord know Vulkan is an available
    /// backend for dense large models (e.g. `llama3.3:70b`) without hardcoding a
    /// model list, mirroring the runtime-driven tagging convention.
    pub fn tag_vulkan_candidates(&mut self) -> Vec<String> {
        if !self.backends.contains_key(backends::VULKAN_BACKEND) {
            return Vec::new();
        }
        let mut tagged = Vec::new();
        for (name, rec) in self.records.iter_mut() {
            if rec.backend.is_none() && backends::is_vulkan_candidate(name) {
                rec.backend = Some(backends::VULKAN_BACKEND.to_string());
                tagged.push(name.clone());
            }
        }
        tagged
    }

    /// Reconcile the registry against the on-disk reality.
    ///
    /// Scans the local manifest tree and (if mounted) the archive manifest tree
    /// and updates records:
    /// - Local model already known → tier becomes `Warm` (left `Hot` if Hot),
    ///   `local_path` set; new → added as `Warm` with `last_requested = mtime`.
    /// - Archive-only model already known → tier `Cold`, `archive_path` set;
    ///   new → added as `Cold`.
    /// - Registry entries found in neither tier are left in place with a warning
    ///   (never deleted).
    ///
    /// A missing/unmounted archive path is skipped with a warning and reconcile
    /// continues with local-only data (it must never crash).
    pub fn reconcile(&mut self) {
        // --- Local scan ---
        let local_manifests = self.local_path.join("manifests");
        let local = scan_manifest_tree(&local_manifests);
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

        for m in &local {
            seen.insert(m.name.clone());
            let protected = self.protected.iter().any(|p| p == &m.name);
            let local_root = self.local_path.to_string_lossy().to_string();
            match self.records.get_mut(&m.name) {
                Some(rec) => {
                    // Known model present locally → at least Warm. Leave Hot.
                    if rec.tier != StorageTier::Hot {
                        rec.tier = StorageTier::Warm;
                    }
                    rec.local_path = Some(local_root);
                    rec.size_bytes = m.size_bytes;
                    rec.protected = rec.protected || protected;
                }
                None => {
                    self.records.insert(
                        m.name.clone(),
                        ModelRecord {
                            name: m.name.clone(),
                            tier: StorageTier::Warm,
                            local_path: Some(local_root),
                            archive_path: None,
                            size_bytes: m.size_bytes,
                            last_loaded: None,
                            last_requested: Some(m.mtime),
                            protected,
                            managed_by: MANAGED_BY_OLLAMA.to_string(),
                            backend: None, gguf_path: None,
                        },
                    );
                }
            }
        }

        // --- Demote records that vanished from local disk ---
        // A model previously recorded as local (Warm/Hot) that the local scan no
        // longer found must have its `local_path` cleared so the archive scan can
        // re-tier it to Cold (or the neither-tier check flags it). Without this,
        // a stale `local_path` keeps `local_path.is_none()` false and the model is
        // never demoted — making reconcile non-idempotent against external removals
        // (e.g. an out-of-band `ollama rm`, or a model archived by another process).
        for rec in self.records.values_mut() {
            // Only Ollama-managed records are demoted from Ollama scans; externally-managed models
            // (e.g. DiffusionGemma) are never present in Ollama manifest trees and must be left alone.
            if rec.managed_by == MANAGED_BY_OLLAMA && rec.local_path.is_some() && !seen.contains(&rec.name) {
                rec.local_path = None;
            }
        }

        // --- Archive scan (skip gracefully if not mounted) ---
        let archive_manifests = self.archive_path.join("manifests");
        if !self.archive_path.exists() {
            tracing::warn!(
                archive_path = %self.archive_path.display(),
                "archive path not present / not mounted; reconciling with local-only data"
            );
        } else {
            let archive = scan_manifest_tree(&archive_manifests);
            let archive_root = self.archive_path.to_string_lossy().to_string();
            for m in &archive {
                seen.insert(m.name.clone());
                let protected = self.protected.iter().any(|p| p == &m.name);
                match self.records.get_mut(&m.name) {
                    Some(rec) => {
                        rec.archive_path = Some(archive_root.clone());
                        // Only models present *only* in archive become Cold.
                        if rec.local_path.is_none() {
                            rec.tier = StorageTier::Cold;
                        }
                        rec.protected = rec.protected || protected;
                    }
                    None => {
                        self.records.insert(
                            m.name.clone(),
                            ModelRecord {
                                name: m.name.clone(),
                                tier: StorageTier::Cold,
                                local_path: None,
                                archive_path: Some(archive_root.clone()),
                                size_bytes: m.size_bytes,
                                last_loaded: None,
                                last_requested: Some(m.mtime),
                                protected,
                                managed_by: MANAGED_BY_OLLAMA.to_string(),
                                backend: None, gguf_path: None,
                            },
                        );
                    }
                }
            }
        }

        // --- Registry entries found in neither tier: warn, keep. ---
        // Externally-managed models (non-Ollama) are expected to be absent from the Ollama scans, so
        // they are not "missing" — only warn for Ollama-managed records.
        for (name, rec) in self.records.iter() {
            if rec.managed_by == MANAGED_BY_OLLAMA && !seen.contains(name) {
                tracing::warn!(
                    model = %name,
                    "registry entry not found on local disk or in archive; keeping record (may be missing)"
                );
            }
        }
    }

    /// Register (or update) a model whose lifecycle is NOT managed by Ollama — e.g. a GGUF served by
    /// `llama-diffusion-daemon`. Such a record is excluded from `reconcile()`'s Ollama-driven re-tiering
    /// (see the `managed_by` checks above), so its tier/paths are authoritative once set here. Returns
    /// the resulting tier. Persisting is the caller's responsibility (`save()`).
    pub fn register_external(
        &mut self,
        name: &str,
        managed_by: &str,
        local_path: Option<String>,
        archive_path: Option<String>,
        size_bytes: u64,
    ) -> StorageTier {
        // Tier reflects reality: Warm if a local copy is present, else Cold (archive-only → pull before
        // use). Never Hot here — VRAM residency is decided by the daemon loading the model on demand.
        let tier = if local_path.is_some() {
            StorageTier::Warm
        } else {
            StorageTier::Cold
        };
        match self.records.get_mut(name) {
            Some(rec) => {
                rec.tier = tier.clone();
                rec.local_path = local_path;
                rec.archive_path = archive_path;
                rec.size_bytes = size_bytes;
                rec.managed_by = managed_by.to_string();
            }
            None => {
                self.records.insert(
                    name.to_string(),
                    ModelRecord {
                        name: name.to_string(),
                        tier: tier.clone(),
                        local_path,
                        archive_path,
                        size_bytes,
                        last_loaded: None,
                        last_requested: None,
                        protected: false,
                        managed_by: managed_by.to_string(),
                        backend: None, gguf_path: None,
                    },
                );
            }
        }
        tier
    }

    /// Register DiffusionGemma (served by `llama-diffusion-daemon`, not Ollama) from env config, so
    /// Chord's tiering and control API are aware of it. Tier follows reality: Warm if the GGUF is on
    /// local disk, else Cold (archive-only → TIER-02 pull before use). No-op (debug log) when neither
    /// `DGEM_MODEL_PATH` nor `DGEM_MODEL_ARCHIVE_PATH` is set — the model lives only on the GPU host, so
    /// off-host chord instances simply don't register it. Never panics; persisting is the caller's job.
    pub fn register_diffusiongemma_from_env(&mut self) {
        let name = std::env::var("DGEM_MODEL_NAME")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "diffusiongemma-26b-a4b".to_string());
        let local = std::env::var("DGEM_MODEL_PATH").ok().filter(|s| !s.is_empty());
        let archive = std::env::var("DGEM_MODEL_ARCHIVE_PATH").ok().filter(|s| !s.is_empty());
        if local.is_none() && archive.is_none() {
            tracing::debug!("DiffusionGemma not registered: DGEM_MODEL_PATH / DGEM_MODEL_ARCHIVE_PATH unset");
            return;
        }
        // A local path only counts if the file actually exists on this host.
        let (local_present, size) = match &local {
            Some(p) => match std::fs::metadata(p) {
                Ok(m) if m.is_file() => (Some(p.clone()), m.len()),
                _ => (None, 0),
            },
            None => (None, 0),
        };
        // Don't register a path-less phantom: if the configured local file is absent and there is no
        // archive location either, there is nothing real to track.
        if local_present.is_none() && archive.is_none() {
            tracing::warn!(
                model = %name,
                "DiffusionGemma not registered: configured DGEM_MODEL_PATH is missing and no archive set"
            );
            return;
        }
        let tier = self.register_external(&name, "llama-diffusion", local_present, archive, size);
        tracing::info!(
            model = %name,
            tier = ?tier,
            "registered DiffusionGemma in model registry (non-Ollama, managed by llama-diffusion-daemon)"
        );
    }

    /// Look up a record by model name.
    pub fn get(&self, name: &str) -> Option<&ModelRecord> {
        self.records.get(name)
    }

    /// Borrow every tracked record (unordered). Used by the TIER-05 control API's
    /// `GET /api/models` to render the full registry without exposing the internal
    /// map. Callers that need a stable order should sort by `name`.
    pub fn all_records(&self) -> impl Iterator<Item = &ModelRecord> {
        self.records.values()
    }

    /// Set a model's `protected` flag explicitly and return the new value.
    /// `None` if the model is unknown. Persisting is the caller's responsibility
    /// (`save()`), matching the rest of the registry's mutate-then-save pattern.
    ///
    /// Note: this toggles only the per-record flag. A name listed in the
    /// configured `MODEL_PROTECTED` set is *always* protected regardless of this
    /// flag (see [`is_protected`]); clearing the flag on such a model has no
    /// effect on `is_protected`.
    pub fn set_protected(&mut self, name: &str, protected: bool) -> Option<bool> {
        let rec = self.records.get_mut(name)?;
        rec.protected = protected;
        Some(rec.protected)
    }

    /// Number of tracked models (mostly for tests/observability).
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the registry holds no records.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Count of records at each tier as `(hot, warm, cold)`. Used at startup to
    /// log the discovered model distribution after `reconcile()`.
    pub fn tier_counts(&self) -> (usize, usize, usize) {
        let (mut hot, mut warm, mut cold) = (0usize, 0usize, 0usize);
        for rec in self.records.values() {
            match rec.tier {
                StorageTier::Hot => hot += 1,
                StorageTier::Warm => warm += 1,
                StorageTier::Cold => cold += 1,
            }
        }
        (hot, warm, cold)
    }

    /// The configured local Ollama root (warm/hot tier location).
    pub fn local_path(&self) -> &Path {
        &self.local_path
    }

    /// The configured archive root (cold tier location).
    pub fn archive_path(&self) -> &Path {
        &self.archive_path
    }

    /// Promote a model to the `Warm` tier after a successful archive pull
    /// (cold → warm). Sets `tier = Warm`, records the `local_path` it now lives
    /// at, and clears nothing (the archive copy still exists). Returns `false`
    /// if the model is unknown.
    ///
    /// The VRAM load (warm → hot) is handled by the existing lifecycle, which
    /// then calls `set_tier(.., Hot)` + `update_last_loaded`.
    pub fn promote_to_warm(&mut self, name: &str, local_path: &str) -> bool {
        let Some(rec) = self.records.get_mut(name) else {
            return false;
        };
        rec.tier = StorageTier::Warm;
        rec.local_path = Some(local_path.to_string());
        true
    }

    /// Set `last_requested` to now for the named model (no-op if unknown).
    pub fn update_last_requested(&mut self, name: &str) {
        if let Some(rec) = self.records.get_mut(name) {
            rec.last_requested = Some(now_epoch_secs());
        }
    }

    /// Set `last_loaded` to now for the named model (no-op if unknown).
    pub fn update_last_loaded(&mut self, name: &str) {
        if let Some(rec) = self.records.get_mut(name) {
            rec.last_loaded = Some(now_epoch_secs());
        }
    }

    /// Test-only: set `last_requested` to an explicit epoch-seconds value so LRU
    /// ordering tests are deterministic (epoch-second resolution otherwise ties
    /// models created within the same wall-clock second).
    #[cfg(test)]
    pub(crate) fn set_last_requested_for_test(&mut self, name: &str, ts: i64) -> bool {
        match self.records.get_mut(name) {
            Some(rec) => {
                rec.last_requested = Some(ts);
                true
            }
            None => false,
        }
    }

    /// Test-only: clear `last_requested` (→ `None`) to simulate a legacy /
    /// never-requested registry entry for cooldown-eviction tests.
    #[cfg(test)]
    pub(crate) fn clear_last_requested_for_test(&mut self, name: &str) -> bool {
        match self.records.get_mut(name) {
            Some(rec) => {
                rec.last_requested = None;
                true
            }
            None => false,
        }
    }

    /// Set a model's tier.
    ///
    /// Protected models are never auto-archived: this method **refuses** to
    /// demote a protected model to `Cold` (i.e. away from a warm/local state),
    /// returning `false` and leaving the tier unchanged. Promotions and warm/hot
    /// transitions for protected models are allowed. Returns `false` if the
    /// model is unknown.
    pub fn set_tier(&mut self, name: &str, tier: StorageTier) -> bool {
        let Some(rec) = self.records.get_mut(name) else {
            return false;
        };
        if rec.protected && tier == StorageTier::Cold {
            tracing::warn!(
                model = %name,
                "refusing to demote protected model to cold tier (auto-archive guard)"
            );
            return false;
        }
        rec.tier = tier;
        true
    }

    /// Warm, non-protected, non-Hot models sorted by `last_requested` ascending
    /// (least-recently-used first; `None` sorts oldest). Returns owned
    /// `(name, last_requested, size_bytes)` tuples so the caller can iterate
    /// without holding a borrow on the registry across awaits.
    ///
    /// This is the candidate set the TIER-03 eviction sweep considers: Hot models
    /// (loaded in VRAM), protected models, and non-Ollama models are excluded here
    /// so callers never even attempt to evict them.
    ///
    /// Non-Ollama models (e.g. DiffusionGemma) are excluded because the evictor
    /// archives via Ollama manifest trees; without this they'd be picked every
    /// sweep and fail with "local manifest not found", logging a warning each pass.
    pub fn warm_eviction_candidates(&self) -> Vec<(String, Option<i64>, u64)> {
        let mut out: Vec<(String, Option<i64>, u64)> = self
            .records
            .values()
            .filter(|r| {
                r.tier == StorageTier::Warm
                    && r.managed_by == MANAGED_BY_OLLAMA
                    && !self.is_protected(&r.name)
            })
            .map(|r| (r.name.clone(), r.last_requested, r.size_bytes))
            .collect();
        // Ascending by last_requested; None (never requested) is treated as the
        // oldest possible so it evicts first.
        out.sort_by_key(|(_, lr, _)| lr.unwrap_or(i64::MIN));
        out
    }

    /// Record that a model now lives only in the archive after an eviction
    /// (warm → cold): clears `local_path`, sets `archive_path`, and demotes the
    /// tier to `Cold`. Refuses (and leaves tier unchanged) for protected models
    /// via [`set_tier`]. Returns `false` if the model is unknown.
    pub fn mark_evicted_to_archive(&mut self, name: &str, archive_path: &str) -> bool {
        if self.records.get(name).is_none() {
            return false;
        }
        // set_tier refuses to demote protected models to Cold — honour that.
        if !self.set_tier(name, StorageTier::Cold) {
            return false;
        }
        if let Some(rec) = self.records.get_mut(name) {
            rec.local_path = None;
            rec.archive_path = Some(archive_path.to_string());
        }
        true
    }

    /// Whether a model is protected (never auto-archived). Consults both the
    /// record flag and the configured protected list.
    pub fn is_protected(&self, name: &str) -> bool {
        if self.protected.iter().any(|p| p == name) {
            return true;
        }
        self.records.get(name).map(|r| r.protected).unwrap_or(false)
    }
}

/// Recursively walk an Ollama `manifests/` tree and return discovered models.
///
/// Ollama lays manifests out as
/// `<manifests>/<registry>/<namespace>/<model>/<tag>` where the leaf is a JSON
/// file. The model name is `<model>:<tag>` for the `library` namespace,
/// otherwise `<namespace>/<model>:<tag>`. The registry/host component
/// (`registry.ollama.ai`, `hf.co`, …) is not part of the name. Returns an empty
/// vec if the root does not exist.
fn scan_manifest_tree(root: &Path) -> Vec<DiscoveredModel> {
    let mut out = Vec::new();
    if !root.exists() {
        return out;
    }
    // Collect every file leaf with its path components relative to `root`.
    let mut leaves: Vec<PathBuf> = Vec::new();
    collect_files(root, &mut leaves);

    for leaf in leaves {
        let rel = match leaf.strip_prefix(root) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let comps: Vec<String> = rel
            .components()
            .filter_map(|c| match c {
                std::path::Component::Normal(s) => Some(s.to_string_lossy().to_string()),
                _ => None,
            })
            .collect();
        // Expect at least: <registry>/<namespace>/<model>/<tag>
        if comps.len() < 4 {
            continue;
        }
        let tag = &comps[comps.len() - 1];
        let model = &comps[comps.len() - 2];
        let namespace = &comps[comps.len() - 3];
        let name = if namespace == "library" {
            format!("{}:{}", model, tag)
        } else {
            format!("{}/{}:{}", namespace, model, tag)
        };

        let size_bytes = parse_manifest_size(&leaf);
        out.push(DiscoveredModel {
            name,
            size_bytes,
            mtime: mtime_epoch_secs(&leaf),
        });
    }
    out
}

/// Collect every manifest leaf file path under `<root>/manifests` (recursively).
/// Used by the TIER-03 eviction's GC-aware blob removal to find which blobs are
/// still referenced by *other* local models before deleting a shared blob.
/// Returns an empty vec if the tree does not exist.
pub(crate) fn collect_manifest_leaves(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    collect_files(&root.join("manifests"), &mut out);
    out
}

/// Recursively collect all file paths under `dir` into `out`.
fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        match entry.file_type() {
            Ok(ft) if ft.is_dir() => collect_files(&path, out),
            Ok(ft) if ft.is_file() => out.push(path),
            _ => {}
        }
    }
}

/// Sum the `size` fields of `config` + every layer in an Ollama manifest file.
/// A missing/unreadable/non-JSON file yields 0 (best-effort).
fn parse_manifest_size(path: &Path) -> u64 {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(_) => return 0,
    };
    let manifest: OllamaManifest = match serde_json::from_str(&text) {
        Ok(m) => m,
        Err(_) => return 0,
    };
    let config_size = manifest.config.map(|c| c.size).unwrap_or(0);
    let layer_size: u64 = manifest.layers.iter().map(|l| l.size).sum();
    config_size + layer_size
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    /// Write a fake Ollama manifest leaf at
    /// `<root>/manifests/<registry>/<namespace>/<model>/<tag>` with the given
    /// config + layer sizes. Returns the leaf path.
    fn write_manifest(
        root: &Path,
        registry: &str,
        namespace: &str,
        model: &str,
        tag: &str,
        config_size: u64,
        layer_sizes: &[u64],
    ) -> PathBuf {
        let dir = root
            .join("manifests")
            .join(registry)
            .join(namespace)
            .join(model);
        fs::create_dir_all(&dir).unwrap();
        let leaf = dir.join(tag);
        let layers: Vec<serde_json::Value> = layer_sizes
            .iter()
            .map(|s| serde_json::json!({ "size": s, "digest": "sha256:x" }))
            .collect();
        let body = serde_json::json!({
            "config": { "size": config_size, "digest": "sha256:c" },
            "layers": layers,
        });
        fs::write(&leaf, serde_json::to_string(&body).unwrap()).unwrap();
        leaf
    }

    fn reg_at(dir: &Path, protected: Vec<String>) -> ModelRegistry {
        ModelRegistry::new(
            dir.join("registry.json"),
            dir.join("local"),
            dir.join("archive"),
            protected,
        )
    }

    #[test]
    fn reconcile_discovers_local_and_archive_models() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        let local = base.join("local");
        let archive = base.join("archive");

        // Local model: library/qwen3:8b → name "qwen3:8b", size 100+200+300=600
        write_manifest(&local, "registry.ollama.ai", "library", "qwen3", "8b", 100, &[200, 300]);
        // Local namespaced model: hf.co/org/model:tag → "org/model:tag"
        write_manifest(&local, "hf.co", "org", "model", "tag", 10, &[5]);
        // Archive-only model: library/llama:70b → "llama:70b"
        write_manifest(&archive, "registry.ollama.ai", "library", "llama", "70b", 1, &[2, 3]);

        let mut reg = reg_at(base, vec![]);
        reg.reconcile();

        let qwen = reg.get("qwen3:8b").expect("local model present");
        assert_eq!(qwen.tier, StorageTier::Warm);
        assert_eq!(qwen.size_bytes, 600);
        assert!(qwen.local_path.is_some());
        assert!(qwen.last_requested.is_some());

        let ns = reg.get("org/model:tag").expect("namespaced local model");
        assert_eq!(ns.tier, StorageTier::Warm);
        assert_eq!(ns.size_bytes, 15);

        let llama = reg.get("llama:70b").expect("archive model present");
        assert_eq!(llama.tier, StorageTier::Cold);
        assert_eq!(llama.size_bytes, 6);
        assert!(llama.local_path.is_none());
        assert!(llama.archive_path.is_some());
    }

    #[test]
    fn reconcile_demotes_warm_to_cold_when_local_removed_but_archived() {
        // Regression: a model that was Warm (present locally) and is later removed
        // from local disk out-of-band, but still exists in the archive, must be
        // re-tiered to Cold on the next reconcile (stale local_path cleared).
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        let local = base.join("local");
        let archive = base.join("archive");

        // Present BOTH locally and in archive → first reconcile sees it Warm.
        let local_leaf =
            write_manifest(&local, "registry.ollama.ai", "library", "embed", "latest", 10, &[20]);
        write_manifest(&archive, "registry.ollama.ai", "library", "embed", "latest", 10, &[20]);

        let mut reg = reg_at(base, vec![]);
        reg.reconcile();
        let rec = reg.get("embed:latest").expect("present after first reconcile");
        assert_eq!(rec.tier, StorageTier::Warm);
        assert!(rec.local_path.is_some());

        // Simulate an out-of-band `ollama rm`: the local manifest disappears.
        std::fs::remove_file(&local_leaf).unwrap();

        reg.reconcile();
        let rec = reg.get("embed:latest").expect("still present after second reconcile");
        assert_eq!(
            rec.tier,
            StorageTier::Cold,
            "removed-from-local but still archived must become Cold"
        );
        assert!(rec.local_path.is_none(), "stale local_path must be cleared");
        assert!(rec.archive_path.is_some());
    }

    #[test]
    fn save_and_load_round_trips_identically() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        write_manifest(
            &base.join("local"),
            "registry.ollama.ai",
            "library",
            "qwen3-coder",
            "30b",
            1000,
            &[2000],
        );
        let mut reg = reg_at(base, vec!["qwen3-coder:30b".to_string()]);
        reg.reconcile();
        reg.save().unwrap();

        let reloaded = ModelRegistry::load_or_new(
            base.join("registry.json"),
            base.join("local"),
            base.join("archive"),
            vec!["qwen3-coder:30b".to_string()],
        );
        let a = reg.get("qwen3-coder:30b").unwrap();
        let b = reloaded.get("qwen3-coder:30b").unwrap();
        assert_eq!(b.name, a.name);
        assert_eq!(b.tier, a.tier);
        assert_eq!(b.size_bytes, a.size_bytes);
        assert_eq!(b.local_path, a.local_path);
        assert_eq!(b.protected, a.protected);
        assert!(b.protected, "protected flag should round-trip true");
        assert_eq!(reloaded.len(), reg.len());
    }

    #[test]
    fn loads_legacy_flat_registry_and_seeds_backends() {
        // P5: a pre-P5 registry is a bare {name: record} map with no `backend`
        // field and no `backends` section. It must still load, default each
        // model's backend to None, and seed the backend catalogue from env.
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("registry.json");
        let legacy = r#"{"qwen3:8b":{"name":"qwen3:8b","tier":"warm","local_path":"/x",
            "archive_path":null,"size_bytes":100,"last_loaded":null,
            "last_requested":null,"protected":false,"managed_by":"ollama"}}"#;
        std::fs::write(&path, legacy).unwrap();
        std::env::set_var("OLLAMA_URL", "http://localhost:11434");

        let reg = ModelRegistry::load_or_new(
            path,
            tmp.path().join("local"),
            tmp.path().join("archive"),
            vec![],
        );
        let r = reg.get("qwen3:8b").expect("legacy record loads");
        assert_eq!(r.backend, None, "untagged legacy model defaults to None");
        assert!(reg.backend("llama-gpu").is_some(), "generic GPU backend seeded");
        assert!(reg.backend("ollama").is_some(), "primary seeded from OLLAMA_URL");
        // Untagged model resolves to the default backend (the primary ollama).
        assert_eq!(reg.backend_for("qwen3:8b").map(|b| b.name.as_str()), Some("ollama"));
        std::env::remove_var("OLLAMA_URL");
    }

    #[test]
    fn set_model_backend_validates_and_tags() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        write_manifest(&base.join("local"), "registry.ollama.ai", "library", "qwen3-coder", "30b", 1, &[1]);
        let mut reg = reg_at(base, vec![]);
        reg.reconcile();
        // Unknown backend rejected; unknown model rejected.
        assert!(!reg.set_model_backend("qwen3-coder:30b", Some("nope".into())));
        assert!(!reg.set_model_backend("ghost:1", Some("llama-gpu".into())));
        // Valid tag applied + resolves.
        assert!(reg.set_model_backend("qwen3-coder:30b", Some("llama-gpu".into())));
        assert_eq!(
            reg.backend_for("qwen3-coder:30b").map(|b| b.name.as_str()),
            Some("llama-gpu")
        );
    }

    #[test]
    fn tag_vulkan_candidates_dense_large_only() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        // A dense-large model (candidate), a MoE coder (not), and a small model (not).
        write_manifest(&base.join("local"), "registry.ollama.ai", "library", "llama3.3", "70b", 1, &[1]);
        write_manifest(&base.join("local"), "registry.ollama.ai", "library", "qwen3-coder", "30b", 1, &[2]);
        write_manifest(&base.join("local"), "registry.ollama.ai", "library", "qwen3", "8b", 1, &[3]);
        let mut reg = reg_at(base, vec![]);
        reg.reconcile();
        // vulkan backend is seeded (always offered), so tagging applies.
        assert!(reg.backend(backends::VULKAN_BACKEND).is_some());
        let tagged = reg.tag_vulkan_candidates();
        assert_eq!(tagged, vec!["llama3.3:70b".to_string()]);
        assert_eq!(
            reg.backend_for("llama3.3:70b").map(|b| b.name.as_str()),
            Some(backends::VULKAN_BACKEND)
        );
        // Non-candidates keep the default backend.
        assert_ne!(
            reg.get("qwen3-coder:30b").and_then(|r| r.backend.as_deref()),
            Some(backends::VULKAN_BACKEND)
        );
        // Idempotent + never overrides an explicit tag.
        assert!(reg.set_model_backend("qwen3:8b", Some("llama-gpu".into())));
        let again = reg.tag_vulkan_candidates();
        assert!(again.is_empty(), "already-tagged candidate not re-tagged");
        assert_eq!(
            reg.get("qwen3:8b").and_then(|r| r.backend.as_deref()),
            Some("llama-gpu"),
            "explicit operator tag is never overridden"
        );
    }

    #[test]
    fn on_disk_not_in_registry_added_as_warm() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        write_manifest(&base.join("local"), "registry.ollama.ai", "library", "mistral", "7b", 1, &[1]);
        let mut reg = reg_at(base, vec![]);
        assert!(reg.get("mistral:7b").is_none());
        reg.reconcile();
        let rec = reg.get("mistral:7b").expect("newly discovered");
        assert_eq!(rec.tier, StorageTier::Warm);
        assert!(rec.last_requested.is_some(), "new local model gets last_requested = mtime");
    }

    #[test]
    fn in_registry_but_missing_is_kept() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        // Empty local + archive trees (dirs exist so archive scan runs).
        fs::create_dir_all(base.join("local").join("manifests")).unwrap();
        fs::create_dir_all(base.join("archive").join("manifests")).unwrap();

        let mut reg = reg_at(base, vec![]);
        // Inject a record for a model that exists nowhere on disk.
        reg.records.insert(
            "ghost:latest".to_string(),
            ModelRecord {
                name: "ghost:latest".to_string(),
                tier: StorageTier::Warm,
                local_path: Some("/old".to_string()),
                archive_path: None,
                size_bytes: 42,
                last_loaded: None,
                last_requested: Some(123),
                protected: false,
                managed_by: MANAGED_BY_OLLAMA.to_string(),
                backend: None, gguf_path: None,
            },
        );
        reg.reconcile();
        // Not deleted — kept in place.
        assert!(reg.get("ghost:latest").is_some(), "missing model must be kept, not deleted");
    }

    #[test]
    fn protected_flag_prevents_demotion_to_cold() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        write_manifest(&base.join("local"), "registry.ollama.ai", "library", "lumina", "latest", 1, &[1]);
        let mut reg = reg_at(base, vec!["lumina:latest".to_string()]);
        reg.reconcile();
        assert!(reg.is_protected("lumina:latest"));
        // Attempt to demote protected model to Cold → refused.
        let changed = reg.set_tier("lumina:latest", StorageTier::Cold);
        assert!(!changed, "protected model must not be demoted to cold");
        assert_eq!(reg.get("lumina:latest").unwrap().tier, StorageTier::Warm);
        // Non-cold tier change is allowed.
        assert!(reg.set_tier("lumina:latest", StorageTier::Hot));
        assert_eq!(reg.get("lumina:latest").unwrap().tier, StorageTier::Hot);
    }

    #[test]
    fn corrupt_json_rebuilds_without_panic() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        let path = base.join("registry.json");
        fs::write(&path, b"{ this is not valid json").unwrap();
        // Must not panic; yields an empty registry that reconcile can repopulate.
        let reg = ModelRegistry::load_or_new(
            path,
            base.join("local"),
            base.join("archive"),
            vec![],
        );
        assert!(reg.is_empty(), "corrupt JSON should rebuild from empty");
    }

    #[test]
    fn missing_archive_mount_is_local_only_no_crash() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        write_manifest(&base.join("local"), "registry.ollama.ai", "library", "phi", "3", 7, &[8]);
        // archive dir intentionally NOT created → simulates unmounted NFS.
        let mut reg = reg_at(base, vec![]);
        reg.reconcile(); // must not crash
        assert!(reg.get("phi:3").is_some(), "local model still discovered");
        assert_eq!(reg.get("phi:3").unwrap().tier, StorageTier::Warm);
    }

    #[test]
    fn hot_tier_is_preserved_through_reconcile() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        write_manifest(&base.join("local"), "registry.ollama.ai", "library", "hotmodel", "1", 1, &[1]);
        let mut reg = reg_at(base, vec![]);
        reg.reconcile();
        assert!(reg.set_tier("hotmodel:1", StorageTier::Hot));
        // Reconcile again — local model that is Hot stays Hot, not demoted to Warm.
        reg.reconcile();
        assert_eq!(reg.get("hotmodel:1").unwrap().tier, StorageTier::Hot);
    }

    #[test]
    fn update_last_requested_sets_timestamp() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        write_manifest(&base.join("local"), "registry.ollama.ai", "library", "m", "1", 1, &[1]);
        let mut reg = reg_at(base, vec![]);
        reg.reconcile();
        reg.update_last_requested("m:1");
        assert!(reg.get("m:1").unwrap().last_requested.unwrap() > 0);
        // Unknown model is a no-op (no panic).
        reg.update_last_requested("does-not-exist");
    }

    #[test]
    fn promote_to_warm_sets_tier_and_local_path() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        // Archive-only (cold) model.
        write_manifest(&base.join("archive"), "registry.ollama.ai", "library", "cold", "1", 1, &[1]);
        let mut reg = reg_at(base, vec![]);
        reg.reconcile();
        assert_eq!(reg.get("cold:1").unwrap().tier, StorageTier::Cold);
        assert!(reg.promote_to_warm("cold:1", "/opt/ollama-models"));
        let rec = reg.get("cold:1").unwrap();
        assert_eq!(rec.tier, StorageTier::Warm);
        assert_eq!(rec.local_path.as_deref(), Some("/opt/ollama-models"));
        // Unknown model → false.
        assert!(!reg.promote_to_warm("nope", "/x"));
    }

    #[test]
    fn parse_manifest_blobs_collects_digests_and_size() {
        let tmp = tempdir().unwrap();
        let leaf = write_manifest(
            tmp.path(),
            "registry.ollama.ai",
            "library",
            "m",
            "1",
            100,
            &[200, 300],
        );
        let blobs = parse_manifest_blobs(&leaf);
        assert_eq!(blobs.total_size, 600);
        // config digest "sha256:c" + two layer digests "sha256:x".
        assert_eq!(blobs.digests.len(), 3);
        assert!(blobs.digests.iter().all(|d| d.starts_with("sha256:")));
    }

    #[test]
    fn manifest_rel_path_maps_name_layouts() {
        assert_eq!(
            manifest_rel_path("qwen3:8b").unwrap(),
            PathBuf::from("registry.ollama.ai/library/qwen3/8b")
        );
        assert_eq!(
            manifest_rel_path("org/model:tag").unwrap(),
            PathBuf::from("registry.ollama.ai/org/model/tag")
        );
        assert_eq!(
            manifest_rel_path("hf.co/org/model:tag").unwrap(),
            PathBuf::from("hf.co/org/model/tag")
        );
        assert!(manifest_rel_path("no-tag").is_none());
    }

    // ── S80 DGEM-03: non-Ollama (external) model registration ──

    #[test]
    fn register_external_local_is_warm_and_survives_reconcile() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        // Ollama trees exist but are empty, so reconcile would otherwise demote unknown records.
        fs::create_dir_all(base.join("local").join("manifests")).unwrap();
        fs::create_dir_all(base.join("archive").join("manifests")).unwrap();
        let mut reg = reg_at(base, vec![]);

        let tier = reg.register_external(
            "diffusiongemma-26b-a4b",
            "llama-diffusion",
            Some("/opt/models/dgem.gguf".to_string()),
            None,
            16_806_810_336,
        );
        assert_eq!(tier, StorageTier::Warm);

        reg.reconcile(); // must NOT clear the external model's local_path or demote it
        let rec = reg.get("diffusiongemma-26b-a4b").expect("external model kept");
        assert_eq!(rec.tier, StorageTier::Warm);
        assert_eq!(rec.managed_by, "llama-diffusion");
        assert_eq!(rec.local_path.as_deref(), Some("/opt/models/dgem.gguf"));
        assert!(!rec.protected);
    }

    #[test]
    fn external_warm_model_is_not_an_eviction_candidate() {
        let tmp = tempdir().unwrap();
        let mut reg = reg_at(tmp.path(), vec![]);
        reg.register_external(
            "diffusiongemma-26b-a4b",
            "llama-diffusion",
            Some("/opt/models/dgem.gguf".to_string()),
            None,
            16_000_000_000,
        );
        // Warm + unprotected, but non-Ollama → must NOT be offered to the Ollama-based evictor.
        assert!(
            reg.warm_eviction_candidates().iter().all(|(n, _, _)| n != "diffusiongemma-26b-a4b"),
            "non-Ollama model must be excluded from eviction candidates"
        );
    }

    #[test]
    fn register_external_archive_only_is_cold() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        let mut reg = reg_at(base, vec![]);
        let tier = reg.register_external(
            "diffusiongemma-26b-a4b",
            "llama-diffusion",
            None,
            Some("/archive/dgem".to_string()),
            0,
        );
        assert_eq!(tier, StorageTier::Cold);
        assert_eq!(reg.get("diffusiongemma-26b-a4b").unwrap().tier, StorageTier::Cold);
    }

    #[test]
    fn legacy_record_without_managed_by_defaults_to_ollama() {
        // A registry JSON written before the managed_by field must deserialize with the default.
        let json = r#"{"m:latest":{"name":"m:latest","tier":"warm","local_path":"/x","archive_path":null,"size_bytes":1,"last_loaded":null,"last_requested":null,"protected":false}}"#;
        let map: std::collections::HashMap<String, ModelRecord> = serde_json::from_str(json).unwrap();
        assert_eq!(map["m:latest"].managed_by, MANAGED_BY_OLLAMA);
    }

    #[test]
    fn register_diffusiongemma_from_env_noop_without_paths() {
        let tmp = tempdir().unwrap();
        let mut reg = reg_at(tmp.path(), vec![]);
        // Ensure the env vars are unset for this check.
        std::env::remove_var("DGEM_MODEL_PATH");
        std::env::remove_var("DGEM_MODEL_ARCHIVE_PATH");
        reg.register_diffusiongemma_from_env();
        assert!(reg.all_records().next().is_none(), "no registration without configured paths");
    }
}
