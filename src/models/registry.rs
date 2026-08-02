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

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::models::backends::{self, Backend};
use crate::models::rope_ingest::{self, RopeIngestOutcome};
use crate::serving::profile::{RopeScaling, RopeScalingMethod};
use terminus_rs::intake::serving::ServingBackend;

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
    /// YARN-02: RoPE/YaRN scaling metadata ingested from this model's OWN GGUF
    /// kv header at registration time (never invented) — see
    /// `rope_ingest::derive_rope_scaling`. `None` ⇒ not yet ingested, OR
    /// ingestion could not safely produce a block (see
    /// [`rope_scaling_note`](Self::rope_scaling_note)). `Some(RopeScaling {
    /// method: RopeScalingMethod::None, .. })` ⇒ ingested and the model
    /// declares no long-context scaling (the EnvSpec "no-op method" shape from
    /// YARN-01). `validated` is always `false` on an ingested block — YARN-03
    /// (not yet built) owns promoting it.
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    /// YARN-02: set when ingestion found scaling metadata it would not safely
    /// pre-fill (e.g. yarn declared without `original_context_length`) — a
    /// human-readable note for an operator to manually configure this model's
    /// rope scaling. `None` when ingestion succeeded (including "no long
    /// context") or has not run yet.
    #[serde(default)]
    pub rope_scaling_note: Option<String>,
    /// CHRD phase3: the model's ARCHITECTURE family (e.g. `"gptoss"`, `"qwen3"`,
    /// `"llama"`), as reported by the model's own GGUF `general.architecture` kv
    /// ([`ingest_arch`](ModelRegistry::ingest_arch)) or recorded by the MINT
    /// sweep profiler from the backend that actually served it
    /// ([`record_served_arch`](ModelRegistry::record_served_arch)).
    ///
    /// Drives arch-aware backend selection
    /// ([`backend_for_arch_aware`](ModelRegistry::backend_for_arch_aware))
    /// INDEPENDENTLY of the serving-profile DB, so a known-incompatible pairing
    /// (a gptoss model tagged to a llama-server backend) is rejected even when
    /// the RoutingMap is empty (DB unreachable). `None` ⇒ arch not yet known —
    /// the arch guard is inert (unchanged routing), so registries written before
    /// this field existed deserialize and route exactly as before.
    #[serde(default)]
    pub arch: Option<String>,
}

/// Default for `ModelRecord::managed_by` so registries written before this field existed still
/// deserialize (every legacy record is an Ollama-managed model).
fn default_managed_by() -> String {
    "ollama".to_string()
}

/// `managed_by` value for an Ollama-managed model (the reconcile default).
pub const MANAGED_BY_OLLAMA: &str = "ollama";

/// `managed_by` value for a model served entirely through a remote OpenRouter
/// endpoint (no local file, no archive, nothing `reconcile()` should touch).
pub const MANAGED_BY_OPENROUTER: &str = "openrouter-api";

/// The "Owl Alpha" slot's actual OpenRouter model ID.
///
/// "Owl Alpha" (`openrouter/owl-alpha`) was the operator's original free
/// 1M-context target, verified 2026-07-03 directly against OpenRouter's own
/// site/API (`openrouter.ai/openrouter/owl-alpha`,
/// `GET /api/v1/models/openrouter/owl-alpha/endpoints`) to genuinely have
/// `context_length: 1048576` and a 2026-04-28 creation date — but that same
/// verification found `"endpoints":[]`: zero active serving endpoints, so
/// nothing is actually reachable there (no per-token price is published for
/// the same reason — there's no active endpoint to price). It's also absent
/// from the public `GET /api/v1/models` catalog. Confirmed dead, not merely
/// repriced, by an independent live authenticated call from a sibling
/// project (Harmony) returning a clean 404 rather than an auth/pricing error.
///
/// Retargeted (2026-07-03, operator-directed) to `nvidia/nemotron-3-ultra-550b-a55b:free`
/// — same free ($0) pricing tier, same ~1M (1,048,576) token context window,
/// and confirmed live (a real chat-completion round trip succeeded) by the
/// same sibling project. Override via `OPENROUTER_OWL_ALPHA_MODEL` if
/// OpenRouter ever reinstates `openrouter/owl-alpha` or this substitute also
/// rotates off free pricing — no code change needed, see
/// [`ModelRegistry::register_openrouter_owl_alpha_from_env`].
pub const OWL_ALPHA_MODEL_ID: &str = "nvidia/nemotron-3-ultra-550b-a55b:free";

/// File-backed registry of all known models.
#[derive(Debug, Clone)]
pub struct ModelRegistry {
    records: HashMap<String, ModelRecord>,
    backends: HashMap<String, Backend>,
    path: PathBuf,
    protected: Vec<String>,
    local_path: PathBuf,
    archive_path: PathBuf,
    /// TRTR-07: RUNTIME-ONLY residency exemption — the model names currently held
    /// resident by the assistant-mode resident set (embedding / router /
    /// personality roles).
    ///
    /// Deliberately **never persisted** and never written into a record's
    /// `protected` flag: the whole point is that a mode swap (Harmony's BLD-11
    /// idle lease, a MINT sweep entering idle) can drop the exemption and make
    /// every one of these models *immediately* reclaimable again. Persisting it —
    /// or folding it into `protected` — would leave a stale pin behind after a
    /// crash, exactly the failure mode `MODEL_PROTECTED` already bit this fleet
    /// with (a name pinned in config that had never even been pulled).
    residency_exempt: HashSet<String>,
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

/// Owned snapshot of the SLOW manifest-tree filesystem scan produced by
/// [`ModelRegistry::scan_disk`] and consumed by
/// [`ModelRegistry::apply_reconcile`]. Opaque to callers (they only move it from
/// the `spawn_blocking` scan into the brief-locked apply). Holding this owned
/// snapshot is what lets reconcile run its NFS/FS I/O WITHOUT the registry lock
/// and then apply the result under the lock in milliseconds — see the S111
/// regression note on [`ModelRegistry::scan_disk`].
///
/// `Send` (all fields are owned `Vec`/`bool`), so it crosses the
/// `spawn_blocking` boundary.
pub struct ReconcileScan {
    /// Models found under the local manifest tree.
    local: Vec<DiscoveredModel>,
    /// Models found under the archive manifest tree (empty when unmounted).
    archive: Vec<DiscoveredModel>,
    /// Whether the archive root was present/mounted at scan time.
    archive_present: bool,
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
            residency_exempt: HashSet::new(),
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
            // Runtime-only: a restart always starts with an EMPTY exemption set,
            // so nothing stays pinned across a crash. The resident set re-applies
            // it when it warms at startup.
            residency_exempt: HashSet::new(),
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
    /// local `always_on` backend, else any local backend.
    ///
    /// Excludes [`BackendKind::OpenRouter`] (and any future remote,
    /// bearer-authenticated kind) from every fallback tier, not just the
    /// `ollama` lookup. This was found by a live test regression while wiring
    /// up the OpenRouter backend: `openrouter` is `always_on: true` (it has no
    /// process to start/stop), so once it's seeded it satisfies
    /// `.find(|b| b.always_on)` in any environment without `OLLAMA_URL`
    /// configured — silently becoming the *implicit* default for every
    /// untagged model, sending ordinary local-model traffic out to a real
    /// internet API with no key. A remote backend must only ever be reached by
    /// a model **explicitly** tagged to it (`backend_for`/`set_model_backend`/
    /// `register_remote_api_model`), never picked as a fallback.
    pub fn default_backend(&self) -> Option<&Backend> {
        self.backends
            .get("ollama")
            .or_else(|| {
                self.backends
                    .values()
                    .find(|b| b.always_on && b.kind != backends::BackendKind::OpenRouter)
            })
            .or_else(|| {
                self.backends
                    .values()
                    .find(|b| b.kind != backends::BackendKind::OpenRouter)
            })
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

    /// The [`ServingBackend`] serving-tier a chord backend NAME belongs to, for
    /// the arch-exclusion cross-check. Only the three profiled serving tiers map:
    /// `llama-gpu` → llama.cpp-GPU, `ollama` (chord's primary Ollama, which
    /// serves the profile's `ollama-gpu` tier on this fleet) → ollama-GPU,
    /// `ollama-cpu` → genuine CPU. Backends with no serving-profile tier
    /// (`vulkan`, `lemonade-coder`, `openrouter`) return `None` and are never
    /// arch-guarded here — their tag is always honored.
    fn serving_tier_of_backend(name: &str) -> Option<ServingBackend> {
        match name {
            "llama-gpu" => Some(ServingBackend::LlamaGpu),
            "ollama" => Some(ServingBackend::OllamaGpu),
            "ollama-cpu" => Some(ServingBackend::Cpu),
            _ => None,
        }
    }

    /// The chord backend NAME that serves a given [`ServingBackend`] tier —
    /// inverse of [`serving_tier_of_backend`](Self::serving_tier_of_backend).
    fn backend_name_for_tier(tier: ServingBackend) -> &'static str {
        match tier {
            ServingBackend::LlamaGpu => "llama-gpu",
            ServingBackend::OllamaGpu => "ollama",
            ServingBackend::Cpu => "ollama-cpu",
        }
    }

    /// ARCH-AWARE backend resolution (CHRD arch-aware fix).
    ///
    /// Like [`backend_for`](Self::backend_for), but a resolution-time guard:
    /// if the model's registry-tagged backend is arch-EXCLUDED for this model
    /// (its serving tier appears in `excluded_tiers`, i.e. `exclusion_reason !=
    /// None` — e.g. a `gpt-oss` model still tagged `llama-gpu` on disk, which
    /// llama-server cannot load), the stale tag is IGNORED and the model
    /// resolves to `chosen_tier` (the serving RoutingMap's chosen USABLE
    /// backend, mapped back to a concrete chord backend), falling back to the
    /// default backend (Ollama) if that mapping names no known backend. This
    /// overrides bad tags already PERSISTED on disk, not just tagging-time.
    ///
    /// `excluded_tiers` / `chosen_tier` are OWNED values the caller extracts
    /// from the [`RoutingMap`](crate::serving::profile::RoutingMap) BEFORE
    /// locking this registry — the routing_map guard must NOT be held across
    /// the registry lock (lock-order inversion vs `list_models`/`get_model`,
    /// which lock the registry first). See
    /// [`resolve_and_ensure`](crate::models::routing::resolve_and_ensure).
    ///
    /// A model with a usable tagged backend (e.g. `qwen3:8b` on `llama-gpu`,
    /// which IS llama-loadable → no exclusion) is NOT touched. A model with no
    /// serving/RoutingMap row (empty `excluded_tiers`, `None` `chosen_tier`)
    /// keeps the current [`backend_for`] behavior (tag/default) — an unprofiled
    /// model is never hard-failed here.
    pub fn backend_for_arch_aware(
        &self,
        model: &str,
        excluded_tiers: &[ServingBackend],
        chosen_tier: Option<ServingBackend>,
    ) -> Option<&Backend> {
        let rec = self.records.get(model);
        let tag = rec.and_then(|r| r.backend.as_deref());
        let arch = rec.and_then(|r| r.arch.as_deref());
        if let Some(name) = tag {
            // A tag is arch-EXCLUDED for this model when EITHER:
            //  (a) the serving-profile RoutingMap marked its tier excluded
            //      (DB-driven — needs a live serving profile), OR
            //  (b) the model's KNOWN architecture cannot be served by this
            //      backend's runtime (data-driven, DB-INDEPENDENT: fires even
            //      when the RoutingMap is empty, which closes the gptoss→
            //      llama-server hang gap the profile-only guard left open when
            //      the intake DB is unreachable).
            let tier_excluded = Self::serving_tier_of_backend(name)
                .map(|tier| excluded_tiers.contains(&tier))
                .unwrap_or(false);
            let arch_excluded = !self.backend_name_can_serve_arch(name, arch);
            if tier_excluded || arch_excluded {
                // Steer to the chosen usable route — but only if IT too can
                // serve this model's arch — then fall back to the default
                // backend (Ollama, which serves every local architecture) if
                // the chosen tier maps to no known/arch-capable chord backend.
                if let Some(chosen) = chosen_tier {
                    let chosen_name = Self::backend_name_for_tier(chosen);
                    if self.backend_name_can_serve_arch(chosen_name, arch) {
                        if let Some(b) = self.backends.get(chosen_name) {
                            return Some(b);
                        }
                    }
                }
                return self.default_backend();
            }
        }
        // Not excluded (or unprofiled / unmapped tag / unknown arch) → unchanged
        // behavior. An unknown arch never hard-fails a model here.
        self.backend_for(model)
    }

    /// CHRD phase3: whether a defined backend NAME can serve architecture
    /// `arch`, resolved by its [`BackendKind`]. An unknown backend name or an
    /// unknown arch (`None`) ⇒ `true` (never hard-fail on missing data). Thin
    /// wrapper over [`backends::kind_can_serve_arch`].
    fn backend_name_can_serve_arch(&self, name: &str, arch: Option<&str>) -> bool {
        match self.backends.get(name) {
            Some(b) => backends::kind_can_serve_arch(b.kind, arch),
            None => true,
        }
    }

    /// CHRD phase3 (profiler / task-26): record the ARCHITECTURE a MINT sweep
    /// observed a model actually served under, so arch-aware routing becomes
    /// DATA-DRIVEN rather than guessed. Additive and non-blocking: a blank arch
    /// or an unknown model is a no-op (returns `false`) — a profiler failure
    /// never breaks a sweep or a route. Persisting is the caller's job
    /// (`save()`), matching the mutate-then-save pattern.
    pub fn record_served_arch(&mut self, model: &str, arch: &str) -> bool {
        let arch = arch.trim();
        if arch.is_empty() {
            return false;
        }
        match self.records.get_mut(model) {
            Some(rec) => {
                rec.arch = Some(arch.to_string());
                true
            }
            None => false,
        }
    }

    /// CHRD phase3: best-effort derive a model's architecture from its OWN GGUF
    /// `general.architecture` kv (reusing the YARN-02 GGUF reader) and stamp it
    /// onto `rec.arch`, returning the derived arch. No-op (`None`) when the model
    /// is unknown, has no `gguf_path`, or the GGUF is unreadable / declares no
    /// architecture — it never fabricates one and never fails. Persisting is the
    /// caller's job (`save()`).
    pub fn ingest_arch(&mut self, name: &str) -> Option<String> {
        let path = self.records.get(name)?.gguf_path.clone()?;
        let arch = rope_ingest::read_gguf_rope_kv(Path::new(&path))?.architecture?;
        let arch = arch.trim().to_string();
        if arch.is_empty() {
            return None;
        }
        if let Some(rec) = self.records.get_mut(name) {
            rec.arch = Some(arch.clone());
        }
        Some(arch)
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

    /// SLOW half of reconcile: walk the local and (if mounted) archive Ollama
    /// manifest trees and return an owned [`ReconcileScan`].
    ///
    /// This is pure filesystem I/O — it takes NO `&self` and holds NO lock, so
    /// callers MUST run it OFF the registry `Mutex` (via
    /// `tokio::task::spawn_blocking`) and only then apply the result under a
    /// brief lock with [`apply_reconcile`](Self::apply_reconcile).
    ///
    /// **Why this split exists (S111 production regression).** The archive scan
    /// walks the QNAP NFS store; with ~66 cold models it took ~84s live. The old
    /// `reconcile(&mut self)` ran that scan while the sweep held the registry
    /// `Mutex` (`registry.lock().await.reconcile()`), so every
    /// `chat_completions` / `update_last_requested` — which take the SAME lock on
    /// every request — froze for the full scan, wedging all inference and the
    /// control API. The lock must NEVER be held across FS/NFS I/O; only across
    /// the fast in-memory apply.
    pub fn scan_disk(local_path: PathBuf, archive_path: PathBuf) -> ReconcileScan {
        let local = scan_manifest_tree(&local_path.join("manifests"));
        // `exists()` on an NFS mount is itself a (cheap) stat; do it here, off
        // the lock, so `apply_reconcile` never touches the filesystem.
        let archive_present = archive_path.exists();
        let archive = if archive_present {
            scan_manifest_tree(&archive_path.join("manifests"))
        } else {
            Vec::new()
        };
        ReconcileScan { local, archive, archive_present }
    }

    /// FAST half of reconcile: apply a [`ReconcileScan`] to the in-memory
    /// records. Pure in-memory work — NO filesystem I/O — so a caller holds the
    /// registry `Mutex` ONLY for this call (milliseconds), never across
    /// [`scan_disk`](Self::scan_disk). Persisting afterwards (`save()`) is the
    /// caller's job.
    ///
    /// Record updates:
    /// - Local model already known → tier becomes `Warm` (left `Hot` if Hot),
    ///   `local_path` set; new → added as `Warm` with `last_requested = now`
    ///   (the warm/pull moment, NOT the manifest file mtime). A freshly pulled
    ///   model must read ~0h idle so the TIER-04 cooldown pass does not delete it
    ///   mid-use; the manifest mtime can be far in the past (blob dedup /
    ///   copy-preserving timestamps), which used to make a just-pulled model look
    ///   ~918h idle and get evicted seconds after the pull (CHRD-75).
    /// - Archive-only model already known → tier `Cold`, `archive_path` set;
    ///   new → added as `Cold`.
    /// - Registry entries found in neither tier are left in place with a warning
    ///   (never deleted).
    /// - A missing/unmounted archive (`archive_present == false`) is skipped with
    ///   a warning; local-only data is applied (it must never crash).
    pub fn apply_reconcile(&mut self, scan: ReconcileScan) {
        let ReconcileScan { local, archive, archive_present } = scan;
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

        // --- Local ---
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
                            // A newly-discovered local model has just been warmed
                            // (pulled) into the local tier — treat discovery as the
                            // last-access time so it reads ~0h idle, never the
                            // possibly-ancient manifest mtime (CHRD-75).
                            last_requested: Some(now_epoch_secs()),
                            protected,
                            managed_by: MANAGED_BY_OLLAMA.to_string(),
                            backend: None, gguf_path: None,
                        rope_scaling: None, rope_scaling_note: None, arch: None,
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

        // --- Archive (skipped gracefully when not mounted) ---
        if !archive_present {
            tracing::warn!(
                archive_path = %self.archive_path.display(),
                "archive path not present / not mounted; reconciling with local-only data"
            );
        } else {
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
                        rope_scaling: None, rope_scaling_note: None, arch: None,
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

        // --- MSM-05: MODEL_PROTECTED env is authoritative, re-applied every reconcile ---
        // The scans above only union `protected` into records they actually touch (a
        // model present locally and/or in archive this pass). A registry-flag loss (or
        // a record that only exists in-memory) must not leave a configured-protected
        // model unprotected. Union the env list into every matching record
        // unconditionally so protection never depends on persisted state alone — this
        // closes the S111 incident where an empty MODEL_PROTECTED + registry loss would
        // have unprotected the pinned production models.
        for name in &self.protected {
            if let Some(rec) = self.records.get_mut(name) {
                rec.protected = true;
            }
        }
    }

    /// Reconcile the registry against on-disk reality, synchronously (scan +
    /// apply in one call). Convenience for startup and tests where blocking is
    /// acceptable (single-threaded startup, no concurrent lock contention).
    ///
    /// **Do NOT call this while holding the registry `Mutex` in a request-serving
    /// process** — it runs the (potentially multi-second NFS) archive scan
    /// inline. The background sweep and the `/api/models/reconcile` control
    /// endpoint instead call [`scan_disk`](Self::scan_disk) in
    /// `spawn_blocking` and then [`apply_reconcile`](Self::apply_reconcile)
    /// under a brief lock — see the S111 note on `scan_disk`.
    pub fn reconcile(&mut self) {
        let scan = Self::scan_disk(self.local_path.clone(), self.archive_path.clone());
        self.apply_reconcile(scan);
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
                        rope_scaling: None, rope_scaling_note: None, arch: None,
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

    /// Register a model served entirely by a remote HTTP API backend (e.g.
    /// OpenRouter) — no local file, no archive, nothing to pull. Tier is
    /// deliberately `Warm`, never `Cold`: TIER-02's pull-on-miss path
    /// (`chat_completions` in `routes.rs`) fetches from the *archive* when a
    /// model is `Cold`, but a remote-API model has no archive to pull from —
    /// tagging it `Cold` would make every request 503 with "archive pull
    /// failed". `Warm` correctly signals "no pull needed, just dispatch".
    ///
    /// Returns `false` (no-op) if `backend` is not a defined backend in the
    /// catalogue, mirroring [`Self::set_model_backend`]'s validation. Callers
    /// should `save()` afterwards to persist.
    pub fn register_remote_api_model(&mut self, name: &str, managed_by: &str, backend: &str) -> bool {
        if !self.backends.contains_key(backend) {
            return false;
        }
        match self.records.get_mut(name) {
            Some(rec) => {
                rec.tier = StorageTier::Warm;
                rec.managed_by = managed_by.to_string();
                rec.backend = Some(backend.to_string());
            }
            None => {
                self.records.insert(
                    name.to_string(),
                    ModelRecord {
                        name: name.to_string(),
                        tier: StorageTier::Warm,
                        local_path: None,
                        archive_path: None,
                        size_bytes: 0,
                        last_loaded: None,
                        last_requested: None,
                        protected: false,
                        managed_by: managed_by.to_string(),
                        backend: Some(backend.to_string()),
                        gguf_path: None,
                        rope_scaling: None,
                        rope_scaling_note: None,
                        arch: None,
                    },
                );
            }
        }
        true
    }

    /// Register the "Owl Alpha" slot (OpenRouter) from env, tagged to the
    /// `openrouter` backend (see `backends::seed_from_env`). **Opt-in**, via
    /// `OPENROUTER_OWL_ALPHA_ENABLED=1` — no-op otherwise. Defaults to
    /// [`OWL_ALPHA_MODEL_ID`] (currently `nvidia/nemotron-3-ultra-550b-a55b:free`,
    /// a live, free, ~1M-context substitute for the original "Owl Alpha" model,
    /// which OpenRouter's own API confirms has zero active serving endpoints
    /// — see that constant's docs for the full history). Override the actual
    /// model routed here via `OPENROUTER_OWL_ALPHA_MODEL` — e.g. if OpenRouter
    /// ever reinstates `openrouter/owl-alpha`, or this substitute rotates off
    /// free pricing — purely via config, no code change needed.
    ///
    /// Returns `false` if disabled OR if the `openrouter` backend is not in
    /// the catalogue (e.g. `seed_from_env` didn't run, or was overridden by a
    /// hand-edited registry file without it).
    pub fn register_openrouter_owl_alpha_from_env(&mut self) -> bool {
        let enabled = std::env::var("OPENROUTER_OWL_ALPHA_ENABLED")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if !enabled {
            return false;
        }
        let model_id = std::env::var("OPENROUTER_OWL_ALPHA_MODEL")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| OWL_ALPHA_MODEL_ID.to_string());
        // `routes.rs::chat_completions` normalizes an untagged (no `:`) model
        // name to `"{model}:latest"` before it EVER looks the model up in this
        // registry (`registry_key`, routes.rs ~L469) — Ollama's own untagged
        // convention. "openrouter/owl-alpha" has no `:`, so registering it
        // verbatim creates a record `backend_for` can never actually match:
        // every real request misses and silently falls through to the default
        // backend instead of OpenRouter. This was caught by an actual live
        // request through a running chord-proxy, not by a unit test — the
        // registry key MUST mirror that same normalization or the "tagged"
        // model is a dead entry.
        let registry_key = if model_id.contains(':') {
            model_id.clone()
        } else {
            format!("{model_id}:latest")
        };
        let ok = self.register_remote_api_model(&registry_key, MANAGED_BY_OPENROUTER, "openrouter");
        if ok {
            tracing::info!(
                model = %model_id,
                registry_key = %registry_key,
                "registered Owl Alpha (OpenRouter, non-Ollama, remote API)"
            );
        } else {
            tracing::warn!(
                model = %model_id,
                "Owl Alpha registration requested (OPENROUTER_OWL_ALPHA_ENABLED=1) but the \
                 'openrouter' backend is not in the catalogue"
            );
        }
        ok
    }

    /// YARN-02: ingest RoPE/YaRN scaling metadata from a model's OWN GGUF kv
    /// header, at registration time, into `rec.rope_scaling` /
    /// `rec.rope_scaling_note`. Reads `gguf_path` off the already-registered
    /// record — call this AFTER the model's `gguf_path` is set (e.g. after
    /// `register_external`, or once an Ollama-managed record's `gguf_path` has
    /// been resolved). No-op (`None`) when the model is unknown or has no
    /// `gguf_path` — there is nothing to read yet.
    ///
    /// Never fabricates a scale factor: see [`rope_ingest::derive_rope_scaling`].
    /// Persisting the updated record is the caller's responsibility (`save()`),
    /// matching the rest of the registry's mutate-then-save pattern.
    pub fn ingest_rope_scaling(&mut self, name: &str) -> Option<RopeIngestOutcome> {
        let path = self.records.get(name)?.gguf_path.clone()?;
        let outcome = match rope_ingest::read_gguf_rope_kv(Path::new(&path)) {
            Some(kv) => rope_ingest::derive_rope_scaling(&kv),
            None => RopeIngestOutcome::ManualConfigRequired(
                "GGUF unreadable or not a valid GGUF file — manual configuration required"
                    .to_string(),
            ),
        };

        let rec = self.records.get_mut(name)?;
        match &outcome {
            RopeIngestOutcome::NoLongContext => {
                rec.rope_scaling = Some(RopeScaling {
                    method: RopeScalingMethod::None,
                    ..Default::default()
                });
                rec.rope_scaling_note = None;
                tracing::debug!(model = %name, "YARN-02: model declares no long-context rope scaling");
            }
            RopeIngestOutcome::Prefilled(rope) => {
                tracing::info!(
                    model = %name,
                    method = rope.method.as_str(),
                    rope_scale = rope.rope_scale,
                    yarn_orig_ctx = rope.yarn_orig_ctx,
                    target_ctx = rope.target_ctx,
                    "YARN-02: pre-filled rope_scaling from model's own GGUF metadata (unvalidated)"
                );
                rec.rope_scaling = Some(*rope);
                rec.rope_scaling_note = None;
            }
            RopeIngestOutcome::ManualConfigRequired(note) => {
                tracing::warn!(
                    model = %name,
                    note = %note,
                    "YARN-02: rope_scaling not pre-filled — manual configuration required"
                );
                rec.rope_scaling = None;
                rec.rope_scaling_note = Some(note.clone());
            }
        }
        Some(outcome)
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

    /// TIER-05 (cold-quota): drop a model's record entirely — used AFTER its
    /// archive files have been GC-aware deleted by the cold-quota pruner, so a
    /// pruned cold model (no local copy, no archive copy) leaves no dangling
    /// record. Returns `true` if a record was removed. Persisting is the
    /// caller's responsibility (`save()`), matching the mutate-then-save pattern.
    ///
    /// This only touches Chord's own registry — it never deletes the model's
    /// Terminus `model_discovery_candidate` / `model_fleet_catalog` row, so the
    /// model stays re-pullable later (the cold-quota re-pullability invariant).
    pub fn remove_record(&mut self, name: &str) -> bool {
        self.records.remove(name).is_some()
    }

    /// Number of records currently at the [`StorageTier::Cold`] tier — the
    /// denominator for the cold-quota min-keep floor and hard-misconfig cap.
    pub fn cold_count(&self) -> usize {
        self.records
            .values()
            .filter(|r| r.tier == StorageTier::Cold)
            .count()
    }

    /// Whether a model is protected (never auto-archived). Consults the record
    /// flag, the configured protected list, and — TRTR-07 — the runtime residency
    /// exemption held by the assistant-mode resident set.
    ///
    /// Every eviction path (`evict_to_archive`, the TIER-03 disk-pressure pass,
    /// the TIER-04 cooldown pass, `evictable_warm_models`, `set_tier`'s
    /// demote-guard) already routes through this one predicate, so a resident
    /// role's model is skipped by the sweep for exactly as long as the resident
    /// set holds it — and becomes reclaimable the instant it is released.
    pub fn is_protected(&self, name: &str) -> bool {
        if self.protected.iter().any(|p| p == name) {
            return true;
        }
        if self.residency_exempt.contains(name) {
            return true;
        }
        self.records.get(name).map(|r| r.protected).unwrap_or(false)
    }

    /// TRTR-07: replace the runtime residency-exemption set with `names`.
    ///
    /// Wholesale replacement (not a union) so a re-warm after an alias repoint
    /// cannot leave a stale exemption behind for a model no longer held. Runtime
    /// only — never persisted, never touches any record's `protected` flag.
    pub fn set_residency_exempt<I>(&mut self, names: I)
    where
        I: IntoIterator<Item = String>,
    {
        self.residency_exempt = names.into_iter().collect();
    }

    /// TRTR-07: drop the whole residency exemption (the mode-swap RELEASE).
    /// Idempotent — clearing an already-empty set is a no-op. After this, every
    /// previously-held model is immediately reclaimable by the eviction sweep.
    pub fn clear_residency_exempt(&mut self) {
        self.residency_exempt.clear();
    }

    /// Whether `name` is currently held by the resident set (TRTR-07).
    pub fn is_residency_exempt(&self, name: &str) -> bool {
        self.residency_exempt.contains(name)
    }

    /// The current residency exemption, sorted (observability / status).
    pub fn residency_exempt(&self) -> Vec<String> {
        let mut v: Vec<String> = self.residency_exempt.iter().cloned().collect();
        v.sort();
        v
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
    fn default_backend_never_implicitly_picks_openrouter() {
        // Regression test for a real bug caught by the test suite while wiring
        // up OpenRouter: `openrouter` is `always_on: true` (no process
        // lifecycle), so in any environment without OLLAMA_URL configured it
        // used to satisfy `.find(|b| b.always_on)` and become the *implicit*
        // default backend for every UNTAGGED model — silently sending ordinary
        // local-model chat traffic to a real internet API with no key. An
        // untagged/unknown model must always resolve to a local backend (or
        // no backend at all), never the remote OpenRouter one.
        std::env::remove_var("OLLAMA_URL");
        std::env::remove_var("OLLAMA_BASE_URL");
        std::env::remove_var("OLLAMA_CPU_URL");
        std::env::remove_var("LLAMA_SERVER_URL");
        let tmp = tempdir().unwrap();
        let reg = reg_at(tmp.path(), vec![]);
        // openrouter is always seeded (like llama-gpu/vulkan)...
        assert!(reg.backend("openrouter").is_some());
        // ...but must never be what an untagged model resolves to.
        let default = reg.default_backend().expect("some local backend is always seeded");
        assert_ne!(default.name, "openrouter");
        assert_ne!(default.kind, backends::BackendKind::OpenRouter);
        let resolved = reg
            .backend_for("some-untagged-model:latest")
            .expect("falls back to a local default");
        assert_ne!(resolved.kind, backends::BackendKind::OpenRouter);
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
        assert!(rec.last_requested.is_some(), "new local model gets last_requested = now (warm/pull moment)");
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
                        rope_scaling: None, rope_scaling_note: None, arch: None,
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
    fn reconcile_rewarms_stale_cold_record_when_local_manifest_exists() {
        // MSM-01: the periodic sweep now calls reconcile() every tick (not just at
        // startup), so a registry entry left stale at Cold (e.g. from a crash right
        // after a pull promoted it locally but before `promote_to_warm`/`save()`
        // landed) must be corrected the next time reconcile runs, purely from disk
        // reality — a local manifest existing is authoritative over a stale tier.
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        write_manifest(&base.join("local"), "registry.ollama.ai", "library", "stale", "1", 10, &[20]);
        let mut reg = reg_at(base, vec![]);
        reg.reconcile();
        assert_eq!(reg.get("stale:1").unwrap().tier, StorageTier::Warm);

        // Force the record stale (simulating drift): mark it Cold with no local_path,
        // as if an external process/crash left the registry out of sync with disk.
        {
            let rec = reg.records.get_mut("stale:1").unwrap();
            rec.tier = StorageTier::Cold;
            rec.local_path = None;
        }
        assert_eq!(reg.get("stale:1").unwrap().tier, StorageTier::Cold);

        // Local manifest still exists on disk → the next reconcile self-heals it.
        reg.reconcile();
        let rec = reg.get("stale:1").unwrap();
        assert_eq!(rec.tier, StorageTier::Warm, "reconcile re-warms a stale-cold record whose local manifest exists");
        assert!(rec.local_path.is_some());
    }

    #[test]
    fn reconcile_reapplies_env_protected_set_authoritatively() {
        // MSM-05: MODEL_PROTECTED must be re-applied on every reconcile, not just
        // unioned in for records the local/archive scans happen to touch — a
        // persisted `protected: false` (e.g. after a registry loss/rebuild) must not
        // leave a configured-protected model unprotected.
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        write_manifest(&base.join("local"), "registry.ollama.ai", "library", "pinned", "1", 1, &[1]);
        let mut reg = reg_at(base, vec!["pinned:1".to_string()]);
        reg.reconcile();
        assert!(reg.get("pinned:1").unwrap().protected);

        // Simulate a registry rebuild that lost the persisted flag.
        reg.records.get_mut("pinned:1").unwrap().protected = false;
        assert!(!reg.get("pinned:1").unwrap().protected, "flag forced false for the test");

        // A second reconcile must re-mark it protected purely from the env list,
        // even though the model was already known (not newly discovered).
        reg.reconcile();
        assert!(
            reg.get("pinned:1").unwrap().protected,
            "MODEL_PROTECTED must be re-applied authoritatively on every reconcile"
        );
    }

    // ── TRTR-07: runtime residency exemption ────────────────────────────────

    #[test]
    fn residency_exemption_protects_then_releases_immediately() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        write_manifest(&base.join("local"), "registry.ollama.ai", "library", "voice", "1", 1, &[1]);
        let mut reg = reg_at(base, vec![]);
        reg.reconcile();
        assert!(!reg.is_protected("voice:1"), "not protected before residency");

        reg.set_residency_exempt(["voice:1".to_string()]);
        assert!(reg.is_residency_exempt("voice:1"));
        assert!(reg.is_protected("voice:1"), "resident model is skipped by eviction");
        assert!(
            !reg.warm_eviction_candidates().iter().any(|(n, _, _)| n == "voice:1"),
            "a resident model must never be an eviction candidate"
        );

        // RELEASE: immediately reclaimable again — this is what makes a mode swap
        // actually hand the VRAM/disk back.
        reg.clear_residency_exempt();
        assert!(!reg.is_protected("voice:1"));
        assert!(reg.warm_eviction_candidates().iter().any(|(n, _, _)| n == "voice:1"));
        // Idempotent.
        reg.clear_residency_exempt();
        assert!(!reg.is_protected("voice:1"));
    }

    #[test]
    fn residency_exemption_is_replaced_not_unioned_and_never_persisted() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        write_manifest(&base.join("local"), "registry.ollama.ai", "library", "a", "1", 1, &[1]);
        write_manifest(&base.join("local"), "registry.ollama.ai", "library", "b", "1", 1, &[2]);
        let mut reg = reg_at(base, vec![]);
        reg.reconcile();

        reg.set_residency_exempt(["a:1".to_string()]);
        // An alias repoint swaps the held target: the OLD one must not linger.
        reg.set_residency_exempt(["b:1".to_string()]);
        assert!(!reg.is_residency_exempt("a:1"), "stale exemption must be dropped");
        assert!(reg.is_residency_exempt("b:1"));
        assert_eq!(reg.residency_exempt(), vec!["b:1".to_string()]);

        // The exemption is runtime-only: it never leaks into the persisted record
        // flag, so a restart can never resurrect a stale pin.
        assert!(!reg.get("b:1").unwrap().protected);
        reg.save().unwrap();
        let reloaded = ModelRegistry::load_or_new(
            base.join("registry.json"),
            base.join("local"),
            base.join("archive"),
            vec![],
        );
        assert!(reloaded.get("b:1").is_some(), "record round-tripped");
        assert!(!reloaded.is_residency_exempt("b:1"), "exemption must not survive a restart");
        assert!(!reloaded.is_protected("b:1"));
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

    #[test]
    fn register_openrouter_owl_alpha_from_env_noop_when_disabled() {
        let tmp = tempdir().unwrap();
        let mut reg = reg_at(tmp.path(), vec![]);
        std::env::remove_var("OPENROUTER_OWL_ALPHA_ENABLED");
        std::env::remove_var("OPENROUTER_OWL_ALPHA_MODEL");
        assert!(!reg.register_openrouter_owl_alpha_from_env());
        let key = format!("{OWL_ALPHA_MODEL_ID}:latest");
        assert!(reg.backend_for(&key).is_none()
            || reg.backend_for(&key).unwrap().name != "openrouter");
    }

    #[test]
    fn register_openrouter_owl_alpha_from_env_uses_the_default_model_verbatim() {
        // OWL_ALPHA_MODEL_ID (currently the Nemotron substitute) already
        // contains ':' (its own "-free" tag), so the untagged->":latest"
        // normalization branch does NOT apply to it — it must register
        // (and be looked up) verbatim, unmodified.
        let tmp = tempdir().unwrap();
        let mut reg = reg_at(tmp.path(), vec![]);
        std::env::set_var("OPENROUTER_OWL_ALPHA_ENABLED", "1");
        std::env::remove_var("OPENROUTER_OWL_ALPHA_MODEL");
        assert!(reg.register_openrouter_owl_alpha_from_env());

        let resolved = reg
            .backend_for(OWL_ALPHA_MODEL_ID)
            .expect("registered verbatim under its own already-tagged id");
        assert_eq!(resolved.name, "openrouter");
        assert_eq!(resolved.kind, backends::BackendKind::OpenRouter);

        // It must NOT also be double-suffixed to "...:free:latest" — backend_for
        // always falls back to default_backend() rather than returning None, so
        // the real assertion is "not registered as OpenRouter", not "is_none()".
        let double_suffixed = format!("{OWL_ALPHA_MODEL_ID}:latest");
        assert_ne!(
            reg.backend_for(&double_suffixed).map(|b| b.kind),
            Some(backends::BackendKind::OpenRouter),
            "an already-tagged model id must not additionally get :latest appended"
        );

        std::env::remove_var("OPENROUTER_OWL_ALPHA_ENABLED");
    }

    #[test]
    fn register_openrouter_owl_alpha_from_env_normalizes_an_untagged_override_to_match_routes_rs() {
        // Regression test for a real bug caught via a live request through a
        // running chord-proxy: `routes.rs::chat_completions` normalizes an
        // untagged model name to "{model}:latest" (registry_key, ~L469)
        // BEFORE calling `backend_for`. Registering a bare, colon-less model
        // id verbatim creates a record no real (normalized) request could
        // ever match. Exercised here via OPENROUTER_OWL_ALPHA_MODEL, since
        // the current default id already contains ':' and no longer takes
        // this branch itself (see the sibling verbatim test above).
        let tmp = tempdir().unwrap();
        let mut reg = reg_at(tmp.path(), vec![]);
        std::env::set_var("OPENROUTER_OWL_ALPHA_ENABLED", "1");
        std::env::set_var("OPENROUTER_OWL_ALPHA_MODEL", "openrouter/some-untagged-model");
        assert!(reg.register_openrouter_owl_alpha_from_env());

        // The exact key chat_completions computes for an untagged request.
        let registry_key = "openrouter/some-untagged-model:latest";
        let resolved = reg
            .backend_for(registry_key)
            .expect("registered under the :latest-normalized key");
        assert_eq!(resolved.name, "openrouter");
        assert_eq!(resolved.kind, backends::BackendKind::OpenRouter);

        // The un-normalized bare ID must NOT be the registry key (that was
        // the bug) — it resolves only via default_backend() fallback, which
        // (per the other regression test) is guaranteed never OpenRouter.
        let bare_lookup = reg.backend_for("openrouter/some-untagged-model");
        assert_ne!(
            bare_lookup.map(|b| b.kind),
            Some(backends::BackendKind::OpenRouter),
            "bare model ID (no :latest) must not accidentally resolve to OpenRouter"
        );

        std::env::remove_var("OPENROUTER_OWL_ALPHA_ENABLED");
        std::env::remove_var("OPENROUTER_OWL_ALPHA_MODEL");
    }

    // --- YARN-02: rope-scaling ingestion at registration time -------------

    const GGUF_T_U32: u32 = 4;
    const GGUF_T_F32: u32 = 6;
    const GGUF_T_STR: u32 = 8;

    fn gguf_string_value(s: &str) -> Vec<u8> {
        let mut v = (s.len() as u64).to_le_bytes().to_vec();
        v.extend_from_slice(s.as_bytes());
        v
    }
    fn gguf_u32_value(n: u32) -> Vec<u8> {
        n.to_le_bytes().to_vec()
    }
    fn gguf_f32_value(n: f32) -> Vec<u8> {
        n.to_le_bytes().to_vec()
    }

    /// Minimal valid GGUF byte buffer: magic + version 3 + tensor_count 0 +
    /// the given kv pairs. Enough to exercise `ingest_rope_scaling` end-to-end
    /// without a real model file (mirrors `models::rope_ingest`'s own test
    /// builder, kept separate since that one is private to its module).
    fn write_test_gguf(path: &Path, kvs: &[(&str, u32, Vec<u8>)]) {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&(kvs.len() as u64).to_le_bytes()); // kv_count
        for (key, type_tag, value) in kvs {
            buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
            buf.extend_from_slice(key.as_bytes());
            buf.extend_from_slice(&type_tag.to_le_bytes());
            buf.extend_from_slice(value);
        }
        fs::write(path, buf).unwrap();
    }

    /// Register a model with no manifest of its own (so `reconcile()` won't
    /// touch it) and a direct `gguf_path`, the shape `ingest_rope_scaling`
    /// reads from.
    fn register_with_gguf(reg: &mut ModelRegistry, name: &str, gguf_path: &Path) {
        reg.register_external(name, "llama-gpu-gguf", None, None, 0);
        reg.records.get_mut(name).unwrap().gguf_path =
            Some(gguf_path.to_string_lossy().to_string());
    }

    #[test]
    fn ingest_rope_scaling_prefills_yarn_from_gguf_metadata() {
        let tmp = tempdir().unwrap();
        let gguf_path = tmp.path().join("model.gguf");
        write_test_gguf(
            &gguf_path,
            &[
                ("qwen2.context_length", GGUF_T_U32, gguf_u32_value(131072)),
                ("qwen2.rope.scaling.type", GGUF_T_STR, gguf_string_value("yarn")),
                ("qwen2.rope.scaling.factor", GGUF_T_F32, gguf_f32_value(4.0)),
                (
                    "qwen2.rope.scaling.original_context_length",
                    GGUF_T_U32,
                    gguf_u32_value(32768),
                ),
            ],
        );

        let mut reg = reg_at(tmp.path(), vec![]);
        register_with_gguf(&mut reg, "qwen2.5:32b", &gguf_path);

        let outcome = reg.ingest_rope_scaling("qwen2.5:32b").expect("model known");
        match outcome {
            RopeIngestOutcome::Prefilled(rope) => {
                assert_eq!(rope.method, RopeScalingMethod::Yarn);
                assert_eq!(rope.rope_scale, 4.0);
                assert_eq!(rope.yarn_orig_ctx, 32768);
                assert_eq!(rope.target_ctx, 131072);
                assert!(!rope.validated, "ingestion never marks validated — YARN-03's job");
            }
            other => panic!("expected Prefilled(yarn), got {other:?}"),
        }

        let rec = reg.get("qwen2.5:32b").unwrap();
        let stored = rec.rope_scaling.expect("stored on the record");
        assert_eq!(stored.method, RopeScalingMethod::Yarn);
        assert!(rec.rope_scaling_note.is_none());
    }

    #[test]
    fn ingest_rope_scaling_records_linear_method() {
        let tmp = tempdir().unwrap();
        let gguf_path = tmp.path().join("model.gguf");
        write_test_gguf(
            &gguf_path,
            &[
                ("llama.context_length", GGUF_T_U32, gguf_u32_value(16384)),
                ("llama.rope.scaling.type", GGUF_T_STR, gguf_string_value("linear")),
                ("llama.rope.scaling.factor", GGUF_T_F32, gguf_f32_value(2.0)),
            ],
        );

        let mut reg = reg_at(tmp.path(), vec![]);
        register_with_gguf(&mut reg, "llama3:8b", &gguf_path);

        match reg.ingest_rope_scaling("llama3:8b").unwrap() {
            RopeIngestOutcome::Prefilled(rope) => {
                assert_eq!(rope.method, RopeScalingMethod::Linear);
                assert_eq!(rope.rope_scale, 2.0);
            }
            other => panic!("expected Prefilled(linear), got {other:?}"),
        }
        assert_eq!(
            reg.get("llama3:8b").unwrap().rope_scaling.unwrap().method,
            RopeScalingMethod::Linear
        );
    }

    #[test]
    fn ingest_rope_scaling_no_long_context_is_method_none() {
        let tmp = tempdir().unwrap();
        let gguf_path = tmp.path().join("model.gguf");
        write_test_gguf(
            &gguf_path,
            &[("llama.context_length", GGUF_T_U32, gguf_u32_value(4096))],
        );

        let mut reg = reg_at(tmp.path(), vec![]);
        register_with_gguf(&mut reg, "tiny:1b", &gguf_path);

        assert_eq!(
            reg.ingest_rope_scaling("tiny:1b").unwrap(),
            RopeIngestOutcome::NoLongContext
        );
        let rec = reg.get("tiny:1b").unwrap();
        assert_eq!(rec.rope_scaling.as_ref().unwrap().method, RopeScalingMethod::None);
        assert!(rec.rope_scaling_note.is_none());
    }

    #[test]
    fn ingest_rope_scaling_missing_orig_ctx_never_fabricates_and_notes_manual_config() {
        // Declares yarn + a factor + a context_length but NOT
        // original_context_length: ingestion must refuse to guess it, and the
        // record must end up with NO rope_scaling block, only a note.
        let tmp = tempdir().unwrap();
        let gguf_path = tmp.path().join("model.gguf");
        write_test_gguf(
            &gguf_path,
            &[
                ("qwen2.rope.scaling.type", GGUF_T_STR, gguf_string_value("yarn")),
                ("qwen2.rope.scaling.factor", GGUF_T_F32, gguf_f32_value(4.0)),
                ("qwen2.context_length", GGUF_T_U32, gguf_u32_value(131072)),
            ],
        );

        let mut reg = reg_at(tmp.path(), vec![]);
        register_with_gguf(&mut reg, "partial-yarn:7b", &gguf_path);

        match reg.ingest_rope_scaling("partial-yarn:7b").unwrap() {
            RopeIngestOutcome::ManualConfigRequired(note) => {
                assert!(note.contains("original_context_length"));
            }
            other => panic!("expected ManualConfigRequired, got {other:?}"),
        }
        let rec = reg.get("partial-yarn:7b").unwrap();
        assert!(rec.rope_scaling.is_none(), "never a fabricated block");
        assert!(rec.rope_scaling_note.is_some());
    }

    #[test]
    fn ingest_rope_scaling_unreadable_gguf_is_manual_config_not_a_crash() {
        let tmp = tempdir().unwrap();
        let bogus_path = tmp.path().join("not-really-a-model.gguf");
        fs::write(&bogus_path, b"definitely not a gguf file").unwrap();

        let mut reg = reg_at(tmp.path(), vec![]);
        register_with_gguf(&mut reg, "bad-file:1b", &bogus_path);

        match reg.ingest_rope_scaling("bad-file:1b").unwrap() {
            RopeIngestOutcome::ManualConfigRequired(_) => {}
            other => panic!("expected ManualConfigRequired, got {other:?}"),
        }
        assert!(reg.get("bad-file:1b").unwrap().rope_scaling.is_none());
    }

    #[test]
    fn ingest_rope_scaling_unknown_model_is_none() {
        let tmp = tempdir().unwrap();
        let mut reg = reg_at(tmp.path(), vec![]);
        assert!(reg.ingest_rope_scaling("nope:1b").is_none());
    }

    #[test]
    fn ingest_rope_scaling_no_gguf_path_is_none() {
        let tmp = tempdir().unwrap();
        let mut reg = reg_at(tmp.path(), vec![]);
        reg.register_external("ollama-managed:1b", "ollama", Some("/warm/x".to_string()), None, 1);
        assert!(reg.ingest_rope_scaling("ollama-managed:1b").is_none());
    }

    // ── S111: reconcile must not hold the registry lock across the FS scan ────

    #[tokio::test]
    async fn reconcile_scan_off_lock_never_blocks_concurrent_update() {
        // Models the production sweep's CORRECT reconcile shape: the slow
        // manifest scan runs in spawn_blocking OFF the registry lock, and only
        // the fast in-memory `apply_reconcile` takes the lock. A concurrent
        // inference-style `update_last_requested` (registry lock) must acquire
        // the lock ~immediately rather than waiting for the whole scan — the
        // regression was the old `registry.lock().await.reconcile()` holding the
        // lock across the ~84s NFS scan and freezing all inference.
        use std::sync::Arc;
        use std::time::{Duration, Instant};
        use tokio::sync::Mutex;

        let tmp = tempdir().unwrap();
        let base = tmp.path();
        write_manifest(&base.join("local"), "registry.ollama.ai", "library", "m", "1", 1, &[1]);
        let mut reg = reg_at(base, vec![]);
        reg.reconcile(); // seed so "m:1" exists
        let registry = Arc::new(Mutex::new(reg));

        // Reconcile the correct way, with an injected slow scan (400ms) that runs
        // entirely inside spawn_blocking (lock is free during it).
        let reg_for_scan = registry.clone();
        let local = base.join("local");
        let archive = base.join("archive");
        let scan_task = tokio::spawn(async move {
            let scan = tokio::task::spawn_blocking(move || {
                std::thread::sleep(Duration::from_millis(400)); // simulate slow NFS scan
                ModelRegistry::scan_disk(local, archive)
            })
            .await
            .unwrap();
            // Brief-locked apply.
            let mut reg = reg_for_scan.lock().await;
            reg.apply_reconcile(scan);
        });

        // Let the scan task enter its spawn_blocking (the lock is now free).
        tokio::time::sleep(Duration::from_millis(50)).await;

        // The concurrent update must NOT wait ~400ms for the scan.
        let start = Instant::now();
        {
            let mut reg = registry.lock().await;
            reg.update_last_requested("m:1");
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_millis(200),
            "update_last_requested blocked for {elapsed:?}; reconcile must not hold the \
             registry lock across the manifest scan"
        );

        scan_task.await.unwrap();
    }

    #[tokio::test]
    async fn concurrent_gc_and_reconcile_apply_both_complete_no_deadlock() {
        // Lock-ordering / disjointness check: GC holds ONLY `disk_op_lock`
        // (never the registry lock), reconcile-apply holds ONLY the registry
        // lock (never disk_op) — so they can never deadlock regardless of
        // interleaving. Assert both complete well within a generous timeout.
        use std::sync::Arc;
        use std::time::Duration;
        use tokio::sync::Mutex;

        let tmp = tempdir().unwrap();
        let base = tmp.path();
        write_manifest(&base.join("local"), "registry.ollama.ai", "library", "m", "1", 1, &[1]);
        std::fs::create_dir_all(base.join("archive")).unwrap();
        let mut reg = reg_at(base, vec![]);
        reg.reconcile();
        let registry = Arc::new(Mutex::new(reg));
        let disk_op = crate::models::eviction::new_disk_op_lock();
        let local = base.join("local");
        let archive = base.join("archive");

        let reg_gc = registry.clone();
        let dl = disk_op.clone();
        let (lg, ag) = (local.clone(), archive.clone());
        let gc = tokio::spawn(async move {
            crate::models::gc::run_gc(&reg_gc, &lg, &ag, &dl, 0).await
        });

        let reg_rc = registry.clone();
        let (lr, ar) = (local.clone(), archive.clone());
        let rc = tokio::spawn(async move {
            let scan =
                tokio::task::spawn_blocking(move || ModelRegistry::scan_disk(lr, ar)).await.unwrap();
            reg_rc.lock().await.apply_reconcile(scan);
        });

        tokio::time::timeout(Duration::from_secs(10), async move {
            gc.await.unwrap();
            rc.await.unwrap();
        })
        .await
        .expect("concurrent GC + reconcile-apply deadlocked (canonical disk_op→registry order broken)");
    }

    // ── CHRD arch-aware backend resolution ──────────────────────────────────

    #[test]
    fn backend_for_arch_aware_overrides_excluded_tag_and_preserves_usable_tag() {
        use crate::models::backends::{BackendKind, Hardware};
        use crate::serving::profile::RoutingMap;
        use terminus_rs::intake::serving::{
            ExclusionReason, ModelId, RecheckTrigger, Runtime, ServingBackend, ServingProfile,
        };

        fn row(
            model: &str,
            backend: ServingBackend,
            best: Runtime,
            excl: ExclusionReason,
        ) -> ServingProfile {
            ServingProfile {
                model_id: ModelId::from(model),
                backend_tag: backend,
                best_runtime: best,
                env_json: "{}".into(),
                tok_s: None,
                vram_or_ram_peak_gb: None,
                cold_load_s: None,
                keep_warm: false,
                fallback_runtime: None,
                exclusion_reason: excl,
                recheck_trigger: RecheckTrigger::None,
                provenance: None,
            }
        }

        fn backend(name: &str, kind: BackendKind, hw: Hardware) -> Backend {
            Backend {
                name: name.into(),
                url: format!("http://127.0.0.1:0/{name}"),
                hardware: hw,
                kind,
                unit: None,
                always_on: false,
                idle_stop_secs: 0,
                launch: None,
                api_key_env: None,
            }
        }

        fn record(name: &str, backend_tag: &str) -> ModelRecord {
            ModelRecord {
                name: name.into(),
                tier: StorageTier::Warm,
                local_path: None,
                archive_path: None,
                size_bytes: 0,
                last_loaded: None,
                last_requested: None,
                protected: false,
                managed_by: MANAGED_BY_OLLAMA.to_string(),
                backend: Some(backend_tag.into()),
                gguf_path: None,
                rope_scaling: None,
                rope_scaling_note: None,
                arch: None,
            }
        }

        let tmp = tempdir().unwrap();
        let mut reg = reg_at(tmp.path(), vec![]);
        reg.upsert_backend(backend(
            "llama-gpu",
            BackendKind::LlamaServer,
            Hardware::Gpu,
        ));
        reg.upsert_backend(backend("ollama", BackendKind::Ollama, Hardware::Cpu));

        // gpt-oss: tagged `llama-gpu` on disk, but that tier is arch-excluded
        // (llama-server can't load gptoss); the usable route is ollama-gpu.
        reg.records
            .insert("gpt-oss:20b".into(), record("gpt-oss:20b", "llama-gpu"));
        // qwen3:8b: tagged `llama-gpu` and genuinely llama-loadable (no exclusion).
        reg.records
            .insert("qwen3:8b".into(), record("qwen3:8b", "llama-gpu"));

        let routing = RoutingMap::load_from(vec![
            row(
                "gpt-oss:20b",
                ServingBackend::LlamaGpu,
                Runtime::LlamaCpp,
                ExclusionReason::PermanentUnknownArch,
            ),
            row(
                "gpt-oss:20b",
                ServingBackend::OllamaGpu,
                Runtime::Ollama,
                ExclusionReason::None,
            ),
            row(
                "qwen3:8b",
                ServingBackend::LlamaGpu,
                Runtime::LlamaCpp,
                ExclusionReason::None,
            ),
        ]);

        // Resolve exactly as production does: extract the owned excluded/chosen
        // tiers from the routing map (guard would be dropped before the registry
        // lock in the live path), then call the arch-aware resolver.
        let resolve = |reg: &ModelRegistry, model: &str| -> String {
            let mid = ModelId::from(model);
            reg.backend_for_arch_aware(
                model,
                &routing.excluded_tiers(&mid),
                routing.chosen_backend(&mid),
            )
            .expect("resolves")
            .name
            .clone()
        };

        // Excluded tag IGNORED even though rec.backend == "llama-gpu"; resolves
        // to the map's chosen usable route (ollama-gpu → chord "ollama").
        assert_eq!(
            resolve(&reg, "gpt-oss:20b"),
            "ollama",
            "gptoss must NOT dispatch to the arch-excluded llama-gpu backend"
        );

        // Usable tag preserved — no regression for a llama-loadable model.
        assert_eq!(
            resolve(&reg, "qwen3:8b"),
            "llama-gpu",
            "a model with no exclusion keeps its llama-gpu tag"
        );

        // A model with NO serving/RoutingMap row keeps current behavior (its
        // tag) — an unprofiled model is never hard-failed by the guard.
        reg.records
            .insert("mystery:1b".into(), record("mystery:1b", "llama-gpu"));
        assert_eq!(
            resolve(&reg, "mystery:1b"),
            "llama-gpu",
            "unprofiled model keeps its tag (unchanged backend_for behavior)"
        );
    }

    // ── CHRD phase3: back-compat deserialization of pre-phase3 registries ─────

    #[test]
    fn legacy_registry_json_deserializes_with_defaulted_new_fields() {
        // A pre-managed_by / pre-arch bare-map registry (the OLDEST on-disk
        // format): only the always-present fields, tagged to the retired
        // `ollama-cpu` backend. It MUST still parse — every field added since
        // (managed_by, backend, gguf_path, rope_scaling*, arch) is
        // `#[serde(default)]`, so an old registry loads without a hard error.
        let legacy = r#"{
            "legacy-model:1b": {
                "name": "legacy-model:1b",
                "tier": "warm",
                "local_path": null,
                "archive_path": null,
                "size_bytes": 123,
                "last_loaded": null,
                "last_requested": null,
                "protected": false,
                "backend": "ollama-cpu"
            }
        }"#;
        let (records, backends) = parse_registry_file(legacy).expect("legacy json parses");
        assert!(backends.is_empty(), "bare-map legacy file has no backends section");
        let rec = records.get("legacy-model:1b").expect("legacy record loaded");
        assert_eq!(rec.arch, None, "arch defaults to None on an old registry");
        assert_eq!(rec.managed_by, MANAGED_BY_OLLAMA, "managed_by defaults to ollama");
        assert_eq!(rec.backend.as_deref(), Some("ollama-cpu"), "old ollama-cpu tag preserved verbatim");
        assert_eq!(rec.rope_scaling, None);
    }

    #[test]
    fn retired_ollama_cpu_tag_resolves_to_default_backend_not_error() {
        use crate::models::backends::{Backend, BackendKind, Hardware};
        // The unified-GTT cleanup removed the `ollama-cpu` backend from the
        // seeded catalogue, but a model may still carry the retired tag on disk.
        // It must map to the unified representation — resolve to the default
        // (ollama) backend — NEVER hard-fail (None) or panic.
        let tmp = tempdir().unwrap();
        let mut reg = reg_at(tmp.path(), vec![]);
        reg.upsert_backend(Backend {
            name: "ollama".into(),
            url: "http://127.0.0.1:11434".into(),
            hardware: Hardware::Gpu,
            kind: BackendKind::Ollama,
            unit: Some("ollama.service".into()),
            always_on: true,
            idle_stop_secs: 0,
            launch: None,
            api_key_env: None,
        });
        let rec = ModelRecord {
            name: "legacy-model:1b".into(),
            tier: StorageTier::Warm,
            local_path: None,
            archive_path: None,
            size_bytes: 0,
            last_loaded: None,
            last_requested: None,
            protected: false,
            managed_by: MANAGED_BY_OLLAMA.to_string(),
            backend: Some("ollama-cpu".into()), // retired tag, not in catalogue
            gguf_path: None,
            rope_scaling: None,
            rope_scaling_note: None,
            arch: None,
        };
        reg.records.insert("legacy-model:1b".into(), rec);

        // Plain backend_for falls back to the default (ollama) — the retired tag
        // names no known backend.
        assert_eq!(reg.backend_for("legacy-model:1b").unwrap().name, "ollama");
        // The arch-aware resolver (empty RoutingMap, no arch) is identical.
        assert_eq!(
            reg.backend_for_arch_aware("legacy-model:1b", &[], None)
                .unwrap()
                .name,
            "ollama"
        );
    }

    // ── CHRD phase3: DB-INDEPENDENT arch guard (empty RoutingMap) ─────────────

    #[test]
    fn arch_guard_rejects_gptoss_on_llama_gpu_without_any_routing_map() {
        use crate::models::backends::{Backend, BackendKind, Hardware};
        let tmp = tempdir().unwrap();
        let mut reg = reg_at(tmp.path(), vec![]);
        let mk = |name: &str, kind: BackendKind| Backend {
            name: name.into(),
            url: format!("http://127.0.0.1:0/{name}"),
            hardware: Hardware::Gpu,
            kind,
            unit: None,
            always_on: name == "ollama",
            idle_stop_secs: 0,
            launch: None,
            api_key_env: None,
        };
        reg.upsert_backend(mk("llama-gpu", BackendKind::LlamaServer));
        reg.upsert_backend(mk("ollama", BackendKind::Ollama));

        let mk_rec = |name: &str, tag: &str, arch: Option<&str>| ModelRecord {
            name: name.into(),
            tier: StorageTier::Warm,
            local_path: None,
            archive_path: None,
            size_bytes: 0,
            last_loaded: None,
            last_requested: None,
            protected: false,
            managed_by: MANAGED_BY_OLLAMA.to_string(),
            backend: Some(tag.into()),
            gguf_path: None,
            rope_scaling: None,
            rope_scaling_note: None,
            arch: arch.map(str::to_string),
        };
        // gptoss tagged llama-gpu, arch KNOWN — but NO RoutingMap rows at all
        // (empty excluded/None chosen: the DB-unreachable case). The arch guard
        // must still steer it off the arch-incompatible llama-gpu backend.
        reg.records.insert(
            "gpt-oss:20b".into(),
            mk_rec("gpt-oss:20b", "llama-gpu", Some("gpt-oss")),
        );
        // A llama-loadable arch on llama-gpu is preserved (no regression).
        reg.records.insert(
            "qwen3:8b".into(),
            mk_rec("qwen3:8b", "llama-gpu", Some("qwen3")),
        );
        // Unknown arch (None) is never hard-failed — keeps its tag.
        reg.records.insert(
            "mystery:1b".into(),
            mk_rec("mystery:1b", "llama-gpu", None),
        );

        assert_eq!(
            reg.backend_for_arch_aware("gpt-oss:20b", &[], None).unwrap().name,
            "ollama",
            "gptoss must NOT dispatch to llama-gpu even with an empty RoutingMap"
        );
        assert_eq!(
            reg.backend_for_arch_aware("qwen3:8b", &[], None).unwrap().name,
            "llama-gpu",
            "a llama-loadable arch keeps its llama-gpu tag (no regression)"
        );
        assert_eq!(
            reg.backend_for_arch_aware("mystery:1b", &[], None).unwrap().name,
            "llama-gpu",
            "unknown arch never hard-fails — unchanged behavior"
        );
    }

    #[test]
    fn every_real_backend_shape_still_resolves_when_tagged() {
        // Deliverable gate: every currently-configured real backend still
        // resolves for a model explicitly tagged to it. Uses the exact backend
        // shapes seed_from_env produces (kinds/always_on), built deterministically
        // to avoid cross-test env races.
        use crate::models::backends::{Backend, BackendKind, Hardware, LaunchSpec};
        let tmp = tempdir().unwrap();
        let mut reg = reg_at(tmp.path(), vec![]);
        let base = |name: &str, kind: BackendKind, hw: Hardware, always_on: bool| Backend {
            name: name.into(),
            url: format!("http://127.0.0.1:0/{name}"),
            hardware: hw,
            kind,
            unit: None,
            always_on,
            idle_stop_secs: 0,
            launch: None,
            api_key_env: None,
        };
        reg.upsert_backend(base("ollama", BackendKind::Ollama, Hardware::Gpu, true));
        reg.upsert_backend(base("lemonade-coder", BackendKind::LlamaServer, Hardware::Gpu, false));
        let mut lg = base("llama-gpu", BackendKind::LlamaServer, Hardware::Gpu, false);
        lg.launch = Some(LaunchSpec {
            bin: "/x/llama-server".into(),
            args: vec![],
            model_arg: "-m".into(),
            model_from: "ollama-blob".into(),
        });
        reg.upsert_backend(lg);
        reg.upsert_backend(base("vulkan", BackendKind::LlamaServer, Hardware::Gpu, false));
        let mut or = base("openrouter", BackendKind::OpenRouter, Hardware::Cpu, true);
        or.api_key_env = Some("OPENROUTER_API_KEY_CHORDHARMONY".into());
        reg.upsert_backend(or);

        // A model tagged to each backend resolves to THAT backend. For llama-server
        // backends use a llama-loadable arch so the phase3 arch guard is a no-op.
        for name in ["ollama", "lemonade-coder", "llama-gpu", "vulkan", "openrouter"] {
            let model = format!("m-{name}:1b");
            reg.records.insert(
                model.clone(),
                ModelRecord {
                    name: model.clone(),
                    tier: StorageTier::Warm,
                    local_path: None,
                    archive_path: None,
                    size_bytes: 0,
                    last_loaded: None,
                    last_requested: None,
                    protected: false,
                    managed_by: MANAGED_BY_OLLAMA.to_string(),
                    backend: Some(name.to_string()),
                    gguf_path: None,
                    rope_scaling: None,
                    rope_scaling_note: None,
                    arch: Some("qwen3".to_string()),
                },
            );
            assert_eq!(
                reg.backend_for(&model).unwrap().name,
                name,
                "explicit tag resolves to its backend: {name}"
            );
            assert_eq!(
                reg.backend_for_arch_aware(&model, &[], None).unwrap().name,
                name,
                "arch-aware resolve preserves a valid backend tag: {name}"
            );
        }
    }

    // ── CHRD phase3: profiler recording ──────────────────────────────────────

    #[test]
    fn record_served_arch_is_additive_and_non_blocking() {
        let tmp = tempdir().unwrap();
        let mut reg = reg_at(tmp.path(), vec![]);
        reg.records.insert(
            "m:1b".into(),
            ModelRecord {
                name: "m:1b".into(),
                tier: StorageTier::Warm,
                local_path: None,
                archive_path: None,
                size_bytes: 0,
                last_loaded: None,
                last_requested: None,
                protected: false,
                managed_by: MANAGED_BY_OLLAMA.to_string(),
                backend: None,
                gguf_path: None,
                rope_scaling: None,
                rope_scaling_note: None,
                arch: None,
            },
        );
        // Known model, real arch → recorded.
        assert!(reg.record_served_arch("m:1b", " gptoss "));
        assert_eq!(reg.get("m:1b").unwrap().arch.as_deref(), Some("gptoss"));
        // Blank arch → no-op (never clobbers with garbage).
        assert!(!reg.record_served_arch("m:1b", "   "));
        assert_eq!(reg.get("m:1b").unwrap().arch.as_deref(), Some("gptoss"));
        // Unknown model → no-op, no panic.
        assert!(!reg.record_served_arch("nope:1b", "llama"));
    }

    #[test]
    fn ingest_arch_no_ops_without_gguf_path_or_model() {
        let tmp = tempdir().unwrap();
        let mut reg = reg_at(tmp.path(), vec![]);
        // Unknown model → None.
        assert_eq!(reg.ingest_arch("nope:1b"), None);
        // Known model with no gguf_path → None (nothing to read), arch untouched.
        reg.records.insert(
            "m:1b".into(),
            ModelRecord {
                name: "m:1b".into(),
                tier: StorageTier::Warm,
                local_path: None,
                archive_path: None,
                size_bytes: 0,
                last_loaded: None,
                last_requested: None,
                protected: false,
                managed_by: MANAGED_BY_OLLAMA.to_string(),
                backend: None,
                gguf_path: None,
                rope_scaling: None,
                rope_scaling_note: None,
                arch: None,
            },
        );
        assert_eq!(reg.ingest_arch("m:1b"), None);
        assert_eq!(reg.get("m:1b").unwrap().arch, None);
    }
}
