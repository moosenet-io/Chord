//! TIER-05: cold-archive disk-quota + SCORE-BASED pruning.
//!
//! The Ask-4 auto-promotion loop grows the cold archive (NFS) monotonically:
//! every nightly discovery cycle can pull new HuggingFace candidates into cold
//! storage. Without a bound the archive mount fills up. This module adds a
//! quota tier ON TOP of the existing warm→cold eviction (TIER-03/04): when the
//! live archive `df` usage exceeds a quota, the **least-qualified** cold models
//! are pruned (deleted from the archive) — "as new models come in, the
//! least-qualified go out" — until usage is back under quota or a safety floor
//! is hit.
//!
//! ## What it reuses (does NOT reinvent)
//! - **GC-aware, shared-blob-safe deletion:** [`super::eviction::fs_remove_model`]
//!   pointed at the ARCHIVE root (the same primitive warm eviction uses on the
//!   LOCAL root), so a content-addressed blob shared by another archived model is
//!   never deleted while still referenced.
//! - **The global [`DiskOpLock`]:** the destructive prune loop holds it, so it can
//!   never interleave with a warm→cold eviction, an archive pull, or the orphan GC.
//! - **The archive-reachability invariant:** no archive mount ⇒ no pruning (never
//!   delete when the NFS mount is absent/unreadable — mirrors the eviction sweep's
//!   `archive_root.exists()` guard).
//! - **The registry** ([`ModelRecord`](super::registry::ModelRecord) tier /
//!   `size_bytes` / `last_requested` / `archive_path`, [`ModelRegistry::is_protected`]).
//! - **The read-only intake DB pool** (the SAME pool `coding_selector` / `snap` /
//!   the lumina alias updater use, via `terminus_rs::config::intake_database_url`)
//!   for the qualification-score join. Chord stays the SINGLE deletion authority.
//!
//! ## Ranking (least-qualified pruned first)
//! Each cold candidate is ranked by, in precedence order:
//!   1. **PRIMARY — lowest measured `assistant_avg_value`** (from the
//!      `model_dual_profile` view; a swept model's measured quality).
//!   2. **FALLBACK — lowest `discovery_score`** (from `model_discovery_candidate`;
//!      the stored practical estimate — CQH-01/F5: the persisted column is
//!      `discovery_score`, NOT `fit_score`; `fit_score` is a derived blend that is
//!      never stored) — used ONLY for un-swept models (no measured value) that are
//!      past the grace window. Measured candidates are always pruned before
//!      fallback ones (a measured-low model is a stronger evict signal than an
//!      unmeasured one).
//!   3. **tie-break — oldest `last_requested`** (LRU).
//!   4. **final tie-break — largest `size_bytes`** (reclaim the most per delete).
//! Only `tier == Cold` models are candidates.
//!
//! ## Safety gates (ALL mandatory — see [`ColdQuotaConfig`])
//! - **Dry-run default ON** (`MODEL_ARCHIVE_QUOTA_DRY_RUN` default `1`): enumerate
//!   + rank + emit a structured `[cold-quota]` log line (ordered would-prune list
//!   + estimated freed bytes) and DELETE NOTHING. Deletion happens only when
//!   explicitly set to `0`. Same discipline as the Ask-4 shadow flip.
//! - **Protected / keep-set exclusion:** never prune [`ModelRegistry::is_protected`]
//!   (granite4.1 / embedders / coder), never Warm/Hot (only Cold are candidates),
//!   never the CURRENT dynamic lumina proxy target(s) (read from
//!   [`crate::routing::lumina_alias::LuminaAliasStore`]), and never a
//!   VRAM-keep-resident model (CQH-01/F4: the resident-set pins, resolved via
//!   [`crate::routing::resident_set::resident_exempt_models`] — the current
//!   single-owner successor to the retired `MODEL_KEEP_RESIDENT` set). The keep
//!   set is queried LIVE at delete time (CQH-01/F2), not from a stale snapshot, so
//!   a mid-pass lumina-alias repoint protects the new target.
//! - **Grace window** (`MODEL_ARCHIVE_QUOTA_GRACE_DAYS` default 14): a model whose
//!   first-seen/ingest time is within the window is exempt (so the sweep can
//!   MEASURE it before it is judged). The `fit_score` fallback ranking applies
//!   only AFTER grace.
//! - **Min-keep floor** (`MODEL_ARCHIVE_QUOTA_MIN_KEEP` default 20): never prune
//!   below N cold models.
//! - **Hard misconfig floor:** abort the whole pass (delete NOTHING) + alert if a
//!   single pass would remove more than `MODEL_ARCHIVE_QUOTA_MAX_PASS_FRACTION`
//!   (default 0.25 ⇒ >25 %) of the cold models, OR the RESOLVED quota (either
//!   path) is 0 (e.g. `MODEL_ARCHIVE_QUOTA_PERCENT=0`) or below
//!   `MODEL_ARCHIVE_QUOTA_MIN_SANE_GB` (default 10 GiB). A fat-fingered
//!   `MODEL_ARCHIVE_QUOTA_GB=1` or `_PERCENT=0` must NOT wipe the archive.
//! - **Re-pullability:** deletes ONLY the archive files + drops the Chord registry
//!   record. It NEVER writes to the intake DB — the model's Terminus
//!   `model_discovery_candidate` / `model_fleet_catalog` row is left intact, so a
//!   pruned model can be re-pulled later. Chord doesn't own those rows.
//! - **GC-aware + idempotent:** recomputed from the registry + live `df` each pass;
//!   partial runs are safe (a re-run simply continues from the current state).
//!
//! ## No literals / secrets (S1/S7/S9)
//! Every threshold is an env var with a documented default; the intake DB URL
//! comes from `terminus_rs::config::intake_database_url` (never a literal, never
//! a second secret path). No host/IP/token literal appears here.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;
use tracing::{info, warn};

use super::eviction::{fs_remove_model, referenced_by_other_manifests, DiskOpLock};
use super::registry::{parse_manifest_blobs, ModelRegistry, StorageTier};
use super::transfer::{
    blob_filename, find_manifest_leaf, nearest_existing_ancestor, DiskSpaceProbe,
};
use crate::routing::lumina_alias::LuminaAliasStore;
use crate::routing::resident_set::{resident_exempt_models, ResidentSetConfig};

/// 1 GiB in bytes (matches `eviction::BYTES_PER_GB`'s GiB convention).
const GIB: u64 = 1_073_741_824;
const BYTES_PER_GB_F: f64 = 1_073_741_824.0;
const SECS_PER_DAY: i64 = 86_400;

fn bytes_to_gb(bytes: u64) -> f64 {
    bytes as f64 / BYTES_PER_GB_F
}

/// Current wall-clock time in epoch seconds (isolated for deterministic tests via
/// the `_at` orchestrator variant).
fn now_epoch_secs() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Config (env-driven, sanitized at the point of use, documented defaults)
// ─────────────────────────────────────────────────────────────────────────────

/// Defaults (also the sane fallbacks the point-of-use sanitizers apply).
const DEFAULT_QUOTA_PERCENT: u8 = 80;
const DEFAULT_GRACE_DAYS: u64 = 14;
const DEFAULT_MIN_KEEP: usize = 20;
const DEFAULT_MAX_PASS_FRACTION: f64 = 0.25;
const DEFAULT_MIN_SANE_GB: u64 = 10;

/// Full cold-quota configuration. All env-driven with documented defaults;
/// co-located here (self-contained `from_env`) exactly like `IngestConfig`
/// (ASK4) and `AliasUpdaterConfig` (CHRD-91390429), the two nearest precedents.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColdQuotaConfig {
    /// `MODEL_ARCHIVE_QUOTA_GB` — ABSOLUTE quota in GiB. `Some(n)` (n>0) TAKES
    /// PRECEDENCE over the percent quota. `None` (unset) ⇒ use the percent. An
    /// EXPLICIT `=0` does NOT land here as `None` — it is captured by
    /// [`explicit_zero_quota`](Self::explicit_zero_quota) and aborts the pass.
    pub quota_gb: Option<u64>,
    /// CQH-01 (F3): `MODEL_ARCHIVE_QUOTA_GB` was EXPLICITLY set to `0`. Previously
    /// an explicit `=0` was filtered to `None` and silently collapsed to the 80%
    /// percent default (a large quota) — the zero-quota abort never fired. A
    /// deliberate `=0` is a misconfiguration (it would try to empty the archive to
    /// the min-keep floor), so it ABORTS the whole pass (prune nothing). An
    /// UNSET/absent var leaves this `false` and uses the percent default, unchanged.
    pub explicit_zero_quota: bool,
    /// `MODEL_ARCHIVE_QUOTA_PERCENT` — quota as a percent OF THE ARCHIVE MOUNT's
    /// total capacity (default 80). Used only when `quota_gb` is unset.
    pub quota_percent: u8,
    /// `MODEL_ARCHIVE_QUOTA_DRY_RUN` — default `1` (ON). When true (any value but
    /// `0`), the pass enumerates + ranks + logs the would-prune list and DELETES
    /// NOTHING. Set to `0` to ARM deletion.
    pub dry_run: bool,
    /// `MODEL_ARCHIVE_QUOTA_GRACE_DAYS` — a model first-seen within this many days
    /// is exempt (default 14); the `fit_score` fallback applies only after grace.
    pub grace_days: u64,
    /// `MODEL_ARCHIVE_QUOTA_MIN_KEEP` — never prune below this many cold models
    /// (default 20).
    pub min_keep: usize,
    /// `MODEL_ARCHIVE_QUOTA_MAX_PASS_FRACTION` — abort the whole pass (delete
    /// nothing) if it would remove more than this fraction of the cold models
    /// (default 0.25 ⇒ >25 %). The fat-finger guard.
    pub max_pass_fraction: f64,
    /// `MODEL_ARCHIVE_QUOTA_MIN_SANE_GB` — a RESOLVED quota below this many GiB is
    /// treated as a misconfiguration and aborts the pass (default 10). Applies to
    /// the resolved quota regardless of which knob produced it (a fat-fingered
    /// absolute `QUOTA_GB`, OR a tiny `QUOTA_PERCENT` that resolves below the
    /// floor). A resolved quota of exactly 0 (e.g. `QUOTA_PERCENT=0`) always
    /// aborts, independent of this floor.
    pub min_sane_gb: u64,
    /// `MODEL_ARCHIVE_QUOTA_FALLBACK_FIT` — default `true`. When false, un-swept
    /// (no measured `assistant_avg_value`) models are NEVER pruned ("measured-only"
    /// mode); the `fit_score` fallback bucket is disabled entirely. The
    /// operator-tunable "un-swept fit_score-fallback vs measured-only" decision.
    pub fallback_fit: bool,
}

impl Default for ColdQuotaConfig {
    fn default() -> Self {
        ColdQuotaConfig {
            quota_gb: None,
            explicit_zero_quota: false,
            quota_percent: DEFAULT_QUOTA_PERCENT,
            dry_run: true, // SAFE default: deploy inert until audited + armed.
            grace_days: DEFAULT_GRACE_DAYS,
            min_keep: DEFAULT_MIN_KEEP,
            max_pass_fraction: DEFAULT_MAX_PASS_FRACTION,
            min_sane_gb: DEFAULT_MIN_SANE_GB,
            fallback_fit: true,
        }
    }
}

/// Parse the DRY-RUN flag with a DELIBERATELY ASYMMETRIC, fail-safe contract:
/// deletion is armed ONLY when the trimmed value is exactly `"0"`. EVERY other
/// value — unset, `"1"`, `"true"`, but also `"false"`/`"no"`/`"off"`/anything
/// typo'd — keeps dry-run ON. This matches the documented contract
/// ("any value but 0 keeps dry-run; set 0 to arm") and means a fat-fingered
/// `MODEL_ARCHIVE_QUOTA_DRY_RUN=false` can NEVER accidentally arm deletion (the
/// #1 arming footgun): the only way to delete is the unambiguous literal `0`.
fn dry_run_from_raw(raw: Option<String>) -> bool {
    match raw {
        Some(v) => v.trim() != "0",
        None => true,
    }
}

/// CQH-01 (F3): classify a raw `MODEL_ARCHIVE_QUOTA_GB` value into
/// `(quota_gb, explicit_zero)`. An EXPLICIT, well-formed `0` (any surrounding
/// whitespace trimmed) is a misconfiguration signal — returned as
/// `(None, true)` so the pass aborts rather than silently using the percent
/// default. A positive integer ⇒ `(Some(n), false)`. Unset, or any
/// non-integer/garbage value ⇒ `(None, false)` (fall through to the percent
/// quota, unchanged). Pure/testable — the point-of-parse for the arm-path fix.
fn parse_quota_gb_raw(raw: Option<&str>) -> (Option<u64>, bool) {
    match raw.map(str::trim) {
        Some(v) => match v.parse::<u64>() {
            Ok(0) => (None, true),        // explicit zero → abort signal
            Ok(n) => (Some(n), false),    // positive absolute quota
            Err(_) => (None, false),      // garbage → percent path (unchanged)
        },
        None => (None, false), // unset → percent path (unchanged)
    }
}

/// Parse a `0`/`1`/`true`/`false`-style boolean env var, defaulting to `default`.
fn env_bool(key: &str, default: bool) -> bool {
    match std::env::var(key)
        .ok()
        .map(|v| v.trim().to_ascii_lowercase())
    {
        Some(v) if v == "0" || v == "false" || v == "no" || v == "off" => false,
        Some(v) if v == "1" || v == "true" || v == "yes" || v == "on" => true,
        _ => default,
    }
}

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(default)
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

impl ColdQuotaConfig {
    /// Read the config from the environment, applying documented defaults for any
    /// unset/malformed var. The DRY-RUN default is ON — omitting the var leaves the
    /// tier inert (logs a would-prune plan, deletes nothing).
    pub fn from_env() -> Self {
        let (quota_gb, explicit_zero_quota) =
            parse_quota_gb_raw(std::env::var("MODEL_ARCHIVE_QUOTA_GB").ok().as_deref());
        let quota_percent = std::env::var("MODEL_ARCHIVE_QUOTA_PERCENT")
            .ok()
            .and_then(|v| v.trim().parse::<u8>().ok())
            .map(|p| p.min(100))
            .unwrap_or(DEFAULT_QUOTA_PERCENT);
        let max_pass_fraction = std::env::var("MODEL_ARCHIVE_QUOTA_MAX_PASS_FRACTION")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|f| f.is_finite() && *f > 0.0 && *f <= 1.0)
            .unwrap_or(DEFAULT_MAX_PASS_FRACTION);
        ColdQuotaConfig {
            quota_gb,
            explicit_zero_quota,
            quota_percent,
            dry_run: dry_run_from_raw(std::env::var("MODEL_ARCHIVE_QUOTA_DRY_RUN").ok()),
            grace_days: env_u64("MODEL_ARCHIVE_QUOTA_GRACE_DAYS", DEFAULT_GRACE_DAYS),
            min_keep: env_usize("MODEL_ARCHIVE_QUOTA_MIN_KEEP", DEFAULT_MIN_KEEP),
            max_pass_fraction,
            min_sane_gb: env_u64("MODEL_ARCHIVE_QUOTA_MIN_SANE_GB", DEFAULT_MIN_SANE_GB),
            fallback_fit: env_bool("MODEL_ARCHIVE_QUOTA_FALLBACK_FIT", true),
        }
    }

    /// Point-of-use sanitizers so ANY construction path (a test literal, a future
    /// caller) still yields a safe value — a `0`/invalid fraction must never
    /// disable the fat-finger guard, and the min-keep/percent stay in range.
    fn sane_max_pass_fraction(&self) -> f64 {
        if self.max_pass_fraction.is_finite()
            && self.max_pass_fraction > 0.0
            && self.max_pass_fraction <= 1.0
        {
            self.max_pass_fraction
        } else {
            DEFAULT_MAX_PASS_FRACTION
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Qualification scores (read-only join over the intake DB)
// ─────────────────────────────────────────────────────────────────────────────

/// The qualification signals joined from the read-only intake DB, keyed by the
/// registry model name (byte-for-byte the `model_fleet_catalog.model_name` /
/// `model_dual_profile.model_id` identity Chord's registry already uses).
#[derive(Debug, Clone, Default)]
pub struct QualificationScores {
    /// Measured quality: mean `assistant_avg_value` from `model_dual_profile`.
    pub assistant_avg_value: HashMap<String, f64>,
    /// Practical estimate: the stored `discovery_score` from
    /// `model_discovery_candidate` (CQH-01/F5 — the persisted column; `fit_score`
    /// is a derived blend that is never stored). The name is kept for continuity
    /// with the "fallback / practical" ranking bucket.
    pub fit_score: HashMap<String, f64>,
}

impl QualificationScores {
    pub fn is_empty(&self) -> bool {
        self.assistant_avg_value.is_empty() && self.fit_score.is_empty()
    }
}

/// A score-source failure — carries no infra detail (host/DSN), same discipline
/// as [`crate::models::coding_selector::SelectorError`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ColdScoreError {
    NotConfigured,
    StoreUnavailable,
}

impl std::fmt::Display for ColdScoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ColdScoreError::NotConfigured => {
                f.write_str("cold-quota score store is not configured")
            }
            ColdScoreError::StoreUnavailable => {
                f.write_str("cold-quota score store is temporarily unavailable")
            }
        }
    }
}

impl std::error::Error for ColdScoreError {}

/// Source of the qualification scores. Abstracted (mirrors
/// `coding_selector::CodeProfileSource`) so unit tests inject fixtures and only a
/// gated live path touches the read-only intake DB.
#[async_trait]
pub trait ColdScoreSource: Send + Sync {
    async fn load_scores(&self) -> Result<QualificationScores, ColdScoreError>;
}

/// Hot-swappable, fail-open handle shared with [`crate::routes::AppState`] and the
/// nightly sweep — `None` until the intake DB connects (same posture as
/// `coding_profile_source` / `score_source`).
pub type SharedColdScoreSource = Arc<Mutex<Option<Arc<dyn ColdScoreSource>>>>;

/// Production source: reads `assistant_avg_value` (measured) and `fit_score`
/// (practical) over a read-only `sqlx::PgPool`. NO literal DSN/host — the pool is
/// built by the caller from `terminus_rs::config::intake_database_url`, the SAME
/// division of responsibility as `DbCodeProfileSource` / `DbScoreSource`.
pub struct DbColdScoreSource {
    pool: sqlx::PgPool,
}

/// CQH-01 (F5): the fallback (practical) score query. The STORED column is
/// `discovery_score` (Terminus `model_discovery_candidate`, DISC-01 DDL) — NOT
/// `fit_score` (a derived blend that is never persisted, which is why the old
/// query errored every pass). Named so a unit test pins the column without a live
/// Postgres.
pub(crate) const FALLBACK_SCORE_SQL: &str = "SELECT model_name, discovery_score::float8 AS f \
     FROM model_discovery_candidate \
     WHERE discovery_score IS NOT NULL";

impl DbColdScoreSource {
    pub fn new(pool: sqlx::PgPool) -> Self {
        DbColdScoreSource { pool }
    }
}

#[async_trait]
impl ColdScoreSource for DbColdScoreSource {
    async fn load_scores(&self) -> Result<QualificationScores, ColdScoreError> {
        use sqlx::Row;

        // (1) Measured quality — mean `assistant_avg_value` per model across the
        // `model_dual_profile` rows that have an assistant profile (a model has
        // one row per backend_tag/mem_config; average them, mirroring the lumina
        // alias updater's `q` aggregation). Read-only.
        let mut assistant_avg_value: HashMap<String, f64> = HashMap::new();
        let q_rows = sqlx::query(
            "SELECT model_id, avg(assistant_avg_value)::float8 AS q \
             FROM model_dual_profile \
             WHERE has_assistant_profile AND assistant_avg_value IS NOT NULL \
             GROUP BY model_id",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| {
            tracing::warn!(error = %e, "[cold-quota] assistant_avg_value query failed");
            ColdScoreError::StoreUnavailable
        })?;
        for r in q_rows {
            let model_id: String = r.get("model_id");
            if let Ok(Some(v)) = r.try_get::<Option<f64>, _>("q") {
                if v.is_finite() {
                    assistant_avg_value.insert(model_id, v);
                }
            }
        }

        // (2) Practical estimate — `discovery_score` from
        // `model_discovery_candidate`, keyed by `model_name` (the byte-for-byte
        // fleet identity, per the DISC-01 schema). CQH-01 (F5): the STORED practical
        // column is `discovery_score DOUBLE PRECISION` (verified against the
        // canonical DDL, Terminus `migrations/S114-disc01-brochure.sql`, and the
        // `DiscoveryCandidate` row struct). `fit_score` is a DERIVED blend computed
        // at selection time and is NEVER persisted, so the old
        // `SELECT ... fit_score` errored every pass ("column fit_score does not
        // exist") and silently degraded the fallback bucket to empty (measured-only)
        // WITHOUT any operator DDL being possible to fix it. Querying the column that
        // actually exists makes the score-based fallback populate in the common case.
        // Still BEST-EFFORT: a genuine query error (DB down / schema older than
        // DISC-01) is logged and degrades to measured-only — never a hard failure;
        // the whole source is fail-open at the call site regardless.
        let mut fit_score: HashMap<String, f64> = HashMap::new();
        match sqlx::query(FALLBACK_SCORE_SQL)
            .fetch_all(&self.pool)
            .await
        {
            Ok(rows) => {
                for r in rows {
                    let model_name: String = r.get("model_name");
                    if let Ok(Some(v)) = r.try_get::<Option<f64>, _>("f") {
                        if v.is_finite() {
                            fit_score.insert(model_name, v);
                        }
                    }
                }
            }
            Err(e) => {
                // Non-fatal: the measured join already succeeded; the practical
                // score is a fallback signal only. Log and continue with an empty
                // fallback map.
                tracing::warn!(
                    error = %e,
                    "[cold-quota] discovery_score query failed (model_discovery_candidate.discovery_score) — \
                     fallback ranking disabled this pass (measured-only)"
                );
            }
        }

        Ok(QualificationScores {
            assistant_avg_value,
            fit_score,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pure ranking + planning core (no I/O — fully unit-testable)
// ─────────────────────────────────────────────────────────────────────────────

/// One cold model's inputs for ranking. `est_freed_bytes` is the GC-aware
/// exclusive archive size (blobs referenced ONLY by this model), computed by the
/// I/O layer; the pure planner treats it as opaque.
#[derive(Debug, Clone, PartialEq)]
pub struct ColdCandidate {
    pub name: String,
    /// Measured `assistant_avg_value` (swept). `None` ⇒ un-swept.
    pub assistant_avg_value: Option<f64>,
    /// Practical stored `discovery_score` (fallback for un-swept-past-grace;
    /// CQH-01/F5). Field name kept for the "fallback" ranking bucket.
    pub fit_score: Option<f64>,
    /// LRU tie-break signal (`None` = oldest).
    pub last_requested: Option<i64>,
    pub size_bytes: u64,
    /// First-seen / ingest epoch-secs (grace-window signal). `None` ⇒ unknown age.
    pub first_seen: Option<i64>,
    /// GC-aware exclusive archive bytes freed by pruning this model.
    pub est_freed_bytes: u64,
}

/// The single rank key: `Measured` (primary) sorts strictly before `Fallback`
/// (a measured-low model is pruned before any un-swept fallback one); within each
/// group, ascending value (lowest qualification pruned first).
#[derive(Debug, Clone, Copy, PartialEq)]
enum RankKey {
    Measured(f64),
    Fallback(f64),
}

fn cmp_rank(a: &RankKey, b: &RankKey) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let val = |k: &RankKey| match k {
        RankKey::Measured(v) | RankKey::Fallback(v) => *v,
    };
    let group = |k: &RankKey| match k {
        RankKey::Measured(_) => 0u8,
        RankKey::Fallback(_) => 1u8,
    };
    match group(a).cmp(&group(b)) {
        Ordering::Equal => val(a).partial_cmp(&val(b)).unwrap_or(Ordering::Equal),
        other => other,
    }
}

/// One planned prune (the would-prune audit item).
#[derive(Debug, Clone, PartialEq)]
pub struct PrunePlanItem {
    pub name: String,
    pub est_freed_bytes: u64,
    /// Human-readable rank rationale (e.g. `"measured avg=12.3"`, `"fit=0.21"`).
    pub reason: String,
}

/// The computed prune plan for one pass. `aborted` set ⇒ delete NOTHING.
#[derive(Debug, Clone, PartialEq)]
pub struct PrunePlan {
    pub items: Vec<PrunePlanItem>,
    pub est_total_freed_bytes: u64,
    /// `Some(reason)` ⇒ a hard-misconfig floor tripped; the whole pass aborts and
    /// deletes nothing.
    pub aborted: Option<String>,
    pub cold_total: usize,
    pub used_bytes: u64,
    pub quota_bytes: u64,
}

/// Resolve the quota in bytes: the absolute `quota_gb` (GiB) if set, else
/// `quota_percent`% of the archive mount's `total_bytes`.
pub fn resolve_quota_bytes(cfg: &ColdQuotaConfig, total_bytes: u64) -> u64 {
    if let Some(gb) = cfg.quota_gb.filter(|g| *g > 0) {
        gb.saturating_mul(GIB)
    } else {
        ((cfg.quota_percent.min(100) as u128 * total_bytes as u128) / 100) as u64
    }
}

/// Whether current archive usage exceeds the quota.
pub fn is_over_quota(used_bytes: u64, quota_bytes: u64) -> bool {
    used_bytes > quota_bytes
}

/// Build the prune plan (pure). Applies eligibility (keep-set, grace, fallback
/// mode), ranks least-qualified-first, greedily selects until under quota / the
/// min-keep floor, then applies the hard-misconfig floors (which abort the whole
/// plan). `keep_set` contains protected names AND the current lumina targets; the
/// caller pre-filters protected/non-cold, but keep-set membership is enforced here
/// too (defense in depth).
#[allow(clippy::too_many_arguments)]
pub fn plan_prune(
    candidates: &[ColdCandidate],
    keep_set: &HashSet<String>,
    cfg: &ColdQuotaConfig,
    now_secs: i64,
    cold_total: usize,
    used_bytes: u64,
    quota_bytes: u64,
) -> PrunePlan {
    // CQH-01 (F3): an explicit MODEL_ARCHIVE_QUOTA_GB=0 aborts the whole plan
    // (defense in depth — the orchestrator also short-circuits before probing).
    if cfg.explicit_zero_quota {
        return PrunePlan {
            items: Vec::new(),
            est_total_freed_bytes: 0,
            aborted: Some(
                "MODEL_ARCHIVE_QUOTA_GB explicitly set to 0 — refusing to prune (suspected \
                 misconfiguration; unset to use the percent quota)"
                    .to_string(),
            ),
            cold_total,
            used_bytes,
            quota_bytes,
        };
    }

    let grace_secs = (cfg.grace_days as i64).saturating_mul(SECS_PER_DAY);

    // ── Eligibility + rank key ──
    let mut keyed: Vec<(&ColdCandidate, RankKey)> = Vec::new();
    for c in candidates {
        if keep_set.contains(&c.name) {
            continue; // protected / current lumina target — never prune.
        }
        let key = match c.assistant_avg_value {
            // Swept: measured quality is the primary signal (grace does not exempt
            // a model that has already been measured).
            Some(v) if v.is_finite() => RankKey::Measured(v),
            _ => {
                // Un-swept. Measured-only mode never prunes these.
                if !cfg.fallback_fit {
                    continue;
                }
                // Grace: exempt if first-seen within the window. Unknown age ⇒
                // exempt (conservative — never prune a model whose age we can't
                // establish).
                let within_grace = match c.first_seen {
                    Some(fs) => now_secs.saturating_sub(fs) < grace_secs,
                    None => true,
                };
                if within_grace {
                    continue;
                }
                match c.fit_score {
                    Some(f) if f.is_finite() => RankKey::Fallback(f),
                    // Past grace but no qualification signal at all ⇒ skip
                    // (conservative: never prune a model we have zero data on).
                    _ => continue,
                }
            }
        };
        keyed.push((c, key));
    }

    // ── Sort least-qualified first, then LRU (oldest first), then size (largest first) ──
    keyed.sort_by(|(a, ka), (b, kb)| {
        cmp_rank(ka, kb)
            .then_with(|| {
                a.last_requested
                    .unwrap_or(i64::MIN)
                    .cmp(&b.last_requested.unwrap_or(i64::MIN))
            })
            .then_with(|| b.size_bytes.cmp(&a.size_bytes))
            .then_with(|| a.name.cmp(&b.name)) // deterministic final tie-break
    });

    // ── Greedy selection: prune down until under quota OR the min-keep floor ──
    let max_prunable = cold_total.saturating_sub(cfg.min_keep);
    let mut items: Vec<PrunePlanItem> = Vec::new();
    let mut freed: u64 = 0;
    for (c, key) in &keyed {
        if items.len() >= max_prunable {
            break; // min-keep floor reached.
        }
        if used_bytes.saturating_sub(freed) <= quota_bytes {
            break; // back under quota.
        }
        freed = freed.saturating_add(c.est_freed_bytes);
        let reason = match key {
            RankKey::Measured(v) => format!("measured assistant_avg_value={v:.3}"),
            RankKey::Fallback(f) => format!("discovery_score={f:.3} (un-swept, past grace)"),
        };
        items.push(PrunePlanItem {
            name: c.name.clone(),
            est_freed_bytes: c.est_freed_bytes,
            reason,
        });
    }

    // ── Hard-misconfig floors → abort (delete nothing) + alert ──
    // Only meaningful when actually over quota (a pass that wouldn't prune has
    // nothing to refuse).
    let mut aborted: Option<String> = None;
    if used_bytes > quota_bytes {
        let min_sane_bytes = cfg.min_sane_gb.saturating_mul(GIB);
        if quota_bytes == 0 {
            // (a0) A resolved quota of 0 (e.g. MODEL_ARCHIVE_QUOTA_PERCENT=0, or a
            // 0-GB absolute) would try to empty the archive down to the min-keep
            // floor — always a misconfiguration. Applies to BOTH the percent and
            // absolute paths.
            aborted = Some(
                "resolved quota is 0 bytes (MODEL_ARCHIVE_QUOTA_PERCENT=0 / GB=0) — refusing to \
                 prune the entire archive (suspected misconfiguration)"
                    .to_string(),
            );
        } else if quota_bytes < min_sane_bytes {
            // (a) resolved quota below the sane minimum — applies to the RESOLVED
            // quota regardless of which knob set it (a fat-fingered absolute
            // QUOTA_GB, OR a tiny percent that resolves below the floor).
            aborted = Some(format!(
                "resolved quota {:.2} GiB is below the sane minimum {} GiB — refusing to prune \
                 (suspected misconfiguration)",
                bytes_to_gb(quota_bytes),
                cfg.min_sane_gb
            ));
        } else if cold_total > 0 && !items.is_empty() {
            // (b) a single pass would remove more than the max-pass fraction of
            // cold models. The cap is `floor(fraction * cold_total)`; for a tiny
            // cold set (cold_total < 1/fraction, e.g. < 4 at 25 %) the cap rounds
            // to 0 and this guard is INACTIVE — the min-keep floor governs those
            // small sets instead (in production min_keep defaults to 20, so
            // pruning only ever happens once cold_total > 20, where the cap is
            // always ≥ 5 and this guard is live).
            let cap = (cfg.sane_max_pass_fraction() * cold_total as f64).floor() as usize;
            if cap >= 1 && items.len() > cap {
                aborted = Some(format!(
                    "plan would prune {} of {} cold models (> {:.0}% cap of {}) in a single pass — \
                     refusing (suspected misconfiguration / quota too low)",
                    items.len(),
                    cold_total,
                    cfg.sane_max_pass_fraction() * 100.0,
                    cap
                ));
            }
        }
    }

    let est_total_freed_bytes = items.iter().map(|i| i.est_freed_bytes).sum();
    PrunePlan {
        items,
        est_total_freed_bytes,
        aborted,
        cold_total,
        used_bytes,
        quota_bytes,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// I/O layer: archive df probe, GC-aware exclusive size, removal, orchestration
// ─────────────────────────────────────────────────────────────────────────────

/// The archive mount's live quota status from a real `df` probe.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArchiveQuotaStatus {
    pub total_bytes: u64,
    pub used_bytes: u64,
    pub quota_bytes: u64,
    pub over_quota: bool,
}

/// Measure the ARCHIVE mount's usage against the quota — a NEW disk-pressure
/// probe on the ARCHIVE mount, DISTINCT from [`super::eviction::check_disk_pressure`]
/// (which probes the LOCAL disk against a percent). Returns `None` when the probe
/// can't determine total/free (never prune on a probe failure — same fail-safe as
/// `check_disk_pressure`).
pub fn archive_quota_status(
    archive_root: &Path,
    cfg: &ColdQuotaConfig,
    probe: &dyn DiskSpaceProbe,
) -> Option<ArchiveQuotaStatus> {
    let target = nearest_existing_ancestor(archive_root);
    let (total, free) = (probe.total_bytes(&target)?, probe.available_bytes(&target)?);
    if total == 0 {
        return None;
    }
    let used = total.saturating_sub(free);
    let quota_bytes = resolve_quota_bytes(cfg, total);
    Some(ArchiveQuotaStatus {
        total_bytes: total,
        used_bytes: used,
        quota_bytes,
        over_quota: is_over_quota(used, quota_bytes),
    })
}

/// GC-aware exclusive archive size of `model`: the sum of the sizes of the blobs
/// it references that NO OTHER archive manifest references. This is the bytes a
/// prune actually frees (a shared blob frees nothing until its last referencer is
/// gone), so it slightly UNDER-counts when two pruned models share a blob — a safe
/// (conservative) direction. Uses the same primitives as warm eviction's local
/// removal, pointed at the archive root.
pub(crate) fn exclusive_archive_bytes(archive_root: &Path, model: &str) -> u64 {
    let Some(manifest) = find_manifest_leaf(archive_root, model) else {
        return 0;
    };
    let blobs = parse_manifest_blobs(&manifest);
    let others = referenced_by_other_manifests(archive_root, &manifest);
    let blobs_dir = archive_root.join("blobs");
    let mut total: u64 = 0;
    for digest in &blobs.digests {
        if others.contains(digest) {
            continue; // shared → frees nothing.
        }
        if let Ok(md) = std::fs::metadata(blobs_dir.join(blob_filename(digest))) {
            total = total.saturating_add(md.len());
        }
    }
    total
}

/// GC-aware archive removal, injectable for tests (mirrors
/// [`super::eviction::LocalEvictor`]).
#[async_trait]
pub trait ColdArchiveRemover: Send + Sync {
    /// Delete `model`'s archive manifest + the blobs it EXCLUSIVELY owns (never a
    /// blob still referenced by another archive manifest).
    async fn remove(&self, model: &str) -> Result<(), String>;
}

/// Production remover: GC-aware filesystem removal under the ARCHIVE root, reusing
/// [`super::eviction::fs_remove_model`] (the exact primitive warm eviction uses on
/// the LOCAL root). Chord stays the single deletion authority.
pub struct FsColdArchiveRemover {
    archive_root: PathBuf,
}

impl FsColdArchiveRemover {
    pub fn new(archive_root: PathBuf) -> Self {
        Self { archive_root }
    }
}

#[async_trait]
impl ColdArchiveRemover for FsColdArchiveRemover {
    async fn remove(&self, model: &str) -> Result<(), String> {
        let root = self.archive_root.clone();
        let model = model.to_string();
        tokio::task::spawn_blocking(move || fs_remove_model(&root, &model))
            .await
            .map_err(|e| format!("join error: {e}"))?
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CQH-01 (F1/F2/F4): the LIVE keep/exempt set
// ─────────────────────────────────────────────────────────────────────────────

/// A source of the "never-prune" model names, queried LIVE (re-evaluated each
/// time it is asked). CQH-01 (F2): the pass no longer takes a frozen `HashSet`
/// snapshotted once at pass start — it takes this handle and RE-QUERIES it in the
/// delete-time revalidation, so a lumina-alias repoint (or a resident-set change)
/// that happens mid-pass protects the new target instead of the stale one.
pub trait LiveKeepSet: Send + Sync {
    /// The current set of model names that must never be pruned: the live dynamic
    /// lumina proxy target(s) + the VRAM keep-resident pins. Cheap enough to call
    /// once per candidate delete. Lock-free (an `ArcSwap` load); its consistency
    /// w.r.t. a repoint comes from the caller holding [`publish_lock`](Self::publish_lock).
    fn current(&self) -> HashSet<String>;

    /// CQH-01 (F1/F2): the alias-publish serialization lock, if this keep-set has a
    /// publisher that could race a delete. The cold-quota pruner `.lock_owned()`s it
    /// and holds it across the WHOLE per-item critical section — the live keep/
    /// resident re-read → `remove_record` → the async `remover.remove().await` — so
    /// no repoint (which also takes this lock in [`LuminaAliasStore::set`]) can
    /// interleave any part of check→record-drop→fs-delete. `None` ⇒ no publisher can
    /// race (e.g. a frozen `HashSet`), so no serialization is needed. Lock order at
    /// the prune commit (the one site holding both) is ALWAYS publish→registry.
    fn publish_lock(&self) -> Option<Arc<Mutex<()>>> {
        None
    }
}

/// A frozen set is trivially a (constant) live set — lets existing callers/tests
/// pass a plain `HashSet` and keeps the pure-planner fixtures unchanged. No
/// publisher can race it, so `publish_lock` is `None` (default).
impl LiveKeepSet for HashSet<String> {
    fn current(&self) -> HashSet<String> {
        self.clone()
    }
}

/// Production keep-set: the current lumina alias targets (queried LIVE off the
/// shared [`LuminaAliasStore`] — an `ArcSwap`, so a background repoint is visible
/// immediately) UNIONED with the VRAM keep-resident pins (CQH-01/F4, resolved via
/// [`resident_exempt_models`] against the SAME live dynamic store + the static
/// aliases). Holding this handle and calling [`current`](Self::current) at delete
/// time is what makes the arm-path re-check see LIVE keep data (F2) and never
/// prune a keep-resident model (F4).
pub struct LuminaResidentKeepSet {
    dynamic: LuminaAliasStore,
    statics: HashMap<String, String>,
    resident_cfg: ResidentSetConfig,
}

impl LuminaResidentKeepSet {
    pub fn new(
        dynamic: LuminaAliasStore,
        statics: HashMap<String, String>,
        resident_cfg: ResidentSetConfig,
    ) -> Self {
        Self {
            dynamic,
            statics,
            resident_cfg,
        }
    }
}

impl LiveKeepSet for LuminaResidentKeepSet {
    fn current(&self) -> HashSet<String> {
        // Live dynamic lumina targets (F2) …
        let mut set: HashSet<String> = self.dynamic.snapshot().into_values().collect();
        // … plus the VRAM keep-resident pins (F4). Both resolved against the SAME
        // live dynamic store so a mid-pass repoint moves the protection with it.
        set.extend(resident_exempt_models(
            &self.resident_cfg,
            &self.dynamic,
            &self.statics,
        ));
        set
    }

    /// CQH-01 (F1/F2): expose the alias-publish lock so the pruner holds it across
    /// its whole per-item critical section (re-read → record-drop → fs delete).
    /// `LuminaAliasStore::set` takes the SAME lock, so a repoint — of a dynamic
    /// lumina target OR a resident-role alias resolved from the dynamic store —
    /// cannot interleave any part of that section. The registry lock (also held)
    /// handles the protect-toggle path (`set_protected` takes the registry mutex);
    /// this closes the alias/ArcSwap path the registry mutex cannot order against.
    fn publish_lock(&self) -> Option<Arc<Mutex<()>>> {
        Some(self.dynamic.publish_lock_arc())
    }
}

/// Run one cold-quota pass with the real wall clock.
#[allow(clippy::too_many_arguments)]
pub async fn run_cold_quota_pass(
    registry: &Arc<Mutex<ModelRegistry>>,
    archive_root: &Path,
    probe: &dyn DiskSpaceProbe,
    scores: &QualificationScores,
    keep: &dyn LiveKeepSet,
    remover: &dyn ColdArchiveRemover,
    cfg: &ColdQuotaConfig,
    disk_op_lock: &DiskOpLock,
) {
    run_cold_quota_pass_at(
        registry,
        archive_root,
        probe,
        scores,
        keep,
        remover,
        cfg,
        disk_op_lock,
        now_epoch_secs(),
    )
    .await
}

/// One cold-quota pass with an injected `now_secs` (deterministic grace in tests).
///
/// Flow: archive-reachability guard → archive `df` probe → over-quota? →
/// enumerate cold candidates + GC-aware sizes → join scores → plan → emit the
/// structured `[cold-quota]` log → (dry-run: stop) → (armed: delete under the
/// disk-op lock, re-checking `df` after each, dropping the registry record).
#[allow(clippy::too_many_arguments)]
pub async fn run_cold_quota_pass_at(
    registry: &Arc<Mutex<ModelRegistry>>,
    archive_root: &Path,
    probe: &dyn DiskSpaceProbe,
    scores: &QualificationScores,
    keep: &dyn LiveKeepSet,
    remover: &dyn ColdArchiveRemover,
    cfg: &ColdQuotaConfig,
    disk_op_lock: &DiskOpLock,
    now_secs: i64,
) {
    // ── CQH-01 (F3): explicit MODEL_ARCHIVE_QUOTA_GB=0 is a misconfiguration ──
    // A deliberate `=0` used to collapse to the 80% percent default; treat it as an
    // abort (prune nothing) rather than silently resolving to a large quota.
    if cfg.explicit_zero_quota {
        warn!(
            "[cold-quota] ABORT: MODEL_ARCHIVE_QUOTA_GB is explicitly set to 0 — refusing to prune \
             (suspected misconfiguration; unset the var to use the percent quota)"
        );
        return;
    }

    // ── Archive-reachability skip invariant (never prune with no archive mount) ──
    if !archive_root.exists() {
        warn!(
            archive_path = %archive_root.display(),
            "[cold-quota] archive path not present / not mounted; skipping cold-quota pass"
        );
        return;
    }

    // ── Archive df probe (fail-safe: unknown df ⇒ never prune) ──
    let Some(status) = archive_quota_status(archive_root, cfg, probe) else {
        warn!("[cold-quota] archive disk usage could not be probed; skipping (never prune on probe failure)");
        return;
    };
    if !status.over_quota {
        tracing::debug!(
            used_gb = bytes_to_gb(status.used_bytes),
            quota_gb = bytes_to_gb(status.quota_bytes),
            total_gb = bytes_to_gb(status.total_bytes),
            "[cold-quota] archive under quota; nothing to prune"
        );
        return;
    }

    // ── Enumerate cold, non-protected, ollama-managed candidates + cold total ──
    let (cold_candidates, cold_total): (Vec<(String, Option<i64>, u64)>, usize) = {
        let reg = registry.lock().await;
        let cands = reg
            .all_records()
            .filter(|r| {
                r.tier == StorageTier::Cold
                    && r.managed_by == super::registry::MANAGED_BY_OLLAMA
                    && !reg.is_protected(&r.name)
            })
            .map(|r| (r.name.clone(), r.last_requested, r.size_bytes))
            .collect();
        (cands, reg.cold_count())
    };

    // ── GC-aware exclusive size + score join (off the registry lock) ──
    let candidates: Vec<ColdCandidate> = {
        let archive_root = archive_root.to_path_buf();
        let cold_candidates = cold_candidates.clone();
        let assistant = scores.assistant_avg_value.clone();
        let fit = scores.fit_score.clone();
        // Filesystem walk (exclusive-size) is blocking → off the reactor.
        tokio::task::spawn_blocking(move || {
            cold_candidates
                .into_iter()
                .map(|(name, last_requested, size_bytes)| {
                    let est_freed_bytes = exclusive_archive_bytes(&archive_root, &name);
                    ColdCandidate {
                        assistant_avg_value: assistant.get(&name).copied(),
                        fit_score: fit.get(&name).copied(),
                        last_requested,
                        size_bytes,
                        // first-seen proxy: the archive manifest mtime captured as
                        // `last_requested` at reconcile for a never-served cold
                        // model (its ingest time). No dependency on an intake
                        // ingest-timestamp column. See the module docs.
                        first_seen: last_requested,
                        est_freed_bytes,
                        name,
                    }
                })
                .collect()
        })
        .await
        .unwrap_or_default()
    };

    // CQH-01 (F2): snapshot the LIVE keep set for PLANNING. The delete-time
    // revalidation re-queries `keep.current()` again so a mid-pass repoint is
    // still honored even though this snapshot is stale by then.
    let keep_snapshot = keep.current();
    let plan = plan_prune(
        &candidates,
        &keep_snapshot,
        cfg,
        now_secs,
        cold_total,
        status.used_bytes,
        status.quota_bytes,
    );

    // ── Structured [cold-quota] audit log (the ordered would-prune list) ──
    let ordered: Vec<String> = plan
        .items
        .iter()
        .map(|i| {
            format!(
                "{} (~{:.2} GiB; {})",
                i.name,
                bytes_to_gb(i.est_freed_bytes),
                i.reason
            )
        })
        .collect();
    info!(
        used_gb = bytes_to_gb(status.used_bytes),
        quota_gb = bytes_to_gb(status.quota_bytes),
        total_gb = bytes_to_gb(status.total_bytes),
        cold_total = plan.cold_total,
        would_prune = plan.items.len(),
        est_freed_gb = bytes_to_gb(plan.est_total_freed_bytes),
        dry_run = cfg.dry_run,
        "[cold-quota] over quota — plan: prune {} of {} cold model(s), est freed ~{:.2} GiB; order=[{}]",
        plan.items.len(),
        plan.cold_total,
        bytes_to_gb(plan.est_total_freed_bytes),
        ordered.join(", "),
    );

    // ── Hard-misconfig floor → abort + alert, delete nothing ──
    if let Some(reason) = &plan.aborted {
        warn!("[cold-quota] ABORT (delete nothing): {reason}");
        return;
    }

    if plan.items.is_empty() {
        tracing::debug!("[cold-quota] over quota but no eligible candidate to prune (all protected/kept/grace/min-keep)");
        return;
    }

    // ── Dry-run: log the plan, delete NOTHING ──
    if cfg.dry_run {
        info!(
            "[cold-quota] DRY-RUN (MODEL_ARCHIVE_QUOTA_DRY_RUN=1): would prune {} model(s) \
             (~{:.2} GiB) — deleting nothing. Set MODEL_ARCHIVE_QUOTA_DRY_RUN=0 to arm.",
            plan.items.len(),
            bytes_to_gb(plan.est_total_freed_bytes),
        );
        return;
    }

    // ── ARMED: delete under the disk-op lock, re-checking df after each prune ──
    let _guard = disk_op_lock.lock().await;
    let mut pruned = 0usize;
    let mut freed_actual: u64 = 0;
    for item in &plan.items {
        // Re-check live df: stop as soon as we're back under quota (idempotent,
        // GC-aware — df reflects what shared-blob deletes actually freed). A probe
        // FAILURE mid-pass STOPS the loop (fail-safe): never keep deleting when we
        // can no longer confirm we're still over quota.
        match archive_quota_status(archive_root, cfg, probe) {
            Some(cur) if !cur.over_quota => break,
            Some(_) => {}
            None => {
                warn!("[cold-quota] archive df unavailable mid-pass; stopping (fail-safe)");
                break;
            }
        }

        // CQH-01 (F1 + F2): ATOMIC revalidate + commit. The plan was computed
        // BEFORE we took the disk-op lock. `disk_op_lock` blocks a concurrent
        // cold→warm pull's copy, but the control-plane registry mutations (a
        // protect toggle, a lumina-alias repoint) do NOT take it — so a model can
        // become protected/kept in the gap between the plan and this delete. The
        // OLD code re-checked eligibility under the registry lock, then DROPPED the
        // lock before the async remove + a separate record-drop — leaving a window
        // where a protect/repoint could still race the delete. Here the final
        // re-check AND the registry-record removal happen under ONE registry lock
        // hold, so the commit point (record removal) is atomic with the decision:
        //   * a model that became protected / promoted to Warm / entered the LIVE
        //     keep set (F2: `keep.current()` re-queried HERE, not the stale
        //     snapshot) is skipped — its archive + record survive;
        //   * once we commit (drop the record under the lock) no concurrent protect
        //     can "save" it, and the fs delete follows.
        // We cannot hold the registry lock across the async `remover.remove().await`
        // (it is fs I/O and the remover may itself need the registry lock — see the
        // TOCTOU test), so the record is committed-removed FIRST, then the fs delete
        // runs. If the fs delete then fails, the archive files are left behind but
        // untracked; the next reconcile re-tiers them as Cold (idempotent), so the
        // pass never wedges. A missing manifest ⇒ already gone / archive
        // unreachable → skip without touching the record.
        if find_manifest_leaf(archive_root, &item.name).is_none() {
            warn!(
                model = %item.name,
                "[cold-quota] archive manifest not found at delete time (already gone / archive unreachable) — skipping"
            );
            continue;
        }

        // CQH-01 (F1/F2, review r3): hold the alias-publish lock across the ENTIRE
        // per-item critical section — the live keep/resident re-read, the
        // `remove_record`, AND the async `remover.remove().await` (the fs delete).
        // `LuminaAliasStore::set` takes the SAME lock, so no repoint can interleave
        // ANY part of check→record-drop→fs-delete; an alias can no longer point at a
        // candidate in the record-drop→fs-delete window and have the scheduled
        // delete wipe a now-live target. Acquired PER ITEM (released each iteration)
        // so the alias updater is never blocked across the whole multi-delete pass,
        // and a repoint BETWEEN items is honored by the next item's re-read.
        //
        // LOCK ORDER (the ONLY site holding both): publish → registry. `set` takes
        // only publish; the registry methods take only registry; `resolve`/`snapshot`
        // take neither. So there is no registry→publish path and no inversion. The
        // registry lock is released BEFORE the fs delete (only publish spans it), so
        // the slow NFS delete never blocks chat/`update_last_requested`.
        let _publish = match keep.publish_lock() {
            Some(m) => Some(m.lock_owned().await),
            None => None,
        };

        // `committed` ⇒ the record was dropped (proceed to fs delete);
        // `min_keep_stop` ⇒ the live floor was hit (stop the whole loop).
        let mut committed = false;
        let mut min_keep_stop = false;
        {
            let mut reg = registry.lock().await;
            if reg.cold_count() <= cfg.min_keep {
                min_keep_stop = true;
            } else {
                // The registry lock additionally serializes the protect-toggle path
                // (`set_protected`). An item that became protected / promoted to Warm
                // / a LIVE keep or resident target is skipped; otherwise drop the
                // record (the Terminus discovery/catalog row is NEVER touched →
                // re-pullable).
                let live_keep = keep.current();
                let still_eligible = reg
                    .get(&item.name)
                    .map(|r| r.tier == StorageTier::Cold)
                    .unwrap_or(false)
                    && !reg.is_protected(&item.name)
                    && !live_keep.contains(&item.name);
                if still_eligible {
                    reg.remove_record(&item.name);
                    if let Err(e) = reg.save() {
                        warn!(
                            "[cold-quota] failed to persist registry after pruning {}: {e}",
                            item.name
                        );
                    }
                    committed = true;
                }
            }
            drop(reg); // release registry BEFORE the (slow, NFS) fs delete
        }
        if min_keep_stop {
            info!("[cold-quota] live cold count at min-keep floor mid-pass; stopping");
            break; // `_publish` drops here
        }
        if !committed {
            warn!(
                model = %item.name,
                "[cold-quota] item no longer eligible at delete time (promoted/protected/kept) — skipping"
            );
            continue; // `_publish` drops here
        }

        // Still under `_publish`: the fs delete cannot race a repoint.
        match remover.remove(&item.name).await {
            Ok(()) => {
                pruned += 1;
                freed_actual = freed_actual.saturating_add(item.est_freed_bytes);
                info!(
                    model = %item.name,
                    est_freed_gb = bytes_to_gb(item.est_freed_bytes),
                    "[cold-quota] pruned cold model from archive ({})",
                    item.reason
                );
            }
            Err(e) => {
                // Record already committed-removed; files remain and will be
                // re-tiered as Cold by the next reconcile (idempotent).
                warn!(
                    model = %item.name,
                    error = %e,
                    "[cold-quota] archive delete failed after record drop; files left for reconcile to re-track"
                );
            }
        }
        drop(_publish); // release the publish lock before the next item
    }
    info!(
        pruned,
        est_freed_gb = bytes_to_gb(freed_actual),
        "[cold-quota] pass complete"
    );
}

/// Convenience wrapper for the two trigger sites (post-ingest pre-flight + the
/// nightly sweep): resolve the score source (fail-open), build the fs remover, and
/// run one pass. A `None`/unconnected score source yields empty scores (measured
/// join absent ⇒ only grace/fit gates matter; the pass still enforces the quota
/// using whatever signals exist). Never panics.
#[allow(clippy::too_many_arguments)]
pub async fn run_cold_quota_pass_with_source(
    registry: &Arc<Mutex<ModelRegistry>>,
    archive_root: &Path,
    probe: &dyn DiskSpaceProbe,
    score_source: &SharedColdScoreSource,
    keep: &dyn LiveKeepSet,
    cfg: &ColdQuotaConfig,
    disk_op_lock: &DiskOpLock,
) {
    let scores = {
        let guard = score_source.lock().await;
        match guard.as_ref() {
            Some(src) => src.load_scores().await.unwrap_or_else(|e| {
                warn!("[cold-quota] score source unavailable ({e}) — proceeding with empty scores");
                QualificationScores::default()
            }),
            None => {
                tracing::debug!(
                    "[cold-quota] score source not configured — proceeding with empty scores"
                );
                QualificationScores::default()
            }
        }
    };
    let remover = FsColdArchiveRemover::new(archive_root.to_path_buf());
    run_cold_quota_pass(
        registry,
        archive_root,
        probe,
        &scores,
        keep,
        &remover,
        cfg,
        disk_op_lock,
    )
    .await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::eviction::new_disk_op_lock;
    use crate::models::registry::ModelRegistry;
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tempfile::tempdir;

    // ── Pure config ────────────────────────────────────────────────────────────

    #[test]
    fn resolve_quota_absolute_takes_precedence() {
        let cfg = ColdQuotaConfig {
            quota_gb: Some(100),
            quota_percent: 80,
            ..ColdQuotaConfig::default()
        };
        // 100 GiB absolute, ignoring the percent-of-total.
        assert_eq!(resolve_quota_bytes(&cfg, 3_000 * GIB), 100 * GIB);
    }

    #[test]
    fn resolve_quota_percent_of_total() {
        let cfg = ColdQuotaConfig {
            quota_gb: None,
            quota_percent: 80,
            ..ColdQuotaConfig::default()
        };
        assert_eq!(resolve_quota_bytes(&cfg, 1000), 800);
        assert!(is_over_quota(801, 800));
        assert!(!is_over_quota(800, 800));
    }

    fn cand(
        name: &str,
        q: Option<f64>,
        fit: Option<f64>,
        last: Option<i64>,
        size: u64,
        first_seen: Option<i64>,
        freed: u64,
    ) -> ColdCandidate {
        ColdCandidate {
            name: name.to_string(),
            assistant_avg_value: q,
            fit_score: fit,
            last_requested: last,
            size_bytes: size,
            first_seen,
            est_freed_bytes: freed,
        }
    }

    const NOW: i64 = 1_700_000_000;
    const DAY: i64 = 86_400;

    /// Config for the PURE ranking tests: the sanity floors use realistic GiB
    /// magnitudes, but these fixtures use tiny byte-scale quotas to exercise
    /// ranking/greedy logic — so disable the min-sane floor (0) to isolate the
    /// behavior under test. The dedicated hard-floor tests set their own cfg.
    fn rank_cfg() -> ColdQuotaConfig {
        ColdQuotaConfig {
            min_sane_gb: 0,
            ..ColdQuotaConfig::default()
        }
    }

    #[test]
    fn dry_run_armed_only_by_literal_zero() {
        // The #1 arming footgun contract: ONLY the literal "0" arms deletion;
        // every other value — including "false"/"no"/"off" and typos — keeps
        // dry-run ON (fail-safe), and unset defaults to dry-run ON.
        assert!(dry_run_from_raw(None), "unset ⇒ dry-run ON");
        assert!(!dry_run_from_raw(Some("0".into())), "\"0\" arms");
        assert!(
            !dry_run_from_raw(Some("  0  ".into())),
            "trimmed \"0\" arms"
        );
        assert!(dry_run_from_raw(Some("1".into())), "\"1\" ⇒ dry-run ON");
        assert!(
            dry_run_from_raw(Some("false".into())),
            "\"false\" must NOT arm (footgun guard)"
        );
        assert!(dry_run_from_raw(Some("off".into())), "\"off\" must NOT arm");
        assert!(dry_run_from_raw(Some("no".into())), "\"no\" must NOT arm");
        assert!(
            dry_run_from_raw(Some("00".into())),
            "\"00\" is not \"0\" ⇒ ON"
        );
    }

    // ── Ranking ──────────────────────────────────────────────────────────────

    #[test]
    fn ranks_lowest_measured_first() {
        // Two swept models; lowest assistant_avg_value pruned first. Quota needs
        // exactly one prune (used 1000, quota 500, each frees 600).
        let cands = vec![
            cand(
                "good",
                Some(100.0),
                None,
                Some(5),
                10,
                Some(NOW - 100 * DAY),
                600,
            ),
            cand(
                "bad",
                Some(10.0),
                None,
                Some(5),
                10,
                Some(NOW - 100 * DAY),
                600,
            ),
        ];
        let plan = plan_prune(&cands, &HashSet::<String>::new(), &rank_cfg(), NOW, 100, 1000, 500);
        assert!(plan.aborted.is_none());
        assert_eq!(plan.items.len(), 1);
        assert_eq!(
            plan.items[0].name, "bad",
            "lowest measured value pruned first"
        );
    }

    #[test]
    fn measured_pruned_before_fallback() {
        // A measured model (even a decent one) is pruned before an un-swept
        // fallback one. Need two prunes.
        let cands = vec![
            cand(
                "measured",
                Some(50.0),
                None,
                Some(1),
                10,
                Some(NOW - 100 * DAY),
                400,
            ),
            cand(
                "unswept",
                None,
                Some(0.01),
                Some(1),
                10,
                Some(NOW - 100 * DAY),
                400,
            ),
        ];
        let plan = plan_prune(&cands, &HashSet::<String>::new(), &rank_cfg(), NOW, 100, 1000, 200);
        assert!(plan.aborted.is_none());
        assert_eq!(plan.items[0].name, "measured");
        assert_eq!(plan.items[1].name, "unswept");
    }

    #[test]
    fn fallback_uses_lowest_fit_past_grace() {
        // Two un-swept models past grace; lowest fit_score pruned first.
        let cands = vec![
            cand(
                "hi_fit",
                None,
                Some(0.9),
                Some(1),
                10,
                Some(NOW - 100 * DAY),
                600,
            ),
            cand(
                "lo_fit",
                None,
                Some(0.1),
                Some(1),
                10,
                Some(NOW - 100 * DAY),
                600,
            ),
        ];
        let plan = plan_prune(&cands, &HashSet::<String>::new(), &rank_cfg(), NOW, 100, 1000, 500);
        assert_eq!(plan.items.len(), 1);
        assert_eq!(plan.items[0].name, "lo_fit");
    }

    #[test]
    fn lru_then_size_tiebreaks() {
        // Equal measured value → oldest last_requested first; equal last_requested
        // → largest size first.
        let cands = vec![
            cand(
                "newer_small",
                Some(10.0),
                None,
                Some(2000),
                10,
                Some(NOW - 100 * DAY),
                100,
            ),
            cand(
                "older",
                Some(10.0),
                None,
                Some(1000),
                10,
                Some(NOW - 100 * DAY),
                100,
            ),
            cand(
                "newer_big",
                Some(10.0),
                None,
                Some(2000),
                99,
                Some(NOW - 100 * DAY),
                100,
            ),
        ];
        let plan = plan_prune(&cands, &HashSet::<String>::new(), &rank_cfg(), NOW, 100, 1000, 999);
        // used 1000, quota 999, each frees 100 → one prune; oldest wins the LRU.
        assert_eq!(plan.items[0].name, "older");
    }

    // ── Exclusions ─────────────────────────────────────────────────────────────

    #[test]
    fn keep_set_excludes_current_target() {
        let mut keep = HashSet::new();
        keep.insert("granite4.1:30b".to_string()); // protected/current lumina target
        let cands = vec![
            cand(
                "granite4.1:30b",
                Some(1.0),
                None,
                Some(1),
                10,
                Some(NOW - 100 * DAY),
                600,
            ),
            cand(
                "prunable",
                Some(5.0),
                None,
                Some(1),
                10,
                Some(NOW - 100 * DAY),
                600,
            ),
        ];
        let plan = plan_prune(&cands, &keep, &rank_cfg(), NOW, 100, 1000, 500);
        assert_eq!(plan.items.len(), 1);
        assert_eq!(
            plan.items[0].name, "prunable",
            "keep-set member never pruned even at lowest score"
        );
    }

    #[test]
    fn grace_exempts_recent_unswept_model() {
        // Un-swept model first-seen 3 days ago, grace 14 → exempt (not a candidate).
        let cands = vec![cand(
            "fresh",
            None,
            Some(0.01),
            Some(NOW - 3 * DAY),
            10,
            Some(NOW - 3 * DAY),
            600,
        )];
        let plan = plan_prune(&cands, &HashSet::<String>::new(), &rank_cfg(), NOW, 100, 1000, 100);
        assert!(
            plan.items.is_empty(),
            "within-grace un-swept model is exempt"
        );
    }

    #[test]
    fn unswept_past_grace_is_prunable_via_fit() {
        let cands = vec![cand(
            "old_unswept",
            None,
            Some(0.01),
            Some(NOW - 30 * DAY),
            10,
            Some(NOW - 30 * DAY),
            600,
        )];
        let plan = plan_prune(&cands, &HashSet::<String>::new(), &rank_cfg(), NOW, 100, 1000, 100);
        assert_eq!(plan.items.len(), 1);
        assert_eq!(plan.items[0].name, "old_unswept");
    }

    #[test]
    fn measured_only_mode_never_prunes_unswept() {
        let cfg = ColdQuotaConfig {
            fallback_fit: false,
            ..ColdQuotaConfig::default()
        };
        let cands = vec![cand(
            "old_unswept",
            None,
            Some(0.01),
            Some(NOW - 30 * DAY),
            10,
            Some(NOW - 30 * DAY),
            600,
        )];
        let plan = plan_prune(&cands, &HashSet::<String>::new(), &cfg, NOW, 100, 1000, 100);
        assert!(
            plan.items.is_empty(),
            "measured-only mode leaves un-swept models alone"
        );
    }

    #[test]
    fn min_keep_floor_caps_prunes() {
        // 3 cold total, min_keep 2 → at most 1 prunable even though quota wants more.
        let cfg = ColdQuotaConfig {
            min_keep: 2,
            min_sane_gb: 0,
            ..ColdQuotaConfig::default()
        };
        let cands = vec![
            cand(
                "a",
                Some(1.0),
                None,
                Some(1),
                10,
                Some(NOW - 100 * DAY),
                100,
            ),
            cand(
                "b",
                Some(2.0),
                None,
                Some(1),
                10,
                Some(NOW - 100 * DAY),
                100,
            ),
            cand(
                "c",
                Some(3.0),
                None,
                Some(1),
                10,
                Some(NOW - 100 * DAY),
                100,
            ),
        ];
        let plan = plan_prune(&cands, &HashSet::<String>::new(), &cfg, NOW, 3, 1000, 1);
        assert!(plan.aborted.is_none());
        assert_eq!(
            plan.items.len(),
            1,
            "min-keep floor caps prunes to cold_total - min_keep"
        );
    }

    #[test]
    fn under_quota_is_noop() {
        let cands = vec![cand(
            "a",
            Some(1.0),
            None,
            Some(1),
            10,
            Some(NOW - 100 * DAY),
            100,
        )];
        // used 400 <= quota 500 shouldn't even be called, but plan is safe: greedy
        // stops immediately.
        let plan = plan_prune(&cands, &HashSet::<String>::new(), &rank_cfg(), NOW, 100, 400, 500);
        assert!(plan.items.is_empty());
    }

    // ── Hard-misconfig floors ────────────────────────────────────────────────

    #[test]
    fn hard_floor_aborts_on_absurd_absolute_quota() {
        // QUOTA_GB=1 → 1 GiB < min_sane 10 GiB → abort, delete nothing.
        let cfg = ColdQuotaConfig {
            quota_gb: Some(1),
            min_sane_gb: 10,
            ..ColdQuotaConfig::default()
        };
        let cands = vec![cand(
            "a",
            Some(1.0),
            None,
            Some(1),
            10,
            Some(NOW - 100 * DAY),
            100,
        )];
        let quota = resolve_quota_bytes(&cfg, 3000 * GIB);
        let plan = plan_prune(&cands, &HashSet::<String>::new(), &cfg, NOW, 100, 3000 * GIB, quota);
        assert!(plan.aborted.is_some(), "absurd absolute quota must abort");
    }

    #[test]
    fn hard_floor_aborts_when_over_pass_fraction() {
        // 10 cold models, 25% cap → 2; a plan needing 6 aborts (delete nothing).
        // min_keep 0 so the greedy plan isn't capped below the fraction guard;
        // min_sane_gb 0 so the fraction guard fires (not the min-sane floor) at
        // this test's tiny byte-scale quota.
        let cfg = ColdQuotaConfig {
            min_keep: 0,
            min_sane_gb: 0,
            ..ColdQuotaConfig::default()
        };
        let mut cands = Vec::new();
        for i in 0..10 {
            cands.push(cand(
                &format!("m{i}"),
                Some(i as f64),
                None,
                Some(1),
                10,
                Some(NOW - 100 * DAY),
                100,
            ));
        }
        // used 1000, quota 400, each frees 100 → wants 6 prunes → 6 > cap(2) → abort.
        let plan = plan_prune(&cands, &HashSet::<String>::new(), &cfg, NOW, 10, 1000, 400);
        assert!(plan.aborted.is_some(), ">25% single-pass prune must abort");
        assert!(plan.aborted.as_ref().unwrap().contains("single pass"));
    }

    // ── I/O orchestration (real fs archive) ────────────────────────────────────

    /// Write a manifest + referenced blobs under `<root>/manifests/.../<tag>`,
    /// returning the model name. Mirrors eviction.rs's test helper.
    fn make_archive_model(root: &Path, model: &str, tag: &str, blob_sizes: &[u64]) -> String {
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

    /// A manifest referencing a SHARED blob digest (so two archived models share
    /// one physical blob file).
    fn make_archive_model_sharing(
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

    /// Probe reporting a fixed total/free on the archive mount.
    struct FixedProbe {
        total: u64,
        free: Arc<std::sync::atomic::AtomicU64>,
    }
    impl DiskSpaceProbe for FixedProbe {
        fn available_bytes(&self, _: &Path) -> Option<u64> {
            Some(self.free.load(Ordering::SeqCst))
        }
        fn total_bytes(&self, _: &Path) -> Option<u64> {
            Some(self.total)
        }
    }

    /// Remover that records calls but never touches the fs (proves dry-run).
    struct SpyRemover(Arc<AtomicUsize>);
    #[async_trait]
    impl ColdArchiveRemover for SpyRemover {
        async fn remove(&self, _model: &str) -> Result<(), String> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    fn armed_cfg() -> ColdQuotaConfig {
        ColdQuotaConfig {
            dry_run: false,
            grace_days: 0, // exercise pruning without grace interference
            min_keep: 0,
            min_sane_gb: 0, // tiny test archive
            ..ColdQuotaConfig::default()
        }
    }

    #[tokio::test]
    async fn orchestrator_over_quota_prunes_least_qualified() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        let archive = base.join("archive");
        make_archive_model(&archive, "lowq", "1", &[100]);
        make_archive_model(&archive, "hiq", "1", &[100]);
        let mut reg = reg_with(base, vec![]);
        reg.reconcile();
        // Both cold (archive-only). Set deterministic timestamps well past grace 0.
        reg.set_last_requested_for_test("lowq:1", 1000);
        reg.set_last_requested_for_test("hiq:1", 1000);
        let registry = Arc::new(Mutex::new(reg));

        let mut scores = QualificationScores::default();
        scores.assistant_avg_value.insert("lowq:1".into(), 5.0);
        scores.assistant_avg_value.insert("hiq:1".into(), 500.0);

        // total 1000, free 100 → used 900; quota 80% = 800 → over by 100. One prune.
        let free = Arc::new(std::sync::atomic::AtomicU64::new(100));
        let probe = FixedProbe {
            total: 1000,
            free: free.clone(),
        };
        let remover = FsColdArchiveRemover::new(archive.clone());
        let lock = new_disk_op_lock();

        run_cold_quota_pass_at(
            &registry,
            &archive,
            &probe,
            &scores,
            &HashSet::<String>::new(),
            &remover,
            &armed_cfg(),
            &lock,
            NOW,
        )
        .await;

        // lowq pruned from archive + registry; hiq kept.
        let reg = registry.lock().await;
        assert!(
            reg.get("lowq:1").is_none(),
            "least-qualified record removed"
        );
        assert!(reg.get("hiq:1").is_some(), "higher-qualified kept");
        assert!(!archive
            .join("manifests/registry.ollama.ai/library/lowq/1")
            .is_file());
        assert!(archive
            .join("manifests/registry.ollama.ai/library/hiq/1")
            .is_file());
    }

    #[tokio::test]
    async fn orchestrator_dry_run_deletes_nothing() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        let archive = base.join("archive");
        make_archive_model(&archive, "lowq", "1", &[100]);
        let mut reg = reg_with(base, vec![]);
        reg.reconcile();
        reg.set_last_requested_for_test("lowq:1", 1000);
        let registry = Arc::new(Mutex::new(reg));
        let mut scores = QualificationScores::default();
        scores.assistant_avg_value.insert("lowq:1".into(), 5.0);

        let free = Arc::new(std::sync::atomic::AtomicU64::new(0)); // 100% used → over
        let probe = FixedProbe { total: 1000, free };
        let spy = SpyRemover(Arc::new(AtomicUsize::new(0)));
        let calls = match &spy {
            SpyRemover(c) => c.clone(),
        };
        let cfg = ColdQuotaConfig {
            dry_run: true,
            grace_days: 0,
            min_keep: 0,
            min_sane_gb: 0,
            ..ColdQuotaConfig::default()
        };
        let lock = new_disk_op_lock();

        run_cold_quota_pass_at(
            &registry,
            &archive,
            &probe,
            &scores,
            &HashSet::<String>::new(),
            &spy,
            &cfg,
            &lock,
            NOW,
        )
        .await;

        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "dry-run must not call the remover"
        );
        assert!(
            registry.lock().await.get("lowq:1").is_some(),
            "dry-run keeps the record"
        );
        assert!(
            archive
                .join("manifests/registry.ollama.ai/library/lowq/1")
                .is_file(),
            "dry-run keeps files"
        );
    }

    #[tokio::test]
    async fn orchestrator_skips_when_archive_absent() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        let archive = base.join("archive"); // never created → unmounted NFS
        let registry = Arc::new(Mutex::new(reg_with(base, vec![])));
        let free = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let probe = FixedProbe { total: 1000, free };
        let spy = SpyRemover(Arc::new(AtomicUsize::new(0)));
        let calls = match &spy {
            SpyRemover(c) => c.clone(),
        };
        let lock = new_disk_op_lock();

        run_cold_quota_pass_at(
            &registry,
            &archive,
            &probe,
            &QualificationScores::default(),
            &HashSet::<String>::new(),
            &spy,
            &armed_cfg(),
            &lock,
            NOW,
        )
        .await;
        assert_eq!(calls.load(Ordering::SeqCst), 0, "no archive → no pruning");
    }

    #[tokio::test]
    async fn orchestrator_under_quota_is_noop() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        let archive = base.join("archive");
        make_archive_model(&archive, "lowq", "1", &[100]);
        let mut reg = reg_with(base, vec![]);
        reg.reconcile();
        let registry = Arc::new(Mutex::new(reg));
        let free = Arc::new(std::sync::atomic::AtomicU64::new(900)); // used 100, quota 800
        let probe = FixedProbe { total: 1000, free };
        let spy = SpyRemover(Arc::new(AtomicUsize::new(0)));
        let calls = match &spy {
            SpyRemover(c) => c.clone(),
        };
        let lock = new_disk_op_lock();

        run_cold_quota_pass_at(
            &registry,
            &archive,
            &probe,
            &QualificationScores::default(),
            &HashSet::<String>::new(),
            &spy,
            &armed_cfg(),
            &lock,
            NOW,
        )
        .await;
        assert_eq!(calls.load(Ordering::SeqCst), 0, "under quota → no pruning");
    }

    #[tokio::test]
    async fn exclusive_bytes_is_gc_aware_for_shared_blob() {
        // Two archived models share a big blob; the shared blob contributes 0 to
        // each model's exclusive size (frees nothing while still referenced).
        let tmp = tempdir().unwrap();
        let archive = tmp.path().join("archive");
        let shared = "sha256:sharedblob";
        make_archive_model_sharing(&archive, "alpha", "1", shared, 1000);
        make_archive_model_sharing(&archive, "beta", "1", shared, 1000);
        // alpha's exclusive size = only its own cfg blob (3 bytes), NOT the shared 1000.
        let ex = exclusive_archive_bytes(&archive, "alpha:1");
        assert_eq!(
            ex, 3,
            "shared blob excluded from exclusive freed-bytes estimate"
        );
    }

    #[tokio::test]
    async fn shared_blob_survives_prune_of_one_referencer() {
        // Pruning alpha must NOT delete a blob beta still references (GC-aware).
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        let archive = base.join("archive");
        let shared = "sha256:sharedblob";
        make_archive_model_sharing(&archive, "alpha", "1", shared, 1000);
        make_archive_model_sharing(&archive, "beta", "1", shared, 1000);
        let mut reg = reg_with(base, vec![]);
        reg.reconcile();
        reg.set_last_requested_for_test("alpha:1", 1000);
        reg.set_last_requested_for_test("beta:1", 2000);
        let registry = Arc::new(Mutex::new(reg));
        let mut scores = QualificationScores::default();
        scores.assistant_avg_value.insert("alpha:1".into(), 1.0); // lowest → pruned first
        scores.assistant_avg_value.insert("beta:1".into(), 500.0);

        // Over quota by a little so exactly one prune happens.
        let free = Arc::new(std::sync::atomic::AtomicU64::new(150));
        let probe = FixedProbe { total: 1000, free };
        let remover = FsColdArchiveRemover::new(archive.clone());
        let cfg = ColdQuotaConfig {
            dry_run: false,
            grace_days: 0,
            min_keep: 1,
            min_sane_gb: 0,
            ..ColdQuotaConfig::default()
        };
        let lock = new_disk_op_lock();

        run_cold_quota_pass_at(
            &registry,
            &archive,
            &probe,
            &scores,
            &HashSet::<String>::new(),
            &remover,
            &cfg,
            &lock,
            NOW,
        )
        .await;

        assert!(
            registry.lock().await.get("alpha:1").is_none(),
            "alpha pruned"
        );
        assert!(
            archive.join("blobs/sha256-sharedblob").is_file(),
            "shared blob still referenced by beta must survive"
        );
    }

    #[tokio::test]
    async fn protected_cold_model_never_pruned() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        let archive = base.join("archive");
        make_archive_model(&archive, "keepme", "1", &[100]);
        let mut reg = reg_with(base, vec!["keepme:1".to_string()]);
        reg.reconcile();
        reg.set_last_requested_for_test("keepme:1", 1000);
        assert!(reg.is_protected("keepme:1"));
        let registry = Arc::new(Mutex::new(reg));
        let mut scores = QualificationScores::default();
        scores.assistant_avg_value.insert("keepme:1".into(), 0.0); // worst score

        let free = Arc::new(std::sync::atomic::AtomicU64::new(0)); // maximally over quota
        let probe = FixedProbe { total: 1000, free };
        let remover = FsColdArchiveRemover::new(archive.clone());
        let lock = new_disk_op_lock();

        run_cold_quota_pass_at(
            &registry,
            &archive,
            &probe,
            &scores,
            &HashSet::<String>::new(),
            &remover,
            &armed_cfg(),
            &lock,
            NOW,
        )
        .await;

        assert!(
            registry.lock().await.get("keepme:1").is_some(),
            "protected model never pruned"
        );
        assert!(archive
            .join("manifests/registry.ollama.ai/library/keepme/1")
            .is_file());
    }

    #[tokio::test]
    async fn toctou_reprotected_item_is_skipped_at_delete_time() {
        // A model planned for pruning that becomes PROTECTED between plan and the
        // per-item delete (simulating a concurrent protect / promotion) must be
        // re-validated under the lock and SKIPPED — its archive + record survive.
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        let archive = base.join("archive");
        make_archive_model(&archive, "lowq", "1", &[100]);
        make_archive_model(&archive, "midq", "1", &[100]);
        make_archive_model(&archive, "hiq", "1", &[100]);
        let mut reg = reg_with(base, vec![]);
        reg.reconcile();
        reg.set_last_requested_for_test("lowq:1", 1000);
        reg.set_last_requested_for_test("midq:1", 1000);
        reg.set_last_requested_for_test("hiq:1", 1000);
        let registry = Arc::new(Mutex::new(reg));
        let mut scores = QualificationScores::default();
        scores.assistant_avg_value.insert("lowq:1".into(), 1.0);
        scores.assistant_avg_value.insert("midq:1".into(), 2.0);
        scores.assistant_avg_value.insert("hiq:1".into(), 500.0);

        // Remover that, when it deletes the FIRST item (lowq), concurrently marks
        // midq protected — exactly the TOCTOU the re-validation must catch.
        struct ReprotectRemover {
            archive_root: PathBuf,
            registry: Arc<Mutex<ModelRegistry>>,
        }
        #[async_trait]
        impl ColdArchiveRemover for ReprotectRemover {
            async fn remove(&self, model: &str) -> Result<(), String> {
                if model == "lowq:1" {
                    let mut reg = self.registry.lock().await;
                    reg.set_protected("midq:1", true);
                }
                fs_remove_model(&self.archive_root, model)
            }
        }
        let remover = ReprotectRemover {
            archive_root: archive.clone(),
            registry: registry.clone(),
        };
        // Maximally over quota so the plan wants to prune several.
        let free = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let probe = FixedProbe { total: 1000, free };
        let lock = new_disk_op_lock();

        run_cold_quota_pass_at(
            &registry,
            &archive,
            &probe,
            &scores,
            &HashSet::<String>::new(),
            &remover,
            &armed_cfg(),
            &lock,
            NOW,
        )
        .await;

        let reg = registry.lock().await;
        assert!(reg.get("lowq:1").is_none(), "lowq pruned");
        assert!(
            reg.get("midq:1").is_some(),
            "midq re-protected mid-pass must be skipped, not pruned"
        );
        assert!(
            archive
                .join("manifests/registry.ollama.ai/library/midq/1")
                .is_file(),
            "midq archive must survive the TOCTOU re-validation"
        );
    }

    #[tokio::test]
    async fn probe_failure_mid_pass_stops_deletion() {
        // If the archive df probe fails MID-PASS, the armed loop must STOP
        // (fail-safe) rather than keep deleting blind.
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        let archive = base.join("archive");
        make_archive_model(&archive, "lowq", "1", &[100]);
        make_archive_model(&archive, "midq", "1", &[100]);
        make_archive_model(&archive, "hiq", "1", &[100]);
        let mut reg = reg_with(base, vec![]);
        reg.reconcile();
        reg.set_last_requested_for_test("lowq:1", 1000);
        reg.set_last_requested_for_test("midq:1", 1000);
        reg.set_last_requested_for_test("hiq:1", 1000);
        let registry = Arc::new(Mutex::new(reg));
        let mut scores = QualificationScores::default();
        scores.assistant_avg_value.insert("lowq:1".into(), 1.0);
        scores.assistant_avg_value.insert("midq:1".into(), 2.0);
        scores.assistant_avg_value.insert("hiq:1".into(), 3.0);

        // `available_bytes` returns Some for the first two status evaluations (top
        // probe + first loop item) then None → the 2nd loop item hits a probe
        // failure and the loop stops. Exactly ONE model is pruned.
        struct FlakyProbe(Arc<AtomicUsize>);
        impl DiskSpaceProbe for FlakyProbe {
            fn total_bytes(&self, _: &Path) -> Option<u64> {
                Some(1000)
            }
            fn available_bytes(&self, _: &Path) -> Option<u64> {
                let n = self.0.fetch_add(1, Ordering::SeqCst);
                if n < 2 {
                    Some(0) // used 1000 > quota 800 → over
                } else {
                    None // probe failure
                }
            }
        }
        let probe = FlakyProbe(Arc::new(AtomicUsize::new(0)));
        let remover = FsColdArchiveRemover::new(archive.clone());
        let lock = new_disk_op_lock();

        run_cold_quota_pass_at(
            &registry,
            &archive,
            &probe,
            &scores,
            &HashSet::<String>::new(),
            &remover,
            &armed_cfg(),
            &lock,
            NOW,
        )
        .await;

        let reg = registry.lock().await;
        let remaining = ["lowq:1", "midq:1", "hiq:1"]
            .iter()
            .filter(|m| reg.get(m).is_some())
            .count();
        assert_eq!(
            remaining, 2,
            "probe failure mid-pass must stop after exactly one prune (fail-safe)"
        );
    }

    // ── CQH-01 (F3): explicit MODEL_ARCHIVE_QUOTA_GB=0 aborts ──────────────────

    #[test]
    fn parse_quota_gb_classifies_explicit_zero_vs_unset() {
        // Unset ⇒ (None, false) — use the percent quota, unchanged.
        assert_eq!(parse_quota_gb_raw(None), (None, false));
        // Explicit, well-formed 0 (any whitespace) ⇒ abort signal (None, true).
        assert_eq!(parse_quota_gb_raw(Some("0")), (None, true));
        assert_eq!(parse_quota_gb_raw(Some("  0 ")), (None, true));
        // Positive ⇒ absolute quota.
        assert_eq!(parse_quota_gb_raw(Some("100")), (Some(100), false));
        // Garbage ⇒ percent path (unchanged), never the abort signal.
        assert_eq!(parse_quota_gb_raw(Some("abc")), (None, false));
        assert_eq!(parse_quota_gb_raw(Some("")), (None, false));
    }

    #[test]
    fn plan_prune_aborts_on_explicit_zero_quota() {
        let cfg = ColdQuotaConfig {
            explicit_zero_quota: true,
            min_sane_gb: 0,
            ..ColdQuotaConfig::default()
        };
        let cands = vec![cand(
            "a",
            Some(1.0),
            None,
            Some(1),
            10,
            Some(NOW - 100 * DAY),
            100,
        )];
        // Even maximally over quota, an explicit =0 aborts (prune nothing).
        let plan = plan_prune(&cands, &HashSet::<String>::new(), &cfg, NOW, 100, 1000, 1);
        assert!(plan.aborted.is_some(), "explicit QUOTA_GB=0 must abort");
        assert!(plan.items.is_empty());
    }

    #[tokio::test]
    async fn orchestrator_aborts_on_explicit_zero_quota() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        let archive = base.join("archive");
        make_archive_model(&archive, "lowq", "1", &[100]);
        let mut reg = reg_with(base, vec![]);
        reg.reconcile();
        reg.set_last_requested_for_test("lowq:1", 1000);
        let registry = Arc::new(Mutex::new(reg));
        let mut scores = QualificationScores::default();
        scores.assistant_avg_value.insert("lowq:1".into(), 5.0);

        let free = Arc::new(std::sync::atomic::AtomicU64::new(0)); // 100% used
        let probe = FixedProbe { total: 1000, free };
        let spy = SpyRemover(Arc::new(AtomicUsize::new(0)));
        let calls = match &spy {
            SpyRemover(c) => c.clone(),
        };
        // ARMED, but explicit zero quota → must prune nothing.
        let cfg = ColdQuotaConfig {
            explicit_zero_quota: true,
            ..armed_cfg()
        };
        let lock = new_disk_op_lock();

        run_cold_quota_pass_at(
            &registry,
            &archive,
            &probe,
            &scores,
            &HashSet::<String>::new(),
            &spy,
            &cfg,
            &lock,
            NOW,
        )
        .await;
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "explicit QUOTA_GB=0 must abort the pass before any delete"
        );
        assert!(registry.lock().await.get("lowq:1").is_some());
    }

    // ── CQH-01 (F5): fallback query uses the column that actually exists ────────

    #[test]
    fn fallback_query_targets_discovery_score_not_fit_score() {
        // The live intake DB stores `discovery_score` (DISC-01 DDL); `fit_score` is
        // a derived blend that is never persisted. The query must reference the
        // real column so the fallback bucket populates without operator DDL.
        assert!(
            FALLBACK_SCORE_SQL.contains("discovery_score"),
            "fallback query must select the stored discovery_score column"
        );
        assert!(
            !FALLBACK_SCORE_SQL.contains("fit_score"),
            "fallback query must NOT reference the non-existent fit_score column"
        );
        assert!(FALLBACK_SCORE_SQL.contains("model_discovery_candidate"));
        assert!(FALLBACK_SCORE_SQL.contains("model_name"));
    }

    #[test]
    fn fallback_bucket_populates_and_ranks_from_practical_score() {
        // With the fallback score present (as the discovery_score-sourced map would
        // provide), an un-swept-past-grace model becomes prunable and ranks by that
        // score — proving the fallback bucket is functional (F5's end state).
        let mut scores = QualificationScores::default();
        scores.fit_score.insert("lo".into(), 0.1);
        scores.fit_score.insert("hi".into(), 0.9);
        let cands = vec![
            cand(
                "hi",
                None,
                scores.fit_score.get("hi").copied(),
                Some(1),
                10,
                Some(NOW - 100 * DAY),
                600,
            ),
            cand(
                "lo",
                None,
                scores.fit_score.get("lo").copied(),
                Some(1),
                10,
                Some(NOW - 100 * DAY),
                600,
            ),
        ];
        let plan = plan_prune(&cands, &HashSet::<String>::new(), &rank_cfg(), NOW, 100, 1000, 500);
        assert_eq!(plan.items.len(), 1);
        assert_eq!(
            plan.items[0].name, "lo",
            "lowest practical (discovery_score-sourced) fallback pruned first"
        );
        assert!(
            plan.items[0].reason.contains("discovery_score"),
            "reason names the practical column"
        );
    }

    // ── CQH-01 (F2): a LIVE keep repoint mid-pass protects the new target ───────

    /// A keep set backed by a shared, MUTABLE cell — models can enter it mid-pass.
    struct MutableKeepSet(Arc<std::sync::Mutex<HashSet<String>>>);
    impl LiveKeepSet for MutableKeepSet {
        fn current(&self) -> HashSet<String> {
            self.0.lock().unwrap().clone()
        }
    }

    #[tokio::test]
    async fn live_keep_repoint_mid_pass_protects_new_target() {
        // F1+F2: a model that ENTERS the live keep set between plan and its delete
        // (e.g. the lumina alias updater repoints lumina onto it) must be skipped by
        // the delete-time re-query — its archive + record survive.
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        let archive = base.join("archive");
        make_archive_model(&archive, "lowq", "1", &[100]);
        make_archive_model(&archive, "midq", "1", &[100]);
        make_archive_model(&archive, "hiq", "1", &[100]);
        let mut reg = reg_with(base, vec![]);
        reg.reconcile();
        reg.set_last_requested_for_test("lowq:1", 1000);
        reg.set_last_requested_for_test("midq:1", 1000);
        reg.set_last_requested_for_test("hiq:1", 1000);
        let registry = Arc::new(Mutex::new(reg));
        let mut scores = QualificationScores::default();
        scores.assistant_avg_value.insert("lowq:1".into(), 1.0);
        scores.assistant_avg_value.insert("midq:1".into(), 2.0);
        scores.assistant_avg_value.insert("hiq:1".into(), 500.0);

        let keep_cell = Arc::new(std::sync::Mutex::new(HashSet::<String>::new()));
        let keep = MutableKeepSet(keep_cell.clone());

        // Remover that, on deleting lowq (first), repoints the live keep set onto
        // midq — exactly the mid-pass alias repoint F2 must honor at delete time.
        struct RepointRemover {
            archive_root: PathBuf,
            keep_cell: Arc<std::sync::Mutex<HashSet<String>>>,
        }
        #[async_trait]
        impl ColdArchiveRemover for RepointRemover {
            async fn remove(&self, model: &str) -> Result<(), String> {
                if model == "lowq:1" {
                    self.keep_cell.lock().unwrap().insert("midq:1".to_string());
                }
                fs_remove_model(&self.archive_root, model)
            }
        }
        let remover = RepointRemover {
            archive_root: archive.clone(),
            keep_cell: keep_cell.clone(),
        };
        let free = Arc::new(std::sync::atomic::AtomicU64::new(0)); // maximally over
        let probe = FixedProbe { total: 1000, free };
        let lock = new_disk_op_lock();

        run_cold_quota_pass_at(
            &registry,
            &archive,
            &probe,
            &scores,
            &keep,
            &remover,
            &armed_cfg(),
            &lock,
            NOW,
        )
        .await;

        let reg = registry.lock().await;
        assert!(reg.get("lowq:1").is_none(), "lowq pruned");
        assert!(
            reg.get("midq:1").is_some(),
            "midq entered the LIVE keep set mid-pass and must be skipped"
        );
        assert!(archive
            .join("manifests/registry.ollama.ai/library/midq/1")
            .is_file());
    }

    // ── CQH-01 (F4): a VRAM keep-resident model in Cold is exempt ──────────────

    #[tokio::test]
    async fn keep_resident_model_in_cold_is_exempt() {
        use crate::routing::resident_set::{ResidentSetConfig, Role};
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        let archive = base.join("archive");
        make_archive_model(&archive, "residentmodel", "1", &[100]);
        make_archive_model(&archive, "prunable", "1", &[100]);
        let mut reg = reg_with(base, vec![]);
        reg.reconcile();
        reg.set_last_requested_for_test("residentmodel:1", 1000);
        reg.set_last_requested_for_test("prunable:1", 1000);
        let registry = Arc::new(Mutex::new(reg));
        let mut scores = QualificationScores::default();
        // Resident model has the WORST score — only its keep-resident status saves it.
        scores.assistant_avg_value.insert("residentmodel:1".into(), 0.0);
        scores.assistant_avg_value.insert("prunable:1".into(), 500.0);

        // Static alias "lumina-embed" → residentmodel:1; the resident Embedding role
        // resolves through it, so resident_exempt_models yields {residentmodel:1}.
        let mut statics = HashMap::new();
        statics.insert("lumina-embed".to_string(), "residentmodel:1".to_string());
        let dynamic = LuminaAliasStore::empty();
        let resident_cfg = ResidentSetConfig {
            enabled: true,
            aliases: vec![(Role::Embedding, "lumina-embed".to_string())],
            ..ResidentSetConfig::default()
        };
        let keep = LuminaResidentKeepSet::new(dynamic, statics, resident_cfg);

        // Confirm the keep set actually contains the resident model.
        assert!(keep.current().contains("residentmodel:1"));

        let free = Arc::new(std::sync::atomic::AtomicU64::new(150)); // over by a little
        let probe = FixedProbe { total: 1000, free };
        let remover = FsColdArchiveRemover::new(archive.clone());
        let cfg = ColdQuotaConfig {
            dry_run: false,
            grace_days: 0,
            min_keep: 1,
            min_sane_gb: 0,
            ..ColdQuotaConfig::default()
        };
        let lock = new_disk_op_lock();

        run_cold_quota_pass_at(
            &registry,
            &archive,
            &probe,
            &scores,
            &keep,
            &remover,
            &cfg,
            &lock,
            NOW,
        )
        .await;

        let reg = registry.lock().await;
        assert!(
            reg.get("residentmodel:1").is_some(),
            "keep-resident model must never be pruned, even at the worst score"
        );
        assert!(
            reg.get("prunable:1").is_none(),
            "the non-resident candidate is the one pruned"
        );
    }

    // ── CQH-01 (F1/F2 review round 3): publish lock spans the fs delete ─────────

    #[tokio::test]
    async fn publish_lock_serializes_set_against_owned_prune_guard() {
        // The core of the fix: `LuminaAliasStore::set` and the pruner's owned
        // publish guard are MUTUALLY EXCLUSIVE. While the pruner holds the owned
        // guard (across its read → record-drop → fs delete), a repoint cannot
        // publish; it applies only once the guard is released.
        use std::sync::atomic::AtomicBool;
        let store = LuminaAliasStore::from_static(&{
            let mut m = HashMap::new();
            m.insert("lumina-fast".to_string(), "old:1".to_string());
            m
        });
        let lockarc = store.publish_lock_arc();
        let guard = lockarc.clone().lock_owned().await; // pruner "in the section"

        // A concurrent repoint cannot even acquire the lock while the guard is held.
        assert!(
            lockarc.try_lock().is_err(),
            "set()/repoint must not acquire the publish lock while the prune guard is held"
        );

        let done = Arc::new(AtomicBool::new(false));
        let handle = {
            let store = store.clone();
            let done = done.clone();
            tokio::spawn(async move {
                store.set("lumina-fast", "new:1".to_string()).await; // blocks on the guard
                done.store(true, Ordering::SeqCst);
            })
        };
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert!(
            !done.load(Ordering::SeqCst),
            "set() must block while the owned prune guard is held"
        );
        assert_eq!(
            store.resolve("lumina-fast").as_deref(),
            Some("old:1"),
            "no repoint may be observed while the guard is held"
        );

        drop(guard); // pruner releases → repoint may now publish
        handle.await.unwrap();
        assert!(done.load(Ordering::SeqCst));
        assert_eq!(
            store.resolve("lumina-fast").as_deref(),
            Some("new:1"),
            "repoint applies once the guard is released"
        );
    }

    /// A remover that, while deleting `victim`, fires a concurrent repoint of
    /// `repoint_key` → `victim` on the SAME store. The orchestrator holds the
    /// publish guard across this whole `remove().await`, so the repoint MUST block
    /// until the delete completes + the guard drops — proving the record-drop→
    /// fs-delete window is guarded (the review-round-3 gap).
    struct RepointDuringDeleteRemover {
        archive_root: PathBuf,
        store: LuminaAliasStore,
        repoint_key: String,
        victim: String,
        observed_blocked_during_delete: Arc<std::sync::atomic::AtomicBool>,
        repoint_handle: Arc<std::sync::Mutex<Option<tokio::task::JoinHandle<()>>>>,
    }
    #[async_trait]
    impl ColdArchiveRemover for RepointDuringDeleteRemover {
        async fn remove(&self, model: &str) -> Result<(), String> {
            if model == self.victim {
                let store = self.store.clone();
                let key = self.repoint_key.clone();
                let victim = self.victim.clone();
                // Fire the repoint concurrently with the fs delete. It shares the
                // publish lock the orchestrator holds → it must block here.
                let h = tokio::spawn(async move { store.set(&key, victim).await });
                *self.repoint_handle.lock().unwrap() = Some(h);
                // Give it a chance to (try to) publish; it must still be blocked, so
                // the repoint must NOT yet be observable (lock-free `resolve`).
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                let still_blocked =
                    self.store.resolve(&self.repoint_key).as_deref() != Some(self.victim.as_str());
                self.observed_blocked_during_delete
                    .store(still_blocked, Ordering::SeqCst);
            }
            fs_remove_model(&self.archive_root, model)
        }
    }

    async fn run_fs_delete_window_case(resident: bool, repoint_key: &str) {
        use crate::routing::resident_set::Role;
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        let archive = base.join("archive");
        make_archive_model(&archive, "victim", "1", &[100]);
        make_archive_model(&archive, "other", "1", &[100]);
        let mut reg = reg_with(base, vec![]);
        reg.reconcile();
        reg.set_last_requested_for_test("victim:1", 1000);
        reg.set_last_requested_for_test("other:1", 1000);
        let registry = Arc::new(Mutex::new(reg));
        let mut scores = QualificationScores::default();
        scores.assistant_avg_value.insert("victim:1".into(), 1.0); // ranked first
        scores.assistant_avg_value.insert("other:1".into(), 500.0);

        // Real store, empty at PLAN time (so victim is planned/evictable at commit).
        let store = LuminaAliasStore::empty();
        let resident_cfg = if resident {
            ResidentSetConfig {
                enabled: true,
                aliases: vec![(Role::Embedding, repoint_key.to_string())],
                ..ResidentSetConfig::default()
            }
        } else {
            ResidentSetConfig {
                enabled: false,
                ..ResidentSetConfig::default()
            }
        };
        let keep = LuminaResidentKeepSet::new(store.clone(), HashMap::new(), resident_cfg);

        let observed_blocked = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let repoint_handle = Arc::new(std::sync::Mutex::new(None));
        let remover = RepointDuringDeleteRemover {
            archive_root: archive.clone(),
            store: store.clone(),
            repoint_key: repoint_key.to_string(),
            victim: "victim:1".to_string(),
            observed_blocked_during_delete: observed_blocked.clone(),
            repoint_handle: repoint_handle.clone(),
        };

        let free = Arc::new(std::sync::atomic::AtomicU64::new(0)); // maximally over
        let probe = FixedProbe { total: 1000, free };
        let lock = new_disk_op_lock();

        run_cold_quota_pass_at(
            &registry,
            &archive,
            &probe,
            &scores,
            &keep,
            &remover,
            &armed_cfg(),
            &lock,
            NOW,
        )
        .await;

        // The repoint (blocked during the delete) completes after the guard drops.
        if let Some(h) = repoint_handle.lock().unwrap().take() {
            h.await.unwrap();
        }

        assert!(
            observed_blocked.load(Ordering::SeqCst),
            "repoint MUST be blocked while the fs delete runs under the publish guard \
             (record-drop→fs-delete window is guarded)"
        );
        let reg = registry.lock().await;
        // victim was legitimately evictable at commit time (not a target then), so
        // it IS deleted — but the repoint could NOT tear that, and only lands after.
        assert!(reg.get("victim:1").is_none(), "victim evicted (was not a target at commit)");
        assert!(
            !archive
                .join("manifests/registry.ollama.ai/library/victim/1")
                .is_file(),
            "victim archive deleted under the guard"
        );
        assert_eq!(
            store.resolve(repoint_key).as_deref(),
            Some("victim:1"),
            "the repoint applied only AFTER the delete completed (serialized by the guard)"
        );
    }

    #[tokio::test]
    async fn repoint_during_fs_delete_is_blocked_alias_path() {
        // F1/F2 r3: an alias repoint during the record-drop→fs-delete window is held
        // off by the publish guard until the delete finishes.
        run_fs_delete_window_case(false, "lumina-fast").await;
    }

    #[tokio::test]
    async fn repoint_during_fs_delete_is_blocked_resident_path() {
        // Same guarantee for a resident-role alias (resolved through the dynamic
        // store) — it shares the identical publish lock.
        run_fs_delete_window_case(true, "lumina-embed").await;
    }
}
