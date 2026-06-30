//! SNAP subsystem configuration.
//!
//! Vendors the `StorageTier` / `StorageLocation` types used by the model
//! inventory scanner (ported from harmony-chord's `config.rs`) and adds a
//! self-contained [`SnapConfig`] read entirely from environment variables.
//!
//! No hostnames, paths, or secrets are hardcoded — every value comes from the
//! environment with neutral, non-PII defaults, consistent with chord's
//! "all config from env / Infisical at runtime" rule.

use serde::{Deserialize, Serialize};

/// A configured model storage location to scan for inventory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageLocation {
    pub name: String,
    pub path: String,
    pub tier: StorageTier,
}

/// Storage tier of a location: hot (fast/local) or warm (bulk/network).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum StorageTier {
    Hot,
    Warm,
}

/// SNAP observability configuration, sourced from env vars.
#[derive(Debug, Clone)]
pub struct SnapConfig {
    /// llama-server base URL (`LLAMA_SERVER_URL`). Empty = not polled.
    pub llama_server_url: String,
    /// Ollama GPU base URL (`OLLAMA_URL`). Empty = not polled.
    pub ollama_url: String,
    /// Ollama CPU base URL (`OLLAMA_CPU_URL`). Empty = not polled.
    pub ollama_cpu_url: String,
    /// vLLM base URL (`CHORD_VLLM_URL`). Empty = not polled.
    pub vllm_url: String,
    /// Health-poll interval in seconds (`SNAP_POLL_INTERVAL_SECS`, default 10).
    pub poll_interval_secs: f64,
    /// Directory for the SNAP request-log JSONL (`SNAP_DATA_DIR`, default
    /// `CHORD_DATA_DIR` or the system temp dir).
    pub data_dir: std::path::PathBuf,
    /// Model storage locations to scan, parsed from `SNAP_STORAGE_LOCATIONS`
    /// (`name:tier:path` entries separated by `;`).
    pub storage_locations: Vec<StorageLocation>,
}

impl SnapConfig {
    /// Build SNAP config from environment variables. Never fails: missing /
    /// malformed values fall back to neutral defaults so SNAP is best-effort
    /// and can never block chord startup.
    pub fn from_env() -> Self {
        let env = |k: &str| std::env::var(k).unwrap_or_default();

        let poll_interval_secs = std::env::var("SNAP_POLL_INTERVAL_SECS")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| *v > 0.0)
            .unwrap_or(10.0);

        let data_dir = std::env::var("SNAP_DATA_DIR")
            .or_else(|_| std::env::var("CHORD_DATA_DIR"))
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::env::temp_dir());

        let storage_locations = parse_storage_locations(&env("SNAP_STORAGE_LOCATIONS"));

        Self {
            llama_server_url: env("LLAMA_SERVER_URL"),
            ollama_url: env("OLLAMA_URL"),
            ollama_cpu_url: env("OLLAMA_CPU_URL"),
            vllm_url: env("CHORD_VLLM_URL"),
            poll_interval_secs,
            data_dir,
            storage_locations,
        }
    }
}

/// Parse `SNAP_STORAGE_LOCATIONS` of the form
/// `name:tier:/abs/path;name2:warm:/other/path`. Unparseable entries are
/// skipped. `tier` is `hot` or `warm` (anything else → warm).
fn parse_storage_locations(raw: &str) -> Vec<StorageLocation> {
    raw.split(';')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .filter_map(|entry| {
            // Split into at most 3 parts so the path may itself contain ':'.
            let mut it = entry.splitn(3, ':');
            let name = it.next()?.trim().to_string();
            let tier_str = it.next()?.trim().to_lowercase();
            let path = it.next()?.trim().to_string();
            if name.is_empty() || path.is_empty() {
                return None;
            }
            let tier = if tier_str == "hot" {
                StorageTier::Hot
            } else {
                StorageTier::Warm
            };
            Some(StorageLocation { name, path, tier })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_multiple_locations() {
        let locs = parse_storage_locations("fast:hot:/srv/models;bulk:warm:/mnt/store");
        assert_eq!(locs.len(), 2);
        assert_eq!(locs[0].name, "fast");
        assert_eq!(locs[0].tier, StorageTier::Hot);
        assert_eq!(locs[0].path, "/srv/models");
        assert_eq!(locs[1].tier, StorageTier::Warm);
    }

    #[test]
    fn empty_yields_no_locations() {
        assert!(parse_storage_locations("").is_empty());
        assert!(parse_storage_locations("   ").is_empty());
    }

    #[test]
    fn unknown_tier_defaults_to_warm() {
        let locs = parse_storage_locations("x:bogus:/p");
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0].tier, StorageTier::Warm);
    }

    #[test]
    fn path_with_colon_preserved() {
        // Only first two ':' are delimiters; the rest is the path.
        let locs = parse_storage_locations("n:hot:/a:b/c");
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0].path, "/a:b/c");
    }
}
