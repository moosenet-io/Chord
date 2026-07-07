// SNAP-03: Model inventory — scans storage locations for GGUF and Ollama models.
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tracing::{debug, warn};

use crate::snap::config::{StorageLocation, StorageTier};

/// A discovered model file or Ollama manifest entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRecord {
    pub name: String,
    pub file_path: PathBuf,
    pub size_bytes: u64,
    pub quant_level: Option<String>,
    pub engine_compat: Vec<String>,
    pub storage_tier: StorageTier,
    pub last_used: Option<DateTime<Utc>>,
    pub loaded: bool,
}

/// Inventory of all discovered models across configured storage locations.
#[derive(Debug, Default)]
pub struct ModelInventory {
    pub records: Vec<ModelRecord>,
}

impl ModelInventory {
    /// Scan all configured storage locations and populate records.
    pub fn scan(locations: &[StorageLocation]) -> Self {
        let mut records = Vec::new();

        for loc in locations {
            let path = Path::new(&loc.path);
            if !path.exists() {
                debug!(path = %loc.path, "Storage location does not exist, skipping");
                continue;
            }

            // Scan for GGUF files
            let gguf_records = scan_gguf(path, &loc.tier);
            records.extend(gguf_records);

            // Scan for Ollama manifest blobs
            let ollama_records = scan_ollama_blobs(path, &loc.tier);
            records.extend(ollama_records);
        }

        Self { records }
    }

    /// Return models not used in the last 30 days, sorted by size descending.
    pub fn cleanup_candidates(&self) -> Vec<&ModelRecord> {
        let threshold = Utc::now() - chrono::Duration::days(30);
        let mut candidates: Vec<&ModelRecord> = self
            .records
            .iter()
            .filter(|r| {
                if let Some(last) = r.last_used {
                    last < threshold
                } else {
                    // No last_used: treat as old
                    true
                }
            })
            .collect();

        candidates.sort_by(|a, b| b.size_bytes.cmp(&a.size_bytes));
        candidates
    }

    /// Filter records by optional tier, loaded state, and search string.
    pub fn filter(
        &self,
        tier: Option<&str>,
        loaded: Option<bool>,
        search: Option<&str>,
    ) -> Vec<&ModelRecord> {
        self.records
            .iter()
            .filter(|r| {
                if let Some(t) = tier {
                    let tier_str = match r.storage_tier {
                        StorageTier::Hot => "hot",
                        StorageTier::Warm => "warm",
                    };
                    if tier_str != t {
                        return false;
                    }
                }
                if let Some(l) = loaded {
                    if r.loaded != l {
                        return false;
                    }
                }
                if let Some(s) = search {
                    if !r.name.to_lowercase().contains(&s.to_lowercase()) {
                        return false;
                    }
                }
                true
            })
            .collect()
    }

    /// Delete a model file from disk by name. Returns an error if file not found.
    pub fn delete_model(&mut self, name: &str) -> Result<PathBuf, String> {
        let idx = self
            .records
            .iter()
            .position(|r| r.name == name)
            .ok_or_else(|| format!("Model '{name}' not found in inventory"))?;

        let path = self.records[idx].file_path.clone();
        std::fs::remove_file(&path)
            .map_err(|e| format!("Failed to delete '{}': {e}", path.display()))?;

        self.records.remove(idx);
        Ok(path)
    }
}

/// Scan a directory tree for *.gguf files.
fn scan_gguf(root: &Path, tier: &StorageTier) -> Vec<ModelRecord> {
    let mut records = Vec::new();
    scan_dir_for_gguf(root, root, tier, &mut records);
    records
}

fn scan_dir_for_gguf(
    root: &Path,
    dir: &Path,
    tier: &StorageTier,
    records: &mut Vec<ModelRecord>,
) {
    let rd = match std::fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(e) => {
            warn!(dir = %dir.display(), error = %e, "Cannot read directory");
            return;
        }
    };

    for entry in rd.flatten() {
        let path = entry.path();
        if path.is_dir() {
            scan_dir_for_gguf(root, &path, tier, records);
        } else if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
            let size_bytes = entry.metadata().map(|m| m.len()).unwrap_or(0);
            let file_name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string();

            // Extract quant level from filename (e.g. Q4_K_M, Q5_K_S, IQ2_XXS)
            let quant_level = extract_quant_from_filename(&file_name);

            // Strip .gguf suffix for the model name
            let name = file_name
                .strip_suffix(".gguf")
                .unwrap_or(&file_name)
                .to_string();

            let last_used = entry
                .metadata()
                .ok()
                .and_then(|m| m.accessed().ok())
                .map(|t| DateTime::<Utc>::from(t));

            records.push(ModelRecord {
                name,
                file_path: path,
                size_bytes,
                quant_level,
                engine_compat: vec!["llama-server".into(), "ollama".into()],
                storage_tier: tier.clone(),
                last_used,
                loaded: false,
            });
        }
    }
}

/// Scan an Ollama blobs directory for manifest entries.
/// Ollama stores models in <root>/models/blobs/ — we look for non-sha files.
fn scan_ollama_blobs(root: &Path, tier: &StorageTier) -> Vec<ModelRecord> {
    let mut records = Vec::new();

    // Try root/manifests path (Ollama registry layout)
    let manifests_dir = root.join("manifests");
    if manifests_dir.is_dir() {
        scan_ollama_manifests(&manifests_dir, root, tier, &mut records);
    }

    // Also try root/models/manifests
    let models_manifests = root.join("models").join("manifests");
    if models_manifests.is_dir() {
        scan_ollama_manifests(&models_manifests, root, tier, &mut records);
    }

    records
}

fn scan_ollama_manifests(
    manifests_dir: &Path,
    blobs_root: &Path,
    tier: &StorageTier,
    records: &mut Vec<ModelRecord>,
) {
    // Keep manifests_dir constant through recursion so build_ollama_model_name
    // can compute paths relative to the top-level manifests directory.
    scan_ollama_manifests_inner(manifests_dir, manifests_dir, blobs_root, tier, records);
}

fn scan_ollama_manifests_inner(
    manifests_root: &Path,
    current_dir: &Path,
    blobs_root: &Path,
    tier: &StorageTier,
    records: &mut Vec<ModelRecord>,
) {
    // Walk manifests to find model tags
    let rd = match std::fs::read_dir(current_dir) {
        Ok(rd) => rd,
        Err(_) => return,
    };

    for entry in rd.flatten() {
        let path = entry.path();
        if path.is_dir() {
            // Recurse deeper, keeping manifests_root fixed
            scan_ollama_manifests_inner(manifests_root, &path, blobs_root, tier, records);
        } else {
            // This is a manifest file (model tag)
            if let Ok(content) = std::fs::read_to_string(&path) {
                if let Ok(manifest) = serde_json::from_str::<serde_json::Value>(&content) {
                    let size_bytes = calc_ollama_size(&manifest, blobs_root);
                    let name = path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown")
                        .to_string();

                    // Build full model name from path components relative to manifests_root
                    let full_name = build_ollama_model_name(&path, manifests_root);
                    let last_used = entry
                        .metadata()
                        .ok()
                        .and_then(|m| m.accessed().ok())
                        .map(|t| DateTime::<Utc>::from(t));

                    records.push(ModelRecord {
                        name: full_name.unwrap_or(name),
                        file_path: path,
                        size_bytes,
                        quant_level: None,
                        engine_compat: vec!["ollama".into()],
                        storage_tier: tier.clone(),
                        last_used,
                        loaded: false,
                    });
                }
            }
        }
    }
}

fn build_ollama_model_name(manifest_path: &Path, manifests_root: &Path) -> Option<String> {
    let rel = manifest_path.strip_prefix(manifests_root).ok()?;
    let parts: Vec<&str> = rel
        .components()
        .filter_map(|c| c.as_os_str().to_str())
        .collect();

    // LIVE-02: Standard Ollama layout is registry/namespace/model/tag (4 parts).
    // We skip registry+namespace and use model:tag.
    match parts.len() {
        4 => Some(format!("{}:{}", parts[2], parts[3])),
        3 => Some(format!("{}/{}:{}", parts[0], parts[1], parts[2])),
        2 => Some(format!("{}:{}", parts[0], parts[1])),
        1 => Some(parts[0].to_string()),
        _ => None,
    }
}

fn calc_ollama_size(manifest: &serde_json::Value, blobs_root: &Path) -> u64 {
    let mut total = 0u64;
    if let Some(layers) = manifest.get("layers").and_then(|v| v.as_array()) {
        for layer in layers {
            if let Some(size) = layer.get("size").and_then(|v| v.as_u64()) {
                total += size;
            } else if let Some(digest) = layer.get("digest").and_then(|v| v.as_str()) {
                // Try to get size from actual blob file
                let blob_name = digest.replace(':', "-");
                let blob_path = blobs_root.join("blobs").join(&blob_name);
                if let Ok(meta) = std::fs::metadata(&blob_path) {
                    total += meta.len();
                }
            }
        }
    }
    total
}

/// Extract quant level from a GGUF filename.
/// Examples: model-Q4_K_M.gguf → Q4_K_M, model.IQ2_XXS.gguf → IQ2_XXS
fn extract_quant_from_filename(filename: &str) -> Option<String> {
    // Common quant patterns
    let patterns = [
        "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
        "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M",
        "IQ4_XS", "IQ4_NL",
        "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
        "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M", "Q4_K_L",
        "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M",
        "Q6_K", "Q8_0", "F16", "F32", "BF16",
    ];

    let upper = filename.to_uppercase();
    for pat in &patterns {
        if upper.contains(pat) {
            return Some(pat.to_string());
        }
    }
    None
}

/// Disk usage for a storage location.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskUsage {
    pub path: String,
    pub total_bytes: u64,
    pub used_bytes: u64,
    pub free_bytes: u64,
    pub model_bytes: u64,
}

/// Get disk usage for a path. Returns zeros if unavailable.
pub fn get_disk_usage(path: &str, model_bytes: u64) -> DiskUsage {
    // Use statvfs via libc is complex; use df subprocess approach for simplicity
    let output = std::process::Command::new("df")
        .args(["-B1", "--output=size,used,avail", path])
        .output();

    let (total_bytes, used_bytes, free_bytes) = if let Ok(out) = output {
        let stdout = String::from_utf8_lossy(&out.stdout);
        let lines: Vec<&str> = stdout.lines().collect();
        if lines.len() >= 2 {
            let parts: Vec<&str> = lines[1].split_whitespace().collect();
            if parts.len() >= 3 {
                let total = parts[0].parse::<u64>().unwrap_or(0);
                let used = parts[1].parse::<u64>().unwrap_or(0);
                let free = parts[2].parse::<u64>().unwrap_or(0);
                (total, used, free)
            } else {
                (0, 0, 0)
            }
        } else {
            (0, 0, 0)
        }
    } else {
        (0, 0, 0)
    };

    DiskUsage {
        path: path.to_string(),
        total_bytes,
        used_bytes,
        free_bytes,
        model_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn make_temp_gguf(dir: &Path, name: &str, size: usize) -> PathBuf {
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&vec![0u8; size]).unwrap();
        path
    }

    #[test]
    fn scan_finds_gguf_files() {
        let tmp = TempDir::new().unwrap();
        make_temp_gguf(tmp.path(), "qwen3-Q4_K_M.gguf", 1024);
        make_temp_gguf(tmp.path(), "llama-Q5_K_S.gguf", 2048);

        let locs = vec![StorageLocation {
            name: "test".into(),
            path: tmp.path().to_string_lossy().to_string(),
            tier: StorageTier::Hot,
        }];

        let inv = ModelInventory::scan(&locs);
        assert_eq!(inv.records.len(), 2, "Expected 2 GGUF files");
        let names: Vec<&str> = inv.records.iter().map(|r| r.name.as_str()).collect();
        assert!(names.iter().any(|n| n.contains("qwen3")));
        assert!(names.iter().any(|n| n.contains("llama")));
    }

    #[test]
    fn scan_nested_gguf_files() {
        let tmp = TempDir::new().unwrap();
        let sub = tmp.path().join("subdir");
        std::fs::create_dir(&sub).unwrap();
        make_temp_gguf(&sub, "model-Q4_K_M.gguf", 512);

        let locs = vec![StorageLocation {
            name: "nested".into(),
            path: tmp.path().to_string_lossy().to_string(),
            tier: StorageTier::Warm,
        }];

        let inv = ModelInventory::scan(&locs);
        assert_eq!(inv.records.len(), 1);
        assert_eq!(inv.records[0].storage_tier, StorageTier::Warm);
    }

    #[test]
    fn cleanup_recommends_old_models() {
        let tmp = TempDir::new().unwrap();
        make_temp_gguf(tmp.path(), "old-model-Q4_K_M.gguf", 4096);

        let locs = vec![StorageLocation {
            name: "test".into(),
            path: tmp.path().to_string_lossy().to_string(),
            tier: StorageTier::Hot,
        }];

        let mut inv = ModelInventory::scan(&locs);
        // Override last_used to simulate old model (60 days ago)
        let old_date = Utc::now() - chrono::Duration::days(60);
        for r in &mut inv.records {
            r.last_used = Some(old_date);
        }

        let candidates = inv.cleanup_candidates();
        assert!(!candidates.is_empty(), "Old model should be a cleanup candidate");
    }

    #[test]
    fn filter_by_tier() {
        let records = vec![
            ModelRecord {
                name: "hot-model".into(),
                file_path: PathBuf::from("/hot/model.gguf"),
                size_bytes: 1000,
                quant_level: None,
                engine_compat: vec![],
                storage_tier: StorageTier::Hot,
                last_used: None,
                loaded: false,
            },
            ModelRecord {
                name: "warm-model".into(),
                file_path: PathBuf::from("/warm/model.gguf"),
                size_bytes: 2000,
                quant_level: None,
                engine_compat: vec![],
                storage_tier: StorageTier::Warm,
                last_used: None,
                loaded: false,
            },
        ];
        let inv = ModelInventory { records };

        let hot = inv.filter(Some("hot"), None, None);
        assert_eq!(hot.len(), 1);
        assert_eq!(hot[0].name, "hot-model");

        let warm = inv.filter(Some("warm"), None, None);
        assert_eq!(warm.len(), 1);
        assert_eq!(warm[0].name, "warm-model");
    }

    #[test]
    fn filter_by_search() {
        let records = vec![
            ModelRecord {
                name: "qwen3-coder-Q4_K_M".into(),
                file_path: PathBuf::from("/models/qwen3.gguf"),
                size_bytes: 1000,
                quant_level: Some("Q4_K_M".into()),
                engine_compat: vec![],
                storage_tier: StorageTier::Hot,
                last_used: None,
                loaded: false,
            },
            ModelRecord {
                name: "llama-3-Q5_K_M".into(),
                file_path: PathBuf::from("/models/llama.gguf"),
                size_bytes: 2000,
                quant_level: Some("Q5_K_M".into()),
                engine_compat: vec![],
                storage_tier: StorageTier::Hot,
                last_used: None,
                loaded: false,
            },
        ];
        let inv = ModelInventory { records };

        let results = inv.filter(None, None, Some("qwen"));
        assert_eq!(results.len(), 1);
        assert!(results[0].name.contains("qwen"));
    }

    #[test]
    fn delete_requires_file_to_exist() {
        let tmp = TempDir::new().unwrap();
        let gguf_path = make_temp_gguf(tmp.path(), "deletable-Q4_K_M.gguf", 128);

        let locs = vec![StorageLocation {
            name: "test".into(),
            path: tmp.path().to_string_lossy().to_string(),
            tier: StorageTier::Hot,
        }];
        let mut inv = ModelInventory::scan(&locs);
        assert_eq!(inv.records.len(), 1);

        let model_name = inv.records[0].name.clone();
        let deleted_path = inv.delete_model(&model_name).unwrap();
        assert_eq!(deleted_path, gguf_path);
        assert!(!gguf_path.exists());
        assert_eq!(inv.records.len(), 0);
    }

    #[test]
    fn extract_quant_levels() {
        assert_eq!(
            extract_quant_from_filename("model-Q4_K_M.gguf"),
            Some("Q4_K_M".into())
        );
        assert_eq!(
            extract_quant_from_filename("model.IQ2_XXS.gguf"),
            Some("IQ2_XXS".into())
        );
        assert_eq!(extract_quant_from_filename("model-F16.gguf"), Some("F16".into()));
        assert_eq!(extract_quant_from_filename("model.gguf"), None);
    }

    #[test]
    fn scan_nonexistent_location_graceful() {
        let locs = vec![StorageLocation {
            name: "missing".into(),
            path: "/tmp/nonexistent_chord_test_path_xyz".into(),
            tier: StorageTier::Hot,
        }];
        let inv = ModelInventory::scan(&locs);
        assert!(inv.records.is_empty(), "No records from missing path");
    }
}
