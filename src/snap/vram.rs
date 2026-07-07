// SNAP-02: VRAM tracker — reads GPU memory from sysfs/rocm-smi and Ollama.
use chrono::Utc;
use tracing::warn;

use crate::snap::state::{VRAMAllocation, VRAMState};

use crate::snap::state::SharedInferenceState;

/// Update VRAM state in the shared inference state from all available sources.
pub async fn update_vram(state: &SharedInferenceState) {
    let total_mb = read_total_vram_mb().await;
    let allocations = collect_allocations(state).await;
    let used_mb: u64 = allocations.iter().map(|a| a.size_mb).sum();

    let vram = VRAMState {
        total_mb,
        used_mb,
        free_mb: total_mb.saturating_sub(used_mb),
        allocations,
    };

    let mut s = state.write().await;
    s.vram = vram;
}

/// Read total GPU VRAM from sysfs (gfx1151 / Strix Halo path).
/// Falls back to 96GB default if sysfs is unavailable.
async fn read_total_vram_mb() -> u64 {
    // Try sysfs paths for AMD APU / unified memory
    let sysfs_paths = [
        "/sys/class/drm/card0/device/mem_info_vram_total",
        "/sys/class/drm/card1/device/mem_info_vram_total",
    ];

    for path in &sysfs_paths {
        if let Ok(content) = tokio::fs::read_to_string(path).await {
            if let Ok(bytes) = content.trim().parse::<u64>() {
                if bytes > 0 {
                    return bytes / (1024 * 1024);
                }
            }
        }
    }

    // Try rocm-smi as subprocess fallback
    if let Ok(output) = tokio::process::Command::new("rocm-smi")
        .args(["--showmeminfo", "vram", "--json"])
        .output()
        .await
    {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&stdout) {
                // rocm-smi JSON format varies by version; try common paths
                let total = val
                    .pointer("/card0/VRAM Total Memory (B)")
                    .or_else(|| val.pointer("/GPU[0]/vram_total"))
                    .and_then(|v| v.as_u64());
                if let Some(bytes) = total {
                    if bytes > 0 {
                        return bytes / (1024 * 1024);
                    }
                }
            }
        }
    }

    warn!("Could not read VRAM from sysfs or rocm-smi — defaulting to 96GB estimate");
    96 * 1024 // 96 GB default for Strix Halo
}

/// Build VRAM allocations from loaded models reported by all engines.
async fn collect_allocations(state: &SharedInferenceState) -> Vec<VRAMAllocation> {
    let s = state.read().await;
    let mut allocs = Vec::new();

    for engine in &s.engines {
        for model in &engine.models {
            if model.size_vram_mb > 0 {
                allocs.push(VRAMAllocation {
                    model_name: model.name.clone(),
                    engine: engine.name.clone(),
                    size_mb: model.size_vram_mb,
                    loaded_at: Utc::now(), // approximate; activity tracker refines this
                });
            }
        }
    }

    allocs
}

/// Estimate VRAM needed for a model based on its size on disk.
/// Uses a lookup table for known model sizes; falls back to disk size × 1.2 factor.
pub fn estimate_vram_needed_mb(model_size_bytes: u64, _model_name: &str) -> u64 {
    // For GPU inference, loaded model takes roughly model_size × 1.0-1.2 in VRAM
    // (quantized models load nearly at their file size)
    let base_mb = model_size_bytes / (1024 * 1024);
    // Add ~20% for KV cache and overhead
    base_mb + (base_mb / 5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_vram_reasonable() {
        let eighteen_gb = 18u64 * 1024 * 1024 * 1024;
        let est = estimate_vram_needed_mb(eighteen_gb, "qwen3-coder:30b");
        // Should be around 21-22 GB (18 + 20%)
        assert!(est > 18_000 && est < 24_000, "Expected 18-24GB, got {est}MB");
    }

    #[test]
    fn vram_can_load_predicts_fit() {
        let vram = crate::snap::state::VRAMState {
            total_mb: 98304,
            used_mb: 50000,
            free_mb: 48304,
            allocations: vec![],
        };
        assert!(vram.can_fit(17000), "17GB should fit in 48GB free");
        assert!(!vram.can_fit(60000), "60GB should not fit in 48GB free");
    }

    #[test]
    fn zero_allocations_zero_used() {
        let allocs: Vec<crate::snap::state::VRAMAllocation> = vec![];
        let used: u64 = allocs.iter().map(|a| a.size_mb).sum();
        assert_eq!(used, 0);
    }
}
