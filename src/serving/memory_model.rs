//! Substrate-aware memory accounting (S85 SRV-11).
//!
//! The residency manager (SRV-05) admits models against *free memory*. On a
//! unified-memory APU that "free memory" question has TWO correct answers
//! depending on the host's BIOS/kernel substrate, so this module carries both
//! accounting models behind one [`MemoryModel`] trait and a **boot-time detector**
//! that selects the right one once per process:
//!
//!   - [`SeparateCeilings`] — **ACTIVE under a fixed BIOS carveout** (the current
//!     ~96 GB GPU / ~31 GB system split). GPU-GTT and CPU-system-RAM are two
//!     independent ceilings; a candidate checks only its own pool's free counter,
//!     and the two pools never cross.
//!   - [`UnifiedPool`] — **DORMANT until dynamic-GTT**. GPU and CPU draw from one
//!     physical ~128 GB pool, so a CPU-tier resident reduces what the GPU can grow
//!     into and vice versa. Written + tested now so a future dynamic-GTT migration
//!     finds the accounting already present and reviewed, not authored under
//!     pressure during the substrate change.
//!
//! ## Detection is per-process, at startup only
//! The substrate cannot change without a reboot (BIOS carveout + GRUB
//! `ttm.pages_limit`), which restarts the process and re-runs detection. There is
//! therefore NO live mid-run accounting swap. [`select_memory_model`] classifies
//! the substrate once and the choice is held for the process lifetime.
//!
//! ## Auto-activation is LOUD, never silent
//! On detecting dynamic-GTT the detector auto-selects [`UnifiedPool`] with NO
//! operator opt-in, but emits a prominent [`ActivationEvent`] (recorded in the
//! coordinator state file as `assumed_memory_model` and surfaced in residency
//! status) so a mis-detection is screaming-obvious rather than invisible.

use std::sync::Arc;

use terminus_rs::intake::serving::Runtime;

use super::eviction::ResidentView;

/// Which physical memory pool a backend's residency consumes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pool {
    /// VRAM / GTT — the GPU runtimes (llama.cpp-rocm, ollama-rocm).
    Gpu,
    /// System RAM — the genuine-CPU runtime (secondary ollama unit).
    Cpu,
}

impl Pool {
    /// The pool a runtime draws from. The two GPU runtimes consume VRAM/GTT; the
    /// CPU runtime consumes system RAM.
    pub fn of_runtime(rt: Runtime) -> Pool {
        match rt {
            Runtime::LlamaCpp | Runtime::Ollama => Pool::Gpu,
            Runtime::Cpu => Pool::Cpu,
        }
    }

    /// Stable lowercase id for events/state.
    pub fn id(self) -> &'static str {
        match self {
            Pool::Gpu => "gpu",
            Pool::Cpu => "cpu",
        }
    }
}

/// A live read of the host's free memory. Any field `None` means that counter was
/// unreadable — the memory model turns that into a FAIL-SAFE "won't fit" rather
/// than guessing (mirrors the SRV-05 unreadable-VRAM rule).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MemorySnapshot {
    /// Free VRAM/GTT in GB (amdgpu counter). `None` ⇒ unreadable.
    pub gpu_free_gb: Option<f64>,
    /// Free system RAM in GB (`MemAvailable`). `None` ⇒ unreadable.
    pub cpu_free_gb: Option<f64>,
    /// Total unified physical pool in GB, when the substrate exposes one
    /// (dynamic-GTT). `None` under a fixed carveout (the pools are independent).
    pub physical_total_gb: Option<f64>,
}

/// The accounting policy: given a candidate's pool, the live snapshot, and the
/// current residents, how much free memory may the candidate draw from?
pub trait MemoryModel: Send + Sync {
    /// Stable id recorded as `assumed_memory_model`.
    fn id(&self) -> &'static str;

    /// The admissible free GB for a candidate in `pool`. `None` ⇒ fail-safe: the
    /// relevant counter was unreadable (or the pool is unsized) → treat as "won't
    /// fit", never an OOM-risking launch.
    fn admissible_free_gb(
        &self,
        pool: Pool,
        snap: &MemorySnapshot,
        residents: &[ResidentView],
    ) -> Option<f64>;
}

/// **ACTIVE under a fixed BIOS carveout.** Two independent ceilings: a GPU
/// candidate checks GPU free, a CPU candidate checks CPU free; the pools never
/// cross. This is the conservative, currently-correct model.
#[derive(Debug, Clone, Copy, Default)]
pub struct SeparateCeilings;

impl MemoryModel for SeparateCeilings {
    fn id(&self) -> &'static str {
        "separate-ceilings"
    }

    fn admissible_free_gb(
        &self,
        pool: Pool,
        snap: &MemorySnapshot,
        _residents: &[ResidentView],
    ) -> Option<f64> {
        // Each pool's live free counter already nets out that pool's residents, so
        // the candidate simply checks its own pool — and ONLY its own pool.
        match pool {
            Pool::Gpu => snap.gpu_free_gb,
            Pool::Cpu => snap.cpu_free_gb,
        }
    }
}

/// **DORMANT until dynamic-GTT.** One physical pool: every resident (GPU *or* CPU)
/// draws from the same ~128 GB, so the admissible free for ANY candidate is the
/// physical total minus the combined draw of all residents. This is the cross-pool
/// effect that [`SeparateCeilings`] cannot see.
#[derive(Debug, Clone, Copy)]
pub struct UnifiedPool {
    /// The physical pool size in GB (set at detection from the substrate).
    pub physical_total_gb: f64,
}

impl MemoryModel for UnifiedPool {
    fn id(&self) -> &'static str {
        "unified-pool"
    }

    fn admissible_free_gb(
        &self,
        _pool: Pool,
        snap: &MemorySnapshot,
        residents: &[ResidentView],
    ) -> Option<f64> {
        // Prefer a live unified counter; otherwise fall back to the detected total.
        let total = snap.physical_total_gb.unwrap_or(self.physical_total_gb);
        if !(total.is_finite() && total > 0.0) {
            // Fail-safe: an unsized unified pool cannot admit anything.
            return None;
        }
        // Combined draw across BOTH pools — a CPU resident reduces GPU headroom.
        let used: f64 = residents.iter().map(|r| r.vram_gb).sum();
        Some((total - used).max(0.0))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Boot-time substrate detection
// ─────────────────────────────────────────────────────────────────────────────

/// Substrate facts read once at boot to classify the host. All in GB.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubstrateInfo {
    /// `mem_info_vram_total` — the BIOS VRAM carveout.
    pub vram_carveout_gb: f64,
    /// `mem_info_gtt_total` — the GTT pool the GPU can address.
    pub gtt_total_gb: f64,
    /// `MemTotal` — system RAM.
    pub system_ram_gb: f64,
}

/// The classified substrate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Substrate {
    /// Large dedicated VRAM carveout + modest GTT — typical fixed-carveout host.
    FixedCarveout,
    /// Tiny BIOS carveout + large GTT — the GPU grows into the shared pool.
    DynamicGtt,
    /// Neither pattern cleanly matches — default to the safer model + warn.
    Ambiguous,
}

/// A dynamic-GTT host carves almost no dedicated VRAM (AMD suggests ~0.5 GB) and
/// exposes a large GTT. Below this carveout we treat the host as "tiny carveout".
const DYNAMIC_CARVEOUT_MAX_GB: f64 = 4.0;
/// ...and a GTT at/above this is "large" (the bulk of the physical pool).
const DYNAMIC_GTT_MIN_GB: f64 = 64.0;
/// A fixed-carveout host dedicates a large VRAM slice up front.
const FIXED_CARVEOUT_MIN_GB: f64 = 32.0;

/// Classify the substrate from its memory facts. Pure + deterministic so the
/// decision is unit-tested without a host.
pub fn classify_substrate(info: &SubstrateInfo) -> Substrate {
    let tiny_carveout = info.vram_carveout_gb <= DYNAMIC_CARVEOUT_MAX_GB;
    let large_gtt = info.gtt_total_gb >= DYNAMIC_GTT_MIN_GB;
    if tiny_carveout && large_gtt {
        Substrate::DynamicGtt
    } else if info.vram_carveout_gb >= FIXED_CARVEOUT_MIN_GB {
        Substrate::FixedCarveout
    } else {
        Substrate::Ambiguous
    }
}

/// The loud, structured record of which accounting model was selected and why.
/// Surfaced in the log + recorded as `assumed_memory_model` in the coordinator
/// state file (SRV-13) so a mis-detection is never invisible.
#[derive(Debug, Clone, PartialEq)]
pub struct ActivationEvent {
    /// The chosen model id: `separate-ceilings` | `unified-pool`.
    pub assumed_memory_model: &'static str,
    /// Why it was chosen (the detected substrate + any fallback note).
    pub trigger: String,
    /// The GPU pool size at boot (carveout under fixed, physical under unified).
    pub gpu_pool_gb: f64,
    /// The CPU/system pool size at boot.
    pub cpu_pool_gb: f64,
    /// `true` when the selection must be ANNOUNCED prominently: auto-activating
    /// unified-pool, or falling back from an ambiguous substrate. `false` only for
    /// the expected fixed-carveout → separate-ceilings default.
    pub loud: bool,
}

/// The selected model plus its activation event.
pub struct ModelSelection {
    pub model: Arc<dyn MemoryModel>,
    pub event: ActivationEvent,
}

/// Select the accounting model for the host's substrate. Called ONCE at startup.
///
///   - **fixed-carveout** → [`SeparateCeilings`] (the active default; `loud=false`).
///   - **dynamic-GTT** → [`UnifiedPool`] auto-activated with `loud=true` and the
///     `dynamic-GTT detected at boot` trigger (never silent).
///   - **ambiguous** → [`SeparateCeilings`] (the SAFER, more conservative GPU
///     ceiling) with `loud=true` and an "investigate" trigger.
pub fn select_memory_model(info: &SubstrateInfo) -> ModelSelection {
    match classify_substrate(info) {
        Substrate::FixedCarveout => ModelSelection {
            model: Arc::new(SeparateCeilings),
            event: ActivationEvent {
                assumed_memory_model: "separate-ceilings",
                trigger: "fixed-carveout detected at boot".to_string(),
                gpu_pool_gb: info.vram_carveout_gb,
                cpu_pool_gb: info.system_ram_gb,
                loud: false,
            },
        },
        Substrate::DynamicGtt => {
            // Under dynamic-GTT the GPU grows into the shared pool; the physical
            // total is (roughly) the carveout + system RAM.
            let physical = info.vram_carveout_gb + info.system_ram_gb;
            ModelSelection {
                model: Arc::new(UnifiedPool {
                    physical_total_gb: physical,
                }),
                event: ActivationEvent {
                    assumed_memory_model: "unified-pool",
                    trigger: format!(
                        "dynamic-GTT detected at boot (carveout={:.0}GB gtt={:.0}GB) → \
                         unified-pool auto-activated",
                        info.vram_carveout_gb, info.gtt_total_gb
                    ),
                    gpu_pool_gb: physical,
                    cpu_pool_gb: physical,
                    loud: true,
                },
            }
        }
        Substrate::Ambiguous => ModelSelection {
            model: Arc::new(SeparateCeilings),
            event: ActivationEvent {
                assumed_memory_model: "separate-ceilings",
                trigger: format!(
                    "ambiguous substrate (carveout={:.0}GB gtt={:.0}GB) → safer default \
                     (separate-ceilings); investigate",
                    info.vram_carveout_gb, info.gtt_total_gb
                ),
                gpu_pool_gb: info.vram_carveout_gb,
                cpu_pool_gb: info.system_ram_gb,
                loud: true,
            },
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resident(id: &str, rt: Runtime, gb: f64) -> ResidentView {
        use super::super::eviction::Tier;
        ResidentView {
            model_id: id.to_string(),
            runtime: rt,
            vram_gb: gb,
            tier: Tier::KeepWarm,
            last_used_tick: 1,
        }
    }

    // ── SeparateCeilings: pools never cross ─────────────────────────────────────

    #[test]
    fn separate_ceilings_gpu_checks_gpu_only() {
        let m = SeparateCeilings;
        let snap = MemorySnapshot {
            gpu_free_gb: Some(40.0),
            cpu_free_gb: Some(8.0),
            physical_total_gb: None,
        };
        // A GPU candidate sees GPU free (40), NOT cpu free.
        assert_eq!(m.admissible_free_gb(Pool::Gpu, &snap, &[]), Some(40.0));
        // A CPU candidate sees CPU free (8), NOT gpu free.
        assert_eq!(m.admissible_free_gb(Pool::Cpu, &snap, &[]), Some(8.0));
    }

    #[test]
    fn separate_ceilings_pool_of_runtime_mapping() {
        assert_eq!(Pool::of_runtime(Runtime::LlamaCpp), Pool::Gpu);
        assert_eq!(Pool::of_runtime(Runtime::Ollama), Pool::Gpu);
        assert_eq!(Pool::of_runtime(Runtime::Cpu), Pool::Cpu);
    }

    #[test]
    fn separate_ceilings_unreadable_pool_fails_safe() {
        let m = SeparateCeilings;
        let snap = MemorySnapshot {
            gpu_free_gb: None, // unreadable
            cpu_free_gb: Some(8.0),
            physical_total_gb: None,
        };
        // GPU counter unreadable → None (won't fit), never a guess.
        assert_eq!(m.admissible_free_gb(Pool::Gpu, &snap, &[]), None);
        // CPU still readable.
        assert_eq!(m.admissible_free_gb(Pool::Cpu, &snap, &[]), Some(8.0));
    }

    // ── UnifiedPool: the cross-pool effect ──────────────────────────────────────

    #[test]
    fn unified_pool_sums_both_pools_against_one_ceiling() {
        let m = UnifiedPool {
            physical_total_gb: 128.0,
        };
        let snap = MemorySnapshot {
            gpu_free_gb: Some(10.0),
            cpu_free_gb: Some(10.0),
            physical_total_gb: Some(128.0),
        };
        // A 30GB GPU resident + a 20GB CPU resident = 50GB of the shared pool used.
        let residents = [
            resident("gpu-model", Runtime::LlamaCpp, 30.0),
            resident("cpu-model", Runtime::Cpu, 20.0),
        ];
        // Admissible free for a GPU candidate = 128 - 50 = 78 (the CPU resident
        // reduced GPU headroom — the effect SeparateCeilings cannot see).
        assert_eq!(
            m.admissible_free_gb(Pool::Gpu, &snap, &residents),
            Some(78.0)
        );
        // ...and identically for a CPU candidate (one shared pool).
        assert_eq!(
            m.admissible_free_gb(Pool::Cpu, &snap, &residents),
            Some(78.0)
        );
    }

    #[test]
    fn unified_pool_cpu_resident_reduces_gpu_headroom_vs_separate() {
        let residents = [resident("cpu-model", Runtime::Cpu, 25.0)];
        let snap = MemorySnapshot {
            gpu_free_gb: Some(90.0),
            cpu_free_gb: Some(5.0),
            physical_total_gb: Some(128.0),
        };
        // SeparateCeilings: a GPU candidate ignores the CPU resident entirely → 90.
        assert_eq!(
            SeparateCeilings.admissible_free_gb(Pool::Gpu, &snap, &residents),
            Some(90.0)
        );
        // UnifiedPool: the same CPU resident eats into the GPU candidate's room.
        let u = UnifiedPool {
            physical_total_gb: 128.0,
        };
        assert_eq!(u.admissible_free_gb(Pool::Gpu, &snap, &residents), Some(103.0));
    }

    #[test]
    fn unified_pool_unsized_fails_safe() {
        let u = UnifiedPool {
            physical_total_gb: 0.0,
        };
        let snap = MemorySnapshot {
            physical_total_gb: None,
            ..Default::default()
        };
        assert_eq!(u.admissible_free_gb(Pool::Gpu, &snap, &[]), None);
    }

    // ── Detection ───────────────────────────────────────────────────────────────

    #[test]
    fn detects_fixed_carveout_host() {
        // Example fixed-carveout host: 96GB carveout, 32GB GTT, 31GB system.
        let info = SubstrateInfo {
            vram_carveout_gb: 96.0,
            gtt_total_gb: 32.0,
            system_ram_gb: 31.0,
        };
        assert_eq!(classify_substrate(&info), Substrate::FixedCarveout);
        let sel = select_memory_model(&info);
        assert_eq!(sel.model.id(), "separate-ceilings");
        assert_eq!(sel.event.assumed_memory_model, "separate-ceilings");
        assert!(!sel.event.loud, "fixed-carveout is the expected default, not loud");
    }

    #[test]
    fn detects_dynamic_gtt_and_auto_activates_loudly() {
        // Post-migration: 0.5GB carveout, 120GB GTT.
        let info = SubstrateInfo {
            vram_carveout_gb: 0.5,
            gtt_total_gb: 120.0,
            system_ram_gb: 127.0,
        };
        assert_eq!(classify_substrate(&info), Substrate::DynamicGtt);
        let sel = select_memory_model(&info);
        assert_eq!(sel.model.id(), "unified-pool");
        assert_eq!(sel.event.assumed_memory_model, "unified-pool");
        // NEGATIVE TEST: unified-pool is NEVER selected silently.
        assert!(sel.event.loud, "dynamic-GTT auto-activation MUST be loud");
        assert!(sel.event.trigger.contains("dynamic-GTT"));
        // physical pool ≈ carveout + system RAM.
        assert!((sel.event.gpu_pool_gb - 127.5).abs() < 1e-9);
    }

    #[test]
    fn ambiguous_substrate_defaults_to_safer_with_warning() {
        // Neither tiny-carveout+large-GTT nor a large fixed carveout.
        let info = SubstrateInfo {
            vram_carveout_gb: 16.0,
            gtt_total_gb: 16.0,
            system_ram_gb: 31.0,
        };
        assert_eq!(classify_substrate(&info), Substrate::Ambiguous);
        let sel = select_memory_model(&info);
        // Safer = SeparateCeilings (more conservative on the GPU ceiling).
        assert_eq!(sel.model.id(), "separate-ceilings");
        assert!(sel.event.loud, "ambiguous fallback must warn");
        assert!(sel.event.trigger.contains("ambiguous"));
    }
}
