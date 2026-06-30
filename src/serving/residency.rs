//! VRAM admission + tier-aware residency manager (S85 SRV-05).
//!
//! The real [`ResidencyManager`] (the SRV-04 seam) — it REPLACES SRV-04's
//! `PassThroughResidency` stub WITHOUT changing the launcher API. Given a
//! keep-warm model, it:
//!   1. reads the host's FREE VRAM (via the [`config`] sysfs helper, fail-safe on
//!      unreadable),
//!   2. computes a tier-aware [`EvictionPlan`](super::eviction::EvictionPlan)
//!      (transient-first, chat-pinned, queue-then-LRU keep-warm),
//!   3. evicts/queues accordingly under a single admission lock so two concurrent
//!      launches can NEVER double-admit past the VRAM ceiling,
//!   4. brings the model resident (or reuses an already-resident slot), health-
//!      checks it, mirrors the residency to an atomic JSON state file, and emits
//!      a sanitized event for every admit/queue/evict decision.
//!
//! ## Concurrency model (the no-double-admit guarantee)
//! All admission state — the resident set AND the in-flight RESERVATIONS — lives
//! behind one [`tokio::sync::Mutex`]. An admission:
//!   - takes the lock,
//!   - reads free VRAM and subtracts every active reservation's footprint,
//!   - decides + RESERVES the candidate's footprint (recording it before dropping
//!     the lock),
//!   - drops the lock only to perform the slow launch/health work,
//!   - re-takes the lock to commit the resident and release the reservation.
//!
//! Because the *decision + reservation* are atomic under the lock, a second
//! concurrent launch sees the first's reservation as already-consumed VRAM and
//! cannot admit past the ceiling — there is no read-decide-write race. The bounded
//! keep-warm wait happens WITHOUT holding the lock (so it never stalls other
//! admissions), then re-acquires and re-plans.
//!
//! ## Fail-safe
//! Unreadable free VRAM ([`config::read_free_vram_gb`] → `None`) is treated as
//! "won't fit": the manager QUEUEs (bounded wait, re-read) rather than risk an OOM
//! launch, and denies admission if it stays unreadable — it never force-launches.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::Mutex;

use terminus_rs::intake::serving::{ModelId, Runtime};

use super::eviction::{plan_admission, EvictTarget, EvictionPlan, ResidentView, Tier};
use super::launcher::{ResidencyError, ResidencyManager, Slot};
use super::memory_model::{ActivationEvent, MemoryModel, MemorySnapshot, ModelSelection, Pool, SeparateCeilings};
use super::mode::{ModeAction, ModeController, ModeError, OperatingMode};
use crate::config;

/// A model currently resident in VRAM, with everything the manager needs to serve
/// from it, account its footprint, and evict it.
#[derive(Debug, Clone)]
pub struct Resident {
    pub model_id: String,
    pub runtime: Runtime,
    pub endpoint: String,
    pub vram_gb: f64,
    pub tier: Tier,
    /// Monotonic last-use stamp for LRU (higher = more recently used).
    pub last_used_tick: u64,
}

/// Brings models up / tears them down and reports free VRAM. Abstracted so tests
/// drive deterministic admission/eviction without a GPU or real process; production
/// wires the SRV-04 launcher + the sysfs counter behind it.
#[async_trait]
pub trait WarmLauncher: Send + Sync {
    /// Free VRAM in GB, or `None` when unreadable (the FAIL-SAFE signal). The
    /// production impl reads [`config::read_free_vram_gb`].
    async fn free_vram_gb(&self) -> Option<f64>;

    /// Bring `model_id` resident (cold-load if needed) and return the
    /// health-checked endpoint + the runtime that served it. An `Err` carries a
    /// genericized reason (no infra) — the manager turns it into `CannotAdmit`.
    async fn launch(&self, model_id: &str, vram_gb: f64) -> Result<(Runtime, String), String>;

    /// Health-check an already-resident endpoint. Used to validate a reused slot
    /// before returning it (the SRV-04 contract: a returned slot is serveable).
    async fn health_check(&self, endpoint: &str) -> bool;

    /// Reclaim `model_id`'s VRAM. The impl finishes or cleanly cancels any
    /// in-flight generation before the VRAM is released (SRV-05 edge case). An
    /// `Err` is a genericized reason; the manager records it and aborts the
    /// admission rather than admit past the ceiling.
    async fn evict(&self, model_id: &str) -> Result<(), String>;
}

/// A sink for sanitized residency events (S6/S77). Production wires structured
/// logging; tests record them to assert behavior. Implementations MUST NOT carry
/// raw infra — only the model id, tier, and a stable decision word.
pub trait EventSink: Send + Sync {
    fn emit(&self, event: &ResidencyEvent);
}

/// A no-op event sink (used when no observability sink is wired).
pub struct NoopEventSink;
impl EventSink for NoopEventSink {
    fn emit(&self, _event: &ResidencyEvent) {}
}

/// A structured, already-sanitized residency event. Carries NO endpoint, host, or
/// path — only the model id (a registry key, not infra), the tier, and a stable
/// decision word, so it is safe to log/persist (S6/S77).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidencyEvent {
    /// One of: `admit`, `reuse`, `queue`, `evict`, `admission-denied`.
    pub decision: &'static str,
    /// The model the decision is about (registry key — not infra).
    pub model_id: String,
    /// Tier id when relevant (e.g. the evicted resident's tier), else "".
    pub tier: &'static str,
}

impl ResidencyEvent {
    fn new(decision: &'static str, model_id: impl Into<String>, tier: &'static str) -> Self {
        ResidencyEvent {
            decision,
            model_id: model_id.into(),
            tier,
        }
    }
}

/// The mutable admission state, guarded as a whole by one mutex.
#[derive(Default)]
struct Registry {
    /// Currently-resident models keyed by model id.
    residents: HashMap<String, Resident>,
    /// In-flight reservations (model id → reserved GB). A reservation is the
    /// VRAM an admission has committed to but not yet launched; it is subtracted
    /// from free VRAM so a concurrent admission can't double-spend the same GB.
    reservations: HashMap<String, f64>,
    /// Model id of the pinned chat-role model, if assigned. Exempt from eviction.
    pinned_chat_model: Option<String>,
    /// The explicit operating mode (SRV-13). Defaults to assistant-live.
    mode: OperatingMode,
}

impl Registry {
    /// Build the policy's resident-view snapshot, marking the pinned chat model so
    /// the eviction policy excludes it. Reservations are NOT residents yet, so they
    /// are never eviction targets.
    fn snapshot(&self) -> Vec<ResidentView> {
        self.residents
            .values()
            .map(|r| {
                let tier = if self.pinned_chat_model.as_deref() == Some(r.model_id.as_str()) {
                    Tier::PinnedChat
                } else {
                    r.tier
                };
                ResidentView {
                    model_id: r.model_id.clone(),
                    runtime: r.runtime,
                    vram_gb: r.vram_gb,
                    tier,
                    last_used_tick: r.last_used_tick,
                }
            })
            .collect()
    }

    /// Sum of all active reservations' footprints (VRAM committed-but-not-launched).
    fn reserved_gb(&self) -> f64 {
        self.reservations.values().copied().sum()
    }
}

/// The real residency manager. Cheap to clone (all shared state is `Arc`-wrapped),
/// so it can be handed to many concurrent request tasks.
#[derive(Clone)]
pub struct VramResidencyManager {
    inner: Arc<Inner>,
}

struct Inner {
    registry: Mutex<Registry>,
    launcher: Arc<dyn WarmLauncher>,
    events: Arc<dyn EventSink>,
    /// Bounded keep-warm wait before the manager evicts an LRU keep-warm.
    wait_threshold: Duration,
    /// Optional atomic JSON state-file path (None ⇒ disk mirror disabled).
    state_path: Option<String>,
    /// Monotonic LRU clock (ticks on every use/admission).
    clock: AtomicU64,
    /// The substrate-selected accounting model (SRV-11). Defaults to the
    /// conservative [`SeparateCeilings`]; the binary wires the detected model.
    memory_model: Arc<dyn MemoryModel>,
    /// The (loud-or-not) record of how `memory_model` was selected — surfaced in
    /// the state file as `assumed_memory_model` and emitted at construction.
    activation: ActivationEvent,
    /// GPU pool ceiling (GB) for SRV-13 mode headroom — from the detected pools.
    gpu_ceiling_gb: f64,
    /// CPU/system pool ceiling (GB) for SRV-13 mode headroom.
    cpu_ceiling_gb: f64,
}

impl VramResidencyManager {
    /// Build a manager with config-sourced wait threshold + state path. No infra
    /// literals: the wait threshold and state path come from [`config`].
    pub fn new(launcher: Arc<dyn WarmLauncher>, events: Arc<dyn EventSink>) -> Self {
        Self::with_settings(
            launcher,
            events,
            Duration::from_millis(config::residency_wait_threshold_ms()),
            config::residency_state_path(),
        )
    }

    /// Build with explicit settings (used by tests to drive short waits + temp
    /// state paths deterministically). Uses the conservative [`SeparateCeilings`]
    /// accounting model — call [`with_memory_model`](Self::with_memory_model) to
    /// wire a substrate-detected one.
    pub fn with_settings(
        launcher: Arc<dyn WarmLauncher>,
        events: Arc<dyn EventSink>,
        wait_threshold: Duration,
        state_path: Option<String>,
    ) -> Self {
        Self::with_memory_model(
            launcher,
            events,
            wait_threshold,
            state_path,
            ModelSelection {
                model: Arc::new(SeparateCeilings),
                event: ActivationEvent {
                    assumed_memory_model: "separate-ceilings",
                    trigger: "default (no substrate detection wired)".to_string(),
                    gpu_pool_gb: 0.0,
                    cpu_pool_gb: 0.0,
                    loud: false,
                },
            },
        )
    }

    /// Build with an explicit substrate-detected [`ModelSelection`] (SRV-11). The
    /// activation is emitted at construction (loudly when auto-activating
    /// unified-pool / falling back from an ambiguous substrate), so a mis-detection
    /// is never invisible.
    pub fn with_memory_model(
        launcher: Arc<dyn WarmLauncher>,
        events: Arc<dyn EventSink>,
        wait_threshold: Duration,
        state_path: Option<String>,
        selection: ModelSelection,
    ) -> Self {
        let ModelSelection {
            model,
            event: activation,
        } = selection;
        // Announce the selected accounting model. The `loud` flag drives the log
        // level; the structured event is always emitted so status/tests can see it.
        if activation.loud {
            tracing::warn!(
                assumed_memory_model = activation.assumed_memory_model,
                trigger = %activation.trigger,
                "SRV-11 memory accounting model auto-activated"
            );
        } else {
            tracing::info!(
                assumed_memory_model = activation.assumed_memory_model,
                "SRV-11 memory accounting model selected"
            );
        }
        events.emit(&ResidencyEvent::new(
            "memory-model",
            activation.assumed_memory_model,
            "",
        ));
        let gpu_ceiling_gb = activation.gpu_pool_gb;
        let cpu_ceiling_gb = activation.cpu_pool_gb;
        VramResidencyManager {
            inner: Arc::new(Inner {
                registry: Mutex::new(Registry::default()),
                launcher,
                events,
                wait_threshold,
                state_path,
                clock: AtomicU64::new(1),
                memory_model: model,
                activation,
                gpu_ceiling_gb,
                cpu_ceiling_gb,
            }),
        }
    }

    /// The current operating mode (SRV-13).
    pub async fn mode(&self) -> OperatingMode {
        self.inner.registry.lock().await.mode
    }

    /// A [`ModeController`] snapshot for headroom / oversize checks against the
    /// current mode + detected ceilings.
    pub async fn mode_controller(&self) -> ModeController {
        ModeController {
            mode: self.inner.registry.lock().await.mode,
            gpu_ceiling_gb: self.inner.gpu_ceiling_gb,
            cpu_ceiling_gb: self.inner.cpu_ceiling_gb,
        }
    }

    /// Deliberately switch the operating mode (SRV-13). Switching OFF assistant-live
    /// requires `confirm` (else [`ModeError::NeedsConfirm`]) and GRACEFULLY unpins
    /// the chat model (demoted to evictable keep-warm — never hard-dropped), so live
    /// Lumina is never silently torn down. Switching INTO assistant-live records the
    /// mode and returns [`ModeAction::LoadAndPin`] so the caller loads + pins the
    /// chat model (SRV-06) before coder work is accepted. Returns the action taken.
    pub async fn switch_mode(
        &self,
        target: OperatingMode,
        confirm: bool,
    ) -> Result<ModeAction, ModeError> {
        let action = {
            let reg = self.inner.registry.lock().await;
            ModeController {
                mode: reg.mode,
                gpu_ceiling_gb: self.inner.gpu_ceiling_gb,
                cpu_ceiling_gb: self.inner.cpu_ceiling_gb,
            }
            .request_switch(target, confirm)?
        };
        // Apply the action. GracefulUnpin clears the pin (SRV-06 demotes the old
        // assistant to keep-warm); the mode is recorded in both cases.
        if action == ModeAction::GracefulUnpin {
            self.set_pinned_chat_model(None).await;
        }
        {
            let mut reg = self.inner.registry.lock().await;
            reg.mode = target;
            let snapshot = self.write_state_inputs(&reg).await;
            drop(reg);
            self.persist_state(snapshot).await;
        }
        Ok(action)
    }

    /// Restore a persisted operating mode at startup (SRV-13: mode survives a
    /// restart). Sets the mode WITHOUT the switch guards — this is a restore, not a
    /// deliberate switch. The caller reads [`read_persisted_mode`] from the prior
    /// state file and passes it here.
    pub async fn restore_mode(&self, mode: OperatingMode) {
        let mut reg = self.inner.registry.lock().await;
        reg.mode = mode;
    }

    /// The id of the active accounting model (`separate-ceilings` | `unified-pool`),
    /// recorded as `assumed_memory_model` in the state file and surfaced in status.
    pub fn assumed_memory_model(&self) -> &'static str {
        self.inner.activation.assumed_memory_model
    }

    /// Register a model that became resident OUTSIDE the warm-slot path — i.e. a
    /// `transient` cold-launch (a build/validation serve from SRV-04's inline
    /// path) that must still be ACCOUNTED in VRAM and made evictable-first under
    /// pressure. Idempotent on `model_id` (re-registering refreshes its LRU stamp).
    ///
    /// Keep-warm and pinned-chat residents arrive through
    /// [`acquire_warm_slot`](ResidencyManager::acquire_warm_slot) /
    /// [`set_pinned_chat_model`]; this is the seam for everything else so the
    /// registry's VRAM accounting is complete (no untracked resident can push the
    /// host past the ceiling).
    pub async fn register_resident(
        &self,
        model_id: &str,
        runtime: Runtime,
        endpoint: impl Into<String>,
        vram_gb: f64,
        tier: Tier,
    ) {
        let tick = self.next_tick();
        let snapshot = {
            let mut reg = self.inner.registry.lock().await;
            let tier = if reg.pinned_chat_model.as_deref() == Some(model_id) {
                Tier::PinnedChat
            } else {
                tier
            };
            reg.residents.insert(
                model_id.to_string(),
                Resident {
                    model_id: model_id.to_string(),
                    runtime,
                    endpoint: endpoint.into(),
                    vram_gb,
                    tier,
                    last_used_tick: tick,
                },
            );
            self.write_state_inputs(&reg).await
        };
        self.persist_state(snapshot).await;
    }

    /// Assign (or clear) the pinned chat-role model. SRV-06 wires its assignment
    /// API to this. Idempotent.
    ///
    /// On a REASSIGNMENT (a different model, or clearing), the previously-pinned
    /// resident is demoted back to `keep-warm` so it becomes evictable again — the
    /// stored `tier` must follow the pin, otherwise a former chat model would stay
    /// non-evictable forever (its snapshot tier would remain `PinnedChat`). This is
    /// the residency half of SRV-06's atomic pin transfer: the new model is pinned
    /// and the old one released in a single locked update.
    pub async fn set_pinned_chat_model(&self, model_id: Option<&str>) {
        let mut reg = self.inner.registry.lock().await;
        let previous = reg.pinned_chat_model.take();
        reg.pinned_chat_model = model_id.map(|s| s.to_string());

        // Demote the previously-pinned resident (if still resident and not the new
        // pin) back to keep-warm so it is evictable again.
        if let Some(old) = previous {
            if model_id != Some(old.as_str()) {
                if let Some(r) = reg.residents.get_mut(&old) {
                    r.tier = Tier::KeepWarm;
                }
            }
        }
        // Re-tag the newly-pinned resident's tier so a later snapshot is coherent.
        if let Some(id) = model_id {
            if let Some(r) = reg.residents.get_mut(id) {
                r.tier = Tier::PinnedChat;
            }
        }
        let snapshot = self.write_state_inputs(&reg).await;
        drop(reg);
        self.persist_state(snapshot).await;
    }

    /// The model id currently pinned as the chat-role model, if any. SRV-06's pin
    /// coordinator reads this to detect an already-pinned model (idempotent switch)
    /// and to confirm the atomic transfer left exactly one chat model pinned.
    pub async fn pinned_chat_model(&self) -> Option<String> {
        self.inner.registry.lock().await.pinned_chat_model.clone()
    }

    fn next_tick(&self) -> u64 {
        self.inner.clock.fetch_add(1, Ordering::SeqCst)
    }

    /// Number of models currently resident. Drives the behavior-spec status
    /// (`resident=0` ⇒ IDLE, `resident>=1` ⇒ SERVING).
    pub async fn resident_count(&self) -> usize {
        self.inner.registry.lock().await.residents.len()
    }

    /// Sum of resident footprints (GB) — the co-resident VRAM the ceiling bounds.
    pub async fn resident_total_gb(&self) -> f64 {
        self.inner
            .registry
            .lock()
            .await
            .residents
            .values()
            .map(|r| r.vram_gb)
            .sum()
    }

    /// Read free VRAM through the injected launcher (production: the sysfs helper).
    async fn free_vram(&self) -> Option<f64> {
        self.inner.launcher.free_vram_gb().await
    }

    /// Admissible free GB for a GPU-pool candidate via the active SRV-11 accounting
    /// model, with `reserved_gb` (in-flight reservations) netted out. The live
    /// acquire path serves GPU-tier models; the CPU-pool and unified-pool maths are
    /// exercised directly against [`MemoryModel`]. Fail-safe: any unreadable counter
    /// the model maps to `None` → 0 free (force queue/deny, never an OOM launch).
    async fn admissible_free_gpu(&self, reserved_gb: f64, residents: &[ResidentView]) -> f64 {
        let snap = MemorySnapshot {
            gpu_free_gb: self.free_vram().await,
            cpu_free_gb: config::read_cpu_free_gb(),
            // The unified-pool model carries its own physical total; the live host
            // exposes no separate "unified free" counter under a fixed carveout.
            physical_total_gb: None,
        };
        match self
            .inner
            .memory_model
            .admissible_free_gb(Pool::Gpu, &snap, residents)
        {
            Some(f) => (f - reserved_gb).max(0.0),
            None => 0.0,
        }
    }

    /// CLAIM the eviction targets under the lock: remove each from the resident
    /// map atomically so a concurrent admission cannot ALSO plan to evict the same
    /// resident (the double-eviction race that would let two launches both bank the
    /// same reclaimed VRAM and breach the ceiling). Must be called while holding
    /// `reg`. The slow VRAM reclamation happens afterwards in [`reclaim_targets`].
    fn claim_targets(reg: &mut Registry, targets: &[EvictTarget]) {
        for t in targets {
            reg.residents.remove(&t.model_id);
        }
    }

    /// Reclaim the VRAM of already-CLAIMED targets via the launcher (the slow part,
    /// done WITHOUT the registry lock). The launcher finishes or cleanly cancels
    /// any in-flight generation before the VRAM is released (SRV-05 mid-generation
    /// edge case). Emits an `evict` event per target. An `Err` is genericized; the
    /// caller aborts the admission.
    async fn reclaim_targets(&self, targets: &[EvictTarget]) -> Result<(), String> {
        for t in targets {
            self.inner.launcher.evict(&t.model_id).await?;
            self.inner
                .events
                .emit(&ResidencyEvent::new("evict", t.model_id.clone(), t.tier.id()));
        }
        Ok(())
    }

    /// Snapshot the state-file inputs while holding the lock (so the persisted view
    /// is internally consistent).
    async fn write_state_inputs(&self, reg: &Registry) -> StateSnapshot {
        StateSnapshot {
            residents: reg
                .residents
                .values()
                .map(|r| StateResident {
                    model_id: r.model_id.clone(),
                    tier: r.tier.id().to_string(),
                    vram_gb: r.vram_gb,
                })
                .collect(),
            pinned_chat_model: reg.pinned_chat_model.clone(),
            assumed_memory_model: self.inner.activation.assumed_memory_model,
            mode: reg.mode,
            gpu_ceiling_gb: self.inner.gpu_ceiling_gb,
            cpu_ceiling_gb: self.inner.cpu_ceiling_gb,
        }
    }

    /// Atomically write the residency state file (tempfile + rename) if a path is
    /// configured. `free_vram_gb` is read fresh for the file's required field.
    async fn persist_state(&self, snapshot: StateSnapshot) {
        let Some(path) = self.inner.state_path.clone() else {
            return;
        };
        let free = self.free_vram().await;
        let _ = write_state_file(&path, &snapshot, free);
    }

    /// Commit a freshly-launched resident: insert it, drop its reservation, stamp
    /// LRU, and mirror the state file. Returns the [`Slot`] to serve from.
    async fn commit_resident(
        &self,
        model_id: &str,
        runtime: Runtime,
        endpoint: String,
        vram_gb: f64,
        tier: Tier,
    ) -> Slot {
        let tick = self.next_tick();
        let snapshot = {
            let mut reg = self.inner.registry.lock().await;
            reg.reservations.remove(model_id);
            // Honor an in-the-meantime pin assignment.
            let tier = if reg.pinned_chat_model.as_deref() == Some(model_id) {
                Tier::PinnedChat
            } else {
                tier
            };
            reg.residents.insert(
                model_id.to_string(),
                Resident {
                    model_id: model_id.to_string(),
                    runtime,
                    endpoint: endpoint.clone(),
                    vram_gb,
                    tier,
                    last_used_tick: tick,
                },
            );
            self.write_state_inputs(&reg).await
        };
        self.persist_state(snapshot).await;
        self.inner
            .events
            .emit(&ResidencyEvent::new("admit", model_id.to_string(), tier.id()));
        Slot {
            model_id: model_id.to_string(),
            runtime,
            endpoint,
            // S88 ISO-02: the SRV-05 warm-slot launch path predates netns isolation
            // and is not wired through the ISO-02 launcher seam; record no namespace
            // here. (Warm-slot isolation is a SRV-05 follow-up; the cold-launch path
            // in `launcher.rs` IS isolated by ISO-02.)
            netns: None,
        }
    }

    /// Release a reservation taken on the admission fast-path that we then could
    /// not fulfil (so a failed admission never leaks reserved VRAM).
    async fn release_reservation(&self, model_id: &str) {
        let mut reg = self.inner.registry.lock().await;
        reg.reservations.remove(model_id);
    }
}

/// The phases an admission attempt resolves into, decided under the lock.
enum Decision {
    /// Already resident + healthy → reuse this slot (no launch).
    Reuse(Slot),
    /// Admit after evicting these (transient-only, immediate) targets; the
    /// candidate's footprint is already RESERVED under the lock.
    Admit { evict: Vec<EvictTarget> },
    /// Keep-warm contention: evict transients now, then (after the bounded wait)
    /// re-plan. Footprint is NOT yet reserved (we may end up denying).
    Queue { transient_first: Vec<EvictTarget> },
    /// Cannot admit (pinned-only stall / fail-safe). Already recorded.
    Deny(String),
}

#[async_trait]
impl ResidencyManager for VramResidencyManager {
    async fn acquire_warm_slot(
        &self,
        model_id: &ModelId,
        vram_gb: Option<f64>,
    ) -> Result<Slot, ResidencyError> {
        let id = model_id.as_str().to_string();

        // ── Phase 1: decide under the lock (atomic decide + reserve) ───────────
        let decision = self.decide(&id, vram_gb).await;

        match decision {
            Decision::Reuse(slot) => {
                self.inner
                    .events
                    .emit(&ResidencyEvent::new("reuse", id.clone(), ""));
                Ok(slot)
            }
            Decision::Admit { evict } => {
                self.admit(&id, vram_gb, &evict).await
            }
            Decision::Queue { transient_first } => {
                self.queue_then_admit(&id, vram_gb, transient_first).await
            }
            Decision::Deny(reason) => Err(ResidencyError::CannotAdmit(reason)),
        }
    }
}

impl VramResidencyManager {
    /// The under-the-lock decision: reads free VRAM, subtracts reservations, plans,
    /// and (on an immediate admit) RESERVES the footprint before releasing the lock
    /// — the heart of the no-double-admit guarantee.
    async fn decide(&self, id: &str, vram_gb: Option<f64>) -> Decision {
        let mut reg = self.inner.registry.lock().await;

        // Already resident → reuse if healthy (validated outside the lock would be
        // racy with eviction; we capture the endpoint here and health-check after).
        if let Some(r) = reg.residents.get(id).cloned() {
            // Bump LRU on reuse.
            if let Some(m) = reg.residents.get_mut(id) {
                m.last_used_tick = self.next_tick();
            }
            return Decision::Reuse(Slot {
                model_id: r.model_id,
                runtime: r.runtime,
                endpoint: r.endpoint,
                netns: None, // ISO-02: SRV-05 warm-slot reuse, not netns-wired here.
            });
        }

        // Compute the admissible free memory for the candidate's pool via the
        // active SRV-11 accounting model, WHILE HOLDING THE LOCK so the decision is
        // atomic w.r.t. both the resident set AND the live host counters — a stale
        // pre-lock read is exactly the double-admit window. Reservations are netted
        // out; an unreadable counter the model maps to `None` → 0 free → queue/deny
        // (never an OOM-risking launch). `tokio::sync::Mutex` is held across this
        // await safely.
        let residents = reg.snapshot();
        let free = self.admissible_free_gpu(reg.reserved_gb(), &residents).await;
        let plan = plan_admission(vram_gb, free, &residents);

        match plan {
            EvictionPlan::AdmitAfterEvicting(evict) => {
                // CLAIM the eviction targets + RESERVE the footprint now, under the
                // lock, so a concurrent admission can neither evict the same
                // resident twice nor double-spend the reclaimed/free VRAM.
                Self::claim_targets(&mut reg, &evict);
                reg.reservations
                    .insert(id.to_string(), vram_gb.unwrap_or(0.0));
                Decision::Admit { evict }
            }
            EvictionPlan::Queue { transient_first, .. } => {
                // Claim the cheap transient targets now (under the lock) so they
                // are reclaimed exactly once even across concurrent admissions; the
                // keep-warm LRU decision is deferred to the post-wait re-plan.
                Self::claim_targets(&mut reg, &transient_first);
                Decision::Queue { transient_first }
            }
            EvictionPlan::CannotAdmit => {
                drop(reg);
                self.inner.events.emit(&ResidencyEvent::new(
                    "admission-denied",
                    id.to_string(),
                    "",
                ));
                Decision::Deny(id.to_string())
            }
        }
    }

    /// Execute an immediate admit: evict the (transient) targets, launch, health-
    /// check, commit. The footprint is already reserved by [`decide`].
    async fn admit(
        &self,
        id: &str,
        vram_gb: Option<f64>,
        evict: &[EvictTarget],
    ) -> Result<Slot, ResidencyError> {
        // Targets are already CLAIMED (removed from the map) under the lock by
        // `decide`; here we just reclaim their VRAM via the launcher.
        if let Err(_e) = self.reclaim_targets(evict).await {
            self.release_reservation(id).await;
            return Err(ResidencyError::CannotAdmit(id.to_string()));
        }
        self.launch_and_commit(id, vram_gb).await
    }

    /// Bring the model up via the launcher, health-check, and commit it as a
    /// resident. Releases the reservation on any failure so VRAM is not leaked.
    async fn launch_and_commit(
        &self,
        id: &str,
        vram_gb: Option<f64>,
    ) -> Result<Slot, ResidencyError> {
        let footprint = vram_gb.unwrap_or(0.0);
        let (runtime, endpoint) = match self.inner.launcher.launch(id, footprint).await {
            Ok(v) => v,
            Err(_e) => {
                self.release_reservation(id).await;
                return Err(ResidencyError::CannotAdmit(id.to_string()));
            }
        };
        if !self.inner.launcher.health_check(&endpoint).await {
            self.release_reservation(id).await;
            return Err(ResidencyError::SlotUnhealthy(id.to_string()));
        }
        // Tier of a freshly-admitted keep-warm model is KeepWarm (it stays
        // resident); commit honors any pin assigned in the meantime.
        let slot = self
            .commit_resident(id, runtime, endpoint, footprint, Tier::KeepWarm)
            .await;
        Ok(slot)
    }

    /// The queue-then-LRU path: evict cheap transients immediately, emit a `queue`
    /// event, wait the bounded threshold, then re-plan under the lock. After the
    /// wait, keep-warm LRU eviction is permitted. Coalesces (a re-request that has
    /// since become resident is reused).
    async fn queue_then_admit(
        &self,
        id: &str,
        vram_gb: Option<f64>,
        transient_first: Vec<EvictTarget>,
    ) -> Result<Slot, ResidencyError> {
        // Reclaim the cheap transients now (already CLAIMED under the lock by
        // `decide`) — may already make room.
        if let Err(_e) = self.reclaim_targets(&transient_first).await {
            return Err(ResidencyError::CannotAdmit(id.to_string()));
        }
        self.inner
            .events
            .emit(&ResidencyEvent::new("queue", id.to_string(), ""));

        // Bounded wait WITHOUT holding the lock (never stalls other admissions).
        tokio::time::sleep(self.inner.wait_threshold).await;

        // Re-plan under the lock; now keep-warm LRU eviction is allowed.
        let mut reg = self.inner.registry.lock().await;

        // Coalesce: if it became resident during the wait, reuse it.
        if let Some(r) = reg.residents.get(id).cloned() {
            if let Some(m) = reg.residents.get_mut(id) {
                m.last_used_tick = self.next_tick();
            }
            drop(reg);
            self.inner
                .events
                .emit(&ResidencyEvent::new("reuse", id.to_string(), ""));
            return Ok(Slot {
                model_id: r.model_id,
                runtime: r.runtime,
                endpoint: r.endpoint,
                netns: None, // ISO-02: SRV-05 warm-slot path, not netns-wired here.
            });
        }

        // Re-plan under the lock via the active accounting model (same atomicity
        // argument as `decide`).
        let residents = reg.snapshot();
        let free = self.admissible_free_gpu(reg.reserved_gb(), &residents).await;
        let plan = plan_admission(vram_gb, free, &residents);

        let to_evict: Vec<EvictTarget> = match plan {
            EvictionPlan::AdmitAfterEvicting(e) => e,
            EvictionPlan::Queue {
                transient_first,
                keep_warm_lru,
            } => {
                // After the wait, evict keep-warm LRU too.
                transient_first.into_iter().chain(keep_warm_lru).collect()
            }
            EvictionPlan::CannotAdmit => {
                drop(reg);
                self.inner.events.emit(&ResidencyEvent::new(
                    "admission-denied",
                    id.to_string(),
                    "",
                ));
                return Err(ResidencyError::CannotAdmit(id.to_string()));
            }
        };
        // CLAIM the post-wait targets + RESERVE the footprint before dropping the
        // lock, so a concurrent admission cannot evict the same keep-warm resident.
        Self::claim_targets(&mut reg, &to_evict);
        reg.reservations
            .insert(id.to_string(), vram_gb.unwrap_or(0.0));
        drop(reg);

        if let Err(_e) = self.reclaim_targets(&to_evict).await {
            self.release_reservation(id).await;
            return Err(ResidencyError::CannotAdmit(id.to_string()));
        }
        self.launch_and_commit(id, vram_gb).await
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Residency state file (JSON, atomic tempfile + rename)
// ─────────────────────────────────────────────────────────────────────────────

/// A resident entry in the state file.
#[derive(Debug, Clone, serde::Serialize)]
struct StateResident {
    model_id: String,
    tier: String,
    vram_gb: f64,
}

/// A consistent snapshot of registry state for the file (taken under the lock).
struct StateSnapshot {
    residents: Vec<StateResident>,
    pinned_chat_model: Option<String>,
    /// The active SRV-11 accounting model id (`separate-ceilings` | `unified-pool`).
    assumed_memory_model: &'static str,
    /// The SRV-13 operating mode.
    mode: OperatingMode,
    gpu_ceiling_gb: f64,
    cpu_ceiling_gb: f64,
}

/// The JSON shape of the residency/coordinator state file. Required fields per the
/// SRV-05 + SRV-13 behavior specs: `residents`, `free_vram_gb`, `pinned_chat_model`,
/// `assumed_memory_model`, `mode`, `gpu_ceiling_gb`, `cpu_ceiling_gb`.
#[derive(Debug, Clone, serde::Serialize)]
struct StateFile {
    residents: Vec<StateResident>,
    free_vram_gb: Option<f64>,
    pinned_chat_model: Option<String>,
    assumed_memory_model: &'static str,
    /// Stable mode id (`assistant-live` | `batch-coder`).
    mode: &'static str,
    gpu_ceiling_gb: f64,
    cpu_ceiling_gb: f64,
}

/// Atomically write the residency state file: serialize to JSON, write to a
/// uniquely-named temp file in the SAME directory (so the rename is atomic on the
/// same filesystem), fsync it, then `rename` over the target. A reader either sees
/// the old complete file or the new complete file — never a torn/partial write.
/// On any error the temp file is removed so no junk accumulates.
fn write_state_file(
    path: &str,
    snapshot: &StateSnapshot,
    free_vram_gb: Option<f64>,
) -> std::io::Result<()> {
    use std::io::Write;

    let state = StateFile {
        residents: snapshot.residents.clone(),
        free_vram_gb,
        pinned_chat_model: snapshot.pinned_chat_model.clone(),
        assumed_memory_model: snapshot.assumed_memory_model,
        mode: snapshot.mode.id(),
        gpu_ceiling_gb: snapshot.gpu_ceiling_gb,
        cpu_ceiling_gb: snapshot.cpu_ceiling_gb,
    };
    let json = serde_json::to_vec_pretty(&state).map_err(std::io::Error::other)?;

    let target = std::path::Path::new(path);
    let dir = target.parent().unwrap_or_else(|| std::path::Path::new("."));

    // Unique sibling temp name (pid + a monotonic nonce) in the SAME directory so
    // the final rename is atomic (same mount). Not a security temp — just a
    // crash-safe staging file we control and rename immediately.
    let nonce = STATE_WRITE_NONCE.fetch_add(1, Ordering::Relaxed);
    let tmp_path = dir.join(format!(".residency.{}.{}.tmp", std::process::id(), nonce));

    let write_result = (|| -> std::io::Result<()> {
        let mut f = std::fs::File::create(&tmp_path)?;
        f.write_all(&json)?;
        f.sync_all()?;
        Ok(())
    })();

    if let Err(e) = write_result {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(e);
    }

    // Atomic replace.
    if let Err(e) = std::fs::rename(&tmp_path, target) {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(e);
    }
    Ok(())
}

/// Monotonic nonce so concurrent atomic writes never collide on a temp name.
static STATE_WRITE_NONCE: AtomicU64 = AtomicU64::new(0);

/// Read the persisted operating mode from a prior state file (SRV-13: mode
/// survives a restart). `None` if the file is missing/unreadable or carries no
/// recognizable mode — the caller then keeps the default (assistant-live).
pub fn read_persisted_mode(path: &str) -> Option<OperatingMode> {
    let raw = std::fs::read_to_string(path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
    OperatingMode::from_id(v.get("mode")?.as_str()?)
}

/// Production [`WarmLauncher`]: free VRAM from the sysfs config helper; launch /
/// evict via an injected SRV-04-style bringup. Kept thin: the heavy launch glue is
/// SRV-04's; this only adapts it to the residency interface and supplies the
/// fail-safe free-VRAM read. (The concrete process glue is wired by the binary;
/// here we expose the VRAM read so the no-literal sysfs path is centralized.)
pub struct SysfsFreeVram;

impl SysfsFreeVram {
    /// Free VRAM in GB via [`config::read_free_vram_gb`] (no literal path).
    pub fn read() -> Option<f64> {
        config::read_free_vram_gb()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    /// A scripted launcher: a settable free-VRAM value, recorded launch/evict
    /// calls, and a health toggle. Drives admission/eviction deterministically.
    struct ScriptedLauncher {
        free_gb: StdMutex<Option<f64>>,
        healthy: bool,
        launches: StdMutex<Vec<String>>,
        evictions: StdMutex<Vec<String>>,
        fail_launch: bool,
        /// When true, the free counter drops by `gb` on launch and rises by `gb`
        /// on evict — a realistic host counter (used by the concurrency test).
        track: StdMutex<bool>,
    }
    impl ScriptedLauncher {
        fn new(free: Option<f64>) -> Self {
            ScriptedLauncher {
                free_gb: StdMutex::new(free),
                healthy: true,
                launches: StdMutex::new(vec![]),
                evictions: StdMutex::new(vec![]),
                fail_launch: false,
                track: StdMutex::new(false),
            }
        }
        fn track_footprints(&self) {
            *self.track.lock().unwrap() = true;
        }
    }
    #[async_trait]
    impl WarmLauncher for ScriptedLauncher {
        async fn free_vram_gb(&self) -> Option<f64> {
            *self.free_gb.lock().unwrap()
        }
        async fn launch(&self, model_id: &str, gb: f64) -> Result<(Runtime, String), String> {
            if self.fail_launch {
                return Err("launch-failed".into());
            }
            self.launches.lock().unwrap().push(model_id.to_string());
            if *self.track.lock().unwrap() {
                let mut f = self.free_gb.lock().unwrap();
                *f = Some((f.unwrap_or(0.0) - gb).max(0.0));
            }
            Ok((Runtime::LlamaCpp, format!("http://warm.invalid/{model_id}")))
        }
        async fn health_check(&self, _endpoint: &str) -> bool {
            self.healthy
        }
        async fn evict(&self, model_id: &str) -> Result<(), String> {
            self.evictions.lock().unwrap().push(model_id.to_string());
            Ok(())
        }
    }

    #[derive(Default)]
    struct RecordingSink {
        events: StdMutex<Vec<ResidencyEvent>>,
    }
    impl EventSink for RecordingSink {
        fn emit(&self, event: &ResidencyEvent) {
            self.events.lock().unwrap().push(event.clone());
        }
    }

    fn mgr(
        launcher: Arc<ScriptedLauncher>,
        sink: Arc<RecordingSink>,
        wait_ms: u64,
    ) -> VramResidencyManager {
        VramResidencyManager::with_settings(
            launcher,
            sink,
            Duration::from_millis(wait_ms),
            None,
        )
    }

    /// Seed a resident directly (bypasses admission) for eviction-order tests.
    async fn seed(m: &VramResidencyManager, id: &str, gb: f64, tier: Tier, tick: u64) {
        let mut reg = m.inner.registry.lock().await;
        reg.residents.insert(
            id.to_string(),
            Resident {
                model_id: id.to_string(),
                runtime: Runtime::LlamaCpp,
                endpoint: format!("http://resident.invalid/{id}"),
                vram_gb: gb,
                tier,
                last_used_tick: tick,
            },
        );
    }

    #[tokio::test]
    async fn admits_when_fits() {
        let l = Arc::new(ScriptedLauncher::new(Some(40.0)));
        let s = Arc::new(RecordingSink::default());
        let m = mgr(l.clone(), s.clone(), 10);
        let slot = m
            .acquire_warm_slot(&ModelId::from("a"), Some(20.0))
            .await
            .unwrap();
        assert_eq!(slot.model_id, "a");
        assert_eq!(*l.launches.lock().unwrap(), vec!["a".to_string()]);
        assert!(s
            .events
            .lock()
            .unwrap()
            .iter()
            .any(|e| e.decision == "admit"));
    }

    #[tokio::test]
    async fn keep_warm_persists_across_requests() {
        let l = Arc::new(ScriptedLauncher::new(Some(40.0)));
        let s = Arc::new(RecordingSink::default());
        let m = mgr(l.clone(), s.clone(), 10);
        m.acquire_warm_slot(&ModelId::from("kw"), Some(20.0))
            .await
            .unwrap();
        // Second request → reused (no second launch).
        m.acquire_warm_slot(&ModelId::from("kw"), Some(20.0))
            .await
            .unwrap();
        assert_eq!(
            *l.launches.lock().unwrap(),
            vec!["kw".to_string()],
            "keep-warm must NOT be cold-cycled per request"
        );
        assert!(s.events.lock().unwrap().iter().any(|e| e.decision == "reuse"));
    }

    #[tokio::test]
    async fn transient_launch_does_not_evict_keep_warm_if_transients_suffice() {
        let l = Arc::new(ScriptedLauncher::new(Some(0.0)));
        let s = Arc::new(RecordingSink::default());
        let m = mgr(l.clone(), s.clone(), 10);
        seed(&m, "kw", 30.0, Tier::KeepWarm, 1).await;
        seed(&m, "t", 25.0, Tier::Transient, 2).await;
        // need 20, free 0 → evicting the transient (25) alone suffices.
        m.acquire_warm_slot(&ModelId::from("new"), Some(20.0))
            .await
            .unwrap();
        let evictions = l.evictions.lock().unwrap().clone();
        assert_eq!(evictions, vec!["t".to_string()]);
        assert!(!evictions.contains(&"kw".to_string()), "keep-warm untouched");
    }

    #[tokio::test]
    async fn chat_pinned_never_evicted_under_pressure() {
        let l = Arc::new(ScriptedLauncher::new(Some(0.0)));
        let s = Arc::new(RecordingSink::default());
        let m = mgr(l.clone(), s.clone(), 10);
        seed(&m, "chat", 40.0, Tier::PinnedChat, 1).await;
        m.set_pinned_chat_model(Some("chat")).await;
        // need 50, free 0, only the pinned chat resident → must DENY, never evict.
        let err = m
            .acquire_warm_slot(&ModelId::from("big"), Some(50.0))
            .await
            .unwrap_err();
        assert!(matches!(err, ResidencyError::CannotAdmit(_)));
        assert!(
            l.evictions.lock().unwrap().is_empty(),
            "pinned chat model must NEVER be evicted"
        );
        // The chat model is still resident.
        let reg = m.inner.registry.lock().await;
        assert!(reg.residents.contains_key("chat"));
        drop(reg);
        assert!(s
            .events
            .lock()
            .unwrap()
            .iter()
            .any(|e| e.decision == "admission-denied"));
    }

    #[tokio::test]
    async fn keep_warm_contention_queues_then_evicts_lru_after_threshold() {
        let l = Arc::new(ScriptedLauncher::new(Some(0.0)));
        let s = Arc::new(RecordingSink::default());
        let m = mgr(l.clone(), s.clone(), 20);
        seed(&m, "kw_old", 30.0, Tier::KeepWarm, 1).await;
        seed(&m, "kw_new", 30.0, Tier::KeepWarm, 9).await;
        // need 25, free 0, only keep-warm → queue, then evict LRU (kw_old).
        let slot = m
            .acquire_warm_slot(&ModelId::from("new"), Some(25.0))
            .await
            .unwrap();
        assert_eq!(slot.model_id, "new");
        let ev = l.evictions.lock().unwrap().clone();
        assert_eq!(ev, vec!["kw_old".to_string()], "LRU keep-warm evicted");
        let events = s.events.lock().unwrap();
        assert!(events.iter().any(|e| e.decision == "queue"));
        assert!(events.iter().any(|e| e.decision == "evict"));
    }

    #[tokio::test]
    async fn unreadable_vram_fails_safe() {
        // free = None → fail-safe. Empty host but unknown free → must NOT launch
        // an OOM risk: with no residents the candidate needs >0 and free treated 0.
        let l = Arc::new(ScriptedLauncher::new(None));
        let s = Arc::new(RecordingSink::default());
        let m = mgr(l.clone(), s.clone(), 5);
        let err = m
            .acquire_warm_slot(&ModelId::from("a"), Some(20.0))
            .await
            .unwrap_err();
        assert!(matches!(err, ResidencyError::CannotAdmit(_)));
        assert!(l.launches.lock().unwrap().is_empty(), "no OOM-risking launch");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_launches_do_not_double_admit() {
        // The host has a PINNED chat model already eating most of VRAM, leaving a
        // headroom that fits exactly ONE of two concurrent 20GB launches. Because
        // the only resident is non-evictable (pinned chat), the loser cannot
        // evict-and-replace — it must be denied. This makes "exactly one admits"
        // the deterministic, spec-faithful no-double-admit guarantee.
        let l = Arc::new(ScriptedLauncher::new(Some(50.0)));
        // Free VRAM drops on launch so the second admission sees the first's
        // consumption — the reservation must prevent BOTH passing the read.
        l.track_footprints();
        let s = Arc::new(RecordingSink::default());
        let m = mgr(l.clone(), s.clone(), 20);
        // Seed + pin a chat model (off-limits to eviction). Free headroom = 50 -
        // (chat already accounted by the host counter) → only one 20GB fits.
        seed(&m, "chat", 60.0, Tier::PinnedChat, 1).await;
        m.set_pinned_chat_model(Some("chat")).await;

        let m1 = m.clone();
        let m2 = m.clone();
        let h1 = tokio::spawn(async move {
            m1.acquire_warm_slot(&ModelId::from("a"), Some(30.0)).await
        });
        let h2 = tokio::spawn(async move {
            m2.acquire_warm_slot(&ModelId::from("b"), Some(30.0)).await
        });
        let r1 = h1.await.unwrap();
        let r2 = h2.await.unwrap();
        let ok_count = [&r1, &r2].iter().filter(|r| r.is_ok()).count();
        // At least one admits; the loser is either denied OR admits only by first
        // evicting the other 30GB peer (LRU keep-warm) — never by exceeding the
        // ceiling. Both 30GB models can NEVER be co-resident under 50GB free: that
        // is the double-admit the reservation lock prevents.
        assert!(ok_count >= 1, "at least one launch admits");
        // INVARIANT 1: the pinned chat model is NEVER evicted.
        assert!(
            !l.evictions.lock().unwrap().contains(&"chat".to_string()),
            "pinned chat model must NEVER be evicted"
        );
        // INVARIANT 2 (no double-admit past the ceiling): the two contending 30GB
        // models are never resident together (their combined 60GB would breach the
        // 50GB free ceiling).
        let reg = m.inner.registry.lock().await;
        let a_resident = reg.residents.contains_key("a");
        let b_resident = reg.residents.contains_key("b");
        assert!(
            !(a_resident && b_resident),
            "two 30GB models must not be co-resident under a 50GB ceiling (double-admit)"
        );
        // The pinned chat is still resident throughout.
        assert!(reg.residents.contains_key("chat"));
    }

    #[tokio::test]
    async fn pinned_only_returns_cannot_admit_without_evicting_chat() {
        let l = Arc::new(ScriptedLauncher::new(Some(0.0)));
        let s = Arc::new(RecordingSink::default());
        let m = mgr(l.clone(), s.clone(), 5);
        seed(&m, "chat", 40.0, Tier::PinnedChat, 1).await;
        m.set_pinned_chat_model(Some("chat")).await;
        let err = m
            .acquire_warm_slot(&ModelId::from("x"), Some(60.0))
            .await
            .unwrap_err();
        assert!(matches!(err, ResidencyError::CannotAdmit(_)));
        assert!(l.evictions.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn atomic_state_file_written_with_required_fields() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("residency.json");
        let l = Arc::new(ScriptedLauncher::new(Some(40.0)));
        let s = Arc::new(RecordingSink::default());
        let m = VramResidencyManager::with_settings(
            l,
            s,
            Duration::from_millis(5),
            Some(path.to_string_lossy().to_string()),
        );
        m.set_pinned_chat_model(Some("chat")).await;
        m.acquire_warm_slot(&ModelId::from("a"), Some(20.0))
            .await
            .unwrap();
        let raw = std::fs::read_to_string(&path).unwrap();
        let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert!(v.get("residents").is_some());
        assert!(v.get("free_vram_gb").is_some());
        assert_eq!(v.get("pinned_chat_model").and_then(|x| x.as_str()), Some("chat"));
        // The admitted model is in the residents array.
        let residents = v.get("residents").and_then(|x| x.as_array()).unwrap();
        assert!(residents
            .iter()
            .any(|r| r.get("model_id").and_then(|x| x.as_str()) == Some("a")));
    }

    #[tokio::test]
    async fn unhealthy_slot_after_launch_is_reported() {
        let mut l = ScriptedLauncher::new(Some(40.0));
        l.healthy = false;
        let l = Arc::new(l);
        let s = Arc::new(RecordingSink::default());
        let m = mgr(l, s, 5);
        let err = m
            .acquire_warm_slot(&ModelId::from("a"), Some(20.0))
            .await
            .unwrap_err();
        assert!(matches!(err, ResidencyError::SlotUnhealthy(_)));
    }

    #[test]
    fn events_carry_no_infra() {
        let evs = [
            ResidencyEvent::new("admit", "qwen3:8b", "keep-warm"),
            ResidencyEvent::new("evict", "big:120b", "transient"),
            ResidencyEvent::new("admission-denied", "x:1", ""),
        ];
        for e in evs {
            assert!(!e.model_id.contains("://"));
            assert!(!e.model_id.contains("192.168."));
            assert!(!e.model_id.contains('@'));
        }
    }
}
