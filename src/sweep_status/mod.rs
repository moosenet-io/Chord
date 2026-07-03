//! Sweep-status observability: "is the fleet's model-benchmarking work
//! healthy right now" for `intake-coder-sweep.service` and
//! `intake-assistant-sweep.service`.
//!
//! Built after a real 7-hour silent jam (GPU pegged 99% busy, zero new DB
//! rows) caused by Ollama's runner subprocess wedging mid-`generate` on a
//! gfx1151 GPU-MoE workload — a known failure mode a host-level watchdog now
//! auto-restarts `ollama.service` for, but which had no observability surface
//! for a human or another system (Harmony/Lumina) to check without SSHing in
//! and running ad hoc SQL against the correlated signals.
//!
//! ## Layout
//! - [`verdict`] — pure `working`/`stuck`/`idle` judgment logic (heavily
//!   unit-tested; this is the actual decision this subsystem exists to make).
//! - [`config`] — env-sourced configuration (poll interval, thresholds, log
//!   path, service names, DB URL resolution).
//! - [`db`] — Postgres polling of `code_profile_runs` (coder sweep) and
//!   `assistant_profile_run` (assistant sweep).
//! - [`ollama`] — local `GET /api/ps` polling (currently loaded model(s)).
//! - [`gpu`] — `/sys/class/drm/card*/device/gpu_busy_percent` reader.
//! - [`systemd`] — `systemctl is-active` check via `tokio::process::Command`.
//! - [`snapshot`] — the JSON shape written to the log / served over HTTP.
//! - [`log`] — daily-rotated JSONL log + retention (see `log`'s module docs
//!   for the rotation-vs-single-file-trim design decision).
//! - [`poll`] — the background tick loop + process-global latest-snapshot
//!   (mirrors `crate::snap`'s `SHARED_STATE` pattern — additive, no `AppState`
//!   change needed).
//! - [`api`] — the two HTTP endpoints, `GET /v1/sweep/status` and
//!   `GET /v1/sweep/status/history`.
//!
//! ## Resilience
//! Every external call (Postgres, Ollama HTTP, `systemctl`, sysfs reads) is
//! wrapped so a failure there degrades to an `unavailable`/`None` marker in
//! the snapshot with a `tracing::warn!` — never a panic, never a crashed
//! poll tick, never a stopped monitor loop.

pub mod api;
pub mod config;
pub mod db;
pub mod gpu;
pub mod log;
pub mod ollama;
pub mod poll;
pub mod snapshot;
pub mod systemd;
pub mod verdict;
