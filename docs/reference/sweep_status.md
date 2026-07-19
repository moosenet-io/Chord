# sweep_status

Sweep-status observability (132 KG nodes, `src/sweep_status/`): "is the fleet's
model-benchmarking work healthy right now" for the intake coder/assistant sweep
services. Built after a real 7-hour silent jam — GPU pegged at 99% busy, zero
new DB rows — caused by an Ollama runner subprocess wedging mid-`generate` on a
GPU-MoE workload. This subsystem gives that failure signature an observable
surface so a human or another system can check without SSHing in and running ad
hoc SQL.

## Key types and functions

| Symbol | Kind | File | What it does |
|---|---|---|---|
| `sweep_status::verdict` | module | `src/sweep_status/verdict.rs` | The pure `working` / `stuck` / `idle` judgment — the decision this subsystem exists to make (heavily unit-tested) |
| `sweep_status::config::SweepMonitorConfig::from_env` | function | `src/sweep_status/config.rs` | Poll interval, thresholds, log path, service names, DB URL resolution |
| `sweep_status::db::SweepDbStats` | struct | `src/sweep_status/db.rs` | Postgres polling of `code_profile_runs` (coder) and `assistant_profile_run` (assistant); `unavailable()` marks a failed probe |
| `sweep_status::ollama::parse_ps_models` | function | `src/sweep_status/ollama.rs` | Parses the local `GET /api/ps` response (currently loaded models) |
| `sweep_status::gpu::read_gpu_busy_percent` | function | `src/sweep_status/gpu.rs` | Reads `/sys/class/drm/card*/device/gpu_busy_percent` |
| `sweep_status::systemd` | module | `src/sweep_status/systemd.rs` | `systemctl is-active` check via `tokio::process::Command` |
| `sweep_status::snapshot` | module | `src/sweep_status/snapshot.rs` | The JSON shape written to the log and served over HTTP |
| `sweep_status::log::SweepStatusLog` | struct | `src/sweep_status/log.rs` | Daily-rotated JSONL log with retention |
| `sweep_status::poll::spawn` | function | `src/sweep_status/poll.rs` | The background tick loop + process-global latest snapshot (mirrors `snap`'s shared-state pattern — no `AppState` change) |
| `sweep_status::api::sweep_status_routes` | function | `src/sweep_status/api.rs` | `GET /v1/sweep/status` and `GET /v1/sweep/status/history` |

## How it connects

`main.rs` spawns the poller at startup; **routes** merges the two endpoints
into the proxy-port router. They are unauthenticated by the same bar as
`/health` and `/v1/audit/summary` — aggregate health only, no identities or
secrets. The verdict correlates four independent signals (GPU busy, fresh DB
rows, loaded models, unit active) so the wedge signature — GPU pegged, service
active, no fresh rows — is distinguishable from both honest work and idleness.
Related but distinct: the control API's sweep *session* cache
(`/api/sweep/session*`, RESIL-02 in `src/sweep_session.rs`) stores a sweep's
action queue for durable resume; this subsystem only observes health.

## Configuration

`CHORD_SWEEP_POLL_INTERVAL_SECS`, `CHORD_SWEEP_STUCK_AGE_SECS`,
`CHORD_SWEEP_GPU_BUSY_THRESHOLD`, `CHORD_SWEEP_STARTUP_GRACE_SECS`,
`CHORD_SWEEP_CODER_SERVICE`, `CHORD_SWEEP_ASSISTANT_SERVICE`,
`CHORD_SWEEP_OLLAMA_URL`, `CHORD_SWEEP_DRM_ROOT`, `CHORD_SWEEP_STATUS_LOG`,
`CHORD_SWEEP_RETENTION_DAYS` (DB URL resolution follows the intake-DB
convention).

## Notes and gaps

- Every external call (Postgres, Ollama HTTP, `systemctl`, sysfs) degrades to
  an `unavailable`/`None` marker with a warning — never a panic, never a
  stopped monitor loop.
- An unconfigured intake DB degrades to `db_configured: false` snapshots at
  startup rather than blocking anything.
- This subsystem observes; it does not restart anything — the auto-restart of a
  wedged runner is a host-level watchdog outside this repo.
