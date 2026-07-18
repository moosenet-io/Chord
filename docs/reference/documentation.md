## Documentation

Component explainers (written from the source in [`src/`](src)) live in
[`docs/`](docs/):

- **[docs/architecture.md](docs/architecture.md)** — component deep-dive: request
  flow, Routing, backend tiers, model registry & storage tiering,
  memory/residency management, the control API, the agentic loop, and the search
  harness, each mapped to its real module/types.
- **[docs/serving.md](docs/serving.md)** — the serving / coordinator subsystem
  (Memory Coordinator, Clean-Swap Launcher, Mode Controller) mapped to the code
  that actually ships, with a present/partial/absent table.
- **[docs/egress.md](docs/egress.md)** — the runtime isolation model: ISO-01
  env-scrub + egress policy and ISO-02 per-runtime network-namespace enforcement,
  the fail-closed posture, and the honest scope boundaries.
- **[docs/snap-persistence.md](docs/snap-persistence.md)** — the optional,
  default-off SNAP → Postgres persistence layer (`CHORD_SNAP_PERSIST`).

**Sweep action-queue cache (RESIL-02, `CHORD_STATE_DIR`).** Chord caches a
sweep's planned action queue + progress cursor so a restarted sweep can resume
from Chord rather than replanning. Three JWT-gated control endpoints:
`POST /api/sweep/session` (register/upsert a queue — idempotent; same queue is a
no-op preserving progress, a different queue replaces it and resets progress),
`GET /api/sweep/session/:id` (remaining keys in queue order + counts; 404
unknown), and `POST /api/sweep/session/:id/advance` (mark keys done — append-only,
idempotent, keys not in the queue ignored). Chord only RECORDS and SERVES the
queue — it never executes it; the Terminus sweep is the executor. Backed by
[`src/sweep_session.rs`](src/sweep_session.rs), persisted to
`<CHORD_STATE_DIR>/sweep_sessions.json` (atomic tempfile+rename) when configured;
unset ⇒ in-memory only (lost on restart). Best-effort, corrupt-tolerant.

**GPU-exclusive lease durability (`CHORD_STATE_DIR`).** The GPU-exclusive lock
([`src/gpu_exclusive.rs`](src/gpu_exclusive.rs)) that hands the single host GPU to
the intake sweep is otherwise in-memory only — a Chord restart mid-sweep would drop
the lease and let a competing job slip in ("CHORD LOCK GAP DETECTED" on the harness
side). When `CHORD_STATE_DIR` is set, Chord persists the lease
(`<CHORD_STATE_DIR>/gpu_exclusive_lease.json`, atomic tempfile+rename) on every
acquire/heartbeat/release and reloads it on startup, honoring the TTL
(`CHORD_GPU_EXCLUSIVE_TTL_SECS`) so an already-abandoned lease never relocks the GPU.
Persistence is best-effort: a missing/corrupt/unwritable file never panics Chord —
it degrades to in-memory-only and logs at warn. Unset ⇒ persistence disabled (the
prior behavior). See also the sweep's Chord-backed resume in `moosenet/Terminus`.
- **[docs/test-results.md](docs/test-results.md)** — the S86 coder-fleet sweep
  results: themed BLITZ vs MULTI-FILE pass-rate charts, leaderboard, table, and
  takeaways.
- **[docs/model-testing-methodology.md](docs/model-testing-methodology.md)**
  — the full model benchmarking methodology (coder + assistant harnesses,
  scoring, judge panel, YaRN collapse detection, `mem_config` hardware
  tagging, gfx1151 backend quirks). The harness code lives in
  [`moosenet/Terminus`](../Terminus)`/src/intake/`, not in this repo.
- **[docs/assistant-results.md](docs/assistant-results.md)** — the S84 ASMT
  assistant-fleet sweep results, generated from Postgres, partial/in-progress.
- **[docs/contributing-results.md](docs/contributing-results.md)** — how to
  benchmark your own hardware and tag results so they aren't blended with
  the numbers here.
- **[docs/README.md](docs/README.md)** — the docs index.

