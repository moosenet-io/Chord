## Idle mode — free the host for the compiler (BLD-09)

The constellation CI/CD compiler (S117) builds on the heavy GPU/big-RAM host. To
hand that host to a build **without taking Chord down**, the compiler asks Chord to
go *idle*: Chord drains in-flight inference, stops its on-demand backends, unloads
every resident model from VRAM (demoting them back to warm on-disk storage so the
system RAM/VRAM they held is freed), records a resume manifest, and enters a
low-footprint wait — then restores full service on request. The admin surface lives
on the **control port** (`CHORD_CONTROL_PORT`, default `8090`) and is JWT-gated with
the same auth as every other endpoint. Implemented in
[`src/admin/idle.rs`](src/admin/idle.rs).

Internally this is a **closed-world transition state machine**
(`Active → EnteringIdle → Idle → Activating → Active`), not a snapshot-then-act flag,
so it is correct under concurrent control calls and concurrent inference. The
`EnteringIdle`/`Activating` markers are installed atomically (compare-and-swap) **before**
any release/restore work, and new inference is admitted only while `Active` — with the
admission counter incremented under the same lock that flips the state. Once idle-mode
flips to `EnteringIdle`, no further request can join the in-flight set, so the drain
that follows is a genuine closed-world drain.

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/admin/idle` | Enter idle: drain, stop providers, release GPU/VRAM, demote resident models, free RAM. Reports freed RAM. Idempotent. |
| `POST` | `/admin/activate` | Restore full service. Idempotent. |
| `GET` | `/admin/idle` | Current `idle`/`active` status + resume manifest + in-flight count. |

**Contract**

- **`POST /admin/idle`** (optional body `{"reason":"compiler"}`) drains in-flight
  requests (bounded by `CHORD_IDLE_DRAIN_SECS`, default 30s — an overrunning request
  is left to finish and flagged in the response), stops every on-demand inference
  backend, evicts all Ollama-resident models from VRAM, demotes any `Hot` registry
  records to `Warm`, and reports freed RAM measured as the `MemAvailable` delta from
  `/proc/meminfo`. Response: `{"status":"idle","changed":true,"freed":{"mem_available_before_gb":…,"mem_available_after_gb":…,"freed_gb":…,"backends_stopped":…,"models_unloaded":…,"models_demoted":…,"inflight_remaining":0,"foreign_gpu_lock_holder":null},…}`.
- **`POST /admin/activate`** clears idle and resumes serving. Models reload **lazily**
  on their next request (Ollama/llama-server cold-load on demand), so activation is
  cheap and non-blocking. Activation also happens **automatically on the first real
  inference request** after idle (lazy-on-request) — the compiler never needs to call
  `/admin/activate` explicitly if traffic resumes on its own — **except** while a
  compiler build lease is still held (see below).
- **Idempotency**: idle-while-idle and activate-while-active are `200` no-ops
  (`changed:false`); a repeat `/admin/idle` never re-runs release or clobbers the
  original manifest. Concurrent `/admin/idle` calls are safe — exactly one runs the
  release side effects; the rest return `changed:false`.
- **Requests during a transition**: an inference request that arrives while idle-mode
  is mid-transition (`EnteringIdle`/`Activating`) gets a short, retryable `503`
  (`{"error":"idle_transition_in_progress"}`, `Retry-After: 2`) rather than being
  admitted into the draining set.
- **Compiler-lease-aware lazy restore**: a real request that arrives while idle
  restores service **only if no compiler build lease is currently held**. If a compiler
  build lease *is* held, the request is shed with a retryable `503`
  (`{"error":"idle_compiler_build_active"}`, `Retry-After: 5`) and the idle manifest +
  watchdog protection are **preserved** — stray traffic can never tear down an active
  build window. A compiler lease is any GPU-exclusive holder whose label matches
  `CHORD_IDLE_COMPILER_LEASE_HOLDERS` (default substrings `compiler,build,bld`); other
  GPU jobs (e.g. the intake sweep harness) are **not** treated as build leases.
- **Watchdog / fail-safe**: a background loop re-activates an idle proxy once the
  watchdog deadline passes (`CHORD_IDLE_WATCHDOG_SECS`, default 3600s) **unless** a
  *compiler build lease* (per `CHORD_IDLE_COMPILER_LEASE_HOLDERS`) is still held — so a
  legitimately long build keeps Chord idle as long as it holds the GPU, but a
  crashed/forgotten compiler (or a stale idle state reloaded after a Chord restart)
  never leaves the proxy silently dead. A non-compiler GPU holder does not extend the
  idle window.
- **Cancellation-safe, budget-bounded transition**: entering idle spans several
  `.await` points (drain, VRAM eviction, tier demote). The `EnteringIdle` phase is held
  by an RAII guard, so if the enter request is cancelled (client disconnect), panics, or
  returns early before completion, the phase deterministically rolls back to `Active` —
  a cancelled enter can never wedge the proxy in `EnteringIdle` (which would 503 all
  inference and block admin enter/activate). The **entire release sequence is bounded by
  `CHORD_IDLE_RELEASE_BUDGET_SECS`** (default 90s): if it overruns, the enter self-aborts
  and the guard rolls back to `Active`. Admission is **closed for the whole
  `EnteringIdle` window** and reopens only after that consistent rollback — never with
  half-stopped backends still admitting. As a defense-in-depth backstop, the watchdog
  force-resolves any transient phase stuck longer than `CHORD_IDLE_STALE_TRANSITION_SECS`
  (default 300s) back to `Active`. **Invariant (clamped at runtime): the stale-recovery
  threshold is always strictly greater than the release budget**, so stale-recovery can
  only ever fire after the release future is already gone — never concurrently with live
  release.
- **Activate persist is hard-gated**: when a state path is configured, `POST /admin/activate`
  (and lazy restore) clears the on-disk idle manifest to `Active` **before** any restore
  work; if that persist fails, the activate is aborted (the proxy stays `Idle`,
  recoverable) and a retryable `503 idle_activate_persist_failed` is returned — it never
  proceeds into restore while the disk still says `Idle`. With no state path configured,
  persistence is a no-op as before.
- **Streaming in-flight accounting**: for streaming inference (chat/completions SSE or
  JSON pass-through, agent SSE), the in-flight guard lives until the streamed body is
  fully consumed / the executor task finishes — not merely until the handler returns
  the `Response` — so `POST /admin/idle` never observes a false "drained" state while a
  response is still streaming.
- **Durability**: when `CHORD_STATE_DIR` is set, the resume manifest is persisted
  (atomic tempfile+rename) so a crash mid-idle leaves a record the watchdog acts on
  after restart. Transient transition markers (`EnteringIdle`/`Activating`) are never
  persisted — a crash mid-transition reloads as `Active` (the GPU-exclusive gate and
  watchdog keep that safe). Unset ⇒ in-memory only (the watchdog still bounds it).
- **GPU lock held by another job**: reported in `foreign_gpu_lock_holder`, **never**
  force-released or killed — that lease may be a legitimate external GPU job.

`/v1/embeddings` also participates in this admission/drain path (its local-first path
dispatches to a GPU-resident embedding model), exactly like `/v1/chat/completions`,
`/v1/agent/execute`, and `/v1/infer`.

| Env var | Purpose | Default |
|---|---|---|
| `CHORD_IDLE_DRAIN_SECS` | Max seconds to wait for in-flight inference to drain before releasing. | `30` |
| `CHORD_IDLE_RELEASE_BUDGET_SECS` | Hard budget for the entire `enter_idle` release sequence; on overrun the enter self-aborts and rolls back to `Active` (admission closed throughout). Must be, and is clamped to be, strictly less than the stale threshold. | `90` |
| `CHORD_IDLE_WATCHDOG_SECS` | Hard timeout after which the watchdog auto-activates (unless a compiler build lease is held). | `3600` |
| `CHORD_IDLE_STALE_TRANSITION_SECS` | Backstop: force-resolve a controller stuck in a transient `EnteringIdle`/`Activating` phase back to `Active` after this many seconds (behind the RAII rollback guard). Clamped up if set ≤ the release budget. | `300` |
| `CHORD_IDLE_COMPILER_LEASE_HOLDERS` | Comma-separated, case-insensitive substrings that mark a GPU-exclusive holder as a *compiler build* lease (protects the idle window from lazy teardown + watchdog). Role labels, not infra identifiers. | `compiler,build,bld` |
| `CHORD_STATE_DIR` | Durable state dir; the idle manifest persists to `<dir>/admin_idle_state.json`. | *(unset → in-memory)* |
| `OLLAMA_URL` | Ollama base used to enumerate/unload resident models on idle. | *(unset → VRAM eviction skipped)* |

No infrastructure hosts/ports are hardcoded — every value above is env-sourced, and
no secret is read or logged on the idle/activate path.

