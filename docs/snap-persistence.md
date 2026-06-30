# Chord — SNAP → Postgres Persistence (1.3.1)

Chord's **SNAP** observability subsystem (`src/snap/`) collects four streams:
request analytics, model inventory, passive activity, and VRAM usage. As of
**chord-proxy 1.3.1** these streams can optionally be persisted to Postgres,
using the SAME conventions the model-intake harnesses use (`sqlx 0.8`, a shared
`PgPool`, one row per event, idempotent `CREATE TABLE IF NOT EXISTS` DDL applied
in-code).

This is **additive and default-OFF**. With persistence disabled, SNAP behaves
exactly as 1.2.0 / 1.3.0 — in-memory state and the JSONL request log only, with
no DB pool opened and zero behavior change.

## The toggle: `CHORD_SNAP_PERSIST`

A single environment flag controls all SNAP DB writes.

| `CHORD_SNAP_PERSIST` | Behavior |
|----------------------|----------|
| unset / `false` / `0` / `off` (default) | In-memory / JSONL only. No pool, no DDL, no writes. |
| `true` / `1` / `yes` / `on` | SNAP opens one shared pool at startup, runs `migrate`, and persists each stream. |

If the flag is on but no database URL resolves, persistence **silently disables**
(logged once at startup) and SNAP runs in-memory — it never crashes the proxy.

## Database: reuses the intake pool (no new secret)

When enabled, SNAP **reuses the existing intake database** — the same one the
serving profile (`DbProfileSource`) and the model-intake harnesses use. The URL
is resolved via `terminus_rs::config::intake_database_url()`, which prefers
`INTAKE_DATABASE_URL` and falls back to `DATABASE_URL`. There is **no new secret,
no separate DSN, and no hardcoded host** anywhere in the SNAP code. One pool is
opened once at startup and shared.

## Tables (created in-code at startup)

`snap::storage::migrate(pool)` issues `CREATE TABLE IF NOT EXISTS` + indexes for
five tables (idempotent, safe to re-run; no `migrations/` directory):

| Table | Stream | Grain | Written by |
|-------|--------|-------|-----------|
| `snap_request_log` | SNAP-05 request analytics | one row per completed request | `RequestLogger::append` (dual-write alongside JSONL) |
| `snap_model_inventory` | SNAP-03 model inventory | one row per model per scan (shared `scan_id`) | inventory snapshot at startup (when storage locations configured) |
| `snap_activity` | SNAP-04 passive activity | one row per active (engine × model) per poll (shared `poll_id`) | health monitor poll loop |
| `snap_vram_sample` | SNAP-02 VRAM totals | one row per persisted poll (`sample_id`) | health monitor poll loop |
| `snap_vram_allocation` | SNAP-02 per-model VRAM | one row per loaded model in a sample (FK → `snap_vram_sample`) | health monitor poll loop |

Cost / savings figures are **derived at read time** from the pricing table (as the
JSONL path already does), not denormalized onto each request row.

## VRAM row-bloat guard

The health monitor can poll every few seconds, which would flood
`snap_vram_sample`. Two guards apply to VRAM writes:

1. **Minimum interval** — `SNAP_VRAM_SAMPLE_SECS` (default **30s**): a VRAM sample
   is written at most once per interval.
2. **Write-on-change** — within an eligible interval, a sample is skipped unless
   the `(used_mb, allocation-count)` fingerprint changed since the last write.

Activity writes are similarly bounded: only models with `active_requests > 0` are
persisted (idle models every poll are dropped).

## Failure policy

SNAP persistence is best-effort observability. Every DB write is off the request's
critical path and any error is logged (`warn!`) and dropped — a database problem
can never fail or slow a proxied inference request.

## Retention

No automatic pruning ships in 1.3.1. Operators who enable persistence should add a
retention policy (e.g. a nightly `DELETE … WHERE created_at < now() - interval 'N
days'`) sized per stream — request-log longer (30–90d), vram/activity shorter
(7–14d).
