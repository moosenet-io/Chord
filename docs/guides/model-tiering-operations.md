# Model tiering operations

Operating the hot/warm/cold storage tiers through the control API
(`CHORD_CONTROL_PORT`, default 8090). All endpoints require the JWT; see
[../reference/models.md](../reference/models.md) for the machinery behind each
call.

## 1. Inspect

```sh
# Every registry record: tier, size, last_requested, protected flag, supports_thinking
curl -s -H "Authorization: Bearer $TOKEN" http://localhost:8090/api/models

# One model (404 if unknown)
curl -s -H "Authorization: Bearer $TOKEN" http://localhost:8090/api/models/<name>

# Disk usage, local + archive
curl -s -H "Authorization: Bearer $TOKEN" http://localhost:8090/api/storage
```

Expected outcome: JSON records whose tiers match on-disk reality. If they look
stale, run a reconcile (step 4).

## 2. Move a model between tiers

```sh
# Warm → Cold (archive-first, verify-then-delete). Hot ⇒ 409; protected ⇒ 403.
curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  http://localhost:8090/api/models/<name>/archive

# Cold → Warm (pull from archive). Insufficient local space ⇒ 507.
curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  http://localhost:8090/api/models/<name>/pull
```

Note: a `Cold` model requested via `/v1/chat/completions` is pulled
transparently anyway — manual pulls are for pre-warming ahead of expected load.

## 3. Protect a model from eviction

```sh
curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  http://localhost:8090/api/models/<name>/protect
```

Protected models can never be demoted to `Cold` — by the sweep or by hand.
`MODEL_PROTECTED` (env) is the authoritative baseline set, re-applied on every
reconcile.

## 4. Maintenance passes

```sh
# Disk-pressure eviction sweep (202 Accepted, runs async)
curl -s -X POST -H "Authorization: Bearer $TOKEN" http://localhost:8090/api/models/sweep

# Reconcile registry against on-disk manifests; returns before/after tier counts
curl -s -X POST -H "Authorization: Bearer $TOKEN" http://localhost:8090/api/models/reconcile

# Orphan-blob GC; returns {orphans_deleted, freed_bytes, errors}
curl -s -X POST -H "Authorization: Bearer $TOKEN" http://localhost:8090/api/storage/gc
```

The background sweep already runs every `MODEL_SWEEP_INTERVAL_SECS` with
reconcile + GC folded in; manual triggers are for immediate relief. For
unattended operation, deploy
[`deploy/model-storage-manager/`](../../deploy/model-storage-manager/) — a
systemd service + timer that drives reconcile → sweep → gc out-of-process with
a heartbeat and alerting.

## Troubleshooting

- **Eviction appears to do nothing**: check `MODEL_WARM_COOLDOWN_HOURS` (0
  disables cooldown demotion — Chord warns at startup) and whether candidates
  are protected. Only warm, non-protected, Ollama-managed models are eligible.
- **Archive operations hang then recover**: the archive copy is bounded by
  `MODEL_ARCHIVE_COPY_TIMEOUT_SECS` (default 1800 s); on timeout the model
  stays Warm and retries next sweep — a stalled NFS write cannot wedge the
  sweep.
