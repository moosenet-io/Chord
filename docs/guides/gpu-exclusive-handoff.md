# GPU-exclusive handoff

Lending the single host GPU to an external GPU-heavy job (typically the
Terminus intake benchmarking sweep) **without stopping Chord**. The old
practice — stopping the Chord service for exclusivity — once left the fleet
backbone dead for days; this lock is the replacement. Mechanics:
[../reference/gpu_exclusive.md](../reference/gpu_exclusive.md).

## 1. Acquire the lock

```sh
curl -s -X POST http://localhost:9099/v1/gpu-exclusive/acquire \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"holder": "intake-coder-sweep"}'
```

Outcomes:

- **Granted** — the lock is yours; Chord's inference paths
  (`/v1/chat/completions`, `/v1/infer`) now return a structured 503 naming your
  holder label, while health checks, tools, and read-only endpoints keep
  serving.
- **409** — a different holder has a live lock. Do not race it; check status
  and coordinate.

## 2. Heartbeat while you work

Re-POST the same acquire with the same `holder` periodically. A same-holder
re-acquire is a heartbeat refresh, not a new grant. If your job crashes and
stops heartbeating, the TTL expires the lock and Chord returns to normal
service on its own — no manual cleanup required.

## 3. Check status (any time, any caller)

```sh
curl -s -H "Authorization: Bearer $TOKEN" http://localhost:9099/v1/gpu-exclusive/status
```

## 4. Release

```sh
curl -s -X POST http://localhost:9099/v1/gpu-exclusive/release \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"holder": "intake-coder-sweep"}'
```

Expected outcome: inference paths resume immediately.

## Monitoring the sweep itself

While a benchmarking sweep holds the GPU, its health is observable without
auth: `GET /v1/sweep/status` correlates GPU busy percent, fresh sweep-DB rows,
loaded models, and systemd unit state into a `working`/`stuck`/`idle` verdict —
the wedge signature (GPU pegged, service active, no fresh rows) shows up as
`stuck`. See [../reference/sweep_status.md](../reference/sweep_status.md).

## Troubleshooting

- **503 from chat completions with a holder name in the body** — that's this
  lock working as designed; check `/v1/gpu-exclusive/status` for who holds it
  and its heartbeat age.
- **The lock gates requests but VRAM is still occupied** — the lock does not
  deallocate models; combine with [idle mode](idle-mode.md) if the external job
  needs the memory freed too.
