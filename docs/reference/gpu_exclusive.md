# gpu_exclusive

GPU-exclusive coordination (40 KG nodes, `src/gpu_exclusive.rs`): a "service
mode" that hands the single host GPU to an external GPU-heavy job — the
Terminus intake benchmarking harness — **without ever taking Chord down**.
Before this existed, the harness got exclusivity by literally stopping the
Chord service, which once left the fleet's backbone proxy dead for three days
of a multi-day sweep. Now Chord stays up (HTTP listeners, health checks,
routing decisions, read-only tools all keep serving) and only *gates* the
GPU/model-inference paths for the duration of the lock.

## The model

- The lock is a single, process-global record (`GPU_EXCLUSIVE`) — one physical
  GPU per Chord process, so this is a hardware resource, not per-request state.
- A grant from FREE (or from an expired/abandoned lock) is a new grant; a
  re-acquire by the **same** holder is a heartbeat refresh (bumps
  `last_heartbeat`, no re-eviction); a live lock held by a **different** holder
  blocks with 409 — two jobs can never silently race the GPU.
- The lock carries a TTL; an expired lock is treated as abandoned and
  re-grantable.

## Key types and functions

| Symbol | Kind | File | What it does |
|---|---|---|---|
| `gpu_exclusive::GpuExclusive::new` | function | `src/gpu_exclusive.rs` | Constructs the coordinator |
| `gpu_exclusive::GpuExclusive::acquire` | function | `src/gpu_exclusive.rs` | Grant / heartbeat-refresh / 409-block, per the model above |
| `gpu_exclusive::GpuExclusive::release` | function | `src/gpu_exclusive.rs` | Explicit release back to FREE |
| `gpu_exclusive::GpuExclusive::active_holder` | function | `src/gpu_exclusive.rs` | The gate the inference handlers consult: `Some(record)` ⇒ structured 503 naming the holder |
| `gpu_exclusive::decide_acquire` / `decide_release` | functions | `src/gpu_exclusive.rs` | The pure decision logic (unit-testable, separate from the shared record) |
| `gpu_exclusive::LockRecord::is_expired` | function | `src/gpu_exclusive.rs` | TTL/abandonment check |

## HTTP surface

On the proxy port, same JWT auth as every other endpoint:

- `POST /v1/gpu-exclusive/acquire` — take or heartbeat the lock (409 if held by
  another live holder)
- `POST /v1/gpu-exclusive/release` — release it
- `GET /v1/gpu-exclusive/status` — current holder, if any

## How it connects

**routes** consults `active_holder` at the top of the inference paths
(`chat_completions`, `infer`) and returns a shared, structured 503 while the
lock is held; the non-GPU surface is unaffected. The external harness calls
acquire with periodic heartbeats for the sweep's duration and releases at the
end; a crashed harness is handled by TTL expiry rather than manual cleanup.

## Configuration

None of its own — behavior is driven by the acquire request (holder label,
TTL) and the shared JWT auth.

## Notes and gaps

- The lock gates *inference*; it does not physically deallocate VRAM. Freeing
  resident models for the sweep is the idle-mode / serving layer's job (see
  [../guides/idle-mode.md](../guides/idle-mode.md) and
  [serving.md](serving.md)).
- The lock is in-process state: a Chord restart clears it (the harness's
  heartbeat re-acquire then re-establishes it).
