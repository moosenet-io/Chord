# Guides

Task-oriented operator guides. Every command targets a real endpoint from
`src/routes.rs` / `src/control.rs`; `$TOKEN` is a JWT signed with
`CHORD_JWT_SECRET` (operator-specific issuing procedure), and ports are the
defaults (`CHORD_PROXY_PORT` 9099, `CHORD_CONTROL_PORT` 8090).

| Guide | When you need it |
|---|---|
| [Model tiering operations](model-tiering-operations.md) | Inspect storage tiers, archive/pull/protect models, run sweeps and GC |
| [GPU-exclusive handoff](gpu-exclusive-handoff.md) | Lend the GPU to a benchmarking sweep without stopping Chord |
| [Idle mode and the activity signal](idle-mode.md) | Free the host for a heavy build; find genuine idle windows |

Deep dives that complement these: [../serving.md](../serving.md) (VRAM
residency and swaps), [../egress.md](../egress.md) (launch isolation),
[../chord-tui.md](../chord-tui.md) (doing all of this from the TUI instead of
curl).
