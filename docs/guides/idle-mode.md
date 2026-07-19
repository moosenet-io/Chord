# Idle mode and the activity signal

Freeing the heavy host for a large build job (the fleet compiler), and telling
"nothing is happening" apart from "something is mid-flight". Two related but
distinct surfaces on the control port (BLD-09 and CHORD-ACT-01, both in
`src/admin/idle.rs`).

## Enter idle mode

```sh
curl -s -X POST -H "Authorization: Bearer $TOKEN" http://localhost:8090/admin/idle
```

Chord drains and releases what it holds — providers, GPU, models, RAM — and
reports the freed memory. The process and both listeners stay up; the point is
that the *host's* heavy resources become available to the build.

## Check the phase

```sh
curl -s -H "Authorization: Bearer $TOKEN" http://localhost:8090/admin/idle
```

## Restore normal service

```sh
curl -s -X POST -H "Authorization: Bearer $TOKEN" http://localhost:8090/admin/activate
```

## The watchdog (why you can't strand Chord idle)

A background watchdog (`admin::idle::watchdog_loop`, spawned at startup, 60 s
cadence) auto-activates Chord if it is left idle past the deadline with no
active compiler GPU-exclusive lease — covering a crashed/forgotten compiler or
a stale idle state reloaded after a restart. Chord is never left silently dead;
letting the watchdog fire is safe, just slower than an explicit activate.

## The activity signal (scheduling into genuine idle windows)

```sh
curl -s -H "Authorization: Bearer $TOKEN" http://localhost:8090/admin/activity
```

Distinct from the idle *mode*: this reports whether inference is **actually in
flight** right now and, if not, how long Chord has been quiet. A build
scheduler should consult this before entering idle mode, so heavy work lands in
genuine quiet windows instead of interrupting live traffic.

## Troubleshooting

- **Idle entered but the build still can't get memory** — check what else holds
  the GPU/RAM outside Chord; idle mode releases Chord-owned resources only.
  `GET /api/vram` (SNAP) shows the device-truth VRAM picture.
- **Chord reactivated by itself** — that's the watchdog: the idle deadline
  passed without a live compiler lease. Re-enter idle with the compiler's lease
  in place.
