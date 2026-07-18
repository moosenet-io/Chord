# model-storage-manager (MSM-06)

The external "sub service with a regular cron" the S111 incident called for: an
out-of-process watchdog/driver that periodically calls Chord's model-storage
control endpoints (`POST /api/models/reconcile`, `POST /api/models/sweep`,
`POST /api/storage/gc`) so the on-disk registry and local/cold tiering stay
correct **even if the in-process eviction sweep stalls or the registry
drifts**. Because it runs as its own systemd unit, it can never be wedged by
Chord's in-process disk-op lock, unlike a stalled NFS write inside Chord
itself (the S111 incident's root cause).

## What it does, each run (every ~15 min via the timer)

1. Reads `CHORD_JWT_SECRET` / `CHORD_CONTROL_PORT` from Chord's own env file
   on this host (`CHORD_ENV_FILE`, default `<path>/.env`) — **never**
   fetched fresh from <secret-manager> by this script (see "Secrets" below).
2. Mints a short-lived HS256 JWT (`sub=lumina`, `role=admin`).
3. Calls, against `127.0.0.1:${CHORD_CONTROL_PORT}`, in order:
   - `POST /api/models/reconcile` — MSM-01/MSM-04: re-sync the registry
     against on-disk reality and persist it.
   - `POST /api/models/sweep` — trigger the existing disk-pressure/cooldown
     eviction sweep.
   - `POST /api/storage/gc` — MSM-03/MSM-04: delete orphan blobs.
4. Reads `GET /api/storage` and compares local disk usage to
   `MSM_HIGH_WATER_PERCENT` (default 90%). If still over the mark after the
   above ran, that's a condition this manager **cannot relieve on its own**
   (e.g. protected models pinning space, or the archive itself is full) — it
   alerts loudly rather than silently repeating a no-op.
5. Appends one JSON heartbeat line to `MSM_HEARTBEAT_FILE` (default
   `/var/log/chord/model-storage-manager.jsonl`).
6. Exits non-zero (and prints an `ALERT:` line to stderr) on any non-2xx
   response, a `curl` failure, or an unrelieved high-water condition.

## Install (on the Chord GPU host)

```sh
sudo mkdir -p <path>/deploy/model-storage-manager
sudo cp model-storage-manager.sh <path>/deploy/model-storage-manager/
sudo chmod +x <path>/deploy/model-storage-manager/model-storage-manager.sh

sudo cp model-storage-manager.service model-storage-manager.timer \
    /etc/systemd/system/

# Verify the User=/Group= in the .service match chord.service's own:
systemctl show chord.service --property=User,Group
# Edit /etc/systemd/system/model-storage-manager.service if they differ.

sudo mkdir -p /var/log/chord
sudo chown <chord-service-user>:<chord-service-group> /var/log/chord

sudo systemctl daemon-reload
sudo systemctl enable --now model-storage-manager.timer

# Confirm:
systemctl list-timers model-storage-manager.timer
sudo systemctl start model-storage-manager.service   # run once immediately
journalctl -u model-storage-manager.service -n 50
tail -n5 /var/log/chord/model-storage-manager.jsonl
```

## Config (all via `Environment=` in the `.service` file, all optional)

| Variable                  | Default                                       | Purpose |
|----------------------------|-----------------------------------------------|---------|
| `CHORD_ENV_FILE`           | `<path>/.env`                             | Where to read `CHORD_JWT_SECRET`/`CHORD_CONTROL_PORT` from |
| `CHORD_CONTROL_HOST`       | `127.0.0.1`                                   | Must stay loopback — this driver runs on the same host as chord-proxy |
| `MSM_HEARTBEAT_FILE`       | `/var/log/chord/model-storage-manager.jsonl`  | Heartbeat log path |
| `MSM_HIGH_WATER_PERCENT`   | `90`                                          | Alert threshold if local usage is still above this after a run |
| `MSM_JWT_TTL_SECS`         | `120`                                        | Minted-JWT lifetime (fresh every run) |
| `MSM_SWEEP_SETTLE_SECS`    | `20`                                          | Wait after the async (202) sweep before the high-water read, so an in-progress eviction isn't mis-read as unrelievable |
| `MSM_CURL_TIMEOUT_SECS`    | `60`                                          | Per-request curl timeout |
| `MSM_ALERT_CMD`            | unset                                        | Optional command the alert message is piped to (e.g. the fleet's Matrix/synapse notifier); when unset, alerts go to stderr/journal + the non-zero exit code, which the timer's own failure surfaces to existing fleet service-monitoring |

## Secrets discipline (no exceptions — see CLAUDE.md / S7)

- **No secret is ever hardcoded** in `model-storage-manager.sh` or either unit
  file. `CHORD_JWT_SECRET` is read at runtime from Chord's own `.env` on this
  host — the same materialization Chord itself uses, never re-fetched from
  <secret-manager> by this script (that would be a second, unsanctioned access path
  to a secret Chord's own <secret-manager> client already owns).
- If `CHORD_JWT_SECRET` is empty/unset, the script sends requests with no
  `Authorization` header at all — matching Chord's own "empty secret disables
  auth cluster-wide" behavior (`src/auth.rs`) rather than fabricating a token.
- The JWT signature (HMAC-SHA256 over the secret) is computed in `python3` with
  the secret passed via an **environment variable**, never as a command-line
  argument — so it never appears in `/proc/<pid>/cmdline` (i.e. it is not
  `ps`-visible). Only the non-secret header/payload are base64url-encoded via
  `openssl`.
- This driver must run **on the same host** as chord-proxy (`CHORD_CONTROL_HOST`
  stays `127.0.0.1`) — it is not a remote client and never should be.

## Synchronous vs asynchronous steps

`reconcile` and `gc` are **synchronous** — the endpoints do the work and return
the result inline, so their response bodies in the heartbeat are authoritative.
`POST /api/models/sweep` is **asynchronous** — it returns `202 Accepted`
immediately and the eviction runs in the background. The driver therefore waits
`MSM_SWEEP_SETTLE_SECS` after kicking the sweep before sampling disk usage, and
the high-water alert is worded as "still high after a settle window" (a
sustained condition across runs is the real operator signal), not a definitive
"the sweep failed".

## Dependencies

`bash`, `curl`, `openssl` (base64url of the non-secret JWT header/payload),
`python3` (JWT HMAC signing with the secret held in-env, plus JSON field
extraction for the heartbeat / high-water check — every fleet host in scope
already carries `python3`; this avoids adding a `jq` dependency).

## Verify / troubleshoot

- `journalctl -u model-storage-manager.service` — run-by-run output, including
  `ALERT:` lines on failure.
- `tail -f /var/log/chord/model-storage-manager.jsonl` — heartbeat history
  (one JSON object per line: `ts`, `status`, `failures`, `used_percent`, and
  the raw `reconcile`/`sweep`/`gc` response bodies).
- Chord down → the run's `curl` calls fail, it alerts, exits non-zero; the
  timer retries on its next `OnUnitActiveSec` tick (15 min).
- `CHORD_JWT_SECRET` rotated without updating `CHORD_ENV_FILE` → the control
  API returns 401, the run alerts clearly (`POST ... returned HTTP 401`)
  rather than failing silently.
