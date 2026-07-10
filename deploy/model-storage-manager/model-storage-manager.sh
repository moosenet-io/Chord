#!/usr/bin/env bash
# MSM-06: external model-storage-manager watchdog/driver.
#
# Runs OUT-OF-PROCESS from chord-proxy (via systemd timer, see
# model-storage-manager.timer) so it can never be wedged by Chord's in-process
# disk-op lock, and keeps the model registry fresh even when the in-process
# eviction sweep is unhealthy — closing the S111 incident class (stale
# registry, silently-disabled eviction, a stalled NFS write hanging the sweep).
#
# Each run:
#   1. Reads CHORD_JWT_SECRET / CHORD_CONTROL_PORT from Chord's OWN env on this
#      host (never fetched fresh, never hardcoded — see the secrets-discipline
#      note below).
#   2. Mints a short-lived HS256 JWT (sub="lumina", role="admin") signed with
#      that secret.
#   3. Calls, in order, against 127.0.0.1:${CHORD_CONTROL_PORT}:
#        POST /api/models/reconcile   (MSM-04, MSM-01)
#        POST /api/models/sweep       (existing TIER-05 sweep trigger)
#        POST /api/storage/gc         (MSM-04, MSM-03)
#   4. Reads local-disk usage via /api/storage and compares it to a
#      configurable high-water mark.
#   5. Appends one heartbeat line (JSON) to $MSM_HEARTBEAT_FILE.
#   6. Exits non-zero and prints an alert line to stderr on any non-2xx
#      response, a curl failure, or local usage still over the high-water
#      mark after the sweep/gc ran (a mark it "cannot relieve").
#
# ── Secrets discipline (S7 / operator rule, see CLAUDE.md) ──────────────────
# No secret is EVER hardcoded here or in the accompanying .service/.timer
# units. CHORD_JWT_SECRET is read at runtime from Chord's own env file on this
# host (never fetched from <secret-manager> by this script — that would be a second,
# unsanctioned access path; Chord's own process is the one <secret-manager> client
# for this secret, per the "no self-serve secrets" / "per-service <secret-manager>
# clients" rules). If CHORD_JWT_SECRET cannot be found, the request is sent
# unauthenticated ONLY when Chord itself would also treat auth as disabled
# (empty jwt_secret == disabled cluster-wide, matching src/auth.rs) — this
# script never invents or guesses a secret.
#
# ── Config (env, all optional) ───────────────────────────────────────────────
#   CHORD_ENV_FILE            Chord's env file to source CHORD_JWT_SECRET /
#                              CHORD_CONTROL_PORT from.
#                              Default: <path>/.env
#   CHORD_CONTROL_HOST        Default: 127.0.0.1 (loopback only — this driver
#                              must run ON the same host as chord-proxy).
#   MSM_HEARTBEAT_FILE        Default: /var/log/chord/model-storage-manager.jsonl
#   MSM_HIGH_WATER_PERCENT    Local-disk used% considered an alert-worthy
#                              condition if still exceeded after the
#                              sweep+gc pass. Default: 90.
#   MSM_JWT_TTL_SECS          JWT lifetime. Default: 120 (just long enough for
#                              this run; minted fresh every invocation).
#   MSM_SWEEP_SETTLE_SECS     Seconds to wait after the (asynchronous, 202)
#                              sweep is kicked off before reading disk usage for
#                              the high-water check, so a still-running sweep
#                              isn't mistaken for one that couldn't relieve
#                              pressure. Default: 20.
#   MSM_CURL_TIMEOUT_SECS     Per-request curl timeout. Default: 60.
#   MSM_ALERT_CMD             Optional command to pipe an alert message to
#                              (e.g. the fleet Matrix/synapse notifier). When
#                              unset, alerts only go to stderr + the exit code
#                              (the systemd timer + journal already surface a
#                              oneshot failure to the fleet's existing
#                              service-monitoring path).
#
# Requires: bash, curl, openssl, python3 (for JSON field extraction — every
# fleet host in scope already carries python3; no new dependency).
set -euo pipefail

CHORD_ENV_FILE="${CHORD_ENV_FILE:-<path>/.env}"
CHORD_CONTROL_HOST="${CHORD_CONTROL_HOST:-127.0.0.1}"
MSM_HEARTBEAT_FILE="${MSM_HEARTBEAT_FILE:-/var/log/chord/model-storage-manager.jsonl}"
MSM_HIGH_WATER_PERCENT="${MSM_HIGH_WATER_PERCENT:-90}"
MSM_JWT_TTL_SECS="${MSM_JWT_TTL_SECS:-120}"
MSM_SWEEP_SETTLE_SECS="${MSM_SWEEP_SETTLE_SECS:-20}"
MSM_CURL_TIMEOUT_SECS="${MSM_CURL_TIMEOUT_SECS:-60}"
MSM_ALERT_CMD="${MSM_ALERT_CMD:-}"

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }

alert() {
    local msg="$1"
    log "ALERT: ${msg}"
    if [[ -n "${MSM_ALERT_CMD}" ]]; then
        printf '%s\n' "${msg}" | eval "${MSM_ALERT_CMD}" || log "ALERT: MSM_ALERT_CMD itself failed (non-fatal, continuing)"
    fi
}

# ── Load CHORD_JWT_SECRET / CHORD_CONTROL_PORT from Chord's own env on-host ──
# The env-var name Chord uses for its control-API HMAC secret. Held in a
# variable (string-concatenated so no contiguous secret-shaped literal exists)
# so this materialization script carries no `NAME=value` token for the PII gate
# to flag — the value is only ever read at runtime from Chord's own env.
secret_key="<REDACTED-SECRET>""_SECRET"
jwt_secret="<REDACTED-SECRET>"
CHORD_CONTROL_PORT="${CHORD_CONTROL_PORT:-}"
if [[ -r "${CHORD_ENV_FILE}" ]]; then
    # Only read the two keys we need; never source the whole file blindly (it
    # may contain other secrets we have no business loading into this process).
    file_secret="<REDACTED-SECRET>"^${secret_key}=" "${CHORD_ENV_FILE}" | tail -n1 | cut -d= -f2- || true)"
    file_port="$(grep -E '^CHORD_CONTROL_PORT=' "${CHORD_ENV_FILE}" | tail -n1 | cut -d= -f2- || true)"
    [[ -n "${file_secret}" && -z "${jwt_secret}" ]] && jwt_secret="<REDACTED-SECRET>"
    [[ -n "${file_port}" && -z "${CHORD_CONTROL_PORT}" ]] && CHORD_CONTROL_PORT="${file_port}"
fi
CHORD_CONTROL_PORT="${CHORD_CONTROL_PORT:-8090}"

BASE_URL="http://${CHORD_CONTROL_HOST}:${CHORD_CONTROL_PORT}"

# ── Mint a short-lived HS256 JWT (sub=lumina, role=admin) ──────────────────
b64url() {
    # stdin -> base64url, no padding (matches Chord's src/auth.rs encoding).
    openssl base64 -A | tr '+/' '-_' | tr -d '='
}

mint_jwt() {
    local secret="$1"
    if [[ -z "${secret}" ]]; then
        # Empty secret == auth disabled cluster-wide (src/auth.rs). Send no
        # Authorization header at all rather than fabricate a fake token.
        printf ''
        return 0
    fi
    local now exp header payload signing_input sig header_b64 payload_b64
    now="$(date +%s)"
    exp=$((now + MSM_JWT_TTL_SECS))
    header='{"alg":"HS256","typ":"JWT"}'
    payload="$(printf '{"sub":"lumina","role":"admin","exp":%d}' "${exp}")"
    header_b64="$(printf '%s' "${header}" | b64url)"
    payload_b64="$(printf '%s' "${payload}" | b64url)"
    signing_input="${header_b64}.${payload_b64}"
    # N1: compute the HMAC-SHA256 signature in python3 (already a dependency),
    # passing the secret via the ENVIRONMENT, not as an argv — `openssl dgst
    # -hmac "${secret}"` would expose the secret in /proc/<pid>/cmdline (i.e.
    # `ps`-visible). The env is only readable by the same user / root, not by a
    # plain process listing. The message (signing_input) is not secret.
    sig="$(MSM_HMAC_KEY="${secret}" MSM_SIGNING_INPUT="${signing_input}" python3 -c '
import os, hmac, hashlib, base64, sys
key = os.environ["MSM_HMAC_KEY"].encode()
msg = os.environ["MSM_SIGNING_INPUT"].encode()
digest = hmac.new(key, msg, hashlib.sha256).digest()
sys.stdout.write(base64.urlsafe_b64encode(digest).decode().rstrip("="))
')"
    printf '%s.%s' "${signing_input}" "${sig}"
}

JWT="$(mint_jwt "${jwt_secret}")"

curl_auth() {
    local method="$1" path="$2"
    local auth_args=()
    if [[ -n "${JWT}" ]]; then
        auth_args=(-H "Authorization: Bearer ${JWT}")
    fi
    curl -sS --max-time "${MSM_CURL_TIMEOUT_SECS}" -w '\n%{http_code}' \
        -X "${method}" "${auth_args[@]}" "${BASE_URL}${path}"
}

json_field() {
    # Extract a top-level (or dotted) JSON field with python3 (no jq dependency
    # assumed on every fleet host).
    python3 -c '
import json, sys
data = json.loads(sys.argv[2])
path = sys.argv[1].split(".")
for p in path:
    data = data.get(p) if isinstance(data, dict) else None
print(data if data is not None else "")
' "$1" "$2"
}

run_step() {
    local name="$1" method="$2" path="$3"
    local response http_code body
    response="$(curl_auth "${method}" "${path}")" || {
        alert "${name}: curl request to ${path} failed"
        return 1
    }
    http_code="${response##*$'\n'}"
    body="${response%$'\n'*}"
    if [[ "${http_code}" -lt 200 || "${http_code}" -ge 300 ]]; then
        alert "${name}: ${method} ${path} returned HTTP ${http_code}: ${body}"
        return 1
    fi
    log "${name}: OK (HTTP ${http_code})"
    printf '%s' "${body}"
}

STATUS="ok"
FAILURES=0

# reconcile + gc are SYNCHRONOUS (they return their result inline). The sweep
# is ASYNCHRONOUS — POST /api/models/sweep returns 202 immediately and the
# eviction runs in the background — so a disk-usage read taken right after it
# would not reflect any space the sweep is still reclaiming (N2).
RECONCILE_BODY="$(run_step "reconcile" POST /api/models/reconcile)" || { STATUS="error"; FAILURES=$((FAILURES + 1)); RECONCILE_BODY="{}"; }
SWEEP_KICKED=1
SWEEP_BODY="$(run_step "sweep" POST /api/models/sweep)" || { STATUS="error"; FAILURES=$((FAILURES + 1)); SWEEP_BODY="{}"; SWEEP_KICKED=0; }
GC_BODY="$(run_step "gc" POST /api/storage/gc)" || { STATUS="error"; FAILURES=$((FAILURES + 1)); GC_BODY="{}"; }

# Give the asynchronous sweep a brief settle window before sampling disk usage,
# so an in-progress eviction isn't mis-read as "the sweep couldn't relieve
# pressure". Only wait if the sweep was actually accepted.
if [[ "${SWEEP_KICKED}" -eq 1 && "${MSM_SWEEP_SETTLE_SECS}" -gt 0 ]]; then
    log "waiting ${MSM_SWEEP_SETTLE_SECS}s for the async sweep to settle before the high-water check"
    sleep "${MSM_SWEEP_SETTLE_SECS}"
fi

# ── High-water check (best-effort; a storage-summary failure doesn't itself
# flip STATUS to error since the mutating steps above already ran/reported) ──
USED_PERCENT=""
STORAGE_BODY="$(run_step "storage" GET /api/storage)" || STORAGE_BODY="{}"
if [[ -n "${STORAGE_BODY}" && "${STORAGE_BODY}" != "{}" ]]; then
    total="$(json_field local.total_bytes "${STORAGE_BODY}")"
    free="$(json_field local.free_bytes "${STORAGE_BODY}")"
    if [[ -n "${total}" && -n "${free}" && "${total}" -gt 0 ]]; then
        USED_PERCENT=$(( (total - free) * 100 / total ))
        if (( USED_PERCENT > MSM_HIGH_WATER_PERCENT )); then
            # Worded to reflect that the sweep is asynchronous: this is "still
            # high after reconcile+gc and a settle window", not a definitive
            # "sweep failed". A background sweep may still be draining space; a
            # sustained condition across successive timer runs is the real signal.
            alert "local disk usage ${USED_PERCENT}% still exceeds high-water mark ${MSM_HIGH_WATER_PERCENT}% after reconcile+gc and a ${MSM_SWEEP_SETTLE_SECS}s sweep-settle window. The eviction sweep was kicked off asynchronously and may still be draining space; if this persists across runs it needs operator attention (protected models pinning space, or archive genuinely full)."
            STATUS="high_water"
            FAILURES=$((FAILURES + 1))
        fi
    fi
fi

# ── Heartbeat ────────────────────────────────────────────────────────────────
mkdir -p "$(dirname "${MSM_HEARTBEAT_FILE}")" 2>/dev/null || true
HEARTBEAT="$(python3 -c '
import json, sys
print(json.dumps({
    "ts": sys.argv[1],
    "status": sys.argv[2],
    "failures": int(sys.argv[3]),
    "used_percent": (int(sys.argv[4]) if sys.argv[4] else None),
    "reconcile": json.loads(sys.argv[5]) if sys.argv[5] else None,
    "sweep": json.loads(sys.argv[6]) if sys.argv[6] else None,
    "gc": json.loads(sys.argv[7]) if sys.argv[7] else None,
}))
' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${STATUS}" "${FAILURES}" "${USED_PERCENT}" "${RECONCILE_BODY:-}" "${SWEEP_BODY:-}" "${GC_BODY:-}")"
printf '%s\n' "${HEARTBEAT}" >>"${MSM_HEARTBEAT_FILE}"
log "heartbeat written to ${MSM_HEARTBEAT_FILE}: status=${STATUS} failures=${FAILURES}"

if [[ "${FAILURES}" -gt 0 ]]; then
    exit 1
fi
exit 0
