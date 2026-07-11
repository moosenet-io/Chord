//! RESIL-02: a durable, session-keyed cache of a sweep's planned ACTION QUEUE
//! plus its progress cursor, so Chord — the component that owns model movement
//! and host-singleton GPU state — can tell a restarted sweep exactly what work
//! remains.
//!
//! ## Why this exists
//! The MINT sweep's planned work lives in the Terminus process plus a file-backed
//! checkpoint on the NAS. That file survives a *Terminus* restart, but Chord
//! itself has no knowledge of the sweep's remaining work, so it cannot help a
//! restarted sweep resume and cannot be the cross-restart authority the operator
//! wants it to be. This module gives Chord a small durable store the sweep can
//! REGISTER a queue with, ADVANCE a cursor against as each unit completes, and
//! query for what REMAINS. Chord only RECORDS and SERVES the queue — it never
//! executes it; the Terminus sweep is the executor (see RESIL-03).
//!
//! ## Model
//! - An [`ActionKey`] is an opaque, caller-defined stable string (Terminus owns
//!   its shape, e.g. `"<run_kind>|<model>|<backend>|<case>"`); Chord treats it as
//!   a token and never parses it.
//! - A [`SweepSession`] is `{ session_id, created_utc, queue, done }`. `remaining`
//!   is `queue` minus `done`, IN QUEUE ORDER.
//! - The [`SweepSessionStore`] holds sessions by id behind an `RwLock`, and — when
//!   a state path is configured (`CHORD_STATE_DIR`) — persists the whole store to
//!   a file (atomic tempfile+rename) on every mutation and reloads it on startup.
//!   Persistence is best-effort: a missing/corrupt/unwritable file never panics
//!   Chord (warn + start empty); path unset ⇒ in-memory only (lost on restart).
//!
//! The pure queue math ([`SweepSession::remaining`], [`register_decision`],
//! [`advance_into`]) is separated from the `RwLock`/clock/IO so it is
//! exhaustively unit-testable with no global state and no disk.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use serde::{Deserialize, Serialize};
use tracing::warn;

/// An opaque, caller-defined stable action key. Chord never parses it.
pub type ActionKey = String;

/// One sweep's registered action queue plus the set of completed keys.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SweepSession {
    pub session_id: String,
    pub created_utc: u64,
    pub queue: Vec<ActionKey>,
    pub done: BTreeSet<ActionKey>,
}

impl SweepSession {
    /// The remaining keys: `queue` minus `done`, preserving queue order.
    pub fn remaining(&self) -> Vec<ActionKey> {
        self.queue
            .iter()
            .filter(|k| !self.done.contains(*k))
            .cloned()
            .collect()
    }

    /// A JSON-friendly summary (counts + remaining) for the control API.
    pub fn summary(&self) -> SessionSummary {
        let remaining = self.remaining();
        SessionSummary {
            session_id: self.session_id.clone(),
            total: self.queue.len(),
            done_count: self.done.len(),
            remaining,
        }
    }
}

/// The control-API view of a session: counts + the remaining keys in order.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SessionSummary {
    pub session_id: String,
    pub total: usize,
    pub done_count: usize,
    pub remaining: Vec<ActionKey>,
}

/// Pure decision for a register/upsert of `session_id` with `queue` at `now`,
/// given any existing session. Idempotent on an identical queue (preserves
/// `done`); a different queue replaces it and resets `done` (a replanned sweep).
pub fn register_decision(
    existing: Option<&SweepSession>,
    session_id: &str,
    queue: Vec<ActionKey>,
    now: u64,
) -> SweepSession {
    match existing {
        // Same id, same queue ⇒ no-op: keep the existing session (and its `done`).
        Some(s) if s.queue == queue => s.clone(),
        // New, or same id with a different queue ⇒ fresh session, empty `done`.
        _ => SweepSession {
            session_id: session_id.to_string(),
            created_utc: now,
            queue,
            done: BTreeSet::new(),
        },
    }
}

/// Pure advance: mark `keys` done on `session`, but ONLY keys that are actually
/// in the queue (a key not in the queue is ignored — opaque, never an error).
/// Append-only + idempotent (re-marking a done key is harmless).
pub fn advance_into(session: &mut SweepSession, keys: &[ActionKey]) {
    let in_queue: BTreeSet<&ActionKey> = session.queue.iter().collect();
    for k in keys {
        if in_queue.contains(k) {
            session.done.insert(k.clone());
        }
    }
}

/// The persisted on-disk shape: a flat list of sessions (a map serializes fine
/// too, but a list keeps the file human-diffable and order-stable).
#[derive(Debug, Default, Serialize, Deserialize)]
struct PersistedStore {
    sessions: Vec<SweepSession>,
}

/// Load the persisted store from `path`. Missing/unreadable/malformed ⇒ empty
/// map with a warn — never a panic.
fn load_persisted(path: &Path) -> HashMap<String, SweepSession> {
    let data = match std::fs::read_to_string(path) {
        Ok(d) => d,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return HashMap::new(),
        Err(e) => {
            warn!(path = %path.display(), error = %e,
                "sweep-session: could not read persisted store (starting empty)");
            return HashMap::new();
        }
    };
    match serde_json::from_str::<PersistedStore>(&data) {
        Ok(store) => store
            .sessions
            .into_iter()
            .map(|s| (s.session_id.clone(), s))
            .collect(),
        Err(e) => {
            warn!(path = %path.display(), error = %e,
                "sweep-session: persisted store is corrupt/unrecognized (starting empty)");
            HashMap::new()
        }
    }
}

/// Atomically persist `sessions` to `path` (tempfile + rename). Best-effort: any
/// IO/serde error is logged at warn and swallowed — persistence must never break
/// a register/advance.
fn persist_state(path: &Path, sessions: &HashMap<String, SweepSession>) {
    let store = PersistedStore {
        sessions: sessions.values().cloned().collect(),
    };
    let json = match serde_json::to_string(&store) {
        Ok(j) => j,
        Err(e) => {
            warn!(error = %e, "sweep-session: failed to serialize store (state not persisted)");
            return;
        }
    };
    if let Some(dir) = path.parent() {
        if let Err(e) = std::fs::create_dir_all(dir) {
            warn!(dir = %dir.display(), error = %e,
                "sweep-session: could not create state dir (state not persisted)");
            return;
        }
    }
    let tmp = path.with_extension("json.tmp");
    if let Err(e) = std::fs::write(&tmp, json.as_bytes()) {
        warn!(path = %tmp.display(), error = %e,
            "sweep-session: could not write temp store file (state not persisted)");
        return;
    }
    if let Err(e) = std::fs::rename(&tmp, path) {
        warn!(path = %path.display(), error = %e,
            "sweep-session: could not atomically install store file (state not persisted)");
        let _ = std::fs::remove_file(&tmp);
    }
}

/// A durable, session-keyed store of sweep action queues. Host-singleton (one
/// sweep host per Chord), mirroring the process-global `GPU_EXCLUSIVE` pattern.
pub struct SweepSessionStore {
    inner: RwLock<HashMap<String, SweepSession>>,
    /// Where the store is persisted across restarts. `None` ⇒ in-memory only.
    state_path: Option<PathBuf>,
}

impl SweepSessionStore {
    /// In-memory only (no persistence) — used by unit tests and when
    /// `CHORD_STATE_DIR` is unset.
    pub fn new_in_memory() -> Self {
        Self {
            inner: RwLock::new(HashMap::new()),
            state_path: None,
        }
    }

    /// Construct with durable persistence at `state_path`, loading any existing
    /// store on startup. A missing/corrupt file starts empty.
    pub fn with_state(state_path: Option<PathBuf>) -> Self {
        let initial = match state_path.as_deref() {
            Some(p) => load_persisted(p),
            None => HashMap::new(),
        };
        Self {
            inner: RwLock::new(initial),
            state_path,
        }
    }

    /// Construct from the environment (`CHORD_STATE_DIR` via config).
    pub fn from_env() -> Self {
        Self::with_state(crate::config::sweep_session_state_path())
    }

    fn persist_locked(&self, sessions: &HashMap<String, SweepSession>) {
        if let Some(path) = self.state_path.as_deref() {
            persist_state(path, sessions);
        }
    }

    /// Register (idempotent upsert) `session_id` with `queue` at `now`. Returns
    /// the resulting session summary.
    pub fn register(&self, session_id: &str, queue: Vec<ActionKey>, now: u64) -> SessionSummary {
        let mut guard = self.inner.write().expect("sweep-session store poisoned");
        let session = register_decision(guard.get(session_id), session_id, queue, now);
        let summary = session.summary();
        guard.insert(session_id.to_string(), session);
        self.persist_locked(&guard);
        summary
    }

    /// The current summary for `session_id`, or `None` if unknown.
    pub fn get(&self, session_id: &str) -> Option<SessionSummary> {
        let guard = self.inner.read().expect("sweep-session store poisoned");
        guard.get(session_id).map(|s| s.summary())
    }

    /// Mark `keys` done on `session_id`. Returns the updated summary, or `None`
    /// if the session is unknown.
    pub fn advance(&self, session_id: &str, keys: &[ActionKey]) -> Option<SessionSummary> {
        let mut guard = self.inner.write().expect("sweep-session store poisoned");
        let summary = match guard.get_mut(session_id) {
            Some(s) => {
                advance_into(s, keys);
                Some(s.summary())
            }
            None => None,
        };
        if summary.is_some() {
            self.persist_locked(&guard);
        }
        summary
    }
}

/// The process-global sweep-session store, mirroring `GPU_EXCLUSIVE`. Handlers
/// consult this; unit tests use isolated `new_in_memory`/`with_state` instances.
pub static SWEEP_SESSIONS: once_cell::sync::Lazy<SweepSessionStore> =
    once_cell::sync::Lazy::new(SweepSessionStore::from_env);

#[cfg(test)]
mod tests {
    use super::*;

    fn q(items: &[&str]) -> Vec<ActionKey> {
        items.iter().map(|s| s.to_string()).collect()
    }

    // ── pure logic ───────────────────────────────────────────────────────────

    #[test]
    fn remaining_is_queue_minus_done_in_order() {
        let mut s = register_decision(None, "sess", q(&["a", "b", "c", "d"]), 0);
        advance_into(&mut s, &q(&["b", "d"]));
        assert_eq!(s.remaining(), q(&["a", "c"]));
    }

    #[test]
    fn register_same_queue_is_noop_preserving_done() {
        let mut s = register_decision(None, "sess", q(&["a", "b"]), 0);
        advance_into(&mut s, &q(&["a"]));
        // Re-register with the identical queue ⇒ keep the existing session + done.
        let s2 = register_decision(Some(&s), "sess", q(&["a", "b"]), 100);
        assert_eq!(s2.done, s.done);
        assert_eq!(s2.created_utc, 0); // unchanged
        assert_eq!(s2.remaining(), q(&["b"]));
    }

    #[test]
    fn register_different_queue_replaces_and_resets_done() {
        let mut s = register_decision(None, "sess", q(&["a", "b"]), 0);
        advance_into(&mut s, &q(&["a"]));
        let s2 = register_decision(Some(&s), "sess", q(&["x", "y", "z"]), 100);
        assert!(s2.done.is_empty());
        assert_eq!(s2.created_utc, 100);
        assert_eq!(s2.remaining(), q(&["x", "y", "z"]));
    }

    #[test]
    fn advance_ignores_keys_not_in_queue() {
        let mut s = register_decision(None, "sess", q(&["a", "b"]), 0);
        advance_into(&mut s, &q(&["a", "ghost"]));
        assert_eq!(s.done, q(&["a"]).into_iter().collect());
        assert_eq!(s.remaining(), q(&["b"]));
    }

    #[test]
    fn advance_is_idempotent() {
        let mut s = register_decision(None, "sess", q(&["a", "b"]), 0);
        advance_into(&mut s, &q(&["a"]));
        advance_into(&mut s, &q(&["a"]));
        assert_eq!(s.done.len(), 1);
    }

    // ── store (in-memory) ────────────────────────────────────────────────────

    #[test]
    fn store_register_get_advance_cycle() {
        let store = SweepSessionStore::new_in_memory();
        assert!(store.get("s").is_none());

        let sum = store.register("s", q(&["a", "b", "c"]), 0);
        assert_eq!(sum.total, 3);
        assert_eq!(sum.done_count, 0);
        assert_eq!(sum.remaining, q(&["a", "b", "c"]));

        let sum = store.advance("s", &q(&["b"])).unwrap();
        assert_eq!(sum.done_count, 1);
        assert_eq!(sum.remaining, q(&["a", "c"]));

        assert!(store.advance("unknown", &q(&["a"])).is_none());
    }

    #[test]
    fn empty_queue_registration_is_valid() {
        let store = SweepSessionStore::new_in_memory();
        let sum = store.register("s", q(&[]), 0);
        assert_eq!(sum.total, 0);
        assert!(sum.remaining.is_empty());
    }

    // ── store (durable) ──────────────────────────────────────────────────────

    #[test]
    fn store_persists_and_reloads_across_restart() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sweep_sessions.json");

        let store = SweepSessionStore::with_state(Some(path.clone()));
        store.register("s", q(&["a", "b", "c"]), 0);
        store.advance("s", &q(&["a"]));
        assert!(path.exists());

        // Simulate a Chord restart: a fresh store loads the same file.
        let restarted = SweepSessionStore::with_state(Some(path.clone()));
        let sum = restarted.get("s").expect("session should survive restart");
        assert_eq!(sum.done_count, 1);
        assert_eq!(sum.remaining, q(&["b", "c"]));
    }

    #[test]
    fn store_corrupt_file_starts_empty_no_panic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sweep_sessions.json");
        std::fs::write(&path, b"{ not valid ").unwrap();

        let store = SweepSessionStore::with_state(Some(path.clone()));
        assert!(store.get("s").is_none());
        // Still functional.
        let sum = store.register("s", q(&["a"]), 0);
        assert_eq!(sum.total, 1);
    }

    #[test]
    fn store_missing_file_starts_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("does-not-exist.json");
        let store = SweepSessionStore::with_state(Some(path));
        assert!(store.get("s").is_none());
    }

    #[test]
    fn in_memory_store_writes_nothing() {
        // new_in_memory has no state_path; register/advance never touch disk.
        let store = SweepSessionStore::new_in_memory();
        store.register("s", q(&["a"]), 0);
        assert!(store.advance("s", &q(&["a"])).is_some());
    }
}
