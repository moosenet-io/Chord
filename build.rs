//! Build script: embed the short git commit hash and the build timestamp into
//! the binary as compile-time env vars. No external crates — uses `std` and a
//! best-effort `git rev-parse`. Both values degrade gracefully to "unknown".

use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    // GIT_HASH — short commit hash, "unknown" if git is unavailable.
    let git_hash = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=GIT_HASH={git_hash}");

    // BUILD_TIME — RFC3339 / ISO-8601 UTC, derived from the system clock with no
    // external date crate. Falls back to "unknown" if the clock is before epoch.
    let build_time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|d| format_rfc3339_utc(d.as_secs()))
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=BUILD_TIME={build_time}");

    // Rebuild whenever the checked-out commit changes. Watching `.git/HEAD`
    // alone is NOT enough: on a fast-forward push the symref content is
    // unchanged and only the branch ref file moves, so cargo never re-runs and
    // GIT_HASH goes stale. Watch HEAD *and* the ref it points at (plus
    // packed-refs, where refs live after `git gc`).
    emit_git_rerun_triggers();
}

/// Emit `cargo:rerun-if-changed` for the git refs that determine `GIT_HASH`,
/// so a fast-forward (which leaves `.git/HEAD` untouched) still rebuilds.
fn emit_git_rerun_triggers() {
    // Resolve the repo's git dir robustly (handles worktrees where `.git` is a
    // file pointer). Fall back to the relative workspace `.git` if git is
    // unavailable.
    let git_dir = Command::new("git")
        .args(["rev-parse", "--absolute-git-dir"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "../../.git".to_string());

    println!("cargo:rerun-if-changed={git_dir}/HEAD");
    println!("cargo:rerun-if-changed={git_dir}/packed-refs");

    // Follow HEAD's symref (e.g. "ref: refs/heads/main") to the concrete ref
    // file so a fast-forward of that branch triggers a rebuild.
    if let Ok(head) = std::fs::read_to_string(format!("{git_dir}/HEAD")) {
        if let Some(ref_path) = head.strip_prefix("ref:").map(str::trim) {
            println!("cargo:rerun-if-changed={git_dir}/{ref_path}");
        }
    }
}

/// Format a Unix timestamp (seconds) as RFC3339 UTC, e.g. `2026-06-12T14:03:55Z`.
/// Pure integer math — no chrono dependency in the build graph.
fn format_rfc3339_utc(secs: u64) -> String {
    let days = secs / 86_400;
    let rem = secs % 86_400;
    let (hour, minute, second) = (rem / 3600, (rem % 3600) / 60, rem % 60);

    // Civil-from-days algorithm (Howard Hinnant), epoch 1970-01-01.
    let z = days as i64 + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as i64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { year + 1 } else { year };

    format!(
        "{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z"
    )
}
