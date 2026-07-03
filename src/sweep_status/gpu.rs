//! GPU busy% reader — `/sys/class/drm/card*/device/gpu_busy_percent`.
//!
//! Async (`tokio::fs`) since this runs inside the sweep-monitor's async tick.
//! The root directory is configurable ([`crate::sweep_status::config::DRM_ROOT_ENV`])
//! so tests can point at a synthetic sysfs tree instead of the real one.

/// Scan `{drm_root}/card<N>/device/gpu_busy_percent` for every bare `cardN`
/// entry (connector subdirectories like `card0-DP-1` are skipped) and return
/// the highest reading found. `None` when the root is unreadable or no card
/// exposes a busy-percent counter (e.g. non-AMD host, or sysfs path moved) —
/// callers must treat this as "unknown", not "0% busy".
pub async fn read_gpu_busy_percent(drm_root: &std::path::Path) -> Option<f64> {
    let mut entries = tokio::fs::read_dir(drm_root).await.ok()?;
    let mut best: Option<f64> = None;
    while let Ok(Some(entry)) = entries.next_entry().await {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !is_bare_card_dir(&name) {
            continue;
        }
        let path = entry.path().join("device/gpu_busy_percent");
        if let Ok(content) = tokio::fs::read_to_string(&path).await {
            if let Some(v) = parse_busy_percent(&content) {
                best = Some(best.map_or(v, |b: f64| b.max(v)));
            }
        }
    }
    best
}

/// `cardN` (bare numeric suffix) — true; anything else (`card0-DP-1`,
/// `renderD128`, ...) — false.
fn is_bare_card_dir(name: &str) -> bool {
    match name.strip_prefix("card") {
        Some(rest) => !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()),
        None => false,
    }
}

/// Parse a `gpu_busy_percent` sysfs file's contents (e.g. `"99\n"`) into a
/// percentage. `None` on anything that doesn't parse as a finite, non-negative
/// number.
fn parse_busy_percent(raw: &str) -> Option<f64> {
    let v: f64 = raw.trim().parse().ok()?;
    (v.is_finite() && v >= 0.0).then_some(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn bare_card_dirs_match() {
        assert!(is_bare_card_dir("card0"));
        assert!(is_bare_card_dir("card1"));
        assert!(is_bare_card_dir("card12"));
    }

    #[test]
    fn connector_and_render_dirs_do_not_match() {
        assert!(!is_bare_card_dir("card0-DP-1"));
        assert!(!is_bare_card_dir("renderD128"));
        assert!(!is_bare_card_dir("card"));
        assert!(!is_bare_card_dir("cardX"));
    }

    #[test]
    fn parse_busy_percent_typical() {
        assert_eq!(parse_busy_percent("99\n"), Some(99.0));
        assert_eq!(parse_busy_percent("0"), Some(0.0));
        assert_eq!(parse_busy_percent("  42  \n"), Some(42.0));
    }

    #[test]
    fn parse_busy_percent_rejects_garbage() {
        assert_eq!(parse_busy_percent("not a number"), None);
        assert_eq!(parse_busy_percent(""), None);
        assert_eq!(parse_busy_percent("-5"), None);
    }

    #[tokio::test]
    async fn reads_highest_busy_percent_across_cards() {
        let tmp = tempfile::tempdir().unwrap();
        for (card, busy) in [("card0", "12"), ("card1", "87"), ("card0-DP-1", "999")] {
            let dir = tmp.path().join(card).join("device");
            fs::create_dir_all(&dir).unwrap();
            fs::write(dir.join("gpu_busy_percent"), busy).unwrap();
        }
        let busy = read_gpu_busy_percent(tmp.path()).await;
        assert_eq!(busy, Some(87.0), "connector dir card0-DP-1 must be ignored, max of real cards is 87");
    }

    #[tokio::test]
    async fn missing_root_is_none() {
        let busy = read_gpu_busy_percent(std::path::Path::new("/nonexistent/drm/root")).await;
        assert!(busy.is_none());
    }

    #[tokio::test]
    async fn root_with_no_busy_files_is_none() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("card0/device")).unwrap();
        let busy = read_gpu_busy_percent(tmp.path()).await;
        assert!(busy.is_none());
    }
}
