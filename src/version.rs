//! Compile-time version metadata for the `chord-proxy` crate.
//!
//! `commit()` / `build_time()` are emitted by `build.rs`. `terminus_version()`
//! reports the compiled-in version of the in-process `terminus-rs` tool library.

/// Crate semantic version (e.g. `1.0.0`).
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Short git commit the binary was built from, or "unknown".
pub fn commit() -> &'static str {
    option_env!("GIT_HASH").unwrap_or("unknown")
}

/// RFC3339 UTC timestamp of the build, or "unknown".
pub fn build_time() -> &'static str {
    option_env!("BUILD_TIME").unwrap_or("unknown")
}

/// Compiled-in version of the `terminus-rs` tool library running in-process.
pub fn terminus_version() -> &'static str {
    terminus_rs::VERSION
}

/// One-line version string for `--version`:
/// `chord-proxy 1.0.0 (<commit>, terminus-rs <terminus_version>)`.
pub fn version_line() -> String {
    format!(
        "chord-proxy {} ({}, terminus-rs {})",
        version(),
        commit(),
        terminus_version()
    )
}
