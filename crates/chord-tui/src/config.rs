//! Persisted configuration (S91 CTUI-01).
//!
//! Persists the instance (fleet) list + UI settings via a TOML file. SECRETS
//! ARE NEVER WRITTEN HERE — an instance stores only a [`SecretRef`] (a vault key
//! name), and the [`SecretValue`] type is not `Serialize`, so it is a compile
//! error to persist a secret value.
//!
//! Robustness:
//!   - **missing config**  → [`Config::default`] (empty fleet), the app then
//!     shows the add-instance prompt.
//!   - **corrupt config**  → the bad file is backed up to `<path>.corrupt-<ts>`,
//!     a fresh default is started, and a warning is surfaced. The fleet is never
//!     silently lost.
//!
//! No infrastructure endpoints are baked in; the file lives under the user's
//! config dir and the fleet is populated by the operator.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::secret::SecretRef;

/// Which control plane an instance speaks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum InstanceKind {
    /// A Chord proxy control endpoint.
    Chord,
    /// A Terminus-fleet control endpoint.
    Terminus,
}

/// A single configured control-plane instance. Contains NO secret value — only
/// a reference (`auth_secret_ref`) to a vault key resolved at connect time.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstanceConfig {
    /// Operator-facing label.
    pub name: String,
    /// Base URL of the control endpoint (e.g. a Chord control port). Supplied by
    /// the operator; never hardcoded.
    pub base_url: String,
    pub kind: InstanceKind,
    /// Vault key name for this instance's auth token. `None` → no auth (e.g. a
    /// jwt-disabled dev instance). Never a literal value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_secret_ref: Option<SecretRef>,
}

/// UI + behavior settings.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Settings {
    /// Per-instance health poll interval (seconds).
    pub poll_interval_secs: u64,
    /// Per-request timeout (seconds) — a dead instance is marked unreachable
    /// after this, and the event loop is never blocked on it.
    pub request_timeout_secs: u64,
    /// MASTER SAFETY FLAG. When false (the default), any stubbed / not-yet-wired
    /// mutation (S85 serving/coordinator/clean-swap) is INERT: the confirm flow
    /// runs but performs no real operation. Kept off until S85 wiring is
    /// verified live.
    pub enable_stubbed_mutations: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            poll_interval_secs: 5,
            request_timeout_secs: 4,
            // Stubbed mutations OFF by default — inert until S85 is wired.
            enable_stubbed_mutations: false,
        }
    }
}

/// Top-level persisted config.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Config {
    #[serde(default)]
    pub instances: Vec<InstanceConfig>,
    #[serde(default)]
    pub settings: Settings,
}

/// Outcome of loading config, carrying any non-fatal warning to surface as a
/// toast (e.g. corrupt-config recovery).
#[derive(Debug)]
pub struct LoadOutcome {
    pub config: Config,
    /// Non-fatal warning to show the operator, if any.
    pub warning: Option<String>,
}

impl Config {
    /// Default config-file path under the user's config dir. Falls back to the
    /// current directory if the platform dir is unavailable.
    pub fn default_path() -> PathBuf {
        dirs::config_dir()
            .map(|d| d.join("chord-tui").join("config.toml"))
            .unwrap_or_else(|| PathBuf::from("chord-tui.config.toml"))
    }

    /// Load from `path`. Missing → default (empty fleet). Corrupt → back up the
    /// bad file + start fresh with a warning. Never panics; never loses the
    /// fleet silently.
    pub fn load(path: &Path) -> LoadOutcome {
        let raw = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return LoadOutcome { config: Config::default(), warning: None };
            }
            Err(e) => {
                return LoadOutcome {
                    config: Config::default(),
                    warning: Some(format!("could not read config ({e}); starting with empty fleet")),
                };
            }
        };

        match toml::from_str::<Config>(&raw) {
            Ok(config) => LoadOutcome { config, warning: None },
            Err(e) => {
                // Corrupt: back up rather than lose it.
                let backup = Self::backup_corrupt(path, &raw);
                let warning = match backup {
                    Ok(bpath) => format!(
                        "config was corrupt ({e}); backed up to {} and started fresh",
                        bpath.display()
                    ),
                    Err(be) => format!(
                        "config was corrupt ({e}); backup ALSO failed ({be}); started fresh in memory"
                    ),
                };
                LoadOutcome { config: Config::default(), warning: Some(warning) }
            }
        }
    }

    fn backup_corrupt(path: &Path, raw: &str) -> std::io::Result<PathBuf> {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let mut backup = path.to_path_buf();
        let fname = format!(
            "{}.corrupt-{ts}",
            path.file_name().and_then(|f| f.to_str()).unwrap_or("config.toml")
        );
        backup.set_file_name(fname);
        std::fs::write(&backup, raw)?;
        Ok(backup)
    }

    /// Persist config to `path`, creating parent dirs. By construction this can
    /// only write [`SecretRef`] names, never [`crate::secret::SecretValue`]s.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let body = toml::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::secret::SecretRef;

    fn sample() -> Config {
        Config {
            instances: vec![InstanceConfig {
                name: "local-chord".into(),
                base_url: "http://127.0.0.1:18090".into(),
                kind: InstanceKind::Chord,
                auth_secret_ref: Some(SecretRef::new("CHORD_JWT_SECRET")),
            }],
            settings: Settings::default(),
        }
    }

    #[test]
    fn missing_config_yields_empty_fleet_no_warning() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("nope.toml");
        let out = Config::load(&p);
        assert!(out.instances_is_empty());
        assert!(out.warning.is_none());
    }

    #[test]
    fn roundtrip_preserves_instances() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.toml");
        let cfg = sample();
        cfg.save(&p).unwrap();
        let out = Config::load(&p);
        assert!(out.warning.is_none());
        assert_eq!(out.config, cfg);
    }

    /// NEGATIVE TEST: a resolved secret value must NEVER appear in the persisted
    /// config file. We serialize a config carrying a SecretRef whose *name*
    /// happens to be a key, and assert the file contains only the reference,
    /// and that the raw secret token text is absent.
    #[test]
    fn secrets_are_never_written_to_config() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.toml");
        let cfg = sample();
        cfg.save(&p).unwrap();
        let raw = std::fs::read_to_string(&p).unwrap();
        // The reference name is fine to persist:
        assert!(raw.contains("CHORD_JWT_SECRET"));
        // No value-bearing token could be present because SecretValue is not
        // Serialize; assert a would-be secret literal is absent as a guardrail.
        assert!(!raw.contains("hunter2"));
        assert!(!raw.to_lowercase().contains("bearer "));
    }

    #[test]
    fn corrupt_config_is_backed_up_and_starts_fresh() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.toml");
        std::fs::write(&p, "this is = = not valid toml [[[").unwrap();
        let out = Config::load(&p);
        assert!(out.instances_is_empty(), "fleet should reset to empty on corrupt");
        assert!(out.warning.as_ref().unwrap().contains("corrupt"));
        // A backup file must exist alongside.
        let backups: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains("corrupt-"))
            .collect();
        assert_eq!(backups.len(), 1, "exactly one corrupt backup expected");
        // And the original bad content is preserved in the backup (not lost).
        let bpath = backups[0].path();
        assert!(std::fs::read_to_string(bpath).unwrap().contains("not valid toml"));
    }

    #[test]
    fn stubbed_mutations_default_off() {
        assert!(!Settings::default().enable_stubbed_mutations);
    }

    // Small helper for tests.
    impl LoadOutcome {
        fn instances_is_empty(&self) -> bool {
            self.config.instances.is_empty()
        }
    }
}
