//! Root-cause fix for a comment-leak audit finding (see the `moosenet/chord`
//! remediation branch `fix/pii-comment-scrub-self-check`): a couple of Rust
//! comments in this repo used to name real internal infrastructure (a
//! specific host, a specific mount path). This test makes that class of leak
//! a hard CI failure instead of something that only gets caught by a manual
//! audit sweep.
//!
//! Chord doesn't depend on Terminus's crate, so the patterns here are a local
//! re-implementation — kept in lockstep, on purpose, with the categories in
//! Terminus's `src/github/pii.rs` gate (`container_id`, `internal_host`,
//! `internal_domain`, `internal_path`, `private_ip`) so the two repos flag the
//! same things.
//!
//! Deliberate test-fixture literals that would otherwise trip this scan can be
//! exempted with a trailing `// pii-test-fixture` comment on the same line —
//! the same marker convention Terminus's repo already uses.
//!
//! NOTE on `internal_path`: Terminus's canonical list includes `<path>/`
//! because from Terminus's point of view that's *someone else's* infra. From
//! inside Chord's own repo, `<path>/` is Chord's own publicly-documented
//! default install path (it appears ~24x as the `model_registry_path`
//! default) — flagging it here would just be Chord tripping over its own
//! shadow, not a leak. It's intentionally left out of the pattern below; see
//! the remediation report for the full rationale.

use regex::Regex;
use std::path::{Path, PathBuf};

struct PiiPatterns {
    container_id: Regex,
    internal_host: Regex,
    internal_domain: Regex,
    internal_path: Regex,
    private_ip: Regex,
}

fn patterns() -> PiiPatterns {
    PiiPatterns {
        container_id: Regex::new(r"\bCT\d{3}\b").expect("container_id regex"),
        internal_host: Regex::new(r"(?i)\b(?:<host>|<host>|<host>|<host>|<host>)\b") // pii-test-fixture
            .expect("internal_host regex"),
        internal_domain: Regex::new(r"moosenet\.online|moosenet\.local") // pii-test-fixture
            .expect("internal_domain regex"),
        // Deliberately omits `<path>/` — see module doc.
        internal_path: Regex::new(r"<path>/|<path>/|<path>/|/mnt/<host>/") // pii-test-fixture
            .expect("internal_path regex"),
        private_ip: Regex::new(
            r"\b(?:192\.168|10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b",
        )
        .expect("private_ip regex"),
    }
}

#[derive(Debug)]
struct Violation {
    file: PathBuf,
    line: usize,
    category: &'static str,
    snippet: String,
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{} [{}]: {}",
            self.file.display(),
            self.line,
            self.category,
            self.snippet.trim()
        )
    }
}

/// Recursively collect every `.rs` file under `dir`, skipping any `target`
/// directory (build artifacts, not source we own) and the `.git` directory.
fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name == "target" || name == ".git" {
                continue;
            }
            collect_rs_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

/// Scan a single file's contents for PII, honoring the `// pii-test-fixture`
/// exemption marker (strip any line carrying it before matching).
fn scan_file(path: &Path, contents: &str, p: &PiiPatterns, out: &mut Vec<Violation>) {
    for (idx, raw_line) in contents.lines().enumerate() {
        if raw_line.contains("pii-test-fixture") {
            continue;
        }
        let line_no = idx + 1;

        let mut push = |category: &'static str, m: &str| {
            out.push(Violation {
                file: path.to_path_buf(),
                line: line_no,
                category,
                snippet: m.to_string(),
            });
        };

        for m in p.container_id.find_iter(raw_line) {
            push("container_id", m.as_str());
        }
        for m in p.internal_host.find_iter(raw_line) {
            push("internal_host", m.as_str());
        }
        for m in p.internal_domain.find_iter(raw_line) {
            push("internal_domain", m.as_str());
        }
        for m in p.internal_path.find_iter(raw_line) {
            push("internal_path", m.as_str());
        }
        for m in p.private_ip.find_iter(raw_line) {
            push("private_ip", m.as_str());
        }
    }
}

/// Root-cause self-check: walk Chord's own source tree (this repo, including
/// the `chord-tui` workspace member) and assert it contains none of the
/// internal-infra identifiers that a public release must never carry.
#[test]
fn no_pii_leaks_in_own_source() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let p = patterns();

    let mut files = Vec::new();
    for sub in ["src", "tests", "crates", "build.rs"] {
        let path = manifest_dir.join(sub);
        if path.is_dir() {
            collect_rs_files(&path, &mut files);
        } else if path.is_file() && path.extension().and_then(|e| e.to_str()) == Some("rs") {
            files.push(path);
        }
    }

    assert!(
        !files.is_empty(),
        "self-check found zero .rs files to scan — walker is broken, not the repo"
    );

    let mut violations = Vec::new();
    for file in &files {
        let Ok(contents) = std::fs::read_to_string(file) else {
            continue;
        };
        scan_file(file, &contents, &p, &mut violations);
    }

    assert!(
        violations.is_empty(),
        "PII self-check found {} violation(s) — tag deliberate test fixtures with \
         `// pii-test-fixture`, otherwise scrub the real leak:\n{}",
        violations.len(),
        violations
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// Unit-level check that the patterns themselves actually fire — this is what
/// would catch a regression in the scanning logic (as opposed to
/// `no_pii_leaks_in_own_source`, which catches a regression in the source).
#[test]
fn scanner_detects_each_category() {
    let p = patterns();
    let mut out = Vec::new();

    scan_file(
        Path::new("synthetic.rs"),
        "// deployed to <host> for testing // pii-test-fixture\n",
        &p,
        &mut out,
    );
    assert!(
        out.is_empty(),
        "pii-test-fixture marker must suppress the match: {out:?}"
    );

    let mut out = Vec::new();
    scan_file(
        Path::new("synthetic.rs"),
        "// deployed to <host> for testing\n", // pii-test-fixture
        &p,
        &mut out,
    );
    assert!(out.iter().any(|v| v.category == "container_id"));

    let mut out = Vec::new();
    scan_file(Path::new("synthetic.rs"), "// ran on <host> last night\n", &p, &mut out); // pii-test-fixture
    assert!(out.iter().any(|v| v.category == "internal_host"));

    let mut out = Vec::new();
    scan_file(
        Path::new("synthetic.rs"),
        "// see git.example.com for the repo\n", // pii-test-fixture
        &p,
        &mut out,
    );
    assert!(out.iter().any(|v| v.category == "internal_domain"));

    let mut out = Vec::new();
    scan_file(
        Path::new("synthetic.rs"),
        "// config lives at /mnt/<host>/some-share\n", // pii-test-fixture
        &p,
        &mut out,
    );
    assert!(out.iter().any(|v| v.category == "internal_path"));

    let mut out = Vec::new();
    scan_file(
        Path::new("synthetic.rs"),
        "// backend at <internal-ip> for now\n", // pii-test-fixture
        &p,
        &mut out,
    );
    assert!(out.iter().any(|v| v.category == "private_ip"));
}
