//! BSUIT-02 binary: generate the `run_score_points` performance-curve report.
//!
//! Read-only. Connects to the intake DB via the same
//! `terminus_rs::config::intake_database_url()` resolver every other Chord DB
//! consumer uses (no literal DSN), loads `run_score_points`, and writes a static
//! HTML page (see [`chord_proxy::batch_report`]). If the table is empty — the
//! current state — it writes an honest empty-state page rather than fabricating
//! data.
//!
//! Usage:
//!   batch-report [OUTPUT_PATH]        # default: ./batch-performance-curves.html

use std::process::ExitCode;

use chord_proxy::batch_report::{
    group_points, render_report, DbScorePointSource, ScorePointSource,
};

#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt().with_env_filter("info").try_init().ok();

    let out_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "batch-performance-curves.html".to_string());

    let Some(db_url) = terminus_rs::config::intake_database_url() else {
        eprintln!(
            "no intake DB configured (set INTAKE_DATABASE_URL or DATABASE_URL) — cannot query \
             run_score_points"
        );
        return ExitCode::FAILURE;
    };

    let pool = match sqlx::PgPool::connect(&db_url).await {
        Ok(p) => p,
        Err(e) => {
            eprintln!("intake DB connect failed: {e}");
            return ExitCode::FAILURE;
        }
    };

    let source = DbScorePointSource::new(pool);
    let points = match source.load_points().await {
        Ok(p) => p,
        Err(e) => {
            eprintln!("query failed: {e}");
            return ExitCode::FAILURE;
        }
    };

    let groups = group_points(&points);
    let generated_at = chrono::Utc::now().to_rfc3339();
    let html = render_report(&groups, &generated_at);

    if let Err(e) = std::fs::write(&out_path, html) {
        eprintln!("failed to write {out_path}: {e}");
        return ExitCode::FAILURE;
    }

    eprintln!(
        "wrote {out_path}: {} point(s) across {} chart group(s){}",
        points.len(),
        groups.len(),
        if points.is_empty() {
            " (empty-state page — run_score_points has no rows yet)"
        } else {
            ""
        }
    );
    ExitCode::SUCCESS
}
