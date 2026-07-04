//! BSUIT-02: performance-curve report generator for `run_score_points`.
//!
//! The intake schema has a long-format `run_score_points` table — one row per
//! measurement point along an axis (`context_tokens`, `tool_count`, …):
//! `(profile_id, axis, x_value, x_label, metric, value)`. It exists so a model's
//! score/throughput can be plotted as a CURVE (e.g. "score vs. context depth"),
//! showing where a model falls off. This module turns those points into a
//! static, dependency-free HTML page: one hand-rolled inline `<svg>` line chart
//! per `(axis, metric)` group, one polyline per model.
//!
//! ## Split: pure rendering core + thin DB shell
//! Same idiom as the rest of Chord (e.g. `coding_proxy`'s pure
//! `select_with_fallback` vs. its `coding_select` handler): [`group_points`] and
//! [`render_report`] are PURE (no I/O) and unit-tested with synthetic points;
//! [`DbScorePointSource`] and the `batch-report` binary are the only parts that
//! touch Postgres. This keeps the SVG geometry testable without a DB and without
//! fabricating "live" data.
//!
//! ## Empty-data honesty
//! As of this writing `run_score_points` has ZERO rows (the context/agent sweeps
//! that populate it aren't the suite currently running against the intake DB —
//! only the coder suite is). That is expected, not a bug. [`render_report`] with
//! no groups renders a clear empty-state page ("no measurement points recorded
//! yet"); it NEVER fabricates sample data. The moment the populating sweeps run,
//! the same generator renders real curves with no code change.
//!
//! ## Design-system compliance (constellation.css)
//! Per this project's rule, every Lumina HTML output links
//! `/shared/constellation.css` and uses ITS variables/classes rather than
//! hardcoded colors. The generated page does exactly that: it links the
//! stylesheet and uses the documented `page` / `card` / `lumina-footer`
//! structural classes and the `var(--accent)` / `var(--text-secondary)` /
//! `var(--bg-secondary)` / `var(--border-color)` CSS variables.
//!
//! constellation.css itself was NOT reachable from the build host at authoring
//! time (the fleet-served `/shared/constellation.css` URL did not respond), so —
//! per the rule's explicit guidance for that case — this module only uses the
//! class/variable names documented in the project brief and does NOT invent a
//! categorical/series color palette it cannot verify. Instead, multiple series
//! in one chart are distinguished by cycling stroke DASH patterns
//! ([`SERIES_DASH`]) over the single documented `var(--accent)` stroke. Once
//! constellation.css's real categorical palette variables are known, swap the
//! per-series stroke to those (see [`series_stroke_style`]); the geometry and
//! structure need no change.

use std::collections::BTreeMap;

use async_trait::async_trait;

/// The `/shared/constellation.css` link every Lumina HTML output must include.
const CSS_LINK: &str = r#"<link rel="stylesheet" href="/shared/constellation.css">"#;

/// Stroke dash patterns cycled per-series to distinguish models within one
/// chart using only the single documented `var(--accent)` color (see the
/// module-level design-system note). Index = series index modulo len.
const SERIES_DASH: &[&str] = &["none", "6 3", "2 3", "8 3 2 3", "1 4", "10 4"];

/// One row from `run_score_points`, joined to `model_profiles` for the model
/// name. Long-format: one measurement point.
#[derive(Debug, Clone, PartialEq)]
pub struct ScorePoint {
    /// `model_profiles.model_name` (resolved via join on `profile_id`).
    pub model_id: String,
    pub profile_id: String,
    /// e.g. `"context_tokens"` or `"tool_count"`.
    pub axis: String,
    pub x_value: f64,
    /// Optional human label for the x tick (e.g. `"32k"`); falls back to
    /// `x_value` when empty.
    pub x_label: String,
    /// e.g. `"score"` or `"throughput"`.
    pub metric: String,
    pub value: f64,
}

/// One model's ordered points within a single `(axis, metric)` chart.
#[derive(Debug, Clone, PartialEq)]
pub struct Series {
    pub model_id: String,
    /// `(x_value, x_label, value)` sorted ascending by `x_value`.
    pub points: Vec<(f64, String, f64)>,
}

/// All series that share one `(axis, metric)` — i.e. one chart.
#[derive(Debug, Clone, PartialEq)]
pub struct SeriesGroup {
    pub axis: String,
    pub metric: String,
    pub series: Vec<Series>,
}

/// Group raw points into per-`(axis, metric)` charts, each holding one series
/// per model, points sorted by `x_value`. Pure. Deterministic ordering:
/// `(axis, metric)` and `model_id` are ordered lexicographically so the report
/// and its tests are stable.
pub fn group_points(points: &[ScorePoint]) -> Vec<SeriesGroup> {
    // (axis, metric) -> model_id -> Vec<(x, label, value)>
    let mut grouped: BTreeMap<(String, String), BTreeMap<String, Vec<(f64, String, f64)>>> =
        BTreeMap::new();
    for p in points {
        grouped
            .entry((p.axis.clone(), p.metric.clone()))
            .or_default()
            .entry(p.model_id.clone())
            .or_default()
            .push((p.x_value, p.x_label.clone(), p.value));
    }

    grouped
        .into_iter()
        .map(|((axis, metric), by_model)| {
            let series = by_model
                .into_iter()
                .map(|(model_id, mut pts)| {
                    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                    Series { model_id, points: pts }
                })
                .collect();
            SeriesGroup { axis, metric, series }
        })
        .collect()
}

/// Minimal XML/HTML text escaping for interpolated data (model names, labels).
fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

/// The per-series stroke style. Currently one documented color
/// (`var(--accent)`) with a cycled dash pattern (see the module-level
/// design-system note); the natural place to swap in constellation.css's real
/// categorical palette once known.
fn series_stroke_style(series_index: usize) -> (&'static str, &'static str) {
    let dash = SERIES_DASH[series_index % SERIES_DASH.len()];
    ("var(--accent)", dash)
}

// Fixed SVG geometry. Kept as constants so the render math is auditable.
const SVG_W: f64 = 720.0;
const SVG_H: f64 = 360.0;
const PAD_L: f64 = 56.0;
const PAD_R: f64 = 180.0; // room for the legend on the right
const PAD_T: f64 = 40.0;
const PAD_B: f64 = 44.0;

/// Render one `(axis, metric)` group as a self-contained inline `<svg>` line
/// chart. Pure. One `var(--accent)` polyline per model, dash-cycled; axes and
/// gridlines use `var(--border-color)` / `var(--text-secondary)`. Returns an
/// empty-state note instead of an SVG when the group has no plottable points.
pub fn render_svg_chart(group: &SeriesGroup) -> String {
    // Collect the numeric extent across every series in this chart.
    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    for s in &group.series {
        for (x, _, y) in &s.points {
            xs.push(*x);
            ys.push(*y);
        }
    }
    if xs.is_empty() {
        return format!(
            "<p class=\"muted\">No points for {} · {}.</p>",
            esc(&group.axis),
            esc(&group.metric)
        );
    }

    let (x_min, x_max) = min_max(&xs);
    let (mut y_min, mut y_max) = min_max(&ys);
    // Always include zero in the y-axis for score/throughput readability, and
    // guard a flat series (y_min == y_max) so we never divide by zero.
    if y_min > 0.0 {
        y_min = 0.0;
    }
    if (y_max - y_min).abs() < f64::EPSILON {
        y_max = y_min + 1.0;
    }

    let plot_w = SVG_W - PAD_L - PAD_R;
    let plot_h = SVG_H - PAD_T - PAD_B;
    let sx = |x: f64| -> f64 {
        if x_max <= x_min {
            PAD_L + plot_w / 2.0
        } else {
            PAD_L + (x - x_min) / (x_max - x_min) * plot_w
        }
    };
    let sy = |y: f64| -> f64 { PAD_T + plot_h - (y - y_min) / (y_max - y_min) * plot_h };

    let mut svg = String::new();
    svg.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{SVG_W}\" height=\"{SVG_H}\" \
         viewBox=\"0 0 {SVG_W} {SVG_H}\" role=\"img\" \
         aria-label=\"{} vs {}\">",
        esc(&group.metric),
        esc(&group.axis)
    ));

    // Axes (color via documented CSS variables; never hardcoded hex).
    svg.push_str(&format!(
        "<line x1=\"{PAD_L}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"var(--border-color)\" stroke-width=\"1\"/>",
        PAD_T + plot_h,
        PAD_L + plot_w,
        PAD_T + plot_h
    ));
    svg.push_str(&format!(
        "<line x1=\"{PAD_L}\" y1=\"{PAD_T}\" x2=\"{PAD_L}\" y2=\"{}\" stroke=\"var(--border-color)\" stroke-width=\"1\"/>",
        PAD_T + plot_h
    ));

    // Y gridlines + tick labels (5 steps).
    for i in 0..=4 {
        let t = i as f64 / 4.0;
        let yv = y_min + t * (y_max - y_min);
        let py = sy(yv);
        svg.push_str(&format!(
            "<line x1=\"{PAD_L}\" y1=\"{py:.1}\" x2=\"{}\" y2=\"{py:.1}\" \
             stroke=\"var(--border-color)\" stroke-width=\"1\" opacity=\"0.4\"/>",
            PAD_L + plot_w
        ));
        svg.push_str(&format!(
            "<text x=\"{}\" y=\"{:.1}\" fill=\"var(--text-secondary)\" font-size=\"11\" \
             text-anchor=\"end\">{:.2}</text>",
            PAD_L - 8.0,
            py + 3.0,
            yv
        ));
    }

    // X tick labels (min and max, using x_label where present).
    let x_label_for = |target: f64| -> String {
        group
            .series
            .iter()
            .flat_map(|s| s.points.iter())
            .find(|(x, lbl, _)| (*x - target).abs() < f64::EPSILON && !lbl.is_empty())
            .map(|(_, lbl, _)| lbl.clone())
            .unwrap_or_else(|| format!("{target:.0}"))
    };
    svg.push_str(&format!(
        "<text x=\"{PAD_L}\" y=\"{}\" fill=\"var(--text-secondary)\" font-size=\"11\" \
         text-anchor=\"start\">{}</text>",
        SVG_H - 16.0,
        esc(&x_label_for(x_min))
    ));
    svg.push_str(&format!(
        "<text x=\"{}\" y=\"{}\" fill=\"var(--text-secondary)\" font-size=\"11\" \
         text-anchor=\"end\">{}</text>",
        PAD_L + plot_w,
        SVG_H - 16.0,
        esc(&x_label_for(x_max))
    ));
    // Axis titles.
    svg.push_str(&format!(
        "<text x=\"{}\" y=\"{}\" fill=\"var(--text-secondary)\" font-size=\"12\" \
         text-anchor=\"middle\">{}</text>",
        PAD_L + plot_w / 2.0,
        SVG_H - 2.0,
        esc(&group.axis)
    ));

    // One polyline per series.
    for (idx, s) in group.series.iter().enumerate() {
        let (stroke, dash) = series_stroke_style(idx);
        let pts: String = s
            .points
            .iter()
            .map(|(x, _, y)| format!("{:.1},{:.1}", sx(*x), sy(*y)))
            .collect::<Vec<_>>()
            .join(" ");
        let dash_attr = if dash == "none" {
            String::new()
        } else {
            format!(" stroke-dasharray=\"{dash}\"")
        };
        svg.push_str(&format!(
            "<polyline points=\"{pts}\" fill=\"none\" stroke=\"{stroke}\" \
             stroke-width=\"2\"{dash_attr}/>"
        ));
        // Point markers.
        for (x, _, y) in &s.points {
            svg.push_str(&format!(
                "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"2.5\" fill=\"{stroke}\"/>",
                sx(*x),
                sy(*y)
            ));
        }
        // Legend entry (right gutter).
        let ly = PAD_T + 6.0 + idx as f64 * 20.0;
        let lx = PAD_L + plot_w + 16.0;
        let ldash = if dash == "none" {
            String::new()
        } else {
            format!(" stroke-dasharray=\"{dash}\"")
        };
        svg.push_str(&format!(
            "<line x1=\"{lx}\" y1=\"{ly:.1}\" x2=\"{}\" y2=\"{ly:.1}\" stroke=\"{stroke}\" \
             stroke-width=\"2\"{ldash}/>",
            lx + 22.0
        ));
        svg.push_str(&format!(
            "<text x=\"{}\" y=\"{:.1}\" fill=\"var(--text-secondary)\" font-size=\"11\">{}</text>",
            lx + 28.0,
            ly + 4.0,
            esc(&s.model_id)
        ));
    }

    svg.push_str("</svg>");
    svg
}

fn min_max(vals: &[f64]) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in vals {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    (min, max)
}

/// Render the full HTML report page. Pure. `generated_at` is injected (not read
/// from the clock here) so the output is deterministic and unit-testable. Links
/// constellation.css and uses only its documented classes/variables (see the
/// module-level design-system note). With no groups, renders an honest
/// empty-state page rather than fabricating data.
pub fn render_report(groups: &[SeriesGroup], generated_at: &str) -> String {
    let mut body = String::new();
    body.push_str("<div class=\"page\">");
    body.push_str(
        "<header class=\"page-header\"><h1>Model performance curves</h1>\
         <p class=\"muted\">Per-model score/throughput along measured axes \
         (source: intake <code>run_score_points</code>).</p></header>",
    );

    if groups.is_empty() {
        // Honest empty state — NOT fabricated sample data. See module docs.
        body.push_str(
            "<div class=\"card\"><h2>No data yet</h2>\
             <p class=\"muted\">No measurement points have been recorded in \
             <code>run_score_points</code> yet. This page renders one line chart \
             per <code>(axis, metric)</code> group the moment the context/agent \
             sweeps that populate that table run against the intake database — no \
             regeneration code change needed.</p></div>",
        );
    } else {
        for g in groups {
            body.push_str("<div class=\"card\">");
            body.push_str(&format!(
                "<h2>{} <span class=\"muted\">vs {}</span></h2>",
                esc(&g.metric),
                esc(&g.axis)
            ));
            body.push_str("<div class=\"chart-wrap\">");
            body.push_str(&render_svg_chart(g));
            body.push_str("</div></div>");
        }
    }

    body.push_str(&format!(
        "<footer class=\"lumina-footer\">Lumina Constellation · Chord batch-report · \
         generated {}</footer>",
        esc(generated_at)
    ));
    body.push_str("</div>");

    format!(
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"UTF-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n\
         <title>Model performance curves</title>\n{CSS_LINK}\n\
         <!-- constellation.css was unreachable at authoring time; classes \
         (page/card/lumina-footer/muted) and vars (--accent/--text-secondary/\
         --border-color) are the documented ones. chart-wrap scrolls wide SVGs. -->\n\
         <style>.chart-wrap{{overflow-x:auto}}</style>\n\
         </head>\n<body>\n{body}\n</body>\n</html>\n"
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Data source (Postgres, read-only)
// ─────────────────────────────────────────────────────────────────────────────

/// A score-point data-source failure. No infra detail — same discipline as the
/// selector errors elsewhere in this crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScorePointError {
    StoreUnavailable,
}

impl std::fmt::Display for ScorePointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("score-point store is temporarily unavailable")
    }
}

impl std::error::Error for ScorePointError {}

/// Source of `run_score_points` rows. Abstracted so unit tests use fixtures and
/// only a gated integration test / the binary hits the real read-only intake DB.
#[async_trait]
pub trait ScorePointSource: Send + Sync {
    async fn load_points(&self) -> Result<Vec<ScorePoint>, ScorePointError>;
}

/// Production [`ScorePointSource`]: reads `run_score_points` joined to
/// `model_profiles` over a `sqlx::PgPool`. Read-only. Mirrors the DB-source
/// pattern in `models::coding_selector` / `models::batch_suitability`.
pub struct DbScorePointSource {
    pool: sqlx::PgPool,
}

impl DbScorePointSource {
    pub fn new(pool: sqlx::PgPool) -> Self {
        DbScorePointSource { pool }
    }
}

#[async_trait]
impl ScorePointSource for DbScorePointSource {
    async fn load_points(&self) -> Result<Vec<ScorePoint>, ScorePointError> {
        use sqlx::Row;

        let rows = sqlx::query(
            "SELECT mp.model_name AS model_id, \
                    rsp.profile_id::text AS profile_id, \
                    rsp.axis, \
                    rsp.x_value::float8 AS x_value, \
                    coalesce(rsp.x_label, '') AS x_label, \
                    rsp.metric, \
                    rsp.value::float8 AS value \
             FROM run_score_points rsp \
             JOIN model_profiles mp ON mp.id = rsp.profile_id \
             ORDER BY rsp.axis, rsp.metric, mp.model_name, rsp.x_value",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| {
            tracing::error!(error = %e, "run_score_points query failed");
            ScorePointError::StoreUnavailable
        })?;

        Ok(rows
            .into_iter()
            .map(|r| ScorePoint {
                model_id: r.get("model_id"),
                profile_id: r.get("profile_id"),
                axis: r.get("axis"),
                x_value: r.get("x_value"),
                x_label: r.get("x_label"),
                metric: r.get("metric"),
                value: r.get("value"),
            })
            .collect())
    }
}

/// Fixed-fixture source for unit tests.
#[derive(Debug, Clone, Default)]
pub struct StaticScorePointSource {
    pub points: Vec<ScorePoint>,
}

#[async_trait]
impl ScorePointSource for StaticScorePointSource {
    async fn load_points(&self) -> Result<Vec<ScorePoint>, ScorePointError> {
        Ok(self.points.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(model: &str, axis: &str, x: f64, label: &str, metric: &str, v: f64) -> ScorePoint {
        ScorePoint {
            model_id: model.to_string(),
            profile_id: format!("id-{model}"),
            axis: axis.to_string(),
            x_value: x,
            x_label: label.to_string(),
            metric: metric.to_string(),
            value: v,
        }
    }

    #[test]
    fn group_points_splits_by_axis_metric_and_model_and_sorts_x() {
        let points = vec![
            pt("m1", "context_tokens", 32000.0, "32k", "score", 4.0),
            pt("m1", "context_tokens", 8000.0, "8k", "score", 4.5),
            pt("m2", "context_tokens", 8000.0, "8k", "score", 3.0),
            pt("m1", "context_tokens", 8000.0, "8k", "throughput", 55.0),
            pt("m1", "tool_count", 4.0, "4", "score", 3.8),
        ];
        let groups = group_points(&points);
        // (context_tokens,score), (context_tokens,throughput), (tool_count,score)
        assert_eq!(groups.len(), 3);
        let cs = groups
            .iter()
            .find(|g| g.axis == "context_tokens" && g.metric == "score")
            .unwrap();
        assert_eq!(cs.series.len(), 2, "two models in that chart");
        let m1 = cs.series.iter().find(|s| s.model_id == "m1").unwrap();
        // Sorted ascending by x_value: 8k then 32k.
        assert_eq!(m1.points[0].0, 8000.0);
        assert_eq!(m1.points[1].0, 32000.0);
    }

    #[test]
    fn render_report_empty_is_honest_no_fabricated_data() {
        let html = render_report(&[], "2026-07-04T00:00:00Z");
        assert!(html.contains(CSS_LINK), "must link constellation.css");
        assert!(html.contains("No data yet"));
        assert!(html.contains("run_score_points"));
        // No svg/polyline invented when there's nothing to plot.
        assert!(!html.contains("<polyline"));
        assert!(!html.contains("<svg"));
    }

    #[test]
    fn render_report_with_points_emits_svg_polyline_and_links_css() {
        let groups = group_points(&[
            pt("qwen3-coder:30b", "context_tokens", 8000.0, "8k", "score", 4.5),
            pt("qwen3-coder:30b", "context_tokens", 32000.0, "32k", "score", 3.9),
            pt("qwen3-coder:30b", "context_tokens", 128000.0, "128k", "score", 2.1),
            pt("gemma3:12b", "context_tokens", 8000.0, "8k", "score", 3.2),
            pt("gemma3:12b", "context_tokens", 32000.0, "32k", "score", 2.0),
        ]);
        let html = render_report(&groups, "2026-07-04T00:00:00Z");
        assert!(html.contains(CSS_LINK));
        assert!(html.contains("<svg"));
        // One polyline per model (2 series in the single chart).
        assert_eq!(html.matches("<polyline").count(), 2);
        // Uses CSS variables, never a hardcoded hex color.
        assert!(html.contains("var(--accent)"));
        assert!(!html.contains('#'), "no hardcoded hex colors allowed");
        // Model names are present (escaped) in the legend.
        assert!(html.contains("qwen3-coder:30b"));
    }

    #[test]
    fn render_svg_chart_handles_single_flat_series_without_div_by_zero() {
        // All-equal y values (flat curve) must not NaN or divide by zero.
        let g = SeriesGroup {
            axis: "tool_count".to_string(),
            metric: "score".to_string(),
            series: vec![Series {
                model_id: "flat".to_string(),
                points: vec![(1.0, "1".into(), 3.0), (2.0, "2".into(), 3.0)],
            }],
        };
        let svg = render_svg_chart(&g);
        assert!(svg.contains("<polyline"));
        assert!(!svg.contains("NaN"));
    }

    #[test]
    fn esc_escapes_markup() {
        assert_eq!(esc("a<b>&\"'"), "a&lt;b&gt;&amp;&quot;&#39;");
    }

    #[tokio::test]
    async fn empty_source_produces_empty_state_page() {
        let source = StaticScorePointSource::default();
        let points = source.load_points().await.unwrap();
        let groups = group_points(&points);
        let html = render_report(&groups, "t");
        assert!(html.contains("No data yet"));
    }
}
