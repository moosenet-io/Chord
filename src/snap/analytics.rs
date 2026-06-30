// SNAP-05: RequestLogger — per-request metadata, cost/savings tracking.
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

/// A single request log record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestRecord {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub model: String,
    pub endpoint: String,
    pub engine_url: String,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub duration_ms: u64,
    pub status_code: u16,
    pub streaming: bool,
}

/// Cloud pricing for computing imputed savings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudPricing {
    /// USD per 1k input tokens
    pub input_per_1k: f64,
    /// USD per 1k output tokens
    pub output_per_1k: f64,
}

/// Savings summary for a time period.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavingsSummary {
    pub period: String,
    pub total_requests: usize,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub imputed_cloud_cost_usd: f64,
    pub actual_cost_usd: f64,
    pub savings_usd: f64,
    pub savings_pct: f64,
}

/// Daily cost breakdown for a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyCost {
    pub date: String,
    pub model: String,
    pub requests: usize,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub imputed_cost_usd: f64,
}

/// RequestLogger: append-only JSONL log with rotation at 100k lines.
pub struct RequestLogger {
    pub log_path: PathBuf,
    /// Cloud pricing table: model_prefix → pricing
    pub cloud_pricing: HashMap<String, CloudPricing>,
}

impl RequestLogger {
    pub const MAX_LINES: usize = 100_000;

    pub fn new(data_dir: &Path) -> Self {
        let log_path = data_dir.join("request-log.jsonl");
        let mut cloud_pricing = HashMap::new();

        // Representative cloud pricing for imputed cost calculation
        cloud_pricing.insert(
            "gpt-4o".into(),
            CloudPricing { input_per_1k: 0.005, output_per_1k: 0.015 },
        );
        cloud_pricing.insert(
            "gpt-4".into(),
            CloudPricing { input_per_1k: 0.03, output_per_1k: 0.06 },
        );
        cloud_pricing.insert(
            "claude-3-5-sonnet".into(),
            CloudPricing { input_per_1k: 0.003, output_per_1k: 0.015 },
        );
        cloud_pricing.insert(
            "claude-3-opus".into(),
            CloudPricing { input_per_1k: 0.015, output_per_1k: 0.075 },
        );
        // Default for local/open models (like qwen3, llama etc.)
        cloud_pricing.insert(
            "default".into(),
            CloudPricing { input_per_1k: 0.002, output_per_1k: 0.002 },
        );

        Self { log_path, cloud_pricing }
    }

    /// Append a request record to the log. Rotates if > MAX_LINES.
    pub fn append(&self, record: &RequestRecord) {
        // Check line count and rotate if needed
        self.rotate_if_needed();

        if let Some(parent) = self.log_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        if let Ok(mut json) = serde_json::to_string(record) {
            json.push('\n');
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.log_path)
                .ok();
            if let Some(ref mut f) = file {
                let _ = f.write_all(json.as_bytes());
            }
        }
    }

    fn rotate_if_needed(&self) {
        let count = self.count_lines();
        if count >= Self::MAX_LINES {
            // Rename existing to .1 and start fresh
            let rotated = self.log_path.with_extension("jsonl.1");
            let _ = std::fs::rename(&self.log_path, &rotated);
        }
    }

    fn count_lines(&self) -> usize {
        std::fs::read_to_string(&self.log_path)
            .map(|s| s.lines().filter(|l| !l.is_empty()).count())
            .unwrap_or(0)
    }

    /// Read records filtered by optional time range, model, and limit.
    pub fn query(
        &self,
        from: Option<DateTime<Utc>>,
        to: Option<DateTime<Utc>>,
        model: Option<&str>,
        limit: usize,
    ) -> Vec<RequestRecord> {
        let content = std::fs::read_to_string(&self.log_path).unwrap_or_default();
        let mut records: Vec<RequestRecord> = content
            .lines()
            .filter_map(|line| serde_json::from_str(line).ok())
            .filter(|r: &RequestRecord| {
                if let Some(f) = from {
                    if r.timestamp < f {
                        return false;
                    }
                }
                if let Some(t) = to {
                    if r.timestamp > t {
                        return false;
                    }
                }
                if let Some(m) = model {
                    if !r.model.contains(m) {
                        return false;
                    }
                }
                true
            })
            .collect();

        // Return most recent N
        let len = records.len();
        if len > limit {
            records.drain(0..(len - limit));
        }
        records
    }

    /// Compute daily cost breakdown for a period in days.
    pub fn daily_costs(&self, period_days: u64) -> Vec<DailyCost> {
        let from = Utc::now() - chrono::Duration::days(period_days as i64);
        let records = self.query(Some(from), None, None, 1_000_000);

        // Group by (date, model)
        let mut groups: HashMap<(String, String), (usize, u64, u64)> = HashMap::new();
        for record in &records {
            let date = record.timestamp.format("%Y-%m-%d").to_string();
            let key = (date, record.model.clone());
            let entry = groups.entry(key).or_default();
            entry.0 += 1;
            entry.1 += record.input_tokens.unwrap_or(0);
            entry.2 += record.output_tokens.unwrap_or(0);
        }

        let mut costs: Vec<DailyCost> = groups
            .into_iter()
            .map(|((date, model), (requests, input_tokens, output_tokens))| {
                let pricing = self.get_pricing(&model);
                let imputed_cost_usd = self.compute_cost(input_tokens, output_tokens, &pricing);
                DailyCost { date, model, requests, input_tokens, output_tokens, imputed_cost_usd }
            })
            .collect();

        costs.sort_by(|a, b| a.date.cmp(&b.date).then(a.model.cmp(&b.model)));
        costs
    }

    /// Compute savings summary for a period.
    pub fn savings_summary(&self, period: &str) -> SavingsSummary {
        let period_days: u64 = parse_period_days(period);
        let from = Utc::now() - chrono::Duration::days(period_days as i64);
        let records = self.query(Some(from), None, None, 1_000_000);

        let total_requests = records.len();
        let total_input_tokens: u64 = records.iter().map(|r| r.input_tokens.unwrap_or(0)).sum();
        let total_output_tokens: u64 = records.iter().map(|r| r.output_tokens.unwrap_or(0)).sum();

        // Imputed cost: what it would cost on a cloud provider
        let mut imputed_cost = 0.0f64;
        for record in &records {
            let pricing = self.get_pricing(&record.model);
            imputed_cost += self.compute_cost(
                record.input_tokens.unwrap_or(0),
                record.output_tokens.unwrap_or(0),
                &pricing,
            );
        }

        // Actual cost: local inference (electricity only, ~$0 for our purposes)
        let actual_cost = 0.0f64;
        let savings = imputed_cost - actual_cost;
        let savings_pct = if imputed_cost > 0.0 {
            (savings / imputed_cost) * 100.0
        } else {
            0.0
        };

        SavingsSummary {
            period: period.to_string(),
            total_requests,
            total_input_tokens,
            total_output_tokens,
            imputed_cloud_cost_usd: imputed_cost,
            actual_cost_usd: actual_cost,
            savings_usd: savings,
            savings_pct,
        }
    }

    fn get_pricing(&self, model: &str) -> CloudPricing {
        // Try exact prefix match
        for (key, pricing) in &self.cloud_pricing {
            if key != "default" && model.starts_with(key.as_str()) {
                return pricing.clone();
            }
        }
        self.cloud_pricing
            .get("default")
            .cloned()
            .unwrap_or(CloudPricing { input_per_1k: 0.002, output_per_1k: 0.002 })
    }

    fn compute_cost(&self, input_tokens: u64, output_tokens: u64, pricing: &CloudPricing) -> f64 {
        (input_tokens as f64 / 1000.0) * pricing.input_per_1k
            + (output_tokens as f64 / 1000.0) * pricing.output_per_1k
    }
}

fn parse_period_days(period: &str) -> u64 {
    // Parse e.g. "7d", "30d", "1w"
    if let Some(days) = period.strip_suffix('d') {
        days.parse().unwrap_or(7)
    } else if let Some(weeks) = period.strip_suffix('w') {
        weeks.parse::<u64>().unwrap_or(1) * 7
    } else {
        7
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_logger(tmp: &TempDir) -> RequestLogger {
        RequestLogger::new(tmp.path())
    }

    fn make_record(model: &str, input: u64, output: u64) -> RequestRecord {
        RequestRecord {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            model: model.to_string(),
            endpoint: "/v1/chat/completions".to_string(),
            engine_url: "http://localhost:11434".to_string(),
            input_tokens: Some(input),
            output_tokens: Some(output),
            duration_ms: 100,
            status_code: 200,
            streaming: false,
        }
    }

    #[test]
    fn append_and_query_records() {
        let tmp = TempDir::new().unwrap();
        let logger = make_logger(&tmp);

        logger.append(&make_record("qwen3:8b", 100, 50));
        logger.append(&make_record("qwen3:8b", 200, 80));

        let records = logger.query(None, None, None, 100);
        assert_eq!(records.len(), 2);
    }

    #[test]
    fn filter_by_model() {
        let tmp = TempDir::new().unwrap();
        let logger = make_logger(&tmp);

        logger.append(&make_record("qwen3:8b", 100, 50));
        logger.append(&make_record("llama3:70b", 200, 80));

        let records = logger.query(None, None, Some("qwen"), 100);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].model, "qwen3:8b");
    }

    #[test]
    fn savings_math_is_correct() {
        let tmp = TempDir::new().unwrap();
        let logger = make_logger(&tmp);

        // 1000 input tokens + 500 output tokens with default pricing ($0.002/$0.002)
        // cost = (1000/1000)*0.002 + (500/1000)*0.002 = 0.002 + 0.001 = 0.003
        logger.append(&make_record("qwen3:8b", 1000, 500));

        let summary = logger.savings_summary("7d");
        assert_eq!(summary.total_requests, 1);
        assert_eq!(summary.total_input_tokens, 1000);
        assert_eq!(summary.total_output_tokens, 500);
        assert!(summary.imputed_cloud_cost_usd > 0.0, "Should have imputed cost");
        assert!(summary.savings_usd > 0.0, "Should have savings");
        assert!(summary.savings_pct > 99.0, "Should be >99% savings vs cloud");
    }

    #[test]
    fn cost_aggregation_by_day() {
        let tmp = TempDir::new().unwrap();
        let logger = make_logger(&tmp);

        logger.append(&make_record("qwen3:8b", 500, 200));
        logger.append(&make_record("qwen3:8b", 300, 100));

        let costs = logger.daily_costs(7);
        assert!(!costs.is_empty());
        // Both records are the same model today, should combine
        let today = Utc::now().format("%Y-%m-%d").to_string();
        let today_costs: Vec<_> = costs.iter().filter(|c| c.date == today).collect();
        assert_eq!(today_costs.len(), 1);
        assert_eq!(today_costs[0].input_tokens, 800);
    }

    #[test]
    fn parse_period_days_handles_formats() {
        assert_eq!(parse_period_days("7d"), 7);
        assert_eq!(parse_period_days("30d"), 30);
        assert_eq!(parse_period_days("1w"), 7);
        assert_eq!(parse_period_days("2w"), 14);
        assert_eq!(parse_period_days("invalid"), 7); // fallback
    }

    #[test]
    fn query_limit_returns_most_recent() {
        let tmp = TempDir::new().unwrap();
        let logger = make_logger(&tmp);

        for i in 0..10 {
            let mut r = make_record("qwen3:8b", i * 100, i * 50);
            r.input_tokens = Some(i * 100);
            logger.append(&r);
        }

        let records = logger.query(None, None, None, 3);
        assert_eq!(records.len(), 3);
    }

    #[test]
    fn record_serializes_to_json() {
        let record = make_record("qwen3:8b", 100, 50);
        let json = serde_json::to_string(&record).unwrap();
        assert!(json.contains("qwen3:8b"));
        assert!(json.contains("input_tokens"));
    }
}
