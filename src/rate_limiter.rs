//! Per-user rate limiting for the Chord proxy.
//!
//! Enforces daily call budgets per user role:
//! - Admin: unlimited (all limits set to u32::MAX)
//! - User: configurable via CHORD_RATE_LLM_USER / CHORD_RATE_TOOL_USER / CHORD_RATE_DEEP_USER
//! - Guest: configurable via CHORD_RATE_LLM_GUEST / CHORD_RATE_TOOL_GUEST / CHORD_RATE_DEEP_GUEST
//!
//! Counters are in-memory; they reset at midnight UTC (lazy check on each request).
//! A background task (optional) can call reset_daily() to sweep all counters.
//!
//! Deep-model calls (model name contains "deep", "opus", or "120b") count against
//! both the `llm` budget and the `deep` budget.
//!
//! All timestamps are in seconds since UNIX_EPOCH (UTC).

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::config::RateLimitConfig;

// ── Time helpers ──────────────────────────────────────────────────────────────

/// Returns current UTC time as seconds since UNIX_EPOCH.
fn now_utc_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Returns the day number (days since UNIX_EPOCH) for the given UTC timestamp.
/// Used for lazy daily reset checks.
fn utc_day(secs: u64) -> u64 {
    secs / 86_400
}

/// Returns the Unix timestamp of the next midnight UTC after `now_secs`.
fn next_midnight_utc(now_secs: u64) -> u64 {
    let today_start = utc_day(now_secs) * 86_400;
    today_start + 86_400
}

// ── Call type ────────────────────────────────────────────────────────────────

/// The type of call being rate-limited.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallType {
    /// LLM chat-completion call
    Llm,
    /// Tool call (list / call / discover each count as 1)
    Tool,
    /// Deep-model LLM call (subset of Llm; counted against both llm and deep budgets)
    Deep,
}

impl CallType {
    /// Returns true if the model name string qualifies as a deep model.
    pub fn is_deep_model(model: &str) -> bool {
        let m = model.to_lowercase();
        m.contains("deep") || m.contains("opus") || m.contains("120b")
    }
}

// ── User role ─────────────────────────────────────────────────────────────────

/// The user's role, extracted from JWT claims.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UserRole {
    Admin,
    User,
    Guest,
}

impl UserRole {
    /// Parse from a role string (JWT claim "role"). Defaults to User.
    pub fn from_claim(claim: Option<&str>) -> Self {
        match claim {
            Some("admin") => UserRole::Admin,
            Some("guest") => UserRole::Guest,
            _ => UserRole::User,
        }
    }
}

// ── Per-user counters ─────────────────────────────────────────────────────────

#[derive(Debug)]
struct UserCounters {
    /// The UTC day number (days since UNIX_EPOCH) these counters are valid for.
    /// Uses sentinel 0 to indicate "never set".
    day: u64,
    llm: u32,
    tool: u32,
    deep: u32,
}

impl Default for UserCounters {
    fn default() -> Self {
        UserCounters { day: 0, llm: 0, tool: 0, deep: 0 }
    }
}

impl UserCounters {
    /// Reset counters if the stored day is not today.
    fn refresh_for_today(&mut self, today_day: u64) {
        if self.day != today_day {
            self.llm = 0;
            self.tool = 0;
            self.deep = 0;
            self.day = today_day;
        }
    }
}

// ── Rate limit result ─────────────────────────────────────────────────────────

/// Outcome of a rate-limit check.
pub struct RateLimitResult {
    /// Whether the call is allowed.
    pub allowed: bool,
    /// Daily limit for the call type.
    pub limit: u32,
    /// Calls remaining after this request (0 if denied).
    pub remaining: u32,
    /// Unix timestamp of next midnight UTC (reset time).
    pub reset_at: u64,
    /// Seconds until reset (for Retry-After header).
    pub retry_after_secs: u64,
}

// ── ProxyRateLimiter ──────────────────────────────────────────────────────────

/// Shared rate limiter. Wrap in `Arc<tokio::sync::Mutex<ProxyRateLimiter>>` for use across
/// Axum handlers.
pub struct ProxyRateLimiter {
    config: RateLimitConfig,
    counters: HashMap<String, UserCounters>,
}

impl ProxyRateLimiter {
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            config,
            counters: HashMap::new(),
        }
    }

    /// Check and, if allowed, record a call.
    ///
    /// For `Deep` calls this also checks and records against the `Llm` budget
    /// (a deep call consumes from both pools).
    pub fn check_and_record(
        &mut self,
        user_id: &str,
        role: UserRole,
        call_type: CallType,
    ) -> RateLimitResult {
        let now = now_utc_secs();
        let today_day = utc_day(now);
        let reset_at = next_midnight_utc(now);
        let retry_after_secs = reset_at.saturating_sub(now);

        // Admin is always unlimited.
        if role == UserRole::Admin {
            let entry = self.counters.entry(user_id.to_string()).or_default();
            entry.refresh_for_today(today_day);
            match call_type {
                CallType::Llm => entry.llm += 1,
                CallType::Tool => entry.tool += 1,
                CallType::Deep => {
                    entry.llm += 1;
                    entry.deep += 1;
                }
            }
            return RateLimitResult {
                allowed: true,
                limit: u32::MAX,
                remaining: u32::MAX,
                reset_at,
                retry_after_secs,
            };
        }

        let (llm_limit, tool_limit, deep_limit) = match role {
            UserRole::User => (
                self.config.user_llm_limit,
                self.config.user_tool_limit,
                self.config.user_deep_limit,
            ),
            UserRole::Guest => (
                self.config.guest_llm_limit,
                self.config.guest_tool_limit,
                self.config.guest_deep_limit,
            ),
            UserRole::Admin => unreachable!(),
        };

        let entry = self.counters.entry(user_id.to_string()).or_default();
        entry.refresh_for_today(today_day);

        match call_type {
            CallType::Llm => {
                let current = entry.llm;
                let limit = llm_limit;
                if current >= limit {
                    return RateLimitResult {
                        allowed: false,
                        limit,
                        remaining: 0,
                        reset_at,
                        retry_after_secs,
                    };
                }
                entry.llm += 1;
                let remaining = limit.saturating_sub(entry.llm);
                RateLimitResult { allowed: true, limit, remaining, reset_at, retry_after_secs }
            }
            CallType::Tool => {
                let current = entry.tool;
                let limit = tool_limit;
                if current >= limit {
                    return RateLimitResult {
                        allowed: false,
                        limit,
                        remaining: 0,
                        reset_at,
                        retry_after_secs,
                    };
                }
                entry.tool += 1;
                let remaining = limit.saturating_sub(entry.tool);
                RateLimitResult { allowed: true, limit, remaining, reset_at, retry_after_secs }
            }
            CallType::Deep => {
                // Deep calls count against BOTH the llm pool AND the deep pool.
                // If either pool is exhausted, deny.
                let llm_current = entry.llm;
                let deep_current = entry.deep;

                if llm_current >= llm_limit {
                    return RateLimitResult {
                        allowed: false,
                        limit: llm_limit,
                        remaining: 0,
                        reset_at,
                        retry_after_secs,
                    };
                }
                if deep_current >= deep_limit {
                    return RateLimitResult {
                        allowed: false,
                        limit: deep_limit,
                        remaining: 0,
                        reset_at,
                        retry_after_secs,
                    };
                }

                entry.llm += 1;
                entry.deep += 1;

                // Report remaining for the more constrained of the two pools.
                let llm_remaining = llm_limit.saturating_sub(entry.llm);
                let deep_remaining = deep_limit.saturating_sub(entry.deep);
                let remaining = llm_remaining.min(deep_remaining);
                RateLimitResult {
                    allowed: true,
                    limit: deep_limit,
                    remaining,
                    reset_at,
                    retry_after_secs,
                }
            }
        }
    }

    /// Sweep all counters: reset any that belong to a past day.
    /// Called by a background task or at midnight.
    pub fn reset_daily(&mut self) {
        let today_day = utc_day(now_utc_secs());
        for counters in self.counters.values_mut() {
            counters.refresh_for_today(today_day);
        }
    }

    /// Expose the raw limit for a (role, call_type) pair — used in tests.
    pub fn limit_for(&self, role: UserRole, call_type: CallType) -> u32 {
        match role {
            UserRole::Admin => u32::MAX,
            UserRole::User => match call_type {
                CallType::Llm => self.config.user_llm_limit,
                CallType::Tool => self.config.user_tool_limit,
                CallType::Deep => self.config.user_deep_limit,
            },
            UserRole::Guest => match call_type {
                CallType::Llm => self.config.guest_llm_limit,
                CallType::Tool => self.config.guest_tool_limit,
                CallType::Deep => self.config.guest_deep_limit,
            },
        }
    }

    /// Test-only: forcibly backdate a user's counters to a past day so the
    /// next request triggers a lazy reset.
    #[cfg(test)]
    pub fn backdate_for_test(&mut self, user_id: &str) {
        let yesterday_day = utc_day(now_utc_secs()).saturating_sub(1);
        if let Some(entry) = self.counters.get_mut(user_id) {
            entry.day = yesterday_day;
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RateLimitConfig;

    fn default_config() -> RateLimitConfig {
        RateLimitConfig {
            user_llm_limit: 200,
            user_tool_limit: 500,
            user_deep_limit: 50,
            guest_llm_limit: 20,
            guest_tool_limit: 50,
            guest_deep_limit: 5,
        }
    }

    fn limiter() -> ProxyRateLimiter {
        ProxyRateLimiter::new(default_config())
    }

    // ── Admin bypass ─────────────────────────────────────────────────────────

    #[test]
    fn test_admin_bypasses_all_limits() {
        let mut rl = limiter();
        // Call well beyond any user/guest limit — admin should always be allowed.
        for _ in 0..300 {
            let res = rl.check_and_record("admin-user", UserRole::Admin, CallType::Llm);
            assert!(res.allowed, "admin must never be blocked");
        }
        for _ in 0..600 {
            let res = rl.check_and_record("admin-user", UserRole::Admin, CallType::Tool);
            assert!(res.allowed);
        }
        for _ in 0..100 {
            let res = rl.check_and_record("admin-user", UserRole::Admin, CallType::Deep);
            assert!(res.allowed);
        }
    }

    // ── User LLM limit ───────────────────────────────────────────────────────

    #[test]
    fn test_user_blocked_at_201st_llm_call() {
        let mut rl = limiter();
        // Calls 1-200 must pass.
        for i in 1..=200 {
            let res = rl.check_and_record("user-a", UserRole::User, CallType::Llm);
            assert!(res.allowed, "call {i} should be allowed");
        }
        // 201st call must be denied.
        let res = rl.check_and_record("user-a", UserRole::User, CallType::Llm);
        assert!(!res.allowed, "201st LLM call must be blocked");
        assert_eq!(res.remaining, 0);
        assert_eq!(res.limit, 200);
    }

    // ── Guest LLM limit ──────────────────────────────────────────────────────

    #[test]
    fn test_guest_blocked_at_21st_llm_call() {
        let mut rl = limiter();
        for i in 1..=20 {
            let res = rl.check_and_record("guest-a", UserRole::Guest, CallType::Llm);
            assert!(res.allowed, "call {i} should be allowed");
        }
        let res = rl.check_and_record("guest-a", UserRole::Guest, CallType::Llm);
        assert!(!res.allowed, "21st LLM call must be blocked");
        assert_eq!(res.remaining, 0);
        assert_eq!(res.limit, 20);
    }

    // ── Deep model counted separately ────────────────────────────────────────

    #[test]
    fn test_deep_model_counted_separately() {
        let config = RateLimitConfig {
            user_llm_limit: 200,
            user_tool_limit: 500,
            user_deep_limit: 5, // low deep limit for fast test
            guest_llm_limit: 20,
            guest_tool_limit: 50,
            guest_deep_limit: 5,
        };
        let mut rl = ProxyRateLimiter::new(config);

        // Use 5 deep calls.
        for i in 1..=5 {
            let res = rl.check_and_record("user-b", UserRole::User, CallType::Deep);
            assert!(res.allowed, "deep call {i} should be allowed");
        }
        // 6th deep call must be blocked by the deep budget (even though llm budget has room).
        let res = rl.check_and_record("user-b", UserRole::User, CallType::Deep);
        assert!(!res.allowed, "6th deep call must be blocked");
        assert_eq!(res.limit, 5);

        // A regular LLM call still uses the shared llm pool; deep calls also consumed from it.
        // 5 deep calls consumed 5 LLM slots, so 195 remain.
        let res = rl.check_and_record("user-b", UserRole::User, CallType::Llm);
        assert!(res.allowed, "regular LLM call should still be allowed (195 remain)");
    }

    // ── Counter reset on date change ─────────────────────────────────────────

    #[test]
    fn test_counters_reset_when_date_changes() {
        let mut rl = limiter();

        // Exhaust guest LLM budget.
        for _ in 0..20 {
            rl.check_and_record("guest-b", UserRole::Guest, CallType::Llm);
        }
        let res = rl.check_and_record("guest-b", UserRole::Guest, CallType::Llm);
        assert!(!res.allowed, "should be blocked after exhaustion");

        // Simulate a date change by backdating the counter.
        rl.backdate_for_test("guest-b");

        // Now the next call should reset and succeed.
        let res = rl.check_and_record("guest-b", UserRole::Guest, CallType::Llm);
        assert!(res.allowed, "first call after date change should be allowed");
    }

    // ── 429 response includes Retry-After ────────────────────────────────────

    #[test]
    fn test_denied_result_includes_retry_after() {
        let mut rl = limiter();

        // Exhaust guest limit.
        for _ in 0..20 {
            rl.check_and_record("guest-c", UserRole::Guest, CallType::Llm);
        }
        let res = rl.check_and_record("guest-c", UserRole::Guest, CallType::Llm);
        assert!(!res.allowed);
        // retry_after_secs should be positive and at most 86400 (one full day).
        assert!(res.retry_after_secs > 0, "Retry-After must be > 0");
        assert!(res.retry_after_secs <= 86_400, "Retry-After must be <= 86400 seconds");
        // reset_at must be in the future.
        let now = now_utc_secs();
        assert!(res.reset_at > now, "reset_at must be in the future");
    }

    // ── Rate limit headers on normal responses ───────────────────────────────

    #[test]
    fn test_rate_limit_headers_present_on_allowed_response() {
        let mut rl = limiter();
        let res = rl.check_and_record("user-c", UserRole::User, CallType::Tool);
        assert!(res.allowed);
        // All header fields must be populated.
        assert_eq!(res.limit, 500); // tool limit for user
        assert_eq!(res.remaining, 499); // 500 - 1
        let now = now_utc_secs();
        assert!(res.reset_at > now);
    }

    // ── No hardcoded values: limits come from config ──────────────────────────

    #[test]
    fn test_limits_come_from_config_not_hardcoded() {
        let custom_config = RateLimitConfig {
            user_llm_limit: 42,
            user_tool_limit: 77,
            user_deep_limit: 7,
            guest_llm_limit: 3,
            guest_tool_limit: 10,
            guest_deep_limit: 1,
        };
        let mut rl = ProxyRateLimiter::new(custom_config);

        // User LLM: exactly 42 allowed.
        for i in 1..=42 {
            let res = rl.check_and_record("u", UserRole::User, CallType::Llm);
            assert!(res.allowed, "call {i} should be allowed");
        }
        let res = rl.check_and_record("u", UserRole::User, CallType::Llm);
        assert!(!res.allowed, "43rd call must be blocked");

        // Guest LLM: exactly 3 allowed.
        for i in 1..=3 {
            let res = rl.check_and_record("g", UserRole::Guest, CallType::Llm);
            assert!(res.allowed, "guest call {i} should be allowed");
        }
        let res = rl.check_and_record("g", UserRole::Guest, CallType::Llm);
        assert!(!res.allowed, "4th guest call must be blocked");
    }

    // ── Deep model detection ─────────────────────────────────────────────────

    #[test]
    fn test_deep_model_detection() {
        assert!(CallType::is_deep_model("deepseek-chat"));
        assert!(CallType::is_deep_model("claude-opus-4-5"));
        assert!(CallType::is_deep_model("mixtral-120b-instruct"));
        assert!(CallType::is_deep_model("DeepSeek-R1")); // case-insensitive
        assert!(!CallType::is_deep_model("claude-sonnet-4-5"));
        assert!(!CallType::is_deep_model("gpt-4o"));
        assert!(!CallType::is_deep_model("haiku-3-5"));
    }

    // ── Role parsing ─────────────────────────────────────────────────────────

    #[test]
    fn test_role_parsing() {
        assert_eq!(UserRole::from_claim(Some("admin")), UserRole::Admin);
        assert_eq!(UserRole::from_claim(Some("guest")), UserRole::Guest);
        assert_eq!(UserRole::from_claim(Some("user")), UserRole::User);
        assert_eq!(UserRole::from_claim(None), UserRole::User);
        assert_eq!(UserRole::from_claim(Some("unknown")), UserRole::User);
    }

    // ── reset_daily sweeps all users ─────────────────────────────────────────

    #[test]
    fn test_reset_daily_clears_counters() {
        let mut rl = limiter();

        // Exhaust two users.
        for _ in 0..20 {
            rl.check_and_record("g1", UserRole::Guest, CallType::Llm);
            rl.check_and_record("g2", UserRole::Guest, CallType::Llm);
        }

        // Backdate both entries.
        rl.backdate_for_test("g1");
        rl.backdate_for_test("g2");

        // reset_daily should pick up today's day and clear all.
        rl.reset_daily();

        let res1 = rl.check_and_record("g1", UserRole::Guest, CallType::Llm);
        let res2 = rl.check_and_record("g2", UserRole::Guest, CallType::Llm);
        assert!(res1.allowed, "g1 should be allowed after reset");
        assert!(res2.allowed, "g2 should be allowed after reset");
    }

    // ── Midnight boundary calculation ────────────────────────────────────────

    #[test]
    fn test_next_midnight_is_within_24_hours() {
        let now = now_utc_secs();
        let midnight = next_midnight_utc(now);
        assert!(midnight > now, "next midnight must be in the future");
        assert!(midnight <= now + 86_400, "next midnight must be within 24 hours");
        // It should be exactly at a day boundary.
        assert_eq!(midnight % 86_400, 0, "midnight must be at a day boundary");
    }
}
