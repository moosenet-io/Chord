//! Core tool-serving allowlist for Chord's external tool catalog.
//!
//! Chord's job is model routing/serving and the build/spec-execution pipeline
//! (gitea/github/plane dispatch, code review via DiffusionGemma, model
//! advisor/serving-profile introspection). It is **not** a general-purpose
//! secrets/personal-utility/ops proxy — that surface belongs to Lumina core
//! talking to Terminus directly, not to anything Chord serves externally.
//!
//! This allowlist governs what `/v1/tools/list`, `/v1/tools/discover`, and
//! `/v1/tools/call` will surface or execute, regardless of what is registered
//! in the upstream MCP backend or the Rust fallback registry (terminus-rs).
//! Tools not on this list are excluded from Chord's served catalog entirely
//! and rejected outright by `/v1/tools/call` — being off the `/v1/tools/list`
//! response is not sufficient on its own, since a caller who already knows a
//! tool name could otherwise still invoke it directly.
//!
//! Scope note: this narrows what Chord *serves* to external callers. It does
//! not change what Chord can *reach* internally (e.g. the upstream MCP
//! backend connection stays fully intact for the subset of calls that pass
//! the allowlist below).

/// Exact tool names always permitted, independent of prefix grouping.
const ALLOWED_EXACT: &[&str] = &[
    // Built-ins
    "health",
    "echo",
    "utc_now",
];

/// Prefixes covering an entire tool family. A tool name is allowed if it
/// starts with any of these.
///
/// - `gitea_` / `github_` — build-pipeline repo dispatch (create repo/PR,
///   push branch, mirror) used by the spec-execution pipeline.
/// - `plane_` — work-item tracking used by the same pipeline.
/// - `dgem_` — DiffusionGemma secondary code reviewer, part of the build
///   pipeline (SKILL v3.4).
/// - `model_advisor_` / `serving_profile` / `serving_residency` /
///   `model_intake` — Chord's own model-routing/serving domain.
const ALLOWED_PREFIXES: &[&str] = &[
    "gitea_",
    "github_",
    "plane_",
    "dgem_",
    "model_advisor_",
    "serving_profile",
    "serving_residency",
    "model_intake",
];

/// Returns true if `name` is in Chord's core served-tool allowlist.
///
/// Anything not covered here (personal-utility tools, secrets/infra-admin
/// tools, and general Lumina-fleet orchestration tools) is explicitly
/// excluded from what Chord serves, even if it is registered in the
/// underlying MCP backend or Rust fallback registry.
pub fn is_core_tool(name: &str) -> bool {
    if ALLOWED_EXACT.contains(&name) {
        return true;
    }
    ALLOWED_PREFIXES.iter().any(|p| name.starts_with(p))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allows_core_build_pipeline_tools() {
        for name in [
            "health",
            "echo",
            "utc_now",
            "gitea_create_repo",
            "gitea_create_pr",
            "github_push_branch",
            "plane_create_work_item",
            "dgem_review",
            "model_advisor_recommend",
            "serving_profile_get",
            "serving_residency_status",
        ] {
            assert!(is_core_tool(name), "{name} should be allowed");
        }
    }

    #[test]
    fn excludes_secret_access_tools() {
        for name in [
            "infisical_get_secret",
            "infisical_get_secrets_batch",
            "infisical_list_secrets",
            "infisical_list_projects",
            "infisical_status",
        ] {
            assert!(!is_core_tool(name), "{name} must be excluded");
        }
    }

    #[test]
    fn excludes_personal_utility_and_admin_tools() {
        for name in [
            "ledger_log",
            "vitals_summary",
            "crucible_status",
            "relay_lubelogger",
            "meridian_journal",
            "odyssey_plan",
            "hearth_add",
            "myelin_report",
            "dura_backup_status",
            "cortex_audit",
            "soma_status",
            "dev_run_command",
            "ansible_run_playbook",
            "reminder_set",
            "wizard_ask",
            "seer_research",
            "commute",
            "weather",
            "portainer_list_containers",
            "litellm_status",
            "prometheus_query",
            "jellyseerr_search",
            "google_calendar_list",
            "google_email_inbox",
            "openhands_run",
            "nexus_send",
            "news_headlines",
            "axon_dispatch",
            "vigil_summary",
            "sentinel_status",
            "routines_list",
            "skills_list",
            "council_convene",
            "network_ping",
        ] {
            assert!(!is_core_tool(name), "{name} must be excluded");
        }
    }

    #[test]
    fn does_not_substring_match_across_families() {
        // e.g. a tool that merely contains "gitea" should not match unless it
        // actually starts with the "gitea_" prefix.
        assert!(!is_core_tool("some_gitea_lookalike"));
        assert!(!is_core_tool("modelling_advisor_fake"));
    }
}
