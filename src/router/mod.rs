//! DOCGEN-03: the Chord SLM router capability.
//!
//! Given a generation request (from the `moosenet/Terminus` documentation
//! engine, or any future in-process caller), [`slm_router::SlmRouter`] decides
//! the inference destination (local high-context / local cheap / OpenRouter
//! frontier-free) per an explicit [`policy::RoutingPolicy`], and executes the
//! generation on the chosen destination — falling back gracefully, never
//! silently, when a destination is unavailable or egress-denied.
//!
//! See the module docs on [`slm_router`] for how this reuses the existing
//! `models::backends`/`models::routing` substrate rather than reimplementing
//! backend resolution or bearer-key handling.
//!
//! DOCGEN-04 ([`eval`]/[`eval_storage`]) is the evaluation sweep that scores
//! candidate routers on ROUTING quality (not raw model quality): decision
//! appropriateness, resulting doc quality, cost, and latency, persisted to
//! Postgres for cross-run comparison.

pub mod eval;
pub mod eval_storage;
pub mod policy;
pub mod slm_router;

pub use eval::{
    fixed_panel, sanity_check_grader, score_decision, summarize_candidate, CandidateEvalSummary,
    DecisionScore, DocGenEvalRequest, Grader, GraderSanityError, HeuristicCompletenessGrader,
};
pub use eval_storage::{normalize_model_id, RouterEvalDbError};
pub use policy::{RoutingDestination, RoutingPolicy, RoutingRequest};
pub use slm_router::{Executor, HttpExecutor, RoutingDecision, SlmRouter, SlmRouterError};
