//! ROUT-06: permanent regression fixture for the hybrid router.
//!
//! This session's live evaluation of `SupraLabs/Supra-Router-51M` scored 47
//! prompts against the daemon's real raw output. That raw transcript was not
//! found recorded anywhere retrievable by this build (no Plane attachment,
//! no committed file, no reachable session log) — per the S92 build task's
//! explicit fallback, this fixture RECONSTRUCTS an equivalent 47-prompt set
//! spanning the same categories the eval report described (simple, complex,
//! ambiguous, ambiguous, and the two confirmed edge cases: trivial
//! single-operation arithmetic and trivial one-line code) rather than
//! re-inventing the wheel from nothing. Each entry pairs a prompt with a
//! *simulated* raw daemon response (in the label-tolerant formats ROUT-02
//! documented, including the omitted-label and garbled-Domain cases) and the
//! expected final route after ROUT-02/03 processing.
//!
//! This exercises the full pipeline end-to-end (`parse_classification` +
//! `route_for`) as one committed, permanent regression test — future
//! model swaps or threshold tuning must not silently change these 47
//! outcomes without a deliberate, reviewed edit to this file.

use chord_proxy::agentic::router_classifier::{parse_classification, route_for, Route};

/// One fixture case: (name, prompt, simulated raw daemon output, expected route).
struct Case {
    name: &'static str,
    prompt: &'static str,
    raw: &'static str,
    expected: Route,
}

fn cases() -> Vec<Case> {
    vec![
        // ── Simple / small-model-appropriate (12) ───────────────────────────
        Case { name: "simple_01_greeting", prompt: "hello there", raw: "Domain: chat | Complexity: 1 | Math: False | Code: False | Route: small", expected: Route::Small },
        Case { name: "simple_02_time", prompt: "what time is it", raw: "Domain: chat | Complexity: 1 | Math: False | Code: False", expected: Route::Small },
        Case { name: "simple_03_weather", prompt: "what's the weather today", raw: "Domain: general | Complexity: 1 | Math: False | Code: False", expected: Route::Small },
        Case { name: "simple_04_lookup", prompt: "what is the capital of France", raw: "Domain: trivia | Complexity: 1 | Math: False | Code: False", expected: Route::Small },
        Case { name: "simple_05_status", prompt: "is the server up", raw: "Domain: ops | Complexity: 1 | Math: False | Code: False", expected: Route::Small },
        Case { name: "simple_06_reminder", prompt: "remind me to call mom at 5pm", raw: "Domain: scheduling | Complexity: 2 | Math: False | Code: False", expected: Route::Small },
        Case { name: "simple_07_definition", prompt: "define recursion", raw: "Domain: general | Complexity: 1 | Math: False | Code: False", expected: Route::Small },
        Case { name: "simple_08_translate", prompt: "translate 'good morning' to spanish", raw: "Domain: language | Complexity: 2 | Math: False | Code: False", expected: Route::Small },
        Case { name: "simple_09_convert_units", prompt: "convert 10 miles to km", raw: "Domain: math | Complexity: 2 | Math: True | Code: False", expected: Route::Big },
        Case { name: "simple_10_joke", prompt: "tell me a joke", raw: "Domain: chat | Complexity: 1 | Math: False | Code: False", expected: Route::Small },
        Case { name: "simple_11_spell", prompt: "how do you spell necessary", raw: "Domain: language | Complexity: 1 | Math: False | Code: False", expected: Route::Small },
        Case { name: "simple_12_yesno", prompt: "is 17 a prime number", raw: "Domain: math | Complexity: 1 | Math: True | Code: False", expected: Route::Big },

        // ── Complex / big-model-appropriate (12) ────────────────────────────
        Case { name: "complex_01_analyze", prompt: "analyze the quarterly sales trends and flag anomalies", raw: "Domain: analytics | Complexity: 4 | Math: True | Code: False", expected: Route::Big },
        Case { name: "complex_02_compare", prompt: "compare these three cloud providers on cost and latency", raw: "Domain: general | Complexity: 4 | Math: False | Code: False", expected: Route::Big },
        Case { name: "complex_03_synthesize", prompt: "synthesize the findings from these five reports into one summary", raw: "Domain: writing | Complexity: 4 | Math: False | Code: False", expected: Route::Big },
        Case { name: "complex_04_debug", prompt: "debug this race condition in a multi-threaded producer-consumer queue", raw: "Domain: coding | Complexity: 5 | Math: False | Code: True", expected: Route::Big },
        Case { name: "complex_05_design", prompt: "design a distributed rate limiter that survives node failures", raw: "Domain: coding | Complexity: 5 | Math: False | Code: True", expected: Route::Big },
        Case { name: "complex_06_proof", prompt: "prove that this recursive algorithm terminates for all inputs", raw: "Domain: math | Complexity: 5 | Math: True | Code: False", expected: Route::Big },
        Case { name: "complex_07_architecture", prompt: "evaluate the tradeoffs between microservices and a monolith for this workload", raw: "Domain: architecture | Complexity: 4 | Math: False | Code: False", expected: Route::Big },
        Case { name: "complex_08_optimize", prompt: "optimize this SQL query joining six tables with subqueries", raw: "Domain: coding | Complexity: 4 | Math: False | Code: True", expected: Route::Big },
        Case { name: "complex_09_explain_why", prompt: "explain why this neural network is not converging", raw: "Domain: ml | Complexity: 4 | Math: True | Code: False", expected: Route::Big },
        Case { name: "complex_10_reason_about", prompt: "reason about the second-order consequences of this pricing change", raw: "Domain: strategy | Complexity: 4 | Math: False | Code: False", expected: Route::Big },
        Case { name: "complex_11_multistep_code", prompt: "implement a thread-safe LRU cache with TTL eviction and metrics hooks", raw: "Domain: coding | Complexity: 4 | Math: False | Code: True", expected: Route::Big },
        Case { name: "complex_12_proof_math", prompt: "derive the closed-form solution for this second-order differential equation", raw: "Domain: math | Complexity: 5 | Math: True | Code: False", expected: Route::Big },

        // ── Confirmed blind-spot: trivial arithmetic (ROUT-03) (6) ──────────
        Case { name: "trivial_math_01", prompt: "12*8", raw: "Domain: math | Complexity: 1 | Math: True | Code: False", expected: Route::Small },
        Case { name: "trivial_math_02", prompt: "45 + 7", raw: "Domain: math | Complexity: 1 | Math: True | Code: False", expected: Route::Small },
        Case { name: "trivial_math_03", prompt: "100 / 4", raw: "Domain: math | Complexity: 2 | Math: True | Code: False", expected: Route::Small },
        Case { name: "trivial_math_04", prompt: "-9 - 3", raw: "Domain: math | Complexity: 1 | Math: True | Code: False", expected: Route::Small },
        Case { name: "trivial_math_05", prompt: "6.5 * 2", raw: "Domain: math | Complexity: 1 | Math: True | Code: False", expected: Route::Small },
        Case { name: "trivial_math_06_still_escalates", prompt: "compute 12*8 as part of this multi-step compound-interest analysis", raw: "Domain: math | Complexity: 4 | Math: True | Code: False", expected: Route::Big },

        // ── Confirmed blind-spot: trivial one-line code (ROUT-03) (6) ───────
        Case { name: "trivial_code_01", prompt: "def f(x): return x+1", raw: "Domain: coding | Complexity: 1 | Math: False | Code: True", expected: Route::Small },
        Case { name: "trivial_code_02", prompt: "lambda x: x * 2", raw: "Domain: coding | Complexity: 1 | Math: False | Code: True", expected: Route::Small },
        Case { name: "trivial_code_03", prompt: "def square(n): return n * n", raw: "Domain: coding | Complexity: 2 | Math: False | Code: True", expected: Route::Small },
        Case { name: "trivial_code_04", prompt: "function double(x) => x * 2", raw: "Domain: coding | Complexity: 1 | Math: False | Code: True", expected: Route::Small },
        Case { name: "trivial_code_05_control_flow_not_trivial", prompt: "def f(x): return x+1 if x > 0 else -x", raw: "Domain: coding | Complexity: 2 | Math: False | Code: True", expected: Route::Big },
        Case { name: "trivial_code_06_multiline_not_trivial", prompt: "def f(x):\n    return x + 1", raw: "Domain: coding | Complexity: 2 | Math: False | Code: True", expected: Route::Big },

        // ── Ambiguous (6) ────────────────────────────────────────────────────
        Case { name: "ambiguous_01_short_but_deep", prompt: "why", raw: "Domain: chat | Complexity: 2 | Math: False | Code: False", expected: Route::Small },
        Case { name: "ambiguous_02_borderline_complexity", prompt: "summarize this email thread", raw: "Domain: writing | Complexity: 3 | Math: False | Code: False", expected: Route::Big },
        Case { name: "ambiguous_03_math_word_no_computation", prompt: "what does the word 'algebra' mean", raw: "Domain: language | Complexity: 1 | Math: False | Code: False", expected: Route::Small },
        Case { name: "ambiguous_04_code_word_no_code", prompt: "what programming language should I learn first", raw: "Domain: advice | Complexity: 2 | Math: False | Code: False", expected: Route::Small },
        // Natural-language request *about* code, not an actual short code
        // snippet — ROUT-03's trivial-code detector intentionally only
        // matches literal minimal snippets (`def f(x): ...`, `lambda ...`),
        // not descriptions of a coding task, so this correctly still escalates.
        Case { name: "ambiguous_05_short_code_request", prompt: "write a one-liner to reverse a string in python", raw: "Domain: coding | Complexity: 1 | Math: False | Code: True", expected: Route::Big },
        Case { name: "ambiguous_06_deep_prefix", prompt: "/deep what should I have for lunch", raw: "Domain: chat | Complexity: 1 | Math: False | Code: False", expected: Route::Small },

        // ── Garbled Domain, still parseable (~11% of real eval) (3) ─────────
        Case { name: "garbled_domain_01", prompt: "analyze this dataset for outliers", raw: "g3n3ral/an4lytics!! | Complexity: 4 | Math: True | Code: False", expected: Route::Big },
        Case { name: "garbled_domain_02", prompt: "hello", raw: "###\u{fffd}corrupt\u{fffd}### Complexity: 1 | Math: False | Code: False", expected: Route::Small },
        Case { name: "garbled_domain_03", prompt: "write a sort function", raw: "??? | Complexity: 2 | Math: False | Code: True", expected: Route::Big },

        // ── Label-omitted output, still parseable (2) ───────────────────────
        Case { name: "label_omitted_01", prompt: "compare two databases", raw: "general/analysis 4/5 math:no code:no", expected: Route::Big },
        Case { name: "label_omitted_02", prompt: "12*8", raw: "1/5 math:yes code:no", expected: Route::Small },
    ]
}

#[test]
fn rout06_permanent_regression_fixture_47_cases() {
    let all_cases = cases();
    assert_eq!(
        all_cases.len(),
        47,
        "this fixture must stay at exactly 47 cases — the eval baseline size"
    );

    let mut failures = Vec::new();
    for case in &all_cases {
        let classification = match parse_classification(case.raw) {
            Ok(c) => c,
            Err(e) => {
                failures.push(format!("{}: parse failed unexpectedly: {:?}", case.name, e));
                continue;
            }
        };
        let route = route_for(case.prompt, &classification);
        if route != case.expected {
            failures.push(format!(
                "{}: expected {:?}, got {:?} (prompt={:?}, raw={:?})",
                case.name, case.expected, route, case.prompt, case.raw
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "ROUT-06 regression fixture failures:\n{}",
        failures.join("\n")
    );
}

#[test]
fn rout06_fixture_covers_all_required_categories() {
    let names: Vec<&str> = cases().iter().map(|c| c.name).collect();
    assert!(names.iter().any(|n| n.starts_with("simple_")));
    assert!(names.iter().any(|n| n.starts_with("complex_")));
    assert!(names.iter().any(|n| n.starts_with("trivial_math_")));
    assert!(names.iter().any(|n| n.starts_with("trivial_code_")));
    assert!(names.iter().any(|n| n.starts_with("ambiguous_")));
    assert!(names.iter().any(|n| n.starts_with("garbled_domain_")));
    assert!(names.iter().any(|n| n.starts_with("label_omitted_")));
}
