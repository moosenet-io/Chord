#!/usr/bin/env python3
"""Render docs/assistant-results.md from live `lumina_intake` Postgres data.

Why a script instead of a hand-frozen table: the assistant sweep (S84
ASMT-01..08) is a mid-run, multi-day fleet sweep — new models and dimensions
land in Postgres between doc updates. A hand-copied snapshot goes stale on
the first new run; this script re-derives the table from the same tables
Chord/Terminus write to, so "re-run the script" is the only maintenance step.

Why Python, not Rust, even though Chord already depends on sqlx/postgres:
this is read-only reporting tooling that ships alongside the docs, not part
of the served proxy binary, and the repo has no existing Rust reporting-CLI
scaffold to extend. A Python script with psycopg2 needs no compile step and
is trivial for a docs-only contributor to run and read. If this grows beyond
a doc-generation script (e.g. becomes a dashboard or a CI gate), porting it
to a `chord-report` Rust binary using the existing sqlx dependency is the
natural next step — flagged here rather than done speculatively.

Usage:
    export CHORD_INTAKE_DB_URL="postgresql://user:pass@host:5432/lumina_intake"
    python3 docs/scripts/query_assistant_results.py > docs/assistant-results.md

Read-only: every query below is a SELECT. This script must never gain an
INSERT/UPDATE/DELETE/DDL statement — the sweep this reads from is live and
shared.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

try:
    import psycopg2
except ImportError:
    print(
        "psycopg2 not installed. `pip install psycopg2-binary` (or "
        "--break-system-packages / a venv) and retry.",
        file=sys.stderr,
    )
    raise

DB_URL_ENV = "CHORD_INTAKE_DB_URL"

DIMENSIONS = [
    "conversation_depth",
    "tool_chaining",
    "memory_integration",
    "personality_latent",
    "personality_prompted",
    "embeddings",
    "yarn_context_depth",
]

PROMPTED_TRAITS = [
    "warm",
    "quirky",
    "curious",
    "direct",
    "held_one_question",
    "no_unasked_prefetch",
    "no_overclaim",
    "voice_under_provocation",
]


def connect():
    url = os.environ.get(DB_URL_ENV)
    if not url:
        print(f"Set {DB_URL_ENV} to a read-only Postgres URL first.", file=sys.stderr)
        sys.exit(1)
    return psycopg2.connect(url)


def fetch_all(cur, sql, params=None):
    cur.execute(sql, params or ())
    cols = [c.name for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def main() -> None:
    conn = connect()
    conn.set_session(readonly=True)
    cur = conn.cursor()

    dims_per_model = fetch_all(
        cur,
        """
        select model_id,
               count(distinct dimension) as dims_covered,
               count(distinct run_id) as runs,
               bool_or(mem_config is not null) as any_mem_config_tagged
        from assistant_dimension_score
        group by 1
        order by 2 desc, 1
        """,
    )

    proximity = {
        r["model_id"]: r
        for r in fetch_all(
            cur,
            """
            select model_id, avg(value) as avg_proximity, count(*) as n
            from assistant_dimension_score
            where dimension = 'personality_latent' and metric = 'proximity_to_lumina'
            group by 1
            """,
        )
    }

    recall_ceiling = {
        r["model_id"]: r
        for r in fetch_all(
            cur,
            """
            select model_id, avg(value) as avg_recall_ceiling_turns
            from assistant_dimension_score
            where dimension = 'conversation_depth' and metric = 'recall_ceiling_turns'
            group by 1
            """,
        )
    }

    coherence = {
        r["model_id"]: r
        for r in fetch_all(
            cur,
            """
            select model_id, avg(value) as avg_coherence
            from assistant_dimension_score
            where dimension = 'conversation_depth' and metric = 'coherence'
            group by 1
            """,
        )
    }

    mem_survival = {
        r["model_id"]: r
        for r in fetch_all(
            cur,
            """
            select model_id, avg(value) as avg_fact_survival_rate
            from assistant_dimension_score
            where dimension = 'memory_integration' and metric = 'fact_survival_rate'
            group by 1
            """,
        )
    }

    chain_acc = {
        r["model_id"]: r
        for r in fetch_all(
            cur,
            """
            select model_id, avg(value) as avg_mean_chain_accuracy
            from assistant_dimension_score
            where dimension = 'tool_chaining' and metric = 'mean_chain_accuracy'
            group by 1
            """,
        )
    }

    prompted_avg = {
        r["model_id"]: r
        for r in fetch_all(
            cur,
            """
            select model_id, avg(value) as avg_prompted_all_traits
            from assistant_dimension_score
            where dimension = 'personality_prompted' and metric = any(%s)
            group by 1
            """,
            (PROMPTED_TRAITS,),
        )
    }

    yarn_count = fetch_all(
        cur,
        "select count(*) as n from assistant_dimension_score where dimension = 'yarn_context_depth'",
    )[0]["n"]

    mem_config_counts = fetch_all(
        cur,
        "select mem_config, count(*) as n from assistant_dimension_score group by 1 order by 1",
    )

    runs = fetch_all(
        cur,
        "select id, started_at, harness_version from assistant_profile_run order by started_at",
    )

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    print(f"<!-- GENERATED by docs/scripts/query_assistant_results.py at {now}. -->")
    print("<!-- Do not hand-edit the tables below; re-run the script instead. -->")
    print()
    print("# Assistant-Fleet Results — S84 ASMT Sweep (gfx1151)")
    print()
    print(
        "Live-generated from `lumina_intake` Postgres "
        "(`assistant_dimension_score`, `assistant_profile_run`). "
        f"Regenerated {now}. This is a **partial, in-progress sweep** — "
        "coverage varies per model; see the per-model dimension count "
        "column below before trusting any single number."
    )
    print()
    print(f"Harness runs recorded (`assistant_profile_run`): {len(runs)}")
    for r in runs:
        print(f"- `{r['id']}` — {r['started_at']} — harness `{r['harness_version']}`")
    print()
    print(
        f"**`yarn_context_depth` rows: {yarn_count}.** "
        "The YaRN-assistant dimension (dim7) is built and unit-tested in "
        "Terminus but has not been run against any model yet — no "
        "context-collapse data exists for any model below."
    )
    print()
    print("**`mem_config` tagging on this table:**")
    for r in mem_config_counts:
        tag = r["mem_config"] if r["mem_config"] is not None else "NULL (untagged)"
        print(f"- `{tag}`: {r['n']} rows")
    print(
        "\nEvery row in `assistant_dimension_score` is currently untagged. "
        "**Do not read the numbers below as `dynamic_gtt`- or "
        "`carveout`-specific** — we do not know which config produced them. "
        "This is a tracked gap (see [model-testing-methodology.md]"
        "(model-testing-methodology.md#known-gaps--honest-limitations))."
    )
    print()
    print("## Per-model summary")
    print()
    print(
        "| Model | Dims covered | Runs | Any mem_config tagged | Proximity-to-Lumina (1-5) "
        "| Recall ceiling (turns) | Coherence (1-5) | Fact survival rate | "
        "Mean tool-chain accuracy | Prompted-adherence avg (1-5) |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")

    def fmt(v, nd=2):
        if v is None:
            return "N/A"
        return f"{v:.{nd}f}"

    for row in dims_per_model:
        m = row["model_id"]
        dims = row["dims_covered"]
        n_runs = row["runs"]
        tagged = "yes" if row["any_mem_config_tagged"] else "no"
        prox = proximity.get(m, {}).get("avg_proximity")
        prox_n = proximity.get(m, {}).get("n")
        rc = recall_ceiling.get(m, {}).get("avg_recall_ceiling_turns")
        coh = coherence.get(m, {}).get("avg_coherence")
        fs = mem_survival.get(m, {}).get("avg_fact_survival_rate")
        ca = chain_acc.get(m, {}).get("avg_mean_chain_accuracy")
        pa = prompted_avg.get(m, {}).get("avg_prompted_all_traits")
        prox_str = f"{fmt(prox)} (n={prox_n})" if prox is not None else "N/A"
        print(
            f"| `{m}` | {dims}/7 | {n_runs} | {tagged} | {prox_str} | "
            f"{fmt(rc, 1)} | {fmt(coh)} | {fmt(fs)} | {fmt(ca)} | {fmt(pa)} |"
        )

    print()
    print(
        "`Dims covered` counts distinct `dimension` labels with at least one "
        "row for that model, out of the 7 possible (6 standard + "
        "`yarn_context_depth`). A model at 4-6/7 has partial coverage — "
        "treat its numbers as directional, not final. `Proximity-to-Lumina` "
        "is the `personality_latent.proximity_to_lumina` metric (raw model, "
        "no Lumina prompt) — a composite closeness-to-target-persona score, "
        "not a pass/fail. `Prompted-adherence avg` is the mean across all "
        "8 `personality_prompted` traits (voice + behavioral), measured "
        "WITH the real 5-layer production prompt loaded — this is the "
        "production-relevant number, `personality_latent` is the "
        "pre-prompt baseline."
    )
    print()
    print(
        "For the full per-dimension, per-metric breakdown (every trait, "
        "every embedding metric, every tool-chaining scenario) query "
        "`assistant_dimension_score` directly — this summary intentionally "
        "collapses to one headline number per dimension for readability. "
        "See [model-testing-methodology.md](model-testing-methodology.md) "
        "for what each dimension/metric means."
    )

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
