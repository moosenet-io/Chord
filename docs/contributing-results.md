# Contributing Your Own Hardware's Results

The coder and assistant sweeps documented in
[model-testing-methodology.md](model-testing-methodology.md),
[test-results.md](test-results.md), and
[assistant-results.md](assistant-results.md) are all measured on one
specific machine: an AMD Strix Halo (`gfx1151`) unified-memory APU, tagged
`dynamic_gtt` (or, for older rows, untagged legacy `carveout`-era runs). If
you have different hardware — a discrete NVIDIA/AMD GPU, a different APU, a
CPU-only box, Apple Silicon — your numbers are not directly comparable to
ours, and **the harness's own `mem_config`/hardware-tagging discipline
exists specifically so contributed data is never silently blended with this
project's Strix Halo numbers.**

## What you need

1. **This repo (`moosenet/Chord`)** — the inference-serving/routing proxy.
   You need it running so the harness talks to a real serving path, not a
   direct-to-backend shortcut (both the coder and assistant harnesses
   deliberately route every inference call through Chord's unified path —
   see [model-testing-methodology.md](model-testing-methodology.md) — so a
   contributed result reflects the same code path production traffic uses).
2. **The [`moosenet/Terminus`](../../Terminus) harness** — `src/intake/`
   contains both harnesses:
   - `code_v2.rs` + `bin/intake_coder_sweep.rs` for coder-quality
   - `intake/assistant/` (`dim1`..`dim7`, `runner.rs`, `judges.rs`) for
     assistant-quality
3. **A Postgres endpoint** you control, with the same schema as
   `lumina_intake` (`code_profile_runs`, `assistant_dimension_score`,
   `assistant_profile_run`, `model_profiles`, `model_operational_profiles`,
   `model_dual_profile` — see `src/intake/storage.rs` and
   `src/intake/assistant/schema.rs` in Terminus for the exact DDL/columns).
   **Do not write into MooseNet's production `lumina_intake` instance** —
   stand up your own database. If a shared upload path is ever built (it
   does not exist today — see gaps below), it will be additive/append-only
   against a submission table, not direct write access to the live sweep.
4. Optionally, the 3-judge panel CLIs (`claude`, `gemini`, `codex`) if you
   want judged (not just deterministic) dimensions scored. The harness
   degrades gracefully — an unavailable/unauthenticated CLI abstains rather
   than failing your run (see `judges.rs`).

## Tagging your hardware distinctly

`mem_config` is a free-text column, not an enum constrained to `carveout`/
`dynamic_gtt` — those are just the two values this project's own sweep host
has used. **Tag your own hardware with its own distinct string** before you
run anything, e.g.:

- `contrib-4090-24gb` for a discrete-GPU contributor
- `contrib-m3max-128gb-unified` for Apple Silicon unified memory
- `contrib-mi300x-192gb` for a different AMD accelerator

The one hard rule, straight from the existing `mem_config` contract
(`storage.rs`): **never leave it `None`/untagged**, and never reuse
`carveout` or `dynamic_gtt` for anything that isn't literally this project's
Strix Halo sweep host under that config. An untagged or mis-tagged row is
worse than no row — it risks being silently averaged into gfx1151 numbers
it has nothing to do with. Set it explicitly on every row your harness run
writes (both `CodeRunRowV2.mem_config` and the assistant
`ScoreSink`/`insert_dimension_score_with_category_and_mem_config` path
accept it as a plain settable field).

## What a submission should look like

There is no automated submission pipeline today (see
[gaps](#current-gap-no-submission-pipeline) below). Until one exists, a
useful submission to open as a PR/issue against this repo includes:

1. **Hardware description** — exact GPU/APU/CPU, memory size and topology
   (unified vs. discrete), and the `mem_config` string you tagged your runs
   with.
2. **Harness version** — the Terminus commit/tag you ran (`harness_version`
   as recorded in `assistant_profile_run`, or the equivalent for a coder
   run).
3. **A Postgres dump or CSV export** of your tagged rows from
   `code_profile_runs` / `assistant_dimension_score` (whichever harness you
   ran) — enough for someone to re-import into a comparison database and
   verify the `mem_config` tag is consistently applied.
4. **Any skipped/non-viable models with reason** — the existing sweep
   records `OlympicCoder-32B`, `ornith-9b-fixed`, and `ornith-35b-fixed` as
   skipped-not-scored (timeout against the patience cap) rather than
   silently omitted or scored zero; do the same for anything that didn't
   run cleanly on your hardware.
5. If you're contributing assistant results, note **whether you had the
   3-judge panel available** — deterministic-only dimensions
   (`recall_ceiling_turns`, `fact_survival_rate`) are directly comparable
   across contributors even without judge access; judged dimensions
   (`coherence`, `personality_prompted`, `personality_latent`) are not
   comparable to panel-scored results if you only had 1 of 3 judges
   available. Record which judges actually responded (vs. abstained) —
   `judges.rs`'s abstain semantics mean a "3-judge" result can quietly be a
   1-judge result if two CLIs weren't authenticated.

## Current gap: no submission pipeline

Being direct about where this stands today: there is no CI job, upload
endpoint, or merge process that ingests external contributor data into a
shared comparison view. This section describes the tagging discipline and
data shape a submission needs so it *can* be compared safely once such a
pipeline exists (or manually, by a maintainer importing a dump) — it is not
claiming that pipeline exists yet. If you want to build it, the natural
starting point is a `contrib_` prefixed table (or a `source_host` column
alongside `mem_config`) in a shared Postgres instance, so contributed rows
are queryable but never joined into the gfx1151 leaderboard without an
explicit filter.
