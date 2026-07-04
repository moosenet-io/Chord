# Model Testing Methodology — gfx1151 (AMD Strix Halo) Fleet Sweeps

This page documents how MooseNet benchmarks local LLMs for two distinct
production roles — **coder** (agentic build/modify tasks routed through
Chord's Harmony plane) and **assistant** (conversational/tool-using
personality-adherent workloads routed through Chord's Lumina plane) — on
AMD's Strix Halo (`gfx1151`) unified-memory APU.

**The harness code does not live in this repository.** Chord is the
inference-serving/routing proxy the harness calls through; the actual test
harness — corpora, scoring, judge orchestration, and the Postgres writer —
lives in a separate repository, **[`moosenet/Terminus`](../../Terminus)**,
under `src/intake/`. Every mechanic described below is cited to the Terminus
source file and doc comment it comes from, not inferred or guessed. If you
are trying to run or extend the harness itself, go read Terminus; this page
is the results/methodology reference, not the harness README.

## Why benchmark on this hardware at all

Strix Halo is a unified-memory APU: GPU and CPU share one physical memory
pool, carved up (or dynamically pooled — see [mem_config](#mem_config-tagging-why-results-are-never-blended-across-hardware-configs)
below) between the two. That changes the tradeoffs that drive model choice
in ways that don't transfer from discrete-GPU benchmarks: throughput is far
more sensitive to how memory is split, MoE models interact badly with one of
the two serving backends (see [Known backend quirks](#known-gfx1151-backend-quirks)),
and the "does this model even run without OOM/hanging" question is itself
sweep data, not an assumption. Chord's whole job — routing, admission,
launch-flag selection — is driven by per-model serving profiles measured by
this harness (see the [Chord README](../README.md)/[architecture.md](architecture.md)).
Getting the benchmarking honest is a prerequisite for Chord's routing being
honest.

## Two harnesses, two roles

### Coder-quality harness (`src/intake/code_v2.rs`, driven by `src/bin/intake_coder_sweep.rs`)

The v2 harness (Terminus doc comment, `code_v2.rs`) deliberately does **not**
test cold one-shot generation from a task description — Terminus's own v1
suite (`code.rs`) did that and "every model scored 0-1," because it's not
what the real build pipeline asks a model to do. v2 instead reproduces the
actual pipeline scenario a coder model faces in production:

> "Here is a spec item (`## Task` / `## FILES` / `## APPROACH` / `## TEST
> PLAN`) and the CURRENT FULL CONTENTS of the real file(s) it must modify,
> plus project context. Output the COMPLETE modified file(s)."

Cases are drawn from a corpus of dependency-minimal, standalone real-derived
crates (`_workspaces/<ws>/`) spanning **Bash, Python, Rust**, and (corpus
support exists for, though not currently exercised — see
[gaps](#known-gaps--honest-limitations)) TypeScript. Each model's output is
applied to a fresh `/tmp` copy of the case workspace (never the whole repo)
and run through a stage-marked validator script that prints:

```
STAGE:COMPILE ok|fail
STAGE:TESTS   ok|fail
STAGE:CHANGE  ok|fail
```

#### Graduated 0–5 scoring (`code_v2.rs::graduated_score`)

| Score | Meaning |
|---|---|
| 5 | Compiles, tests pass, change is correct, **and** an LLM idiom-quality judge rates it ≥4 |
| 4 | Compiles, tests pass, change is correct (idiom judge <4 or unavailable) |
| 3 | Compiles, existing/model tests pass, but the targeted change is incomplete (`STAGE:CHANGE fail`) |
| 2 | Compiles but tests fail or the fix is only partially correct |
| 1 | Doesn't compile, but a recognizable attempt was produced |
| 0 | No usable code extracted / refusal / nothing to grade |

This is a genuinely graduated scale, not a pass/fail collapse: score 3 vs 2
vs 1 all represent distinct, useful failure modes for triaging a candidate
model, and only 0 means "nothing to grade at all."

#### First-pass vs. retry (`code_v2.rs::should_retry`)

A first-pass score of 1 or 2 (compiles-but-wrong, or a recognizable-but-
non-compiling attempt) triggers **one retry pass** with the validator's
feedback fed back to the model. The recorded `retry_score` is only ever
allowed to raise the effective score (`retry.max(first_pass)`), never lower
it — a model that produces a near-miss and then over-corrects into
something worse is still credited for its first-pass showing.

#### BLITZ vs. MULTI-FILE

Cases are also split into **BLITZ** (single-file edits) and **MULTI-FILE**
(2+ files touched in one case) buckets. As documented in
[`test-results.md`](test-results.md), multi-file editing is the harness's
best discriminator between genuinely strong coder models and models that
only look good on easy single-file edits.

### Assistant-quality harness (`src/intake/assistant/`)

Six core dimensions run for every nominated model
(`runner::SUITE_DIMENSIONS`), plus a seventh that only runs for models
flagged `yarn_capable`:

| # | Module | Dimension label | What it measures |
|---|---|---|---|
| 1 | `dim1_conversation.rs` | `conversation_depth` | How many conversation turns a model sustains before degrading, on two axes: a deterministic planted-fact **recall ceiling** (`recall_ceiling_turns` — the deepest turn where recall still meets the corpus threshold), and judged **coherence/on-voice** quality (1–5 panel, mean + SD) sampled at several depths. |
| 2 | `dim2_toolchain.rs` | `tool_chaining` | Multi-step tool chaining where the user's intent is **implicit across turns** — the model must infer a 3–5 step chain with data handoff between steps from a conversation that never names the tools. (Explicit-task tool chains are the pre-existing S83 agent-scenario corpus, referenced by id, not re-authored here.) |
| 3 | `dim3_memory.rs` | `memory_integration` | Whether planted facts survive the **real** production 3-tier memory pipeline (`compat::conversation::buffer::ConversationBuffer`) across sessions: plant in session 1, force a summarization cycle, run an unrelated session 2, probe recall in session 3. Reports `fact_survival_rate` split by whether the fact was compressed into a summary or still lived in the verbatim buffer — isolating recall-from-summary from raw recall. |
| 4 | `dim4_ocean.rs` | `personality_latent` | Big Five (OCEAN) traits on the **raw model with no Lumina prompt loaded** — what training baked in, so weak-fit base models can be filtered before layering the persona on top. A `is_base_only` guard asserts the prompt carries no Lumina marker. |
| 5 | `dim5_prompted.rs` | `personality_prompted` | The production-config measurement, with the **real 5-layer Lumina system prompt** assembled by the canonical `PromptAssembler` (`[identity] [rules] [capabilities] [style] [now]`) loaded. Scores voice adherence (`warm`, `quirky`, `curious`, `direct`) and behavioral adherence under pressure (`held_one_question`, `no_unasked_prefetch`, `no_overclaim`, `voice_under_provocation`), all panel-scored 1–5. |
| 6 | `dim6_embeddings.rs` | `embeddings` | A separate sub-harness for embedding models (not chat models): ranks docs by cosine similarity on a public IR benchmark subset (MTEB/BEIR-style) **and** a small hand-labeled set drawn from real Engram memory data, then reports the **public-vs-Engram delta** as a domain-fit signal — a model strong on public IR but weak on Engram is flagged as a domain mismatch, not marked as a bad model. |
| 7 | `dim7_yarn_depth.rs` | `yarn_context_depth` | YaRN context-extension quality degradation (see below). Not in the standard suite — driven explicitly per `yarn_capable`-flagged model. |

Every dimension runs inference through Chord's **unified path**
(`intake::context::generate` / `chat_with_tools` → `intake::infer::infer_with_metrics`),
never a direct Ollama call — the harness is a client of the same proxy
production traffic uses, so a model's measured behavior is the behavior
Chord will actually serve.

**Degradation is data, not a crash.** Every dimension module states the same
contract in its doc comments: a timeout, truncation, transport error, empty
response, or refusal at a given turn/depth is recorded as *degradation at
that point* and the run stops there — nothing panics or aborts the sweep.

#### 3-judge panel + retry/abstain contract (`judges.rs`)

Subjective scoring (coherence, personality, idiom quality) goes through a
panel of three provider CLIs — Claude, Gemini, Codex — each shelled out to
independently (mirroring how the coder validator shells out to `bash`).
Every judge prompt ends in a fixed JSON-only contract string
(`JSON_CONTRACT_SUFFIX`): "respond with ONLY a JSON object mapping each
trait to an integer 1–5." A judge that violates the contract is retried
**exactly once** with a terse reminder; a second violation makes that judge
**abstain** for the item (its raw output is retained, redacted, for audit —
`RAW_AUDIT_MAX = 2000` bytes). The panel still produces a result from the
remaining judges rather than discarding the item. If a judge CLI is not
installed or not authenticated, it abstains for the whole run with an
operator warning rather than failing the sweep.

#### YaRN collapse detection (`dim7_yarn_depth.rs`)

For a model served with YaRN rope-scaling
(`--rope-scaling yarn --rope-scale N --yarn-orig-ctx <native> -c <extended>`),
this dimension probes **assistant quality, not code quality**, at rising
actual context-token depths: native baseline, then 30%, 60%, 100% of the
YaRN-extended target context (`DepthRung::ORDER`, shallowest first). It
reuses dim1's exact recall-probe and coherence-judge primitives, scored at
token depth instead of turn depth.

The discipline that matters for public credibility: **the ladder stops at
the first collapse** (a recall miss, or — when judges are configured — a
weak/unconfirmed coherence score) instead of grinding on to the advertised
maximum context. `stopped_early = true` and the collapse rung are recorded
explicitly. This means a model's *reported* context ceiling in this harness
is the actual token depth where quality broke, not the vendor's advertised
number — a model advertised as "128K context" that only holds coherent
recall to 30% of its YaRN-extended target will show that as its ceiling,
not 128K.

As of this writing **zero rows exist in Postgres for `yarn_context_depth`**
across any model (see [Results](#results) — the dimension is built and
unit-tested in Terminus but has not yet been run against the live fleet).

## `mem_config` tagging: why results are never blended across hardware configs

Every result row — coder (`code_profile_runs.mem_config`) and assistant
(`assistant_dimension_score.mem_config`) — carries an optional `mem_config`
column. This exists because the **same physical hardware** can be running
under two genuinely different memory configurations that are not
comparable:

- **`carveout`** — the older, static VRAM/GTT split.
- **`dynamic_gtt`** — a dynamically-sized GTT pool (currently 120GB on the
  production sweep host).

Terminus's own test suite (`storage.rs`) asserts the field defaults to
`None` and is explicitly settable — the comment on the struct field is
blunt about the intent: *"before this column existed (the preserved
baseline) or by callers that don't yet track it — NEVER assume `None` means
a specific config."* In other words, `None` is "unknown/untagged," not "the
default config" — a row with no tag must never be silently treated as
equivalent to a `carveout` or `dynamic_gtt` row. Cross-config comparisons
(e.g. "is this model faster under carveout or dynamic_gtt?") are only valid
when both sides are actually tagged; an untagged row answers neither
question.

**Current tagging coverage is partial** (see [gaps](#known-gaps--honest-limitations)):
the coder harness has both tagged (`dynamic_gtt`, 1,400 rows) and untagged
legacy rows (837 rows) in production Postgres today; the assistant harness
has **zero** tagged rows — every one of its 2,362 rows is untagged. This is
recorded as a real, current gap, not glossed over.

## Known gfx1151 backend quirks

Two serving backends are used in production:

- **Ollama (ROCm)** — the always-on GPU backend (`ollama`, port `11434`)
  used for GPU passes.
- **Ollama (CPU-only instance)** — `ollama-cpu`, a separate CPU-only
  instance used for CPU baseline passes.

A third path, **Vulkan via `llama-server`** (the `llama-gpu` backend tag),
exists in the stack but is **deliberately not used for MoE models** because
of a known wedge/hang bug. From `intake_coder_sweep.rs`:

> "serve GPU passes via `ollama-rocm` (the always-on `ollama` backend,
> `:11434`), NOT `llama-server` (`llama-gpu`), which wedges on MoE models on
> this Vulkan stack (S84: MiniMax/Ornith; S86: ornith-1.0). `ollama-rocm`
> serves dense AND MoE cleanly (proven on qwen3-coder)."

`acquire.rs` encodes this as a per-model `Gfx1151Class`:

- `Confirmed` — dense, Vulkan-validated: Ollama GPU first, then CPU.
- `Experimental` — MoE-on-Vulkan likely to hang: try ROCm with
  `HSA_OVERRIDE_GFX_VERSION` set; if it still hangs on both, **skip with a
  recorded reason** rather than force it.
- `Unknown` — needs a bounded smoke test before committing to the full
  suite.

**Do not read this as "Vulkan is broadly unsafe."** It is specifically the
MoE-on-Vulkan combination that hangs on this stack; dense models are
Vulkan-validated (`Confirmed`) and run there. If you are evaluating a new
MoE candidate on gfx1151, route it through ROCm (with `HSA_OVERRIDE_GFX_VERSION`
if needed) first, and treat a bounded smoke test as mandatory before
trusting a `Confirmed`/dense assumption for an unfamiliar model family.

## Results

See:
- **[test-results.md](test-results.md)** — the coder-fleet sweep (S86),
  complete: 17 scored models + 3 skipped-as-non-viable, full leaderboard and
  charts.
- **[assistant-results.md](assistant-results.md)** — the assistant-fleet
  sweep, **partial and ongoing**: 12 models with at least one dimension of
  data, coverage ranges from 4/7 to 7/7 dimensions per model, generated
  directly from the same Postgres tables described above by
  [`scripts/query_assistant_results.py`](scripts/query_assistant_results.py).

## Known gaps / honest limitations

These are real, current gaps — not hedging language. They are listed so a
reader (or contributor) knows exactly what is and isn't trustworthy today:

- **`yarn_context_depth` has zero rows.** The dimension is built, documented,
  and unit-tested in Terminus, but has not been run against any model in the
  live fleet yet. No context-collapse-point data exists in Postgres today.
- **Assistant `mem_config` tagging is entirely untagged (100% `NULL`).**
  Every one of the 12 assistant models' 2,362 dimension-score rows carries
  no `mem_config` value. We do not know, from Postgres alone, whether any
  given assistant score was measured under `carveout` or `dynamic_gtt`. The
  coder harness is partially tagged (1,400 of 2,237 rows); the assistant
  harness needs the same backfill.
- **`model_operational_profiles` (context-ceiling / throughput-by-context-size
  table) has 7 rows, all with every numeric column `NULL`** except the model
  name. `max_context_safe`, `quality_degradation_point`,
  `throughput_at_{2k,8k,16k,32k,64k}` — none of it is populated yet. This
  table exists in schema and is queried by the results script, but there is
  currently no operational-profile data to report.
- **The coder suite is Bash/Python/Rust only.** TypeScript/JS corpus cases
  exist in the harness design but did not run in the S86 sweep (a harness
  tag bug, per `test-results.md`) — front-end-heavy models are not
  represented in the coder leaderboard.
- **Best-batch dedup, not all-batch averaging**, is used for the coder
  leaderboard: a model's canonical score is its best clean-GPU batch, to
  avoid a degraded/contaminated CPU-era re-run silently dragging its score
  down. This is a deliberate, documented choice (see `test-results.md`), not
  an oversight — but it means the published number is a "best observed,"
  not a mean.
- **Assistant sweep coverage is uneven across models**: some models have 2
  full profile runs (`assistant_profile_run`, harness `s84-asmt-01`) across
  7 dimensions; others have a single run and only 4–6 dimensions. Treat
  single-run, low-dimension-count rows as directional, not final.
- **Three nominated coder models were skipped as non-viable**
  (`OlympicCoder-32B`, `ornith-9b-fixed`, `ornith-35b-fixed`) — they hit
  per-case timeouts against the patience cap before producing a scoreable
  result. This is recorded as a skip with reason, not folded in as a zero.

## Contributing your own hardware's results

See **[contributing-results.md](contributing-results.md)**.

## References

- **YaRN** — "Yet another RoPE extensioN" — a RoPE (Rotary Position
  Embedding) frequency-scaling technique for extending a model's usable
  context window beyond its trained length. This harness treats YaRN's
  *advertised* extended context as a hypothesis to be probed, not a given —
  see [YaRN collapse detection](#yarn-collapse-detection-dim7_yarn_depthrs)
  above. (We are not citing a specific paper/version here beyond the
  technique name — if you know the canonical citation and want to add it,
  see [contributing-results.md](contributing-results.md).)
- **Graduated / multi-shot code evaluation vs. single-shot** — the general
  practice of scoring code-generation quality on a multi-level rubric
  (compiles → tests pass → behaviorally correct → idiomatic) rather than a
  single pass/fail bit, and of allowing a bounded retry pass. This harness's
  specific 0–5 scale and single-retry rule are project-specific design
  (`code_v2.rs`), not a reproduction of an external benchmark's exact rubric.
- **LLM-judge-panel scoring for subjective quality** — using multiple
  independent LLM "judges" to score subjective dimensions (coherence,
  personality-trait adherence, idiom quality) and aggregating/handling
  disagreement, rather than relying on a single judge or purely automatic
  metrics. This harness's specific contract (fixed JSON-only suffix,
  retry-once, abstain-on-second-failure, redacted raw-output audit trail) is
  documented in `judges.rs` and described above; we are not claiming this
  reproduces any particular published judge-panel methodology verbatim.
