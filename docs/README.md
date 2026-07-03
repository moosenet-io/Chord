# Chord — Documentation

Component explainers for Chord (`chord-proxy`), the inference manager for local
LLM fleets. Every page is written from the actual source in [`../src/`](../src)
and names the real modules and types behind each component.

## Contents

- **[architecture.md](architecture.md)** — full component deep-dive: the two
  listeners, the request flow through `/v1/chat/completions`, and each component
  (Routing, backend tiers, model registry & storage tiering, memory/residency
  management, the control API, the agentic loop, and the search harness) mapped to
  its real module/types. References the
  [architecture diagram](../assets/architecture.svg).
- **[serving.md](serving.md)** — the serving / coordinator subsystem: how the
  diagram's **Memory Coordinator** (SRV-11), **Clean-Swap Launcher** (SRV-12), and
  **Mode Controller** (SRV-13) boxes map onto the code that actually ships, with an
  explicit present/partial/absent table.
- **[egress.md](egress.md)** — runtime isolation: ISO-01 env-scrub + egress policy
  and the ISO-02 per-runtime network-namespace kernel enforcement (serve = no
  route, pull = allow-list), fail-closed posture, and honest scope.
- **[snap-persistence.md](snap-persistence.md)** — the optional, default-off
  (`CHORD_SNAP_PERSIST`) SNAP → Postgres persistence of the analytics, inventory,
  activity, and VRAM observability streams.

## Model Testing / Benchmarking

- **[model-testing-methodology.md](model-testing-methodology.md)** — the
  full benchmarking methodology: what the coder and assistant harnesses
  measure and why, graduated 0-5 scoring, the 3-judge panel retry/abstain
  contract, YaRN collapse detection, `mem_config` hardware tagging, known
  gfx1151 backend quirks (ROCm vs. Vulkan vs. CPU), and an honest gaps list.
  The harness code itself lives in
  [`moosenet/Terminus`](../../Terminus)`/src/intake/` — this repo only hosts
  the results and the query tooling.
- **[test-results.md](test-results.md)** — the **S86 coder-fleet sweep** on
  `gfx1151` (MINT v2 harness, `qwen3:8b` judge): themed per-model BLITZ vs
  MULTI-FILE pass-rate charts, an overall leaderboard, the full results table, and
  takeaways. Generated charts live in [`charts/`](charts/). Complete (17
  scored models, 3 skipped-as-non-viable).
- **[assistant-results.md](assistant-results.md)** — the **S84 ASMT
  assistant-fleet sweep**: 7 dimensions (conversation depth, tool chaining,
  memory integration, OCEAN personality, prompted personality/behavioral
  drift, embeddings, YaRN context-depth), data-driven, generated straight
  from Postgres by
  [`scripts/query_assistant_results.py`](scripts/query_assistant_results.py).
  **Partial/in-progress** — 12 models with 4-7 of 7 dimensions covered; see
  the gaps list in model-testing-methodology.md before citing any number.
- **[contributing-results.md](contributing-results.md)** — how to run
  either harness on your own hardware and tag your results so they're never
  silently blended with the gfx1151 numbers here.
