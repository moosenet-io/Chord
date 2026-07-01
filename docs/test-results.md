# Test Results — S86 Coder-Fleet Sweep

These are results from the **S86 coder-fleet sweep** run on the `gfx1151` (Radeon 8060S / Strix Halo) serving host, using the **MINT v2 coder harness** with **`qwen3:8b`** as the pass/fail judge. Each model runs a fixed 31-case suite of `build_modify` tasks across Bash, Python, and Rust. Cases split into **BLITZ** (single-file edits, 14 cases) and **MULTI-FILE** (2+ files touched, 17 cases). A case *passes* when the produced change is correct **and** compiles **and** the run recorded no error; results are deduplicated to the latest sweep per model.

## Pass rate by model

![BLITZ vs MULTI-FILE pass rate by model](charts/coder-sweep-blitz-vs-multi.svg)

![Overall pass-rate leaderboard](charts/coder-sweep-leaderboard.svg)

## Results

| Model | BLITZ | MULTI | Overall |
|---|---|---|---|
| `qwen3-coder-next:latest` | 100% (14/14) | 65% (11/17) | **81%** (25/31) |
| `qwen3-coder:30b` | 100% (14/14) | 65% (11/17) | **81%** (25/31) |
| `qwen2.5-coder:32b-instruct` | 21% (3/14) | 53% (9/17) | **39%** (12/31) |
| `codestral:latest` | 50% (7/14) | 24% (4/17) | **35%** (11/31) |
| `devstral:24b` | 36% (5/14) | 29% (5/17) | **32%** (10/31) |
| `deepcoder:14b` | 14% (2/14) | 29% (5/17) | **23%** (7/31) |
| `qwen2.5-coder:14b-instruct` | 7% (1/14) | 18% (3/17) | **13%** (4/31) |
| `olmo2:13b` | 7% (1/14) | 6% (1/17) | **6%** (2/31) |
| `gemma3:12b` | 0% (0/14) | 6% (1/17) | **3%** (1/31) |
| `qwen3.5:27b` | 0% (0/14) | 0% (0/17) | **0%** (0/31) |
| `qwen3:32b` | 0% (0/14) | 0% (0/17) | **0%** (0/31) |
| `starcoder2:15b` | 0% (0/14) | 0% (0/17) | **0%** (0/31) |
| `yi-coder:9b` | 0% (0/14) | 0% (0/17) | **0%** (0/31) |

## Takeaways

- **`qwen3-coder:30b` is the top MoE and is now the served model** — it ties for first at **81% overall (25/31)** with a perfect **100% BLITZ** score, and it is what the production serving deployment on `gfx1151` now runs. `qwen3-coder-next:latest` matches it point-for-point on this suite.
- **Multi-file is where the fleet separates.** Both leaders clear BLITZ perfectly but still drop to **65% on MULTI-FILE**, and every other model falls further on multi-file work — the coordination of edits across 2+ files, not single-file correctness, is the real discriminator.
- **Base and general-purpose models are weak here.** General/instruct-tuned but non-coder-specialized models (`gemma3:12b`, `olmo2:13b`, and the `qwen3.5:27b` / `qwen3:32b` general chat models) land at or near **0%**, confirming that coder-specialized weights matter for this harness. `qwen2.5-coder:32b-instruct` is the strongest of the non-MoE coders at 39% overall, driven mostly by multi-file (53%).
- **Coverage caveats.** The suite is Bash/Python/Rust only — there is **no TypeScript/JS coverage** yet, so front-end-heavy models are not represented. Additionally, **7 tail candidate models were deferred** from this sweep when the GPU was claimed by the production serving deployment; they will be folded into a follow-up run.
