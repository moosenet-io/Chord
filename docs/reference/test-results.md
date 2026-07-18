## Test Results

Results from the **S86 coder-fleet sweep** on `gfx1151` (MINT v2 harness,
`qwen3:8b` judge) — full charts, table, and takeaways in
[`docs/test-results.md`](docs/test-results.md). Assistant-fleet results
(S84 ASMT sweep, in progress) are in
[`docs/assistant-results.md`](docs/assistant-results.md). Full methodology
in [`docs/model-testing-methodology.md`](docs/model-testing-methodology.md).

[![BLITZ vs MULTI-FILE pass rate by model](docs/charts/coder-sweep-blitz-vs-multi.svg)](docs/test-results.md)

`qwen3-coder:30b` tops the fleet at **81% overall** with a perfect BLITZ score and
is now the served model on `gfx1151`; `qwen2.5-coder:14b-instruct` is the best value
at **77%**, tying the MoEs on multi-file. The completed sweep folds in the tail
models, including the standout Hugging Face find `Seed-Coder-8B` at **61%** — the
strongest multi-file showing outside the leaders for an 8B model. Multi-file
coordination is where the fleet separates, and base/general models score near zero.

