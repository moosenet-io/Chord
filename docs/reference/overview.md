## Overview

Chord is the inference manager for the Lumina / Harmony stack. It fronts one or
more local LLM backends and decides, per request, **which model runs on which
backend, with which launch flags, and whether it can fit in memory at all**.

Lumina runs interactive assistant workloads; Harmony runs agentic coder
workloads. Both share the same finite pool of accelerator memory on
unified-memory APUs, where a single misjudged context size or a leaked process
can starve the next model. Chord exists to make that sharing safe and
deterministic: it measures each model's serving profile, routes accordingly,
admits work only when memory genuinely allows, swaps models cleanly, and tracks
the operating mode of the whole fleet.

Chord is also the constellation's **MCP tool front door**: it embeds the
`terminus-rs` tool registry and exposes it over an authenticated HTTP surface
(`/v1/tools/*`), so agents reach the build-pipeline toolset (Gitea, GitHub,
Plane, DiffusionGemma review, model-serving introspection) and the LLM backends
through one JWT-guarded service. See [MCP tool dispatch](#mcp-tool-dispatch)
below.

Chord ships as the standalone crate `chord-proxy`, extracted from the former
`lumina-constellation` monorepo into this repository.

