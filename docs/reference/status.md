## Status

Chord ships as the standalone Rust crate `chord-proxy`, version **1.4**. The
core inference manager — routing, the substrate-aware Memory Coordinator, the
verified Clean-Swap Launcher, the Mode Controller, the SNAP observability
subsystem, and per-runtime kernel egress isolation — is in place, as is the MCP
tool-dispatch surface, the personal federation, the S92 hybrid agentic router,
and Chord's own <secret-manager> client.

The core `/v1/tools/*` registry embeds **`terminus-rs` pinned at `1.1.0`** (the
serving-profile types + intake DB config + the tool registry), resolved from the
Gitea crate registry. That pin currently **predates** the latest Terminus
plane-helper additions (`PLANE_PAT_<NAME>` multi-identity, the shared-Redis GET
cache / rate-limit queue, the prefix registry, and the module-management tools).
A dependency bump to pick those up is **tracked and pending** — operator-gated on
the new `terminus-rs` being published to the Gitea registry, then a Chord rebuild
and redeploy. Until that lands, those newest plane-helper tools are reachable via
the standalone `terminus_personal` path (personal federation), **not** the core
`/v1/tools/*` route on the currently deployed binary. The S86 coder-fleet
benchmark charts and leaderboard are published (see
[Test Results](#test-results) below).

