# mcp_proxy (+ catalog, fallback, allowlist)

The MCP tool surface (37 + 34 KG nodes: `src/mcp_proxy.rs`, `src/catalog.rs`,
plus `src/fallback.rs` and `src/tool_allowlist.rs`). `McpProxy` routes tool
requests between callers (Lumina core, agents) and the MCP backend, and falls
back to in-process Rust tools (the compiled-in `terminus-rs` registry) when the
backend is unavailable — so the tool surface degrades, it never disappears.

## Key types and functions

| Symbol | Kind | File | What it does |
|---|---|---|---|
| `mcp_proxy::McpProxy::new` | function | `src/mcp_proxy.rs` | The filtered proxy used for the default tool surface (allowlist-scoped) |
| `mcp_proxy::McpProxy::new_unfiltered` | function | `src/mcp_proxy.rs` | The deliberately unfiltered variant used only for the personal-backend federation proxy |
| `mcp_proxy::McpProxy::tool_list` | function | `src/mcp_proxy.rs` | The merged catalog fetch behind `/v1/tools/list` |
| `mcp_proxy::FallbackTool` | trait | `src/mcp_proxy.rs` | The in-process tool contract — lives here so chord-proxy never depends on terminus-rs types directly |
| `mcp_proxy::FallbackRegistry::register` / `call` / `contains` | functions | `src/mcp_proxy.rs` | The fallback registry; `contains` is the single highest-ranked function in the repo's call graph |
| `mcp_proxy::FallbackRegistry::as_catalog_entries` | function | `src/mcp_proxy.rs` | Exposes Rust tools as catalog entries for merging |
| `fallback::build_fallback_registry` / `TerminusToolProxy` | function / struct | `src/fallback.rs` | Bridges `terminus_rs::ToolRegistry` (after `register_all`) into `FallbackTool`s — one proxy per tool, lock-free concurrent calls over a shared `Arc` |
| `catalog::ToolCatalog::new` / `update` / `find` / `all` | functions | `src/catalog.rs` | The merged catalog with a `CHORD_CATALOG_CACHE_SECS` cache (default 5 minutes) |
| `catalog::ToolEntry::from_mcp` / `from_rust` | functions | `src/catalog.rs` | Entry constructors; `source` is `"mcp"` or `"chord"` — MCP tools win name conflicts, Rust tools take the fallback position |
| `catalog::extract_tool_result` / `parse_mcp_tools` | functions | `src/catalog.rs` | MCP response parsing |
| `tool_allowlist` | module | `src/tool_allowlist.rs` | The core-serving allowlist: governs what `/v1/tools/list|discover|call` will surface **or execute**, regardless of what either registry contains — knowing a hidden tool's name is not enough to call it |
| `session::McpSession::with_token` | function | `src/session.rs` | Authenticated MCP backend session |

## How it connects

**routes** (`tools_list` / `tools_call` / `tools_discover`) and **agentic**
(every guarded tool call in the loop) both dispatch through the same `McpProxy`
instances built in `main.rs`. The primary proxy tries the MCP backend
(`MCP_BACKEND_URL`) first and falls back to the Rust registry; the catalog
merges both sources. A second, optional proxy federates a personal tool backend
(`PERSONAL_BACKEND_URL`) under `/v1/personal/tools/*` — built `new_unfiltered`
with an **empty** fallback registry on purpose: it is a pure passthrough to the
personal backend's own catalog and must never silently serve Chord's in-process
tools under the personal routes, and it is never merged into the default
catalog.

## Configuration

`MCP_BACKEND_URL`, `MCP_BACKEND_TOKEN`, `CHORD_CATALOG_CACHE_SECS`,
`CHORD_TOOL_TIMEOUT_SECS`, `PERSONAL_BACKEND_URL`, `PERSONAL_BACKEND_TOKEN`.

## Notes and gaps

- The allowlist boundary is deliberate scope-setting: Chord serves model
  routing and build-pipeline tooling externally; general-purpose
  secrets/personal-utility/ops tools belong to the fleet's tool hub, not
  Chord's served catalog. The federation routes exist precisely so the personal
  catalog stays a separate, opt-in surface.
- With the MCP backend down, only tools that exist in the Rust registry keep
  working — backend-only tools 503 until it returns.
- Tool execution timeouts are governed by `CHORD_TOOL_TIMEOUT_SECS`; audit
  records carry only tool names and outcomes, never arguments.
