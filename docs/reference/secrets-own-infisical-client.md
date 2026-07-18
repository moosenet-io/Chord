## Secrets (own <secret-manager> client)

Chord fetches its own runtime secrets rather than having them written into a
unit file or brokered by another service. A shared, dependency-light <secret-manager>
Universal-Auth client ([`crates/chord-secrets`](crates/chord-secrets), used by
both `chord-proxy` and the `chord-tui` control client) authenticates with
Chord's own machine identity (`INFISICAL_URL` / `INFISICAL_CLIENT_ID` /
`INFISICAL_CLIENT_SECRET` plus `CHORD_INFISICAL_PROJECT_ID` /
`CHORD_INFISICAL_ENVIRONMENT` / `CHORD_INFISICAL_SECRET_PATH`). The client
itself is intentionally stateless — it authenticates fresh per call and keeps no
token cache or background refresh thread. `chord-proxy` uses it for a one-shot
startup fetch of values such as `CHORD_JWT_SECRET` / `CHORD_API_KEY` /
`OPENROUTER_API_KEY` (EMBED-01's OpenRouter fallback key); the
`chord-tui` control client wraps it in an `InfisicalSecretManager` whose TTL
cache coalesces concurrent cold-cache misses so its poll tasks never stampede
<secret-manager> with a re-auth storm. When <secret-manager> config is absent the sanctioned
env-var fallback is used — no secret is ever hardcoded.

