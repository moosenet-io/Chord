# chord-secrets

Chord's own <secret-manager> Universal Auth client (CSEC-01). Shared by:

- the root `chord-proxy` binary, which fetches `CHORD_JWT_SECRET`/`CHORD_API_KEY`
  from <secret-manager> at startup (CSEC-02), and
- `chord-tui`'s `InfisicalSecretManager` (CSEC-03).

## Why a separate crate

Chord authenticates to <secret-manager> directly, with its own bootstrap identity — this
is a standing architectural decision, not brokered through `terminus_personal` or
any other fleet service, since some internal hops aren't TLS-terminated. Both
Chord consumers of this flow need the exact same Universal Auth request shape, so
it lives once here rather than being duplicated in `chord-proxy` and `chord-tui`.

## Configuration (env vars, no literals)

- `INFISICAL_URL` — base URL of the <secret-manager> instance.
- `INFISICAL_CLIENT_ID` / `INFISICAL_CLIENT_SECRET` — Universal Auth machine
  identity credentials for Chord's own bootstrap identity.

All three must be set (and non-empty) for `InfisicalConfig::is_configured()` to
return `true`. When any are missing, callers get a clean "not configured" signal
(`SecretError::NotConfigured`) rather than a hard failure — this is deliberate so
a deployment that hasn't migrated to <secret-manager>-backed secrets yet keeps working
off its static environment.

## API

```rust
use chord_secrets::{InfisicalConfig, fetch_secrets_batch, fetch_secret};

let config = InfisicalConfig::from_env();
if config.is_configured() {
    let secrets = fetch_secrets_batch(&config, project_id, environment, secret_path).await?;
}
```

No background refresh thread or TTL cache — this crate authenticates fresh per
call, matching Chord's one-shot startup-fetch use case. Secret values are never
logged or included in any `Debug`/`Display` output from this crate.
