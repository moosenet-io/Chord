//! Axum middleware that wraps every route handler to produce audit log entries.
//!
//! The middleware:
//!   1. Records the start time.
//!   2. Extracts the JWT `sub` claim from the `Authorization` header (if present)
//!      to populate `user_id`. When absent or invalid the user is logged as
//!      `"anonymous"`. The raw token is never stored.
//!   3. After the inner handler returns, maps the HTTP status code to an audit
//!      `Status` and logs the entry.
//!
//! Sensitive content (request bodies, tool arguments, LLM messages) is never
//! inspected or stored — the middleware operates only on headers and the
//! response status code.

use std::sync::Arc;
use std::task::{Context, Poll};

use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::Response,
};
use futures_util::future::BoxFuture;
use tower::{Layer, Service};

use crate::audit::{AuditEntry, AuditLogger, RequestType, Status, Timer};
use crate::auth::{extract_bearer, validate_jwt};

// ── Layer (factory) ───────────────────────────────────────────────────────────

/// Tower [`Layer`] that injects [`AuditMiddleware`] into the service stack.
#[derive(Clone)]
pub struct AuditLayer {
    logger: Arc<AuditLogger>,
    jwt_secret: String,
}

impl AuditLayer {
    pub fn new(logger: Arc<AuditLogger>, jwt_secret: String) -> Self {
        Self { logger, jwt_secret }
    }
}

impl<S> Layer<S> for AuditLayer {
    type Service = AuditMiddleware<S>;

    fn layer(&self, inner: S) -> Self::Service {
        AuditMiddleware {
            inner,
            logger: self.logger.clone(),
            jwt_secret: self.jwt_secret.clone(),
        }
    }
}

// ── Middleware service ────────────────────────────────────────────────────────

/// Tower [`Service`] that wraps every request with audit logging.
#[derive(Clone)]
pub struct AuditMiddleware<S> {
    inner: S,
    logger: Arc<AuditLogger>,
    jwt_secret: String,
}

impl<S> Service<Request<Body>> for AuditMiddleware<S>
where
    S: Service<Request<Body>, Response = Response> + Send + Clone + 'static,
    S::Future: Send + 'static,
    S::Error: Send + 'static,
{
    type Response = Response;
    type Error = S::Error;
    type Future = BoxFuture<'static, Result<Self::Response, Self::Error>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<Body>) -> Self::Future {
        let timer = Timer::start();
        let logger = self.logger.clone();
        let jwt_secret = self.jwt_secret.clone();

        // Derive audit fields from the request BEFORE moving it into the handler.
        let method = req.method().clone();
        let uri    = req.uri().path().to_string();
        let request_type = path_to_request_type(&uri);
        let target = String::new();
        let user_id = extract_user_id(req.headers(), &jwt_secret);
        // Extract raw token for auth-failure logging (hashed before storage).
        let raw_token: Option<String> = req
            .headers()
            .get("Authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|h| extract_bearer(h).ok())
            .map(|t| t.to_string());

        let mut inner = self.inner.clone();

        Box::pin(async move {
            let response: Response = inner.call(req).await?;
            let duration_ms = timer.elapsed_ms();
            let status_code = response.status();

            // Skip health and audit endpoints — no recursive audit noise.
            let should_log = !uri.eq("/health") && !uri.starts_with("/v1/audit");

            if should_log {
                let audit_status = http_status_to_audit_status(status_code);
                let error_msg = if audit_status != Status::Success {
                    Some(format!("HTTP {}", status_code.as_u16()))
                } else {
                    None
                };

                if status_code == StatusCode::UNAUTHORIZED {
                    // Log auth failure with hashed token prefix (never the raw token).
                    logger.log_auth_failure(raw_token.as_deref(), duration_ms);
                } else {
                    let entry = if audit_status == Status::Success {
                        AuditEntry::success(&user_id, request_type, &target, duration_ms)
                    } else {
                        AuditEntry::failed(
                            &user_id,
                            request_type,
                            &target,
                            duration_ms,
                            audit_status,
                            error_msg,
                        )
                    };
                    logger.log_entry(&entry);
                }
            }

            let _ = (method, target); // consumed
            Ok(response)
        })
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Map URI path to a `RequestType`.
fn path_to_request_type(path: &str) -> RequestType {
    match path {
        "/v1/tools/list"     => RequestType::ToolList,
        "/v1/tools/call"     => RequestType::ToolCall,
        "/v1/tools/discover" => RequestType::ToolDiscover,
        // Auth failures: middleware logs RequestType::AuthFailure directly; this
        // path is reached only for non-401 requests on unrecognised routes.
        _                    => RequestType::ToolList,
    }
}

/// Extract `sub` from the JWT in the `Authorization` header.
/// Returns `"anonymous"` if the header is absent, malformed, or invalid.
/// The raw token value is never returned or stored.
fn extract_user_id(headers: &HeaderMap, jwt_secret: &str) -> String {
    let auth_header = match headers.get("Authorization").and_then(|v| v.to_str().ok()) {
        Some(h) => h,
        None => return "anonymous".to_string(),
    };
    let token = match extract_bearer(auth_header) {
        Ok(t) => t,
        Err(_) => return "anonymous".to_string(),
    };
    match validate_jwt(token, jwt_secret) {
        Ok(claims) => claims.sub,
        Err(_) => "anonymous".to_string(),
    }
}

/// Convert HTTP status code to an audit `Status`.
fn http_status_to_audit_status(code: StatusCode) -> Status {
    match code.as_u16() {
        200..=299 => Status::Success,
        504       => Status::Timeout,
        _         => Status::Error,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_http_status_to_audit_status_success() {
        assert_eq!(http_status_to_audit_status(StatusCode::OK), Status::Success);
        assert_eq!(http_status_to_audit_status(StatusCode::CREATED), Status::Success);
    }

    #[test]
    fn test_http_status_to_audit_status_timeout() {
        assert_eq!(http_status_to_audit_status(StatusCode::GATEWAY_TIMEOUT), Status::Timeout);
    }

    #[test]
    fn test_http_status_to_audit_status_error() {
        assert_eq!(http_status_to_audit_status(StatusCode::BAD_GATEWAY), Status::Error);
        assert_eq!(http_status_to_audit_status(StatusCode::UNAUTHORIZED), Status::Error);
        assert_eq!(http_status_to_audit_status(StatusCode::NOT_FOUND), Status::Error);
    }

    #[test]
    fn test_path_to_request_type() {
        assert_eq!(path_to_request_type("/v1/tools/list"), RequestType::ToolList);
        assert_eq!(path_to_request_type("/v1/tools/call"), RequestType::ToolCall);
        assert_eq!(path_to_request_type("/v1/tools/discover"), RequestType::ToolDiscover);
    }

    #[test]
    fn test_extract_user_id_anonymous_when_no_header() {
        let headers = HeaderMap::new();
        let user = extract_user_id(&headers, "secret");
        assert_eq!(user, "anonymous");
    }

    #[test]
    fn test_extract_user_id_anonymous_on_invalid_jwt() {
        let mut headers = HeaderMap::new();
        headers.insert("Authorization", "Bearer invalid.token.here".parse().unwrap());
        let user = extract_user_id(&headers, "secret");
        assert_eq!(user, "anonymous");
    }

    #[test]
    fn test_extract_user_id_when_secret_empty() {
        // When secret is empty, auth is disabled → sub returns "lumina".
        let mut headers = HeaderMap::new();
        headers.insert("Authorization", "Bearer anything".parse().unwrap());
        let user = extract_user_id(&headers, "");
        assert_eq!(user, "lumina");
    }
}
