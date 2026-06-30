use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProxyError {
    #[error("MCP backend error: {0}")]
    McpBackend(String),

    #[error("MCP session error: {0}")]
    Session(String),

    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    #[error("Tool execution error: {0}")]
    ToolExecution(String),

    #[error("Timeout executing tool: {0}")]
    Timeout(String),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Configuration error: {0}")]
    Config(String),
}

#[derive(Debug, Error)]
pub enum AuthError {
    #[error("Missing Authorization header")]
    MissingHeader,

    #[error("Invalid Authorization header format")]
    InvalidFormat,

    #[error("Invalid JWT token: {0}")]
    InvalidToken(String),

    #[error("JWT expired")]
    Expired,

    #[error("Invalid JWT subject")]
    InvalidSubject,
}
