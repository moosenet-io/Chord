//! Chord "Backends" panel (S91 CTUI-02).
//!
//! Read-only view of the inference backends (llama.cpp-rocm, ollama-rocm, cpu,
//! vulkan-radv, …) derived from the stable model registry. No mutations.

use crate::modes::chord::chord_client::BackendStatus;

/// One formatted backend row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BackendDisplay {
    pub name: String,
    pub loaded: String,
}

pub fn display_backend(b: &BackendStatus) -> BackendDisplay {
    BackendDisplay {
        name: b.name.clone(),
        loaded: format!("{} loaded", b.loaded_models),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_backend_row() {
        let b = BackendStatus { name: "vulkan-radv".into(), loaded_models: 2 };
        let d = display_backend(&b);
        assert_eq!(d.name, "vulkan-radv");
        assert_eq!(d.loaded, "2 loaded");
    }
}
