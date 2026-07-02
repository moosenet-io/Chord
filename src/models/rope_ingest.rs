//! YARN-02: model-config ingestion — read a model's OWN RoPE/YaRN metadata from
//! its GGUF key/value header at registration time, so the pre-filled
//! [`RopeScaling`] block Chord stores reflects what the model was actually
//! tuned for, never a value Chord invented. This is the producer half of
//! YARN-01's consumer-side [`RopeScaling`]/[`RopeScalingMethod`] types.
//!
//! ## Why a hand-rolled GGUF kv reader
//! Nothing in this crate parses GGUF binary metadata yet: `snap/inventory.rs`
//! only scans *file names* for `.gguf` / quant tags, and `models/registry.rs`
//! only tracks paths/sizes from Ollama's JSON manifests — there was no existing
//! "read a GGUF kv" mechanism to reuse. This module adds the minimal GGUF-spec
//! kv-header reader needed to answer "what rope config does this model
//! declare": magic + version + counts, then walks every metadata kv pair far
//! enough to type-skip values it doesn't need (the format is sequential, so a
//! value has to be parsed to advance past it even when its key is irrelevant).
//! It deliberately stops after the kv-metadata block, before the tensor-info
//! section — every key this module reads lives in that block.
//!
//! Key matching is by suffix (`*.context_length`, `*.rope.scaling.type`, …)
//! rather than a hardcoded architecture prefix, since GGUF namespaces every
//! architecture-specific key under `<arch>.` and the architecture varies
//! per-model (`general.architecture` names it, but callers don't need to know
//! it up front to read the rope keys).
//!
//! ## Never fabricate a scale factor
//! [`derive_rope_scaling`] only ever produces a [`RopeScaling`] block using
//! numbers the GGUF itself states (`rope_scale`, `yarn_orig_ctx`, `target_ctx`).
//! When a required value is missing, it returns
//! [`RopeIngestOutcome::ManualConfigRequired`] with a log-worthy note instead of
//! guessing. The yarn fine-tune quartet (`ext_factor`/`attn_factor`/
//! `beta_slow`/`beta_fast`) falls back to llama.cpp's OWN stock defaults when
//! the GGUF doesn't override them — that is matching what the runtime would
//! default to anyway, not inventing a per-model value; the load-bearing,
//! never-guessed fields (`rope_scale`, `yarn_orig_ctx`) are always
//! metadata-sourced.
//!
//! ## GGUF vs. model-card / HF-config disagreement
//! Per the YARN-02 edge cases: when a model's card or HF `config.json` claims a
//! context the GGUF doesn't support, the GGUF wins — this module only reads
//! the GGUF (what's actually loaded), so there is no separate model-card path
//! to reconcile against. A caller that also has HF/config-derived values should
//! prefer whatever this module returns and log the discrepancy itself.
//!
//! ## `override-kv`-style context override
//! Some Qwen GGUF conversions understate their own `<arch>.context_length` kv
//! relative to the long-context config they were actually tuned for (fixed at
//! launch time via llama.cpp's `--override-kv <arch>.context_length=int:N`).
//! Full override-kv plumbing is out of scope here (YARN-02 only needs to not
//! silently ignore it): when a yarn/linear block's own `context_length` kv does
//! not exceed the declared `original_context_length`, [`derive_rope_scaling`]
//! still pre-fills the block from the stated numbers but logs a warning
//! flagging that an `--override-kv` context bump may be required.

use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

use crate::serving::profile::{RopeScaling, RopeScalingMethod};

/// llama.cpp's stock YaRN fine-tune defaults, used ONLY when the GGUF itself
/// does not state a value. These match what llama.cpp applies by default when
/// `--rope-scaling yarn` is requested without explicit overrides — reusing
/// them is not per-model fabrication, it's matching the runtime's own
/// architecture-standard constants. The values that ARE per-model and are
/// NEVER defaulted are `rope_scale` and `yarn_orig_ctx`.
const YARN_DEFAULT_EXT_FACTOR: f64 = 1.0;
const YARN_DEFAULT_ATTN_FACTOR: f64 = 1.0;
const YARN_DEFAULT_BETA_SLOW: f64 = 1.0;
const YARN_DEFAULT_BETA_FAST: f64 = 32.0;

/// The subset of a GGUF's kv-metadata block relevant to RoPE/YaRN scaling
/// ingestion. `None` fields mean the key was absent from this model's GGUF —
/// NOT zero, NOT unknown-but-assumed.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GgufRopeKv {
    /// `general.architecture` (e.g. `"qwen2"`, `"llama"`), if present.
    pub architecture: Option<String>,
    /// `<arch>.context_length` — the context the GGUF itself declares.
    pub context_length: Option<u32>,
    /// `<arch>.rope.scaling.type` (e.g. `"none"`, `"linear"`, `"yarn"`, or a
    /// non-Chord method such as `"ntk"`), lowercased.
    pub scaling_type: Option<String>,
    /// `<arch>.rope.scaling.factor` (falls back to the legacy
    /// `<arch>.rope.scale_linear` key when present instead).
    pub scaling_factor: Option<f64>,
    /// `<arch>.rope.scaling.original_context_length` — the yarn "trained on"
    /// context before extension.
    pub orig_ctx: Option<u32>,
    /// `<arch>.rope.scaling.attn_factor`, if the GGUF states one explicitly.
    pub attn_factor: Option<f64>,
}

/// Result of [`derive_rope_scaling`]: what YARN-02 ingestion concluded about a
/// model's own rope configuration.
#[derive(Debug, Clone, PartialEq)]
pub enum RopeIngestOutcome {
    /// The model states no long-context scaling (`method` absent or `"none"`).
    /// Maps to `RopeScaling { method: RopeScalingMethod::None, .. }` — the
    /// EnvSpec "no-op method" shape from YARN-01.
    NoLongContext,
    /// The model's own metadata fully supports an unvalidated pre-fill.
    /// `validated` is always `false` here — YARN-03 (not yet built) owns
    /// promoting a block to validated.
    Prefilled(RopeScaling),
    /// The model declares scaling but is missing a value this ingestion will
    /// not guess (e.g. yarn without `original_context_length`). The `String`
    /// is a human-readable note for the caller to log; no `RopeScaling` is
    /// produced.
    ManualConfigRequired(String),
}

/// A scalar value read out of one GGUF kv entry. Only the variants ingestion
/// actually needs; arrays are walked (to keep the reader position correct)
/// but never materialized as a value here (see [`skip_or_read_value`]).
#[derive(Debug, Clone, PartialEq)]
enum GgufScalar {
    UInt(u64),
    Int(i64),
    Float(f64),
    Str(String),
}

impl GgufScalar {
    fn as_u32(&self) -> Option<u32> {
        match self {
            GgufScalar::UInt(v) => u32::try_from(*v).ok(),
            GgufScalar::Int(v) => u32::try_from(*v).ok(),
            _ => None,
        }
    }

    fn as_f64(&self) -> Option<f64> {
        match self {
            GgufScalar::Float(v) => Some(*v),
            GgufScalar::UInt(v) => Some(*v as f64),
            GgufScalar::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            GgufScalar::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

fn read_exact_buf<R: Read, const N: usize>(r: &mut R) -> io::Result<[u8; N]> {
    let mut buf = [0u8; N];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

fn read_u8(r: &mut impl Read) -> io::Result<u8> {
    Ok(read_exact_buf::<_, 1>(r)?[0])
}
fn read_u16(r: &mut impl Read) -> io::Result<u16> {
    Ok(u16::from_le_bytes(read_exact_buf(r)?))
}
fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    Ok(u32::from_le_bytes(read_exact_buf(r)?))
}
fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    Ok(u64::from_le_bytes(read_exact_buf(r)?))
}
fn read_i8(r: &mut impl Read) -> io::Result<i8> {
    Ok(read_exact_buf::<_, 1>(r)?[0] as i8)
}
fn read_i16(r: &mut impl Read) -> io::Result<i16> {
    Ok(i16::from_le_bytes(read_exact_buf(r)?))
}
fn read_i32(r: &mut impl Read) -> io::Result<i32> {
    Ok(i32::from_le_bytes(read_exact_buf(r)?))
}
fn read_i64(r: &mut impl Read) -> io::Result<i64> {
    Ok(i64::from_le_bytes(read_exact_buf(r)?))
}
fn read_f32(r: &mut impl Read) -> io::Result<f32> {
    Ok(f32::from_le_bytes(read_exact_buf(r)?))
}
fn read_f64(r: &mut impl Read) -> io::Result<f64> {
    Ok(f64::from_le_bytes(read_exact_buf(r)?))
}

/// GGUF string: `u64` length prefix + that many raw bytes (no NUL terminator).
/// Invalid UTF-8 is lossily replaced rather than failing the whole parse — a
/// garbled string in an unrelated key must not abort ingestion of the keys we
/// actually want.
fn read_gguf_string(r: &mut impl Read) -> io::Result<String> {
    let len = read_u64(r)?;
    let len = usize::try_from(len)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "gguf string length overflow"))?;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

/// GGUF metadata value-type tags (see the GGUF spec / `ggml.h`
/// `gguf_metadata_value_type`).
const GGUF_TYPE_UINT8: u32 = 0;
const GGUF_TYPE_INT8: u32 = 1;
const GGUF_TYPE_UINT16: u32 = 2;
const GGUF_TYPE_INT16: u32 = 3;
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_UINT64: u32 = 10;
const GGUF_TYPE_INT64: u32 = 11;
const GGUF_TYPE_FLOAT64: u32 = 12;

/// Read (or, for array elements we don't materialize, just advance past) one
/// value of the given GGUF type. Every branch consumes exactly the bytes that
/// type occupies, so the reader position stays correct regardless of whether
/// the caller cares about this particular key — required because GGUF kv
/// pairs are stored back-to-back with no per-entry length prefix to skip by.
fn skip_or_read_value(r: &mut impl Read, value_type: u32) -> io::Result<Option<GgufScalar>> {
    match value_type {
        GGUF_TYPE_UINT8 => Ok(Some(GgufScalar::UInt(read_u8(r)? as u64))),
        GGUF_TYPE_INT8 => Ok(Some(GgufScalar::Int(read_i8(r)? as i64))),
        GGUF_TYPE_UINT16 => Ok(Some(GgufScalar::UInt(read_u16(r)? as u64))),
        GGUF_TYPE_INT16 => Ok(Some(GgufScalar::Int(read_i16(r)? as i64))),
        GGUF_TYPE_UINT32 => Ok(Some(GgufScalar::UInt(read_u32(r)? as u64))),
        GGUF_TYPE_INT32 => Ok(Some(GgufScalar::Int(read_i32(r)? as i64))),
        GGUF_TYPE_FLOAT32 => Ok(Some(GgufScalar::Float(read_f32(r)? as f64))),
        GGUF_TYPE_BOOL => Ok(Some(GgufScalar::UInt(read_u8(r)? as u64))),
        GGUF_TYPE_STRING => Ok(Some(GgufScalar::Str(read_gguf_string(r)?))),
        GGUF_TYPE_UINT64 => Ok(Some(GgufScalar::UInt(read_u64(r)?))),
        GGUF_TYPE_INT64 => Ok(Some(GgufScalar::Int(read_i64(r)?))),
        GGUF_TYPE_FLOAT64 => Ok(Some(GgufScalar::Float(read_f64(r)?))),
        GGUF_TYPE_ARRAY => {
            let elem_type = read_u32(r)?;
            let len = read_u64(r)?;
            // Arrays (e.g. the tokenizer vocab) can be huge; ingestion never
            // needs their contents, only to advance past them correctly.
            for _ in 0..len {
                skip_or_read_value(r, elem_type)?;
            }
            Ok(None)
        }
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unrecognized gguf metadata value type",
        )),
    }
}

/// Read the RoPE/YaRN-relevant subset of a GGUF file's kv-metadata header.
///
/// Returns `None` when the file cannot be opened, isn't a valid GGUF (bad
/// magic), or the kv section is truncated/malformed — every one of those is
/// "we don't know this model's rope config", which the caller (registration)
/// turns into a manual-configuration note rather than a crash or a guess.
pub fn read_gguf_rope_kv(path: &Path) -> Option<GgufRopeKv> {
    let file = File::open(path).ok()?;
    let mut r = BufReader::new(file);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).ok()?;
    if &magic != b"GGUF" {
        return None;
    }
    let version = read_u32(&mut r).ok()?;
    if version < 2 {
        // v1 used 32-bit counts; every model Chord serves today is v2/v3+.
        // Not worth supporting a format llama.cpp itself dropped.
        return None;
    }
    let _tensor_count = read_u64(&mut r).ok()?;
    let kv_count = read_u64(&mut r).ok()?;

    let mut out = GgufRopeKv::default();
    for _ in 0..kv_count {
        let key = read_gguf_string(&mut r).ok()?;
        let value_type = read_u32(&mut r).ok()?;
        let value = skip_or_read_value(&mut r, value_type).ok()?;

        if key == "general.architecture" {
            out.architecture = value.as_ref().and_then(GgufScalar::as_str).map(str::to_string);
        } else if key.ends_with(".context_length") {
            if let Some(v) = value.as_ref().and_then(GgufScalar::as_u32) {
                out.context_length = Some(v);
            }
        } else if key.ends_with(".rope.scaling.type") {
            out.scaling_type = value
                .as_ref()
                .and_then(GgufScalar::as_str)
                .map(|s| s.to_lowercase());
        } else if key.ends_with(".rope.scaling.factor") || key.ends_with(".rope.scale_linear") {
            if let Some(v) = value.as_ref().and_then(GgufScalar::as_f64) {
                out.scaling_factor = Some(v);
            }
        } else if key.ends_with(".rope.scaling.original_context_length") {
            if let Some(v) = value.as_ref().and_then(GgufScalar::as_u32) {
                out.orig_ctx = Some(v);
            }
        } else if key.ends_with(".rope.scaling.attn_factor") {
            if let Some(v) = value.as_ref().and_then(GgufScalar::as_f64) {
                out.attn_factor = Some(v);
            }
        }
        // Every other key (tokenizer arrays, quantization metadata, etc.) is
        // intentionally ignored — `skip_or_read_value` already advanced past
        // it above, which is all correctness requires.
    }
    Some(out)
}

/// Derive a [`RopeIngestOutcome`] from a model's GGUF-declared rope metadata.
/// Never fabricates `rope_scale`/`yarn_orig_ctx`/`target_ctx` — see the module
/// docs for exactly which fields are allowed to fall back to llama.cpp's own
/// stock defaults (the yarn fine-tune quartet only) and which never are.
pub fn derive_rope_scaling(kv: &GgufRopeKv) -> RopeIngestOutcome {
    let method_str = kv.scaling_type.as_deref().unwrap_or("none");

    match method_str {
        "" | "none" => RopeIngestOutcome::NoLongContext,

        "linear" => match (kv.scaling_factor, kv.context_length) {
            (Some(scale), Some(target_ctx)) => RopeIngestOutcome::Prefilled(RopeScaling {
                method: RopeScalingMethod::Linear,
                rope_scale: scale,
                target_ctx,
                validated: false,
                ..Default::default()
            }),
            _ => RopeIngestOutcome::ManualConfigRequired(format!(
                "model declares rope.scaling.type=linear but GGUF is missing {} — refusing to \
                 fabricate a scale factor; manual configuration required",
                missing_linear_fields(kv)
            )),
        },

        // NTK isn't in Chord's RopeScalingMethod enum (none|linear|yarn, per
        // YARN-01). Per YARN-02 guidance: treat it as linear rather than force
        // yarn, and say so — the scale factor is still whatever the model
        // states, never invented.
        "ntk" | "ntk-aware" => {
            tracing::info!(
                target: "chord.models.rope_ingest",
                "model declares NTK rope scaling (not in Chord's method enum) — \
                 recording as linear per YARN-02 handling"
            );
            match (kv.scaling_factor, kv.context_length) {
                (Some(scale), Some(target_ctx)) => RopeIngestOutcome::Prefilled(RopeScaling {
                    method: RopeScalingMethod::Linear,
                    rope_scale: scale,
                    target_ctx,
                    validated: false,
                    ..Default::default()
                }),
                _ => RopeIngestOutcome::ManualConfigRequired(format!(
                    "model declares rope.scaling.type=ntk (treated as linear) but GGUF is \
                     missing {} — refusing to fabricate a scale factor; manual configuration \
                     required",
                    missing_linear_fields(kv)
                )),
            }
        }

        "yarn" => match (kv.scaling_factor, kv.orig_ctx, kv.context_length) {
            (Some(scale), Some(orig_ctx), Some(target_ctx)) => {
                if target_ctx <= orig_ctx {
                    // The GGUF's own context_length doesn't reflect an
                    // extension over orig_ctx — a known Qwen-family quirk
                    // where the effective long context needs an
                    // `--override-kv <arch>.context_length=int:N` bump at
                    // launch. Not built here (out of scope); just don't
                    // silently ignore it.
                    tracing::warn!(
                        target: "chord.models.rope_ingest",
                        orig_ctx,
                        target_ctx,
                        "yarn declared but GGUF context_length does not exceed \
                         original_context_length — an --override-kv context bump may be \
                         required at launch; pre-filling block from stated values anyway"
                    );
                }
                RopeIngestOutcome::Prefilled(RopeScaling {
                    method: RopeScalingMethod::Yarn,
                    rope_scale: scale,
                    yarn_orig_ctx: orig_ctx,
                    target_ctx,
                    ext_factor: YARN_DEFAULT_EXT_FACTOR,
                    attn_factor: kv.attn_factor.unwrap_or(YARN_DEFAULT_ATTN_FACTOR),
                    beta_slow: YARN_DEFAULT_BETA_SLOW,
                    beta_fast: YARN_DEFAULT_BETA_FAST,
                    validated: false,
                })
            }
            _ => RopeIngestOutcome::ManualConfigRequired(format!(
                "model declares rope.scaling.type=yarn but GGUF is missing {} — refusing to \
                 fabricate a scale factor; manual configuration required",
                missing_yarn_fields(kv)
            )),
        },

        other => RopeIngestOutcome::ManualConfigRequired(format!(
            "model declares unrecognized rope.scaling.type={other:?} — manual configuration \
             required"
        )),
    }
}

/// Human-readable list of the yarn fields this model's GGUF is missing, for
/// the `ManualConfigRequired` note.
fn missing_yarn_fields(kv: &GgufRopeKv) -> String {
    let mut missing = Vec::new();
    if kv.scaling_factor.is_none() {
        missing.push("rope.scaling.factor");
    }
    if kv.orig_ctx.is_none() {
        missing.push("rope.scaling.original_context_length");
    }
    if kv.context_length.is_none() {
        missing.push("context_length");
    }
    missing.join(", ")
}

/// Human-readable list of the linear/ntk fields this model's GGUF is missing.
fn missing_linear_fields(kv: &GgufRopeKv) -> String {
    let mut missing = Vec::new();
    if kv.scaling_factor.is_none() {
        missing.push("rope.scaling.factor");
    }
    if kv.context_length.is_none() {
        missing.push("context_length");
    }
    missing.join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Minimal valid-GGUF-header builder: magic + version 3 + tensor_count 0 +
    /// the given kv pairs. Enough to exercise `read_gguf_rope_kv` without a
    /// real model file.
    struct GgufBuilder {
        buf: Vec<u8>,
        kv_count: u64,
    }

    impl GgufBuilder {
        fn new() -> Self {
            let mut buf = Vec::new();
            buf.extend_from_slice(b"GGUF");
            buf.extend_from_slice(&3u32.to_le_bytes()); // version
            buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
            buf.extend_from_slice(&0u64.to_le_bytes()); // kv_count placeholder, patched in finish()
            GgufBuilder { buf, kv_count: 0 }
        }

        fn push_key(&mut self, key: &str) {
            self.buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
            self.buf.extend_from_slice(key.as_bytes());
        }

        fn kv_string(&mut self, key: &str, value: &str) -> &mut Self {
            self.push_key(key);
            self.buf.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
            self.buf.extend_from_slice(&(value.len() as u64).to_le_bytes());
            self.buf.extend_from_slice(value.as_bytes());
            self.kv_count += 1;
            self
        }

        fn kv_u32(&mut self, key: &str, value: u32) -> &mut Self {
            self.push_key(key);
            self.buf.extend_from_slice(&GGUF_TYPE_UINT32.to_le_bytes());
            self.buf.extend_from_slice(&value.to_le_bytes());
            self.kv_count += 1;
            self
        }

        fn kv_f32(&mut self, key: &str, value: f32) -> &mut Self {
            self.push_key(key);
            self.buf.extend_from_slice(&GGUF_TYPE_FLOAT32.to_le_bytes());
            self.buf.extend_from_slice(&value.to_le_bytes());
            self.kv_count += 1;
            self
        }

        /// An array kv (e.g. a tiny stand-in for a tokenizer vocab list), to
        /// prove the reader correctly skips past array payloads.
        fn kv_string_array(&mut self, key: &str, values: &[&str]) -> &mut Self {
            self.push_key(key);
            self.buf.extend_from_slice(&GGUF_TYPE_ARRAY.to_le_bytes());
            self.buf.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
            self.buf.extend_from_slice(&(values.len() as u64).to_le_bytes());
            for v in values {
                self.buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
                self.buf.extend_from_slice(v.as_bytes());
            }
            self.kv_count += 1;
            self
        }

        fn write_to(&self, path: &Path) {
            // Patch the kv_count placeholder (offset 4 magic-skip is wrong;
            // layout is: magic(4) + version(4) + tensor_count(8) + kv_count(8)).
            let mut out = self.buf.clone();
            out[16..24].copy_from_slice(&self.kv_count.to_le_bytes());
            let mut f = File::create(path).unwrap();
            f.write_all(&out).unwrap();
        }
    }

    #[test]
    fn reads_yarn_declared_model() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        GgufBuilder::new()
            .kv_string("general.architecture", "qwen2")
            .kv_u32("qwen2.context_length", 131072)
            .kv_string("qwen2.rope.scaling.type", "yarn")
            .kv_f32("qwen2.rope.scaling.factor", 4.0)
            .kv_u32("qwen2.rope.scaling.original_context_length", 32768)
            .kv_string_array("tokenizer.ggml.tokens", &["<s>", "</s>", "hello"])
            .write_to(&path);

        let kv = read_gguf_rope_kv(&path).expect("valid gguf parses");
        assert_eq!(kv.architecture.as_deref(), Some("qwen2"));
        assert_eq!(kv.context_length, Some(131072));
        assert_eq!(kv.scaling_type.as_deref(), Some("yarn"));
        assert_eq!(kv.scaling_factor, Some(4.0));
        assert_eq!(kv.orig_ctx, Some(32768));

        let outcome = derive_rope_scaling(&kv);
        match outcome {
            RopeIngestOutcome::Prefilled(rope) => {
                assert_eq!(rope.method, RopeScalingMethod::Yarn);
                assert_eq!(rope.rope_scale, 4.0);
                assert_eq!(rope.yarn_orig_ctx, 32768);
                assert_eq!(rope.target_ctx, 131072);
                assert!(!rope.validated, "YARN-03 owns validation, not ingestion");
                assert!(rope.is_plausible());
            }
            other => panic!("expected Prefilled(yarn), got {other:?}"),
        }
    }

    #[test]
    fn reads_linear_declared_model() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        GgufBuilder::new()
            .kv_string("general.architecture", "llama")
            .kv_u32("llama.context_length", 16384)
            .kv_string("llama.rope.scaling.type", "linear")
            .kv_f32("llama.rope.scaling.factor", 2.0)
            .write_to(&path);

        let kv = read_gguf_rope_kv(&path).expect("valid gguf parses");
        let outcome = derive_rope_scaling(&kv);
        match outcome {
            RopeIngestOutcome::Prefilled(rope) => {
                assert_eq!(rope.method, RopeScalingMethod::Linear);
                assert_eq!(rope.rope_scale, 2.0);
                assert_eq!(rope.target_ctx, 16384);
                assert!(!rope.validated);
            }
            other => panic!("expected Prefilled(linear), got {other:?}"),
        }
    }

    #[test]
    fn reads_no_long_context_model() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        GgufBuilder::new()
            .kv_string("general.architecture", "llama")
            .kv_u32("llama.context_length", 4096)
            .write_to(&path);

        let kv = read_gguf_rope_kv(&path).expect("valid gguf parses");
        assert_eq!(kv.scaling_type, None);
        assert_eq!(derive_rope_scaling(&kv), RopeIngestOutcome::NoLongContext);
    }

    #[test]
    fn explicit_none_method_is_also_no_long_context() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        GgufBuilder::new()
            .kv_string("llama.rope.scaling.type", "none")
            .write_to(&path);
        let kv = read_gguf_rope_kv(&path).unwrap();
        assert_eq!(derive_rope_scaling(&kv), RopeIngestOutcome::NoLongContext);
    }

    #[test]
    fn yarn_missing_orig_ctx_never_fabricates_a_factor() {
        // Declares yarn + a factor + a context_length, but NOT
        // original_context_length — ingestion must refuse to guess it.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        GgufBuilder::new()
            .kv_string("qwen2.rope.scaling.type", "yarn")
            .kv_f32("qwen2.rope.scaling.factor", 4.0)
            .kv_u32("qwen2.context_length", 131072)
            .write_to(&path);

        let kv = read_gguf_rope_kv(&path).unwrap();
        assert_eq!(kv.orig_ctx, None);
        match derive_rope_scaling(&kv) {
            RopeIngestOutcome::ManualConfigRequired(note) => {
                assert!(note.contains("original_context_length"));
                assert!(!note.to_lowercase().contains("guess"));
            }
            other => panic!("expected ManualConfigRequired, got {other:?}"),
        }
    }

    #[test]
    fn yarn_missing_factor_never_fabricates_a_factor() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        GgufBuilder::new()
            .kv_string("qwen2.rope.scaling.type", "yarn")
            .kv_u32("qwen2.rope.scaling.original_context_length", 32768)
            .kv_u32("qwen2.context_length", 131072)
            .write_to(&path);

        let kv = read_gguf_rope_kv(&path).unwrap();
        match derive_rope_scaling(&kv) {
            RopeIngestOutcome::ManualConfigRequired(note) => {
                assert!(note.contains("rope.scaling.factor"));
            }
            other => panic!("expected ManualConfigRequired, got {other:?}"),
        }
    }

    #[test]
    fn ntk_is_recorded_as_linear_not_forced_to_yarn() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        GgufBuilder::new()
            .kv_string("llama.rope.scaling.type", "ntk")
            .kv_f32("llama.rope.scaling.factor", 2.0)
            .kv_u32("llama.context_length", 32768)
            .write_to(&path);

        let kv = read_gguf_rope_kv(&path).unwrap();
        match derive_rope_scaling(&kv) {
            RopeIngestOutcome::Prefilled(rope) => {
                assert_eq!(rope.method, RopeScalingMethod::Linear);
                assert_eq!(rope.rope_scale, 2.0);
            }
            other => panic!("expected Prefilled(linear) for ntk, got {other:?}"),
        }
    }

    #[test]
    fn unrecognized_method_requires_manual_config() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        GgufBuilder::new()
            .kv_string("llama.rope.scaling.type", "bogus")
            .write_to(&path);
        let kv = read_gguf_rope_kv(&path).unwrap();
        match derive_rope_scaling(&kv) {
            RopeIngestOutcome::ManualConfigRequired(note) => assert!(note.contains("bogus")),
            other => panic!("expected ManualConfigRequired, got {other:?}"),
        }
    }

    #[test]
    fn override_kv_mismatch_still_prefills_but_warns() {
        // context_length does NOT exceed original_context_length — the
        // override-kv quirk. Ingestion still pre-fills from stated numbers
        // (it doesn't invent anything) but this path is exercised to prove it
        // doesn't panic or silently drop the block.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        GgufBuilder::new()
            .kv_string("qwen2.rope.scaling.type", "yarn")
            .kv_f32("qwen2.rope.scaling.factor", 4.0)
            .kv_u32("qwen2.rope.scaling.original_context_length", 32768)
            .kv_u32("qwen2.context_length", 32768)
            .write_to(&path);
        let kv = read_gguf_rope_kv(&path).unwrap();
        match derive_rope_scaling(&kv) {
            RopeIngestOutcome::Prefilled(rope) => {
                assert_eq!(rope.target_ctx, 32768);
                assert_eq!(rope.yarn_orig_ctx, 32768);
            }
            other => panic!("expected Prefilled despite the mismatch, got {other:?}"),
        }
    }

    #[test]
    fn non_gguf_file_yields_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("not-a-model.gguf");
        std::fs::write(&path, b"not a gguf file at all").unwrap();
        assert!(read_gguf_rope_kv(&path).is_none());
    }

    #[test]
    fn missing_file_yields_none() {
        let path = Path::new("/nonexistent/definitely-not-here.gguf");
        assert!(read_gguf_rope_kv(path).is_none());
    }

    #[test]
    fn attn_factor_override_from_gguf_is_honored() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        GgufBuilder::new()
            .kv_string("qwen2.rope.scaling.type", "yarn")
            .kv_f32("qwen2.rope.scaling.factor", 4.0)
            .kv_u32("qwen2.rope.scaling.original_context_length", 32768)
            .kv_u32("qwen2.context_length", 131072)
            .kv_f32("qwen2.rope.scaling.attn_factor", 0.5)
            .write_to(&path);
        let kv = read_gguf_rope_kv(&path).unwrap();
        match derive_rope_scaling(&kv) {
            RopeIngestOutcome::Prefilled(rope) => assert_eq!(rope.attn_factor, 0.5),
            other => panic!("expected Prefilled, got {other:?}"),
        }
    }
}
