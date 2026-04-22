//! GGUF-to-GGUF quantization via `llama_gguf::gguf::quantize_model`.
//!
//! This is offline pipeline tooling (a separate concern from the `Engine`
//! runtime) but it lives in `lf-inference` because it wraps the same
//! upstream crate and doesn't earn its own workspace member yet.
//!
//! **Vocabulary note.** llama.cpp popularized labels like `Q4_K_M` that
//! denote a *mixture*: attention tensors at Q6_K, FFN tensors at Q4_K,
//! etc. `llama_gguf 0.14` exposes uniform k-quant variants only —
//! every tensor gets the same type. We accept the llama.cpp-style `_M`
//! suffix on input for familiarity, but strip it and apply the uniform
//! type. Mixed-precision quantization is a future enhancement.

use std::path::{Path, PathBuf};
use std::time::Instant;

use llama_gguf::gguf::{GgmlType, QuantizeOptions, quantize_model};
use serde::Serialize;

/// Which quantization type to apply. A subset of `llama_gguf`'s
/// `GgmlType` — the ones that make sense to expose as CLI-facing
/// targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizeTarget {
    Q4K,
    Q5K,
    Q6K,
    Q8_0,
}

#[derive(Debug, thiserror::Error)]
#[error("unknown quantization target: {0:?} (known: Q4_K, Q5_K, Q6_K, Q8_0)")]
pub struct ParseTargetError(pub String);

impl QuantizeTarget {
    /// Parse a CLI-friendly string. Case-insensitive. The llama.cpp
    /// `_M` suffix (e.g., `Q4_K_M`) is stripped and treated as the
    /// same uniform k-quant — see module docstring.
    pub fn parse(s: &str) -> Result<Self, ParseTargetError> {
        let upper = s.trim().to_ascii_uppercase();
        // Strip llama.cpp mixed-precision suffixes we don't implement.
        let normalized = upper
            .strip_suffix("_M")
            .or_else(|| upper.strip_suffix("-M"))
            .or_else(|| upper.strip_suffix("_S"))
            .or_else(|| upper.strip_suffix("-S"))
            .unwrap_or(&upper);
        match normalized {
            "Q4_K" | "Q4K" => Ok(Self::Q4K),
            "Q5_K" | "Q5K" => Ok(Self::Q5K),
            "Q6_K" | "Q6K" => Ok(Self::Q6K),
            "Q8_0" | "Q8" => Ok(Self::Q8_0),
            _ => Err(ParseTargetError(s.to_owned())),
        }
    }

    /// Canonical label for on-disk filenames and reports.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Q8_0 => "Q8_0",
        }
    }

    fn to_ggml_type(self) -> GgmlType {
        match self {
            Self::Q4K => GgmlType::Q4K,
            Self::Q5K => GgmlType::Q5K,
            Self::Q6K => GgmlType::Q6K,
            Self::Q8_0 => GgmlType::Q8_0,
        }
    }
}

/// Given an input GGUF path and a target, produce a sensible output
/// filename alongside it (`model.gguf` + `Q4_K` → `model.Q4_K.gguf`).
pub fn derive_output_name(input: &Path, target: QuantizeTarget) -> PathBuf {
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    PathBuf::from(format!("{stem}.{}.gguf", target.label()))
}

/// Result of a single quantization pass. `Serialize` so the CLI can
/// emit a TOML report.
#[derive(Debug, Clone, Serialize)]
pub struct QuantizeReport {
    pub target: String,
    pub input_path: PathBuf,
    pub output_path: PathBuf,
    pub input_bytes: u64,
    pub output_bytes: u64,
    /// `output_bytes / input_bytes`. Lower = better compression.
    pub reduction_ratio: f64,
    pub tensors_total: usize,
    pub tensors_quantized: usize,
    pub tensors_skipped: usize,
    pub duration_secs: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum QuantizeError {
    #[error("parsing target: {0}")]
    ParseTarget(#[from] ParseTargetError),
    #[error("llama_gguf quantize: {0}")]
    Gguf(String),
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
}

/// Apply `target` to `input`, writing the quantized GGUF to `output`.
/// `threads` is passed through to `QuantizeOptions::threads`; pass 0 to
/// let `llama_gguf` pick a default (currently 4).
pub fn quantize(
    input: &Path,
    output: &Path,
    target: QuantizeTarget,
    threads: usize,
) -> Result<QuantizeReport, QuantizeError> {
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let options = QuantizeOptions {
        target_type: target.to_ggml_type(),
        threads: if threads == 0 {
            QuantizeOptions::default().threads
        } else {
            threads
        },
        ..Default::default()
    };

    let start = Instant::now();
    let stats = quantize_model(input, output, &options, None)
        .map_err(|e| QuantizeError::Gguf(format!("{e:?}")))?;
    let duration = start.elapsed();

    let reduction = if stats.bytes_original == 0 {
        0.0
    } else {
        stats.bytes_quantized as f64 / stats.bytes_original as f64
    };

    Ok(QuantizeReport {
        target: target.label().to_string(),
        input_path: input.to_path_buf(),
        output_path: output.to_path_buf(),
        input_bytes: stats.bytes_original as u64,
        output_bytes: stats.bytes_quantized as u64,
        reduction_ratio: reduction,
        tensors_total: stats.tensors_total,
        tensors_quantized: stats.tensors_quantized,
        tensors_skipped: stats.tensors_skipped,
        duration_secs: duration.as_secs_f64(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_accepts_every_canonical_name() {
        for (input, expected) in [
            ("Q4_K", QuantizeTarget::Q4K),
            ("Q5_K", QuantizeTarget::Q5K),
            ("Q6_K", QuantizeTarget::Q6K),
            ("Q8_0", QuantizeTarget::Q8_0),
        ] {
            assert_eq!(QuantizeTarget::parse(input).unwrap(), expected);
        }
    }

    #[test]
    fn parse_is_case_insensitive() {
        assert_eq!(QuantizeTarget::parse("q4_k").unwrap(), QuantizeTarget::Q4K);
        assert_eq!(QuantizeTarget::parse("q8_0").unwrap(), QuantizeTarget::Q8_0);
    }

    #[test]
    fn parse_strips_llama_cpp_m_suffix() {
        assert_eq!(
            QuantizeTarget::parse("Q4_K_M").unwrap(),
            QuantizeTarget::Q4K
        );
        assert_eq!(
            QuantizeTarget::parse("Q5_K_M").unwrap(),
            QuantizeTarget::Q5K
        );
    }

    #[test]
    fn parse_strips_s_suffix_too() {
        assert_eq!(
            QuantizeTarget::parse("Q4_K_S").unwrap(),
            QuantizeTarget::Q4K
        );
    }

    #[test]
    fn parse_rejects_unknown() {
        let err = QuantizeTarget::parse("IQ2_XXS").unwrap_err();
        assert!(err.0.contains("IQ2_XXS"));
    }

    #[test]
    fn parse_accepts_compact_forms() {
        assert_eq!(QuantizeTarget::parse("Q4K").unwrap(), QuantizeTarget::Q4K);
        assert_eq!(QuantizeTarget::parse("q8").unwrap(), QuantizeTarget::Q8_0);
    }

    #[test]
    fn label_roundtrips_through_parse() {
        for t in [
            QuantizeTarget::Q4K,
            QuantizeTarget::Q5K,
            QuantizeTarget::Q6K,
            QuantizeTarget::Q8_0,
        ] {
            assert_eq!(QuantizeTarget::parse(t.label()).unwrap(), t);
        }
    }

    #[test]
    fn derive_output_name_inserts_target_label() {
        let p = derive_output_name(Path::new("/tmp/starcoder2-3b.gguf"), QuantizeTarget::Q4K);
        assert_eq!(p, PathBuf::from("starcoder2-3b.Q4_K.gguf"));
    }

    #[test]
    fn derive_output_name_handles_pathless_input() {
        let p = derive_output_name(Path::new("raw.gguf"), QuantizeTarget::Q8_0);
        assert_eq!(p, PathBuf::from("raw.Q8_0.gguf"));
    }

    #[test]
    fn report_serializes_to_toml() {
        let r = QuantizeReport {
            target: "Q4_K".into(),
            input_path: PathBuf::from("/tmp/in.gguf"),
            output_path: PathBuf::from("/tmp/out.Q4_K.gguf"),
            input_bytes: 6_000_000_000,
            output_bytes: 1_900_000_000,
            reduction_ratio: 0.316666,
            tensors_total: 224,
            tensors_quantized: 200,
            tensors_skipped: 24,
            duration_secs: 45.2,
        };
        let toml_text = toml::to_string(&r).unwrap();
        assert!(toml_text.contains("target = \"Q4_K\""));
        assert!(toml_text.contains("input_bytes = 6000000000"));
        assert!(toml_text.contains("reduction_ratio = 0.316666"));
    }
}
