//! `local-ferris quantize` — produce one or more quantized GGUFs from
//! a raw (F16/F32) input and optionally emit a TOML size/quality report.

use std::fs;

use anyhow::{Context, Result};
use lf_inference::quantize::{QuantizeReport, QuantizeTarget, derive_output_name, quantize};
use serde::Serialize;

use crate::cli::QuantizeArgs;

pub fn run(args: QuantizeArgs) -> Result<()> {
    if !args.input.exists() {
        anyhow::bail!("input GGUF does not exist: {}", args.input.display());
    }
    fs::create_dir_all(&args.out_dir)
        .with_context(|| format!("creating output dir {}", args.out_dir.display()))?;

    let targets: Vec<QuantizeTarget> = args
        .targets
        .iter()
        .map(|s| QuantizeTarget::parse(s).with_context(|| format!("parsing --targets entry `{s}`")))
        .collect::<Result<Vec<_>>>()?;
    if targets.is_empty() {
        anyhow::bail!("no quantization targets provided");
    }

    let mut reports: Vec<QuantizeReport> = Vec::with_capacity(targets.len());
    for target in &targets {
        let out_name = derive_output_name(&args.input, *target);
        let out_path = args.out_dir.join(&out_name);
        eprintln!(
            "[quantize] {} → {} ({})",
            args.input.display(),
            out_path.display(),
            target.label()
        );
        let report = quantize(&args.input, &out_path, *target, args.threads)
            .with_context(|| format!("quantizing to {}", target.label()))?;
        println!(
            "  {:<6}  {:>10.1} MiB → {:>10.1} MiB  ({:>5.1}%)  {:>5} tensors  {:>6.1} s",
            report.target,
            bytes_to_mib(report.input_bytes),
            bytes_to_mib(report.output_bytes),
            report.reduction_ratio * 100.0,
            report.tensors_quantized,
            report.duration_secs,
        );
        reports.push(report);
    }

    if let Some(report_path) = &args.report {
        let doc = ReportDoc {
            version: 1,
            quantize: reports,
        };
        let text = toml::to_string_pretty(&doc).context("serializing report")?;
        fs::write(report_path, text)
            .with_context(|| format!("writing report to {}", report_path.display()))?;
        println!("report: {}", report_path.display());
    }

    Ok(())
}

#[derive(Debug, Clone, Serialize)]
struct ReportDoc {
    version: u32,
    quantize: Vec<QuantizeReport>,
}

fn bytes_to_mib(n: u64) -> f64 {
    (n as f64) / (1024.0 * 1024.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn report_doc_roundtrips_through_toml() {
        let doc = ReportDoc {
            version: 1,
            quantize: vec![QuantizeReport {
                target: "Q4_K".into(),
                input_path: PathBuf::from("/tmp/in.gguf"),
                output_path: PathBuf::from("/tmp/out.Q4_K.gguf"),
                input_bytes: 6_000_000_000,
                output_bytes: 1_900_000_000,
                reduction_ratio: 0.3166666,
                tensors_total: 224,
                tensors_quantized: 200,
                tensors_skipped: 24,
                duration_secs: 45.2,
            }],
        };
        let s = toml::to_string_pretty(&doc).unwrap();
        assert!(s.contains("version = 1"));
        assert!(s.contains("[[quantize]]"));
        assert!(s.contains("target = \"Q4_K\""));
    }

    #[test]
    fn bytes_to_mib_is_sane() {
        assert_eq!(bytes_to_mib(0), 0.0);
        let mib = bytes_to_mib(1024 * 1024);
        assert!((mib - 1.0).abs() < 1e-9);
    }
}
