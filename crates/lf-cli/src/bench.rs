//! `local-ferris bench` — tokens-per-second harness across one or more
//! GGUF files. Typical use: point it at sibling quantizations of the
//! same base model (Q4_K_M, Q5_K_M, Q8_0) to see the size/speed trade.
//!
//! Methodology is deliberately simple: load each model, run
//! `generate_streaming` for a fixed `--max-tokens` budget, count tokens
//! as they arrive, divide by wall-clock. No warmup window, no
//! statistical aggregation — just raw rows. Good enough to spot order-of-
//! magnitude regressions; rigorous benchmarking is a future concern.

use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use lf_inference::{Engine, EngineConfig, GenerateOutcome};

use crate::cli::BenchArgs;

/// Default prompt used when the user doesn't supply one. Raw text (no
/// chat template) because we're measuring throughput, not output
/// quality — content is irrelevant as long as the model generates
/// tokens at its natural rate.
const DEFAULT_PROMPT: &str = "Write a short Rust function that returns the Fibonacci number at index `n` using iteration:\n\n```rust\n";

#[derive(Debug, Clone)]
struct BenchRow {
    label: String,
    run: u32,
    load_ms: u128,
    tokens: usize,
    elapsed: Duration,
    cancelled: bool,
}

impl BenchRow {
    fn tokens_per_sec(&self) -> f64 {
        let secs = self.elapsed.as_secs_f64();
        if secs <= 0.0 {
            0.0
        } else {
            self.tokens as f64 / secs
        }
    }
}

pub fn run(args: BenchArgs) -> Result<()> {
    if let Some(n) = args.threads {
        match lf_inference::init_threads(n) {
            Ok(actual) => eprintln!("[using {actual} inference threads]"),
            Err(e) => bail!(e),
        }
    }

    let prompt = resolve_prompt(args.prompt.as_deref(), args.prompt_file.as_deref())?;

    let cancel = Arc::new(AtomicBool::new(false));
    install_ctrlc_handler(Arc::clone(&cancel))?;

    let mut rows: Vec<BenchRow> = Vec::new();

    'outer: for model_path in &args.models {
        let label = label_from_path(model_path);
        let model_path_str = model_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("model path is not UTF-8: {}", model_path.display()))?
            .to_owned();

        eprintln!("[bench {label}] loading {}", model_path.display());
        let t_load = Instant::now();
        let config = EngineConfig {
            model_path: model_path_str,
            max_tokens: args.max_tokens,
            max_context_len: Some(args.ctx as usize),
            use_gpu: false,
            ..Default::default()
        };
        let engine = Engine::load(config)
            .with_context(|| format!("failed to load {}", model_path.display()))?;
        let load_ms = t_load.elapsed().as_millis();
        eprintln!("[bench {label}] loaded in {load_ms} ms");

        for run_idx in 1..=args.runs {
            if cancel.load(Ordering::SeqCst) {
                eprintln!("[cancelled before run {run_idx}]");
                break 'outer;
            }
            eprintln!(
                "[bench {label}] run {run_idx}/{} generating {} tokens…",
                args.runs, args.max_tokens
            );
            let mut tokens = 0usize;
            let t_gen = Instant::now();
            let outcome = engine
                .generate_streaming(&prompt, args.max_tokens, &cancel, |_| tokens += 1)
                .with_context(|| format!("generation failed for {label} run {run_idx}"))?;
            let elapsed = t_gen.elapsed();
            rows.push(BenchRow {
                label: label.clone(),
                run: run_idx,
                load_ms,
                tokens,
                elapsed,
                cancelled: outcome.was_cancelled(),
            });
            if matches!(outcome, GenerateOutcome::Cancelled { .. }) {
                break 'outer;
            }
        }
    }

    if rows.is_empty() {
        bail!("no benchmark rows produced (all runs cancelled or failed)");
    }

    println!();
    println!("{}", format_table(&rows));
    Ok(())
}

fn resolve_prompt(inline: Option<&str>, file: Option<&Path>) -> Result<String> {
    if let Some(text) = inline {
        return Ok(text.to_owned());
    }
    if let Some(path) = file {
        return fs::read_to_string(path)
            .with_context(|| format!("failed to read prompt from {}", path.display()));
    }
    Ok(DEFAULT_PROMPT.to_owned())
}

/// Extract a short human-readable label from a model path. Drops the
/// directory and the `.gguf` extension, leaving the filename stem.
fn label_from_path(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.to_owned())
        .unwrap_or_else(|| path.display().to_string())
}

fn format_table(rows: &[BenchRow]) -> String {
    use std::fmt::Write;

    const HEADERS: [&str; 6] = ["label", "run", "load_ms", "tokens", "elapsed_ms", "tok/s"];

    let cells: Vec<[String; 6]> = rows
        .iter()
        .map(|r| {
            [
                if r.cancelled {
                    format!("{} (cancelled)", r.label)
                } else {
                    r.label.clone()
                },
                r.run.to_string(),
                r.load_ms.to_string(),
                r.tokens.to_string(),
                r.elapsed.as_millis().to_string(),
                format!("{:.2}", r.tokens_per_sec()),
            ]
        })
        .collect();

    let mut widths = [0usize; 6];
    for (i, h) in HEADERS.iter().enumerate() {
        widths[i] = h.len();
    }
    for row in &cells {
        for (i, cell) in row.iter().enumerate() {
            widths[i] = widths[i].max(cell.len());
        }
    }

    let mut out = String::new();
    let _ = writeln!(
        out,
        " {}",
        HEADERS
            .iter()
            .enumerate()
            .map(|(i, h)| format!("{:<width$}", h, width = widths[i]))
            .collect::<Vec<_>>()
            .join("  ")
    );
    let _ = writeln!(
        out,
        " {}",
        widths
            .iter()
            .map(|w| "─".repeat(*w))
            .collect::<Vec<_>>()
            .join("  ")
    );
    for row in &cells {
        let _ = writeln!(
            out,
            " {}",
            row.iter()
                .enumerate()
                .map(|(i, c)| format!("{:<width$}", c, width = widths[i]))
                .collect::<Vec<_>>()
                .join("  ")
        );
    }
    out
}

fn install_ctrlc_handler(cancel: Arc<AtomicBool>) -> Result<()> {
    ctrlc::set_handler(move || {
        cancel.store(true, Ordering::SeqCst);
    })
    .context("failed to install Ctrl-C handler")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_from_path_strips_extension_and_dir() {
        assert_eq!(
            label_from_path(Path::new("/models/starcoder2-3b.Q4_K_M.gguf")),
            "starcoder2-3b.Q4_K_M"
        );
        assert_eq!(
            label_from_path(Path::new("./phi-3.5-mini.Q5_K_M.gguf")),
            "phi-3.5-mini.Q5_K_M"
        );
    }

    #[test]
    fn label_from_path_falls_back_to_display_when_no_stem() {
        // Path with no file component — degenerate but shouldn't panic.
        assert_eq!(label_from_path(Path::new("/")), "/");
    }

    #[test]
    fn tokens_per_sec_handles_zero_elapsed() {
        let row = BenchRow {
            label: "x".into(),
            run: 1,
            load_ms: 0,
            tokens: 42,
            elapsed: Duration::ZERO,
            cancelled: false,
        };
        assert_eq!(row.tokens_per_sec(), 0.0);
    }

    #[test]
    fn tokens_per_sec_basic_arithmetic() {
        let row = BenchRow {
            label: "q4".into(),
            run: 1,
            load_ms: 1000,
            tokens: 100,
            elapsed: Duration::from_secs(25),
            cancelled: false,
        };
        assert!((row.tokens_per_sec() - 4.0).abs() < 1e-9);
    }

    #[test]
    fn format_table_includes_header_and_all_rows() {
        // Round-number durations so the formatted tok/s is unambiguous
        // under `{:.2}` rounding.
        let rows = vec![
            BenchRow {
                label: "starcoder2-3b.Q4_K_M".into(),
                run: 1,
                load_ms: 2340,
                tokens: 100,
                elapsed: Duration::from_secs(25),
                cancelled: false,
            },
            BenchRow {
                label: "starcoder2-3b.Q8_0".into(),
                run: 1,
                load_ms: 5610,
                tokens: 50,
                elapsed: Duration::from_secs(25),
                cancelled: false,
            },
        ];
        let out = format_table(&rows);
        assert!(out.contains("label"));
        assert!(out.contains("tok/s"));
        assert!(out.contains("starcoder2-3b.Q4_K_M"));
        assert!(out.contains("starcoder2-3b.Q8_0"));
        assert!(out.contains("4.00"), "output:\n{out}"); // 100 / 25
        assert!(out.contains("2.00"), "output:\n{out}"); // 50 / 25
    }

    #[test]
    fn format_table_marks_cancelled_rows() {
        let rows = vec![BenchRow {
            label: "partial".into(),
            run: 1,
            load_ms: 100,
            tokens: 7,
            elapsed: Duration::from_secs(1),
            cancelled: true,
        }];
        let out = format_table(&rows);
        assert!(out.contains("partial (cancelled)"));
    }
}
