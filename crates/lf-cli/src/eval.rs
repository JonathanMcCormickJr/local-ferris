//! `local-ferris eval` — task-completion eval suite for calibrating a
//! quantized model against a held-out Rust corpus.
//!
//! This is the "quality" half of the size/quality report that pairs
//! with `local-ferris quantize`. Each case in the suite is a raw
//! completion prompt (no chat templating) plus a short expected
//! substring the generated output should contain. The harness runs
//! every case, records per-case pass/fail + output preview, and emits
//! a TOML report.
//!
//! **Perplexity note.** Real perplexity computation would require
//! driving per-token forward passes and reading logits, which
//! `llama_gguf 0.14` does not expose at the Engine level. That's
//! genuine follow-up work; substring task-completion carries the
//! quality signal for the size/quality report today. See README §3.

use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::{Context, Result};
use lf_inference::{Engine, EngineConfig};
use serde::{Deserialize, Serialize};

use crate::cli::EvalArgs;

const MAX_PREVIEW_CHARS: usize = 240;

#[derive(Debug, Clone, Deserialize)]
pub struct Suite {
    #[serde(default)]
    pub version: u32,
    #[serde(default, rename = "case")]
    pub cases: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Case {
    pub name: String,
    pub prompt: String,
    pub expect: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
}

fn default_max_tokens() -> usize {
    128
}

#[derive(Debug, Clone, Serialize)]
pub struct CaseResult {
    pub name: String,
    pub passed: bool,
    pub generated_tokens: usize,
    pub duration_secs: f64,
    pub expected_substring: String,
    /// Truncated to [`MAX_PREVIEW_CHARS`] to keep reports readable.
    pub output_preview: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct EvalReport {
    pub version: u32,
    pub model: String,
    pub suite: String,
    pub total: usize,
    pub passed: usize,
    pub pass_rate: f64,
    pub duration_secs: f64,
    #[serde(rename = "case")]
    pub cases: Vec<CaseResult>,
}

pub fn run(args: EvalArgs) -> Result<()> {
    let suite = load_suite(&args.suite)
        .with_context(|| format!("loading suite from {}", args.suite.display()))?;
    if suite.version != 0 && suite.version != 1 {
        eprintln!(
            "[warn] suite version {} is newer than this harness understands (1); \
             proceeding but some fields may be ignored",
            suite.version
        );
    }
    if suite.cases.is_empty() {
        anyhow::bail!("suite {} contains no cases", args.suite.display());
    }

    if !Path::new(&args.model).exists() {
        anyhow::bail!("model GGUF does not exist: {}", args.model);
    }

    if let Some(n) = args.threads {
        match lf_inference::init_threads(n) {
            Ok(actual) => eprintln!("[using {actual} inference threads]"),
            Err(e) => anyhow::bail!(e),
        }
    }

    let cancel = Arc::new(AtomicBool::new(false));
    {
        let cancel = Arc::clone(&cancel);
        ctrlc::set_handler(move || cancel.store(true, Ordering::SeqCst))
            .context("installing Ctrl-C handler")?;
    }

    let config = EngineConfig {
        model_path: args.model.clone(),
        temperature: args.temperature,
        max_context_len: Some(args.ctx as usize),
        use_gpu: false,
        ..Default::default()
    };
    let engine = Engine::load(config).context("loading model")?;

    let mut case_results = Vec::with_capacity(suite.cases.len());
    let t_total = Instant::now();

    for (i, case) in suite.cases.iter().enumerate() {
        eprintln!(
            "[eval {}/{}] {} (max_tokens={})",
            i + 1,
            suite.cases.len(),
            case.name,
            case.max_tokens
        );
        if cancel.load(Ordering::SeqCst) {
            eprintln!("[cancelled before case `{}`]", case.name);
            break;
        }

        let mut output = String::new();
        let t_case = Instant::now();
        let outcome = engine
            .generate_streaming(&case.prompt, case.max_tokens, &cancel, |chunk| {
                output.push_str(chunk);
            })
            .with_context(|| format!("generating for case `{}`", case.name))?;
        let duration_secs = t_case.elapsed().as_secs_f64();

        let passed = matches_expected(&output, &case.expect);
        let truncated = truncate_preview(&output);

        println!(
            "  {:<6} {:<30} {:>3} tok  {:>6.2}s  {}",
            if passed { "PASS" } else { "FAIL" },
            case.name,
            outcome.tokens(),
            duration_secs,
            if passed {
                String::new()
            } else {
                format!("expected to contain {:?}", case.expect)
            },
        );

        case_results.push(CaseResult {
            name: case.name.clone(),
            passed,
            generated_tokens: outcome.tokens(),
            duration_secs,
            expected_substring: case.expect.clone(),
            output_preview: truncated,
        });
    }

    let total = case_results.len();
    let passed = case_results.iter().filter(|c| c.passed).count();
    let pass_rate = if total == 0 {
        0.0
    } else {
        passed as f64 / total as f64
    };

    println!();
    println!(
        "summary: {passed}/{total} passed ({:.1}%), {:.2}s total",
        pass_rate * 100.0,
        t_total.elapsed().as_secs_f64()
    );

    if let Some(report_path) = &args.report {
        let report = EvalReport {
            version: 1,
            model: args.model,
            suite: args.suite.display().to_string(),
            total,
            passed,
            pass_rate,
            duration_secs: t_total.elapsed().as_secs_f64(),
            cases: case_results,
        };
        let text = toml::to_string_pretty(&report).context("serializing report")?;
        fs::write(report_path, text)
            .with_context(|| format!("writing report to {}", report_path.display()))?;
        println!("report: {}", report_path.display());
    }

    Ok(())
}

fn load_suite(path: &Path) -> Result<Suite> {
    let text = fs::read_to_string(path)?;
    let suite: Suite = toml::from_str(&text)?;
    Ok(suite)
}

fn matches_expected(output: &str, expected: &str) -> bool {
    output.contains(expected)
}

fn truncate_preview(s: &str) -> String {
    if s.chars().count() <= MAX_PREVIEW_CHARS {
        s.to_owned()
    } else {
        let prefix: String = s.chars().take(MAX_PREVIEW_CHARS).collect();
        format!("{prefix}…")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suite_parses_two_cases() {
        let toml_text = r#"
version = 1

[[case]]
name = "fib"
prompt = "fn fib(n: u32) -> u64 {\n"
expect = "let mut"
max_tokens = 128

[[case]]
name = "vec-sum"
prompt = "fn sum(xs: &[i32]) -> i32 {\n"
expect = "iter"
"#;
        let suite: Suite = toml::from_str(toml_text).unwrap();
        assert_eq!(suite.version, 1);
        assert_eq!(suite.cases.len(), 2);
        assert_eq!(suite.cases[0].name, "fib");
        assert_eq!(suite.cases[0].max_tokens, 128);
        // default max_tokens fires when the field is omitted
        assert_eq!(suite.cases[1].max_tokens, 128);
    }

    #[test]
    fn suite_empty_is_valid() {
        let s: Suite = toml::from_str("").unwrap();
        assert!(s.cases.is_empty());
    }

    #[test]
    fn matches_expected_substring_hits() {
        assert!(matches_expected(
            "    let mut a = 0;\n    let mut b = 1;\n",
            "let mut"
        ));
    }

    #[test]
    fn matches_expected_substring_misses() {
        assert!(!matches_expected("xs.iter().sum()", "let mut"));
    }

    #[test]
    fn truncate_preview_is_noop_for_short_strings() {
        assert_eq!(truncate_preview("hello"), "hello");
    }

    #[test]
    fn truncate_preview_adds_ellipsis_when_over_limit() {
        let big: String = "a".repeat(MAX_PREVIEW_CHARS * 2);
        let out = truncate_preview(&big);
        assert!(out.ends_with('…'));
        assert_eq!(out.chars().count(), MAX_PREVIEW_CHARS + 1);
    }

    #[test]
    fn report_serializes_to_toml_with_expected_shape() {
        let r = EvalReport {
            version: 1,
            model: "./raw.Q4_K.gguf".into(),
            suite: "evals/rust_suite.toml".into(),
            total: 2,
            passed: 1,
            pass_rate: 0.5,
            duration_secs: 12.3,
            cases: vec![
                CaseResult {
                    name: "fib".into(),
                    passed: true,
                    generated_tokens: 42,
                    duration_secs: 6.0,
                    expected_substring: "let mut".into(),
                    output_preview: "let mut a = 0;".into(),
                },
                CaseResult {
                    name: "sum".into(),
                    passed: false,
                    generated_tokens: 50,
                    duration_secs: 6.3,
                    expected_substring: "iter".into(),
                    output_preview: "xs.into_iter()".into(),
                },
            ],
        };
        let s = toml::to_string_pretty(&r).unwrap();
        assert!(s.contains("version = 1"));
        assert!(s.contains("passed = 1"));
        assert!(s.contains("pass_rate = 0.5"));
        assert!(s.contains("[[case]]"));
        assert!(s.contains("name = \"fib\""));
        assert!(s.contains("name = \"sum\""));
    }
}
