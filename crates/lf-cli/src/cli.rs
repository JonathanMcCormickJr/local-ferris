use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    name = "local-ferris",
    version,
    about = "Local, CPU-only Rust-first AI pair programmer",
    long_about = None,
)]
pub struct Cli {
    /// Path to a config file (defaults to ~/.config/local-ferris/config.toml).
    #[arg(long, global = true, value_name = "PATH")]
    pub config: Option<PathBuf>,

    /// Log verbosity: error, warn, info, debug, trace.
    #[arg(long, global = true, default_value = "info", value_name = "LEVEL")]
    pub log_level: String,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Interactive chat with the local model.
    Chat(ChatArgs),
    /// One-shot completion (stdin → stdout).
    Complete(CompleteArgs),
    /// Build or refresh the RAG index over a workspace.
    Index(IndexArgs),
    /// Fetch pinned, checksum-verified model weights.
    Download(DownloadArgs),
    /// Measure tokens-per-second across one or more GGUF files.
    Bench(BenchArgs),
}

#[derive(clap::Args, Debug)]
pub struct ChatArgs {
    /// GGUF model path or alias from the manifest.
    #[arg(long, value_name = "PATH_OR_ALIAS")]
    pub model: Option<String>,

    /// Inference threads (default: physical cores). Sets rayon's global
    /// thread pool, which `llama_gguf`'s CPU backend dispatches through.
    #[arg(long, value_name = "N")]
    pub threads: Option<usize>,

    /// Context window in tokens. Capped at 8192 per the reference-hardware
    /// budget.
    #[arg(
        long,
        default_value_t = 4096,
        value_name = "N",
        value_parser = clap::value_parser!(u32).range(1..=8192),
    )]
    pub ctx: u32,

    /// Prompt template. `auto` inspects the GGUF `general.architecture`
    /// header and picks chatml / llama3 / gemma / phi3.
    #[arg(long, default_value = "auto", value_name = "NAME",
          value_parser = ["auto", "chatml", "llama3", "gemma", "phi3"])]
    pub template: String,

    /// Optional system prompt.
    #[arg(long, value_name = "TEXT")]
    pub system: Option<String>,

    /// Maximum new tokens to emit.
    #[arg(long, default_value_t = 512, value_name = "N")]
    pub max_tokens: usize,

    /// Sampling temperature.
    #[arg(long, default_value_t = 0.7)]
    pub temperature: f32,
}

#[derive(clap::Args, Debug)]
pub struct CompleteArgs {
    /// GGUF model path or alias from the manifest.
    #[arg(long, value_name = "PATH_OR_ALIAS")]
    pub model: Option<String>,

    /// Inference threads (default: physical cores). Sets rayon's global
    /// thread pool, which `llama_gguf`'s CPU backend dispatches through.
    #[arg(long, value_name = "N")]
    pub threads: Option<usize>,

    /// Context window in tokens. Capped at 8192 per the reference-hardware
    /// budget in README §"Target hardware baseline".
    #[arg(
        long,
        default_value_t = 4096,
        value_name = "N",
        value_parser = clap::value_parser!(u32).range(1..=8192),
    )]
    pub ctx: u32,

    /// Maximum new tokens to emit.
    #[arg(long, default_value_t = 512, value_name = "N")]
    pub max_tokens: usize,

    /// Sampling temperature.
    #[arg(long, default_value_t = 0.2)]
    pub temperature: f32,
}

#[derive(clap::Args, Debug)]
pub struct IndexArgs {
    /// Workspace root to index (defaults to the current directory).
    #[arg(long, value_name = "PATH")]
    pub workspace: Option<PathBuf>,

    /// Fetch current dependency docs from docs.rs / crates.io / lib.rs at
    /// index time. Requires network; off by default to honor the offline
    /// posture.
    #[arg(long)]
    pub auto_fetch_docs: bool,

    /// Rebuild from scratch instead of doing an incremental update.
    #[arg(long)]
    pub rebuild: bool,
}

#[derive(clap::Args, Debug)]
pub struct BenchArgs {
    /// One or more GGUF files to benchmark (typically sibling
    /// quantizations of the same base model).
    #[arg(value_name = "GGUF", required = true)]
    pub models: Vec<PathBuf>,

    /// Prompt text to drive generation. If neither this nor
    /// `--prompt-file` is provided, a short built-in Rust prompt is used.
    #[arg(long, value_name = "TEXT", conflicts_with = "prompt_file")]
    pub prompt: Option<String>,

    /// Path to a file whose contents are used as the prompt.
    #[arg(long, value_name = "PATH", conflicts_with = "prompt")]
    pub prompt_file: Option<PathBuf>,

    /// Tokens to generate per run.
    #[arg(long, default_value_t = 128, value_name = "N")]
    pub max_tokens: usize,

    /// Inference threads (sets rayon's global pool).
    #[arg(long, value_name = "N")]
    pub threads: Option<usize>,

    /// Context window in tokens.
    #[arg(
        long,
        default_value_t = 2048,
        value_name = "N",
        value_parser = clap::value_parser!(u32).range(1..=8192),
    )]
    pub ctx: u32,

    /// Runs per model. Each produces one row in the output table.
    #[arg(long, default_value_t = 1, value_name = "N",
          value_parser = clap::value_parser!(u32).range(1..=100))]
    pub runs: u32,
}

#[derive(clap::Args, Debug)]
pub struct DownloadArgs {
    /// Manifest entry or alias to download (e.g. `starcoder2-3b.Q4_K_M`).
    #[arg(value_name = "ALIAS")]
    pub alias: String,

    /// Manifest file to read (defaults to manifests/models.toml).
    #[arg(long, value_name = "PATH")]
    pub manifest: Option<PathBuf>,

    /// Verify an already-downloaded artifact without refetching.
    #[arg(long)]
    pub verify_only: bool,
}

pub fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Chat(args) => crate::chat::run(args),
        Command::Complete(args) => crate::complete::run(args),
        Command::Index(_) => not_implemented("index"),
        Command::Download(args) => crate::download::run(args),
        Command::Bench(args) => crate::bench::run(args),
    }
}

fn not_implemented(name: &str) -> Result<()> {
    bail!("`{name}` is not yet implemented — scaffold only (see README roadmap)")
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn parses_chat_subcommand() {
        let cli = Cli::try_parse_from(["local-ferris", "chat", "--threads", "4"]).unwrap();
        match cli.command {
            Command::Chat(args) => assert_eq!(args.threads, Some(4)),
            _ => panic!("expected Chat"),
        }
    }

    #[test]
    fn parses_index_auto_fetch_docs_flag() {
        let cli = Cli::try_parse_from(["local-ferris", "index", "--auto-fetch-docs"]).unwrap();
        match cli.command {
            Command::Index(args) => assert!(args.auto_fetch_docs),
            _ => panic!("expected Index"),
        }
    }

    #[test]
    fn download_requires_alias() {
        let result = Cli::try_parse_from(["local-ferris", "download"]);
        assert!(result.is_err());
    }

    #[test]
    fn complete_ctx_default_is_4096() {
        let cli = Cli::try_parse_from(["local-ferris", "complete", "--model", "x.gguf"]).unwrap();
        match cli.command {
            Command::Complete(args) => assert_eq!(args.ctx, 4096),
            _ => panic!("expected Complete"),
        }
    }

    #[test]
    fn complete_ctx_accepts_within_range() {
        let cli = Cli::try_parse_from([
            "local-ferris",
            "complete",
            "--model",
            "x.gguf",
            "--ctx",
            "2048",
        ])
        .unwrap();
        match cli.command {
            Command::Complete(args) => assert_eq!(args.ctx, 2048),
            _ => panic!("expected Complete"),
        }
    }

    #[test]
    fn complete_ctx_rejects_above_ceiling() {
        let result = Cli::try_parse_from([
            "local-ferris",
            "complete",
            "--model",
            "x.gguf",
            "--ctx",
            "16384",
        ]);
        assert!(result.is_err(), "--ctx 16384 should be rejected");
    }

    #[test]
    fn chat_template_defaults_to_auto() {
        let cli = Cli::try_parse_from(["local-ferris", "chat", "--model", "x.gguf"]).unwrap();
        match cli.command {
            Command::Chat(args) => assert_eq!(args.template, "auto"),
            _ => panic!("expected Chat"),
        }
    }

    #[test]
    fn chat_template_rejects_unknown_name() {
        let result = Cli::try_parse_from([
            "local-ferris",
            "chat",
            "--model",
            "x.gguf",
            "--template",
            "mistral-instruct",
        ]);
        assert!(result.is_err(), "unknown template name should be rejected");
    }

    #[test]
    fn chat_accepts_system_prompt_and_all_named_templates() {
        for t in ["chatml", "llama3", "gemma", "phi3"] {
            let cli = Cli::try_parse_from([
                "local-ferris",
                "chat",
                "--model",
                "x.gguf",
                "--template",
                t,
                "--system",
                "You are Ferris.",
            ])
            .unwrap();
            match cli.command {
                Command::Chat(args) => {
                    assert_eq!(args.template, t);
                    assert_eq!(args.system.as_deref(), Some("You are Ferris."));
                }
                _ => panic!("expected Chat"),
            }
        }
    }

    #[test]
    fn bench_requires_at_least_one_model() {
        let result = Cli::try_parse_from(["local-ferris", "bench"]);
        assert!(result.is_err());
    }

    #[test]
    fn bench_accepts_multiple_positional_models() {
        let cli = Cli::try_parse_from(["local-ferris", "bench", "/a.gguf", "/b.gguf", "/c.gguf"])
            .unwrap();
        match cli.command {
            Command::Bench(args) => {
                assert_eq!(args.models.len(), 3);
                assert_eq!(args.runs, 1);
                assert_eq!(args.max_tokens, 128);
                assert_eq!(args.ctx, 2048);
            }
            _ => panic!("expected Bench"),
        }
    }

    #[test]
    fn bench_prompt_and_prompt_file_are_mutually_exclusive() {
        let result = Cli::try_parse_from([
            "local-ferris",
            "bench",
            "/a.gguf",
            "--prompt",
            "hi",
            "--prompt-file",
            "/tmp/p.txt",
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn bench_runs_range_is_validated() {
        assert!(Cli::try_parse_from(["local-ferris", "bench", "/a.gguf", "--runs", "0"]).is_err());
        assert!(
            Cli::try_parse_from(["local-ferris", "bench", "/a.gguf", "--runs", "500"]).is_err()
        );
    }

    #[test]
    fn chat_ctx_rejects_above_ceiling() {
        let result = Cli::try_parse_from([
            "local-ferris",
            "chat",
            "--model",
            "x.gguf",
            "--ctx",
            "16384",
        ]);
        assert!(result.is_err(), "chat --ctx 16384 should be rejected");
    }

    #[test]
    fn complete_ctx_rejects_zero() {
        let result = Cli::try_parse_from([
            "local-ferris",
            "complete",
            "--model",
            "x.gguf",
            "--ctx",
            "0",
        ]);
        assert!(result.is_err(), "--ctx 0 should be rejected");
    }
}
