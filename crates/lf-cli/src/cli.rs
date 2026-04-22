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
}

#[derive(clap::Args, Debug)]
pub struct ChatArgs {
    /// GGUF model path or alias from the manifest.
    #[arg(long, value_name = "PATH_OR_ALIAS")]
    pub model: Option<String>,

    /// Inference threads (default: physical cores).
    #[arg(long, value_name = "N")]
    pub threads: Option<usize>,

    /// Context window in tokens.
    #[arg(long, default_value_t = 4096, value_name = "N")]
    pub ctx: usize,
}

#[derive(clap::Args, Debug)]
pub struct CompleteArgs {
    /// GGUF model path or alias from the manifest.
    #[arg(long, value_name = "PATH_OR_ALIAS")]
    pub model: Option<String>,

    /// Inference threads (default: physical cores).
    #[arg(long, value_name = "N")]
    pub threads: Option<usize>,

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
        Command::Chat(_) => not_implemented("chat"),
        Command::Complete(_) => not_implemented("complete"),
        Command::Index(_) => not_implemented("index"),
        Command::Download(_) => not_implemented("download"),
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
}
