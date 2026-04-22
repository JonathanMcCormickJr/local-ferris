//! `local-ferris complete` — read a prompt from stdin, stream the
//! completion to stdout, honor Ctrl-C.

use std::io::{self, Read, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, bail};
use lf_inference::{Engine, EngineConfig, GenerateOutcome};

use crate::cli::CompleteArgs;

pub fn run(args: CompleteArgs) -> Result<()> {
    let model_path = args
        .model
        .ok_or_else(|| anyhow::anyhow!("--model is required (path to a GGUF file)"))?;

    let mut prompt = String::new();
    io::stdin()
        .read_to_string(&mut prompt)
        .context("failed to read prompt from stdin")?;
    if prompt.is_empty() {
        bail!("no prompt on stdin; pipe one in or redirect from a file");
    }

    if let Some(n) = args.threads {
        match lf_inference::init_threads(n) {
            Ok(actual) => eprintln!("[using {actual} inference threads]"),
            Err(e) => bail!(e),
        }
    }

    let cancel = Arc::new(AtomicBool::new(false));
    install_ctrlc_handler(Arc::clone(&cancel))?;

    let config = EngineConfig {
        model_path,
        temperature: args.temperature,
        max_tokens: args.max_tokens,
        max_context_len: Some(args.ctx as usize),
        use_gpu: false,
        ..Default::default()
    };

    let engine = Engine::load(config).context("failed to load model")?;

    let stdout = io::stdout();
    let mut out = stdout.lock();

    let outcome = engine
        .generate_streaming(&prompt, args.max_tokens, &cancel, |chunk| {
            let _ = out.write_all(chunk.as_bytes());
            let _ = out.flush();
        })
        .context("generation failed")?;

    writeln!(out)?;
    match outcome {
        GenerateOutcome::Finished { tokens } => {
            eprintln!("[generated {tokens} tokens]");
        }
        GenerateOutcome::Cancelled { tokens } => {
            eprintln!("[cancelled after {tokens} tokens]");
        }
    }
    Ok(())
}

fn install_ctrlc_handler(cancel: Arc<AtomicBool>) -> Result<()> {
    ctrlc::set_handler(move || {
        cancel.store(true, Ordering::SeqCst);
    })
    .context("failed to install Ctrl-C handler")
}
