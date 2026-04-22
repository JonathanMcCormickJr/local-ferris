//! `local-ferris chat` — single-turn chat driven by a named prompt
//! template. Multi-turn history is a future UX concern; this scaffolds
//! the template plumbing that Phase 1 requires.

use std::io::{self, Read, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, bail};
use lf_inference::prompt::{self, Message, PromptTemplate, Role};
use lf_inference::{Engine, EngineConfig, GenerateOutcome};

use crate::cli::ChatArgs;

pub fn run(args: ChatArgs) -> Result<()> {
    let model_path = args
        .model
        .ok_or_else(|| anyhow::anyhow!("--model is required (path to a GGUF file)"))?;

    let mut user_message = String::new();
    io::stdin()
        .read_to_string(&mut user_message)
        .context("failed to read user message from stdin")?;
    if user_message.trim().is_empty() {
        bail!("no user message on stdin; pipe one in or redirect from a file");
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

    let template = resolve_template(&args.template, &engine);
    eprintln!("[template: {}]", template.name());

    let mut messages: Vec<Message> = Vec::with_capacity(2);
    if let Some(sys) = args.system {
        messages.push(Message::new(Role::System, sys));
    }
    messages.push(Message::new(Role::User, user_message.trim_end()));

    let prompt = template.format(&messages, true);

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

fn resolve_template(name: &str, engine: &Engine) -> Box<dyn PromptTemplate> {
    if name == "auto" {
        let arch = engine
            .gguf()
            .and_then(|f| f.data.get_string("general.architecture"));
        match arch {
            Some(a) => {
                eprintln!("[template auto: detected architecture = {a}]");
                prompt::from_architecture(a)
            }
            None => {
                eprintln!("[template auto: architecture unknown, falling back to chatml]");
                prompt::from_architecture("")
            }
        }
    } else {
        // clap already validated the value against the allowed list, so
        // by_name must succeed here.
        prompt::by_name(name)
            .unwrap_or_else(|| panic!("clap let through an unexpected template name: {name}"))
    }
}

fn install_ctrlc_handler(cancel: Arc<AtomicBool>) -> Result<()> {
    ctrlc::set_handler(move || {
        cancel.store(true, Ordering::SeqCst);
    })
    .context("failed to install Ctrl-C handler")
}
