#![forbid(unsafe_code)]

//! Thin, testable wrapper around [`llama_gguf`]'s inference engine.
//!
//! The module intentionally does *not* try to hide `llama_gguf` behind a
//! sealed façade — it re-exports the upstream [`EngineConfig`] so callers
//! can configure sampling directly. The value added here is:
//!
//! 1. A cancellation-aware streaming call (`Engine::generate_streaming`)
//!    driven by an `AtomicBool`, which is what the CLI wires to Ctrl-C.
//! 2. A token-pump loop (`pump`) that is generic over the iterator type
//!    so it can be exercised in unit tests without a real GGUF file.

use std::sync::atomic::{AtomicBool, Ordering};

pub mod prompt;
pub mod quantize;

pub use llama_gguf::engine::EngineConfig;
pub use prompt::{Message, PromptTemplate, Role};

/// Error returned when the rayon global thread pool has already been
/// initialized with a different thread count than requested.
#[derive(Debug, thiserror::Error)]
#[error(
    "rayon global thread pool already initialized with {actual} threads; cannot switch to {requested}"
)]
pub struct ThreadInitMismatch {
    pub requested: usize,
    pub actual: usize,
}

/// Install `n` as the rayon global thread pool size.
///
/// `llama_gguf`'s CPU backend dispatches all parallelism through rayon, so
/// setting rayon's global pool is equivalent to setting `n_threads` for
/// every `Engine` created afterwards.
///
/// Idempotent when called repeatedly with the same count; returns
/// [`ThreadInitMismatch`] if the pool has already been configured with a
/// different count (rayon's global pool can only be set once per process).
pub fn init_threads(n: usize) -> Result<usize, ThreadInitMismatch> {
    match rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global()
    {
        Ok(()) => Ok(n),
        Err(_) => {
            let actual = rayon::current_num_threads();
            if actual == n {
                Ok(actual)
            } else {
                Err(ThreadInitMismatch {
                    requested: n,
                    actual,
                })
            }
        }
    }
}

/// Outcome of a streaming generation pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerateOutcome {
    /// Stream ran to completion (hit `max_tokens` or end-of-stream).
    Finished { tokens: usize },
    /// Cancellation was requested before the stream completed.
    Cancelled { tokens: usize },
}

impl GenerateOutcome {
    pub fn tokens(&self) -> usize {
        match self {
            Self::Finished { tokens } | Self::Cancelled { tokens } => *tokens,
        }
    }

    pub fn was_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled { .. })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    /// The underlying `llama_gguf` crate returned an error. The exact error
    /// type is deliberately collapsed to a string — `llama_gguf`'s error
    /// enum is a moving target and we do not want to leak it into the
    /// `lf-inference` public API.
    #[error("llama_gguf: {0}")]
    LlamaGguf(String),
}

/// Streaming inference engine.
pub struct Engine {
    inner: llama_gguf::engine::Engine,
}

impl Engine {
    /// Load a model according to the supplied configuration.
    pub fn load(config: EngineConfig) -> Result<Self, EngineError> {
        let inner = llama_gguf::engine::Engine::load(config)
            .map_err(|e| EngineError::LlamaGguf(format!("{e:?}")))?;
        Ok(Self { inner })
    }

    /// Access the underlying GGUF file if one is loaded. Returns `None`
    /// for non-GGUF models (e.g. ONNX). Used by the CLI's `--template auto`
    /// path to read `general.architecture` from the header.
    pub fn gguf(&self) -> Option<&llama_gguf::gguf::GgufFile> {
        self.inner.gguf()
    }

    /// Drive the streaming generation loop, invoking `on_token` for each
    /// decoded chunk. Checks `cancel` before consuming each item; returns
    /// [`GenerateOutcome::Cancelled`] as soon as the flag is observed set.
    pub fn generate_streaming<F>(
        &self,
        prompt: &str,
        max_tokens: usize,
        cancel: &AtomicBool,
        on_token: F,
    ) -> Result<GenerateOutcome, EngineError>
    where
        F: FnMut(&str),
    {
        let stream = self.inner.generate_streaming(prompt, max_tokens);
        pump(stream, cancel, on_token)
    }
}

/// Iterate `stream` while respecting `cancel`, delivering each successful
/// chunk to `on_token`.
///
/// Kept generic over the iterator so unit tests can substitute a plain
/// `Vec<Result<String, _>>` and verify cancellation semantics without a
/// real GGUF file on disk.
pub fn pump<I, S, E, F>(
    stream: I,
    cancel: &AtomicBool,
    mut on_token: F,
) -> Result<GenerateOutcome, EngineError>
where
    I: IntoIterator<Item = Result<S, E>>,
    S: AsRef<str>,
    E: std::fmt::Debug,
    F: FnMut(&str),
{
    let mut tokens = 0usize;
    for item in stream {
        if cancel.load(Ordering::SeqCst) {
            return Ok(GenerateOutcome::Cancelled { tokens });
        }
        let chunk = item.map_err(|e| EngineError::LlamaGguf(format!("{e:?}")))?;
        tokens += 1;
        on_token(chunk.as_ref());
    }
    Ok(GenerateOutcome::Finished { tokens })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    #[derive(Debug, thiserror::Error)]
    #[error("fake stream error")]
    struct FakeErr;

    fn ok_tokens(tokens: &[&str]) -> Vec<Result<String, FakeErr>> {
        tokens.iter().map(|t| Ok((*t).to_string())).collect()
    }

    #[test]
    fn runs_to_completion_when_not_cancelled() {
        let cancel = AtomicBool::new(false);
        let captured = RefCell::new(String::new());

        let outcome = pump(ok_tokens(&["hel", "lo", " world"]), &cancel, |t| {
            captured.borrow_mut().push_str(t);
        })
        .unwrap();

        assert_eq!(outcome, GenerateOutcome::Finished { tokens: 3 });
        assert_eq!(&*captured.borrow(), "hello world");
    }

    #[test]
    fn returns_cancelled_immediately_when_flag_preset() {
        let cancel = AtomicBool::new(true);
        let mut hits = 0;

        let outcome = pump(ok_tokens(&["a", "b", "c"]), &cancel, |_| hits += 1).unwrap();

        assert_eq!(outcome, GenerateOutcome::Cancelled { tokens: 0 });
        assert_eq!(hits, 0);
    }

    #[test]
    fn stops_mid_stream_when_flag_flipped() {
        let cancel = AtomicBool::new(false);
        let hits = RefCell::new(0usize);

        // Flip the cancel flag from inside the on_token callback after the
        // second token so the third never lands.
        let outcome = pump(ok_tokens(&["a", "b", "c", "d"]), &cancel, |_| {
            let mut n = hits.borrow_mut();
            *n += 1;
            if *n == 2 {
                cancel.store(true, Ordering::SeqCst);
            }
        })
        .unwrap();

        assert_eq!(outcome, GenerateOutcome::Cancelled { tokens: 2 });
        assert_eq!(*hits.borrow(), 2);
    }

    #[test]
    fn propagates_stream_error() {
        let cancel = AtomicBool::new(false);
        let stream: Vec<Result<String, FakeErr>> = vec![Ok("a".into()), Err(FakeErr)];

        let err = pump(stream, &cancel, |_| {}).unwrap_err();
        match err {
            EngineError::LlamaGguf(msg) => assert!(msg.contains("FakeErr")),
        }
    }

    #[test]
    fn outcome_accessors() {
        assert_eq!(GenerateOutcome::Finished { tokens: 5 }.tokens(), 5);
        assert!(GenerateOutcome::Cancelled { tokens: 3 }.was_cancelled());
        assert!(!GenerateOutcome::Finished { tokens: 0 }.was_cancelled());
    }
}
