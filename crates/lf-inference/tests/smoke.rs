//! Smoke tests for the `llama_gguf` integration.
//!
//! The default-backend test runs unconditionally and is enough to prove the
//! crate actually links. The GGUF-open test is gated behind `#[ignore]` so
//! `cargo test` stays offline and fast; run it explicitly when you want to
//! validate against real weights.

use std::env;
use std::path::PathBuf;

/// Verifies `llama_gguf` is linked and its default backend constructs.
/// No model file required.
#[test]
fn default_backend_initializes() {
    let _backend = llama_gguf::default_backend();
}

/// Opens a stock GGUF model from disk and confirms the header parses.
///
/// Run with:
///
/// ```sh
/// LF_SMOKE_MODEL=/path/to/model.gguf \
///     cargo test -p lf-inference --test smoke -- --ignored
/// ```
///
/// If `LF_SMOKE_MODEL` is unset, the test falls back to
/// `test-fixtures/smoke.gguf` at the repo root (gitignored).
///
/// A suitable, non-PRC-origin smoke target is
/// `TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf` (Apache-2.0, ~637 MiB).
#[test]
#[ignore = "requires a GGUF model on disk; set LF_SMOKE_MODEL"]
fn opens_a_stock_gguf_model() -> anyhow::Result<()> {
    let path = resolve_model_path()?;
    let path_str = path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("model path is not UTF-8: {}", path.display()))?;

    llama_gguf::GgufFile::open(path_str)
        .map_err(|e| anyhow::anyhow!("GgufFile::open failed for {}: {e:?}", path.display()))?;

    Ok(())
}

fn resolve_model_path() -> anyhow::Result<PathBuf> {
    if let Ok(env_path) = env::var("LF_SMOKE_MODEL") {
        return Ok(PathBuf::from(env_path));
    }

    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test-fixtures/smoke.gguf");
    if fixture.exists() {
        return Ok(fixture);
    }

    anyhow::bail!(
        "no GGUF model available; set LF_SMOKE_MODEL=/path/to/model.gguf \
         or place one at test-fixtures/smoke.gguf (gitignored)"
    )
}
