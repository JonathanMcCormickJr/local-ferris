# Local Ferris 💻🦀

A local, CPU-friendly AI pair programmer for Rust-first software development.

Local Ferris is designed to run **entirely on a consumer-grade CPU** with no
network access at inference time. Every stage of the pipeline — model
download, light fine-tuning, quantization, inference, and retrieval — is
bounded by what an ordinary desktop can do in a reasonable amount of time.

## Goals

- **Local-only.** No telemetry, no hosted inference, no remote embeddings.
  The binary must run fully offline after the one-time model download.
- **CPU-first.** No CUDA, ROCm, Metal, or accelerator assumptions. The
  target is AVX2 + FMA (Haswell-era and newer). AVX-512 / VNNI are a bonus,
  not a requirement.
- **Rust-native, end-to-end.** The runtime, tokenizer glue, RAG index, CLI,
  and GGUF inference kernels are written in Rust. No linkage against
  `llama.cpp` or any other C/C++ inference engine — inference runs on the
  [`llama_gguf`](https://docs.rs/llama-gguf/latest/llama_gguf/) pure-Rust
  crate.
- **Rust ecosystem fluency.** The assistant is expected to have working
  mastery of the Rust toolchain and ecosystem: Cargo (workspaces,
  features, profiles, build scripts, target selection), crates.io and
  lib.rs for crate discovery / reputation signals, docs.rs as the
  canonical API reference, and `std` / `core` / `alloc` boundaries. The
  RAG pipeline is designed around these surfaces (see §5).
- **Trusted-jurisdiction model sourcing.** Weights and embedding models
  must originate from organizations outside the jurisdiction of the
  People's Republic of China. This rules out DeepSeek, Qwen, Yi, BGE,
  GLM, InternLM, and other PRC-origin releases regardless of license.
  The intent is supply-chain provenance, not a statement about model
  quality.
- **Security-conscious.** Treat models, weights, and indexed corpora as
  untrusted inputs. Sandbox any code the model proposes to run. Reproducible
  builds and pinned checksums throughout.

## Target hardware baseline

The reference development machine is used as the floor, not the ceiling:

| Resource | Reference floor                       |
| -------- | ------------------------------------- |
| CPU      | Intel i5-4590 — 4 cores / 4 threads   |
| SIMD     | AVX2, FMA, AES-NI (no AVX-512)        |
| RAM      | 16 GiB usable                         |
| Disk     | ~20 GiB free for weights + RAG index  |
| GPU      | none                                  |

These constraints drive every downstream choice: model size ≤ 3B params,
≤ 4-bit weights at runtime, context window ≤ 8k tokens by default, RAG
embeddings ≤ 384 dims.

## Architecture

```
  ┌───────────────────────────────────────────────────────────────┐
  │                        local-ferris CLI                       │
  └───────────────┬───────────────────────┬───────────────────────┘
                  │                       │
         ┌────────▼────────┐      ┌───────▼─────────┐
         │  RAG retriever  │      │ Inference core  │
         │  (tantivy +     │      │ (llama_gguf,    │
         │  flat cosine)   │      │  pure-Rust GGUF)│
         └────────┬────────┘      └───────┬─────────┘
                  │                       │
         ┌────────▼────────┐      ┌───────▼─────────┐
         │ Embedding model │      │ Coder LLM (GGUF │
         │ (all-MiniLM-    │      │  Q4_K_M / Q5_0) │
         │  L6-v2, INT8)   │      │                 │
         └─────────────────┘      └─────────────────┘
```

## Implementation plan

### 1. Model acquisition

- Pull weights from Hugging Face via a pinned revision + SHA-256 manifest.
  Downloads go through a dedicated `download` subcommand; the runtime binary
  itself never makes network calls.
- Candidate base models (decoder-only, code-pretrained, small enough to
  quantize and run comfortably in ≤ 4 GiB RAM, and sourced outside PRC
  jurisdiction):
  - **StarCoder2-3B** (primary — BigCode / ServiceNow + Hugging Face,
    OpenRAIL-M; viable on this floor at Q4_K_M).
  - **CodeGemma-2B** (Google DeepMind, Gemma terms; strong small-model
    code generation).
  - **Phi-3.5-mini-instruct** (Microsoft, MIT; stretch target at 3.8B
    params with Q4_K_M).
- Embedding model: **all-MiniLM-L6-v2** (sentence-transformers /
  Microsoft Research, Apache-2.0, 384-dim, ~22M params) exported to
  ONNX and quantized to INT8 for CPU inference via `ort`.

### 2. Light fine-tuning (LoRA on CPU)

Full fine-tuning is out of scope for this hardware. Instead:

- Train a **LoRA adapter** (rank 8–16, alpha 16–32) on a curated Rust
  corpus: idiomatic patterns from `rustc`, `tokio`, `serde`, and
  user-provided repos. Target ≤ 50 MiB of adapter weights.
- Drive training from Rust: prepare shards with `tokenizers` + `arrow2`,
  then invoke a minimal Python trainer (`peft` + `transformers`, CPU
  backend) as a one-shot subprocess. This keeps the Python dependency off
  the inference path.
- Budget: a few thousand examples, ≤ 1 epoch, overnight on 4 cores. The
  goal is style / idiom nudging, not teaching new capabilities.
- Merge the adapter into a fresh GGUF file; keep the base weights on disk
  unmodified so adapters can be swapped.

### 3. Quantization (≤ 8-bit, typically 4-bit)

- Convert merged weights to GGUF and quantize using Rust-native tooling
  built on `llama_gguf` (the same crate that powers inference). No
  `llama.cpp` toolchain is required anywhere in the pipeline.
  - Default: **Q4_K_M** (≈4.5 bits/weight, best size/quality on AVX2).
  - Precision fallback: **Q5_K_M** when quality regressions are observed
    on the Rust eval set.
  - Ceiling: **Q8_0** for calibration runs.
- Record a calibration diff: generate completions for a fixed suite of
  Rust prompts before and after quantization; fail the build if
  perplexity on held-out Rust code grows by more than a configured
  threshold.

### 4. Inference runtime

- Runtime: **[`llama_gguf`](https://docs.rs/llama-gguf/latest/llama_gguf/)**
  crate — a pure-Rust GGUF loader and k-quant inference engine. No C/C++
  link-time dependency, no FFI shim, no `build.rs` toolchain detection;
  the crate ships AVX2 / FMA kernels gated behind target-feature
  detection so the reference floor is hit without any user tuning.
- Feature gating: `llama_gguf` is pulled in with `default-features = false`.
  `lf-inference` enables only `cpu`. `lf-rag` additionally enables
  `rag-sqlite`. `lf-download` enables `huggingface` for model fetching.
  The `server`, `client`, `cli`, `cuda`, `metal`, `dx12`, `vulkan`,
  `distributed`, and `hailo` features are **never** enabled — they
  would either introduce network listeners, GPU assumptions, or
  remote-inference clients that contradict §6.
- Threading: expose `n_threads` (default = physical cores, here 4) and
  `n_batch` through config.
- Context: default 4096 tokens, configurable up to 8192. Use
  sliding-window trimming rather than growing context indefinitely.
- Sampling: temperature 0.2 for code completion, 0.7 for chat;
  repetition penalty 1.1; top-p 0.95.
- Expected throughput on the reference machine: ~3–5 tok/s at Q4_K_M
  for StarCoder2-3B, ~5–8 tok/s at Q4_K_M for CodeGemma-2B. This is the
  target the UX is designed around (streamed output, interruptible
  generation).

### 5. Retrieval-augmented generation

- **Backbone:** `llama_gguf`'s RAG primitives via the `rag-sqlite`
  feature — a local SQLite-backed vector store (no PostgreSQL, no
  external services). The crate also ships a PostgreSQL/pgvector
  backend under the plain `rag` feature; we explicitly do **not** enable
  that one because it would conflict with the offline, single-binary
  posture.
- **Index target:** the user's current workspace plus any explicitly
  added crates / docs. No implicit crawling.
- **Chunking:** tree-sitter-based splitting for `.rs` files (function /
  impl / module granularity), paragraph splitting for Markdown. Chunks
  are fed into `llama_gguf`'s ingest API.
- **Dense index:** all-MiniLM-L6-v2 embeddings produced by `lf-embed`
  (ONNX / `ort`, INT8) and stored in the `rag-sqlite` vector store.
  At ~10k chunks the built-in SQLite vector scan is comfortably faster
  than we need and avoids a second index implementation.
- **Lexical index:** `tantivy` for BM25 over the same chunks, held
  alongside the vector store in `lf-rag` rather than inside
  `llama_gguf`.
- **Hybrid retrieval:** reciprocal rank fusion over the `llama_gguf`
  dense retrieval and the `tantivy` BM25 retrieval, then re-rank with
  the LLM's own log-probs on the top 20 candidates.
- **Rust ecosystem sources.** The indexer understands Rust-specific
  corpora as first-class citizens: the current workspace's source tree,
  the resolved `Cargo.lock` graph, any path / git / registry dependency
  the workspace pulls in, and the user's local `rustup` doc bundle
  (`rustup doc --path`) for `std` / `core` / `alloc` / Cargo / Rust book
  content. Crate reputation metadata from crates.io (downloads, owners,
  yanked versions) and lib.rs (category, reverse-deps) is indexed
  alongside API docs so the model can make informed recommendations.
- **Auto-fetch latest dependencies' docs (opt-in).** When the
  `rag.auto_fetch_dep_docs` setting is `true`, the `index` subcommand
  resolves every direct dependency from `Cargo.toml` / `Cargo.lock` at
  its pinned version and pulls the corresponding docs.rs HTML bundle,
  the crates.io metadata record, and the lib.rs category summary. The
  fetched corpus is checksum-recorded into `manifests/docs.sha256` so
  the indexed view remains reproducible and auditable. Fetching happens
  **only** inside the `index` subcommand — the inference binary still
  has every HTTP client compiled out (see §6). The setting is `false`
  by default to honor the offline posture; enabling it is an explicit
  opt-in to occasional network access at index time.
- **Context assembly:** deterministic prompt template with explicit
  `<retrieved path="…">` tags so the model (and audit logs) can cite
  sources. docs.rs / crates.io / lib.rs chunks carry their canonical URL
  and version in the tag so citations round-trip back to upstream.

### 6. Security properties

- **No ambient network.** The inference binary is built with a feature
  flag that compiles out every HTTP client; network code lives only in
  the separate `download` subcommand.
- **Supply-chain hygiene.** `cargo-deny` and `cargo-audit` in CI;
  `Cargo.lock` committed; all model downloads checked against a
  committed `manifests/*.sha256`.
- **Untrusted-model posture.** Treat model outputs as data, never as
  commands. Any "run this" suggestion is surfaced as a diff / proposal
  the user approves; execution happens in a `bwrap` / `landlock`
  sandbox with a read-only rootfs and no network.
- **Prompt-injection mitigation.** Retrieved chunks are wrapped in
  clearly delimited, non-executable markers and stripped of anything
  resembling instructions-to-the-assistant before being shown to the
  model.
- **Minimal unsafe.** `#![forbid(unsafe_code)]` in every crate we own;
  `unsafe` only acceptable in vendored FFI shims.
- **Reproducible builds.** Pinned Rust toolchain via `rust-toolchain.toml`;
  deterministic `cargo build --locked`.

## Repository layout (planned)

```
local-ferris/
├── Cargo.toml              # workspace root
├── crates/
│   ├── lf-cli/             # user-facing binary
│   ├── lf-inference/       # llama_gguf runtime, sampling, prompts
│   ├── lf-rag/             # chunking, embeddings, tantivy, fusion
│   ├── lf-embed/           # ONNX embedding model via `ort`
│   ├── lf-docs/            # docs.rs / crates.io / lib.rs fetcher + parser
│   ├── lf-download/        # pinned, checksum-verified model fetcher
│   └── lf-sandbox/         # bwrap/landlock wrappers for code exec
├── manifests/              # SHA-256 pins for every downloadable asset
├── evals/                  # Rust prompt suite + scoring harness
└── scripts/
    ├── finetune_lora.py    # one-shot CPU LoRA trainer
    └── quantize.sh         # HF → GGUF → Q4_K_M pipeline
```

## Progress checklist

### Phase 0 — foundations
- [x] Convert to a Cargo workspace with the crate layout above
- [x] Pin Rust toolchain (`rust-toolchain.toml`, stable 1.95.0)
- [x] `deny.toml` committed (license allowlist, advisory policy,
      source allowlist)
- [ ] Wire `cargo-deny` + `cargo-audit` into CI
- [x] `#![forbid(unsafe_code)]` in every owned crate
- [ ] Baseline CLI skeleton (`clap`, subcommands: `chat`, `complete`,
      `index`, `download`)

### Phase 1 — inference
- [ ] Add `llama_gguf` dependency and a smoke test against a stock
      GGUF model
- [ ] Streaming token output with cancel-on-Ctrl-C
- [ ] Configurable threads / batch / context
- [ ] Prompt template with system / user / tool roles
- [ ] Benchmark harness reporting tok/s for each quant level

### Phase 2 — model pipeline
- [ ] `lf-download` with SHA-256 manifest verification
- [ ] HF → GGUF conversion script
- [ ] Quantize to Q4_K_M, Q5_K_M, Q8_0 and commit size/quality report
- [ ] Calibration eval suite (perplexity + task completion on held-out
      Rust snippets)

### Phase 3 — fine-tuning
- [ ] Curate Rust fine-tune corpus (license-clean, deduped)
- [ ] LoRA training script (CPU, `peft`) + reproducibility seed
- [ ] Adapter merge + re-quantize pipeline
- [ ] Before/after eval on the Rust prompt suite

### Phase 4 — RAG
- [ ] Tree-sitter-based Rust chunker
- [ ] Markdown chunker
- [ ] all-MiniLM-L6-v2 ONNX INT8 embedding path (`ort`)
- [ ] Flat `f16` dense index with SIMD cosine scan
- [ ] `tantivy` BM25 index
- [ ] Reciprocal-rank fusion + log-prob re-ranking
- [ ] Prompt-injection sanitizer for retrieved text
- [ ] `Cargo.toml` / `Cargo.lock` parser and dependency resolver
- [ ] docs.rs / crates.io / lib.rs fetcher with per-source rate limits
      and checksum manifests (`manifests/docs.sha256`)
- [ ] `rustup doc --path` bundler for offline std / core / Cargo docs
- [ ] `rag.auto_fetch_dep_docs` setting wired end-to-end (default
      `false`, opt-in network at index time only)

### Phase 5 — security hardening
- [ ] Network-free build profile (compile-time feature gate)
- [ ] `bwrap` + `landlock` sandbox for model-proposed commands
- [ ] Diff-review UX for any write / exec suggestion
- [ ] Threat model document covering prompt injection, model
      poisoning, and index tampering
- [ ] Reproducible-build verification (two independent builds match
      bit-for-bit)

### Phase 6 — UX polish
- [ ] Workspace auto-indexing with incremental updates on file save
- [ ] `.local-ferrisignore` support
- [ ] Session transcripts with source citations
- [ ] Config file (`~/.config/local-ferris/config.toml`) with sane
      defaults for the reference hardware

## Non-goals

- GPU acceleration paths (CUDA, Metal, ROCm, Vulkan).
- Training models from scratch.
- Multi-tenant / server deployments.
- Agentic autonomous execution without a human in the loop.
