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

- **HF → raw GGUF** (external, Python-based): `llama_gguf` does not
  ingest safetensors, so this leg of the pipeline leans on an existing
  converter — canonically `llama.cpp/convert_hf_to_gguf.py` (pure
  Python; does **not** require the llama.cpp C/C++ engine to be built).
  A wrapper lives at `scripts/convert_hf_to_gguf.sh`; it also stamps a
  ready-to-paste `manifests/models.toml` stanza with the freshly
  computed SHA-256. A pure-Rust converter (safetensors parse + per-
  architecture tensor-name mapping + tokenizer export) is genuine
  future work, not a Phase 2 deliverable.
- **Quantization** (Rust-native): once a raw GGUF exists, downstream
  quantization runs through `llama_gguf::gguf::quantize_model` with
  `QuantizeOptions` — no external toolchain. Exposed via
  `local-ferris quantize`, which optionally emits a TOML size/quality
  report.
  - Default: **Q4_K** (≈4.5 bits/weight, best size/quality on AVX2).
  - Precision fallback: **Q5_K** when quality regressions are observed
    on the Rust eval set.
  - Ceiling: **Q8_0** for calibration runs.
  - Vocabulary note: llama.cpp popularized `Q4_K_M` / `Q5_K_M` — those
    suffixes denote a *mixture* (attention tensors at one type, FFN at
    another). `llama_gguf 0.14` only ships uniform k-quants, so the CLI
    accepts the `_M` suffix as a familiarity alias and strips it; every
    tensor gets the same target. Mixed-precision quantization is a
    future enhancement.
- **Calibration** is handled by `local-ferris eval`, which runs a
  Rust task-completion suite (`evals/rust_suite.toml`) against a
  quantized model and reports per-case pass/fail plus an aggregate
  pass rate. Two reports (pre-quant vs post-quant) can be diff'd to
  catch regressions introduced by the chosen target type.
  - **Perplexity is not yet wired.** Real perplexity needs per-token
    logits from a teacher-forced forward pass; `llama_gguf 0.14` does
    not expose that at the `Engine` level (only `model() -> &dyn Model`
    with the caller driving the forward pass themselves). Task-
    completion pass rate carries the quality signal in the size/quality
    report for now; perplexity is genuine follow-up work.

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
- Threading: `llama_gguf`'s CPU backend dispatches all parallelism
  through rayon. We expose `--threads N` on the inference subcommands;
  it calls `rayon::ThreadPoolBuilder::build_global` before any `Engine`
  is constructed. Default is rayon's default (physical cores — 4 on the
  reference machine). A `BatchedEngine` with an explicit `n_batch` knob
  exists upstream but targets multi-stream server workloads; it is not
  wired in here because this is single-prompt interactive inference.
- Context: default 4096 tokens, configurable up to 8192 via `--ctx N`,
  plumbed through `EngineConfig::max_context_len`. Use sliding-window
  trimming rather than growing context indefinitely.
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
- [x] Wire `cargo-deny` + `cargo-audit` into CI
- [x] `#![forbid(unsafe_code)]` in every owned crate
- [x] Baseline CLI skeleton (`clap`, subcommands: `chat`, `complete`,
      `index`, `download`)

### Phase 1 — inference
- [x] Add `llama_gguf` dependency and a smoke test against a stock
      GGUF model
- [x] Streaming token output with cancel-on-Ctrl-C
- [x] Configurable threads and context (batch is a `BatchedEngine`
      concern upstream; out of scope for single-stream inference — see §4)
- [x] Prompt template with system / user / tool roles
- [x] Benchmark harness reporting tok/s for each quant level

### Phase 2 — model pipeline
- [x] `lf-download` with SHA-256 manifest verification
- [x] HF → GGUF conversion script (external Python converter; see
      `scripts/convert_hf_to_gguf.sh` and §3 for the scope boundary)
- [x] Quantize to Q4_K, Q5_K, Q8_0 and commit size/quality report
      (uniform k-quants — see §3 vocabulary note)
- [x] Calibration eval suite — task-completion on held-out Rust
      snippets (`evals/rust_suite.toml`, driven by `local-ferris eval`).
      Perplexity deferred pending a per-token-logits API in
      `llama_gguf`; see §3.

### Phase 3 — fine-tuning
- [x] Curate Rust fine-tune corpus (license-clean, deduped) —
      `corpus/manifest.toml` + `scripts/build_corpus.py`
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

## Development

### Running the smoke test

`crates/lf-inference/tests/smoke.rs` holds two integration tests:

- `default_backend_initializes` runs on every `cargo test` invocation.
  It proves that `llama_gguf` is actually linked into the build, with no
  model file required.
- `opens_a_stock_gguf_model` is marked `#[ignore]` so it does **not**
  run in default `cargo test` (or in CI). It opens a real GGUF file via
  `llama_gguf::GgufFile::open` and asserts the header parses.

To exercise the ignored test locally, grab a small non-PRC-origin GGUF
(TinyLlama-1.1B-Chat-v1.0 Q4_K_M at ~637 MiB works well — Apache-2.0,
originated at Singapore University of Technology and Design) and point
the test at it via the `LF_SMOKE_MODEL` environment variable:

```sh
LF_SMOKE_MODEL=/path/to/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf \
    cargo test -p lf-inference --test smoke -- --ignored
```

Alternatively, drop the file at `test-fixtures/smoke.gguf` in the repo
root; the test resolves that path as a fallback. The `test-fixtures/`
directory is gitignored, so committing large weights is not a risk.

### Converting a new HF model

HF safetensors → GGUF is the one step in the pipeline that is **not**
Rust-native today (see §3). `scripts/convert_hf_to_gguf.sh` wraps an
external Python converter; the canonical one is
`llama.cpp/convert_hf_to_gguf.py` (pure Python — clone the llama.cpp
repo, you do not need to build its C/C++ engine).

```sh
# one-time setup: clone the Python converter
git clone --depth 1 https://github.com/ggerganov/llama.cpp ~/src/llama.cpp

# convert an HF snapshot (local dir or previously `huggingface-cli snapshot-download`ed)
CONVERTER=~/src/llama.cpp/convert_hf_to_gguf.py \
    scripts/convert_hf_to_gguf.sh \
    ~/hf-cache/bigcode/starcoder2-3b \
    ./starcoder2-3b.gguf
```

The script prints a ready-to-paste `manifests/models.toml` stanza with
the output's size and SHA-256 already filled in; you add `url`,
`license`, `source`, and `provenance_note` after uploading. Then
quantize (next Phase 2 item) and publish.

### Building the fine-tune corpus

LoRA training lives in Python (§2), and so does corpus prep:
`scripts/build_corpus.py` reads `corpus/manifest.toml`, clones each
declared source at its pinned revision, walks the include paths for
`.rs` files, deduplicates by content hash across the whole corpus,
and emits a JSONL file plus a TOML stats companion. Standard-library
Python 3.11+ only — no pip install needed beyond `tomllib` (stdlib)
and `git` on `$PATH`.

The manifest ships with every revision set to a `REPLACE-ME`
placeholder; the script refuses to build until they are pinned to
real upstream refs. Pinning is a deliberate, auditable step — the
commit you choose determines the license text and the code your
fine-tune will imitate. Do this in its own commit so reviewers can
see which revisions are load-bearing.

```sh
# 1. Pin revisions in corpus/manifest.toml (one commit).
# 2. Build:
scripts/build_corpus.py \
    --manifest corpus/manifest.toml \
    --output corpus/out/corpus.jsonl \
    --stats corpus/out/stats.toml

# 3. Feed corpus/out/corpus.jsonl into the LoRA trainer (§2 of README).
```

`corpus/.cache/` (git clones) and `corpus/out/` (generated JSONL +
stats) are gitignored so nothing large lands in-repo.

### Running the eval harness

`local-ferris eval` runs a TOML-defined task-completion suite against
a GGUF model and reports per-case pass/fail plus an aggregate pass
rate. The default suite ships at `evals/rust_suite.toml` and seeds a
handful of canonical Rust completion prompts (iterative Fibonacci,
slice iterator composition, `Result`/`?` propagation, `Option` match,
struct constructor, `fmt::Display` impl). Each case is a raw prompt
(no chat-template wrapping) paired with a short `expect` substring —
loose on purpose, because the goal is a regression signal, not a
grade.

```sh
local-ferris eval \
    --model ./quant/raw.Q4_K.gguf \
    --report ./eval-Q4_K.toml

# compare two reports by eye (or via `diff`) to catch quant regressions
diff <(cat eval-Q4_K.toml)  <(cat eval-Q8_0.toml)
```

Output is a TOML report with `[[case]]` stanzas and a top-level
`pass_rate`. Exit code is 0 regardless of pass rate — the harness is
a measurement, not a gate. Wire it into CI as a gate in whatever shape
makes sense for your workflow. Ctrl-C during a run cancels the
in-flight case cleanly and still prints the partial summary.

### Producing quantized GGUFs

Given a raw GGUF (F16 / F32 — the output of
`scripts/convert_hf_to_gguf.sh`), run `local-ferris quantize` to
produce one or more quantized variants in a single invocation:

```sh
local-ferris quantize \
    --input ./raw.gguf \
    --out-dir ./quant \
    --targets Q4_K,Q5_K,Q8_0 \
    --report ./quant/report.toml
```

Per-target output lands at `<stem>.<target>.gguf` (e.g.
`raw.Q4_K.gguf`). When `--report` is set, a TOML summary is written
with one `[[quantize]]` stanza per target:

```toml
version = 1

[[quantize]]
target = "Q4_K"
input_bytes = 6012345678
output_bytes = 1923456789
reduction_ratio = 0.32
tensors_total = 224
tensors_quantized = 200
tensors_skipped = 24
duration_secs = 45.2
```

Quality measurement (perplexity against held-out Rust code) is the
next Phase 2 item and ties into `evals/`; the ratio + tensor counts
here are the "size" half of "size/quality report".

### Downloading pinned models

Network access is confined to the `download` subcommand; the rest of the
workspace builds and runs offline. `manifests/models.toml` maps each
artifact alias to a URL, a SHA-256 pin, and optional metadata
(`size_bytes`, `license`, `source`, `provenance_note`). The file ships
empty — populate entries only after verifying a real download against
its hash (`sha256sum path/to/file.gguf`). The point of the manifest is
trust, so placeholder hashes are not acceptable in commits.

```sh
# fetch and verify
local-ferris download starcoder2-3b.Q4_K_M

# re-verify an already-cached file without refetching
local-ferris download starcoder2-3b.Q4_K_M --verify-only

# use a different manifest
local-ferris download foo --manifest ./alt-manifest.toml
```

Artifacts land in the per-user cache directory
(`$XDG_CACHE_HOME/local-ferris/models` on Linux,
`~/Library/Caches/local-ferris/models` on macOS). Downloads stream into
`<name>.part` while a SHA-256 is computed on the fly; the file is
atomically renamed into place only after the hash matches. A partial
file that fails verification is discarded automatically.

### Benchmarking quantizations

The `local-ferris bench` subcommand measures tokens/second across one
or more GGUF files. Typical use is to point it at sibling quantizations
of the same base model to compare the size/speed trade:

```sh
local-ferris bench \
    ~/models/starcoder2-3b.Q4_K_M.gguf \
    ~/models/starcoder2-3b.Q5_K_M.gguf \
    ~/models/starcoder2-3b.Q8_0.gguf \
    --max-tokens 128 --threads 4
```

Output is a plain table with `label | run | load_ms | tokens |
elapsed_ms | tok/s`. Labels are derived from filename stems. Runs
default to 1; pass `--runs N` for multiple rows per model. Ctrl-C
during a run aborts cleanly and prints the partial table. The
methodology is deliberately simple — no warmup window, no statistical
aggregation — so these numbers spot order-of-magnitude regressions
rather than claim rigor.

## Non-goals

- GPU acceleration paths (CUDA, Metal, ROCm, Vulkan).
- Training models from scratch.
- Multi-tenant / server deployments.
- Agentic autonomous execution without a human in the loop.
