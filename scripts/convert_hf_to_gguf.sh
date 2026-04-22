#!/usr/bin/env bash
# convert_hf_to_gguf.sh — HF weights → raw (unquantized) GGUF.
#
# Usage:
#   scripts/convert_hf_to_gguf.sh <hf-repo-or-dir> <output.gguf>
#
# Environment:
#   CONVERTER   Path to the Python converter to invoke. No default —
#               you must set this explicitly. The canonical tool is
#               llama.cpp/convert_hf_to_gguf.py (pure Python; does NOT
#               require the llama.cpp C/C++ inference engine to be
#               built).  Any drop-in with compatible CLI works.
#   PYTHON      Python interpreter to use. Defaults to `python3`.
#   DTYPE       Precision of the output GGUF: f32, f16, bf16, or auto.
#               Defaults to f16.
#
# On success, prints a ready-to-paste `manifests/models.toml` stanza
# (minus `url`, which you fill in after uploading the artifact). Every
# field that can be derived locally is pre-filled, including a freshly
# computed sha256.
#
# Why this is a shell wrapper, not a Rust binary: a Rust-native
# HF→GGUF converter needs `safetensors` parsing plus per-architecture
# tensor-name mapping plus tokenizer export — it is a real piece of
# work that earns its own phase. Until then, conversion leans on the
# existing Python tool; see README §3 for the scope boundary.
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
usage: convert_hf_to_gguf.sh <hf-repo-or-dir> <output.gguf>

Required env: CONVERTER=<path-to-convert_hf_to_gguf.py>
Optional env: PYTHON (default python3), DTYPE (default f16)
EOF
    exit 2
}

if [[ $# -ne 2 ]]; then
    usage
fi

HF_SRC="$1"
OUT_GGUF="$2"

: "${CONVERTER:?CONVERTER env var is required (path to convert_hf_to_gguf.py)}"
: "${PYTHON:=python3}"
: "${DTYPE:=f16}"

if [[ ! -e "$CONVERTER" ]]; then
    echo "error: CONVERTER not found: $CONVERTER" >&2
    exit 1
fi

if [[ ! -e "$HF_SRC" ]]; then
    echo "error: HF source path does not exist: $HF_SRC" >&2
    echo "       (for remote HF repos, clone or snapshot-download first)" >&2
    exit 1
fi

OUT_DIR="$(dirname "$OUT_GGUF")"
if [[ ! -d "$OUT_DIR" ]]; then
    echo "error: output directory does not exist: $OUT_DIR" >&2
    exit 1
fi

echo "[convert] $HF_SRC → $OUT_GGUF (dtype=$DTYPE)" >&2
"$PYTHON" "$CONVERTER" "$HF_SRC" --outfile "$OUT_GGUF" --outtype "$DTYPE"

if [[ ! -s "$OUT_GGUF" ]]; then
    echo "error: converter produced no output at $OUT_GGUF" >&2
    exit 1
fi

SIZE_BYTES="$(wc -c < "$OUT_GGUF" | tr -d ' ')"
SHA256="$(sha256sum "$OUT_GGUF" | awk '{print $1}')"
ALIAS="$(basename "$OUT_GGUF" .gguf)"

cat <<EOF

# Paste into manifests/models.toml after uploading the artifact:
[models."${ALIAS}"]
url = "https://REPLACE-ME/${ALIAS}.gguf"
sha256 = "${SHA256}"
size_bytes = ${SIZE_BYTES}
license = "REPLACE-ME"
source = "REPLACE-ME"
provenance_note = "REPLACE-ME (must be non-PRC origin — see README §Goals)."
EOF
