#!/usr/bin/env python3
"""Assemble a Rust fine-tune corpus from corpus/manifest.toml.

Standard-library-only (Python 3.11+ for `tomllib`). Reads the manifest,
refuses REPLACE-ME revisions, clones each listed source into a cache
directory, checks out the pinned revision, walks the declared `include`
directories for `.rs` files, hashes content for whole-corpus
deduplication, and writes one JSONL record per surviving file plus a
TOML stats companion.

Why Python and not a local-ferris subcommand: LoRA fine-tuning itself
is Python (peft/transformers), so the corpus-prep step lives in the
same ecosystem — users will extend this script in the same shell they
run training from. The Rust workspace stays focused on inference and
pipeline artifacts.

Usage:
    scripts/build_corpus.py \\
        --manifest corpus/manifest.toml \\
        --output corpus/out/corpus.jsonl \\
        --cache corpus/.cache \\
        --stats corpus/out/stats.toml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import tomllib


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default="corpus/manifest.toml")
    ap.add_argument("--output", default="corpus/out/corpus.jsonl")
    ap.add_argument("--cache", default="corpus/.cache")
    ap.add_argument("--stats", default="corpus/out/stats.toml")
    args = ap.parse_args()

    manifest_path = pathlib.Path(args.manifest)
    if not manifest_path.exists():
        print(f"error: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    with manifest_path.open("rb") as f:
        manifest = tomllib.load(f)

    sources = manifest.get("source", [])
    if not sources:
        print(f"error: manifest {manifest_path} has no [[source]] entries", file=sys.stderr)
        return 1

    # Refuse to proceed with unpinned revisions — the point of the
    # manifest is auditability, and REPLACE-ME defeats that.
    unpinned = [s["name"] for s in sources if s.get("revision", "").startswith("REPLACE-ME")]
    if unpinned:
        print(
            "error: the following sources have REPLACE-ME revisions; "
            "pin them to real upstream refs before building a corpus:",
            file=sys.stderr,
        )
        for name in unpinned:
            print(f"  - {name}", file=sys.stderr)
        return 1

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = pathlib.Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    seen_hashes: set[str] = set()
    files_seen = files_included = dupes = total_bytes = 0

    with out_path.open("w", encoding="utf-8") as out:
        for src in sources:
            name = src["name"]
            repo = src["repo"]
            rev = src["revision"]
            license_ = src["license"]
            includes = src.get("include", ["."])

            clone_dir = cache_dir / name
            if not clone_dir.exists():
                print(f"[corpus] cloning {name} from {repo}", file=sys.stderr)
                subprocess.run(
                    ["git", "clone", "--quiet", repo, str(clone_dir)],
                    check=True,
                )
            print(f"[corpus] {name} → checkout {rev}", file=sys.stderr)
            # Fetch tags in case `rev` is a tag that wasn't on the
            # default clone (some repos ship tags separately).
            subprocess.run(
                ["git", "-C", str(clone_dir), "fetch", "--quiet", "--tags"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(clone_dir), "checkout", "--quiet", "--detach", rev],
                check=True,
            )

            for inc in includes:
                inc_dir = clone_dir / inc
                if not inc_dir.exists():
                    print(
                        f"[corpus] warn: include path missing for {name}: {inc}",
                        file=sys.stderr,
                    )
                    continue
                for path in inc_dir.rglob("*.rs"):
                    if not path.is_file():
                        continue
                    try:
                        content = path.read_text(encoding="utf-8")
                    except UnicodeDecodeError:
                        continue  # skip non-UTF-8 files silently
                    files_seen += 1
                    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
                    if digest in seen_hashes:
                        dupes += 1
                        continue
                    seen_hashes.add(digest)
                    rel = path.relative_to(clone_dir).as_posix()
                    record = {
                        "source": name,
                        "license": license_,
                        "path": rel,
                        "content": content,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    files_included += 1
                    total_bytes += len(content.encode("utf-8"))

    stats_path = pathlib.Path(args.stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        "# Generated by scripts/build_corpus.py — do not edit by hand.\n"
        "version = 1\n"
        f'manifest = "{manifest_path.as_posix()}"\n'
        f"sources = {len(sources)}\n"
        f"files_seen = {files_seen}\n"
        f"files_included = {files_included}\n"
        f"duplicates_removed = {dupes}\n"
        f"total_bytes = {total_bytes}\n"
        f'output = "{out_path.as_posix()}"\n'
    )

    print(
        f"corpus: {files_included} files ({total_bytes:,} bytes, "
        f"{dupes} duplicates removed) → {out_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
