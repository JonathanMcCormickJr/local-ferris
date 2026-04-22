//! `local-ferris download` — resolve an alias in the manifest, fetch it
//! (or just re-verify an existing cached file) and report the result.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::cli::DownloadArgs;

const DEFAULT_MANIFEST: &str = "manifests/models.toml";

pub fn run(args: DownloadArgs) -> Result<()> {
    let manifest_path = args
        .manifest
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MANIFEST));
    let manifest = lf_download::Manifest::from_path(&manifest_path)
        .with_context(|| format!("failed to load manifest at {}", manifest_path.display()))?;

    let entry = manifest.get(&args.alias).ok_or_else(|| {
        let known: Vec<&str> = manifest.aliases().collect();
        anyhow::anyhow!(
            "alias `{}` not found in {}. Known aliases: {}",
            args.alias,
            manifest_path.display(),
            if known.is_empty() {
                "(manifest is empty)".to_string()
            } else {
                known.join(", ")
            },
        )
    })?;

    let cache_dir = lf_download::default_cache_dir().context("resolving cache dir")?;
    let dest = cache_path_for(&cache_dir, &args.alias, &entry.url);

    if args.verify_only {
        if !dest.exists() {
            bail!(
                "nothing to verify at {} — run without --verify-only to fetch",
                dest.display()
            );
        }
        lf_download::verify_sha256(&dest, &entry.sha256)
            .with_context(|| format!("verifying {}", dest.display()))?;
        println!("verified {}: {}", args.alias, dest.display());
        return Ok(());
    }

    if dest.exists() {
        eprintln!(
            "[download] already present at {}; re-verifying instead of refetching",
            dest.display()
        );
        match lf_download::verify_sha256(&dest, &entry.sha256) {
            Ok(()) => {
                println!("verified {}: {}", args.alias, dest.display());
                return Ok(());
            }
            Err(e) => {
                eprintln!("[download] existing file failed verification ({e}); refetching");
                let _ = std::fs::remove_file(&dest);
            }
        }
    }

    eprintln!("[download] {} ← {}", dest.display(), entry.url);
    lf_download::download_and_verify(entry, &dest)
        .with_context(|| format!("downloading {} from {}", args.alias, entry.url))?;
    println!("downloaded {}: {}", args.alias, dest.display());
    Ok(())
}

/// Pick a local filename for the cached artifact. Prefer the alias as
/// the stem with an extension guessed from the URL's trailing component
/// (so the file on disk is self-describing). Falls back to just the
/// alias if the URL has no useful extension.
fn cache_path_for(cache_dir: &Path, alias: &str, url: &str) -> PathBuf {
    let ext = url_extension(url);
    let filename = match ext {
        Some(e) => format!("{alias}.{e}"),
        None => alias.to_string(),
    };
    cache_dir.join(filename)
}

fn url_extension(url: &str) -> Option<&str> {
    // Strip query / fragment before looking at the path.
    let path = url.split(['?', '#']).next().unwrap_or(url);
    let tail = path.rsplit('/').next()?;
    let (_, ext) = tail.rsplit_once('.')?;
    if ext.is_empty() || ext.len() > 16 {
        None
    } else {
        Some(ext)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_path_uses_alias_and_guesses_extension() {
        let dir = Path::new("/cache");
        let p = cache_path_for(
            dir,
            "starcoder2-3b.Q4_K_M",
            "https://huggingface.co/foo/bar/resolve/main/starcoder2-3b.Q4_K_M.gguf",
        );
        assert_eq!(p, PathBuf::from("/cache/starcoder2-3b.Q4_K_M.gguf"));
    }

    #[test]
    fn cache_path_strips_query_string_before_guessing_ext() {
        let dir = Path::new("/cache");
        let p = cache_path_for(
            dir,
            "foo",
            "https://example.com/path/model.gguf?download=true",
        );
        assert_eq!(p, PathBuf::from("/cache/foo.gguf"));
    }

    #[test]
    fn cache_path_falls_back_to_alias_only_when_no_extension() {
        let dir = Path::new("/cache");
        let p = cache_path_for(dir, "foo", "https://example.com/api/v1/blob");
        assert_eq!(p, PathBuf::from("/cache/foo"));
    }

    #[test]
    fn url_extension_rejects_long_pseudo_extensions() {
        assert_eq!(
            url_extension("https://example.com/x.thisisprobablynotanextension"),
            None
        );
    }
}
