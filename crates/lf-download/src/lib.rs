#![forbid(unsafe_code)]

//! Pinned, checksum-verified model fetching.
//!
//! All network I/O in `local-ferris` is confined to this crate so the
//! rest of the workspace can honor the "no ambient network" posture
//! (see README §6). Artifacts are described in a TOML manifest that
//! lists one entry per alias with a URL and a SHA-256 pin; downloads
//! are streamed to `<dest>.part`, hashed on the fly, and atomically
//! renamed into place only after the hash matches.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    #[serde(default)]
    pub models: BTreeMap<String, ManifestEntry>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ManifestEntry {
    pub url: String,
    pub sha256: String,
    pub size_bytes: Option<u64>,
    pub license: Option<String>,
    pub source: Option<String>,
    pub provenance_note: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    #[error("reading manifest: {0}")]
    Io(#[from] std::io::Error),
    #[error("parsing manifest TOML: {0}")]
    Toml(#[from] toml::de::Error),
    #[error(
        "invalid sha256 pin for `{alias}`: must be 64 hex chars, got {actual_len} chars: {actual:?}"
    )]
    InvalidSha {
        alias: String,
        actual: String,
        actual_len: usize,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum VerifyError {
    #[error("file I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("sha256 mismatch: expected {expected}, got {actual}")]
    Mismatch { expected: String, actual: String },
}

#[derive(Debug, thiserror::Error)]
pub enum DownloadError {
    #[error("file I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("HTTP transport: {0}")]
    Http(#[from] reqwest::Error),
    #[error("HTTP status {status} from {url}")]
    HttpStatus {
        status: reqwest::StatusCode,
        url: String,
    },
    #[error("sha256 mismatch: expected {expected}, got {actual} (partial file discarded)")]
    Mismatch { expected: String, actual: String },
}

#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    #[error(
        "unable to determine user cache directory; set XDG_CACHE_HOME or HOME, or pass an explicit path"
    )]
    NoHomeDir,
}

impl Manifest {
    pub fn from_toml_str(s: &str) -> Result<Self, ManifestError> {
        let m: Manifest = toml::from_str(s)?;
        m.validate()?;
        Ok(m)
    }

    pub fn from_path(path: &Path) -> Result<Self, ManifestError> {
        let s = fs::read_to_string(path)?;
        Self::from_toml_str(&s)
    }

    pub fn get(&self, alias: &str) -> Option<&ManifestEntry> {
        self.models.get(alias)
    }

    pub fn aliases(&self) -> impl Iterator<Item = &str> {
        self.models.keys().map(String::as_str)
    }

    fn validate(&self) -> Result<(), ManifestError> {
        for (alias, entry) in &self.models {
            let s = &entry.sha256;
            if s.len() != 64 || !s.chars().all(|c| c.is_ascii_hexdigit()) {
                return Err(ManifestError::InvalidSha {
                    alias: alias.clone(),
                    actual: s.clone(),
                    actual_len: s.len(),
                });
            }
        }
        Ok(())
    }
}

/// Default per-user cache directory for downloaded artifacts
/// (`$XDG_CACHE_HOME/local-ferris/models`, falling back to the
/// platform equivalent).
pub fn default_cache_dir() -> Result<PathBuf, CacheError> {
    let proj =
        directories::ProjectDirs::from("", "", "local-ferris").ok_or(CacheError::NoHomeDir)?;
    Ok(proj.cache_dir().join("models"))
}

/// Stream `path` through a SHA-256 hasher and compare to `expected_hex`
/// (case-insensitive). Used both after download and by `--verify-only`.
pub fn verify_sha256(path: &Path, expected_hex: &str) -> Result<(), VerifyError> {
    let mut file = File::open(path)?;
    let actual = hash_reader(&mut file)?;
    if actual.eq_ignore_ascii_case(expected_hex) {
        Ok(())
    } else {
        Err(VerifyError::Mismatch {
            expected: expected_hex.to_owned(),
            actual,
        })
    }
}

/// Download `entry.url` to `dest`, streaming through a SHA-256 hasher.
/// On hash match the `.part` file is atomically renamed to `dest`; on
/// mismatch it is discarded and a `DownloadError::Mismatch` is returned.
///
/// Intermediate progress is logged to stderr every ~16 MiB.
pub fn download_and_verify(entry: &ManifestEntry, dest: &Path) -> Result<(), DownloadError> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    let part = part_path(dest);

    let client = reqwest::blocking::Client::builder()
        .user_agent(concat!("local-ferris/", env!("CARGO_PKG_VERSION")))
        .build()?;
    let mut response = client.get(&entry.url).send()?;
    if !response.status().is_success() {
        return Err(DownloadError::HttpStatus {
            status: response.status(),
            url: entry.url.clone(),
        });
    }

    let mut file = File::create(&part)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    let mut total: u64 = 0;
    let mut next_log: u64 = 16 * 1024 * 1024;

    loop {
        let n = response.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        file.write_all(&buf[..n])?;
        total += n as u64;
        if total >= next_log {
            eprintln!(
                "[download] {} MiB streamed{}",
                total / (1024 * 1024),
                entry
                    .size_bytes
                    .map(|t| format!(" of {} MiB", t / (1024 * 1024)))
                    .unwrap_or_default(),
            );
            next_log += 16 * 1024 * 1024;
        }
    }
    file.flush()?;
    drop(file);

    let actual = hex::encode(hasher.finalize());
    if !actual.eq_ignore_ascii_case(&entry.sha256) {
        let _ = fs::remove_file(&part);
        return Err(DownloadError::Mismatch {
            expected: entry.sha256.clone(),
            actual,
        });
    }
    fs::rename(&part, dest)?;
    Ok(())
}

fn hash_reader<R: Read>(r: &mut R) -> io::Result<String> {
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = r.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

fn part_path(dest: &Path) -> PathBuf {
    // Append ".part" to the final path component rather than replacing
    // the extension — keeps `foo.gguf` → `foo.gguf.part` so the partial
    // is obviously not a drop-in for `foo.gguf`.
    let mut s = dest.as_os_str().to_owned();
    s.push(".part");
    PathBuf::from(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "lf-download-test-{}-{}-{name}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn manifest_parses_minimal_entry() {
        let s = r#"
[models."starcoder2-3b.Q4_K_M"]
url = "https://example.com/foo.gguf"
sha256 = "0000000000000000000000000000000000000000000000000000000000000000"
"#;
        let m = Manifest::from_toml_str(s).unwrap();
        let e = m.get("starcoder2-3b.Q4_K_M").unwrap();
        assert_eq!(e.url, "https://example.com/foo.gguf");
        assert!(e.license.is_none());
        assert!(e.size_bytes.is_none());
    }

    #[test]
    fn manifest_parses_rich_entry() {
        let s = r#"
[models.phi-3_5-mini]
url = "https://huggingface.co/example/phi-3.5-mini-Q4_K_M.gguf"
sha256 = "aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899"
size_bytes = 2500000000
license = "MIT"
source = "Microsoft"
provenance_note = "Non-PRC origin."
"#;
        let m = Manifest::from_toml_str(s).unwrap();
        let e = m.get("phi-3_5-mini").unwrap();
        assert_eq!(e.size_bytes, Some(2_500_000_000));
        assert_eq!(e.license.as_deref(), Some("MIT"));
        assert_eq!(e.source.as_deref(), Some("Microsoft"));
        assert_eq!(e.provenance_note.as_deref(), Some("Non-PRC origin."));
    }

    #[test]
    fn manifest_rejects_short_sha() {
        let s = r#"
[models.foo]
url = "https://example.com/x"
sha256 = "deadbeef"
"#;
        let err = Manifest::from_toml_str(s).unwrap_err();
        match err {
            ManifestError::InvalidSha {
                alias, actual_len, ..
            } => {
                assert_eq!(alias, "foo");
                assert_eq!(actual_len, 8);
            }
            other => panic!("expected InvalidSha, got {other:?}"),
        }
    }

    #[test]
    fn manifest_rejects_non_hex_sha() {
        let s = r#"
[models.foo]
url = "https://example.com/x"
sha256 = "ZZZZ000000000000000000000000000000000000000000000000000000000000"
"#;
        let err = Manifest::from_toml_str(s).unwrap_err();
        assert!(matches!(err, ManifestError::InvalidSha { .. }));
    }

    #[test]
    fn manifest_accepts_uppercase_hex() {
        let s = r#"
[models.foo]
url = "https://example.com/x"
sha256 = "B94D27B9934D3E08A52E52D7DA7DABFAC484EFE37A5380EE9088F7ACE2EFCDE9"
"#;
        let m = Manifest::from_toml_str(s).unwrap();
        assert!(m.get("foo").is_some());
    }

    #[test]
    fn manifest_empty_is_valid() {
        let m = Manifest::from_toml_str("").unwrap();
        assert!(m.aliases().next().is_none());
    }

    #[test]
    fn get_unknown_alias_returns_none() {
        let s = r#"
[models.foo]
url = "https://example.com/x"
sha256 = "0000000000000000000000000000000000000000000000000000000000000000"
"#;
        let m = Manifest::from_toml_str(s).unwrap();
        assert!(m.get("bar").is_none());
    }

    #[test]
    fn verify_sha256_matches_known_hash_of_hello_world() {
        // sha256("hello world") = b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9
        let path = scratch_path("hello.bin");
        std::fs::write(&path, b"hello world").unwrap();
        let res = verify_sha256(
            &path,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
        );
        let _ = std::fs::remove_file(&path);
        res.unwrap();
    }

    #[test]
    fn verify_sha256_is_case_insensitive() {
        let path = scratch_path("hello-upper.bin");
        std::fs::write(&path, b"hello world").unwrap();
        let res = verify_sha256(
            &path,
            "B94D27B9934D3E08A52E52D7DA7DABFAC484EFE37A5380EE9088F7ACE2EFCDE9",
        );
        let _ = std::fs::remove_file(&path);
        res.unwrap();
    }

    #[test]
    fn verify_sha256_reports_mismatch() {
        let path = scratch_path("mismatch.bin");
        std::fs::write(&path, b"hello world").unwrap();
        let res = verify_sha256(
            &path,
            "0000000000000000000000000000000000000000000000000000000000000000",
        );
        let _ = std::fs::remove_file(&path);
        match res.unwrap_err() {
            VerifyError::Mismatch { expected, actual } => {
                assert_eq!(
                    expected,
                    "0000000000000000000000000000000000000000000000000000000000000000"
                );
                assert_eq!(
                    actual,
                    "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
                );
            }
            other => panic!("expected Mismatch, got {other:?}"),
        }
    }

    #[test]
    fn part_path_appends_suffix_after_extension() {
        let dest = Path::new("/tmp/models/foo.gguf");
        assert_eq!(part_path(dest), PathBuf::from("/tmp/models/foo.gguf.part"));
    }

    #[test]
    fn default_cache_dir_is_deterministic() {
        // Not asserting absolute path (varies by host); only that it
        // doesn't error and ends in the expected tail.
        let p = default_cache_dir().unwrap();
        assert!(
            p.ends_with("local-ferris/models") || p.ends_with("local-ferris\\models"),
            "unexpected cache dir: {p:?}"
        );
    }
}
