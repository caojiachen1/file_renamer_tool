use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone)]
pub struct ScannedFile {
    pub path: PathBuf,
    pub size: u64,
}

pub fn normalize_extension(ext: &str) -> String {
    let lower = ext.to_lowercase();
    if lower.starts_with('.') {
        lower
    } else {
        format!(".{}", lower)
    }
}

pub fn should_process_file(path: &Path, allowed: &std::collections::HashSet<String>) -> bool {
    if allowed.is_empty() {
        return true;
    }
    match path.extension().and_then(|e| e.to_str()) {
        Some(ext) => allowed.contains(&normalize_extension(ext)),
        None => false,
    }
}

pub fn scan_directory(
    directory: &Path,
    recursive: bool,
    allowed_extensions: &std::collections::HashSet<String>,
) -> Vec<ScannedFile> {
    let mut files = Vec::with_capacity(5000);

    let result = if recursive {
        scan_recursive(directory, allowed_extensions, &mut files)
    } else {
        scan_flat(directory, allowed_extensions, &mut files)
    };

    if let Err(e) = result {
        eprintln!("Error scanning directory: {}", e);
    }

    files.sort_by(|a, b| a.size.cmp(&b.size));
    files
}

fn scan_flat(
    dir: &Path,
    allowed: &std::collections::HashSet<String>,
    out: &mut Vec<ScannedFile>,
) -> Result<(), std::io::Error> {
    let entries = fs::read_dir(dir)?;

    for entry in entries.flatten() {
        let path = entry.path();
        match path.metadata() {
            Ok(meta) => {
                if meta.is_file() && should_process_file(&path, allowed) {
                    out.push(ScannedFile {
                        path,
                        size: meta.len(),
                    });
                }
            }
            Err(e) => {
                eprintln!("Warning: Error accessing file: {}", e);
                continue;
            }
        }
    }

    Ok(())
}

fn scan_recursive(
    dir: &Path,
    allowed: &std::collections::HashSet<String>,
    out: &mut Vec<ScannedFile>,
) -> Result<(), std::io::Error> {
    let entries = fs::read_dir(dir)?;

    for entry in entries.flatten() {
        let path = entry.path();
        match path.metadata() {
            Ok(meta) => {
                if meta.is_file() && should_process_file(&path, allowed) {
                    out.push(ScannedFile {
                        path,
                        size: meta.len(),
                    });
                } else if meta.is_dir() {
                    if let Err(e) = scan_recursive(&path, allowed, out) {
                        eprintln!("Warning: Error scanning subdirectory: {}", e);
                    }
                }
            }
            Err(e) => {
                eprintln!("Warning: Error accessing entry: {}", e);
                continue;
            }
        }
    }

    Ok(())
}
