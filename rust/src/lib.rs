pub mod hash;
pub mod scanner;
pub mod file_ops;
mod workers;

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

pub use hash::Algorithm;
pub use scanner::ScannedFile;
pub use file_ops::RenameError;
pub use workers::SharedReporter;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessingMode {
    SingleThread,
    MultiThread,
    BatchMode,
}

pub struct ProcessOptions {
    pub directory: PathBuf,
    pub algorithm: Algorithm,
    pub recursive: bool,
    pub dry_run: bool,
    pub allowed_extensions: HashSet<String>,
    pub num_threads: usize,
    pub batch_size: usize,
    pub mode: ProcessingMode,
}

/// Trait for receiving progress output from processing.
/// Implement this to forward output to Tauri events, stdout, etc.
pub trait ProgressReporter: Send + Sync {
    fn on_output(&self, text: &str);
}

/// Process files in the given directory with the specified options.
/// Output is sent through the reporter.
pub fn process_directory(options: ProcessOptions, reporter: Arc<dyn ProgressReporter + Send + Sync>) {
    match options.mode {
        ProcessingMode::SingleThread => {
            workers::process_single_threaded(
                &options.directory,
                options.algorithm,
                options.recursive,
                options.dry_run,
                options.allowed_extensions,
                reporter,
            );
        }
        ProcessingMode::MultiThread => {
            workers::process_multi_threaded(
                &options.directory,
                options.algorithm,
                options.recursive,
                options.dry_run,
                options.allowed_extensions,
                options.num_threads,
                reporter,
            );
        }
        ProcessingMode::BatchMode => {
            workers::process_batch_mode(
                &options.directory,
                options.algorithm,
                options.recursive,
                options.dry_run,
                options.allowed_extensions,
                options.num_threads,
                options.batch_size,
                reporter,
            );
        }
    }
}
