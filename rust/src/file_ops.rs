use std::path::Path;

pub struct RenameError {
    pub error_code: String,
    pub error_message: String,
    pub suggestion: String,
}

pub fn rename_file(old_path: &Path, new_path: &Path) -> Result<(), RenameError> {
    if new_path.exists() {
        return Err(RenameError {
            error_code: "FILE_EXISTS".to_string(),
            error_message: format!("Target file already exists: {}", new_path.display()),
            suggestion: "Delete the existing file or use a different naming scheme".to_string(),
        });
    }

    match std::fs::metadata(old_path) {
        Ok(_) => {}
        Err(e) => {
            return Err(RenameError {
                error_code: "SOURCE_ACCESS_ERROR".to_string(),
                error_message: format!("Cannot access source file: {}", e),
                suggestion: "Check file permissions or if file is locked by another process".to_string(),
            });
        }
    }

    match std::fs::rename(old_path, new_path) {
        Ok(_) => Ok(()),
        Err(e) => {
            let error_code = format!("RENAME_FAILED_{}", e.raw_os_error().unwrap_or(-1));
            let suggestion = match e.raw_os_error() {
                Some(5) => "Access denied. Check file permissions or if file is locked".to_string(),
                Some(32) => "File is locked by another process".to_string(),
                Some(2) => "Source file was moved or deleted during processing".to_string(),
                Some(3) => "Target directory does not exist or path is invalid".to_string(),
                Some(123) => "Invalid filename. Check for illegal characters".to_string(),
                Some(112) => "Disk is full. Free up disk space".to_string(),
                Some(19) => "Disk is write-protected".to_string(),
                _ => format!("OS error {}: {}", e.raw_os_error().unwrap_or(-1), e),
            };

            Err(RenameError {
                error_code,
                error_message: e.to_string(),
                suggestion,
            })
        }
    }
}
