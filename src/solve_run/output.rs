//! Publish complete JSON values without exposing partially written replacements.
use serde::Serialize;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, thiserror::Error)]
#[error("JSON output {stage} failed for {path}: {source}")]
pub struct PublicationError {
    pub path: PathBuf,
    pub stage: &'static str,
    #[source]
    pub source: Box<dyn std::error::Error + Send + Sync>,
}

pub struct JsonPublisher {
    path: PathBuf,
    #[cfg(test)]
    failure: Option<&'static str>,
}

impl JsonPublisher {
    pub fn create(path: &Path, value: &impl Serialize) -> Result<Self, PublicationError> {
        let mut publisher = Self {
            path: path.to_owned(),
            #[cfg(test)]
            failure: None,
        };
        publisher.write(value, true)?;
        Ok(publisher)
    }

    pub fn publish(&mut self, value: &impl Serialize) -> Result<(), PublicationError> {
        self.write(value, false)
    }

    fn error(
        &self,
        stage: &'static str,
        source: impl Into<Box<dyn std::error::Error + Send + Sync>>,
    ) -> PublicationError {
        PublicationError {
            path: self.path.clone(),
            stage,
            source: source.into(),
        }
    }

    fn write(&mut self, value: &impl Serialize, exclusive: bool) -> Result<(), PublicationError> {
        let bytes = serde_json::to_vec_pretty(value).map_err(|e| self.error("serialize", e))?;
        let parent = self
            .path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        fs::create_dir_all(parent).map_err(|e| self.error("create parent", e))?;
        let (temp, mut file) =
            Temporary::create(parent).map_err(|e| self.error("create temporary file", e))?;
        let inject = |_stage| -> std::io::Result<()> {
            #[cfg(test)]
            if self.failure == Some(_stage) {
                return Err(std::io::Error::other("injected publication failure"));
            }
            Ok(())
        };
        inject("write")
            .and_then(|()| file.write_all(&bytes))
            .map_err(|e| self.error("write", e))?;
        inject("flush")
            .and_then(|()| file.flush())
            .map_err(|e| self.error("flush", e))?;
        drop(file);
        if exclusive {
            // hard_link publishes fully written bytes and refuses an existing destination.
            fs::hard_link(&temp.path, &self.path)
                .map_err(|e| self.error("create exclusive destination", e))?;
        } else {
            inject("replace")
                .and_then(|()| fs::rename(&temp.path, &self.path))
                .map_err(|e| self.error("replace", e))?;
        }
        Ok(())
    }
}

struct Temporary {
    path: PathBuf,
}
impl Temporary {
    fn create(parent: &Path) -> std::io::Result<(Self, File)> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        for _ in 0..128 {
            let path = parent.join(format!(
                ".solve-json-{}-{}.tmp",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(file) => return Ok((Self { path }, file)),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e),
            }
        }
        Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "temporary filename attempts exhausted",
        ))
    }
}
impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn write_flush_and_replace_failures_leave_previous_json_intact() {
        for stage in ["write", "flush", "replace"] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("out.json");
            let mut publisher = JsonPublisher::create(&path, &vec![1]).unwrap();
            let before = fs::read(&path).unwrap();
            publisher.failure = Some(stage);
            let error = publisher.publish(&vec![2]).unwrap_err();
            assert_eq!(error.stage, stage);
            assert!(error.to_string().contains("injected publication failure"));
            assert_eq!(fs::read(&path).unwrap(), before);
            assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
        }
    }
}
