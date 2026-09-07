#[path = "../src/solve_run/output.rs"]
mod output;

use output::JsonPublisher;
use serde::Serialize;

#[test]
fn exclusive_creation_and_atomic_updates_preserve_previous_results() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("nested/result.json");
    let mut publisher = JsonPublisher::create(&path, &vec![1, 2]).unwrap();
    let initial = std::fs::read(&path).unwrap();
    assert!(JsonPublisher::create(&path, &vec![99]).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), initial);
    publisher.publish(&vec![3, 4]).unwrap();
    assert_eq!(
        serde_json::from_slice::<Vec<i32>>(&std::fs::read(&path).unwrap()).unwrap(),
        vec![3, 4]
    );
}

struct Invalid;
impl Serialize for Invalid {
    fn serialize<S: serde::Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
        Err(serde::ser::Error::custom(
            "deliberate serialization failure",
        ))
    }
}

#[test]
fn serialization_failure_preserves_published_bytes_and_cleans_temporary_files() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("result.json");
    let mut publisher = JsonPublisher::create(&path, &vec![7]).unwrap();
    let before = std::fs::read(&path).unwrap();
    let error = publisher.publish(&Invalid).unwrap_err().to_string();
    assert!(error.contains("serialize") && error.contains("deliberate serialization failure"));
    assert!(error.contains(path.to_str().unwrap()));
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
}

#[test]
fn actual_parent_and_replacement_failures_have_context() {
    let dir = tempfile::tempdir().unwrap();
    let parent = dir.path().join("file");
    std::fs::write(&parent, b"existing").unwrap();
    let path = parent.join("result.json");
    let error = JsonPublisher::create(&path, &1).err().unwrap();
    let cause = std::error::Error::source(&error)
        .unwrap()
        .downcast_ref::<std::io::Error>()
        .unwrap();
    assert!(matches!(
        cause.kind(),
        std::io::ErrorKind::AlreadyExists | std::io::ErrorKind::NotADirectory
    ));
    assert!(error.to_string().contains(&cause.to_string()));
    let error = error.to_string();
    assert!(
        error.contains(path.to_str().unwrap()) && error.contains("parent"),
        "{error}"
    );
    assert_eq!(std::fs::read(&parent).unwrap(), b"existing");
    let path = dir.path().join("result.json");
    let mut publisher = JsonPublisher::create(&path, &1).unwrap();
    std::fs::remove_file(&path).unwrap();
    std::fs::create_dir(&path).unwrap();
    let error = publisher.publish(&2).unwrap_err();
    let cause = std::error::Error::source(&error)
        .unwrap()
        .downcast_ref::<std::io::Error>()
        .unwrap();
    assert!(cause.raw_os_error().is_some());
    assert!(error.to_string().contains(&cause.to_string()));
    let error = error.to_string();
    assert!(
        error.contains("replace") && error.contains(path.to_str().unwrap()),
        "{error}"
    );
    assert!(path.is_dir());
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 2);
}
