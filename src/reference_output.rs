//! Shared native and scalar publication primitives for fixed reference bundles.
use crate::itebd_auto::ItebdState;
use anyhow::{ensure, Context, Result};
use std::{
    fs,
    io::Write,
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
};
pub(crate) fn exact_state(original: &ItebdState, restored: &ItebdState) -> Result<()> {
    use crate::itebd_state_view::ItebdStateRef;
    ensure!(
        matches!(
            (original, restored),
            (ItebdState::Real(_), ItebdState::Real(_))
        ),
        "terminal checkpoint backend differs"
    );
    let a = ItebdStateRef::from(original);
    let b = ItebdStateRef::from(restored);
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    for (x, y) in [(a.a(), b.a()), (a.b(), b.b())] {
        ensure!(
            x.gamma.indices == y.gamma.indices
                && bits(&x.gamma.to_vec::<f64>()?) == bits(&y.gamma.to_vec::<f64>()?),
            "terminal tensor exact-bit round-trip differs"
        );
    }
    ensure!(
        bits(a.lambda_ab()) == bits(b.lambda_ab()) && bits(a.lambda_ba()) == bits(b.lambda_ba()),
        "terminal Schmidt exact-bit round-trip differs"
    );
    Ok(())
}
pub(crate) fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    // Staging is a sibling of the reserved directory, never a completed bundle member.
    static SERIAL: AtomicU64 = AtomicU64::new(0);
    let parent = path.parent().context("publication path has no parent")?;
    let staging_parent = parent
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let staging = staging_parent.join(format!(
        ".reference-{}-{}.pending",
        std::process::id(),
        SERIAL.fetch_add(1, Ordering::Relaxed)
    ));
    let result = (|| -> Result<()> {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&staging)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&staging, path)?;
        Ok(())
    })();
    result.with_context(|| {
        format!(
            "publish {} (pending evidence {})",
            path.display(),
            staging.display()
        )
    })
}
pub(crate) fn quote_csv(value: &str) -> String {
    if value.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.into()
    }
}
pub(crate) fn parse_csv(text: &str) -> Result<Vec<Vec<String>>> {
    let mut rows = Vec::new();
    let mut row = Vec::new();
    let mut field = String::new();
    let mut chars = text.chars().peekable();
    let mut quoted = false;
    let mut closed = false;
    while let Some(c) = chars.next() {
        if quoted {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    field.push('"');
                } else {
                    quoted = false;
                    closed = true;
                }
            } else {
                field.push(c);
            }
        } else {
            match c {
                '"' if field.is_empty() && !closed => quoted = true,
                ',' => {
                    row.push(std::mem::take(&mut field));
                    closed = false;
                }
                '\n' | '\r' => {
                    if c == '\r' {
                        ensure!(chars.next() == Some('\n'), "CSV bare carriage return");
                    }
                    row.push(std::mem::take(&mut field));
                    rows.push(std::mem::take(&mut row));
                    closed = false;
                }
                _ => {
                    ensure!(!closed && c != '"', "CSV malformed quotation");
                    field.push(c);
                }
            }
        }
    }
    ensure!(!quoted, "CSV unterminated quotation");
    if !row.is_empty() || !field.is_empty() || closed {
        row.push(field);
        rows.push(row);
    }
    Ok(rows)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn csv_quotes_and_real_publication_errors() -> Result<()> {
        let text = format!(
            "a,b\r\n{},{}\n",
            quote_csv("comma,quote\""),
            quote_csv("two\nlines")
        );
        assert_eq!(
            parse_csv(&text)?,
            vec![vec!["a", "b"], vec!["comma,quote\"", "two\nlines"]]
        );
        for bad in ["\"unclosed", "\"closed\"x\n", "bare\rnext"] {
            assert!(parse_csv(bad).is_err());
        }
        let temp = tempfile::tempdir()?;
        let file = temp.path().join("blocker");
        std::fs::write(&file, b"keep")?;
        let error = atomic_write(&file.join("child"), b"new").unwrap_err();
        assert!(format!("{error:#}").contains("publish"));
        assert_eq!(std::fs::read(file)?, b"keep");
        Ok(())
    }
}
