extern crate monolith;
extern crate tempfile;

use std::fs::File;
use std::io::{self, Write};
use std::marker::PhantomData;
use std::path::Path;

use monolith::dataset::{Load, Loader};
use monolith::io::prelude::*;
use monolith::lang::{Sentence, Token};
use monolith::preprocessing::{TextPreprocessor, Vocab};
use tempfile::NamedTempFile;

#[derive(Debug)]
pub struct Reader<R, T> {
    inner: R,
    _phantom: PhantomData<T>,
}

impl<R: io::Read, T> Reader<R, T> {
    pub fn new(inner: R) -> Self {
        Reader {
            inner: inner,
            _phantom: PhantomData,
        }
    }
}

impl<T> FileOpen for Reader<io::BufReader<File>, T> {
    fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        Ok(Self::new(io::BufReader::new(File::open(path)?)))
    }
}

impl<'a, R: io::BufRead> Read for Reader<R, Sentence<Token<'a>>> {
    type Item = Sentence<Token<'a>>;

    fn read_upto(&mut self, num: usize, buf: &mut Vec<Self::Item>) -> io::Result<usize> {
        let mut count = 0;
        let mut line = String::new();
        while count < num {
            match self.inner.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => {
                    let trimmed_line = line.trim();
                    if !trimmed_line.is_empty() {
                        buf.push(Sentence::new(trimmed_line));
                        count += 1;
                    }
                }
                Err(ref e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
            line.clear();
        }
        Ok(count)
    }
}

static RAW_TEXT: &'static str = r#"
Pierre Vinken , 61 years old , will join the board as a nonexecutive director Nov. 29 .
No , it was n't Black Monday .
John loves Mary .
"#;

#[test]
fn test_loader() {
    let mut tmpfile = NamedTempFile::new().unwrap();
    write!(tmpfile.as_mut(), "{}", RAW_TEXT).unwrap();

    let preprocessor = TextPreprocessor::new(Vocab::new());
    let mut loader = Loader::<Reader<io::BufReader<File>, Sentence<Token>>, _>::new(preprocessor);
    loader.fix();
    let dataset = loader.load(tmpfile.path()).unwrap();
    assert_eq!(dataset[0], vec![0; 18]);
    assert_eq!(dataset[1], vec![0; 8]);
    assert_eq!(dataset[2], vec![0; 4]);
    loader.unfix();
    let dataset = loader.load(tmpfile.path()).unwrap();
    assert_eq!(
        dataset[0],
        &[1, 2, 3, 4, 5, 6, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    );
    assert_eq!(dataset[1], &[18, 3, 19, 20, 21, 22, 23, 17]);
    assert_eq!(dataset[2], &[24, 25, 26, 17]);
    loader.fix();
    let dataset = loader.load(tmpfile.path()).unwrap();
    assert_eq!(
        dataset[0],
        &[1, 2, 3, 4, 5, 6, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    );
    assert_eq!(dataset[1], &[18, 3, 19, 20, 21, 22, 23, 17]);
    assert_eq!(dataset[2], &[24, 25, 26, 17]);
}
