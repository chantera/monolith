use std::error;
use std::fs::File;
use std::io;
use std::marker::PhantomData;
use std::path::Path;
use std::usize::MAX as USIZE_MAX;

pub mod prelude;

pub trait Read {
    type Item;

    fn read(&mut self, buf: &mut Vec<Self::Item>) -> io::Result<usize> {
        self.read_upto(USIZE_MAX, buf)
    }

    fn read_upto(&mut self, num: usize, buf: &mut Vec<Self::Item>) -> io::Result<usize>;
}

pub trait FromLine: Sized {
    type Err: Into<Box<error::Error + Send + Sync>>;

    fn from_line(line: &str) -> Result<Self, Self::Err>;
}

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

    #[inline]
    pub fn inner(&self) -> &R {
        &self.inner
    }

    #[inline]
    pub fn inner_mut(&mut self) -> &mut R {
        &mut self.inner
    }
}

impl<T> Reader<io::BufReader<File>, T> {
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        Ok(Self::new(io::BufReader::new(File::open(path)?)))
    }
}

impl<R: io::Seek, T> io::Seek for Reader<R, T> {
    fn seek(&mut self, pos: io::SeekFrom) -> io::Result<u64> {
        self.inner.seek(pos)
    }
}

impl<R: io::BufRead, T: FromLine> Read for Reader<R, T> {
    type Item = T;

    fn read_upto(&mut self, num: usize, buf: &mut Vec<Self::Item>) -> io::Result<usize> {
        read_upto(&mut self.inner, num, buf)
    }
}

pub fn read_upto<R: io::BufRead, T: FromLine>(
    reader: &mut R,
    num: usize,
    buf: &mut Vec<T>,
) -> io::Result<usize> {
    let mut count = 0;
    let mut line = String::new();
    while count < num {
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {
                buf.push(try!(T::from_line(&line).map_err(|e| {
                    io::Error::new(io::ErrorKind::InvalidData, e)
                })));
                count += 1;
            }
            Err(ref e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
        line.clear();
    }
    Ok(count)
}
