use std::io::Result as IOResult;
use std::marker::PhantomData;
use std::ops;
use std::path::Path;
use std::slice::Iter;
use std::usize::MAX as USIZE_MAX;

use io::{BufFileReader, FileOpen, Read};
use preprocessing::Preprocess;

#[cfg(feature = "dataset-conll")]
pub mod conll;

#[derive(Debug)]
pub struct Dataset<T> {
    items: Vec<T>,
}

impl<T> Dataset<T> {
    pub fn new() -> Self {
        Dataset { items: vec![] }
    }

    pub fn from_items(items: Vec<T>) -> Self {
        Dataset { items: items }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Dataset { items: Vec::with_capacity(capacity) }
    }

    // pub fn batches(size: usize, shuffle: bool) -> Batches {}
    // TODO: implement

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn iter(&self) -> Iter<T> {
        self.items.iter()
    }
}

impl<T> ops::Index<usize> for Dataset<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &T {
        ops::Index::index(&self.items, index)
    }
}

impl<T> ops::IndexMut<usize> for Dataset<T> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut T {
        ops::IndexMut::index_mut(&mut self.items, index)
    }
}

impl<T> ops::Index<ops::Range<usize>> for Dataset<T> {
    type Output = [T];

    #[inline]
    fn index(&self, index: ops::Range<usize>) -> &[T] {
        ops::Index::index(&self.items, index)
    }
}

impl<T> ops::Index<ops::RangeTo<usize>> for Dataset<T> {
    type Output = [T];

    #[inline]
    fn index(&self, index: ops::RangeTo<usize>) -> &[T] {
        ops::Index::index(&self.items, index)
    }
}

impl<T> ops::Index<ops::RangeFrom<usize>> for Dataset<T> {
    type Output = [T];

    #[inline]
    fn index(&self, index: ops::RangeFrom<usize>) -> &[T] {
        ops::Index::index(&self.items, index)
    }
}

impl<T> ops::Index<ops::RangeFull> for Dataset<T> {
    type Output = [T];

    #[inline]
    fn index(&self, _index: ops::RangeFull) -> &[T] {
        &self.items
    }
}

impl<T> ops::IndexMut<ops::Range<usize>> for Dataset<T> {
    #[inline]
    fn index_mut(&mut self, index: ops::Range<usize>) -> &mut [T] {
        ops::IndexMut::index_mut(&mut self.items, index)
    }
}

impl<T> ops::IndexMut<ops::RangeTo<usize>> for Dataset<T> {
    #[inline]
    fn index_mut(&mut self, index: ops::RangeTo<usize>) -> &mut [T] {
        ops::IndexMut::index_mut(&mut self.items, index)
    }
}

impl<T> ops::IndexMut<ops::RangeFrom<usize>> for Dataset<T> {
    #[inline]
    fn index_mut(&mut self, index: ops::RangeFrom<usize>) -> &mut [T] {
        ops::IndexMut::index_mut(&mut self.items, index)
    }
}

impl<T> ops::IndexMut<ops::RangeFull> for Dataset<T> {
    #[inline]
    fn index_mut(&mut self, _index: ops::RangeFull) -> &mut [T] {
        &mut self.items
    }
}

pub trait Load {
    type Item;

    fn load<P: AsRef<Path>>(&mut self, file: P) -> IOResult<Dataset<Self::Item>> {
        self.load_until(file, USIZE_MAX)
    }

    fn load_until<P: AsRef<Path>>(&mut self, file: P, size: usize)
        -> IOResult<Dataset<Self::Item>>;
}

pub struct Loader<R, P> {
    _reader: PhantomData<R>,
    preprocessor: P,
    enable_fit: bool,
}

impl<T, R: Read<Item = T>, P: Preprocess<T>> Loader<R, P> {
    pub fn new(preprocessor: P) -> Self {
        Loader {
            _reader: PhantomData,
            preprocessor: preprocessor,
            enable_fit: true,
        }
    }

    pub fn unfix(&mut self) {
        self.enable_fit = true;
    }

    pub fn fix(&mut self) {
        self.enable_fit = false;
    }
}

impl<T, P: Preprocess<T>, R: FileOpen + Read<Item = T>> Load for Loader<R, P> {
    type Item = P::Output;

    fn load_until<PATH: AsRef<Path>>(
        &mut self,
        file: PATH,
        size: usize,
    ) -> IOResult<Dataset<Self::Item>> {
        let mut reader = R::open(file)?;
        let mut buf = vec![];
        reader.read_upto(size, &mut buf)?;
        let items = if self.enable_fit {
            self.preprocessor.fit_transform(buf.into_iter())
        } else {
            self.preprocessor.transform(buf.into_iter())
        };
        Ok(Dataset::from_items(items))
    }
}

pub type StdLoader<T, P> = Loader<BufFileReader<T>, P>;
