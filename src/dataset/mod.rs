use std::cmp::min;
use std::io::Result as IOResult;
use std::marker::PhantomData;
use std::ops;
use std::path::Path;
use std::slice::Iter;
use std::usize::MAX as USIZE_MAX;

use io::{BufFileReader, FileOpen, Read};
use preprocessing::Preprocess;
use utils::rand::{Rng, thread_rng};

#[cfg(feature = "dataset-conll")]
pub mod conll;

pub struct Batches<'a, T: 'a> {
    batch_size: usize,
    offset: usize,
    samples: &'a [T],
    n_samples: usize,
    indices: Vec<usize>,
}

impl<'a, T> Iterator for Batches<'a, T> {
    type Item = Vec<&'a T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.offset < self.n_samples {
            let batch_indices =
                &self.indices[self.offset..min(self.offset + self.batch_size, self.n_samples)];
            let batch: Vec<&'a T> = batch_indices
                .iter()
                .map(|&index| &self.samples[index])
                .collect();
            self.offset += self.batch_size;
            Some(batch)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remain = self.n_samples - self.offset;
        let bound = (remain as f64 / self.batch_size as f64).ceil() as usize;
        (bound, Some(bound))
    }

    #[inline]
    fn count(self) -> usize {
        (self.n_samples as f64 / self.batch_size as f64).ceil() as usize
    }
}

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

    pub fn batch<'a>(&'a self, size: usize, shuffle: bool) -> Batches<'a, T> {
        let mut indices = (0..self.items.len()).collect::<Vec<_>>();
        if shuffle {
            thread_rng().shuffle(&mut indices);
        }
        Batches {
            batch_size: size,
            offset: 0,
            samples: &self.items,
            n_samples: self.items.len(),
            indices: indices,
        }
    }

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

#[macro_export]
macro_rules! sort_batch {
    ($var:ident, $col:tt) => {
        $var.sort_by(|a, b| b.$col.len().cmp(&a.$col.len()));
    };
    ($var:ident) => {
        sort_batch!($var, 0);
    };
}

#[macro_export]
macro_rules! take_cols {
    (($($name:ident:$col:tt),+); $var:ident, $cap:expr) => {
        $(
            let mut $name = Vec::with_capacity($cap);
        )+
        for entry in $var.into_iter() {
            $(
                $name.push(&entry.$col);
            )+
        }
    };
    (($($name:ident:$col:tt),+); $var:ident) => {
        $(
            let mut $name = Vec::new();
        )+
        for entry in $var.into_iter() {
            $(
                $name.push(&entry.$col);
            )+
        }
    };
}

#[macro_export]
macro_rules! transpose {
    ($($var:ident),+) => {
        $(
            let $var = monolith::dataset::transpose_sequence($var, None);
        )+
    };
}

pub fn transpose_sequence<Batch, Seq, T: Clone>(xs: Batch, pad: Option<T>) -> Vec<Vec<T>>
where
    Batch: AsRef<[Seq]>,
    Seq: AsRef<[T]>,
{
    let batch_size = xs.as_ref().len();
    let max_len = xs.as_ref()[0].as_ref().len();
    match pad {
        Some(p) => {
            let mut ys = vec![vec![p; batch_size]; max_len];
            for (sample_idx, x) in xs.as_ref().iter().enumerate() {
                for (t, x_t) in x.as_ref().iter().enumerate() {
                    ys[t][sample_idx] = x_t.clone();
                }
            }
            ys
        }
        None => {
            let mut ys = vec![Vec::with_capacity(batch_size); max_len];
            for x in xs.as_ref().iter() {
                for (t, x_t) in x.as_ref().iter().enumerate() {
                    ys[t].push(x_t.clone());
                }
            }
            ys
        }
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

#[derive(Debug)]
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

    pub fn preprocessor(&self) -> &P {
        &self.preprocessor
    }

    pub fn dispose(self) -> P {
        self.preprocessor
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

#[cfg(feature = "serialize")]
mod serialize {
    use std::fmt;
    use std::marker::PhantomData;

    use serde::{Deserialize, Serialize};
    use serde::ser::{Serializer, SerializeStruct};
    use serde::de;

    use super::Loader;

    impl<R, SP: Serialize> Serialize for Loader<R, SP> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let mut s = serializer.serialize_struct("Loader", 3)?;
            s.serialize_field("_reader", &self._reader)?;
            s.serialize_field("preprocessor", &self.preprocessor)?;
            s.serialize_field("enable_fit", &self.enable_fit)?;
            s.end()
        }
    }

    impl<R, SP: Serialize> Loader<R, SP> {
        pub fn hello(&self) {
            println!("Hello World!");
        }
    }

    impl<'de, R, DP: Deserialize<'de>> Deserialize<'de> for Loader<R, DP> {
        fn deserialize<D: de::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            deserializer.deserialize_struct("Loader", LOADER_FIELDS, LoaderVisitor::<R, DP>::new())
        }
    }

    const LOADER_FIELDS: &'static [&'static str] = &["_reader", "preprocessor", "enable_fit"];

    struct LoaderVisitor<R, P> {
        _reader: PhantomData<R>,
        _preprocessor: PhantomData<P>,
    }

    impl<R, P> LoaderVisitor<R, P> {
        fn new() -> Self {
            LoaderVisitor {
                _reader: PhantomData,
                _preprocessor: PhantomData,
            }
        }
    }

    impl<'de, R, DP: Deserialize<'de>> de::Visitor<'de> for LoaderVisitor<R, DP> {
        type Value = Loader<R, DP>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("struct Loader")
        }

        fn visit_seq<V>(self, mut seq: V) -> Result<Loader<R, DP>, V::Error>
        where
            V: de::SeqAccess<'de>,
        {
            let reader = seq.next_element()?.ok_or_else(
                || de::Error::invalid_length(0, &self),
            )?;
            let preprocessor = seq.next_element()?.ok_or_else(
                || de::Error::invalid_length(1, &self),
            )?;
            let enable_fit = seq.next_element()?.ok_or_else(
                || de::Error::invalid_length(2, &self),
            )?;
            let loader = Loader {
                _reader: reader,
                preprocessor: preprocessor,
                enable_fit: enable_fit,
            };
            Ok(loader)
        }

        fn visit_map<V>(self, mut map: V) -> Result<Loader<R, DP>, V::Error>
        where
            V: de::MapAccess<'de>,
        {
            let mut reader = None;
            let mut preprocessor = None;
            let mut enable_fit = None;
            while let Some(key) = map.next_key()? {
                match key {
                    LoaderField::Reader => {
                        if reader.is_some() {
                            return Err(de::Error::duplicate_field("_reader"));
                        }
                        reader = Some(map.next_value()?);
                    }
                    LoaderField::Preprocessor => {
                        if preprocessor.is_some() {
                            return Err(de::Error::duplicate_field("preprocessor"));
                        }
                        preprocessor = Some(map.next_value()?);
                    }
                    LoaderField::EnableFit => {
                        if enable_fit.is_some() {
                            return Err(de::Error::duplicate_field("enable_fit"));
                        }
                        enable_fit = Some(map.next_value()?);
                    }
                }
            }
            let reader = reader.ok_or_else(|| de::Error::missing_field("reader"))?;
            let preprocessor = preprocessor.ok_or_else(
                || de::Error::missing_field("preprocessor"),
            )?;
            let enable_fit = enable_fit.ok_or_else(
                || de::Error::missing_field("enable_fit"),
            )?;
            let loader = Loader {
                _reader: reader,
                preprocessor: preprocessor,
                enable_fit: enable_fit,
            };
            Ok(loader)
        }
    }

    enum LoaderField {
        Reader,
        Preprocessor,
        EnableFit,
    }

    impl<'de> Deserialize<'de> for LoaderField {
        fn deserialize<D>(deserializer: D) -> Result<LoaderField, D::Error>
        where
            D: de::Deserializer<'de>,
        {
            deserializer.deserialize_identifier(LoaderFieldVisitor)
        }
    }

    struct LoaderFieldVisitor;

    impl<'de> de::Visitor<'de> for LoaderFieldVisitor {
        type Value = LoaderField;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("`_reader`, `preprocessor` or `enable_fit`")
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            match value {
                "_reader" => Ok(LoaderField::Reader),
                "preprocessor" => Ok(LoaderField::Preprocessor),
                "enable_fit" => Ok(LoaderField::EnableFit),
                _ => Err(de::Error::unknown_field(value, LOADER_FIELDS)),
            }
        }
    }
}
