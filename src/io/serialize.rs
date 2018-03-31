use std::cmp;
use std::io as std_io;
use std::marker::PhantomData;

use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;
use serde_json;

use io as mod_io;

#[derive(Debug, Clone, Copy)]
pub enum Format {
    Json,
    JsonPretty,
    Msgpack,
}

pub struct Serializer<IO, T> {
    _phantom: PhantomData<T>,
    inner: IO,
    buf: Vec<String>,
    format: Format,
}

impl<IO, T> Serializer<IO, T> {
    pub fn new(io: IO, format: Format) -> Self {
        Serializer {
            _phantom: PhantomData,
            inner: io,
            buf: vec![],
            format: format,
        }
    }

    pub fn inner(&self) -> &IO {
        &self.inner
    }
}

impl<IO, T: Serialize> Serializer<IO, T> {
    pub fn serialize(&self, data: &T) -> std_io::Result<Vec<u8>> {
        serialize(data, self.format)
    }
}

impl<'a, IO, T: Deserialize<'a>> Serializer<IO, T> {
    pub fn deserialize(&self, bytes: &'a [u8]) -> std_io::Result<T> {
        deserialize(bytes, self.format)
    }
}

pub fn serialize<T: Serialize>(data: &T, format: Format) -> std_io::Result<Vec<u8>> {
    match format {
        Format::Json => {
            match serde_json::to_string(data) {
                Ok(s) => Ok(s.into_bytes()),
                Err(e) => Err(std_io::Error::new(std_io::ErrorKind::InvalidData, e)),
            }
        }
        Format::JsonPretty => {
            match serde_json::to_string_pretty(data) {
                Ok(s) => Ok(s.into_bytes()),
                Err(e) => Err(std_io::Error::new(std_io::ErrorKind::InvalidData, e)),
            }
        }
        Format::Msgpack => Err(std_io::Error::new(
            std_io::ErrorKind::Other,
            "Not Supported",
        )),
    }
}

pub fn deserialize<'a, T: Deserialize<'a>>(bytes: &'a [u8], format: Format) -> std_io::Result<T> {
    match format {
        Format::Json | Format::JsonPretty => {
            serde_json::from_slice(bytes).map_err(|e| {
                std_io::Error::new(std_io::ErrorKind::InvalidData, e)
            })
        }
        Msgpack => Err(std_io::Error::new(
            std_io::ErrorKind::Other,
            "Not Supported",
        )),
    }
}

impl<T: Serialize, IO: std_io::Write> mod_io::Write for Serializer<IO, T> {
    type Item = T;

    fn write(&mut self, buf: &[Self::Item]) -> std_io::Result<usize> {
        for item in buf {
            let mut bytes = self.serialize(item)?;
            bytes.push(b'\n');
            self.inner.write(&bytes);
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> std_io::Result<()> {
        self.inner.flush()
    }
}

impl<T: DeserializeOwned, IO: std_io::Read> mod_io::Read for Serializer<IO, T> {
    type Item = T;

    fn read_upto(&mut self, num: usize, buf: &mut Vec<Self::Item>) -> std_io::Result<usize> {
        let mut count = 0;
        let mut w_buf = Vec::new();
        let mut r_buf = vec![0; 1024];
        let mut pos = 0;
        let mut cap = 0;
        while count < num {
            // if already consumed, then read next buffer.
            if pos >= cap {
                debug_assert!(pos == cap);
                cap = self.inner.read(&mut r_buf)?;
                pos = 0;
            }
            // search from the current buffer.
            let available = &r_buf[pos..cap];
            let used = match available.iter().position(|&b| b == b'\n') {
                Some(i) => {
                    w_buf.extend_from_slice(&available[..i]);
                    buf.push(self.deserialize(&w_buf)?);
                    count += 1;
                    w_buf.clear();
                    i + 1
                }
                None => {
                    w_buf.extend_from_slice(available);
                    available.len()
                }
            };
            pos = cmp::min(pos + used, cap); // consumed
            if used == 0 {
                break;
            }
        }
        Ok(count)
    }
}

impl<T, IO: std_io::Seek> std_io::Seek for Serializer<IO, T> {
    fn seek(&mut self, pos: std_io::SeekFrom) -> std_io::Result<u64> {
        self.inner.seek(pos)
    }
}
