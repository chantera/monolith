use std::io as std_io;
use std::marker::PhantomData;
use std::u32::MAX as U32_MAX;

use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use serde::{Deserialize, Serialize};
use serde::de::{DeserializeOwned, Deserializer as SerdeDeserializer};
use serde::ser::Serializer as SerdeSerializer;
use serde_json;
use rmp_serde;

use io as mod_io;
use lang::RcString;

#[derive(Debug, Clone, Copy)]
pub enum Format {
    Json,
    JsonPretty,
    Msgpack,
}

pub struct Serializer<IO, T> {
    _phantom: PhantomData<T>,
    inner: IO,
    format: Format,
}

impl<IO, T> Serializer<IO, T> {
    pub fn new(io: IO, format: Format) -> Self {
        Serializer {
            _phantom: PhantomData,
            inner: io,
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
        Format::Msgpack => {
            rmp_serde::to_vec(data).map_err(|e| {
                std_io::Error::new(std_io::ErrorKind::InvalidData, e)
            })
        }
    }
}

pub fn deserialize<'a, T: Deserialize<'a>>(bytes: &'a [u8], format: Format) -> std_io::Result<T> {
    match format {
        Format::Json | Format::JsonPretty => {
            serde_json::from_slice(bytes).map_err(|e| {
                std_io::Error::new(std_io::ErrorKind::InvalidData, e)
            })
        }
        Format::Msgpack => {
            rmp_serde::from_slice(bytes).map_err(|e| {
                std_io::Error::new(std_io::ErrorKind::InvalidData, e)
            })
        }
    }
}

impl<T: Serialize, IO: std_io::Write> mod_io::Write for Serializer<IO, T> {
    type Item = T;

    fn write(&mut self, buf: &[Self::Item]) -> std_io::Result<usize> {
        for item in buf {
            let mut bytes = self.serialize(item)?;
            let len = bytes.len();
            if len > U32_MAX as usize {
                return Err(std_io::Error::new(
                    std_io::ErrorKind::Other,
                    format!("only supports {} length byte stream", U32_MAX),
                ));
            }
            self.inner.write_u32::<BigEndian>(len as u32)?;
            bytes.push(b'\n');
            self.inner.write_all(&bytes)?;
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
        while count < num {
            let len = match self.inner.read_u32::<BigEndian>() {
                Ok(n) => n as usize,
                Err(e) => {
                    match e.kind() {
                        std_io::ErrorKind::UnexpectedEof => break,
                        _ => return Err(e),
                    }
                }
            };
            let mut data = vec![0; len + 1];
            self.inner.read_exact(&mut data)?;
            if data[len] != b'\n' {
                return Err(std_io::Error::new(
                    std_io::ErrorKind::InvalidData,
                    "broken data",
                ));
            }
            buf.push(self.deserialize(&data[..len])?);
            count += 1;
        }
        Ok(count)
    }
}

impl<T, IO: std_io::Seek> std_io::Seek for Serializer<IO, T> {
    fn seek(&mut self, pos: std_io::SeekFrom) -> std_io::Result<u64> {
        self.inner.seek(pos)
    }
}

impl Serialize for RcString {
    #[inline]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: SerdeSerializer,
    {
        serializer.serialize_str(self)
    }
}

impl<'de> Deserialize<'de> for RcString {
    fn deserialize<D>(deserializer: D) -> Result<RcString, D::Error>
    where
        D: SerdeDeserializer<'de>,
    {
        String::deserialize(deserializer).map(|s| RcString::new(s.to_string()))
    }
}
