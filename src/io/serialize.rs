use std::fs;
use std::io::{self as std_io, Read, Write};
use std::path::Path;

use rmp_serde;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json;

#[derive(Debug, Clone, Copy)]
pub enum Format {
    Json,
    JsonPretty,
    Msgpack,
}

pub fn serialize<T: Serialize>(data: &T, format: Format) -> std_io::Result<Vec<u8>> {
    match format {
        Format::Json => match serde_json::to_string(data) {
            Ok(s) => Ok(s.into_bytes()),
            Err(e) => Err(std_io::Error::new(std_io::ErrorKind::InvalidData, e)),
        },
        Format::JsonPretty => match serde_json::to_string_pretty(data) {
            Ok(s) => Ok(s.into_bytes()),
            Err(e) => Err(std_io::Error::new(std_io::ErrorKind::InvalidData, e)),
        },
        Format::Msgpack => rmp_serde::to_vec(data)
            .map_err(|e| std_io::Error::new(std_io::ErrorKind::InvalidData, e)),
    }
}

pub fn write_to<P: AsRef<Path>, T: Serialize>(
    data: &T,
    path: P,
    format: Format,
) -> std_io::Result<()> {
    let file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    let mut writer = std_io::BufWriter::new(file);
    writer.write(&serialize(data, format)?)?;
    writer.flush()
}

pub fn deserialize<'a, T: Deserialize<'a>>(bytes: &'a [u8], format: Format) -> std_io::Result<T> {
    match format {
        Format::Json | Format::JsonPretty => serde_json::from_slice(bytes)
            .map_err(|e| std_io::Error::new(std_io::ErrorKind::InvalidData, e)),
        Format::Msgpack => rmp_serde::from_slice(bytes)
            .map_err(|e| std_io::Error::new(std_io::ErrorKind::InvalidData, e)),
    }
}

pub fn read_from<P: AsRef<Path>, T: DeserializeOwned>(
    path: P,
    format: Format,
) -> std_io::Result<T> {
    let mut reader = std_io::BufReader::new(fs::File::open(path)?);
    let mut buf = Vec::new();
    reader.read_to_end(&mut buf)?;
    deserialize(&buf, format)
}
