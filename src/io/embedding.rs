use std::fs::File;
use std::io as std_io;
use std::path::Path;

use csv;

#[derive(Debug, Deserialize)]
struct EmbedRecord {
    word: String,
    value: Vec<f32>,
}

impl Into<(String, Vec<f32>)> for EmbedRecord {
    fn into(self) -> (String, Vec<f32>) {
        (self.word, self.value)
    }
}

const DEFAULT_CAPACITY: usize = 400000;

pub fn load_embeddings<P: AsRef<Path>>(
    file: P,
    delimiter: u8,
    has_header: bool,
) -> Result<Vec<(String, Vec<f32>)>, std_io::Error> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(has_header)
        .delimiter(delimiter)
        .quoting(false)
        .from_reader(File::open(file)?);
    let mut entries = Vec::with_capacity(DEFAULT_CAPACITY);
    for result in reader.deserialize() {
        let record: EmbedRecord = result?;
        entries.push(record.into());
    }
    Ok(entries)
}
