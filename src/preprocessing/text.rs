use std::borrow::Borrow;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

use lang::RcString;

#[derive(Debug)]
pub struct Vocab {
    s2i: HashMap<RcString, u32>,
    i2s: Vec<RcString>,
    freq: Vec<u32>,
    embeddings: Option<Vec<Vec<f32>>>,
}

const DEFAULT_CAPACITY: usize = 32;
static UNKNOWN_TOKEN: &'static str = "<UNK>";

impl Vocab {
    pub fn new() -> Self {
        Self::with_capacity_and_default_token(DEFAULT_CAPACITY, UNKNOWN_TOKEN.to_string())
    }

    pub fn from_file<P: AsRef<Path>, S: Into<String>>(
        file: P,
        default_token: S,
    ) -> Result<Self, io::Error> {
        // @TODO: support following formats:
        // 1. vocab: word per line
        //  (example)
        //    will
        //    their
        // 2. vocab with embeddings: word and embedding per line
        //  (example)
        //    will 0.81544 0.30171 0.5472 0.46581 ...
        //    their 0.41519 0.13167 -0.0569 -0.56765 ...
        // 3. vocab and embeddings: word and embeddings (separated)
        //  (example)
        //  <vocab file>
        //    will
        //    their
        //  <embeddings file>
        //    0.81544 0.30171 0.5472 0.46581 ...
        //    0.41519 0.13167 -0.0569 -0.56765 ...
        //
        // currently supports the format `2` only.

        let (words, embeddings) = load_embeddings(file, " ")?;
        let mut v = Self::with_capacity_and_default_token(words.len(), default_token.into());
        words.into_iter().for_each(|word| { v.add(word); });
        v.embeddings = Some(embeddings);
        Ok(v)
    }

    pub fn with_default_token(default_token: String) -> Self {
        Self::with_capacity_and_default_token(DEFAULT_CAPACITY, default_token)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_default_token(capacity, UNKNOWN_TOKEN.to_string())
    }

    pub fn with_capacity_and_default_token(capacity: usize, default_token: String) -> Self {
        let mut v = Vocab {
            s2i: HashMap::with_capacity(capacity),
            i2s: Vec::with_capacity(capacity),
            freq: Vec::with_capacity(capacity),
            embeddings: None,
        };
        v.add(default_token);
        v
    }

    pub fn add(&mut self, word: String) -> u32 {
        match self.s2i.get(&word[..]) {
            Some(v) => {
                let id = *v;
                if id > 0 {
                    self.freq[id as usize] += 1;
                }
                return id;
            }
            _ => {}
        }
        let id = self.i2s.len() as u32;
        let rc = RcString::new(word);
        self.i2s.push(rc.clone());
        self.s2i.insert(rc, id);
        self.freq.push(0);
        id
    }

    pub fn get<Q: Borrow<str> + ?Sized>(&self, word: &Q) -> u32 {
        self.s2i.get(word.borrow()).map(|v| *v).unwrap_or_else(|| 0)
    }

    pub fn freq(&self, id: u32) -> Option<u32> {
        self.freq.get(id as usize).map(|v| *v)
    }

    pub fn lookup(&self, id: u32) -> Option<&str> {
        self.i2s.get(id as usize).map(|v| v.as_str())
    }

    pub fn size(&self) -> usize {
        self.i2s.len()
    }

    pub fn embed(&self) -> Option<&Vec<Vec<f32>>> {
        self.embeddings.as_ref()
    }
}

pub fn load_embeddings<P: AsRef<Path>>(
    file: P,
    delimiter: &str,
) -> Result<(Vec<String>, Vec<Vec<f32>>), io::Error> {
    let reader = io::BufReader::new(File::open(file)?);
    let mut vocab = vec![];
    let mut embeddings = vec![];
    for line in reader.lines() {
        let l = line?;
        let mut cols = l.split(delimiter);
        let mut embedding = vec![];
        match cols.next() {
            Some(word) => {
                vocab.push(word.to_string());
            }
            None => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "A word must be placed first in a line",
                ));
            }
        }
        for value in cols {
            match value.parse::<f32>() {
                Ok(v) => {
                    embedding.push(v);
                }
                Err(e) => {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, e));
                }
            }
        }
        embeddings.push(embedding);
    }
    Ok((vocab, embeddings))
}

trait Tokenize {
    fn tokenize(sentence: &str) -> Vec<String>;
}
