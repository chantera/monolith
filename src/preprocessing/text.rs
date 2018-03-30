use std::borrow::Borrow;
use std::collections::HashMap;
use std::io as std_io;
use std::path::Path;

use io::embedding as embed_io;
use lang::RcString;
use rand::thread_rng;
use rand::distributions;
use rand::distributions::range::RangeImpl;

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
    ) -> Result<Self, std_io::Error> {
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

        let mut entries = embed_io::load_embeddings(file, b' ', false)?;
        let capacity = entries.len() + 1;
        assert!(capacity > 1);
        let mut embeddings = Vec::with_capacity(capacity);

        let default_token = default_token.into();
        let default = entries.iter().position(
            |ref entry| entry.0 == default_token,
        );
        match default {
            Some(index) => {
                let entry = entries.remove(index);
                embeddings.push(entry.1);
            }
            None => {
                let dim = entries[0].1.len();
                let uniform = distributions::range::RangeFloat::<f32>::new(-1.0, 1.0);
                let mut value = Vec::with_capacity(dim);
                let mut rng = thread_rng();
                for _ in 0..dim {
                    value.push(uniform.sample(&mut rng));
                }
                embeddings.push(value);
            }
        }

        let mut v = Self::with_capacity_and_default_token(capacity, default_token);
        for entry in entries.into_iter() {
            v.add(entry.0);
            embeddings.push(entry.1);
        }
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

trait Tokenize {
    fn tokenize(sentence: &str) -> Vec<String>;
}
