use std::borrow::Borrow;
use std::collections::HashMap;

use lang::RcString;

#[derive(Debug)]
pub struct Vocab {
    s2i: HashMap<RcString, u32>,
    i2s: Vec<RcString>,
    freq: Vec<u32>,
}

const DEFAULT_CAPACITY: usize = 32;
static UNKNOWN_TOKEN: &'static str = "<UNK>";

impl Vocab {
    pub fn new() -> Self {
        Self::with_capacity_and_default_token(DEFAULT_CAPACITY, UNKNOWN_TOKEN.to_string())
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
}


// trait Tokenize {
//     fn tokenize(sentence: &str) -> Vec<> {
//     }
// }
//
// fn tokenize()

// struct VocaburaryPreprocessor<F> where F: Fn() -> Vec<&str> {
//     tokenizer: F
// }
//
// struct EmbeddingPreprocessor {}
