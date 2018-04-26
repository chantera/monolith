use std::borrow::{Borrow, Cow};
use std::collections::HashMap;
use std::error;
use std::fmt;
#[cfg(feature = "serialize")]
use std::io as std_io;
#[cfg(feature = "serialize")]
use std::path::Path;

#[cfg(feature = "app")]
use io::cache::{self, FromCache, IntoCache};
#[cfg(feature = "serialize")]
use io::embedding as embed_io;
use lang::RcString;
#[cfg(feature = "serialize")]
use rand::distributions::{self, Distribution};
#[cfg(feature = "app")]
use uuid::{Uuid, NAMESPACE_OID as UUID_NAMESPACE_OID};

#[cfg(feature = "serialize")]
use utils::rand::thread_rng;

#[derive(Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct Vocab {
    s2i: HashMap<RcString, u32>,
    i2s: Vec<RcString>,
    freq: Vec<usize>,
    embeddings: Option<Vec<Vec<f32>>>,
}

const DEFAULT_CAPACITY: usize = 32;
static UNKNOWN_TOKEN: &'static str = "<UNK>";

impl Vocab {
    pub fn new() -> Self {
        Self::with_capacity_and_default_token(DEFAULT_CAPACITY, UNKNOWN_TOKEN)
    }

    #[cfg(feature = "serialize")]
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
        debug_assert!(capacity > 1);
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
                let uniform = distributions::Uniform::from(-1.0..1.0);
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
        v.embeddings = Some(embeddings);
        Ok(v)
    }

    #[cfg(feature = "app")]
    pub fn from_cache_or_file<P: AsRef<Path>, S: Into<String>>(
        file: P,
        default_token: S,
    ) -> Result<Self, std_io::Error> {
        let s = default_token.into();
        let hash = Uuid::new_v5(
            &UUID_NAMESPACE_OID,
            &format!("{}={}", file.as_ref().to_str().unwrap(), s),
        ).to_string();
        if Vocab::has_cache(&hash) {
            Vocab::from_cache(&hash)
        } else {
            let v = Vocab::from_file(file, s)?;
            Vocab::into_cache(&v, &hash)?;
            Ok(v)
        }
    }

    pub fn with_default_token<S: Into<String>>(default_token: S) -> Self {
        Self::with_capacity_and_default_token(DEFAULT_CAPACITY, default_token)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_default_token(capacity, UNKNOWN_TOKEN)
    }

    pub fn with_capacity_and_default_token<S: Into<String>>(
        capacity: usize,
        default_token: S,
    ) -> Self {
        let mut v = Vocab {
            s2i: HashMap::with_capacity(capacity),
            i2s: Vec::with_capacity(capacity),
            freq: Vec::with_capacity(capacity),
            embeddings: None,
        };
        v.add(default_token.into());
        v
    }

    pub fn add<'a, S: Into<Cow<'a, str>> + ?Sized>(&mut self, word: S) -> u32 {
        let word = word.into();
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
        let rc = RcString::new(word.into_owned());
        self.i2s.push(rc.clone());
        self.s2i.insert(rc, id);
        self.freq.push(1);
        id
    }

    pub fn get<Q: Borrow<str> + ?Sized>(&self, word: &Q) -> u32 {
        self.s2i.get(word.borrow()).map(|v| *v).unwrap_or_else(|| 0)
    }

    pub fn contains<Q: Borrow<str> + ?Sized>(&self, word: &Q) -> bool {
        self.s2i.contains_key(word.borrow())
    }

    pub fn increment(&mut self, id: u32) -> Option<usize> {
        if id == 0 {
            debug_assert!(self.freq[id as usize] == 1);
            Some(1)
        } else {
            self.freq.get_mut(id as usize).map(|count| {
                *count += 1;
                *count
            })
        }
    }

    pub fn freq(&self, id: u32) -> Option<usize> {
        self.freq.get(id as usize).map(|v| *v)
    }

    pub fn lookup(&self, id: u32) -> Option<&str> {
        self.i2s.get(id as usize).map(|v| v.as_str())
    }

    pub fn size(&self) -> usize {
        self.i2s.len()
    }

    pub fn embed(&self) -> Result<&Vec<Vec<f32>>, Error> {
        match self.embeddings.as_ref() {
            Some(embeddings) => {
                if self.size() == embeddings.len() {
                    Ok(embeddings)
                } else {
                    Err(Error::InvalidOperation("uninitialized words exist"))
                }
            }
            None => Err(Error::InvalidOperation("vocab does not use embeddings")),
        }
    }

    #[cfg(feature = "app")]
    pub fn init_embed(&mut self) -> Result<(), Error> {
        let vocab_size = self.size();
        match self.embeddings.as_mut() {
            Some(embeddings) => {
                let num_uninitialized_words = vocab_size - embeddings.len();
                if num_uninitialized_words == 0 {
                    return Ok(());
                }
                embeddings.reserve(num_uninitialized_words);
                let dim = embeddings[0].len();
                let uniform = distributions::Uniform::from(-1.0..1.0);
                let mut rng = thread_rng();
                for _ in 0..num_uninitialized_words {
                    let mut value = Vec::with_capacity(dim);
                    for _ in 0..dim {
                        value.push(uniform.sample(&mut rng));
                    }
                    embeddings.push(value);
                }
                Ok(())
            }
            None => Err(Error::InvalidOperation("vocab does not use embeddings")),
        }
    }

    pub fn has_embed(&self) -> bool {
        self.embeddings.is_some()
    }
}

#[cfg(feature = "app")]
impl_cache!(Vocab);

#[derive(Debug)]
pub enum Error {
    InvalidOperation(&'static str),
}

impl Error {
    pub fn as_str(&self) -> &'static str {
        match *self {
            Error::InvalidOperation(message) => message,
        }
    }
}

impl error::Error for Error {
    fn description(&self) -> &str {
        self.as_str()
    }

    fn cause(&self) -> Option<&error::Error> {
        None
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

trait Tokenize {
    fn tokenize(sentence: &str) -> Vec<String>;
}
