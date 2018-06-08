use std::io::Result as IOResult;
use std::marker::PhantomData;
use std::path::Path;

use monolith::dataset::{conll, Dataset, Load, StdLoader};
use monolith::io::serialize;
use monolith::lang::{Sentence, Tokenized};
use monolith::preprocessing::{Preprocess, Vocab};
use serde::de::DeserializeOwned;
use serde::ser::Serialize;
use slog::Logger;

#[derive(Debug, Serialize, Deserialize)]
pub struct Preprocessor<M> {
    _marker: PhantomData<M>,
    pub(crate) word_v: Vocab,
    pub(crate) char_v: Option<Vocab>,
    pub(crate) postag_v: Option<Vocab>,
    pub(crate) label_v: Vocab,
}

impl<M> Preprocessor<M> {
    pub(crate) fn from_vocabs(
        mut word_v: Vocab,
        mut char_v: Option<Vocab>,
        mut postag_v: Option<Vocab>,
        mut label_v: Vocab,
    ) -> Self {
        if word_v.has_embed() {
            word_v.init_embed().unwrap();
        }
        if let Some(ref mut v) = char_v {
            if v.has_embed() {
                v.init_embed().unwrap();
            }
        }
        if let Some(ref mut v) = postag_v {
            if v.has_embed() {
                v.init_embed().unwrap();
            }
        }
        if label_v.has_embed() {
            label_v.init_embed().unwrap();
        }
        Preprocessor {
            _marker: PhantomData,
            word_v,
            char_v,
            postag_v,
            label_v,
        }
    }

    pub fn word_vocab(&self) -> &Vocab {
        &self.word_v
    }

    pub fn char_vocab(&self) -> Option<&Vocab> {
        self.char_v.as_ref()
    }

    pub fn postag_vocab(&self) -> Option<&Vocab> {
        self.postag_v.as_ref()
    }

    pub fn label_vocab(&self) -> &Vocab {
        &self.label_v
    }

    pub fn map_with_fitting<T: Tokenized>(
        &mut self,
        tokens: &[T],
    ) -> (Vec<u32>, Option<Vec<Vec<u32>>>, Option<Vec<u32>>, Vec<u32>) {
        let len = tokens.len();
        let mut word_ids = Vec::with_capacity(len);
        let mut char_ids = self.char_vocab().map(|_| Vec::with_capacity(len));
        let mut postag_ids = self.postag_vocab().map(|_| Vec::with_capacity(len));
        let mut label_ids = Vec::with_capacity(len);
        let fix_word = self.word_v.has_embed();
        tokens.iter().for_each(|token| {
            let form = token.form();
            word_ids.push(if fix_word {
                let id = self.word_v.get(&form.to_lowercase());
                self.word_v.increment(id);
                id
            } else {
                self.word_v.add(form.to_lowercase())
            });
            if let Some(ref mut buf) = char_ids.as_mut() {
                let mut v = self.char_v.as_mut().unwrap();
                buf.push(form.chars().map(|c| v.add(c.to_string())).collect());
            }
            if let Some(ref mut buf) = postag_ids.as_mut() {
                let mut v = self.postag_v.as_mut().unwrap();
                buf.push(v.add(token.postag().unwrap().to_string()));
            }
            label_ids.push(self.label_v.add(token.deprel().unwrap()));
        });
        (word_ids, char_ids, postag_ids, label_ids)
    }

    pub fn map_without_fitting<T: Tokenized>(
        &self,
        tokens: &[T],
    ) -> (Vec<u32>, Option<Vec<Vec<u32>>>, Option<Vec<u32>>, Vec<u32>) {
        let len = tokens.len();
        let mut word_ids = Vec::with_capacity(len);
        let mut char_ids = self.char_vocab().map(|_| Vec::with_capacity(len));
        let mut postag_ids = self.postag_vocab().map(|_| Vec::with_capacity(len));
        let mut label_ids = Vec::with_capacity(len);
        tokens.iter().for_each(|token| {
            let form = token.form();
            word_ids.push(self.word_v.get(&form.to_lowercase()));
            if let Some(ref mut buf) = char_ids.as_mut() {
                let v = self.char_vocab().unwrap();
                buf.push(form.chars().map(|c| v.get(&c.to_string())).collect());
            }
            if let Some(ref mut buf) = postag_ids.as_mut() {
                let v = self.postag_vocab().unwrap();
                buf.push(v.get(token.postag().unwrap()));
            }
            label_ids.push(self.label_v.get(token.deprel().unwrap()));
        });
        (word_ids, char_ids, postag_ids, label_ids)
    }
}

pub type Loader<'a, P> = StdLoader<Sentence<conll::Token<'a>>, P>;

pub fn load<'a, P, P1, P2, P3, P4>(
    train_file: P1,
    valid_file: Option<P2>,
    embed_file: Option<P3>,
    save_to: Option<P4>,
    logger: &Logger,
) -> IOResult<(Dataset<P::Output>, Option<Dataset<P::Output>>, P)>
where
    P: Preprocess<Sentence<conll::Token<'a>>> + From<Vocab> + Serialize + DeserializeOwned,
    P1: AsRef<Path>,
    P2: AsRef<Path>,
    P3: AsRef<Path>,
    P4: AsRef<Path>,
{
    let mut loader = Loader::new(P::from(match embed_file {
        Some(f) => {
            info!(logger, "embed file: {}", f.as_ref().display());
            Vocab::from_cache_or_file(f, "<UNK>")?
        }
        None => {
            info!(logger, "embed file: None");
            Vocab::new()
        }
    }));
    info!(logger, "train file: {}", train_file.as_ref().display());
    let train_dataset = loader.load(train_file)?;
    loader.fix();
    let valid_dataset = match valid_file {
        Some(f) => {
            info!(logger, "valid file: {}", f.as_ref().display());
            Some(loader.load(f)?)
        }
        None => {
            info!(logger, "valid file: None");
            None
        }
    };
    if let Some(ref path) = save_to.as_ref() {
        let path = format!("{}-loader.json", path.as_ref().to_str().unwrap());
        info!(logger, "saving the loader to {} ...", path);
        serialize::write_to(&loader, path, serialize::Format::Json).unwrap();
    }
    Ok((train_dataset, valid_dataset, loader.dispose()))
}
