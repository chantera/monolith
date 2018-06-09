pub use monolith::dataset::Load;
use monolith::dataset::{conll, StdLoader};
use monolith::lang::prelude::*;
use monolith::preprocessing::{Preprocess, Vocab};

static CHAR_PADDING: &'static str = "<PAD>";

/// (word_ids, char_ids, option(sentence), tag_ids) of a sentence.
pub type Sample<T> = (Vec<u32>, Vec<Vec<u32>>, Option<T>, Vec<u32>);

#[derive(Debug, Serialize, Deserialize)]
pub struct Preprocessor {
    word_v: Vocab,
    char_v: Vocab,
    postag_v: Vocab,
}

impl Preprocessor {
    pub fn new(word_v: Vocab) -> Self {
        let mut char_v = Vocab::new();
        let pad_id = char_v.add(CHAR_PADDING);
        assert!(pad_id == 1);
        Preprocessor {
            word_v,
            char_v,
            postag_v: Vocab::with_default_token("NN"),
        }
    }

    pub fn word_vocab(&self) -> &Vocab {
        &self.word_v
    }

    pub fn char_vocab(&self) -> &Vocab {
        &self.char_v
    }

    pub fn postag_vocab(&self) -> &Vocab {
        &self.postag_v
    }
}

impl<S: Phrasal> Preprocess<S> for Preprocessor {
    type Output = Sample<S>;

    fn fit_each(&mut self, x: &S) -> Option<Self::Output> {
        let len = x.len();
        let mut word_ids = Vec::with_capacity(len);
        let mut char_ids = Vec::with_capacity(len);
        let mut postag_ids = Vec::with_capacity(len);
        let fix_word = self.word_v.has_embed();
        x.iter().skip(1).for_each(|token| {
            let form = token.form();
            word_ids.push(if fix_word {
                let id = self.word_v.get(&form.to_lowercase());
                self.word_v.increment(id);
                id
            } else {
                self.word_v.add(form.to_lowercase())
            });
            char_ids.push(
                form.chars()
                    .map(|c| self.char_v.add(c.to_string()))
                    .collect(),
            );
            postag_ids.push(self.postag_v.add(token.postag().unwrap().to_string()));
        });
        Some((word_ids, char_ids, None, postag_ids))
    }

    fn transform_each(&self, x: S) -> Self::Output {
        let len = x.len();
        let mut word_ids = Vec::with_capacity(len);
        let mut char_ids = Vec::with_capacity(len);
        let mut postag_ids = Vec::with_capacity(len);
        x.iter().skip(1).for_each(|token| {
            let form = token.form();
            word_ids.push(self.word_v.get(&form.to_lowercase()));
            char_ids.push(
                form.chars()
                    .map(|c| self.char_v.get(&c.to_string()))
                    .collect(),
            );
            postag_ids.push(self.postag_v.get(token.postag().unwrap()));
        });
        (word_ids, char_ids, Some(x), postag_ids)
    }
}

pub type Loader<'a, P> = StdLoader<Sentence<conll::Token<'a>>, P>;
