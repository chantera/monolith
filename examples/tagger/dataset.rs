use monolith::lang::prelude::*;
use monolith::dataset::{conll, StdLoader};
pub use monolith::dataset::Load;
use monolith::preprocessing::Preprocess;
use monolith::preprocessing::Vocab;

static CHAR_PADDING: &'static str = "<PAD>";

/// (word_ids, char_ids, tag_ids) of a sentence.
pub type Sample = (Vec<u32>, Vec<Vec<u32>>, Vec<u32>);

#[derive(Debug, Serialize, Deserialize)]
pub struct Preprocessor {
    word_v: Vocab,
    char_v: Vocab,
    pos_v: Vocab,
}

impl Preprocessor {
    pub fn new(mut word_v: Vocab) -> Self {
        word_v.disable_serializing_embeddings();
        let mut char_v = Vocab::new();
        let pad_id = char_v.add(CHAR_PADDING);
        assert!(pad_id == 1);
        Preprocessor {
            word_v: word_v,
            char_v: char_v,
            pos_v: Vocab::with_default_token("NN"),
        }
    }

    pub fn word_vocab(&self) -> &Vocab {
        &self.word_v
    }

    pub fn char_vocab(&self) -> &Vocab {
        &self.char_v
    }

    pub fn pos_vocab(&self) -> &Vocab {
        &self.pos_v
    }
}

impl<T: Phrasal> Preprocess<T> for Preprocessor {
    type Output = Sample;

    fn fit_each(&mut self, x: &T) -> Option<Self::Output> {
        let mut word_ids = vec![];
        let mut char_ids = vec![];
        let mut pos_ids = vec![];
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
                token
                    .form()
                    .chars()
                    .map(|c| self.char_v.add(c.to_string()))
                    .collect(),
            );
            pos_ids.push(self.pos_v.add(token.postag().unwrap().to_string()));
        });
        Some((word_ids, char_ids, pos_ids))
    }

    fn transform_each(&self, x: T) -> Self::Output {
        let mut word_ids = vec![];
        let mut char_ids = vec![];
        let mut pos_ids = vec![];
        x.iter().skip(1).for_each(|token| {
            let form = token.form();
            word_ids.push(self.word_v.get(&form.to_lowercase()));
            char_ids.push(
                token
                    .form()
                    .chars()
                    .map(|c| self.char_v.get(&c.to_string()))
                    .collect(),
            );
            pos_ids.push(self.pos_v.get(token.postag().unwrap()));
        });
        (word_ids, char_ids, pos_ids)
    }
}

pub type Loader<'a, P> = StdLoader<Sentence<conll::Token<'a>>, P>;
