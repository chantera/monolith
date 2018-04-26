use lang::{Phrasal, Tokenized};
pub use self::text::*;

mod text;

pub trait Preprocess<T> {
    type Output;

    fn fit<I: Iterator<Item = T>>(&mut self, xs: I) {
        xs.for_each(|x| { self.fit_each(&x); });
    }

    #[allow(unused_variables)]
    fn fit_each(&mut self, x: &T) -> Option<Self::Output> {
        None
    }

    fn transform<I: Iterator<Item = T>>(&self, xs: I) -> Vec<Self::Output> {
        // TODO: return iterator
        xs.map(|x| self.transform_each(x)).collect()
    }

    fn transform_each(&self, x: T) -> Self::Output;

    fn fit_transform<I: Iterator<Item = T>>(&mut self, xs: I) -> Vec<Self::Output> {
        // TODO: return iterator
        xs.map(|x| self.fit_transform_each(x)).collect()
    }

    fn fit_transform_each(&mut self, x: T) -> Self::Output {
        match self.fit_each(&x) {
            Some(y) => y,
            None => self.transform_each(x),
        }
    }
}

pub struct TextPreprocessor {
    vocab: Vocab,
}

impl TextPreprocessor {
    pub fn new(vocab: Vocab) -> Self {
        TextPreprocessor { vocab: vocab }
    }
}

impl<T: Phrasal> Preprocess<T> for TextPreprocessor {
    type Output = Vec<u32>;

    fn fit_each(&mut self, x: &T) -> Option<Self::Output> {
        let word_ids = x.iter()
            .map(|token| self.vocab.add(token.form().to_lowercase()))
            .collect();
        Some(word_ids)
    }

    fn transform_each(&self, x: T) -> Self::Output {
        x.iter()
            .map(|token| self.vocab.get(&token.form().to_lowercase()))
            .collect()
    }
}
