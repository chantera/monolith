use std::marker::PhantomData;

use monolith::lang::prelude::*;
use monolith::dataset::{conll, StdLoader};
pub use monolith::dataset::Load;
use monolith::preprocessing::Preprocess;
use monolith::preprocessing::Vocab;
use monolith::syntax::transition;
use monolith::syntax::transition::prelude::*;

use models::ChenManning14Feature;

static WORD_PADDING: &'static str = "<PAD>";
static POSTAG_PADDING: &'static str = "<PAD>";
static LABEL_PADDING: &'static str = "<PAD>";

pub struct Preprocessor<O> {
    _output_type: PhantomData<O>,
    word_v: Vocab,
    postag_v: Vocab,
    label_v: Vocab,
}

impl<O> Preprocessor<O> {
    pub fn new(mut word_v: Vocab) -> Self {
        let mut postag_v = Vocab::with_default_token("NN");
        let mut label_v = Vocab::with_default_token("dep");
        word_v.add(WORD_PADDING);
        postag_v.add(POSTAG_PADDING);
        label_v.add(LABEL_PADDING);
        Preprocessor {
            _output_type: PhantomData,
            word_v: word_v,
            postag_v: postag_v,
            label_v: label_v,
        }
    }

    pub fn word_vocab(&self) -> &Vocab {
        &self.word_v
    }

    pub fn postag_vocab(&self) -> &Vocab {
        &self.postag_v
    }

    pub fn label_vocab(&self) -> &Vocab {
        &self.label_v
    }

    pub fn map_with_fit<T: Tokenized>(&mut self, tokens: &[T]) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        let len = tokens.len();
        let mut word_ids = Vec::with_capacity(len);
        let mut postag_ids = Vec::with_capacity(len);
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
            postag_ids.push(self.postag_v.add(token.postag().unwrap().to_string()));
            label_ids.push(self.label_v.add(token.deprel().unwrap().to_string()));
        });
        (word_ids, label_ids, postag_ids)
    }

    pub fn map<T: Tokenized>(&self, tokens: &[T]) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        let len = tokens.len();
        let mut word_ids = Vec::with_capacity(len);
        let mut postag_ids = Vec::with_capacity(len);
        let mut label_ids = Vec::with_capacity(len);
        tokens.iter().for_each(|token| {
            let form = token.form();
            word_ids.push(self.word_v.get(&form.to_lowercase()));
            postag_ids.push(self.postag_v.get(token.postag().unwrap()));
            label_ids.push(self.label_v.get(token.deprel().unwrap()));
        });
        (word_ids, label_ids, postag_ids)
    }
}

impl Preprocessor<Vec<ChenManning14Feature>> {
    /// map a sentence to a feature sequence and an action sequence
    pub fn extract_features_and_actions<T: Tokenized>(
        &self,
        word_ids: &[u32],
        postag_ids: &[u32],
        label_ids: &[u32],
        heads: &[u32],
    ) -> (Vec<ChenManning14Feature>, Vec<u32>) {
        let word_pad_id = self.word_v.get(WORD_PADDING);
        let postag_pad_id = self.postag_v.get(POSTAG_PADDING);
        let label_pad_id = self.label_v.get(LABEL_PADDING);
        let (state, features) =
            transition::GoldState::with_feature_extract::<transition::ArcStandard, _, _>(
                &heads,
                &label_ids,
                |state| {
                    ChenManning14Feature::extract(
                        state,
                        &word_ids,
                        &postag_ids,
                        &label_ids,
                        word_pad_id,
                        postag_pad_id,
                        label_pad_id,
                    )
                },
            ).unwrap();
        let actions = Vec::from(state.actions());
        (features, actions)
    }
}

impl<P: Phrasal> Preprocess<P> for Preprocessor<Vec<ChenManning14Feature>> {
    type Output = (Vec<ChenManning14Feature>, Vec<u32>);

    fn fit_each(&mut self, x: &P) -> Option<Self::Output> {
        let (word_ids, postag_ids, label_ids) = self.map_with_fit(x.tokens());
        let heads: Vec<u32> = x.iter().map(|token| token.head().unwrap() as u32).collect();
        let sample = self.extract_features_and_actions::<<P as Phrasal>::Token>(
            &word_ids,
            &postag_ids,
            &label_ids,
            &heads,
        );
        Some(sample)
    }

    fn transform_each(&self, x: P) -> Self::Output {
        let (word_ids, postag_ids, label_ids) = self.map(x.tokens());
        let heads: Vec<u32> = x.iter().map(|token| token.head().unwrap() as u32).collect();
        let sample = self.extract_features_and_actions::<<P as Phrasal>::Token>(
            &word_ids,
            &postag_ids,
            &label_ids,
            &heads,
        );
        sample
    }
}

pub type Loader<'a, P> = StdLoader<Sentence<conll::Token<'a>>, P>;
