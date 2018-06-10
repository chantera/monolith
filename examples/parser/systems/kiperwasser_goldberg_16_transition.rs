//! Simple and Accurate Dependency Parsing Using Bidirectional LSTM Feature Representations
//! Eliyahu Kiperwasser and Yoav Goldberg, TACL, 2016.
//! https://www.transacl.org/ojs/index.php/tacl/article/view/885
//!
//! Our implementation is different from the original one in the following:
//! (1) word embeddings: The original model concatenates randomly initialized word embeddings
//! and pretrained ones. Both are tuned during training. Ours uses either of them and does not
//! updates those when using pretrained ones.
//! (2) parameter initialization: We might use different initialization for parameters. Original
//! model uses PyCNN default initializers.
//! (3) dropout: We apply dropout to all layers while the authors employ "word dropout".
//! (4) training: We use a static oracle and cross entropy loss instead of a dynamic oracle and
//! max-margin loss.

use std::f32::NEG_INFINITY as F32_NEG_INFINITY;
use std::marker::PhantomData;

use monolith::lang::prelude::*;
use monolith::models::*;
use monolith::preprocessing::{Preprocess, Vocab};
use monolith::syntax::transition::prelude::*;
use monolith::syntax::transition::{self, ArcStandard, State}; // TODO(chanetera): use ArcHybrid
use primitiv::functions as F;
use primitiv::{Node, Variable};

use dataset;
use models;

static WORD_PADDING: &'static str = "<PAD>";
static POSTAG_PADDING: &'static str = "<PAD>";

const NUM_BILSTM_LAYERS: usize = 2;
const NUM_MLP_LAYERS: usize = 2;

/// (word_ids, postag_ids, features, eval_data(sentence), actions)
pub type Sample<T> = (Vec<u32>, Vec<u32>, Vec<Feature>, Option<T>, Vec<u32>);

#[derive(Debug, Serialize, Deserialize)]
pub struct Preprocessor {
    v_mapper: dataset::VocabMapper,
}

impl Preprocessor {
    pub fn new(mut word_v: Vocab) -> Self {
        word_v.add(WORD_PADDING);
        let mut postag_v = Vocab::new();
        postag_v.add(POSTAG_PADDING);
        let label_v = Vocab::with_default_token("dep");
        let v_mapper = dataset::VocabMapper::new(word_v, None, Some(postag_v), label_v);
        Preprocessor { v_mapper }
    }

    pub fn word_vocab(&self) -> &Vocab {
        self.v_mapper.word_vocab()
    }

    pub fn word_pad_id(&self) -> u32 {
        self.word_vocab().get(WORD_PADDING)
    }

    pub fn postag_vocab(&self) -> &Vocab {
        self.v_mapper.postag_vocab().unwrap()
    }

    pub fn postag_pad_id(&self) -> u32 {
        self.postag_vocab().get(POSTAG_PADDING)
    }

    pub fn label_vocab(&self) -> &Vocab {
        self.v_mapper.label_vocab()
    }

    pub fn extract_features_and_actions<T: Tokenized>(
        &self,
        heads: &[u32],
        labels: &[u32],
    ) -> (Vec<Feature>, Vec<u32>) {
        let heads = transition::projectivize(heads);
        let (state, features) = transition::GoldState::with_feature_extract::<
            transition::ArcStandard,
            _,
            _,
        >(&heads, &labels, |state| Feature::extract(state))
            .unwrap();
        let actions = Vec::from(state.actions());
        (features, actions)
    }
}

impl From<Vocab> for Preprocessor {
    fn from(word_v: Vocab) -> Self {
        Preprocessor::new(word_v)
    }
}

impl<S: Phrasal> Preprocess<S> for Preprocessor {
    type Output = Sample<S>;

    fn fit_each(&mut self, x: &S) -> Option<Self::Output> {
        let (word_ids, _char_ids, postag_ids, label_ids) =
            self.v_mapper.map_with_fitting(x.tokens());
        let heads: Vec<u32> = x.iter().map(|token| token.head().unwrap() as u32).collect();
        let (features, actions) =
            self.extract_features_and_actions::<<S as Phrasal>::Token>(&heads, &label_ids);
        let sample = (word_ids, postag_ids.unwrap(), features, None, actions);
        Some(sample)
    }

    fn transform_each(&self, x: S) -> Self::Output {
        let (word_ids, _char_ids, postag_ids, label_ids) =
            self.v_mapper.map_without_fitting(x.tokens());
        let heads: Vec<u32> = x.iter().map(|token| token.head().unwrap() as u32).collect();
        let (features, actions) =
            self.extract_features_and_actions::<<S as Phrasal>::Token>(&heads, &label_ids);
        let sample = (word_ids, postag_ids.unwrap(), features, Some(x), actions);
        sample
    }
}

#[derive(Debug, Model, Serialize, Deserialize)]
pub struct BistTransitionModel<V: Variable> {
    #[primitiv(submodel)]
    word_embed: Embed,
    #[primitiv(submodel)]
    postag_embed: Embed,
    #[primitiv(submodel)]
    bilstm: BiLSTM<V>,
    #[primitiv(submodel)]
    pad_linear: Linear,
    #[primitiv(submodel)]
    mlp: MLP,
    dropout_rate: f32,
    word_pad_id: u32,
    postag_pad_id: u32,
}

impl<V: Variable> BistTransitionModel<V> {
    pub fn new(word_pad_id: u32, postag_pad_id: u32, dropout_rate: f32) -> Self {
        BistTransitionModel {
            word_embed: Embed::new(),
            postag_embed: Embed::new(),
            bilstm: BiLSTM::new(NUM_BILSTM_LAYERS, dropout_rate),
            pad_linear: Linear::default(),
            mlp: MLP::new(NUM_MLP_LAYERS, Activation::Tanh, dropout_rate),
            dropout_rate,
            word_pad_id,
            postag_pad_id,
        }
    }

    pub fn init<W: EmbedInitialize>(
        &mut self,
        word_embed: W,
        postag_vocab_size: u32,
        postag_embed_size: u32,
        lstm_hidden_size: u32,
        mlp_unit: u32,
        out_size: u32,
    ) {
        self.word_embed.init_by(word_embed);
        self.postag_embed.init(postag_vocab_size, postag_embed_size);
        self.bilstm.init(
            self.word_embed.embed_size() + postag_embed_size,
            lstm_hidden_size,
        );
        self.pad_linear
            .init(self.bilstm.input_size(), self.bilstm.output_size());
        self.mlp.init(
            &[
                self.bilstm.output_size() * Feature::num_indices() as u32,
                mlp_unit,
            ],
            out_size as u32,
        );
    }

    pub fn encode<WordIDs: AsRef<[u32]>, PostagIDs: AsRef<[u32]>>(
        &mut self,
        words: &[WordIDs],
        postags: &[PostagIDs],
        train: bool,
    ) -> Vec<V> {
        let xs: Vec<V> = self
            .word_embed
            .forward_iter(words.iter())
            .zip(self.postag_embed.forward_iter(postags.iter()))
            .map(|(x_w, x_p): (V, V)| {
                F::concat(
                    &[
                        F::dropout(x_w, self.dropout_rate, train),
                        F::dropout(x_p, self.dropout_rate, train),
                    ],
                    0,
                )
            })
            .collect();
        let hs_batch = transpose_sequence(self.bilstm.forward(xs, None, train));
        hs_batch
    }

    pub fn decode<FSeq: AsRef<[Feature]>>(&mut self, xs: &[FSeq], hs: &[V], train: bool) -> V {
        let pad = F::tanh(self.pad_linear.forward(F::concat(
            [
                self.word_embed.lookup::<V>(&[self.word_pad_id]),
                self.postag_embed.lookup::<V>(&[self.postag_pad_id]),
            ],
            0,
        )));
        let fs = F::batch::concat(
            xs.iter()
                .zip(hs)
                .map(|(features, vs)| self.populate_feature(features.as_ref(), vs, &pad))
                .collect::<Vec<V>>(),
        );
        let ys = self.mlp.forward(fs, train);
        ys
    }

    fn populate_feature(&self, features: &[Feature], hs: &V, pad: &V) -> V {
        let batch_size = features.len();
        let feature_size = Feature::num_indices();
        let mut ids = Vec::with_capacity(batch_size * feature_size);
        for f in features {
            for index in [f.b0, f.s0, f.s1, f.s2].into_iter() {
                ids.push(index.map(|i| i + 1).unwrap_or(0));
            }
        }
        let feature_lookup = F::concat([pad, &F::transpose(hs)], 1);
        let fs = F::batch::split(F::pick(feature_lookup, &ids, 1), ids.len() as u32);
        let fs = F::batch::concat(
            (0..ids.len())
                .filter(|i| i % feature_size == 0)
                .map(|i| F::concat(&fs[i..i + feature_size], 0))
                .collect::<Vec<V>>(),
        ); // ([dim * feature_size], batch_size)
        fs
    }

    pub fn forward<WordIDs: AsRef<[u32]>, PostagIDs: AsRef<[u32]>, FSeq: AsRef<[Feature]>>(
        &mut self,
        words: &[WordIDs],
        postags: &[PostagIDs],
        features: &[FSeq],
        train: bool,
    ) -> V {
        let hs = self.encode(words, postags, train);
        let ys = self.decode(features, &hs, train);
        ys
    }

    pub fn loss<Actions: AsRef<[u32]>>(&mut self, ys: &V, ts: &[Actions]) -> V {
        let batch_size = ts.len() as u32;
        let mut actions = Vec::with_capacity(ys.shape().batch() as usize);
        ts.iter()
            .for_each(|t| actions.extend_from_slice(t.as_ref()));
        let loss = F::batch::sum(F::softmax_cross_entropy_with_ids(ys, &actions, 0));
        loss / batch_size
    }

    pub fn accuracy<Actions: AsRef<[u32]>>(&mut self, ys: &V, ts: &[Actions]) -> (u32, u32) {
        let mut correct = 0;
        let mut count = 0;
        let ys = ys.argmax(0);
        for actions in ts {
            for t in actions.as_ref() {
                if ys[count] == *t {
                    correct += 1;
                }
                count += 1;
            }
        }
        (correct, count as u32)
    }

    pub fn parse<WordIDs: AsRef<[u32]>, PostagIDs: AsRef<[u32]>>(
        &mut self,
        words: &[WordIDs],
        postags: &[PostagIDs],
    ) -> Vec<models::ParserOutput> {
        let hs = self.encode(&words, &postags, false);
        let mut states: Vec<(usize, State)> = hs
            .iter()
            .map(|h| State::new(h.shape().at(0) as u32))
            .enumerate()
            .collect();
        let mut target_states: Vec<(usize, *mut State)> = states
            .iter_mut()
            .map(|&mut (index, ref mut state)| (index, state as *mut _))
            .collect();
        while target_states.len() > 0 {
            let features: Vec<_> = target_states
                .iter()
                .map(|&(_index, state)| {
                    let feature = Feature::extract(unsafe { &*state });
                    [feature]
                })
                .collect();
            let ys = self.decode(&features, &hs, false);
            let num_actions = ys.shape().at(0) as usize;
            let action_scores = ys.to_vector();
            for i in (0..target_states.len()).rev() {
                let state: &mut State = unsafe { &mut *target_states.as_mut_slice()[i].1 };
                let mut best_action = None;
                let mut best_score = F32_NEG_INFINITY;
                for action in 0..num_actions {
                    let score = action_scores[i * num_actions + action];
                    if score > best_score && ArcStandard::is_allowed(action as u32, state) {
                        best_action = Some(action as u32);
                        best_score = score;
                    }
                }
                ArcStandard::apply(best_action.unwrap(), state).unwrap();
                if ArcStandard::is_terminal(state) {
                    target_states.remove(i);
                }
            }
        }
        let outputs = states
            .into_iter()
            .map(|(_index, state)| state.into_arcs())
            .collect();
        outputs
    }
}

#[derive(Debug)]
pub struct Feature {
    pub b0: Option<u32>,
    pub s0: Option<u32>,
    pub s1: Option<u32>,
    pub s2: Option<u32>,
}

impl Feature {
    pub fn extract(state: &State) -> Self {
        Feature {
            b0: state.buffer(0),
            s0: state.stack(0),
            s1: state.stack(1),
            s2: state.stack(2),
        }
    }

    #[inline]
    pub fn num_indices() -> usize {
        4
    }
}

pub type ParserBuilder<'a> = models::ParserBuilder<'a, BistTransitionModel<Node>>;

impl<'a> ParserBuilder<'a> {
    pub fn build(self) -> BistTransitionModel<Node> {
        if self.word_pad_id.is_none() {
            panic!("`word_pad_id` must be set before builder.build() is called.");
        }
        if self.postag_pad_id.is_none() {
            panic!("`postag_pad_id` must be set before builder.build() is called.");
        }
        if self.out_size.is_none() {
            panic!("`out_size` must be set before builder.build() is called.");
        }
        let mut model = BistTransitionModel::new(
            self.word_pad_id.unwrap(),
            self.postag_pad_id.unwrap(),
            self.dropout_rate,
        );
        match self.word_embed {
            Some(values) => {
                model.init(
                    values,
                    self.postag_vocab_size,
                    self.postag_embed_size,
                    self.lstm_hidden_size,
                    self.mlp_unit,
                    self.out_size.unwrap(),
                );
            }
            None => {
                model.init(
                    (self.word_vocab_size, self.word_embed_size),
                    self.postag_vocab_size,
                    self.postag_embed_size,
                    self.lstm_hidden_size,
                    self.mlp_unit,
                    self.out_size.unwrap(),
                );
            }
        }
        model
    }
}

impl<'a> Default for ParserBuilder<'a> {
    fn default() -> Self {
        ParserBuilder {
            _model_type: PhantomData,
            word_vocab_size: 60000,
            word_embed_size: 100,
            word_embed: None,
            word_pad_id: None,
            postag_vocab_size: 64,
            postag_embed_size: 25,
            postag_pad_id: None,
            label_vocab_size: 0, // unused
            label_embed_size: 0, // unused
            label_pad_id: None,  // unused
            lstm_hidden_size: 125,
            mlp_unit: 100,
            out_size: None,
            dropout_rate: 0.5,
        }
    }
}
