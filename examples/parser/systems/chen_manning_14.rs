//! A Fast and Accurate Dependency Parser using Neural Networks
//! Danqi Chen and Christopher D. Manning, EMNLP, 2014
//! http://aclweb.org/anthology/D14-1082

use std::f32::NEG_INFINITY as F32_NEG_INFINITY;
use std::marker::PhantomData;
use std::u32::MAX as U32_MAX;

use monolith::lang::prelude::*;
use monolith::models::*;
use monolith::preprocessing::{Preprocess, Vocab};
use monolith::syntax::projectivize;
use monolith::syntax::transition::prelude::*;
use monolith::syntax::transition::{self, ArcStandard, State};
use primitiv::functions as F;
use primitiv::initializers as I;
use primitiv::{Parameter, Tensor, Variable};

use dataset;
use models;

static WORD_PADDING: &'static str = "<PAD>";
static POSTAG_PADDING: &'static str = "<PAD>";
static LABEL_PADDING: &'static str = "<PAD>";

/// (features, eval_data(word_ids, postag_ids, sentence), actions)
pub type Sample<T> = (Vec<Feature>, Option<(Vec<u32>, Vec<u32>, T)>, Vec<u32>);

#[derive(Debug, Serialize, Deserialize)]
pub struct Preprocessor {
    v_mapper: dataset::VocabMapper,
}

impl Preprocessor {
    pub fn new(mut word_v: Vocab) -> Self {
        let _pad_id = word_v.add(WORD_PADDING);
        let mut postag_v = Vocab::new();
        let pad_id = postag_v.add(POSTAG_PADDING);
        assert!(pad_id == 1);
        let mut label_v = Vocab::with_default_token("dep");
        let pad_id = label_v.add(LABEL_PADDING);
        assert!(pad_id == 1);
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

    pub fn label_pad_id(&self) -> u32 {
        self.label_vocab().get(LABEL_PADDING)
    }

    /// map a sentence to a feature sequence and an action sequence
    pub fn extract_features_and_actions<T: Tokenized>(
        &self,
        word_ids: &[u32],
        postag_ids: &[u32],
        label_ids: &[u32],
        heads: &[u32],
    ) -> (Vec<Feature>, Vec<u32>) {
        let word_pad_id = self.word_pad_id();
        let postag_pad_id = self.postag_pad_id();
        let label_pad_id = self.label_pad_id();
        let heads = projectivize(heads);
        let (state, features) = transition::GoldState::with_feature_extract::<ArcStandard, _, _>(
            &heads,
            &label_ids,
            |state| {
                Feature::extract(
                    state,
                    &word_ids,
                    &postag_ids,
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
        let (features, actions) = self.extract_features_and_actions::<<S as Phrasal>::Token>(
            &word_ids,
            &postag_ids.unwrap(),
            &label_ids,
            &heads,
        );
        let sample = (features, None, actions);
        Some(sample)
    }

    fn transform_each(&self, x: S) -> Self::Output {
        let (word_ids, _char_ids, postag_ids, label_ids) =
            self.v_mapper.map_without_fitting(x.tokens());
        let heads: Vec<u32> = x.iter().map(|token| token.head().unwrap() as u32).collect();
        let (features, actions) = self.extract_features_and_actions::<<S as Phrasal>::Token>(
            &word_ids,
            postag_ids.as_ref().unwrap(),
            &label_ids,
            &heads,
        );
        let sample = (features, Some((word_ids, postag_ids.unwrap(), x)), actions);
        sample
    }
}

#[derive(Debug, Model, Serialize, Deserialize)]
pub struct ChenManning14Model {
    #[primitiv(submodel)]
    word_embed: Embed,
    #[primitiv(submodel)]
    postag_embed: Embed,
    #[primitiv(submodel)]
    label_embed: Embed,
    pw1: Parameter,
    pb1: Parameter,
    pw2: Parameter,
    dropout_rate: f32,
}

impl ChenManning14Model {
    pub fn new(dropout_rate: f32) -> Self {
        ChenManning14Model {
            word_embed: Embed::new(),
            postag_embed: Embed::new(),
            label_embed: Embed::new(),
            pw1: Parameter::new(),
            pb1: Parameter::new(),
            pw2: Parameter::new(),
            dropout_rate,
        }
    }

    pub fn init<W: EmbedInitialize>(
        &mut self,
        word_embed: W,
        postag_vocab_size: u32,
        postag_embed_size: u32,
        label_vocab_size: u32,
        label_embed_size: u32,
        mlp_unit: u32,
        out_size: u32,
    ) {
        let initializer = I::Uniform::new(-0.01, 0.01);
        self.word_embed.init_by(word_embed);
        self.postag_embed
            .init_by_initializer(postag_vocab_size, postag_embed_size, &initializer);
        self.label_embed
            .init_by_initializer(label_vocab_size, label_embed_size, &initializer);
        let feature_dim = self.word_embed.embed_size() * (NUM_CM14_WORD_FEATURES as u32)
            + postag_embed_size * (NUM_CM14_POSTAG_FEATURES as u32)
            + label_embed_size * (NUM_CM14_LABEL_FEATURES as u32);
        self.pw1
            .init_by_initializer([mlp_unit, feature_dim], &initializer);
        self.pb1
            .init_by_initializer([mlp_unit], &I::Constant::new(0.0));
        self.pw2
            .init_by_initializer([out_size as u32, mlp_unit], &initializer);
    }

    pub fn forward<V: Variable, FSeq: AsRef<[Feature]>>(&mut self, xs: &[FSeq], train: bool) -> V {
        let xs: V = self.lookup_features(xs, train);
        let w1: V = F::parameter(&mut self.pw1);
        let b1: V = F::parameter(&mut self.pb1);
        let w2: V = F::parameter(&mut self.pw2);
        let hs = F::pown(F::matmul(w1, xs) + b1, 3);
        let ys = F::matmul(w2, F::dropout(hs, self.dropout_rate, train));
        ys
    }

    /// Computes feature representations
    ///
    /// This flattens the slice of features and returns a varible whose shape is ([dim], n) where
    /// n is the number of `Feature` objects.
    ///
    /// # Implementation notes
    ///
    /// This first gathers ids for lookup without duplication, and then maps the ids to vectors and
    /// retrieves representation.
    fn lookup_features<V: Variable, FSeq: AsRef<[Feature]>>(
        &mut self,
        xs: &[FSeq],
        train: bool,
    ) -> V {
        let num_samples = xs.iter().fold(0, |sum, x| sum + x.as_ref().len());
        let mut word_ids: Vec<u32> = Vec::with_capacity(1024);
        let mut postag_ids: Vec<u32> = Vec::with_capacity(64);
        let mut label_ids: Vec<u32> = Vec::with_capacity(64);
        let mut word_indices: Vec<usize> = Vec::with_capacity(num_samples * NUM_CM14_WORD_FEATURES);
        let mut postag_indices: Vec<usize> =
            Vec::with_capacity(num_samples * NUM_CM14_POSTAG_FEATURES);
        let mut label_indices: Vec<usize> =
            Vec::with_capacity(num_samples * NUM_CM14_LABEL_FEATURES);
        for features in xs {
            for feature in features.as_ref() {
                word_indices.extend(feature.words.iter().map(|id| or_insert(&mut word_ids, *id)));
                postag_indices.extend(
                    feature
                        .postags
                        .iter()
                        .map(|id| or_insert(&mut postag_ids, *id)),
                );
                label_indices.extend(
                    feature
                        .labels
                        .iter()
                        .map(|id| or_insert(&mut label_ids, *id)),
                );
            }
        }
        let xs_words = F::batch::split(
            F::dropout(
                self.word_embed.lookup::<V>(&word_ids),
                self.dropout_rate,
                train,
            ),
            word_ids.len() as u32,
        );
        let xs_postags = F::batch::split(
            F::dropout(
                self.postag_embed.lookup::<V>(&postag_ids),
                self.dropout_rate,
                train,
            ),
            postag_ids.len() as u32,
        );
        let xs_labels = F::batch::split(
            F::dropout(
                self.label_embed.lookup::<V>(&label_ids),
                self.dropout_rate,
                train,
            ),
            label_ids.len() as u32,
        );
        let xs_features: Vec<V> = (0..num_samples)
            .map(|i| {
                let w_iter = word_indices
                    [(i * NUM_CM14_WORD_FEATURES)..((i + 1) * NUM_CM14_WORD_FEATURES)]
                    .iter()
                    .map(|idx| &xs_words[*idx]);
                let p_iter = postag_indices
                    [(i * NUM_CM14_POSTAG_FEATURES)..((i + 1) * NUM_CM14_POSTAG_FEATURES)]
                    .iter()
                    .map(|idx| &xs_postags[*idx]);
                let l_iter = label_indices
                    [(i * NUM_CM14_LABEL_FEATURES)..((i + 1) * NUM_CM14_LABEL_FEATURES)]
                    .iter()
                    .map(|idx| &xs_labels[*idx]);
                F::concat(w_iter.chain(p_iter).chain(l_iter).collect::<Vec<&V>>(), 0)
            })
            .collect();
        F::batch::concat(xs_features)
    }

    pub fn loss<V: Variable, Actions: AsRef<[u32]>>(&mut self, ys: &V, ts: &[Actions]) -> V {
        let batch_size = ts.len() as u32;
        let mut actions = Vec::with_capacity(ys.shape().batch() as usize);
        ts.iter()
            .for_each(|t| actions.extend_from_slice(t.as_ref()));
        let loss = F::batch::sum(F::softmax_cross_entropy_with_ids(ys, &actions, 0));
        loss / batch_size
    }

    pub fn accuracy<V: Variable, Actions: AsRef<[u32]>>(
        &mut self,
        ys: &V,
        ts: &[Actions],
    ) -> (u32, u32) {
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
        word_pad_id: u32,
        postag_pad_id: u32,
        label_pad_id: u32,
    ) -> Vec<models::ParserOutput> {
        let words: Vec<&[u32]> = words.into_iter().map(|x| x.as_ref()).collect();
        let postags: Vec<&[u32]> = postags.into_iter().map(|x| x.as_ref()).collect();
        let mut states: Vec<(usize, State)> = words
            .iter()
            .map(|x| State::new(x.len() as u32))
            .enumerate()
            .collect();
        let mut target_states: Vec<(usize, *mut State)> = states
            .iter_mut()
            .map(|&mut (index, ref mut state)| (index, state as *mut _))
            .collect();
        while target_states.len() > 0 {
            let features: Vec<_> = target_states
                .iter()
                .map(|&(index, state)| {
                    let feature = Feature::extract(
                        unsafe { &*state },
                        words[index],
                        postags[index],
                        word_pad_id,
                        postag_pad_id,
                        label_pad_id,
                    );
                    [feature]
                })
                .collect();
            let ys: Tensor = self.forward(&features, false);
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
            .iter()
            .map(|&(_index, ref state)| (state.heads().to_vec(), state.labels().to_vec()))
            .collect();
        outputs
    }
}

#[inline]
fn or_insert<T: PartialEq>(vec: &mut Vec<T>, value: T) -> usize {
    vec.iter().position(|v| *v == value).unwrap_or_else(|| {
        let index = vec.len();
        vec.push(value);
        index
    })
}

/// Feature template of [Chen and Manning, 2014, EMNLP]
///
/// ## The choice of S^w, S^t, S^l
/// Following (Zhang and Nivre, 2011), we pick a rich set of elements for our final parser.
/// In detail, S^w contains n_w = 18 elements:
/// (1) The top 3 words on the stack and buffer:
///     s_1, s_2, s_3, b_1, b_2, b_3;
/// (2) The first and second leftmost / rightmost children of the top two words on the stack:
///     lc_1(s_i), rc_1(s_i), lc_2(s_i), rc_2(s_i), i = 1, 2.
/// (3) The leftmost of leftmost / rightmost of rightmost children of the top two words on the stack:
///     lc_1(lc_1(s_i)), rc_1(rc_1(s_i)), i = 1, 2.
/// We use the corresponding POS tags for S^t (n_t = 18), and the corresponding arc labels of words
/// excluding those 6 words on the stack/buffer for S^l (n_l = 12).
/// A good advantage of our parser is that we can add a rich set of elements cheaply,
/// instead of hand-crafting many more indicator features.
#[derive(Debug)]
pub struct Feature {
    pub words: [u32; NUM_CM14_WORD_FEATURES],
    pub postags: [u32; NUM_CM14_POSTAG_FEATURES],
    pub labels: [u32; NUM_CM14_LABEL_FEATURES],
}

const NUM_CM14_WORD_FEATURES: usize = 18;
const NUM_CM14_POSTAG_FEATURES: usize = 18;
const NUM_CM14_LABEL_FEATURES: usize = 12;
const PAD_ID: u32 = U32_MAX - 1024;

impl Feature {
    pub fn extract(
        state: &State,
        words: &[u32],
        postags: &[u32],
        pad_word: u32,
        pad_postag: u32,
        pad_label: u32,
    ) -> Self {
        let s0 = state.stack(0).unwrap_or(PAD_ID);
        let s1 = state.stack(1).unwrap_or(PAD_ID);
        let s2 = state.stack(2).unwrap_or(PAD_ID);
        let b0 = state.buffer(0).unwrap_or(PAD_ID);
        let b1 = state.buffer(1).unwrap_or(PAD_ID);
        let b2 = state.buffer(2).unwrap_or(PAD_ID);
        let lc1_s0 = state.leftmost(s0, None).unwrap_or(PAD_ID);
        let rc1_s0 = state.rightmost(s0, None).unwrap_or(PAD_ID);
        let lc2_s0 = state.leftmost(s0, Some(lc1_s0 + 1)).unwrap_or(PAD_ID);
        let rc2_s0 = state.rightmost(s0, Some(rc1_s0 - 1)).unwrap_or(PAD_ID);
        let lc1_s1 = state.leftmost(s1, None).unwrap_or(PAD_ID);
        let rc1_s1 = state.rightmost(s1, None).unwrap_or(PAD_ID);
        let lc2_s1 = state.leftmost(s1, Some(lc1_s1 + 1)).unwrap_or(PAD_ID);
        let rc2_s1 = state.rightmost(s1, Some(rc1_s1 - 1)).unwrap_or(PAD_ID);
        let lc1_lc1_s0 = state.leftmost(lc1_s0, None).unwrap_or(PAD_ID);
        let rc1_rc1_s0 = state.rightmost(rc1_s0, None).unwrap_or(PAD_ID);
        let lc1_lc1_s1 = state.leftmost(lc1_s1, None).unwrap_or(PAD_ID);
        let rc1_rc1_s1 = state.rightmost(rc1_s1, None).unwrap_or(PAD_ID);
        let s0 = s0 as usize;
        let s1 = s1 as usize;
        let s2 = s2 as usize;
        let b0 = b0 as usize;
        let b1 = b1 as usize;
        let b2 = b2 as usize;
        let lc1_s0 = lc1_s0 as usize;
        let rc1_s0 = rc1_s0 as usize;
        let lc2_s0 = lc2_s0 as usize;
        let rc2_s0 = rc2_s0 as usize;
        let lc1_s1 = lc1_s1 as usize;
        let rc1_s1 = rc1_s1 as usize;
        let lc2_s1 = lc2_s1 as usize;
        let rc2_s1 = rc2_s1 as usize;
        let lc1_lc1_s0 = lc1_lc1_s0 as usize;
        let rc1_rc1_s0 = rc1_rc1_s0 as usize;
        let lc1_lc1_s1 = lc1_lc1_s1 as usize;
        let rc1_rc1_s1 = rc1_rc1_s1 as usize;
        Feature {
            words: [
                words.get(s0).map(|id| *id).unwrap_or(pad_word),
                words.get(s1).map(|id| *id).unwrap_or(pad_word),
                words.get(s2).map(|id| *id).unwrap_or(pad_word),
                words.get(b0).map(|id| *id).unwrap_or(pad_word),
                words.get(b1).map(|id| *id).unwrap_or(pad_word),
                words.get(b2).map(|id| *id).unwrap_or(pad_word),
                words.get(lc1_s0).map(|id| *id).unwrap_or(pad_word),
                words.get(rc1_s0).map(|id| *id).unwrap_or(pad_word),
                words.get(lc2_s0).map(|id| *id).unwrap_or(pad_word),
                words.get(rc2_s0).map(|id| *id).unwrap_or(pad_word),
                words.get(lc1_s1).map(|id| *id).unwrap_or(pad_word),
                words.get(rc1_s1).map(|id| *id).unwrap_or(pad_word),
                words.get(lc2_s1).map(|id| *id).unwrap_or(pad_word),
                words.get(rc2_s1).map(|id| *id).unwrap_or(pad_word),
                words.get(lc1_lc1_s0).map(|id| *id).unwrap_or(pad_word),
                words.get(rc1_rc1_s0).map(|id| *id).unwrap_or(pad_word),
                words.get(lc1_lc1_s1).map(|id| *id).unwrap_or(pad_word),
                words.get(rc1_rc1_s1).map(|id| *id).unwrap_or(pad_word),
            ],
            postags: [
                postags.get(s0).map(|id| *id).unwrap_or(pad_postag),
                postags.get(s1).map(|id| *id).unwrap_or(pad_postag),
                postags.get(s2).map(|id| *id).unwrap_or(pad_postag),
                postags.get(b0).map(|id| *id).unwrap_or(pad_postag),
                postags.get(b1).map(|id| *id).unwrap_or(pad_postag),
                postags.get(b2).map(|id| *id).unwrap_or(pad_postag),
                postags.get(lc1_s0).map(|id| *id).unwrap_or(pad_postag),
                postags.get(rc1_s0).map(|id| *id).unwrap_or(pad_postag),
                postags.get(lc2_s0).map(|id| *id).unwrap_or(pad_postag),
                postags.get(rc2_s0).map(|id| *id).unwrap_or(pad_postag),
                postags.get(lc1_s1).map(|id| *id).unwrap_or(pad_postag),
                postags.get(rc1_s1).map(|id| *id).unwrap_or(pad_postag),
                postags.get(lc2_s1).map(|id| *id).unwrap_or(pad_postag),
                postags.get(rc2_s1).map(|id| *id).unwrap_or(pad_postag),
                postags.get(lc1_lc1_s0).map(|id| *id).unwrap_or(pad_postag),
                postags.get(rc1_rc1_s0).map(|id| *id).unwrap_or(pad_postag),
                postags.get(lc1_lc1_s1).map(|id| *id).unwrap_or(pad_postag),
                postags.get(rc1_rc1_s1).map(|id| *id).unwrap_or(pad_postag),
            ],
            labels: [
                state.label(lc1_s0 as u32).unwrap_or(pad_label),
                state.label(rc1_s0 as u32).unwrap_or(pad_label),
                state.label(lc2_s0 as u32).unwrap_or(pad_label),
                state.label(rc2_s0 as u32).unwrap_or(pad_label),
                state.label(lc1_s1 as u32).unwrap_or(pad_label),
                state.label(rc1_s1 as u32).unwrap_or(pad_label),
                state.label(lc2_s1 as u32).unwrap_or(pad_label),
                state.label(rc2_s1 as u32).unwrap_or(pad_label),
                state.label(lc1_lc1_s0 as u32).unwrap_or(pad_label),
                state.label(rc1_rc1_s0 as u32).unwrap_or(pad_label),
                state.label(lc1_lc1_s1 as u32).unwrap_or(pad_label),
                state.label(rc1_rc1_s1 as u32).unwrap_or(pad_label),
            ],
        }
    }
}

pub type ParserBuilder<'a> = models::ParserBuilder<'a, ChenManning14Model>;

impl<'a> ParserBuilder<'a> {
    pub fn build(self) -> ChenManning14Model {
        let out_size = self
            .out_size
            .unwrap_or_else(|| self.label_vocab_size * 2 + 1);
        let mut model = ChenManning14Model::new(self.dropout_rate);
        match self.word_embed {
            Some(values) => {
                model.init(
                    values,
                    self.postag_vocab_size,
                    self.postag_embed_size,
                    self.label_vocab_size,
                    self.label_embed_size,
                    self.mlp_unit,
                    out_size,
                );
            }
            None => {
                model.init(
                    (
                        self.word_vocab_size,
                        self.word_embed_size,
                        I::Uniform::new(-0.01, 0.01),
                    ),
                    self.postag_vocab_size,
                    self.postag_embed_size,
                    self.label_vocab_size,
                    self.label_embed_size,
                    self.mlp_unit,
                    out_size,
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
            word_embed_size: 50,
            word_embed: None,
            word_pad_id: None, // unused
            postag_vocab_size: 64,
            postag_embed_size: 50,
            postag_pad_id: None, // unused
            label_vocab_size: 64,
            label_embed_size: 50,
            label_pad_id: None,  // unused
            lstm_hidden_size: 0, // unused
            mlp_unit: 200,
            out_size: None,
            dropout_rate: 0.5,
        }
    }
}
