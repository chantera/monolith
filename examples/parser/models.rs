use std::marker::PhantomData;
use std::rc::Rc;
use std::f32::NEG_INFINITY as F32_NEG_INFINITY;
use std::u32::MAX as U32_MAX;

use monolith::lang::prelude::*;
use monolith::models::*;
use monolith::preprocessing::Vocab;
use monolith::syntax::transition::prelude::*;
use monolith::syntax::transition::{ArcStandard, State};
use monolith::training;
use primitiv::Model;
use primitiv::Node;
use primitiv::Parameter;
use primitiv::node_functions as F;
use primitiv::initializers as I;
use regex::Regex;
use slog::Logger;

/// tuple of (heads, labels)
pub type ParserOutput = (Vec<Option<u32>>, Vec<Option<u32>>);

#[derive(Debug, Serialize, Deserialize)]
pub struct ChenManning14Model {
    #[serde(skip)]
    model: Model,
    word_embed: Embed,
    postag_embed: Embed,
    label_embed: Embed,
    #[serde(skip)]
    pw1: Parameter,
    #[serde(skip)]
    pb1: Parameter,
    #[serde(skip)]
    pw2: Parameter,
    dropout_rate: f32,
}

impl ChenManning14Model {
    pub fn new(dropout: f32) -> Self {
        let mut m = ChenManning14Model {
            model: Model::new(),
            word_embed: Embed::new(),
            postag_embed: Embed::new(),
            label_embed: Embed::new(),
            pw1: Parameter::new(),
            pb1: Parameter::new(),
            pw2: Parameter::new(),
            dropout_rate: dropout,
        };
        m.reload();
        m
    }

    pub fn reload(&mut self) {
        if self.model.get_submodel("word_embed").is_none() {
            self.word_embed.reload();
            self.model.add_submodel("word_embed", &mut self.word_embed);
        }
        if self.model.get_submodel("postag_embed").is_none() {
            self.postag_embed.reload();
            self.model.add_submodel(
                "postag_embed",
                &mut self.postag_embed,
            );
        }
        if self.model.get_submodel("label_embed").is_none() {
            self.label_embed.reload();
            self.model.add_submodel(
                "label_embed",
                &mut self.label_embed,
            );
        }
        if self.model.get_parameter("pw1").is_none() {
            self.model.add_parameter("pw1", &mut self.pw1);
        }
        if self.model.get_parameter("pb1").is_none() {
            self.model.add_parameter("pb1", &mut self.pb1);
        }
        if self.model.get_parameter("pw2").is_none() {
            self.model.add_parameter("pw2", &mut self.pw2);
        }
    }

    pub fn init(
        &mut self,
        word_vocab_size: usize,
        word_embed_size: u32,
        postag_vocab_size: usize,
        postag_embed_size: u32,
        label_vocab_size: usize,
        label_embed_size: u32,
        mlp_unit: u32,
        out_size: usize,
    ) {
        self.word_embed.init(word_vocab_size, word_embed_size);
        self.word_embed.update_enabled = true;
        self.init_common(
            postag_vocab_size,
            postag_embed_size,
            label_vocab_size,
            label_embed_size,
            mlp_unit,
            out_size,
        );
    }

    pub fn init_by_values<Entries, Values>(
        &mut self,
        word_embed: Entries,
        postag_vocab_size: usize,
        postag_embed_size: u32,
        label_vocab_size: usize,
        label_embed_size: u32,
        mlp_unit: u32,
        out_size: usize,
    ) where
        Entries: AsRef<[Values]>,
        Values: AsRef<[f32]>,
    {
        self.word_embed.init_by_values(word_embed);
        self.word_embed.update_enabled = false;
        self.init_common(
            postag_vocab_size,
            postag_embed_size,
            label_vocab_size,
            label_embed_size,
            mlp_unit,
            out_size,
        );
    }

    fn init_common(
        &mut self,
        postag_vocab_size: usize,
        postag_embed_size: u32,
        label_vocab_size: usize,
        label_embed_size: u32,
        mlp_unit: u32,
        out_size: usize,
    ) {
        self.postag_embed.init(postag_vocab_size, postag_embed_size);
        self.label_embed.init(label_vocab_size, label_embed_size);
        let feature_dim = self.word_embed.embed_size() * (NUM_CM14_WORD_FEATURES as u32) +
            postag_embed_size * (NUM_CM14_POSTAG_FEATURES as u32) +
            label_embed_size * (NUM_CM14_LABEL_FEATURES as u32);
        self.pw1.init_by_initializer(
            [mlp_unit, feature_dim],
            &I::XavierUniform::new(1.0),
        );
        self.pb1.init_by_initializer(
            [mlp_unit],
            &I::Constant::new(0.0),
        );
        self.pw2.init_by_initializer(
            [out_size as u32, mlp_unit],
            &I::XavierUniform::new(1.0),
        );
    }

    pub fn forward<FeatureBatch, Features>(&mut self, xs: FeatureBatch, train: bool) -> Node
    where
        FeatureBatch: AsRef<[Features]>,
        Features: AsRef<[ChenManning14Feature]>,
    {
        let num_samples = xs.as_ref().iter().fold(0, |sum, x| sum + x.as_ref().len());
        let mut word_ids: Vec<&[u32]> = Vec::with_capacity(num_samples);
        let mut postag_ids: Vec<&[u32]> = Vec::with_capacity(num_samples);
        let mut label_ids: Vec<&[u32]> = Vec::with_capacity(num_samples);
        for features in xs.as_ref() {
            for feature in features.as_ref() {
                word_ids.push(&feature.words);
                postag_ids.push(&feature.postags);
                label_ids.push(&feature.labels);
            }
        }
        let xs_words = self.word_embed.forward(word_ids);
        let xs_postags = self.postag_embed.forward(postag_ids);
        let xs_labels = self.label_embed.forward(label_ids);
        let xs_features: Vec<Node> = xs_words
            .into_iter()
            .zip(xs_postags.into_iter())
            .zip(xs_labels.into_iter())
            .map(|((x_w, x_p), x_l)| {
                let x = F::batch::concat(
                    [
                        F::dropout(x_w, self.dropout_rate, train),
                        F::dropout(x_p, self.dropout_rate, train),
                        F::dropout(x_l, self.dropout_rate, train),
                    ],
                );
                let x = F::concat(F::batch::split(x, NUM_CM14_FEATURES as u32), 0);
                x
            })
            .collect();
        let xs_features = F::batch::concat(xs_features);
        let w1 = F::parameter(&mut self.pw1);
        let b1 = F::parameter(&mut self.pb1);
        let w2 = F::parameter(&mut self.pw2);
        let hs = F::pown(F::matmul(w1, xs_features) + b1, 3);
        let ys = F::matmul(w2, F::dropout(hs, self.dropout_rate, train));
        ys
    }

    pub fn loss<ActionBatch, Actions>(&mut self, ys: &Node, ts: ActionBatch) -> Node
    where
        ActionBatch: AsRef<[Actions]>,
        Actions: AsRef<[u32]>,
    {
        let batch_size = ts.as_ref().len() as u32;
        let mut actions = Vec::with_capacity(ys.shape().batch() as usize);
        ts.as_ref().iter().for_each(
            |t| actions.extend_from_slice(t.as_ref()),
        );
        let loss = F::batch::sum(F::softmax_cross_entropy_with_ids(ys, &actions, 0));
        loss / batch_size
    }

    pub fn accuracy<ActionBatch, Actions>(&mut self, ys: &Node, ts: ActionBatch) -> (u32, u32)
    where
        ActionBatch: AsRef<[Actions]>,
        Actions: AsRef<[u32]>,
    {
        let mut correct = 0;
        let mut count = 0;
        let ys = ys.argmax(0);
        for actions in ts.as_ref() {
            for t in actions.as_ref() {
                if ys[count] == *t {
                    correct += 1;
                }
                count += 1;
            }
        }
        (correct, count as u32)
    }

    pub fn parse<WordBatch, PostagBatch, WordIDs, PostagIDs>(
        &mut self,
        words: WordBatch,
        postags: PostagBatch,
        word_pad_id: u32,
        postag_pad_id: u32,
        label_pad_id: u32,
    ) -> Vec<ParserOutput>
    where
        WordBatch: AsRef<[WordIDs]>,
        PostagBatch: AsRef<[PostagIDs]>,
        WordIDs: AsRef<[u32]>,
        PostagIDs: AsRef<[u32]>,
    {
        let words: Vec<&[u32]> = words.as_ref().into_iter().map(|x| x.as_ref()).collect();
        let postags: Vec<&[u32]> = postags.as_ref().into_iter().map(|x| x.as_ref()).collect();
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
                    let feature = ChenManning14Feature::extract(
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
            let ys = self.forward(features, false);
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
            .map(|&(_index, ref state)| {
                (state.heads().to_vec(), state.labels().to_vec())
            })
            .collect();
        outputs
    }
}

impl_model!(ChenManning14Model, model);

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
pub struct ChenManning14Feature {
    pub words: [u32; NUM_CM14_WORD_FEATURES],
    pub postags: [u32; NUM_CM14_POSTAG_FEATURES],
    pub labels: [u32; NUM_CM14_LABEL_FEATURES],
}

const NUM_CM14_WORD_FEATURES: usize = 18;
const NUM_CM14_POSTAG_FEATURES: usize = 18;
const NUM_CM14_LABEL_FEATURES: usize = 12;
const NUM_CM14_FEATURES: usize = NUM_CM14_WORD_FEATURES + NUM_CM14_POSTAG_FEATURES +
    NUM_CM14_LABEL_FEATURES;
const PAD_ID: u32 = U32_MAX - 1024;

impl ChenManning14Feature {
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
        ChenManning14Feature {
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

#[derive(Debug)]
pub struct ParserBuilder<'a, M> {
    _model_type: PhantomData<M>,
    word_vocab_size: usize,
    word_embed_size: u32,
    word_embed: Option<&'a Vec<Vec<f32>>>,
    postag_vocab_size: usize,
    postag_embed_size: u32,
    label_vocab_size: usize,
    label_embed_size: u32,
    mlp_unit: u32,
    out_size: Option<usize>,
    dropout_rate: f32,
}

impl<'a, M> ParserBuilder<'a, M> {
    pub fn word(mut self, vocab_size: usize, embed_size: u32) -> Self {
        self.word_vocab_size = vocab_size;
        self.word_embed_size = embed_size;
        self
    }

    pub fn word_embed(mut self, values: &'a Vec<Vec<f32>>) -> Self {
        self.word_embed = Some(values);
        self
    }

    pub fn postag(mut self, vocab_size: usize, embed_size: u32) -> Self {
        self.postag_vocab_size = vocab_size;
        self.postag_embed_size = embed_size;
        self
    }

    pub fn label(mut self, vocab_size: usize, embed_size: u32) -> Self {
        self.label_vocab_size = vocab_size;
        self.label_embed_size = embed_size;
        self
    }

    pub fn dropout(mut self, p: f32) -> Self {
        self.dropout_rate = p;
        self
    }

    pub fn mlp(mut self, unit: u32) -> Self {
        self.mlp_unit = unit;
        self
    }

    #[allow(dead_code)]
    pub fn out(mut self, size: usize) -> Self {
        self.out_size = Some(size);
        self
    }
}

impl<'a> ParserBuilder<'a, ChenManning14Model> {
    pub fn build(self) -> ChenManning14Model {
        let out_size = self.out_size.unwrap_or_else(
            || self.label_vocab_size * 2 + 1,
        );
        let mut model = ChenManning14Model::new(self.dropout_rate);
        match self.word_embed {
            Some(values) => {
                model.init_by_values(
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
                    self.word_vocab_size,
                    self.word_embed_size,
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

impl<'a> Default for ParserBuilder<'a, ChenManning14Model> {
    fn default() -> Self {
        ParserBuilder {
            _model_type: PhantomData,
            word_vocab_size: 60000,
            word_embed_size: 50,
            word_embed: None,
            postag_vocab_size: 64,
            postag_embed_size: 50,
            label_vocab_size: 64,
            label_embed_size: 50,
            mlp_unit: 200,
            out_size: None,
            dropout_rate: 0.5,
        }
    }
}

#[derive(Debug)]
pub struct Evaluator {
    unlabeled_match: usize,
    labeled_match: usize,
    count: usize,
    num_sentences: usize,
    label_vocab: *const Vocab,
    re: Regex,
    logger: Rc<Logger>,
}

impl Evaluator {
    pub fn new<L: Into<Rc<Logger>>>(label_v: &Vocab, logger: L) -> Self {
        Evaluator {
            unlabeled_match: 0,
            labeled_match: 0,
            count: 0,
            num_sentences: 0,
            label_vocab: label_v,
            re: Regex::new(r"^[^\s\d\w]+$").unwrap(),
            logger: logger.into(),
        }
    }

    pub fn reset(&mut self) {
        self.unlabeled_match = 0;
        self.labeled_match = 0;
        self.num_sentences = 0;
        self.count = 0;
    }

    pub fn evaluate<S: Phrasal>(
        &mut self,
        predicted_heads: &[Option<u32>],
        predicted_labels: &[Option<u32>],
        sentence: &S,
    ) {
        let label_vocab = unsafe { &*self.label_vocab };
        self.num_sentences += 1;
        for (i, token) in sentence.iter().enumerate().skip(1) {
            if self.is_punct(token.form()) {
                continue;
            }
            self.count += 1;
            if predicted_heads[i] == Some(token.head().unwrap() as u32) {
                self.unlabeled_match += 1;
                if let Some(label) = predicted_labels[i] {
                    if label_vocab.lookup(label).unwrap() == token.deprel().unwrap() {
                        self.labeled_match += 1;
                    }
                }
            }
        }
    }

    pub fn is_punct(&self, word: &str) -> bool {
        self.re.is_match(word)
    }

    pub fn uas(&self) -> Result<f32, training::Error> {
        if self.count == 0 {
            return Err(training::Error::ZeroDivision);
        }
        let score = (self.unlabeled_match as f64 / self.count as f64) as f32 * 100.0;
        Ok(score)
    }

    pub fn las(&self) -> Result<f32, training::Error> {
        if self.count == 0 {
            return Err(training::Error::ZeroDivision);
        }
        let score = (self.labeled_match as f64 / self.count as f64) as f32 * 100.0;
        Ok(score)
    }

    pub fn report(&self) {
        if self.count > 0 {
            info!(
                self.logger,
                "#samples: {}, UAS: {:.6}, LAS: {:.6}",
                self.num_sentences,
                self.uas().unwrap(),
                self.las().unwrap()
            );
        } else {
            info!(
                self.logger,
                "#samples: {}, UAS: NaN, LAS: NaN",
                self.num_sentences,
            );
        }
    }
}

impl<S: Phrasal> training::Callback<Option<(Vec<ParserOutput>, Vec<*const S>)>> for Evaluator {
    fn on_epoch_validate_begin(
        &mut self,
        _info: &training::TrainingInfo<Option<(Vec<ParserOutput>, Vec<*const S>)>>,
    ) {
        self.reset();
    }

    fn on_epoch_validate_end(
        &mut self,
        _info: &training::TrainingInfo<Option<(Vec<ParserOutput>, Vec<*const S>)>>,
    ) {
        self.report();
    }

    fn on_batch_end(
        &mut self,
        info: &training::TrainingInfo<Option<(Vec<ParserOutput>, Vec<*const S>)>>,
    ) {
        if !info.train {
            let &(ref outputs, ref sentences) = info.output.as_ref().unwrap().as_ref().unwrap();
            for (&(ref heads, ref labels), sentence) in outputs.iter().zip(sentences) {
                let sentence: &S = unsafe { &**sentence };
                self.evaluate(&heads, &labels, sentence);
            }
        }
    }
}
