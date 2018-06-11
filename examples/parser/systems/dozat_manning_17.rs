//! Deep Biaffine Attention for Neural Dependency Parsing
//! Timothy Dozat, Christopher D. Manning, ICLR 2017
//! https://openreview.net/forum?id=Hk95PK9le
//!
//!
//! orthogonal initialization
//! bucketing
//! pretrained word vector
//! unknown words

use std::marker::PhantomData;

use monolith::lang::prelude::*;
use monolith::models::*;
use monolith::preprocessing::{Preprocess, Vocab};
use primitiv::functions as F;
use primitiv::{Node, Variable};

use dataset;
use models;

const NUM_BILSTM_LAYERS: usize = 3;
const LABEL_MLP_UNIT_SIZE: u32 = 100;

/// (word_ids, postag_ids, eval_data(sentence), (heads, labels))
pub type Sample<T> = (Vec<u32>, Vec<u32>, Option<T>, (Vec<u32>, Vec<u32>));

#[derive(Debug, Serialize, Deserialize)]
pub struct Preprocessor {
    v_mapper: dataset::VocabMapper,
}

impl Preprocessor {
    pub fn new(word_v: Vocab) -> Self {
        let postag_v = Vocab::new();
        let label_v = Vocab::with_default_token("dep");
        let v_mapper = dataset::VocabMapper::new(word_v, None, Some(postag_v), label_v);
        Preprocessor { v_mapper }
    }

    pub fn word_vocab(&self) -> &Vocab {
        self.v_mapper.word_vocab()
    }

    pub fn postag_vocab(&self) -> &Vocab {
        self.v_mapper.postag_vocab().unwrap()
    }

    pub fn label_vocab(&self) -> &Vocab {
        self.v_mapper.label_vocab()
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
        let sample = (word_ids, postag_ids.unwrap(), None, (heads, label_ids));
        Some(sample)
    }

    fn transform_each(&self, x: S) -> Self::Output {
        let (word_ids, _char_ids, postag_ids, label_ids) =
            self.v_mapper.map_without_fitting(x.tokens());
        let heads: Vec<u32> = x.iter().map(|token| token.head().unwrap() as u32).collect();
        let sample = (word_ids, postag_ids.unwrap(), Some(x), (heads, label_ids));
        sample
    }
}

#[derive(Debug, Model, Serialize, Deserialize)]
pub struct DozatManning17Model<V: Variable> {
    #[primitiv(submodel)]
    word_embed: Embed,
    #[primitiv(submodel)]
    postag_embed: Embed,
    #[primitiv(submodel)]
    bilstm: BiLSTM<V>,
    #[primitiv(submodel)]
    arc_head_linear: Linear,
    #[primitiv(submodel)]
    arc_dep_linear: Linear,
    #[primitiv(submodel)]
    label_head_linear: Linear,
    #[primitiv(submodel)]
    label_dep_linear: Linear,
    #[primitiv(submodel)]
    arc_bilinear: Bilinear,
    #[primitiv(submodel)]
    label_bilinear: Bilinear,
    dropout_rate: f32,
}

impl<V: Variable> DozatManning17Model<V> {
    pub fn new(dropout_rate: f32) -> Self {
        DozatManning17Model {
            word_embed: Embed::new(),
            postag_embed: Embed::new(),
            bilstm: BiLSTM::new(NUM_BILSTM_LAYERS, dropout_rate),
            arc_head_linear: Linear::default(),
            arc_dep_linear: Linear::default(),
            label_head_linear: Linear::default(),
            label_dep_linear: Linear::default(),
            arc_bilinear: Bilinear::new((true, false, false)),
            label_bilinear: Bilinear::new((true, true, true)),
            dropout_rate,
        }
    }

    pub fn init<W: EmbedInitialize>(
        &mut self,
        word_embed: W,
        postag_vocab_size: u32,
        postag_embed_size: u32,
        label_vocab_size: u32,
        lstm_hidden_size: u32,
        arc_mlp_unit: u32,
        label_mlp_unit: u32,
    ) {
        self.word_embed.init_by(word_embed);
        self.postag_embed.init(postag_vocab_size, postag_embed_size);
        self.bilstm.init(
            self.word_embed.embed_size() + postag_embed_size,
            lstm_hidden_size,
        );
        let bilstm_out_size = self.bilstm.output_size();
        self.arc_head_linear.init(bilstm_out_size, arc_mlp_unit);
        self.arc_dep_linear.init(bilstm_out_size, arc_mlp_unit);
        self.label_head_linear.init(bilstm_out_size, label_mlp_unit);
        self.label_dep_linear.init(bilstm_out_size, label_mlp_unit);
        self.arc_bilinear.init(arc_mlp_unit, arc_mlp_unit, 1);
        self.label_bilinear
            .init(label_mlp_unit, label_mlp_unit, label_vocab_size);
    }

    pub fn forward<WordIDs: AsRef<[u32]>, PostagIDs: AsRef<[u32]>>(
        &mut self,
        words: &[WordIDs],
        postags: &[PostagIDs],
        train: bool,
    ) -> (V, V) {
        let dropout_rate = self.dropout_rate;
        let dropout = |x: &V| -> V { F::dropout(x, dropout_rate, train) };
        let xs: Vec<V> = self
            .word_embed
            .forward_iter(words.iter())
            .zip(self.postag_embed.forward_iter(postags.iter()))
            .map(|(x_w, x_p): (V, V)| F::concat([dropout(&x_w), dropout(&x_p)], 0))
            .collect();
        let (hs, _lenghts) = pad_sequence(self.bilstm.forward(xs, None, train), 0.0);
        let hs_arc_head = F::lrelu(self.arc_head_linear.forward(dropout(&hs)));
        let hs_arc_dep = F::lrelu(self.arc_dep_linear.forward(dropout(&hs)));
        let hs_label_head = F::lrelu(self.label_head_linear.forward(dropout(&hs)));
        let hs_label_dep = F::lrelu(self.label_dep_linear.forward(dropout(&hs)));
        let hs_arc_scores = self
            .arc_bilinear
            .forward(dropout(&hs_arc_head), dropout(&hs_arc_dep));
        let hs_label_scores = self
            .label_bilinear
            .forward(dropout(&hs_label_head), dropout(&hs_label_dep));
        let dims = hs_label_scores.shape().dims();
        let hs_arc_scores = F::reshape(hs_arc_scores, [dims[0], dims[2]]);
        (hs_arc_scores, hs_label_scores)
    }

    pub fn loss<Heads: AsRef<[u32]>, LabelIDs: AsRef<[u32]>>(
        &mut self,
        ys_heads: &V,
        ys_labels: &V,
        ts_heads: &[Heads],
        ts_labels: &[LabelIDs],
    ) -> V {
        let batch_size = ts_heads.len();
        let arc_iter = F::batch::split(ys_heads, batch_size as u32)
            .into_iter()
            .zip(ts_heads);
        let label_iter = F::batch::split(ys_labels, batch_size as u32)
            .into_iter()
            .zip(ts_labels);
        let mut arc_loss: Vec<V> = Vec::with_capacity(batch_size);
        let label_loss: Vec<V> = arc_iter
            .zip(label_iter)
            .map(|((y_heads, t_heads), (y_labels, t_labels))| {
                // println!("y: {}, t: {}", y_heads.shape(), t_heads.as_ref().len());
                let heads = y_heads.argmax(0);
                let len = t_heads.as_ref().len();

                arc_loss.push(F::batch::sum(F::softmax_cross_entropy_with_ids(
                    F::batch::concat(&F::split(y_heads, 1, heads.len() as u32)[..len]),
                    t_heads.as_ref(),
                    0,
                )));
                // println!("arc_loss: {}", arc_loss[arc_loss.len() - 1].shape());
                let label_scores = F::pick(
                    F::batch::concat(&F::split(y_labels, 2, heads.len() as u32)[..len]),
                    &heads[..len],
                    0,
                );
                // println!(
                //     "y: {}, t: {}",
                //     label_scores.shape(),
                //     t_labels.as_ref().len()
                // );
                // F::softmax_cross_entropy_with_ids(label_scores, t_labels.as_ref(), 1)
                let loss = F::batch::sum(F::softmax_cross_entropy_with_ids(
                    label_scores,
                    t_labels.as_ref(),
                    1,
                ));
                // println!("label_loss: {}", loss.shape());
                loss
            })
            .collect();
        let loss = F::sum_vars(arc_loss) + F::sum_vars(label_loss);
        loss / (batch_size as u32)
    }

    pub fn accuracy<Heads: AsRef<[u32]>, LabelIDs: AsRef<[u32]>>(
        &mut self,
        ys_heads: &V,
        ys_labels: &V,
        ts_heads: &[Heads],
        ts_labels: &[LabelIDs],
    ) -> (u32, u32) {
        let mut correct_heads = 0;
        let mut correct_labels = 0;
        let mut count = 0;
        let batch_size = ts_heads.len();
        let arc_iter = F::batch::split(ys_heads, batch_size as u32)
            .into_iter()
            .zip(ts_heads);
        let label_iter = F::batch::split(ys_labels, batch_size as u32)
            .into_iter()
            .zip(ts_labels);
        for ((y_heads, t_heads), (y_labels, t_labels)) in arc_iter.zip(label_iter) {
            let heads = y_heads.argmax(0);
            let len = t_heads.as_ref().len();
            let labels = F::pick(
                F::batch::concat(&F::split(y_labels, 2, heads.len() as u32)[..len]),
                &heads[..len],
                0,
            ).argmax(1);
            for (i, (t_head, t_label)) in t_heads.as_ref().iter().zip(t_labels.as_ref()).enumerate()
            {
                if heads[i] == *t_head {
                    correct_heads += 1;
                }
                if labels[i] == *t_label {
                    correct_labels += 1;
                }
                count += 1;
            }
            // println!(
            //     "y: {}, t: {}",
            //     label_scores.argmax(1).len(),
            //     t_labels.as_ref().len()
            // );
        }
        (correct_heads + correct_labels, count * 2)
        // let mut arc_loss: Vec<V> = Vec::with_capacity(batch_size);
        // let label_loss: Vec<V> = arc_iter
        //     .zip(label_iter)
        //     .map(|((y_heads, t_heads), (y_labels, t_labels))| {
        //         // println!("y: {}, t: {}", y_heads.shape(), t_heads.as_ref().len());
        //         let heads = y_heads.argmax(0);
        //         let len = t_heads.as_ref().len();
        //
        //         arc_loss.push(F::batch::sum(F::softmax_cross_entropy_with_ids(
        //             F::batch::concat(&F::split(y_heads, 1, heads.len() as u32)[..len]),
        //             t_heads.as_ref(),
        //             0,
        //         )));
        //         // println!("arc_loss: {}", arc_loss[arc_loss.len() - 1].shape());
        //         let label_scores = F::pick(
        //             F::batch::concat(&F::split(y_labels, 2, heads.len() as u32)[..len]),
        //             &heads[..len],
        //             0,
        //         );
        //         // println!(
        //         //     "y: {}, t: {}",
        //         //     label_scores.shape(),
        //         //     t_labels.as_ref().len()
        //         // );
        //         // F::softmax_cross_entropy_with_ids(label_scores, t_labels.as_ref(), 1)
        //         let loss = F::batch::sum(F::softmax_cross_entropy_with_ids(
        //             label_scores,
        //             t_labels.as_ref(),
        //             1,
        //         ));
        //         // println!("label_loss: {}", loss.shape());
        //         loss
        //     })
        //     .collect();
        // let loss = F::sum_vars(arc_loss) + F::sum_vars(label_loss);
        // loss / (batch_size as u32)
    }
}

pub type ParserBuilder<'a> = models::ParserBuilder<'a, DozatManning17Model<Node>>;

impl<'a> ParserBuilder<'a> {
    pub fn build(self) -> DozatManning17Model<Node> {
        if self.label_vocab_size == 0 {
            panic!("`label_vocab_size` must be set before builder.build() is called.");
        }
        let mut model = DozatManning17Model::new(self.dropout_rate);
        match self.word_embed {
            Some(values) => {
                model.init(
                    values,
                    self.postag_vocab_size,
                    self.postag_embed_size,
                    self.label_vocab_size,
                    self.lstm_hidden_size,
                    self.mlp_unit,
                    LABEL_MLP_UNIT_SIZE,
                );
            }
            None => {
                model.init(
                    (self.word_vocab_size, self.word_embed_size),
                    self.postag_vocab_size,
                    self.postag_embed_size,
                    self.label_vocab_size,
                    self.lstm_hidden_size,
                    self.mlp_unit,
                    LABEL_MLP_UNIT_SIZE,
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
            word_pad_id: None, // unused
            postag_vocab_size: 64,
            postag_embed_size: 100,
            postag_pad_id: None, // unused
            label_vocab_size: 0,
            label_embed_size: 0, // unused
            label_pad_id: None,  // unused
            lstm_hidden_size: 400,
            mlp_unit: 500,
            out_size: None, // unused
            dropout_rate: 0.33,
        }
    }
}
