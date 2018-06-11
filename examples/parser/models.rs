use std::error::Error;
use std::marker::PhantomData;
use std::path::Path;
use std::rc::Rc;

use monolith::dataset::Dataset;
use monolith::io::serialize;
use monolith::lang::{Phrasal, Tokenized};
use monolith::preprocessing::Vocab;
use monolith::training;
use primitiv::Model;
use primitiv::Optimizer;
use regex::Regex;
use serde::de::DeserializeOwned;
use serde::ser::Serialize;
use slog::Logger;

/// tuple of (heads, labels)
pub type ParserOutput = (Vec<Option<u32>>, Vec<Option<u32>>);

#[derive(Debug)]
pub struct ParserBuilder<'a, M> {
    pub(crate) _model_type: PhantomData<M>,
    pub(crate) word_vocab_size: u32,
    pub(crate) word_embed_size: u32,
    pub(crate) word_embed: Option<&'a Vec<Vec<f32>>>,
    pub(crate) word_pad_id: Option<u32>,
    pub(crate) postag_vocab_size: u32,
    pub(crate) postag_embed_size: u32,
    pub(crate) postag_pad_id: Option<u32>,
    pub(crate) label_vocab_size: u32,
    pub(crate) label_embed_size: u32,
    pub(crate) label_pad_id: Option<u32>,
    pub(crate) lstm_hidden_size: u32,
    pub(crate) mlp_unit: u32,
    pub(crate) out_size: Option<u32>,
    pub(crate) dropout_rate: f32,
}

impl<'a, M> ParserBuilder<'a, M> {
    pub fn word(mut self, vocab_size: u32, embed_size: u32, pad_id: Option<u32>) -> Self {
        self.word_vocab_size = vocab_size;
        self.word_embed_size = embed_size;
        self.word_pad_id = pad_id;
        self
    }

    pub fn word_embed(mut self, values: &'a Vec<Vec<f32>>, pad_id: Option<u32>) -> Self {
        self.word_embed = Some(values);
        self.word_pad_id = pad_id;
        self
    }

    pub fn postag(mut self, vocab_size: u32, embed_size: u32, pad_id: Option<u32>) -> Self {
        self.postag_vocab_size = vocab_size;
        self.postag_embed_size = embed_size;
        self.postag_pad_id = pad_id;
        self
    }

    pub fn label(mut self, vocab_size: u32, embed_size: u32, pad_id: Option<u32>) -> Self {
        self.label_vocab_size = vocab_size;
        self.label_embed_size = embed_size;
        self.label_pad_id = pad_id;
        self
    }

    #[allow(unused)]
    pub fn lstm(mut self, hidden_size: u32) -> Self {
        self.lstm_hidden_size = hidden_size;
        self
    }

    #[allow(unused)]
    pub fn mlp(mut self, unit: u32) -> Self {
        self.mlp_unit = unit;
        self
    }

    pub fn out(mut self, size: u32) -> Self {
        self.out_size = Some(size);
        self
    }

    #[allow(unused)]
    pub fn dropout(mut self, p: f32) -> Self {
        self.dropout_rate = p;
        self
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
                "#samples: {}, UAS: NaN, LAS: NaN", self.num_sentences,
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

pub fn train<F, FO, S: Phrasal, M, O: Optimizer, T, P: AsRef<Path>>(
    mut forward: F,
    mut model: M,
    mut optimizer: O,
    train_dataset: Dataset<T>,
    valid_dataset: Option<Dataset<T>>,
    n_epochs: u32,
    batch_size: usize,
    label_vocab: Option<&Vocab>,
    save_to: Option<P>,
    logger: &Logger,
) -> Result<(), Box<Error + Send + Sync>>
where
    F: FnMut(&mut M, Vec<&T>, bool) -> FO,
    FO: Into<training::ForwardFnOutput<Option<(Vec<ParserOutput>, Vec<*const S>)>>>,
    M: Model + Serialize + DeserializeOwned,
{
    optimizer.add_model(&mut model);

    let saver = save_to.map(|path| {
        let arch_path = format!("{}-parser.arch.json", path.as_ref().to_str().unwrap());
        serialize::write_to(&model, arch_path, serialize::Format::Json).unwrap();
        let model_path = format!("{}-parser", path.as_ref().to_str().unwrap());
        let mut c = training::callbacks::Saver::new(&model, &model_path);
        c.set_interval(1);
        c.save_from(5);
        c.save_best(valid_dataset.is_some());
        c
    });

    let mut trainer =
        training::Trainer::new(optimizer, |batch, train| forward(&mut model, batch, train));
    let child_logger = Rc::new(logger.new(o!()));
    trainer.show_progress();
    trainer.enable_report(child_logger.clone(), 1);
    if let Some(v) = label_vocab {
        trainer.add_callback("evaluator", Evaluator::new(v, child_logger.clone()));
    }
    if let Some(c) = saver {
        trainer.add_callback("saver", c);
    }

    trainer.fit(train_dataset, valid_dataset, n_epochs, batch_size);
    Ok(())
}
