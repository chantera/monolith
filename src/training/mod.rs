use std::fmt::Debug;
use std::marker::PhantomData;

use pbr::ProgressBar;
use primitiv::{Graph, Node, Optimizer};
use slog::Logger;

use dataset::Dataset;
use logging::{LoggerBuilder, Stream};

#[derive(Debug)]
pub struct Trainer<O, F, T> {
    optimizer: O,
    forward: F,
    _sample_type: PhantomData<T>,
    logger: Option<Logger>,
    callbacks: Vec<(u32, usize, String, Box<Callback>)>,
}

impl<O: Optimizer, F, T> Trainer<O, F, T>
where
    F: FnMut(Vec<&T>, bool) -> Node,
{
    pub fn new(optimizer: O, forward: F) -> Self {
        Trainer {
            optimizer: optimizer,
            forward: forward,
            _sample_type: PhantomData,
            logger: None,
            callbacks: vec![],
        }
    }

    pub fn fit(
        &mut self,
        train_dataset: Dataset<T>,
        valid_dataset: Option<Dataset<T>>,
        n_epochs: u32,
        batch_size: usize,
    ) {
        self.callbacks.sort_by(|cb1, cb2| {
            (-(cb1.0 as i32), cb1.1).cmp(&(-(cb2.0 as i32), cb2.1))
        });
        // if self.logger.is_none() {
        //     let logger = LoggerBuilder::new(Stream::StdOut).build(o!());
        //     self.logger = Some(logger);
        // }
        // let logger = self.logger.as_ref().unwrap();

        let mut g = Graph::new();
        Graph::set_default(&mut g);

        for epoch in 1..n_epochs + 1 {
            let mut train_info = TrainingInfo {
                n_epochs: n_epochs,
                epoch: epoch,
                data_size: train_dataset.len(),
                train: true,
                loss: None,
                batch_size: None,
                batch_index: None,
                batch_loss: None,
            };

            // info!(self.logger.as_ref()unwrap(), "epoch: {}", epoch);
            self.process_batches(&mut g, &train_dataset, batch_size, true);

            if let Some(ref v_data) = valid_dataset {
                let mut valid_info = TrainingInfo {
                    n_epochs: n_epochs,
                    epoch: epoch,
                    data_size: train_dataset.len(),
                    train: true,
                    loss: None,
                    batch_size: None,
                    batch_index: None,
                    batch_loss: None,
                };
                self.process_batches(&mut g, v_data, batch_size, false);
            }
        }
    }

    fn process_batches(
        &mut self,
        g: &mut Graph,
        dataset: &Dataset<T>,
        batch_size: usize,
        train: bool,
    ) {
        // let logger = self.logger.as_ref().unwrap();
        let mut pbar = ProgressBar::new(dataset.len() as u64);
        let mut train_loss = 0.0;
        for batch in dataset.batch(batch_size, train) {
            let size = batch.len();
            g.clear();
            self.optimizer.reset_gradients();
            let loss = (self.forward)(batch, train);
            train_loss += loss.to_float();
            loss.backward();
            self.optimizer.update();
            pbar.add(size as u64);
        }
        pbar.finish();
        // info!(logger, "loss: {}", train_loss);
    }

    pub fn add_callback<S: Into<String>, C: Into<Box<Callback>>>(&mut self, name: S, callback: C) {
        self.add_callback_with_priority(name, callback, 1000);
    }

    pub fn add_callback_with_priority<S: Into<String>, C: Into<Box<Callback>>>(
        &mut self,
        name: S,
        callback: C,
        priority: u32,
    ) {
        let name = name.into();
        self.remove_callback(&name);
        let index = self.callbacks.len();
        self.callbacks.push(
            (priority, index, name, callback.into()),
        );
    }

    pub fn has_callback(&self, name: &str) -> bool {
        self.callbacks.iter().any(|cb| &cb.2 == &name)
    }

    pub fn remove_callback(&mut self, name: &str) {
        let mut index: Option<usize> = None;
        for (i, cb) in self.callbacks.iter().enumerate() {
            if &cb.2 == &name {
                index = Some(i);
                break;
            }
        }
        if let Some(i) = index {
            self.callbacks.remove(i);
        }
    }

    pub fn set_logger(&mut self, logger: Logger) {
        self.logger = Some(logger);
    }

    fn notify(&mut self, event: Event, info: &TrainingInfo) {
        match event {
            Event::EpochBegin => {
                self.callbacks.iter_mut().for_each(
                    |cb| cb.3.on_epoch_begin(info),
                );
            }
            Event::EpochEnd => {
                self.callbacks.iter_mut().for_each(
                    |cb| cb.3.on_epoch_end(info),
                );
            }
            Event::EpochTrainBegin => {
                self.callbacks.iter_mut().for_each(|cb| {
                    cb.3.on_epoch_train_begin(info)
                });
            }
            Event::EpochTrainEnd => {
                self.callbacks.iter_mut().for_each(
                    |cb| cb.3.on_epoch_train_end(info),
                );
            }
            Event::EpochValidateBegin => {
                self.callbacks.iter_mut().for_each(|cb| {
                    cb.3.on_epoch_validate_begin(info)
                });
            }
            Event::EpochValidateEnd => {
                self.callbacks.iter_mut().for_each(|cb| {
                    cb.3.on_epoch_validate_end(info)
                });
            }
            Event::BatchBegin => {
                self.callbacks.iter_mut().for_each(
                    |cb| cb.3.on_batch_begin(info),
                );
            }
            Event::BatchEnd => {
                self.callbacks.iter_mut().for_each(
                    |cb| cb.3.on_batch_end(info),
                );
            }
        }
    }
    // y=None,
    // batch_size=32,
    // epochs=10,
    // validation_data=None,
    // verbose=True):
}

#[derive(Debug, Clone)]
pub struct TrainingInfo {
    pub n_epochs: u32,
    pub epoch: u32,
    pub data_size: usize,
    pub train: bool,
    pub loss: Option<f32>,
    pub batch_size: Option<usize>,
    pub batch_index: Option<usize>,
    pub batch_loss: Option<f32>,
}

#[derive(Debug)]
enum Event {
    EpochBegin,
    EpochEnd,
    EpochTrainBegin,
    EpochTrainEnd,
    EpochValidateBegin,
    EpochValidateEnd,
    BatchBegin,
    BatchEnd,
}

pub trait Callback: Debug {
    fn on_epoch_begin(&mut self, info: &TrainingInfo) {}
    fn on_epoch_end(&mut self, info: &TrainingInfo) {}
    fn on_epoch_train_begin(&mut self, info: &TrainingInfo) {}
    fn on_epoch_train_end(&mut self, info: &TrainingInfo) {}
    fn on_epoch_validate_begin(&mut self, info: &TrainingInfo) {}
    fn on_epoch_validate_end(&mut self, info: &TrainingInfo) {}
    fn on_batch_begin(&mut self, info: &TrainingInfo) {}
    fn on_batch_end(&mut self, info: &TrainingInfo) {}
}

impl<'a, C: Callback + 'a> From<C> for Box<Callback + 'a> {
    fn from(cb: C) -> Box<Callback + 'a> {
        Box::new(cb)
    }
}
