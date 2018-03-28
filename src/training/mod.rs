use std::io::Stderr;
use std::marker::PhantomData;

use primitiv::{Graph, Node, Optimizer};
use slog::Logger;

use dataset::Dataset;

pub mod callbacks;

pub struct ForwardFnOutput<U>(Node, U);

impl<U> From<(Node, U)> for ForwardFnOutput<U> {
    fn from(x: (Node, U)) -> Self {
        ForwardFnOutput(x.0, x.1)
    }
}

impl From<Node> for ForwardFnOutput<Option<f32>> {
    fn from(x: Node) -> Self {
        ForwardFnOutput(x, None)
    }
}

pub struct Trainer<O, F, T, U> {
    optimizer: O,
    forward: F,
    _sample_type: PhantomData<T>,
    _output_type: PhantomData<U>,
    callbacks: Vec<(u32, usize, String, Box<Callback<U>>)>,
}

impl<O: Optimizer, F, T, U, FO> Trainer<O, F, T, U>
where
    F: FnMut(Vec<&T>, bool) -> FO,
    FO: Into<ForwardFnOutput<U>>,
{
    pub fn new(optimizer: O, forward: F) -> Self {
        Trainer {
            optimizer: optimizer,
            forward: forward,
            _sample_type: PhantomData,
            _output_type: PhantomData,
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

        let mut g = Graph::new();
        Graph::set_default(&mut g);

        let n_samples = train_dataset.len();
        self.notify(Event::TrainBegin, &TrainingInfo::new(n_epochs, n_samples));

        for epoch in 1..n_epochs + 1 {
            let mut train_info = TrainingInfo::new(n_epochs, n_samples);
            train_info.epoch = epoch;
            self.notify(Event::EpochBegin, &train_info);

            self.process_batches(&mut g, &train_dataset, batch_size, &mut train_info);
            if let Some(ref v_data) = valid_dataset {
                let mut valid_info = TrainingInfo::new(n_epochs, v_data.len());
                valid_info.epoch = epoch;
                valid_info.train = false;
                self.process_batches(&mut g, v_data, batch_size, &mut valid_info);
            }

            self.notify(Event::EpochEnd, &train_info);
        }

        self.notify(Event::TrainEnd, &TrainingInfo::new(n_epochs, n_samples));
    }

    fn process_batches(
        &mut self,
        g: &mut Graph,
        dataset: &Dataset<T>,
        batch_size: usize,
        info: &mut TrainingInfo<U>,
    ) {
        let train = info.train;
        self.notify(
            if train {
                Event::EpochTrainBegin
            } else {
                Event::EpochValidateBegin
            },
            &info,
        );

        let mut epoch_loss = 0.0;
        for (batch_index, batch) in dataset.batch(batch_size, train).enumerate() {
            info.batch_size = Some(batch.len());
            info.batch_index = Some(batch_index);
            self.notify(Event::BatchBegin, &info);

            g.clear();
            let ret = (self.forward)(batch, train).into();
            let (loss, output) = (ret.0, ret.1);
            let loss_value = loss.to_float();
            epoch_loss += loss_value;
            if train {
                self.optimizer.reset_gradients();
                loss.backward();
                self.optimizer.update();
            }

            info.batch_loss = Some(loss_value);
            info.output = Some(output);
            self.notify(Event::BatchEnd, &info);
        }

        info.loss = Some(epoch_loss);
        self.notify(
            if train {
                Event::EpochTrainEnd
            } else {
                Event::EpochValidateEnd
            },
            &info,
        );
    }

    pub fn add_callback<S: Into<String>, C: Callback<U> + 'static>(
        &mut self,
        name: S,
        callback: C,
    ) {
        self.add_callback_with_priority(name, callback, 1000);
    }

    pub fn add_callback_with_priority<S: Into<String>, C: Callback<U> + 'static>(
        &mut self,
        name: S,
        callback: C,
        priority: u32,
    ) {
        let name = name.into();
        self.remove_callback(&name);
        let index = self.callbacks.len();
        self.callbacks.push(
            (priority, index, name, Box::new(callback)),
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

    fn notify(&mut self, event: Event, info: &TrainingInfo<U>) {
        match event {
            Event::TrainBegin => {
                self.callbacks.iter_mut().for_each(
                    |cb| cb.3.on_train_begin(info),
                );
            }
            Event::TrainEnd => {
                self.callbacks.iter_mut().for_each(
                    |cb| cb.3.on_train_end(info),
                );
            }
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

    pub fn show_progress(&mut self) {
        self.add_callback("progressbar", callbacks::ProgressBar::<Stderr>::new());
    }
}

impl<O: Optimizer, F, T, U, FO> Trainer<O, F, T, Option<U>>
where
    F: FnMut(Vec<&T>, bool) -> FO,
    FO: Into<ForwardFnOutput<Option<U>>>,
{
    pub fn enable_report(&mut self, logger: Logger) {
        self.add_callback("reporter", callbacks::Reporter::new(logger));
    }
}

impl<O: Optimizer, F, T, FO> Trainer<O, F, T, (u32, u32)>
where
    F: FnMut(Vec<&T>, bool) -> FO,
    FO: Into<ForwardFnOutput<(u32, u32)>>,
{
    pub fn enable_report(&mut self, logger: Logger) {
        self.add_callback("reporter", callbacks::Reporter::new(logger));
    }
}

#[derive(Debug, Clone)]
pub struct TrainingInfo<U> {
    pub n_epochs: u32,
    pub epoch: u32,
    pub data_size: usize,
    pub train: bool,
    pub loss: Option<f32>,
    pub batch_size: Option<usize>,
    pub batch_index: Option<usize>,
    pub batch_loss: Option<f32>,
    pub output: Option<U>,
}

impl<U> TrainingInfo<U> {
    pub fn new(n_epochs: u32, data_size: usize) -> Self {
        TrainingInfo {
            n_epochs: n_epochs,
            epoch: 0,
            data_size: data_size,
            train: true,
            loss: None,
            batch_size: None,
            batch_index: None,
            batch_loss: None,
            output: None,
        }
    }
}

#[derive(Debug)]
enum Event {
    TrainBegin,
    TrainEnd,
    EpochBegin,
    EpochEnd,
    EpochTrainBegin,
    EpochTrainEnd,
    EpochValidateBegin,
    EpochValidateEnd,
    BatchBegin,
    BatchEnd,
}

pub trait Callback<U> {
    #[allow(unused_variables)]
    fn on_train_begin(&mut self, info: &TrainingInfo<U>) {}
    #[allow(unused_variables)]
    fn on_train_end(&mut self, info: &TrainingInfo<U>) {}
    #[allow(unused_variables)]
    fn on_epoch_begin(&mut self, info: &TrainingInfo<U>) {}
    #[allow(unused_variables)]
    fn on_epoch_end(&mut self, info: &TrainingInfo<U>) {}
    #[allow(unused_variables)]
    fn on_epoch_train_begin(&mut self, info: &TrainingInfo<U>) {}
    #[allow(unused_variables)]
    fn on_epoch_train_end(&mut self, info: &TrainingInfo<U>) {}
    #[allow(unused_variables)]
    fn on_epoch_validate_begin(&mut self, info: &TrainingInfo<U>) {}
    #[allow(unused_variables)]
    fn on_epoch_validate_end(&mut self, info: &TrainingInfo<U>) {}
    #[allow(unused_variables)]
    fn on_batch_begin(&mut self, info: &TrainingInfo<U>) {}
    #[allow(unused_variables)]
    fn on_batch_end(&mut self, info: &TrainingInfo<U>) {}
}
