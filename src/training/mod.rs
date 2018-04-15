use std::error;
use std::fmt;
use std::io::Stderr;
use std::marker::PhantomData;
use std::ops;

use primitiv::{Graph, Node, Optimizer};
use slog::Logger;

use dataset::Dataset;

pub mod callbacks;

#[derive(Debug)]
pub struct ForwardFnOutput<U> {
    loss: Node,
    accuracy: Option<Accuracy>,
    other: U,
}

/// `forward` must return one of the following types:
/// - loss: Node
/// - (loss: Node, accuracy: A) where A: Into<Accuracy>
/// - (loss: Node, accuracy: ()) where A: Into<Accuracy>
/// - (loss: Node, accuracy: Option<A>) where A: Into<Accuracy>
/// - (loss: Node, accuracy: A, other1: U1, other2: U2, ...) where A: Into<Accuracy>
/// - (loss: Node, accuracy: (), other1: U1, other2: U2, ...)
/// - (loss: Node, accuracy: Option<A>, other1: U1, other2: U2, ...) where A: Into<Accuracy>

impl From<Node> for ForwardFnOutput<()> {
    fn from(x: Node) -> Self {
        ForwardFnOutput {
            loss: x,
            accuracy: None,
            other: (),
        }
    }
}

macro_rules! impl_from_for_output {
    ($($field:tt:$type:tt),*) => {
        impl<A: Into<Accuracy>, $($type),*> From<(Node, A, $($type),*)>
            for ForwardFnOutput<($($type),*)> {
            fn from(x: (Node, A, $($type),*)) -> Self {
                ForwardFnOutput {
                    loss: x.0,
                    accuracy: Some(x.1.into()),
                    other: ($(x.$field),*),
                }
            }
        }

        impl<$($type),*> From<(Node, (), $($type),*)>
            for ForwardFnOutput<($($type),*)> {
            fn from(x: (Node, (), $($type),*)) -> Self {
                ForwardFnOutput {
                    loss: x.0,
                    accuracy: None,
                    other: ($(x.$field),*),
                }
            }
        }

        impl<A: Into<Accuracy>, $($type),*> From<(Node, Option<A>, $($type),*)>
            for ForwardFnOutput<($($type),*)> {
            fn from(x: (Node, Option<A>, $($type),*)) -> Self {
                ForwardFnOutput {
                    loss: x.0,
                    accuracy: x.1.map(|x1| x1.into()),
                    other: ($(x.$field),*),
                }
            }
        }
    };
}

impl_from_for_output!(2: U2, 3: U3, 4: U4, 5: U5, 6: U6, 7: U7);
impl_from_for_output!(2: U2, 3: U3, 4: U4, 5: U5, 6: U6);
impl_from_for_output!(2: U2, 3: U3, 4: U4, 5: U5);
impl_from_for_output!(2: U2, 3: U3, 4: U4);
impl_from_for_output!(2: U2, 3: U3);
impl_from_for_output!(2: U2);
impl_from_for_output!();

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
        let mut epoch_accuracy = Accuracy::new(0, 0);
        for (batch_index, batch) in dataset.batch(batch_size, train).enumerate() {
            info.batch_size = Some(batch.len());
            info.batch_index = Some(batch_index);
            self.notify(Event::BatchBegin, &info);

            g.clear();
            let ret = (self.forward)(batch, train).into();
            let (loss, accuracy, output) = (ret.loss, ret.accuracy, ret.other);
            let loss_value = loss.to_float();
            epoch_loss += loss_value;
            if let Some(ref acc) = accuracy {
                epoch_accuracy += acc;
            }
            if train {
                self.optimizer.reset_gradients();
                loss.backward();
                self.optimizer.update();
            }

            info.batch_loss = Some(loss_value);
            info.batch_accuracy = accuracy;
            info.output = Some(output);
            self.notify(Event::BatchEnd, &info);
        }

        info.loss = Some(epoch_loss);
        if epoch_accuracy.total() > 0 {
            info.accuracy = Some(epoch_accuracy);
        }
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

    pub fn enable_report(&mut self, logger: Logger) {
        self.add_callback("reporter", callbacks::Reporter::new(logger));
    }

    pub fn show_progress(&mut self) {
        self.add_callback("progressbar", callbacks::ProgressBar::<Stderr>::new());
    }
}

#[derive(Debug, Clone)]
pub struct TrainingInfo<U> {
    pub n_epochs: u32,
    pub epoch: u32,
    pub data_size: usize,
    pub train: bool,
    pub loss: Option<f32>,
    pub accuracy: Option<Accuracy>,
    pub batch_size: Option<usize>,
    pub batch_index: Option<usize>,
    pub batch_loss: Option<f32>,
    pub batch_accuracy: Option<Accuracy>,
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
            accuracy: None,
            batch_size: None,
            batch_index: None,
            batch_loss: None,
            batch_accuracy: None,
            output: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Accuracy {
    correct: usize,
    total: usize,
}

impl Accuracy {
    pub fn new(correct: usize, total: usize) -> Self {
        Accuracy {
            correct: correct,
            total: total,
        }
    }

    fn accuracy(&self) -> Result<f32, Error> {
        if self.total == 0 {
            return Err(Error::ZeroDivision);
        }
        Ok((self.correct as f64 / self.total as f64) as f32)
    }

    #[allow(dead_code)]
    #[inline]
    fn correct(&self) -> usize {
        self.correct
    }

    #[inline]
    fn total(&self) -> usize {
        self.total
    }
}

impl fmt::Display for Accuracy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.accuracy() {
            Ok(acc) => write!(f, "accuracy: {} / {} = {}", self.correct, self.total, acc),
            Err(_) => write!(f, "accuracy: {} / {} = Nan", self.correct, self.total),
        }
    }
}

impl ops::Add<Accuracy> for Accuracy {
    type Output = Accuracy;

    fn add(self, other: Accuracy) -> Self::Output {
        Accuracy {
            correct: self.correct + other.correct,
            total: self.total + other.total,
        }
    }
}

impl<'a> ops::Add<Accuracy> for &'a Accuracy {
    type Output = Accuracy;

    fn add(self, other: Accuracy) -> Self::Output {
        Accuracy {
            correct: self.correct + other.correct,
            total: self.total + other.total,
        }
    }
}

impl<'a> ops::Add<&'a Accuracy> for Accuracy {
    type Output = Accuracy;

    fn add(self, other: &'a Accuracy) -> Self::Output {
        Accuracy {
            correct: self.correct + other.correct,
            total: self.total + other.total,
        }
    }
}

impl<'a, 'b> ops::Add<&'b Accuracy> for &'a Accuracy {
    type Output = Accuracy;

    fn add(self, other: &'b Accuracy) -> Self::Output {
        Accuracy {
            correct: self.correct + other.correct,
            total: self.total + other.total,
        }
    }
}

impl ops::AddAssign<Accuracy> for Accuracy {
    fn add_assign(&mut self, other: Accuracy) {
        *self = Accuracy {
            correct: self.correct + other.correct,
            total: self.total + other.total,
        };
    }
}

impl<'a> ops::AddAssign<&'a Accuracy> for Accuracy {
    fn add_assign(&mut self, other: &'a Accuracy) {
        *self = Accuracy {
            correct: self.correct + other.correct,
            total: self.total + other.total,
        };
    }
}

macro_rules! impl_from_for_accuracy {
    ($type:ty) => {
        impl From<($type, $type)> for Accuracy {
            fn from(x: ($type, $type)) -> Self {
                Accuracy::new(x.0 as usize, x.1 as usize)
            }
        }
    };
}

impl_from_for_accuracy!(u8);
impl_from_for_accuracy!(u16);
impl_from_for_accuracy!(u32);
impl_from_for_accuracy!(u64);
impl_from_for_accuracy!(usize);

#[derive(Debug)]
pub enum Error {
    ZeroDivision,
}

impl Error {
    pub fn as_str(&self) -> &'static str {
        match *self {
            Error::ZeroDivision => "division by zero",
        }
    }
}

impl error::Error for Error {
    fn description(&self) -> &str {
        self.as_str()
    }

    fn cause(&self) -> Option<&error::Error> {
        None
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.as_str())
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
