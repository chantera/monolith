use std::io::{stderr, Stderr, stdout, Stdout, Write};

use pbr::ProgressBar as ProgressBarImpl;
use slog::Logger;

use training::{Accuracy, Callback, TrainingInfo};

#[derive(Debug)]
pub struct Reporter {
    logger: Logger,
}

impl Reporter {
    pub fn new(logger: Logger) -> Self {
        Reporter { logger: logger }
    }

    pub fn report(
        &self,
        label: &str,
        epoch: u32,
        n_samples: usize,
        loss: f32,
        accuracy: Option<&Accuracy>,
    ) {
        match accuracy {
            Some(acc) => {
                match acc.accuracy() {
                    Ok(acc_value) => {
                        info!(
                            self.logger,
                            "[{}] epoch {} - #samples: {}, loss: {:.8}, accuracy: {:.8}",
                            label,
                            epoch,
                            n_samples,
                            loss,
                            acc_value,
                        );
                    }
                    Err(_) => {
                        info!(
                            self.logger,
                            "[{}] epoch {} - #samples: {}, loss: {:.8}, accuracy: NaN",
                            label,
                            epoch,
                            n_samples,
                            loss
                        );
                    }
                }
            }
            None => {
                info!(
                    self.logger,
                    "[{}] epoch {} - #samples: {}, loss: {:.8}",
                    label,
                    epoch,
                    n_samples,
                    loss
                );
            }
        }
    }
}

impl<U> Callback<U> for Reporter {
    fn on_epoch_train_end(&mut self, info: &TrainingInfo<U>) {
        self.report(
            "training",
            info.epoch,
            info.data_size,
            info.loss.unwrap(),
            info.accuracy.as_ref(),
        );
    }

    fn on_epoch_validate_end(&mut self, info: &TrainingInfo<U>) {
        self.report(
            "validation",
            info.epoch,
            info.data_size,
            info.loss.unwrap(),
            info.accuracy.as_ref(),
        );
    }
}

pub struct ProgressBar<W: Write> {
    pbar: Option<ProgressBarImpl<W>>,
}

impl<W: Write> ProgressBar<W> {
    pub fn new() -> Self {
        ProgressBar { pbar: None }
    }

    pub fn init(&mut self, handle: W, total: u64) {
        self.pbar = Some(ProgressBarImpl::on(handle, total));
    }

    pub fn add(&mut self, i: u64) -> u64 {
        self.pbar.as_mut().unwrap().add(i)
    }

    pub fn finish(&mut self) {
        self.pbar.as_mut().unwrap().finish();
    }
}

impl<U> Callback<U> for ProgressBar<Stdout> {
    fn on_epoch_train_begin(&mut self, info: &TrainingInfo<U>) {
        self.init(stdout(), info.data_size as u64);
        self.pbar.as_mut().unwrap().tick()
    }

    fn on_batch_end(&mut self, info: &TrainingInfo<U>) {
        self.add(info.batch_size.unwrap() as u64);
    }

    fn on_epoch_train_end(&mut self, _info: &TrainingInfo<U>) {
        self.finish();
    }
}

impl<U> Callback<U> for ProgressBar<Stderr> {
    fn on_epoch_train_begin(&mut self, info: &TrainingInfo<U>) {
        self.init(stderr(), info.data_size as u64);
        self.pbar.as_mut().unwrap().tick()
    }

    fn on_batch_end(&mut self, info: &TrainingInfo<U>) {
        self.add(info.batch_size.unwrap() as u64);
    }

    fn on_epoch_train_end(&mut self, _info: &TrainingInfo<U>) {
        self.finish();
    }
}
