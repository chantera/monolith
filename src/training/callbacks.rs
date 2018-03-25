use std::io::{stderr, Stderr, stdout, Stdout, Write};

use pbr::ProgressBar as ProgressBarImpl;
use slog::Logger;

use training::{Callback, TrainingInfo};

#[derive(Debug)]
pub struct Reporter {
    logger: Logger,
}

impl Reporter {
    pub fn new(logger: Logger) -> Self {
        Reporter { logger: logger }
    }
}

impl Callback for Reporter {
    fn on_epoch_train_end(&mut self, info: &TrainingInfo) {
        info!(
            self.logger,
            "[training] epoch {} - #samples: {}, loss: {:.8}",
            info.epoch,
            info.data_size,
            info.loss.unwrap(),
        );
    }

    fn on_epoch_validate_end(&mut self, info: &TrainingInfo) {
        info!(
            self.logger,
            "[validation] epoch {} - #samples: {}, loss: {:.8}",
            info.epoch,
            info.data_size,
            info.loss.unwrap(),
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

impl Callback for ProgressBar<Stdout> {
    fn on_epoch_train_begin(&mut self, info: &TrainingInfo) {
        self.init(stdout(), info.data_size as u64);
        self.pbar.as_mut().unwrap().tick()
    }

    fn on_batch_end(&mut self, info: &TrainingInfo) {
        self.add(info.batch_size.unwrap() as u64);
    }

    fn on_epoch_train_end(&mut self, _info: &TrainingInfo) {
        self.finish();
    }
}

impl Callback for ProgressBar<Stderr> {
    fn on_epoch_train_begin(&mut self, info: &TrainingInfo) {
        self.init(stderr(), info.data_size as u64);
        self.pbar.as_mut().unwrap().tick()
    }

    fn on_batch_end(&mut self, info: &TrainingInfo) {
        self.add(info.batch_size.unwrap() as u64);
    }

    fn on_epoch_train_end(&mut self, _info: &TrainingInfo) {
        self.finish();
    }
}
