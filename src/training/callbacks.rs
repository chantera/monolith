use std::io::{self as std_io, stderr, Stderr, stdout, Stdout, Write};
use std::path::{Path, PathBuf};
use std::rc::Rc;

use pbr::ProgressBar as ProgressBarImpl;
use primitiv::ModelImpl;
use slog::Logger;

use training::{Accuracy, Callback, TrainingInfo};
use utils;

#[derive(Debug)]
pub struct Reporter {
    logger: Rc<Logger>,
    interval: u32,
}

impl Reporter {
    pub fn new<L: Into<Rc<Logger>>>(logger: L, interval: u32) -> Self {
        Reporter {
            logger: logger.into(),
            interval: interval,
        }
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
        if info.epoch % self.interval == 0 {
            self.report(
                "training",
                info.epoch,
                info.data_size,
                info.loss.unwrap(),
                info.accuracy.as_ref(),
            );
        }
    }

    fn on_epoch_validate_end(&mut self, info: &TrainingInfo<U>) {
        if info.epoch % self.interval == 0 {
            self.report(
                "validation",
                info.epoch,
                info.data_size,
                info.loss.unwrap(),
                info.accuracy.as_ref(),
            );
        }
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

pub static MODEL_FILE_EXT: &'static str = "mdl";

#[derive(Debug)]
pub struct Saver<M: ModelImpl> {
    base_path: PathBuf,
    model: *const M,
    interval: u32,
    save_from: u32,
    save_best: bool,
    best_accuracy: f32,
}

impl<M: ModelImpl> Saver<M> {
    pub fn new<P: AsRef<Path>>(model: &M, dir: P, basename: &str) -> Self {
        let mut base_path = utils::path::expandtilde(dir);
        base_path.push(basename);
        base_path.set_extension(MODEL_FILE_EXT);
        Saver {
            model: model,
            base_path: base_path,
            interval: 1,
            save_from: 1,
            save_best: false,
            best_accuracy: 0.0,
        }
    }

    pub fn set_interval(&mut self, interval: u32) {
        self.interval = interval;
    }

    pub fn save_from(&mut self, epoch: u32) {
        self.save_from = epoch;
    }

    pub fn save_best(&mut self, enabled: bool) {
        self.save_best = enabled;
    }

    pub fn save(&self, id: Option<&str>) -> std_io::Result<()> {
        let model = unsafe { &*self.model };
        match id {
            Some(s) => {
                let ext = self.base_path.extension().unwrap().to_str().unwrap();
                let path = self.base_path.with_extension(format!("{}.{}", s, ext));
                eprintln!("saving the model to {} ...", path.to_str().unwrap());
                model.save(path, true)
            }
            None => {
                eprintln!(
                    "saving the model to {} ...",
                    self.base_path.to_str().unwrap()
                );
                model.save(&self.base_path, true)
            }
        }
    }
}

impl<U, M: ModelImpl> Callback<U> for Saver<M> {
    fn on_epoch_train_end(&mut self, info: &TrainingInfo<U>) {
        if !self.save_best && info.epoch >= self.save_from && info.epoch % self.interval == 0 {
            self.save(Some(&info.epoch.to_string()[..])).unwrap();
        }
    }

    fn on_epoch_validate_end(&mut self, info: &TrainingInfo<U>) {
        if self.save_best && info.epoch >= self.save_from {
            if self.interval > 1 {
                // save model to a new file every interval
                if let Some(ref acc) = info.accuracy {
                    if let Ok(value) = acc.accuracy() {
                        if value > self.best_accuracy {
                            let offset = ((info.epoch - 1) / self.interval) * self.interval + 1;
                            let id = format!(
                                "{}-{}",
                                offset,
                                (offset + self.interval - 1).min(info.n_epochs)
                            );
                            self.save(Some(&id)).unwrap();
                            self.best_accuracy = value;
                        }
                    }
                }
                if info.epoch % self.interval == 0 {
                    self.best_accuracy = 0.0;
                }
            } else {
                // save model to a new file (identical)
                if let Some(ref acc) = info.accuracy {
                    if let Ok(value) = acc.accuracy() {
                        if value > self.best_accuracy {
                            self.save(None).unwrap();
                            self.best_accuracy = value;
                        }
                    }
                }
            }
        }
    }
}
