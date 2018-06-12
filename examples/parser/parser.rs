#[macro_use]
extern crate monolith;
#[macro_use]
extern crate primitiv;
extern crate regex;
extern crate serde;
#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate slog;

use std::error::Error;
use std::path::{Path, PathBuf};

use monolith::app::prelude::*;
use monolith::lang::Phrasal;
use monolith::utils::primitiv as primitiv_utils;
use primitiv::{devices, optimizers, Optimizer};
use slog::Logger;

use self::systems::System;
mod dataset;
mod models;
mod systems;

pub fn train<P1, P2, P3, P4>(
    system: System,
    train_file: P1,
    valid_file: Option<P2>,
    embed_file: Option<P3>,
    n_epochs: u32,
    batch_size: usize,
    learning_rate: f32,
    save_to: Option<P4>,
    logger: &Logger,
) -> Result<(), Box<Error + Send + Sync>>
where
    P1: AsRef<Path>,
    P2: AsRef<Path>,
    P3: AsRef<Path>,
    P4: AsRef<Path>,
{
    macro_rules! load_dataset {
        ($module:ident) => {
            dataset::load_train_dataset::<systems::$module::Preprocessor, _, _, _, _>(
                train_file,
                valid_file,
                embed_file,
                save_to.as_ref().map(|path| {
                    if path.as_ref().is_dir() {
                        path.as_ref().join("loader.json")
                    } else {
                        format!("{}-loader.json", path.as_ref().to_str().unwrap()).into()
                    }
                }),
                logger,
            )
        };
    }
    match system {
        System::ChenManning14 => {
            let (train_dataset, valid_dataset, preprocessor) = load_dataset!(chen_manning_14)?;
            let builder = {
                let word_vocab = preprocessor.word_vocab();
                let mut builder = systems::chen_manning_14::ParserBuilder::default()
                    .word(word_vocab.size() as u32, 50, None)
                    .postag(preprocessor.postag_vocab().size() as u32, 50, None)
                    .label(preprocessor.label_vocab().size() as u32, 50, None);
                info!(logger, "builder: {:?}", builder);
                if word_vocab.has_embed() {
                    builder = builder.word_embed(word_vocab.embed()?, None)
                }
                builder
            };
            let mut optimizer = optimizers::AdaGrad::new(learning_rate, 1e-8);
            optimizer.set_weight_decay(1e-8);

            let word_pad_id = preprocessor.word_pad_id();
            let postag_pad_id = preprocessor.postag_pad_id();
            let label_pad_id = preprocessor.label_pad_id();

            models::train(
                |model, batch, train: bool| {
                    take_cols!((features:0, eval_data:1, actions:2); batch, batch_size);
                    let ys = model.forward(&features, train);
                    let loss = model.loss(&ys, &actions);
                    let accuracy = model.accuracy(&ys, &actions);
                    if !train {
                        let eval_data: Vec<&(Vec<u32>, Vec<u32>, _)> =
                            eval_data.into_iter().map(|x| x.as_ref().unwrap()).collect();
                        take_cols!((word_ids:0, postag_ids:1, sentences:2); eval_data, batch_size);
                        let predicted_heads_and_labels = model.parse(
                            &word_ids,
                            &postag_ids,
                            word_pad_id,
                            postag_pad_id,
                            label_pad_id,
                        );
                        let sentences: Vec<*const _> =
                            sentences.into_iter().map(|x| x as *const _).collect();
                        (
                            loss,
                            accuracy,
                            Some((predicted_heads_and_labels, sentences)),
                        )
                    } else {
                        (loss, accuracy, None)
                    }
                },
                builder.build(),
                optimizer,
                train_dataset,
                valid_dataset,
                n_epochs,
                batch_size,
                Some(preprocessor.label_vocab()),
                save_to,
                logger,
            )
        }
        System::KiperwasserGoldberg16Transition => {
            let (train_dataset, valid_dataset, preprocessor) =
                load_dataset!(kiperwasser_goldberg_16_transition)?;
            let builder = {
                let word_vocab = preprocessor.word_vocab();
                let word_pad_id = preprocessor.word_pad_id();
                let postag_vocab = preprocessor.postag_vocab();
                let postag_pad_id = preprocessor.postag_pad_id();
                let mut builder =
                    systems::kiperwasser_goldberg_16_transition::ParserBuilder::default()
                        .word(word_vocab.size() as u32, 100, Some(word_pad_id))
                        .postag(postag_vocab.size() as u32, 25, Some(postag_pad_id))
                        .out(preprocessor.label_vocab().size() as u32 * 2 + 1);
                info!(logger, "builder: {:?}", builder);
                if word_vocab.has_embed() {
                    builder = builder.word_embed(word_vocab.embed()?, Some(word_pad_id))
                }
                builder
            };
            let optimizer = optimizers::Adam::new(learning_rate, 0.9, 0.999, 1e-8);

            models::train(
                |model, mut batch, train: bool| {
                    sort_batch!(batch);
                    take_cols!((words:0, postags:1, features:2, sentences:3, actions:4); batch, batch_size);
                    transpose!(words, postags);
                    let ys = model.forward(&words, &postags, &features, train);
                    let loss = model.loss(&ys, &actions);
                    let accuracy = model.accuracy(&ys, &actions);
                    if !train {
                        let predicted_heads_and_labels = model.parse_with_cache(&words, &postags);
                        let sentences: Vec<*const _> = sentences
                            .into_iter()
                            .map(|x| x.as_ref().unwrap() as *const _)
                            .collect();
                        (
                            loss,
                            accuracy,
                            Some((predicted_heads_and_labels, sentences)),
                        )
                    } else {
                        (loss, accuracy, None)
                    }
                },
                builder.build(),
                optimizer,
                train_dataset,
                valid_dataset,
                n_epochs,
                batch_size,
                Some(preprocessor.label_vocab()),
                save_to,
                logger,
            )
        }
        System::DozatManning17 => {
            let (train_dataset, valid_dataset, preprocessor) = load_dataset!(dozat_manning_17)?;
            let builder = {
                let word_vocab = preprocessor.word_vocab();
                let mut builder = systems::dozat_manning_17::ParserBuilder::default()
                    .word(word_vocab.size() as u32, 100, None)
                    .postag(preprocessor.postag_vocab().size() as u32, 100, None)
                    .label(preprocessor.label_vocab().size() as u32, 0, None);
                info!(logger, "builder: {:?}", builder);
                if word_vocab.has_embed() {
                    builder = builder.word_embed(word_vocab.embed()?, None)
                }
                builder
            };
            let optimizer = optimizers::Adam::new(learning_rate, 0.9, 0.999, 1e-8);

            models::train(
                |model, mut batch, train: bool| {
                    sort_batch!(batch);
                    take_cols!((words:0, postags:1, sentences:2, ts:3); batch, batch_size);
                    take_cols!((heads:0, labels:1); ts, batch_size);
                    transpose!(words, postags);
                    let ys = model.forward(&words, &postags, train);
                    let loss = model.loss(&ys.0, &ys.1, &heads, &labels);
                    let accuracy = model.accuracy(&ys.0, &ys.1, &heads, &labels);
                    if !train {
                        let mut lengths = Vec::with_capacity(sentences.len());
                        let sentences: Vec<*const _> = sentences
                            .into_iter()
                            .map(|x| {
                                let s = x.as_ref().unwrap();
                                lengths.push(s.len());
                                s as *const _
                            })
                            .collect();
                        let predicted_heads_and_labels = model.parse_from_scores(
                            &ys.0,
                            &ys.1,
                            &lengths,
                            systems::dozat_manning_17::GraphAlgorithm::None,
                        );
                        (
                            loss,
                            accuracy,
                            Some((predicted_heads_and_labels, sentences)),
                        )
                    } else {
                        (loss, accuracy, None)
                    }
                },
                builder.build(),
                optimizer,
                train_dataset,
                valid_dataset,
                n_epochs,
                batch_size,
                Some(preprocessor.label_vocab()),
                save_to,
                logger,
            )
        }
    }
}

pub fn test<P1: AsRef<Path>, P2: AsRef<Path>>(
    system: System,
    test_file: P1,
    model_file: P2,
    logger: &Logger,
) -> Result<(), Box<Error + Send + Sync>> {
    let loader_file = {
        let dir = model_file.as_ref().parent().unwrap().to_str().unwrap();
        let s = model_file.as_ref().file_name().unwrap().to_str().unwrap();
        match s.rsplitn(2, '-').skip(1).next() {
            Some(prefix) => format!("{}/{}-loader.json", dir, prefix),
            None => format!("{}/loader.json", dir),
        }
    };
    macro_rules! load_dataset {
        ($module:ident) => {
            dataset::load_test_dataset::<systems::$module::Preprocessor, _, _>(
                test_file,
                loader_file,
                logger,
            )
        };
    }
    match system {
        System::ChenManning14 => {
            let (test_dataset, preprocessor) = load_dataset!(chen_manning_14)?;

            let word_pad_id = preprocessor.word_pad_id();
            let postag_pad_id = preprocessor.postag_pad_id();
            let label_pad_id = preprocessor.label_pad_id();

            models::test(
                |model: &mut systems::chen_manning_14::ChenManning14Model, batch| {
                    take_cols!((features:0, eval_data:1, actions:2); batch);
                    let ys = model.forward(&features, false);
                    let loss = model.loss(&ys, &actions);
                    let accuracy = model.accuracy(&ys, &actions);
                    let eval_data: Vec<&(Vec<u32>, Vec<u32>, _)> =
                        eval_data.into_iter().map(|x| x.as_ref().unwrap()).collect();
                    take_cols!((word_ids:0, postag_ids:1, sentences:2); eval_data);
                    let predicted_heads_and_labels = model.parse(
                        &word_ids,
                        &postag_ids,
                        word_pad_id,
                        postag_pad_id,
                        label_pad_id,
                    );
                    let sentences: Vec<*const _> =
                        sentences.into_iter().map(|x| x as *const _).collect();
                    (loss, accuracy, (predicted_heads_and_labels, sentences))
                },
                test_dataset,
                model_file,
                preprocessor.label_vocab(),
                logger,
            )
        }
        System::KiperwasserGoldberg16Transition => Ok(()),
        System::DozatManning17 => Ok(()),
    }
}

#[derive(StructOpt, Debug)]
#[structopt(name = "parser")]
struct Args {
    #[structopt(flatten)]
    common: CommonArgs,
    #[structopt(subcommand)]
    command: Command,
}

#[derive(StructOpt, Debug)]
enum Command {
    #[structopt(name = "train", about = "Trains model")]
    Train(Train),
    #[structopt(name = "test", about = "Tests model")]
    Test(Test),
}

#[derive(StructOpt, Debug)]
struct Train {
    /// A training data file
    #[structopt(name = "INPUT", parse(from_os_str))]
    input: PathBuf,
    /// A validation data file
    #[structopt(name = "VFILE", parse(from_os_str))]
    valid_file: Option<PathBuf>,
    /// Number of examples in each mini-batch
    #[structopt(long = "batch", default_value = "32")]
    batch_size: usize,
    /// GPU device ID (negative value indicates CPU)
    #[structopt(long = "device", default_value = "-1")]
    device: i32,
    /// A file of pretrained word embeddings
    #[structopt(long = "embed", parse(from_os_str))]
    embed_file: Option<PathBuf>,
    /// Learning rate for an optimizer
    #[structopt(long = "lrate")]
    learning_rate: Option<f32>,
    /// Number of sweeps over the dataset to train
    #[structopt(long = "epoch", default_value = "20")]
    n_epochs: u32,
    /// Directory for saved models
    #[structopt(long = "save", parse(from_os_str))]
    save_to: Option<PathBuf>,
    /// Parser system
    #[structopt(long = "system", default_value = "cm14", parse(try_from_str = "System::from_str"))]
    system: System,
}

#[derive(StructOpt, Debug)]
struct Test {
    /// A testing data file
    #[structopt(name = "INPUT", parse(from_os_str))]
    input: PathBuf,
    /// A model file
    #[structopt(name = "MODEL", parse(from_os_str))]
    model: PathBuf,
    /// GPU device ID (negative value indicates CPU)
    #[structopt(long = "device", default_value = "-1")]
    device: i32,
    /// Parser system
    #[structopt(long = "system", default_value = "cm14", parse(try_from_str = "System::from_str"))]
    system: System,
}

main!(|args: Args, context: Context| match args.command {
    Command::Train(ref c) => {
        let mut dev = primitiv_utils::select_device(c.device);
        devices::set_default(&mut *dev);
        let output_path = c.save_to.as_ref().map(|dir| {
            let mut path = dir.clone();
            path.push(format!(
                "{}-{}",
                context.accesstime.format("%Y%m%d"),
                context.accessid
            ));
            path
        });
        info!(&context.logger, "execute subcommand: {:?}", c);
        train(
            c.system,
            &c.input,
            c.valid_file.as_ref(),
            c.embed_file.as_ref(),
            c.n_epochs,
            c.batch_size,
            c.learning_rate.unwrap_or(c.system.default_learning_rate()),
            output_path,
            &context.logger,
        )
    }
    Command::Test(ref c) => {
        let mut dev = primitiv_utils::select_device(c.device);
        info!(&context.logger, "execute subcommand: {:?}", c);
        devices::set_default(&mut *dev);
        test(c.system, &c.input, &c.model, &context.logger)
    }
});
