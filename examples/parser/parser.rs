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
use monolith::utils::primitiv as primitiv_utils;
use primitiv::{devices, optimizers};
use slog::Logger;

use self::systems::System;
mod dataset;
mod models;
mod systems;

fn train<P1, P2, P3, P4>(
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
    let system = System::ChenManning14;
    match system {
        System::ChenManning14 => {
            let (train_dataset, valid_dataset, preprocessor): (
                _,
                _,
                systems::chen_manning_14::Preprocessor,
            ) = dataset::load(
                train_file,
                valid_file,
                embed_file,
                save_to.as_ref().map(|path| path.as_ref().to_path_buf()),
                logger,
            )?;

            let builder = {
                let word_vocab = preprocessor.word_vocab();
                let mut builder = systems::chen_manning_14::ParserBuilder::default()
                    .postag(preprocessor.postag_vocab().unwrap().size(), 50)
                    .label(preprocessor.label_vocab().size(), 50)
                    .mlp(200)
                    .dropout(0.5);
                info!(logger, "builder: {:?}", builder);
                builder = if word_vocab.has_embed() {
                    builder.word_embed(word_vocab.embed()?)
                } else {
                    builder.word(word_vocab.size(), 50)
                };
                builder
            };

            let word_pad_id = preprocessor.word_pad_id();
            let postag_pad_id = preprocessor.postag_pad_id();
            let label_pad_id = preprocessor.label_pad_id();

            models::train(
                |model, batch, train: bool| {
                    take_cols!((features:0, eval_data:1, actions:2); batch, batch_size);
                    let ys = model.forward(features, train);
                    let loss = model.loss(&ys, &actions);
                    let accuracy = model.accuracy(&ys, &actions);
                    if !train {
                        let eval_data: Vec<&(Vec<u32>, Vec<u32>, _)> =
                            eval_data.into_iter().map(|x| x.as_ref().unwrap()).collect();
                        take_cols!((word_ids:0, postag_ids:1, sentences:2); eval_data, batch_size);
                        let predicted_heads_and_labels = model.parse(
                            word_ids,
                            postag_ids,
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
                optimizers::AdaGrad::new(learning_rate, 1e-8),
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

fn test<P: AsRef<Path>>(_file: P) -> Result<(), Box<Error + Send + Sync>> {
    Ok(())
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
    #[structopt(long = "lrate", default_value = "0.001")]
    learning_rate: f32,
    /// Number of sweeps over the dataset to train
    #[structopt(long = "epoch", default_value = "20")]
    n_epochs: u32,
    /// Directory for saved models
    #[structopt(long = "save", parse(from_os_str))]
    save_to: Option<PathBuf>,
}

#[derive(StructOpt, Debug)]
struct Test {
    /// A testing data file
    #[structopt(name = "INPUT", parse(from_os_str))]
    input: PathBuf,
    /// GPU device ID (negative value indicates CPU)
    #[structopt(long = "device", default_value = "-1")]
    device: i32,
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
            &c.input,
            c.valid_file.as_ref(),
            c.embed_file.as_ref(),
            c.n_epochs,
            c.batch_size,
            c.learning_rate,
            output_path,
            &context.logger,
        )
    }
    Command::Test(ref c) => {
        let mut dev = primitiv_utils::select_device(c.device);
        info!(&context.logger, "execute subcommand: {:?}", c);
        devices::set_default(&mut *dev);
        test(&c.input)
    }
});
