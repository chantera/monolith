#[macro_use]
extern crate monolith;
#[macro_use]
extern crate primitiv;
#[macro_use]
extern crate slog;

use std::error::Error;
use std::path::{Path, PathBuf};
use std::result::Result;

use monolith::app::prelude::*;
use monolith::preprocessing::Vocab;
use monolith::training::Trainer;
use monolith::training::callbacks::Saver;
use monolith::utils::primitiv as primitiv_utils;
use primitiv::*;
use slog::Logger;

use self::dataset::*;
use self::models::*;
mod dataset;
mod models;

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
    let (train_dataset, valid_dataset, mut model) = {
        let mut loader = Loader::new(Preprocessor::new(match embed_file {
            Some(f) => {
                eprint!(
                    "load embedding from `{}` ... ",
                    f.as_ref().to_str().unwrap()
                );
                let v = Vocab::from_cache_or_file(f, "<UNK>")?;
                eprintln!("done.");
                v
            }
            None => Vocab::new(),
        }));

        let train_dataset = loader.load(train_file)?;
        loader.fix();
        let valid_dataset = match valid_file {
            Some(f) => Some(loader.load(f)?),
            None => None,
        };
        let preprocessor = loader.dispose();
        let word_vocab = preprocessor.word_vocab();

        let mut builder = ParserBuilder::<ChenManning14Model>::default()
            .postag(preprocessor.postag_vocab().size(), 50)
            .label(preprocessor.label_vocab().size(), 50)
            .mlp(200)
            .dropout(0.5);
        info!(logger, "builder: {:?}", builder);
        builder = if word_vocab.has_embed() {
            builder.word_embed(word_vocab.embed()?)
        } else {
            builder.word(word_vocab.size(), 50)
        };
        let mut model = builder.build();
        (train_dataset, valid_dataset, model)
    };

    let mut optimizer = optimizers::AdaGrad::new(learning_rate, 1e-8);
    optimizer.add_model(&mut model);
    let saver = save_to.map(|dir| {
        let mut c = Saver::new(&model, dir, "parser");
        c.set_interval(1);
        c.save_from(10);
        c.save_best(true);
        c
    });

    let mut trainer = Trainer::new(optimizer, |batch: Vec<
        &(Vec<ChenManning14Feature>,
          Vec<u32>),
    >,
     train: bool| {
        take_cols!((features:0, actions:1); batch, batch_size);
        let ys = model.forward(features, train);
        let loss = model.loss(&ys, &actions);
        let accuracy = model.accuracy(&ys, &actions);
        (loss, accuracy)
    });
    trainer.show_progress();
    trainer.enable_report(logger.new(o!()), 1);
    if let Some(c) = saver {
        trainer.add_callback("saver", c);
    }
    trainer.fit(train_dataset, valid_dataset, n_epochs, batch_size);
    Ok(())
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
        println!("train with a file: {:?}", c.input);
        let mut dev = primitiv_utils::select_device(c.device);
        devices::set_default(&mut *dev);
        train(
            &c.input,
            c.valid_file.as_ref(),
            c.embed_file.as_ref(),
            c.n_epochs,
            c.batch_size,
            c.learning_rate,
            c.save_to.as_ref(),
            &context.logger,
        )
    }
    Command::Test(ref c) => {
        println!("test with a file: {:?}", c.input);
        let mut dev = primitiv_utils::select_device(0);
        devices::set_default(&mut *dev);
        test(&c.input)
    }
});