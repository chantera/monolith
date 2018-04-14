#[macro_use]
extern crate monolith;
#[macro_use]
extern crate primitiv;
#[macro_use]
extern crate slog;

use std::error::Error;
use std::path::Path;
use std::result::Result;

use monolith::app::prelude::*;
use monolith::preprocessing::Vocab;
use monolith::training::Trainer;
use primitiv::*;
use slog::Logger;

use self::dataset::*;
use self::models::*;
use self::utils::*;
mod dataset;
mod models;
mod utils;

fn train<P1: AsRef<Path>, P2: AsRef<Path>, P3: AsRef<Path>>(
    train_file: P1,
    valid_file: Option<P2>,
    embed_file: Option<P3>,
    n_epochs: u32,
    batch_size: usize,
    logger: &Logger,
) -> Result<(), Box<Error + Send + Sync>> {
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

        let mut model = TaggerBuilder::new()
            .word(preprocessor.word_vocab().size(), 100)
            .char(preprocessor.char_vocab().size(), 32)
            .lstm(200)
            .mlp(100)
            .dropout(0.5)
            .out(preprocessor.pos_vocab().size())
            .build();
        (train_dataset, valid_dataset, model)
    };

    let mut optimizer = optimizers::Adam::default();
    optimizer.set_weight_decay(1e-6);
    optimizer.set_gradient_clipping(5.0);
    optimizer.add_model(&mut model);

    let mut trainer = Trainer::new(optimizer, |mut batch: Vec<&Sample>, train: bool| {
        sort_batch!(batch);
        take_cols!((words:0, chars:1, postags:2); batch, batch_size);
        transpose!(words, chars, postags);
        let ys = model.forward(words, chars, train);
        let loss = model.loss(&ys, &postags);
        let accuracy = model.accuracy(&ys, &postags);
        (loss, accuracy)
    });
    trainer.show_progress();
    trainer.enable_report(logger.new(o!("child" => "test")));
    trainer.fit(train_dataset, valid_dataset, n_epochs, batch_size);

    Ok(())
}

fn test<P: AsRef<Path>>(_file: P) -> Result<(), Box<Error + Send + Sync>> {
    Ok(())
}

#[derive(StructOpt, Debug)]
#[structopt(name = "tagger")]
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
    #[structopt(name = "INPUT")]
    input: String,
    /// A validation data file
    #[structopt(name = "VFILE")]
    valid_file: Option<String>,
    /// Number of examples in each mini-batch
    #[structopt(long = "batch", default_value = "32")]
    batch_size: usize,
    /// GPU device ID (negative value indicates CPU)
    #[structopt(long = "device", default_value = "-1")]
    device: i32,
    /// A file of pretrained word embeddings
    #[structopt(long = "embed")]
    embed_file: Option<String>,
    /// Number of sweeps over the dataset to train
    #[structopt(long = "epoch", default_value = "20")]
    n_epochs: u32,
}

#[derive(StructOpt, Debug)]
struct Test {
    /// A testing data file
    input: String,
    /// GPU device ID (negative value indicates CPU)
    #[structopt(long = "device", default_value = "-1")]
    device: i32,
}

main!(|args: Args, context: Context| match args.command {
    Command::Train(ref c) => {
        println!("train with a file: {}", c.input);
        let mut dev = select_device(c.device);
        devices::set_default(&mut *dev);
        train(
            &c.input,
            c.valid_file.as_ref(),
            c.embed_file.as_ref(),
            c.n_epochs,
            c.batch_size,
            &context.logger,
        )
    }
    Command::Test(ref c) => {
        println!("test with a file: {}", c.input);
        let mut dev = select_device(0);
        devices::set_default(&mut *dev);
        test(&c.input)
    }
});
