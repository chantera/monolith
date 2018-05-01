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

pub fn train<P1, P2, P3, P4>(
    train_file: P1,
    valid_file: Option<P2>,
    embed_file: Option<P3>,
    n_epochs: u32,
    batch_size: usize,
    learning_rate: f32,
    weight_decay_strength: f32,
    gradient_clipping_threshold: f32,
    save_to: Option<P4>,
    logger: &Logger,
) -> Result<(), Box<Error + Send + Sync>>
where
    P1: AsRef<Path>,
    P2: AsRef<Path>,
    P3: AsRef<Path>,
    P4: AsRef<Path>,
{
    // load datasets and build the NN model
    let (train_dataset, valid_dataset, mut model) = {
        let mut loader = Loader::new(Preprocessor::new(match embed_file {
            Some(f) => {
                info!(logger, "embed file: {}", f.as_ref().to_str().unwrap());
                Vocab::from_cache_or_file(f, "<UNK>")?
            }
            None => {
                info!(logger, "embed file: None");
                Vocab::new()
            }
        }));

        info!(
            logger,
            "train file: {}",
            train_file.as_ref().to_str().unwrap()
        );
        let train_dataset = loader.load(train_file)?;
        loader.fix();
        let valid_dataset = match valid_file {
            Some(f) => {
                info!(logger, "valid file: {}", f.as_ref().to_str().unwrap());
                Some(loader.load(f)?)
            }
            None => {
                info!(logger, "valid file: None");
                None
            }
        };
        let preprocessor = loader.dispose();
        let word_vocab = preprocessor.word_vocab();

        let mut builder = TaggerBuilder::new()
            .char(preprocessor.char_vocab().size(), 50)
            .lstm(200)
            .mlp(100)
            .dropout(0.5)
            .out(preprocessor.pos_vocab().size());
        info!(logger, "builder: {:?}", builder);
        builder = if word_vocab.has_embed() {
            builder.word_embed(word_vocab.embed()?)
        } else {
            builder.word(word_vocab.size(), 100)
        };
        let mut model = builder.build();
        (train_dataset, valid_dataset, model)
    };

    // configure an optimizer
    let mut optimizer = optimizers::Adam::default();
    optimizer.set_learning_rate_scaling(learning_rate);
    if weight_decay_strength > 0.0 {
        optimizer.set_weight_decay(weight_decay_strength);
    }
    if gradient_clipping_threshold > 0.0 {
        optimizer.set_gradient_clipping(gradient_clipping_threshold);
    }
    optimizer.add_model(&mut model);

    // initialize a model saver
    let saver = save_to.map(|dir| {
        let mut c = Saver::new(&model, dir, "tagger");
        c.set_interval(1);
        c.save_from(10);
        c.save_best(true);
        c
    });

    // create trainer with a forward function and register callbacks
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
    trainer.enable_report(logger.new(o!()), 1);
    if let Some(c) = saver {
        trainer.add_callback("saver", c);
    }

    // start training
    trainer.fit(train_dataset, valid_dataset, n_epochs, batch_size);

    Ok(())
}

pub fn test<P: AsRef<Path>>(_file: P) -> Result<(), Box<Error + Send + Sync>> {
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
    /// Gradient clipping threshold
    #[structopt(long = "gclip", default_value = "5.0")]
    gradient_clipping_threshold: f32,
    /// Learning rate for an optimizer
    #[structopt(long = "lrate", default_value = "0.001")]
    learning_rate: f32,
    /// Number of sweeps over the dataset to train
    #[structopt(long = "epoch", default_value = "20")]
    n_epochs: u32,
    /// Directory for saved models
    #[structopt(long = "save", parse(from_os_str))]
    save_to: Option<PathBuf>,
    /// Weight decay strength
    #[structopt(long = "wdecay", default_value = "1e-6")]
    weight_decay_strength: f32,
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
        train(
            &c.input,
            c.valid_file.as_ref(),
            c.embed_file.as_ref(),
            c.n_epochs,
            c.batch_size,
            c.learning_rate,
            c.weight_decay_strength,
            c.gradient_clipping_threshold,
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
