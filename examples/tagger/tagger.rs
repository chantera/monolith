#[macro_use]
extern crate monolith;
#[macro_use]
extern crate primitiv;
#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate slog;

use std::error::Error;
use std::path::{Path, PathBuf};

use monolith::app::prelude::*;
use monolith::io::serialize;
use monolith::lang::prelude::*;
use monolith::preprocessing::Vocab;
use monolith::training;
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
    let save_to = save_to.map(|path| path.as_ref().to_path_buf());

    // load datasets and build the NN model
    let (train_dataset, valid_dataset, mut model) = {
        let mut loader = Loader::new(Preprocessor::new(match embed_file {
            Some(f) => {
                info!(logger, "embed file: {}", f.as_ref().display());
                Vocab::from_cache_or_file(f, "<UNK>")?
            }
            None => {
                info!(logger, "embed file: None");
                Vocab::new()
            }
        }));

        info!(logger, "train file: {}", train_file.as_ref().display());
        let train_dataset = loader.load(train_file)?;
        loader.fix();
        let valid_dataset = match valid_file {
            Some(f) => {
                info!(logger, "valid file: {}", f.as_ref().display());
                Some(loader.load(f)?)
            }
            None => {
                info!(logger, "valid file: None");
                None
            }
        };
        if let Some(ref path) = save_to.as_ref() {
            let path = format!("{}-loader.json", path.to_str().unwrap());
            info!(logger, "saving the loader to {} ...", path);
            serialize::write_to(&loader, path, serialize::Format::Json).unwrap();
        }
        let preprocessor = loader.into_preprocessor();
        let word_vocab = preprocessor.word_vocab();

        let mut builder = TaggerBuilder::new()
            .char(preprocessor.char_vocab().size() as u32, 50)
            .lstm(200)
            .mlp(100)
            .dropout(0.5)
            .out(preprocessor.postag_vocab().size() as u32);
        info!(logger, "builder: {:?}", builder);
        builder = if word_vocab.has_embed() {
            builder.word_embed(word_vocab.embed()?)
        } else {
            builder.word(word_vocab.size() as u32, 100)
        };
        let model = builder.build();
        (train_dataset, valid_dataset, model)
    };

    // configure an optimizer
    let mut optimizer = optimizers::Adam::new(learning_rate, 0.9, 0.999, 1e-8);
    if weight_decay_strength > 0.0 {
        optimizer.set_weight_decay(weight_decay_strength);
    }
    if gradient_clipping_threshold > 0.0 {
        optimizer.set_gradient_clipping(gradient_clipping_threshold);
    }
    optimizer.add_model(&mut model);

    // initialize a model saver
    let saver = save_to.map(|path| {
        let arch_path = format!("{}-tagger.arch.json", path.to_str().unwrap());
        serialize::write_to(&model, arch_path, serialize::Format::Json).unwrap();
        let model_path = format!("{}-tagger", path.to_str().unwrap());
        let mut c = training::callbacks::Saver::new(&model, &model_path);
        c.set_interval(1);
        c.save_from(5);
        c.save_best(valid_dataset.is_some());
        c
    });

    // create trainer with a forward function and register callbacks
    let mut trainer =
        training::Trainer::new(optimizer, |mut batch: Vec<&Sample<_>>, train: bool| {
            sort_batch!(batch);
            take_cols!((words:0, chars:1, sentences:2, postags:3); batch, batch_size);
            transpose!(words, chars, postags);
            let ys = model.forward(&words, &chars, train);
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

pub fn test<P1: AsRef<Path>, P2: AsRef<Path>>(
    test_file: P1,
    model_file: P2,
    logger: &Logger,
) -> Result<(), Box<Error + Send + Sync>> {
    // load datasets and the NN model
    let (test_dataset, mut model, preprocessor) = {
        let (loader_file, arch_file) = {
            let dir = model_file.as_ref().parent().unwrap().to_str().unwrap();
            let s = model_file.as_ref().file_name().unwrap().to_str().unwrap();
            let splits: Vec<&str> = s.split('-').take(2).collect();
            let loader_file = format!("{}/{}-{}-loader.json", dir, splits[0], splits[1]);
            let arch_file = format!("{}/{}-{}-tagger.arch.json", dir, splits[0], splits[1]);
            (loader_file, arch_file)
        };
        info!(logger, "loading the loader from {} ...", loader_file);
        let mut loader =
            serialize::read_from::<_, Loader<Preprocessor>>(loader_file, serialize::Format::Json)?;

        info!(logger, "test file: {}", test_file.as_ref().display());
        loader.fix();
        let test_dataset = loader.load(test_file)?;

        let mut model: Tagger<Tensor> = serialize::read_from(arch_file, serialize::Format::Json)?;
        model.load(model_file, true)?;
        (test_dataset, model, loader.into_preprocessor())
    };
    let postag_vocab = preprocessor.postag_vocab();

    let mut overall_loss = 0.0;
    let mut overall_accuracy = training::Accuracy::new(0, 0);

    for mut batch in test_dataset.batch(32, false) {
        sort_batch!(batch);
        take_cols!((words:0, chars:1, sentences:2, postags:3); batch, 32);
        transpose!(words, chars, postags);
        let ys = model.forward(&words, &chars, false);
        let loss = model.loss(&ys, &postags);
        let accuracy = model.accuracy(&ys, &postags);

        let mut predicted_postags_batch =
            vec![Vec::with_capacity(ys.len()); ys[0].shape().batch() as usize];
        for y_batch in ys {
            for (index, y) in y_batch.argmax(0).into_iter().enumerate() {
                predicted_postags_batch[index].push(y);
            }
        }
        for (sentence, predicted_postags) in sentences.iter().zip(predicted_postags_batch) {
            let token_iter = sentence.as_ref().unwrap().iter();
            for (token, postag) in token_iter.skip(1).zip(predicted_postags) {
                print!("{}/{} ", token.form(), postag_vocab.lookup(postag).unwrap());
            }
            println!("");
        }

        overall_loss += loss.to_float();
        overall_accuracy += accuracy;
    }
    info!(logger, "loss: {}, {}", overall_loss, overall_accuracy);

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
    /// A model file
    #[structopt(name = "MODEL", parse(from_os_str))]
    model: PathBuf,
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
            c.weight_decay_strength,
            c.gradient_clipping_threshold,
            output_path,
            &context.logger,
        )
    }
    Command::Test(ref c) => {
        let mut dev = primitiv_utils::select_device(c.device);
        info!(&context.logger, "execute subcommand: {:?}", c);
        devices::set_default(&mut *dev);
        test(&c.input, &c.model, &context.logger)
    }
});
