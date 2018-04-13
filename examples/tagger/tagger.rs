#[macro_use]
extern crate clap;
#[macro_use]
extern crate monolith;
#[macro_use]
extern crate primitiv;
#[macro_use]
extern crate slog;

use std::error::Error;
use std::path::Path;
use std::result::Result;

use monolith::logging;
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
) -> Result<(), Box<Error>> {
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

fn test<P: AsRef<Path>>(_file: P) -> Result<(), Box<Error>> {
    Ok(())
}

fn main() {
    let matches = clap_app!(
        @app (app_from_crate!())
        (@setting ArgRequiredElseHelp)
        (@subcommand train =>
            (about: "Trains model")
            (@arg INPUT: +required "A training data file")
            (@arg batch_size: -b --batchsize default_value("32") "Number of examples in each mini-batch")
            (@arg device: -d --device default_value("-1") "GPU device ID (negative value indicates CPU)")
            (@arg embed_file: --embed +takes_value "A file of pretrained word embeddings")
            (@arg n_epochs: -e --epoch default_value("20") "Number of sweeps over the dataset to train")
            (@arg valid_file: --vfile +takes_value "A validation data file")
        )
        (@subcommand test =>
            (about: "Tests model")
            (@arg INPUT: +required "A testing data file")
        )
    ).get_matches();

    let mut log_config = logging::Config::default();
    log_config.logdir = "target".to_string();
    log_config.mkdir = false;
    log_config.filename = "your_log_file_path.log".to_string();
    let logger = logging::AppLogger::new(log_config).unwrap();
    info!(logger, "info");
    warn!(logger, "hello world");

    let result = match matches.subcommand() {
        ("train", Some(m)) => {
            println!("train with a file: {}", m.value_of("INPUT").unwrap());
            let batch_size = m.value_of("batch_size").unwrap().parse::<usize>().unwrap();
            let device_id = m.value_of("device").unwrap().parse::<i32>().unwrap();
            let n_epochs = m.value_of("n_epochs").unwrap().parse::<u32>().unwrap();
            let mut dev = select_device(device_id);
            devices::set_default(&mut *dev);
            train(
                m.value_of("INPUT").unwrap(),
                m.value_of("valid_file"),
                m.value_of("embed_file"),
                n_epochs,
                batch_size,
                logger.get_inner(),
            )
        }
        ("test", Some(m)) => {
            println!("test with a file: {}", m.value_of("INPUT").unwrap());
            let mut dev = select_device(0);
            devices::set_default(&mut *dev);
            test(m.value_of("INPUT").unwrap())
        }
        _ => unreachable!(),
    };

    let _exit_code = match result {
        Ok(_) => 0,
        Err(e) => {
            println!("{}", e);
            1
        }
    };
    // std::process::exit(exit_code);
}
