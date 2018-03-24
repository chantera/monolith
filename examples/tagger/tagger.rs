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

use monolith::dataset::transpose_sequence;
use monolith::logging::{LoggerBuilder, Stream};
use monolith::preprocessing::Vocab;
use primitiv::*;
use slog::Logger;

use self::dataset::*;
use self::models::*;
mod dataset;
mod models;

fn train<P: AsRef<Path>>(
    train_file: P,
    _valid_file: Option<P>,
    n_epoch: u32,
    batch_size: usize,
    logger: &Logger,
) -> Result<(), Box<Error>> {
    let mut dev = devices::Naive::new(); // let mut dev = D::CUDA::new(0);
    devices::set_default(&mut dev);

    let mut loader = Loader::new(Preprocessor::new(Vocab::new()));
    let train_dataset = loader.load(train_file)?;

    let mut model = TaggerBuilder::new()
        .word(10000, 100)
        .char(200, 32)
        .lstm(400)
        .mlp(100)
        .dropout(0.5)
        .out(64)
        .build();

    let mut optimizer = optimizers::Adam::default();
    optimizer.set_weight_decay(1e-6);
    optimizer.set_gradient_clipping(5.0);
    optimizer.add_model(&mut model);

    let mut g = Graph::new();
    Graph::set_default(&mut g);

    for epoch in 1..n_epoch + 1 {
        info!(logger, "epoch: {}", epoch);
        let mut train_loss = 0.0;
        for mut batch in train_dataset.batch(batch_size, true) {
            sort_batch!(batch);
            take_cols!((words:0, chars:1, postags:2); batch, batch_size);
            // transpose!(words, chars, postags);
            let words = transpose_sequence(words, Some(0));
            let postags = transpose_sequence(postags, Some(0));
            g.clear();
            let ys = model.forward(words, chars, true);

            optimizer.reset_gradients();
            let loss = model.loss(&ys, postags);
            train_loss += loss.to_float();
            loss.backward();
        }
        info!(logger, "batch loss: {}", train_loss);
    }
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
        )
        (@subcommand test =>
            (about: "Tests model")
            (@arg INPUT: +required "A testing data file")
        )
    ).get_matches();

    let file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("target/your_log_file_path.log")
        .unwrap();
    let logger =
        LoggerBuilder::new(Stream::StdOut).build_with(LoggerBuilder::new(Stream::File(file)), o!());
    info!(logger, "info");
    warn!(logger, "hello world");

    let result = match matches.subcommand() {
        ("train", Some(m)) => {
            println!("train with a file: {}", m.value_of("INPUT").unwrap());
            train(m.value_of("INPUT").unwrap(), None, 20, 32, &logger)
        }
        ("test", Some(m)) => {
            println!("test with a file: {}", m.value_of("INPUT").unwrap());
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
