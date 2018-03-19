#[macro_use]
extern crate monolith;

#[macro_use]
extern crate clap;

use std::error::Error;
use std::path::Path;
use std::process::exit;
use std::result::Result;

use monolith::preprocessing::Vocab;

use self::dataset::*;
mod dataset;

fn train<P: AsRef<Path>>(
    train_file: P,
    valid_file: Option<P>,
    n_epoch: u32,
    batch_size: usize,
) -> Result<(), Box<Error>> {
    let mut loader = Loader::new(Preprocessor::new(Vocab::new()));
    let train_dataset = loader.load(train_file)?;

    for epoch in 1..n_epoch + 1 {
        for batch in train_dataset.batch(batch_size, true) {
            take_cols!((words:0, chars:1, postags:2); batch, batch_size);
            // let ys = model.forward(words, chars);
            // let loss = model.loss(ys, postags);
            // loss.backward();
        }
    }
    Ok(())
}

fn test<P: AsRef<Path>>(file: P) -> Result<(), Box<Error>> {
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

    let result = match matches.subcommand() {
        ("train", Some(m)) => {
            println!("train with a file: {}", m.value_of("INPUT").unwrap());
            train(m.value_of("INPUT").unwrap(), None, 20, 32)
        }
        ("test", Some(m)) => {
            println!("test with a file: {}", m.value_of("INPUT").unwrap());
            test(m.value_of("INPUT").unwrap())
        }
        _ => unreachable!(),
    };

    exit(match result {
        Ok(_) => 0,
        Err(e) => {
            println!("{}", e);
            1
        }
    })
}
