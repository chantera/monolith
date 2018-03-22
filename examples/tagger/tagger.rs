#[macro_use]
extern crate clap;
#[macro_use]
extern crate monolith;
#[macro_use]
extern crate primitiv;

use std::error::Error;
use std::path::Path;
use std::process::exit;
use std::result::Result;

use monolith::dataset::transpose_sequence;
use monolith::preprocessing::Vocab;
use primitiv::*;

use self::dataset::*;
use self::models::*;
mod dataset;
mod models;

fn train<P: AsRef<Path>>(
    train_file: P,
    valid_file: Option<P>,
    n_epoch: u32,
    batch_size: usize,
) -> Result<(), Box<Error>> {
    let mut dev = devices::Naive::new(); // let mut dev = D::CUDA::new(0);
    devices::set_default(&mut dev);

    let mut loader = Loader::new(Preprocessor::new(Vocab::new()));
    let train_dataset = loader.load(train_file)?;

    let mut model = Tagger::new(0.5);
    model.init(10000, 100, 200, 32, 400);

    let mut optimizer = optimizers::Adam::default();
    optimizer.set_weight_decay(1e-6);
    optimizer.set_gradient_clipping(5.0);
    optimizer.add_model(&mut model);

    let mut g = Graph::new();
    Graph::set_default(&mut g);

    for epoch in 1..n_epoch + 1 {
        for mut batch in train_dataset.batch(batch_size, true) {
            sort_batch!(batch);
            take_cols!((words:0, chars:1, postags:2); batch, batch_size);
            // transpose!(words, chars, postags);
            let words = transpose_sequence(words, Some(0));
            transpose!(chars, postags);
            // println!("graph: {}", g.dump("dot"));
            g.clear();
            let ys = model.forward(words, chars, true);

            optimizer.reset_gradients();
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
