use std::fs;
use std::io as std_io;
use std::path::PathBuf;

use utils;

pub use self::app::*;
pub use self::main::*;

mod app;
mod main;
pub mod prelude;

pub static APP_DIR: &'static str = "~/.monolith";

pub fn app_dir() -> std_io::Result<PathBuf> {
    let path = utils::path::expandtilde(APP_DIR);
    if !path.exists() {
        fs::create_dir(&path)?;
    }
    Ok(path)
}
