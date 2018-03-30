extern crate csv;
extern crate rand;
#[macro_use]
extern crate serde_derive;

#[cfg(feature = "training")]
extern crate pbr;
#[cfg(feature = "logging")]
#[macro_use]
extern crate slog;
#[cfg(feature = "logging")]
extern crate slog_async;
#[cfg(feature = "logging")]
extern crate slog_term;
#[cfg(feature = "models")]
#[macro_use]
extern crate primitiv;

#[macro_use]
pub mod dataset;
pub mod io;
pub mod lang;
#[cfg(feature = "logging")]
pub mod logging;
#[cfg(feature = "models")]
pub mod models;
pub mod preprocessing;
// #[cfg(feature = "syntax")]
// pub mod syntax;
#[cfg(feature = "training")]
pub mod training;
