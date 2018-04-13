extern crate libc;
extern crate rand;

#[cfg(feature = "app")]
#[macro_use]
extern crate chan;
#[cfg(feature = "app")]
extern crate chan_signal;
#[cfg(feature = "app")]
extern crate uuid;
#[cfg(feature = "logging")]
extern crate chrono;
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
#[cfg(feature = "serialize")]
extern crate byteorder;
#[cfg(feature = "serialize")]
extern crate csv;
#[cfg(feature = "serialize")]
extern crate serde;
#[cfg(feature = "serialize")]
#[macro_use]
extern crate serde_derive;
#[cfg(feature = "serialize")]
extern crate serde_json;
#[cfg(feature = "serialize")]
extern crate rmp_serde;
#[cfg(feature = "training")]
extern crate pbr;

#[cfg(feature = "app")]
pub mod app;
#[macro_use]
pub mod dataset;
#[macro_use]
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
pub mod utils;
