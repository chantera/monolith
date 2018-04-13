use std::error;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf, MAIN_SEPARATOR};
use std::io as std_io;

use chrono::prelude::*;
pub use slog::FilterLevel as Level;
use slog::{Discard, Drain, Duplicate, Fuse, LevelFilter, Level as LogLevel, Logger, OwnedKV,
           SendSyncRefUnwindSafeKV};
use slog_async::Async;
use slog_term::{CompactFormat, Decorator, FullFormat, PlainDecorator, TermDecorator};

use utils;

#[derive(Debug)]
pub enum Stream {
    StdOut,
    StdErr,
    File(File),
    Null,
}

impl Stream {
    pub fn is_null(&self) -> bool {
        match *self {
            Stream::Null => true,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Format {
    Full,
    Compact,
}

#[derive(Debug)]
pub struct LoggerBuilder {
    stream: Stream,
    level: Level,
    format: Format,
}

impl LoggerBuilder {
    pub fn new(stream: Stream) -> Self {
        LoggerBuilder {
            stream: stream,
            level: Level::Debug,
            format: Format::Full,
        }
    }

    pub fn level(mut self, l: Level) -> Self {
        self.level = l;
        self
    }

    pub fn format(mut self, f: Format) -> Self {
        self.format = f;
        self
    }

    pub fn build<T>(self, values: OwnedKV<T>) -> Logger
    where
        T: SendSyncRefUnwindSafeKV + 'static,
    {
        match self.build_drain() {
            Some(drain) => Logger::root(drain.fuse(), values),
            None => Logger::root(Discard, values),
        }
    }

    fn build_drain(&self) -> Option<LevelFilter<Fuse<Async>>> {
        match self.level {
            Level::Off => {
                return None;
            }
            _ => {}
        }
        match self.stream {
            Stream::StdOut => {
                let drain = self.build_drain_from_decorator(TermDecorator::new().stdout().build());
                Some(drain)
            }
            Stream::StdErr => {
                let drain = self.build_drain_from_decorator(TermDecorator::new().stderr().build());
                Some(drain)
            }
            Stream::File(ref f) => {
                let drain =
                    self.build_drain_from_decorator(PlainDecorator::new(f.try_clone().unwrap()));
                Some(drain)
            }
            Stream::Null => None,
        }
    }

    fn build_drain_from_decorator<D: Decorator + Send + 'static>(
        &self,
        decorator: D,
    ) -> LevelFilter<Fuse<Async>> {
        let drain = match self.format {
            Format::Compact => {
                let drain = CompactFormat::new(decorator).use_local_timestamp().build();
                Async::new(drain.fuse()).build()
            }
            Format::Full => {
                let drain = FullFormat::new(decorator).use_local_timestamp().build();
                Async::new(drain.fuse()).build()
            }
        };
        let drain = LevelFilter::new(
            drain.fuse(),
            LogLevel::from_usize(self.level.as_usize()).unwrap(),
        );
        drain
    }

    pub fn build_with<T>(self, other: LoggerBuilder, values: OwnedKV<T>) -> Logger
    where
        T: SendSyncRefUnwindSafeKV + 'static,
    {
        match other.build_drain() {
            Some(d2) => {
                match self.build_drain() {
                    Some(d1) => Logger::root(Duplicate::new(d1, d2).fuse(), values),
                    None => Logger::root(d2.fuse(), values),
                }
            }
            None => {
                match self.build_drain() {
                    Some(d1) => Logger::root(d1.fuse(), values),
                    None => Logger::root(Discard, values),
                }
            }
        }
    }
}

#[derive(Debug)]
pub enum Error {
    InvalidOption,
    Other(std_io::Error),
}

impl error::Error for Error {
    fn description(&self) -> &str {
        match *self {
            Error::InvalidOption => "invalid option",
            Error::Other(ref e) => e.description(),
        }
    }

    fn cause(&self) -> Option<&error::Error> {
        match *self {
            Error::InvalidOption => None,
            Error::Other(ref e) => e.cause(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::InvalidOption => "invalid option".fmt(f),
            Error::Other(ref err) => err.fmt(f),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub level: Level,
    pub verbosity: Level,
    pub logdir: String,
    pub mkdir: bool,
    pub filename: String,
    pub filemode: char,
    pub fileprefix: Option<String>,
    pub filesuffix: Option<String>,
    pub format: Format,
    pub use_stderr: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            level: Level::Debug,
            verbosity: Level::Trace,
            logdir: "./".to_string(),
            mkdir: false,
            filename: "%Y%m%d.log".to_string(),
            filemode: 'a',
            fileprefix: None,
            filesuffix: None,
            format: Format::Full,
            use_stderr: false,
        }
    }
}

pub fn create_logger<C: Into<Config>>(config: C) -> Result<Logger, Error> {
    create_logger_with_kv_and_time(config, o!(), Local::now())
}

pub fn create_logger_with_kv<C: Into<Config>, T>(
    config: C,
    values: OwnedKV<T>,
) -> Result<Logger, Error>
where
    T: SendSyncRefUnwindSafeKV + 'static,
{
    create_logger_with_kv_and_time(config, values, Local::now())
}

pub fn create_logger_with_kv_and_time<C: Into<Config>, T, Tz: TimeZone>(
    config: C,
    values: OwnedKV<T>,
    datetime: DateTime<Tz>,
) -> Result<Logger, Error>
where
    T: SendSyncRefUnwindSafeKV + 'static,
    Tz::Offset: fmt::Display,
{
    let c = config.into();
    let fstream = match c.level {
        Level::Off => Stream::Null,
        _ => {
            let mut options = OpenOptions::new();
            options.create(true).write(true);
            let mut enable_numbering = false;
            match c.filemode {
                'w' => {
                    options.truncate(true);
                }
                'a' => {
                    options.append(true);
                }
                'n' => {
                    options.truncate(true);
                    enable_numbering = true;
                }
                _ => {
                    return Err(Error::InvalidOption);
                }
            }
            let path = resolve_filepath(
                &c.logdir,
                &c.filename,
                c.fileprefix.as_ref().map(|s| s.as_str()),
                c.filesuffix.as_ref().map(|s| s.as_str()),
                datetime,
                c.mkdir,
                enable_numbering,
            ).map_err(|e| Error::Other(e))?;
            let file = options.open(path).map_err(|e| Error::Other(e))?;
            Stream::File(file)
        }
    };

    let vstream = if c.use_stderr {
        Stream::StdErr
    } else {
        Stream::StdOut
    };
    let logger = LoggerBuilder::new(vstream)
        .level(c.verbosity)
        .format(c.format)
        .build_with(
            LoggerBuilder::new(fstream).level(c.level).format(c.format),
            values,
        );
    Ok(logger)
}

fn resolve_filepath<P1: AsRef<Path>, P2: AsRef<Path>, Tz: TimeZone>(
    dir: P1,
    filename: P2,
    prefix: Option<&str>,
    suffix: Option<&str>,
    time: DateTime<Tz>,
    mkdir: bool,
    numbering: bool,
) -> Result<PathBuf, std_io::Error>
where
    Tz::Offset: fmt::Display,
{
    let dir = utils::path::expandtilde(dir);
    if dir.is_dir() {
        // pass
    } else if mkdir {
        fs::create_dir(&dir)?;
    } else {
        return Err(std_io::Error::new(
            std_io::ErrorKind::NotFound,
            format!(
                "file `{}` is not a directory",
                dir.to_str().unwrap()
            ),
        ));
    }

    let filename = filename.as_ref();
    if filename.to_str().unwrap().contains(MAIN_SEPARATOR) {
        return Err(std_io::Error::new(
            std_io::ErrorKind::InvalidInput,
            "filename must not contain the separator",
        ));
    }

    let stem = filename.file_stem().and_then(|s| s.to_str()).ok_or_else(
        || {
            std_io::Error::new(std_io::ErrorKind::InvalidInput, "invalid filename")
        },
    )?;
    let stem = format!(
        "{}{}{}",
        prefix.unwrap_or(""),
        time.format(stem),
        suffix.unwrap_or("")
    );
    let ext = filename
        .extension()
        .map(|s| format!(".{}", s.to_str().unwrap()))
        .unwrap_or("".to_string());

    if numbering {
        let mut number = 0;
        loop {
            let path = dir.join(format!("{}-{}{}", stem, number, ext));
            if !path.exists() {
                return Ok(path);
            }
            number += 1;
        }

    } else {
        Ok(dir.join(format!("{}{}", stem, ext)))
    }
}
