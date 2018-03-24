use std::fs::File;

pub use slog::Level;
use slog::{Discard, Drain, Duplicate, Fuse, LevelFilter, Logger, OwnedKV, SendSyncRefUnwindSafeKV};
use slog_async::Async;
use slog_term::{CompactFormat, Decorator, FullFormat, PlainDecorator, TermDecorator};

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

#[derive(Debug)]
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
        let drain = LevelFilter::new(drain.fuse(), Level::Info);
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
