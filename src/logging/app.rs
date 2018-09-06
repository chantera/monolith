use std::env;
use std::fs;
use std::io::Write;
use std::panic;
use std::path::PathBuf;
use std::thread;
use std::time::Duration;

use backtrace::Backtrace;
use chrono::prelude::*;
pub use slog::FilterLevel as Level;
use slog::{Discard, Logger, Record};
use uuid::{Uuid, NAMESPACE_OID as UUID_NAMESPACE_OID};

use super::{create_logger_with_kv_and_time, Config, Error};

#[derive(Debug)]
pub struct AppLogger {
    inner: Logger,
    config: Config,
    accessid: String,
    accesstime: DateTime<Local>,
    filepath: Option<PathBuf>,
}

impl AppLogger {
    pub fn new<C: Into<Config>>(config: C) -> Result<Self, Error> {
        let accesstime = Local::now();
        let accessid =
            Uuid::new_v5(&UUID_NAMESPACE_OID, &accesstime.to_string()).to_string()[..8].to_string();
        let c = config.into();

        let (inner, filepath) = create_logger_with_kv_and_time(
            c.clone(),
            o!("accessid" => accessid.clone()),
            &accesstime,
        )?;
        let mut logger = AppLogger {
            inner: inner,
            config: c,
            accessid: accessid,
            accesstime: accesstime,
            filepath: filepath,
        };
        AppLogger::initialize(&mut logger);
        Ok(logger)
    }

    fn initialize(&mut self) {
        info!(
            self,
            "LOG Start with ACCESSID=[{}] ACCESSTIME=[{}]",
            self.accessid,
            self.accesstime.to_rfc3339(),
        );
    }

    fn finalize(&mut self) {
        let processtime = Local::now()
            .signed_duration_since(self.accesstime)
            .num_milliseconds() as f64 * 1e-3;
        info!(
            self,
            "LOG End with ACCESSID=[{}] ACCESSTIME=[{}] PROCESSTIME=[{}]",
            self.accessid,
            self.accesstime.to_rfc3339(),
            processtime,
        );
        self.inner = Logger::root(Discard, o!());
        if let Some(ref path) = self.filepath {
            thread::sleep(Duration::from_millis(1));
            let result = fs::OpenOptions::new()
                .append(true)
                .open(path)
                .map(|mut file| write!(file, "\n").map(|_| ()).and_then(|()| file.flush()));
            match result {
                Ok(_) => {}
                Err(e) => eprintln!("unable to write a newline: {}", e),
            }
        }
    }

    #[inline]
    pub fn log(&self, record: &Record) {
        self.inner.log(record);
    }

    pub fn get_inner(&self) -> &Logger {
        &self.inner
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn accessid(&self) -> &str {
        &self.accessid
    }

    pub fn accesstime(&self) -> &DateTime<Local> {
        &self.accesstime
    }

    pub fn create(&self) -> Logger {
        self.inner.new(o!())
    }
}

impl Drop for AppLogger {
    fn drop(&mut self) {
        AppLogger::finalize(self);
    }
}

pub fn enable_log_panic(logger: Logger) {
    panic::set_hook(Box::new(move |info| {
        let backtrace = Backtrace::new();
        let current_thread = thread::current();
        let thread_name = current_thread.name().unwrap_or("<unnamed>");
        let message = match info.payload().downcast_ref::<&'static str>() {
            Some(s) => *s,
            None => match info.payload().downcast_ref::<String>() {
                Some(s) => &**s,
                None => "Box<Any>",
            },
        };
        let loc_str = info
            .location()
            .map(|location| {
                format!(
                    ", {}:{}:{}",
                    location.file(),
                    location.line(),
                    location.column(),
                )
            })
            .unwrap_or_else(|| "".to_string());
        error!(
            logger,
            "thread '{}' panicked at '{}'{}", thread_name, message, loc_str
        );
        let enabled_backtrace = match env::var_os("RUST_BACKTRACE") {
            Some(ref val) if val != "0" => true,
            _ => false,
        };
        if enabled_backtrace {
            error!(logger, "{:?}", backtrace);
        } else {
            eprintln!("note: Run with `RUST_BACKTRACE=1` for a backtrace.");
        }
    }));
}
