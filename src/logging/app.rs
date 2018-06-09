use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::thread;
use std::time::Duration;

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
