use std::error::Error;
use std::io as std_io;
use std::fmt;
use std::fs;
use std::path::PathBuf;
use std::process;
use std::thread;
use std::time::Duration;

use chan;
use chan_signal::{self, Signal};
use libc;
use slog::Logger;

use logging::{AppLogger, Config as LogConfig};
use utils;

pub static APP_DIR: &'static str = "~/.monolith";

pub fn app_dir() -> std_io::Result<PathBuf> {
    let path = utils::path::expandtilde(APP_DIR);
    if !path.exists() {
        fs::create_dir(&path)?;
    }
    Ok(path)
}

#[derive(Debug)]
struct AppError {
    code: i32,
    error: Box<Error + Send + Sync>,
}

impl AppError {
    pub fn new<E>(code: i32, error: E) -> AppError
    where
        E: Into<Box<Error + Send + Sync>>,
    {
        AppError {
            code: code,
            error: error.into(),
        }
    }

    pub fn code(&self) -> i32 {
        self.code
    }
}

impl Error for AppError {
    fn description(&self) -> &str {
        self.error.description()
    }

    fn cause(&self) -> Option<&Error> {
        self.error.cause()
    }
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} (code: {})", self.error, self.code)
    }
}

pub struct App {
    main_fn: Box<FnMut(Logger) -> Result<(), Box<Error + Send + Sync>> + Send + 'static>,
    handle_signal: bool,
    exit_on_finish: bool,
    log_config: LogConfig,
}

impl App {
    pub fn new() -> Self {
        App {
            main_fn: Box::new(|logger| {
                info!(logger, "Hello World!");
                Ok(())
            }),
            handle_signal: true,
            exit_on_finish: false,
            log_config: LogConfig::default(),
        }
    }

    pub fn main<F>(mut self, f: F) -> Self
    where
        F: FnMut(Logger) -> Result<(), Box<Error + Send + Sync>> + Send + 'static,
    {
        self.main_fn = Box::new(f);
        self
    }

    pub fn run(self) {
        let (mut main_fn, handle_signal, exit_on_finish, log_config) = (
            self.main_fn,
            self.handle_signal,
            self.exit_on_finish,
            self.log_config,
        );
        let signal = if handle_signal {
            // `notify` must be called before any other threads are spawned in the process.
            Some(chan_signal::notify(&[Signal::INT, Signal::TERM]))
        } else {
            None
        };
        // an async logger spawns threads internally.
        let code = match AppLogger::new(log_config) {
            Ok(logger) => {
                let result = if let Some(signal) = signal {
                    let (sdone, rdone) = chan::sync(0);
                    let child_logger = logger.create();
                    thread::spawn(move || {
                        sdone.send((*main_fn)(child_logger).map_err(|e| AppError::new(1, e)));
                        let _ = sdone;
                    });
                    let mut retval;
                    chan_select! {
                        signal.recv() -> s => {
                            let (code, err) = s.map(|val| {
                                (signal_to_i32(val), format!("receive a signal: {:?}", val))
                            }).unwrap_or_else(|| (1, "failed to receive a signal".to_string()));
                            retval = Err(AppError::new(code, err));
                        },
                        rdone.recv() -> r => {
                            retval = r.unwrap_or_else(|| {
                                Err(AppError::new(1, "failed to receive a result"))
                            });
                        }
                    }
                    retval
                } else {
                    (*main_fn)(logger.create()).map_err(|e| AppError::new(1, e))
                };
                let code = match result {
                    Ok(_) => 0,
                    Err(e) => {
                        error!(logger, "{}", e);
                        128 + e.code()
                    }
                };
                drop(logger);
                thread::sleep(Duration::from_millis(1));
                code
            }
            Err(e) => {
                eprintln!("{}", e);
                1
            }
        };
        if exit_on_finish {
            process::exit(code);
        }
    }
}

impl fmt::Debug for App {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("App")
            .field("handle_signal", &self.handle_signal)
            .field("exit_on_finish", &self.exit_on_finish)
            .field("log_config", &self.log_config)
            .finish()
    }
}

fn signal_to_i32(signal: Signal) -> i32 {
    match signal {
        Signal::HUP => libc::SIGHUP,
        Signal::INT => libc::SIGINT,
        Signal::QUIT => libc::SIGQUIT,
        Signal::ILL => libc::SIGILL,
        Signal::ABRT => libc::SIGABRT,
        Signal::FPE => libc::SIGFPE,
        Signal::KILL => libc::SIGKILL,
        Signal::SEGV => libc::SIGSEGV,
        Signal::PIPE => libc::SIGPIPE,
        Signal::ALRM => libc::SIGALRM,
        Signal::TERM => libc::SIGTERM,
        Signal::USR1 => libc::SIGUSR1,
        Signal::USR2 => libc::SIGUSR2,
        Signal::CHLD => libc::SIGCHLD,
        Signal::CONT => libc::SIGCONT,
        Signal::STOP => libc::SIGSTOP,
        Signal::TSTP => libc::SIGTSTP,
        Signal::TTIN => libc::SIGTTIN,
        Signal::TTOU => libc::SIGTTOU,
        Signal::BUS => libc::SIGBUS,
        Signal::PROF => libc::SIGPROF,
        Signal::SYS => libc::SIGSYS,
        Signal::TRAP => libc::SIGTRAP,
        Signal::URG => libc::SIGURG,
        Signal::VTALRM => libc::SIGVTALRM,
        Signal::XCPU => libc::SIGXCPU,
        Signal::XFSZ => libc::SIGXFSZ,
        Signal::IO => libc::SIGIO,
        Signal::WINCH => libc::SIGWINCH,
        Signal::__NonExhaustiveMatch => unreachable!(),
    }
}
