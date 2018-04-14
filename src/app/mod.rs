use std::error::Error;
use std::io as std_io;
use std::fmt;
use std::fs;
use std::mem;
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

#[derive(Debug)]
pub struct Context {
    pub logger: Logger,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub handle_signal: bool,
    pub exit_on_finish: bool,
    pub logging: LogConfig,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            handle_signal: true,
            exit_on_finish: false,
            logging: LogConfig::default(),
        }
    }
}

pub struct App {
    config: Config,
    main_fn: Option<Box<FnMut(Context) -> Result<(), Box<Error + Send + Sync>> + Send + 'static>>,
    receiver: Option<chan::Receiver<Signal>>,
    logger: Option<AppLogger>,
    context: Option<Context>,
}

impl App {
    pub fn new() -> Self {
        App::with_config(Config::default())
    }

    pub fn with_config<C: Into<Config>>(config: C) -> Self {
        App {
            config: config.into(),
            main_fn: None,
            receiver: None,
            logger: None,
            context: None,
        }
    }

    pub fn main<F>(mut self, f: F) -> Self
    where
        F: FnMut(Context) -> Result<(), Box<Error + Send + Sync>> + Send + 'static,
    {
        self.main_fn = Some(Box::new(f));
        self
    }

    pub fn run(mut self) {
        let mut code = App::initialize(&mut self);
        if code.is_ok() {
            code = App::exec(&mut self);
        };
        App::finalize(&mut self); // `finalize` must not fail.
        if self.config.exit_on_finish {
            let retcode = code.unwrap_or_else(|c| c);
            process::exit(retcode);
        }
    }

    #[inline]
    fn exec(&mut self) -> Result<i32, i32> {
        let mut main_fn = mem::replace(&mut self.main_fn, None).unwrap();
        let receiver = mem::replace(&mut self.receiver, None);
        let context = mem::replace(&mut self.context, None).unwrap();
        let logger = self.logger.as_ref().unwrap();

        info!(logger, "*** [START] ***");
        let result = if let Some(ref signal) = receiver {
            let (sdone, rdone) = chan::sync(0);
            thread::spawn(move || {
                sdone.send((*main_fn)(context).map_err(|e| AppError::new(1, e)));
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
            (*main_fn)(context).map_err(|e| AppError::new(1, e))
        };
        let (result, code) = match result {
            Ok(_) => {
                let c = 0;
                (Ok(c), c)
            }
            Err(e) => {
                error!(logger, "{}", e);
                let c = 128 + e.code();
                (Err(c), c)
            }
        };
        info!(logger, "application finished. (code: {})", code);
        info!(logger, "*** [DONE] ***");
        result
    }

    #[inline]
    fn initialize(&mut self) -> Result<i32, i32> {
        if self.main_fn.is_none() {
            eprintln!("`main` must be called before running");
            return Err(1);
        }
        if self.config.handle_signal {
            // `notify` must be called before any other threads are spawned in the process.
            self.receiver = Some(chan_signal::notify(&[Signal::INT, Signal::TERM]));
        }
        match AppLogger::new(self.config.logging.clone()) {
            // an async logger spawns threads internally.
            Ok(logger) => {
                let context = Context { logger: logger.create() };
                self.logger = Some(logger);
                self.context = Some(context);
                Ok(0)
            }
            Err(e) => {
                eprintln!("{}", e);
                Err(1)
            }
        }
    }

    #[inline]
    fn finalize(&mut self) {
        self.main_fn = None;
        self.logger = None;
        self.receiver = None;
        self.context = None;
        thread::sleep(Duration::from_millis(1));
    }
}

impl fmt::Debug for App {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("App").field("config", &self.config).finish()
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
