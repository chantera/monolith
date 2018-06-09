use std::path::PathBuf;

use app::Config;
use logging;

pub trait FromArgs {
    fn from_args() -> Self
    where
        Self: Sized;
}

#[macro_export]
macro_rules! main {
    (|$args:ident: $sopt:ty, $ctx:ident: Context| $body:block; default) => {
        fn main() {
            let $args = <$sopt>::from_args();
            let config = Config::default();
            App::from_config(config)
                .main(move |$ctx: Context| $body)
                .run();
        }
    };
    (|$args:ident: $sopt:ty, $ctx:ident: Context| $body:block; @$field:ident) => {
        fn main() {
            let $args = <$sopt>::from_args();
            App::from_config($args.$field.clone())
                .main(move |$ctx: Context| $body)
                .run();
        }
    };
    (|$args:ident: $sopt:ty, $ctx:ident: Context| $body:block) => {
        main!(|$args: $sopt, $ctx: Context| $body; @common);
    };
    (|$args:ident: $sopt:ty, $ctx:ident: Context| $body:expr) => {
        main!(|$args: $sopt, $ctx: Context| { $body });
    };
}

#[derive(StructOpt, Debug, Clone)]
pub struct CommonArgs {
    /// Activate debug mode
    #[structopt(short = "d", long = "debug")]
    debug: bool,
    /// Log directory
    #[structopt(long = "logdir", parse(from_os_str))]
    logdir: Option<PathBuf>,
    /// Verbose mode (-v, -vv, -vvv, etc.)
    #[structopt(short = "v", long = "verbose", parse(from_occurrences))]
    verbose: u8,
}

impl From<CommonArgs> for Config {
    fn from(args: CommonArgs) -> Config {
        // TODO(chantera) implement
        let mut config = Config::default();
        if let Some(ref logdir) = args.logdir {
            config.logging.logdir = logdir.to_string_lossy().into_owned();
            config.logging.level = logging::Level::Debug;
        }
        config
    }
}
