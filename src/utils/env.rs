use std::env;
use std::error;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::str::FromStr;

#[derive(Debug)]
pub enum VarError {
    NotPresent,
    NotUnicode(OsString),
    Parse(Box<error::Error + Send + Sync>),
}

impl fmt::Display for VarError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            VarError::NotPresent => write!(f, "environment variable not found"),
            VarError::NotUnicode(ref s) => {
                write!(f, "environment variable was not valid unicode: {:?}", s)
            }
            VarError::Parse(ref e) => e.fmt(f),
        }
    }
}

impl error::Error for VarError {
    fn description(&self) -> &str {
        match *self {
            VarError::NotPresent => "environment variable not found",
            VarError::NotUnicode(..) => "environment variable was not valid unicode",
            VarError::Parse(ref e) => e.description(),
        }
    }

    fn cause(&self) -> Option<&error::Error> {
        match *self {
            VarError::NotPresent => None,
            VarError::NotUnicode(..) => None,
            VarError::Parse(ref e) => e.cause(),
        }
    }
}

pub fn var<K: AsRef<OsStr>, T: FromStr>(key: K) -> Result<T, VarError>
where
    <T as FromStr>::Err: Into<Box<error::Error + Send + Sync>>,
{
    match env::var(key) {
        Ok(s) => s.parse::<T>().map_err(|e| VarError::Parse(e.into())),
        Err(e) => {
            match e {
                env::VarError::NotPresent => Err(VarError::NotPresent),
                env::VarError::NotUnicode(s) => Err(VarError::NotUnicode(s)),
            }
        }
    }
}
