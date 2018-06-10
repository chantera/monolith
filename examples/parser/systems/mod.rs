use std::error;
use std::fmt;

pub mod chen_manning_14;
pub mod kiperwasser_goldberg_16_transition;

#[derive(Debug, Clone, Copy)]
pub enum System {
    ChenManning14,
    KiperwasserGoldberg16Transition,
}

impl System {
    pub fn from_str(s: &str) -> Result<System, Error> {
        match s {
            "ChenManning14" | "cm14" => Ok(System::ChenManning14),
            "KiperwasserGoldberg16Transition" | "kg16t" => {
                Ok(System::KiperwasserGoldberg16Transition)
            }
            _ => Err(Error::NotFound),
        }
    }

    pub fn default_learning_rate(&self) -> f32 {
        match *self {
            System::ChenManning14 => 0.01,
            System::KiperwasserGoldberg16Transition => 0.001,
        }
    }
}

#[derive(Debug)]
pub enum Error {
    NotFound,
}

impl Error {
    pub fn to_str(&self) -> &str {
        match *self {
            Error::NotFound => "not found",
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "System error: {}", self.to_str())
    }
}

impl error::Error for Error {
    fn description(&self) -> &str {
        self.to_str()
    }
}
