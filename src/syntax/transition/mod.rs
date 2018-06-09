use std::error;
use std::fmt;

pub use self::arc_standard::*;
pub use self::state::*;

mod arc_standard;
pub mod prelude;
mod state;

pub type Index = u32;
pub type Action = u32;

pub trait TransitionState {
    fn step(&self) -> usize {
        self.actions().len()
    }

    fn num_tokens(&self) -> usize;

    fn stack_top(&self) -> Option<Index>;

    fn stack(&self, position: Index) -> Option<Index>;

    fn stack_size(&self) -> usize;

    fn is_stack_empty(&self) -> bool {
        self.stack_size() == 0
    }

    fn buffer_head(&self) -> Option<Index>;

    fn buffer(&self, position: Index) -> Option<Index>;

    fn buffer_size(&self) -> usize;

    fn is_buffer_empty(&self) -> bool {
        self.buffer_size() == 0
    }

    fn head(&self, index: Index) -> Option<Index>;

    fn heads(&self) -> &[Option<Index>];

    fn label(&self, index: Index) -> Option<Index>;

    fn labels(&self) -> &[Option<Index>];

    fn leftmost(&self, index: Index, check_from: Option<Index>) -> Option<Index>;

    fn rightmost(&self, index: Index, check_from: Option<Index>) -> Option<Index>;

    fn actions(&self) -> &[Action];
}

pub trait TransitionMutableState: TransitionState {
    fn advance(&mut self) -> Result<(), Error>;

    fn push(&mut self, index: Index) -> Result<(), Error>;

    fn pop(&mut self) -> Result<Index, Error>;

    fn add_arc(&mut self, index: Index, head: Index, label: Index) -> Result<(), Error>;

    fn record(&mut self, action: Action) -> Result<(), Error>;
}

pub trait TransitionSystem {
    fn num_action_types() -> usize;

    fn num_defined_actions(num_labels: usize) -> usize;

    fn estimate_num_actions(num_tokens: usize) -> usize;

    fn apply<S: TransitionMutableState>(action: Action, state: &mut S) -> Result<(), Error>;

    fn is_allowed<S: TransitionState>(action: Action, state: &S) -> bool;

    fn is_terminal<S: TransitionState>(state: &S) -> bool;

    fn get_oracle<S: TransitionState>(
        state: &S,
        gold_heads: &[Index],
        gold_labels: &[Index],
    ) -> Option<Action>;
}

#[derive(Debug)]
pub enum Error {
    InvalidOperation,
    InvalidArgument,
}

impl Error {
    pub fn as_str(&self) -> &'static str {
        match *self {
            Error::InvalidOperation => "invalid operation",
            Error::InvalidArgument => "invalid argument",
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl error::Error for Error {
    fn description(&self) -> &str {
        self.as_str()
    }

    fn cause(&self) -> Option<&error::Error> {
        None
    }
}

pub fn projectivize(heads: &[u32]) -> Vec<u32> {
    // https://github.com/tensorflow/models/blob/7d30a017fe50b648be6dee544f8059bde52db562/syntaxnet/syntaxnet/document_filters.cc#L296
    let mut heads: Vec<i32> = heads.iter().map(|head| *head as i32).collect();
    let num_tokens = heads.len();
    let mut left: Vec<i32> = vec![-1; num_tokens];
    let mut right: Vec<i32> = vec![-1; num_tokens];
    loop {
        for i in 0..num_tokens {
            left[i] = -1;
            right[i] = num_tokens as i32;
        }

        for (i, head) in heads.iter().enumerate() {
            let l = (i as i32).min(*head);
            let r = (i as i32).max(*head);
            for j in (l + 1)..r {
                let j = j as usize;
                if left[j] < l {
                    left[j] = l;
                }
                if right[j] > r {
                    right[j] = r;
                }
            }
        }

        let mut deepest_arc = -1;
        let mut max_depth = 0;
        for (i, head) in heads.iter().enumerate() {
            if *head == 0 {
                continue;
            }
            let l = (i as i32).min(*head);
            let r = (i as i32).max(*head);
            let left_bound = left[l as usize].max(left[r as usize]);
            let right_bound = right[l as usize].min(right[r as usize]);

            if l < left_bound || r > right_bound {
                let mut depth = 0;
                let mut j = i;
                while j != 0 {
                    depth += 1;
                    j = heads[j] as usize;
                }
                if depth > max_depth {
                    deepest_arc = i as i32;
                    max_depth = depth;
                }
            }
        }

        if deepest_arc == -1 {
            return heads.iter().map(|head| *head as u32).collect();
        }

        let lifted_head = heads[heads[deepest_arc as usize] as usize];
        heads[deepest_arc as usize] = lifted_head;
    }
}
