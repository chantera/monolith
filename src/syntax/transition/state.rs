use std::u32::MAX as U32_MAX;

use super::{Index, Action, TransitionState, TransitionMutableState, TransitionSystem, Error};

#[inline]
fn default_capacity(num_tokens: usize) -> usize {
    2 * (num_tokens - 1)
}

#[derive(Debug)]
pub struct State {
    num_tokens: Index,
    stack: Vec<Index>,
    buffer: Option<Index>,
    heads: Vec<Option<Index>>,
    labels: Vec<Option<Index>>,
    actions: Vec<Action>,
}

impl State {
    pub fn new(num_tokens: u32) -> Self {
        State::with_capacity(num_tokens, default_capacity(num_tokens as usize))
    }

    pub fn with_capacity(num_tokens: u32, capacity: usize) -> Self {
        let n = num_tokens as usize;
        State {
            num_tokens: num_tokens,
            stack: vec![0],
            buffer: Some(1),
            heads: vec![None; n],
            labels: vec![None; n],
            actions: Vec::with_capacity(capacity),
        }
    }
}

impl TransitionState for State {
    fn step(&self) -> usize {
        self.actions.len()
    }

    fn num_tokens(&self) -> usize {
        self.num_tokens as usize
    }

    fn stack_top(&self) -> Option<Index> {
        self.stack.last().map(|&i| i)
    }

    fn stack(&self, position: Index) -> Option<Index> {
        let position = position as usize;
        let stack_size = self.stack.len();
        if position < stack_size {
            self.stack.get(stack_size - 1 - position).map(|&i| i)
        } else {
            None
        }
    }

    fn stack_size(&self) -> usize {
        self.stack.len()
    }

    fn is_stack_empty(&self) -> bool {
        self.stack.is_empty()
    }

    fn buffer_head(&self) -> Option<Index> {
        self.buffer
    }

    fn buffer(&self, position: Index) -> Option<Index> {
        if let Some(buffer) = self.buffer {
            let index = buffer + position;
            if index < self.num_tokens {
                return Some(index);
            }
        }
        None
    }

    fn buffer_size(&self) -> usize {
        match self.buffer {
            Some(buffer) => (self.num_tokens - buffer) as usize,
            None => 0,
        }
    }

    fn is_buffer_empty(&self) -> bool {
        self.buffer.is_none()
    }

    fn head(&self, index: Index) -> Option<Index> {
        if index < self.num_tokens {
            return self.heads[index as usize];
        }
        None
    }

    fn heads(&self) -> &[Option<Index>] {
        &self.heads
    }

    fn label(&self, index: Index) -> Option<Index> {
        if index < self.num_tokens {
            return self.labels[index as usize];
        }
        None
    }

    fn labels(&self) -> &[Option<Index>] {
        &self.labels
    }

    fn leftmost(&self, index: Index, check_from: Option<Index>) -> Option<Index> {
        if index < self.num_tokens {
            let check_from = match check_from {
                Some(val) => val,
                None => 0,
            };
            if check_from < index {
                let expected = Some(index);
                for i in check_from..index {
                    if self.heads[i as usize] == expected {
                        return Some(i);
                    }
                }
            }
        }
        None
    }

    fn rightmost(&self, index: Index, check_from: Option<Index>) -> Option<Index> {
        if index < self.num_tokens {
            let check_from = match check_from {
                Some(val) => (val + 1).min(self.num_tokens),
                None => 0,
            };
            if check_from > index {
                let expected = Some(index);
                for i in (index..check_from).rev() {
                    if self.heads[i as usize] == expected {
                        return Some(i);
                    }
                }
            }
        }
        None
    }

    fn actions(&self) -> &[Action] {
        &self.actions
    }
}

impl TransitionMutableState for State {
    fn advance(&mut self) -> Result<(), Error> {
        match self.buffer {
            Some(buffer) => {
                if buffer == self.num_tokens - 1 {
                    self.buffer = None;
                } else {
                    self.buffer = Some(buffer + 1);
                }
                Ok(())
            }
            None => Err(Error::InvalidOperation),
        }
    }

    fn push(&mut self, index: Index) -> Result<(), Error> {
        self.stack.push(index);
        Ok(())
    }

    fn pop(&mut self) -> Result<Index, Error> {
        self.stack.pop().ok_or(Error::InvalidOperation)
    }

    fn add_arc(&mut self, index: Index, head: Index, label: Index) -> Result<(), Error> {
        if index >= self.num_tokens {
            Err(Error::InvalidOperation)
        } else if head >= self.num_tokens {
            Err(Error::InvalidOperation)
        } else if index == head {
            Err(Error::InvalidOperation)
        } else {
            let val = &mut self.heads[index as usize];
            match *val {
                Some(_) => Err(Error::InvalidOperation),
                None => {
                    *val = Some(head);
                    self.labels[index as usize] = Some(label);
                    Ok(())
                }
            }
        }
    }

    fn record(&mut self, action: Action) -> Result<(), Error> {
        self.actions.push(action);
        Ok(())
    }
}

#[derive(Debug)]
pub struct GoldState {
    internal: State,
}

impl GoldState {
    pub fn new<T: TransitionSystem>(heads: &[Index], labels: &[Index]) -> Result<Self, Error> {
        let n = heads.len();
        if n > (U32_MAX as usize) {
            Err(Error::InvalidArgument)
        } else if n != labels.len() {
            Err(Error::InvalidArgument)
        } else {
            let capacity = T::estimate_num_actions(n);
            let mut internal = State::with_capacity(n as u32, capacity);
            while !T::is_terminal(&internal) {
                let action = T::get_oracle(&internal, heads, labels).unwrap();
                T::apply(action, &mut internal)?;
            }
            let state = GoldState { internal: internal };
            Ok(state)
        }
    }

    pub fn with_feature_extract<T: TransitionSystem, FO, F: FnMut(&State) -> FO>(
        heads: &[Index],
        labels: &[Index],
        mut extract: F,
    ) -> Result<(Self, Vec<FO>), Error> {
        let n = heads.len();
        if n > (U32_MAX as usize) {
            Err(Error::InvalidArgument)
        } else if n != labels.len() {
            Err(Error::InvalidArgument)
        } else {
            let capacity = T::estimate_num_actions(n);
            let mut internal = State::with_capacity(n as u32, capacity);
            let mut features = Vec::with_capacity(capacity);
            while T::is_terminal(&internal) {
                features.push(extract(&internal));
                let action = T::get_oracle(&internal, heads, labels).unwrap();
                T::apply(action, &mut internal)?;
            }
            let state = GoldState { internal: internal };
            Ok((state, features))
        }
    }
}

impl TransitionState for GoldState {
    fn step(&self) -> usize {
        self.internal.step()
    }

    fn num_tokens(&self) -> usize {
        self.internal.num_tokens()
    }

    fn stack_top(&self) -> Option<Index> {
        self.internal.stack_top()
    }

    fn stack(&self, position: Index) -> Option<Index> {
        self.internal.stack(position)
    }

    fn stack_size(&self) -> usize {
        self.internal.stack_size()
    }

    fn is_stack_empty(&self) -> bool {
        self.internal.is_stack_empty()
    }

    fn buffer_head(&self) -> Option<Index> {
        self.internal.buffer_head()
    }

    fn buffer(&self, position: Index) -> Option<Index> {
        self.internal.buffer(position)
    }

    fn buffer_size(&self) -> usize {
        self.internal.buffer_size()
    }

    fn is_buffer_empty(&self) -> bool {
        self.internal.is_buffer_empty()
    }

    fn head(&self, index: Index) -> Option<Index> {
        self.internal.head(index)
    }

    fn heads(&self) -> &[Option<Index>] {
        self.internal.heads()
    }

    fn label(&self, index: Index) -> Option<Index> {
        self.internal.label(index)
    }

    fn labels(&self) -> &[Option<Index>] {
        self.internal.labels()
    }

    fn leftmost(&self, index: Index, check_from: Option<Index>) -> Option<Index> {
        self.internal.rightmost(index, check_from)
    }

    fn rightmost(&self, index: Index, check_from: Option<Index>) -> Option<Index> {
        self.internal.leftmost(index, check_from)
    }

    fn actions(&self) -> &[Action] {
        self.internal.actions()
    }
}
