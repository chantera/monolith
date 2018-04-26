use super::{Index, Action, TransitionState, TransitionMutableState, TransitionSystem, Error};

#[derive(Debug)]
pub enum ArcStandardActionType {
    Shift,
    LeftArc(Index),
    RightArc(Index),
}

impl ArcStandardActionType {
    pub fn from_action(action: Action) -> Self {
        if action == 0 {
            ArcStandardActionType::Shift
        } else {
            let label = (action - 1) >> 1;
            if (!action & 1) == 0 {
                ArcStandardActionType::LeftArc(label)
            } else {
                ArcStandardActionType::RightArc(label)
            }
        }
    }

    pub fn into_action(self) -> Action {
        match self {
            ArcStandardActionType::Shift => 0,
            ArcStandardActionType::LeftArc(label) => 1 + (label << 1),
            ArcStandardActionType::RightArc(label) => 2 + (label << 1),
        }
    }

    pub fn to_string(&self) -> String {
        match *self {
            ArcStandardActionType::Shift => "Shift".to_string(),
            ArcStandardActionType::LeftArc(label) => format!("LeftArc({})", label),
            ArcStandardActionType::RightArc(label) => format!("RightArc({})", label),
        }
    }

    pub fn num_action_types() -> usize {
        3
    }

    pub fn num_defined_actions(num_labels: usize) -> usize {
        1 + 2 * num_labels
    }
}

#[derive(Debug)]
pub struct ArcStandard;

impl ArcStandard {
    /// Shift: (s, i|b, A) => (s|i, b, A)
    pub fn apply_shift<S: TransitionMutableState>(state: &mut S) -> Result<(), Error> {
        debug_assert!(ArcStandard::is_allowed_shift(state));
        match state.buffer_head() {
            Some(b0) => {
                state.push(b0)?;
                state.advance()
            }
            None => Err(Error::InvalidOperation),
        }
    }

    /// Left Arc: (s|i|j, b, A) => (s|j, b, A +(j,l,i))
    pub fn apply_left_arc<S: TransitionMutableState>(
        state: &mut S,
        label: Index,
    ) -> Result<(), Error> {
        debug_assert!(ArcStandard::is_allowed_left_arc(state));
        let s0 = state.pop()?;
        let s1 = state.pop()?;
        state.add_arc(s1, s0, label)?;
        state.push(s0)
    }

    /// Right Arc: (s|i|j, b, A) => (s|i, b, A +(i,l,j))
    pub fn apply_right_arc<S: TransitionMutableState>(
        state: &mut S,
        label: Index,
    ) -> Result<(), Error> {
        debug_assert!(ArcStandard::is_allowed_right_arc(state));
        let s0 = state.pop()?;
        let s1 = state.stack_top().ok_or(Error::InvalidOperation)?;
        state.add_arc(s0, s1, label)
    }

    pub fn is_allowed_shift<S: TransitionState>(state: &S) -> bool {
        !state.is_buffer_empty()
    }

    pub fn is_allowed_left_arc<S: TransitionState>(state: &S) -> bool {
        state.stack_size() > 2
    }

    pub fn is_allowed_right_arc<S: TransitionState>(state: &S) -> bool {
        state.stack_size() > 1
    }

    pub fn done_children_right_of<S: TransitionState>(
        state: &S,
        gold_heads: &[Index],
        head: Index,
    ) -> bool {
        if let Some(mut index) = state.buffer_head() {
            let num_tokens = state.num_tokens() as u32;
            while index < num_tokens {
                let actual_head = gold_heads[index as usize];
                if actual_head == head {
                    return false;
                }
                index = if actual_head > index {
                    actual_head
                } else {
                    index + 1
                };
            }
        }
        true
    }
}

impl TransitionSystem for ArcStandard {
    fn num_action_types() -> usize {
        ArcStandardActionType::num_action_types()
    }

    fn num_defined_actions(num_labels: usize) -> usize {
        ArcStandardActionType::num_defined_actions(num_labels)
    }

    fn estimate_num_actions(num_tokens: usize) -> usize {
        2 * (num_tokens - 1)
    }

    fn apply<S: TransitionMutableState>(action: Action, state: &mut S) -> Result<(), Error> {
        match ArcStandardActionType::from_action(action) {
            ArcStandardActionType::Shift => ArcStandard::apply_shift(state)?,
            ArcStandardActionType::LeftArc(label) => ArcStandard::apply_left_arc(state, label)?,
            ArcStandardActionType::RightArc(label) => ArcStandard::apply_right_arc(state, label)?,
        }
        state.record(action)
    }

    fn is_allowed<S: TransitionState>(action: Action, state: &S) -> bool {
        match ArcStandardActionType::from_action(action) {
            ArcStandardActionType::Shift => ArcStandard::is_allowed_shift(state),
            ArcStandardActionType::LeftArc(_label) => ArcStandard::is_allowed_left_arc(state),
            ArcStandardActionType::RightArc(_label) => ArcStandard::is_allowed_right_arc(state),
        }
    }

    fn is_terminal<S: TransitionState>(state: &S) -> bool {
        state.is_buffer_empty() && state.stack_size() < 2
    }

    fn get_oracle<S: TransitionState>(
        state: &S,
        gold_heads: &[Index],
        gold_labels: &[Index],
    ) -> Option<Action> {
        if state.stack_size() >= 2 {
            let s0 = state.stack_top().unwrap();
            let s1 = state.stack(1).unwrap();
            if gold_heads[s0 as usize] == s1 &&
                ArcStandard::done_children_right_of(state, gold_heads, s0)
            {
                Some(
                    ArcStandardActionType::RightArc(gold_labels[s0 as usize]).into_action(),
                )
            } else if gold_heads[s1 as usize] == s0 {
                Some(
                    ArcStandardActionType::LeftArc(gold_labels[s1 as usize]).into_action(),
                )
            } else {
                Some(ArcStandardActionType::Shift.into_action())
            }
        } else if !state.is_buffer_empty() {
            Some(ArcStandardActionType::Shift.into_action())
        } else {
            None
        }
    }
}
