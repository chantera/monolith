extern crate monolith;

mod test_utils;

#[cfg(feature = "syntax")]
mod tests {
    use monolith::syntax::transition;
    use monolith::syntax::transition::prelude::*;
    use monolith::lang::prelude::*;
    use monolith::preprocessing::Vocab;

    use super::test_utils;

    #[cfg(feature = "dataset-conll")]
    #[test]
    fn test_arc_stardard_oracle() {
        let mut label_v = Vocab::with_default_token("root");
        let sentences = test_utils::mock::provide_conll_tokens();
        for sentence in &sentences {
            let sentence_len = sentence.len();
            let mut gold_heads = Vec::<u32>::with_capacity(sentence_len);
            let mut gold_labels = Vec::<u32>::with_capacity(sentence_len);
            for token in sentence.iter() {
                gold_heads.push(token.head().unwrap() as u32);
                gold_labels.push(label_v.add(token.deprel().unwrap()));
            }

            let capacity = transition::ArcStandard::estimate_num_actions(sentence_len);
            let mut state = transition::State::with_capacity(sentence_len as u32, capacity);
            while !transition::ArcStandard::is_terminal(&state) {
                let action = transition::ArcStandard::get_oracle(&state, &gold_heads, &gold_labels);
                assert!(action.is_some());
                assert!(
                    transition::ArcStandard::is_allowed(action.unwrap(), &state),
                    "action: `{:?}` is not allowed to apply to the state `{:?}`",
                    transition::ArcStandardActionType::from_action(action.unwrap()),
                    state
                );
                let result = transition::ArcStandard::apply(action.unwrap(), &mut state);
                assert!(result.is_ok());
            }
            let actions = state.actions();
            assert!(
                actions.len() == capacity,
                "num_actions: {}, estimate: {}",
                actions.len(),
                capacity
            );
            let heads_match = state.heads().iter().zip(&gold_heads).skip(1).all(
                |(&head, &gold_head)| head == Some(gold_head),
            );
            assert!(heads_match);
            let labels_match = state.labels().iter().zip(&gold_labels).skip(1).all(
                |(&label, &gold_label)| label == Some(gold_label),
            );
            assert!(labels_match);

            let result =
                transition::GoldState::new::<transition::ArcStandard>(&gold_heads, &gold_labels);
            assert!(result.is_ok());
            let gold_state = result.unwrap();
            assert!(actions == gold_state.actions());
        }
    }
}
