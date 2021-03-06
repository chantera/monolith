extern crate monolith;

#[cfg(feature = "app")]
use monolith::io::cache::{Cache, FromCache, IntoCache};
use monolith::lang::{Sentence, Token};
use monolith::preprocessing::{Preprocess, TextPreprocessor, Vocab};

static SENTENCE1: &'static str =
    "Pierre Vinken , 61 years old , will join the board as a nonexecutive director Nov. 29 .";
static SENTENCE2: &'static str = "No , it was n't Black Monday .";
static SENTENCE3: &'static str = "John loves Mary .";

fn get_samples<'a>() -> Vec<Sentence<Token<'a>>> {
    let text = vec![SENTENCE1, SENTENCE2];
    text.into_iter().map(|raw| Sentence::new(raw)).collect()
}

#[test]
fn test_preprocessor() {
    let sentences = get_samples();
    for (i, s) in sentences.iter().enumerate() {
        println!("sentence{}: {}", i, s);
    }
    let mut preprocessor = TextPreprocessor::new(Vocab::new());
    let word_ids = preprocessor.fit_transform(sentences.into_iter());
    assert_eq!(
        word_ids[0],
        &[1, 2, 3, 4, 5, 6, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    );
    assert_eq!(word_ids[1], &[18, 3, 19, 20, 21, 22, 23, 17]);

    let sentence3 = Sentence::new(SENTENCE3);
    let word_ids = preprocessor.transform(vec![sentence3.clone()].into_iter());
    assert_eq!(word_ids[0], &[0, 0, 0, 17]);
    preprocessor.fit(vec![sentence3.clone()].into_iter());
    let word_ids = preprocessor.transform(vec![sentence3].into_iter());
    assert_eq!(word_ids[0], &[24, 25, 26, 17]);
}

#[cfg(feature = "app")]
#[test]
fn test_serialize() {
    let mut v = Vocab::new();
    assert_eq!(v.add("Hello".to_string()), 1);
    assert_eq!(v.add("World".to_string()), 2);
    assert_eq!(v.add("!".to_string()), 3);

    v.into_cache("test_serialize").unwrap();
    assert!(Cache::new("vocab").exists("test_serialize"));
    let mut v2 = Vocab::from_cache("test_serialize").unwrap();
    assert_eq!(v2.add("Hello".to_string()), 1);
    assert_eq!(v2.add("World".to_string()), 2);
    assert_eq!(v2.add("!".to_string()), 3);
}
