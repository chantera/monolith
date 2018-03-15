use std::borrow::Cow;
use std::fmt;
use std::ops::{Deref, Index};

use lang::{Phrasal, Tokenized};

#[derive(Debug)]
pub struct Token<'a> {
    id: usize,
    form: Cow<'a, str>,
    lemma: Option<Cow<'a, str>>,
    postag: Option<Cow<'a, str>>,
    head: Option<usize>,
    deprel: Option<Cow<'a, str>>,
}

impl<'a> Tokenized for Token<'a> {
    fn id(&self) -> usize {
        self.id
    }

    fn form(&self) -> &str {
        &self.form
    }

    fn lemma(&self) -> Option<&str> {
        self.lemma.as_ref().map(|x| x.deref())
    }

    fn postag(&self) -> Option<&str> {
        self.postag.as_ref().map(|x| x.deref())
    }

    fn head(&self) -> Option<usize> {
        self.head
    }

    fn deprel(&self) -> Option<&str> {
        self.deprel.as_ref().map(|x| x.deref())
    }
}

impl<'a> fmt::Display for Token<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "id: {}, form: {}", self.id, self.form)
    }
}

#[derive(Debug)]
pub struct Sentence<T: Tokenized> {
    raw: String,
    tokens: Vec<T>,
}

impl<T: Tokenized> Sentence<T> {
    fn new(tokens: Vec<T>) -> Self {
        Sentence {
            raw: tokens
                .iter()
                .map(|t| t.form().to_string())
                .collect::<Vec<String>>()
                .join(" "),
            tokens: tokens,
        }
    }
}

impl<T: Tokenized> fmt::Display for Sentence<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "raw: {}", self.raw)
    }
}

impl<T: Tokenized> Phrasal for Sentence<T> {
    type Token = T;

    fn from_tokens(tokens: Vec<T>) -> Self {
        Sentence {
            raw: tokens
                .iter()
                .map(|t| t.form().to_string())
                .collect::<Vec<String>>()
                .join(" "),
            tokens: tokens,
        }
    }

    fn raw(&self) -> &str {
        &self.raw
    }

    fn token(&self, index: usize) -> Option<&Self::Token> {
        self.tokens.get(index)
    }
}

impl<T: Tokenized> Index<usize> for Sentence<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.tokens[index]
    }
}
