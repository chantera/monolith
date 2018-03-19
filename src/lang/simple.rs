use std::borrow::Cow;
use std::fmt;
use std::ops::{Deref, Index};

use lang::{Phrasal, Tokenized};

#[derive(Clone, Debug)]
pub struct Token<'a> {
    id: usize,
    form: Cow<'a, str>,
    lemma: Option<Cow<'a, str>>,
    postag: Option<Cow<'a, str>>,
    head: Option<usize>,
    deprel: Option<Cow<'a, str>>,
}

impl<'a> Token<'a> {
    pub fn new<S: Into<Cow<'a, str>>>(
        id: usize,
        form: S,
        lemma: Option<S>,
        postag: Option<S>,
        head: Option<usize>,
        deprel: Option<S>,
    ) -> Self {
        Token {
            id: id,
            form: form.into(),
            lemma: lemma.map(|s| s.into()),
            postag: postag.map(|s| s.into()),
            head: head,
            deprel: deprel.map(|s| s.into()),
        }
    }
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

#[derive(Clone, Debug)]
pub struct Sentence<T: Tokenized> {
    raw: String,
    tokens: Vec<T>,
}

impl<'a> Sentence<Token<'a>> {
    pub fn new<S: Into<String>>(raw: S) -> Self {
        let raw = raw.into();
        let tokens = raw.split(" ")
            .enumerate()
            .map(|(i, word)| {
                Token::new(i, word.to_string(), None, None, None, None)
            })
            .collect::<Vec<Token>>();
        Sentence {
            raw: raw,
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

    fn tokens(&self) -> &Vec<Self::Token> {
        &self.tokens
    }
}

impl<T: Tokenized> Index<usize> for Sentence<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.tokens[index]
    }
}
