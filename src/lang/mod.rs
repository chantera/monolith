pub(crate) use self::rcstring::RcString;
pub use self::simple::*;

pub mod prelude;
mod rcstring;
mod simple;

pub trait Tokenized {
    fn id(&self) -> usize;
    fn form(&self) -> &str;
    fn lemma(&self) -> Option<&str>;
    fn postag(&self) -> Option<&str>;
    fn head(&self) -> Option<usize>;
    fn deprel(&self) -> Option<&str>;
}

pub trait Phrasal {
    type Token: Tokenized;

    fn from_tokens(tokens: Vec<Self::Token>) -> Self;
    fn raw(&self) -> &str;

    fn token(&self, index: usize) -> Option<&Self::Token>;
}
