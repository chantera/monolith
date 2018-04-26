use std::borrow::Cow;
use std::fmt;
use std::io as std_io;
use std::ops::Deref;

use io as mod_io;
use lang::{Phrasal, Sentence, Tokenized};

pub trait ConllTokenized: Tokenized + mod_io::FromLine {
    fn cpostag(&self) -> Option<&str>;
    fn feats(&self) -> Option<Vec<&str>>;
    fn phead(&self) -> Option<usize>;
    fn pdeprel(&self) -> Option<&str>;
    fn root() -> Self;
}

#[derive(Debug)]
pub struct Token<'a> {
    id: usize,
    form: Cow<'a, str>,
    lemma: Option<Cow<'a, str>>,
    cpostag: Option<Cow<'a, str>>,
    postag: Option<Cow<'a, str>>,
    feats: Option<Vec<Cow<'a, str>>>,
    head: Option<usize>,
    deprel: Option<Cow<'a, str>>,
    phead: Option<usize>,
    pdeprel: Option<Cow<'a, str>>,
}

impl<'a> Token<'a> {
    pub fn new<S: Into<Cow<'a, str>>>(
        id: usize,
        form: S,
        lemma: Option<S>,
        cpostag: Option<S>,
        postag: Option<S>,
        feats: Option<Vec<S>>,
        head: Option<usize>,
        deprel: Option<S>,
        phead: Option<usize>,
        pdeprel: Option<S>,
    ) -> Self {
        Token {
            id: id,
            form: form.into(),
            lemma: lemma.map(|s| s.into()),
            cpostag: cpostag.map(|s| s.into()),
            postag: postag.map(|s| s.into()),
            feats: feats.map(|s| s.into_iter().map(|v| v.into()).collect::<Vec<_>>()),
            head: head,
            deprel: deprel.map(|s| s.into()),
            phead: phead,
            pdeprel: pdeprel.map(|s| s.into()),
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

impl<'a> ConllTokenized for Token<'a> {
    fn cpostag(&self) -> Option<&str> {
        self.cpostag.as_ref().map(|x| x.deref())
    }

    fn feats(&self) -> Option<Vec<&str>> {
        self.feats.as_ref().map(|x| {
            x.iter().map(|x| x.deref()).collect::<Vec<&str>>()
        })
    }

    fn phead(&self) -> Option<usize> {
        self.phead
    }

    fn pdeprel(&self) -> Option<&str> {
        self.pdeprel.as_ref().map(|x| x.deref())
    }

    fn root() -> Self {
        Self::new(
            0,
            "<ROOT>",
            Some("<ROOT>"),
            Some("ROOT"),
            Some("ROOT"),
            None,
            Some(0),
            Some("root"),
            None,
            None,
        )
    }
}

impl<'a> fmt::Display for Token<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "id: {}, form: {}", self.id, self.form)
    }
}

static CONLL_FIELD_DELIMITER: &'static str = "\t";
static CONLL_FEATS_DELIMITER: &'static str = "|";
static CONLL_EMPTY_FIELD: &'static str = "_";

#[inline]
fn parse_conll_required_usize_field(field: &str) -> Result<usize, std_io::Error> {
    field.parse::<usize>().map_err(|e| {
        std_io::Error::new(std_io::ErrorKind::InvalidData, e)
    })
}

#[inline]
fn parse_conll_optional_usize_field(field: &str) -> Result<Option<usize>, std_io::Error> {
    if field == CONLL_EMPTY_FIELD {
        Ok(None)
    } else {
        Ok(Some(field.parse::<usize>().map_err(|e| {
            std_io::Error::new(std_io::ErrorKind::InvalidData, e)
        })?))
    }
}

#[inline]
fn parse_conll_required_str_field(field: &str) -> Result<&str, std_io::Error> {
    Ok(field)
}

#[inline]
fn parse_conll_optional_str_field(field: &str) -> Result<Option<&str>, std_io::Error> {
    if field == CONLL_EMPTY_FIELD {
        Ok(None)
    } else {
        Ok(Some(field))
    }
}

fn require<T>(option: Option<T>) -> Result<T, std_io::Error> {
    match option {
        Some(val) => Ok(val),
        None => Err(std_io::Error::new(
            std_io::ErrorKind::InvalidData,
            "The value must not be `None`",
        )),
    }
}

impl<'a> mod_io::FromLine for Token<'a> {
    type Err = std_io::Error;

    fn from_line(line: &str) -> Result<Token<'a>, Self::Err> {
        let mut cols = line.split(CONLL_FIELD_DELIMITER);
        let token = Token::new(
            require(cols.next()).and_then(
                parse_conll_required_usize_field,
            )?,
            require(cols.next())
                .and_then(parse_conll_required_str_field)
                .map(|s| s.to_string())?,
            require(cols.next())
                .and_then(parse_conll_optional_str_field)?
                .map(|s| s.to_string()),
            require(cols.next())
                .and_then(parse_conll_optional_str_field)?
                .map(|s| s.to_string()),
            require(cols.next())
                .and_then(parse_conll_optional_str_field)?
                .map(|s| s.to_string()),
            require(cols.next())
                .and_then(parse_conll_optional_str_field)?
                .map(|s| {
                    s.split(CONLL_FEATS_DELIMITER)
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>()
                }),
            require(cols.next()).and_then(
                parse_conll_optional_usize_field,
            )?,
            require(cols.next())
                .and_then(parse_conll_optional_str_field)?
                .map(|s| s.to_string()),
            require(cols.next()).and_then(
                parse_conll_optional_usize_field,
            )?,
            require(cols.next())
                .and_then(parse_conll_optional_str_field)?
                .map(|s| s.to_string()),
        );
        if cols.next() == None {
            Ok(token)
        } else {
            Err(std_io::Error::new(
                std_io::ErrorKind::InvalidData,
                "str has more than 10 fields",
            ))
        }
    }
}

pub fn read_upto<R, S, T>(reader: &mut R, num: usize, buf: &mut Vec<S>) -> std_io::Result<usize>
where
    R: std_io::BufRead,
    S: Phrasal<Token = T>,
    T: ConllTokenized,
{
    let mut count = 0;
    let mut line = String::new();
    let mut tokens = vec![<T as ConllTokenized>::root()];
    while count < num {
        match reader.read_line(&mut line) {
            Ok(0) => {
                if tokens.len() > 1 {
                    buf.push(S::from_tokens(tokens));
                    count += 1;
                }
                break;
            }
            Ok(_) => {
                let line_trimmed = line.trim();
                if line_trimmed.is_empty() {
                    if tokens.len() > 1 {
                        buf.push(S::from_tokens(tokens));
                        count += 1;
                    }
                    tokens = vec![T::root()];
                } else if line_trimmed.starts_with("#") {
                    continue;
                } else {
                    tokens.push(try!(T::from_line(&line_trimmed).map_err(|e| {
                        std_io::Error::new(std_io::ErrorKind::InvalidData, e)
                    })));
                }
            }
            Err(ref e) if e.kind() == std_io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
        line.clear();
    }
    Ok(count)
}

pub type Reader<'a, R> = mod_io::Reader<R, Sentence<Token<'a>>>;

impl<'a, R: std_io::BufRead> mod_io::Read for Reader<'a, R> {
    type Item = Sentence<Token<'a>>;

    fn read_upto(&mut self, num: usize, buf: &mut Vec<Self::Item>) -> std_io::Result<usize> {
        read_upto(self.inner_mut(), num, buf)
    }
}
