use std::iter::Peekable;

use lang::{Phrasal, Tokenized};
pub use self::text::*;

mod text;

#[derive(Clone, Debug)]
pub struct Transform<I, C> {
    iter: I,
    caller: C,
    fit: bool,
}

impl<I, P, T> Iterator for Transform<I, P>
where
    I: Iterator<Item = T>,
    P: Preprocess<T>,
{
    type Item = P::Output;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(x) = self.iter.next() {
            if self.fit {
                Some(self.caller.fit_transform_each(x))
            } else {
                Some(self.caller.transform_each(x))
            }
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

pub trait Preprocess<T> {
    type Output;

    fn fit<I: Iterator<Item = T>>(&mut self, xs: I) -> Peekable<I> {
        let mut iter = xs.peekable();
        while let Some(x) = iter.peek() {
            self.fit_each(x);
        }
        iter
    }

    #[allow(unused_variables)]
    fn fit_each(&mut self, x: &T) -> Option<Self::Output> {
        None
    }

    fn transform<I: Iterator<Item = T>>(&self, xs: I) -> Transform<I, &Self> {
        Transform {
            iter: xs,
            caller: self,
            fit: false,
        }
    }

    fn transform_each(&self, x: T) -> Self::Output;

    fn fit_transform<I: Iterator<Item = T>>(&mut self, xs: I) -> Transform<I, &Self> {
        Transform {
            iter: xs,
            caller: self,
            fit: true,
        }
    }

    fn fit_transform_each(&mut self, x: T) -> Self::Output {
        match self.fit_each(&x) {
            Some(y) => y,
            None => self.transform_each(x),
        }
    }
}
