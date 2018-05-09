use std::borrow::Borrow;
use std::convert::From;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::rc::Rc;

pub struct RcString(Rc<String>);

impl RcString {
    pub fn new(string: String) -> Self {
        Self::from(string)
    }

    pub fn as_str(&self) -> &str {
        &self
    }
}

impl Deref for RcString {
    type Target = String;

    #[inline(always)]
    fn deref(&self) -> &String {
        &&self.0
    }
}

impl Clone for RcString {
    #[inline]
    fn clone(&self) -> RcString {
        Self::from(self.0.clone())
    }
}

impl PartialEq for RcString {
    #[inline(always)]
    fn eq(&self, other: &RcString) -> bool {
        self.0 == other.0
    }

    #[inline(always)]
    fn ne(&self, other: &RcString) -> bool {
        self.0 != other.0
    }
}

impl Eq for RcString {}

impl Hash for RcString {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl fmt::Display for RcString {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl fmt::Debug for RcString {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl fmt::Pointer for RcString {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl From<String> for RcString {
    fn from(string: String) -> Self {
        Self::from(Rc::from(string))
    }
}

impl From<Rc<String>> for RcString {
    fn from(item: Rc<String>) -> Self {
        RcString(item)
    }
}

impl Borrow<str> for RcString {
    #[inline]
    fn borrow(&self) -> &str {
        self
    }
}

#[cfg(feature = "serialize")]
mod serialize {
    use serde::{Deserialize, Serialize};
    use serde::de::Deserializer;
    use serde::ser::Serializer;

    use super::RcString;

    impl Serialize for RcString {
        #[inline]
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            serializer.serialize_str(self)
        }
    }

    impl<'de> Deserialize<'de> for RcString {
        fn deserialize<D>(deserializer: D) -> Result<RcString, D::Error>
        where
            D: Deserializer<'de>,
        {
            String::deserialize(deserializer).map(|s| RcString::new(s.to_string()))
        }
    }
}
