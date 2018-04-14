use std::ffi::CStr;
use std::fmt;
use std::io::{Error, Result};
use std::mem;

use libc::{c_char, uname as libc_uname, utsname};

#[derive(Debug)]
pub struct Uname {
    pub sysname: String,
    pub nodename: String,
    pub release: String,
    pub version: String,
    pub machine: String,
    pub domainname: String,
}

impl Uname {
    pub fn new() -> Result<Self> {
        let mut n = unsafe { mem::zeroed() };
        let r = unsafe { libc_uname(&mut n) };
        if r == 0 {
            Ok(From::from(n))
        } else {
            Err(Error::last_os_error())
        }
    }
}

impl fmt::Display for Uname {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{} {} {} {} {} {}",
            self.sysname,
            self.nodename,
            self.release,
            self.version,
            self.machine,
            self.domainname
        )
    }
}

#[inline]
fn to_cstr(buf: &[c_char]) -> &CStr {
    unsafe { CStr::from_ptr(buf.as_ptr()) }
}

impl From<utsname> for Uname {
    fn from(n: utsname) -> Self {
        Uname {
            sysname: to_cstr(&n.sysname[..]).to_string_lossy().into_owned(),
            nodename: to_cstr(&n.nodename[..]).to_string_lossy().into_owned(),
            release: to_cstr(&n.release[..]).to_string_lossy().into_owned(),
            version: to_cstr(&n.version[..]).to_string_lossy().into_owned(),
            machine: to_cstr(&n.machine[..]).to_string_lossy().into_owned(),
            domainname: to_cstr(&n.domainname[..]).to_string_lossy().into_owned(),
        }
    }
}

pub fn uname() -> Result<Uname> {
    Uname::new()
}
