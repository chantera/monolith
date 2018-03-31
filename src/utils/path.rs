use std::io as std_io;
use std::env;
use std::ffi::{CStr, CString};
use std::mem;
use std::path::{Path, PathBuf, MAIN_SEPARATOR};
use std::ptr;

use libc;

pub fn expandtilde<P: AsRef<Path>>(path: P) -> PathBuf {
    let path_str = path.as_ref().to_str().unwrap();
    if !path_str.starts_with('~') {
        return path.as_ref().to_path_buf();
    }
    let i = path_str.find(MAIN_SEPARATOR).unwrap_or(path_str.len());
    let userhome: Option<PathBuf> = if i == 1 {
        env::home_dir()
    } else {
        match home_dir(&path_str[1..i]) {
            Ok(dir) => Some(dir),
            Err(_) => None,
        }
    };
    match userhome {
        Some(mut home) => {
            if i < path_str.len() - 1 {
                home.push(&path_str[i + 1..]);
            }
            home
        }
        None => path.as_ref().to_path_buf(),
    }
}

pub fn home_dir(username: &str) -> std_io::Result<PathBuf> {
    let username = username.as_bytes();
    let mut getpw_string_buf = [0; 4096];
    let mut passwd: libc::passwd = unsafe { mem::zeroed() };
    let mut passwd_out: *mut libc::passwd = ptr::null_mut();
    let result = if username.is_empty() {
        let uid = unsafe { libc::getuid() };
        unsafe {
            libc::getpwuid_r(
                uid,
                &mut passwd as *mut _,
                getpw_string_buf.as_mut_ptr(),
                getpw_string_buf.len() as libc::size_t,
                &mut passwd_out as *mut _,
            )
        }
    } else {
        let username = match CString::new(username) {
            Ok(name) => name,
            Err(_) => return Err(std_io::Error::from_raw_os_error(libc::ENOENT)),
        };
        unsafe {
            libc::getpwnam_r(
                username.as_ptr(),
                &mut passwd as *mut _,
                getpw_string_buf.as_mut_ptr(),
                getpw_string_buf.len() as libc::size_t,
                &mut passwd_out as *mut _,
            )
        }
    };
    if result == 0 {
        let s = unsafe { CStr::from_ptr(passwd.pw_dir) }.to_str();
        match s {
            Ok(s) => Ok(PathBuf::from(s)),
            Err(e) => Err(std_io::Error::new(std_io::ErrorKind::InvalidData, e)),
        }
    } else {
        Err(std_io::Error::from_raw_os_error(result))
    }
}
