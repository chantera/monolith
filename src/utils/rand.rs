use std::cell::UnsafeCell;
use std::rc::Rc;

pub use rand::Rng;
use rand::{RngCore, CryptoRng, SeedableRng, EntropyRng, IsaacRng, Error};
use rand::distributions::{Distribution, Uniform};
use rand::prng::hc128::Hc128Core;
use rand::reseeding::ReseedingRng;

use utils;

static ENV_RNG_SEED: &'static str = "SEED";
const THREAD_RNG_RESEED_THRESHOLD: u64 = 32 * 1024 * 1024; // 32 MiB

#[derive(Clone, Debug)]
pub struct ThreadRng {
    rng: Rc<UnsafeCell<ReseedingRng<Hc128Core, IsaacRng>>>,
}

pub fn env_seed() -> Option<u64> {
    ENV_SEED.with(|seed| seed.map(|ref v| *v))
}

thread_local!(
    static ENV_SEED: Option<u64> = {
        match utils::env::var::<_, u64>(ENV_RNG_SEED) {
            Ok(val) => Some(val),
            Err(e) => {
                match e {
                    utils::env::VarError::NotPresent => None,
                    _ => {
                        panic!(
                            "could not retrieve environment variable `{}`: {}",
                            ENV_RNG_SEED,
                            e
                        );
                    }
                }
            }
        }
    };

    static THREAD_RNG_KEY: Rc<UnsafeCell<ReseedingRng<Hc128Core, IsaacRng>>> = {
        let mut seeding_source = match env_seed() {
            Some(val) => IsaacRng::new_from_u64(val),
            None => {
                IsaacRng::from_rng(EntropyRng::new()).unwrap_or_else(|err| {
                    panic!("could not initialize thread_rng: {}", err);
                })
            }
        };
        let r = Hc128Core::from_rng(&mut seeding_source).unwrap_or_else(|err| {
            panic!("could not initialize thread_rng: {}", err);
        });
        let rng = ReseedingRng::new(r, THREAD_RNG_RESEED_THRESHOLD, seeding_source);
        Rc::new(UnsafeCell::new(rng))
    };
);

pub fn thread_rng() -> ThreadRng {
    ThreadRng { rng: THREAD_RNG_KEY.with(|t| t.clone()) }
}

impl RngCore for ThreadRng {
    #[inline(always)]
    fn next_u32(&mut self) -> u32 {
        unsafe { (*self.rng.get()).next_u32() }
    }

    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        unsafe { (*self.rng.get()).next_u64() }
    }

    fn fill_bytes(&mut self, bytes: &mut [u8]) {
        unsafe { (*self.rng.get()).fill_bytes(bytes) }
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
        unsafe { (*self.rng.get()).try_fill_bytes(dest) }
    }
}

impl CryptoRng for ThreadRng {}

pub fn random<T>() -> T
where
    Uniform: Distribution<T>,
{
    thread_rng().gen()
}
