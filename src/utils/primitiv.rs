use primitiv::{Device, devices};
use utils::rand::env_seed;

#[cfg(all(feature = "primitiv-cuda", feature = "primitiv-eigen"))]
pub fn select_device(id: i32) -> Box<Device> {
    if id >= 0 {
        let id = id as u32;
        if let Some(seed) = env_seed() {
            let seed = seed as u32;
            eprintln!("Use CUDA device: id={}, seed={}", id, seed);
            Box::new(devices::CUDA::new_with_seed(id, seed))
        } else {
            eprintln!("Use CUDA device: id={}", id);
            Box::new(devices::CUDA::new(id))
        }
    } else {
        if let Some(seed) = env_seed() {
            let seed = seed as u32;
            eprintln!("Use Eigen device: seed={}", seed);
            Box::new(devices::Eigen::new_with_seed(seed))
        } else {
            eprintln!("Use Eigen device");
            Box::new(devices::Eigen::new())
        }
    }
}

#[cfg(all(feature = "primitiv-cuda", not(feature = "primitiv-eigen")))]
pub fn select_device(id: i32) -> Box<Device> {
    if id >= 0 {
        if let Some(seed) = env_seed() {
            let seed = seed as u32;
            eprintln!("Use CUDA device: id={}, seed={}", id, seed);
            Box::new(devices::CUDA::new_with_seed(id, seed))
        } else {
            eprintln!("Use CUDA device: id={}", id);
            Box::new(devices::CUDA::new(id))
        }
    } else {
        if let Some(seed) = env_seed() {
            let seed = seed as u32;
            eprintln!("Use Naive device: seed={}", seed);
            Box::new(devices::Naive::new_with_seed(seed))
        } else {
            eprintln!("Use Naive device");
            Box::new(devices::Naive::new())
        }
    }
}

#[cfg(all(not(feature = "primitiv-cuda"), feature = "primitiv-eigen"))]
#[allow(unused_variables)]
pub fn select_device(id: i32) -> Box<Device> {
    if let Some(seed) = env_seed() {
        let seed = seed as u32;
        eprintln!("Use Eigen device: seed={}", seed);
        Box::new(devices::Eigen::new_with_seed(seed))
    } else {
        eprintln!("Use Eigen device");
        Box::new(devices::Eigen::new())
    }
}

#[cfg(all(not(feature = "primitiv-cuda"), not(feature = "primitiv-eigen")))]
#[allow(unused_variables)]
pub fn select_device(id: i32) -> Box<Device> {
    if let Some(seed) = env_seed() {
        let seed = seed as u32;
        eprintln!("Use Naive device: seed={}", seed);
        Box::new(devices::Naive::new_with_seed(seed))
    } else {
        eprintln!("Use Naive device");
        Box::new(devices::Naive::new())
    }
}
