use primitiv::{Device, devices};

#[cfg(all(feature = "primitiv-cuda", feature = "primitiv-eigen"))]
pub fn select_device(id: i32) -> Box<Device> {
    if id >= 0 {
        eprintln!("Use CUDA device: id={}", id);
        Box::new(devices::CUDA::new(id as u32))
    } else {
        eprintln!("Use Eigen device");
        Box::new(devices::Eigen::new())
    }
}

#[cfg(all(feature = "primitiv-cuda", not(feature = "primitiv-eigen")))]
pub fn select_device(id: i32) -> Box<Device> {
    if id >= 0 {
        eprintln!("Use CUDA device: id={}", id);
        Box::new(devices::CUDA::new(id as u32))
    } else {
        eprintln!("Use Naive device");
        Box::new(devices::Naive::new())
    }
}

#[cfg(all(not(feature = "primitiv-cuda"), feature = "primitiv-eigen"))]
#[allow(unused_variables)]
pub fn select_device(id: i32) -> Box<Device> {
    eprintln!("Use Eigen device");
    Box::new(devices::Eigen::new())
}

#[cfg(all(not(feature = "primitiv-cuda"), not(feature = "primitiv-eigen")))]
#[allow(unused_variables)]
pub fn select_device(id: i32) -> Box<Device> {
    eprintln!("Use Naive device");
    Box::new(devices::Naive::new())
}
