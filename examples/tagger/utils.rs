use primitiv::{Device, devices};

#[cfg(feature = "primitiv-cuda")]
pub fn select_device(id: i32) -> Box<Device> {
    if id >= 0 {
        Box::new(devices::CUDA::new(id as u32))
    } else {
        Box::new(devices::Naive::new())
    }
}

#[cfg(not(feature = "primitiv-cuda"))]
#[allow(unused_variables)]
pub fn select_device(id: i32) -> Box<Device> {
    Box::new(devices::Naive::new())
}
