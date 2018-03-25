use primitiv::{Device, devices};

#[cfg(primitiv_cuda)]
pub fn select_device(id: i32) -> Box<Device> {
    if id >= 0 {
        Box::new(devices::CUDA::new(id))
    } else {
        Box::new(devices::Naive::new())
    }
}

#[cfg(not(primitiv_cuda))]
#[allow(unused_variables)]
pub fn select_device(id: i32) -> Box<Device> {
    Box::new(devices::Naive::new())
}
