extern crate tch;
use tch::{Tensor, Kind, nn, Device};

fn main() {
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let t = Tensor::zeros(&[1, 10], (Kind::Float, device));
    let t = t * 2;
    let p = t.size();
    println!("{:?}",t.size());
    t.print();
}