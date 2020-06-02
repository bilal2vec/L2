use l2::tensor::*;

#[test]
fn allocate_tensor() {
    let _t = Tensor::zeros(&[2, 2]).unwrap();
}
