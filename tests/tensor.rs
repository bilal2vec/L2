use l2::tensor::*;

#[test]
fn allocate_tensor() {
    let _t = Tensor {
        data: vec![0.0; 16],
    };
}
