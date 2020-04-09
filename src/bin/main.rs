use l2::tensor::*;

fn main() {
    let t = Tensor {
        data: vec![1.0, 2.0, 3.0, 4.0],
        shape: vec![4],
        strides: vec![1],
    };

    let x = t.slice(&[[0, 4]]);

    println!("{:?}", t);
    println!("{:?}", x);
}
