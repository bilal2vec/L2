use l2::tensor::*;

fn main() {
    let t = Tensor::zeros(&[4, 4]);

    println!("{:?}", t);
}
