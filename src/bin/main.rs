use l2::tensor::*;

fn main() {
    let t = Tensor::zeros(&[2, 2]);

    let x = t[&[0]];

    println!("{}", x);

    println!("hello {:?}", t);
}
