use l2;
use l2::tensor::*;

fn main() -> Result<(), l2::errors::TensorError> {
    let a = Tensor::normal(&[1, 2], 0.0, 1.0)?;
    let b = Tensor::normal(&[2, 2], 0.0, 1.0)?;

    let c = &a + &b;

    println!("{:?}", c);

    Ok(())
}
