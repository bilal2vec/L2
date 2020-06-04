use l2;
use l2::tensor::*;

fn main() -> Result<(), l2::errors::TensorError> {
    let x = Tensor::new(vec![-2.0], &[1])?;
    let y = Tensor::new(vec![5.0], &[1])?;

    let q = &x + &y;

    let z = Tensor::new(vec![-4.0], &[1])?;

    let f = &q * &z;

    println!("{:?}", f);

    let derivative = Tensor::new_no_grad(vec![1.0], &[1])?;

    f.backward(&derivative);

    Ok(())
}
