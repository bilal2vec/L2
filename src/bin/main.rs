use l2::tensor::*;

#[allow(clippy::many_single_char_names)]
fn main() -> Result<(), l2::errors::TensorError> {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]).unwrap();

    let c = l2::matmul(&a, &b)?;

    c.backward();

    println!("{}", c);

    Ok(())
}
