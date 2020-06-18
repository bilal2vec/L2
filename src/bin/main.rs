use l2::tensor::*;

#[allow(clippy::many_single_char_names)]
fn main() -> Result<(), l2::errors::TensorError> {
    let x: Tensor = Tensor::normal(&[2, 4], 0.0, 1.0)?;
    let y: Tensor = Tensor::normal(&[4, 1], 0.0, 1.0)?;

    let z: Tensor = l2::matmul(&x, &y)?;

    z.backward();

    println!("{}", z);

    Ok(())
}
