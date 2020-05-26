use l2;
use l2::tensor::*;

fn main() -> Result<(), l2::errors::TensorError> {
    let x = Tensor::new(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
        &[2, 2, 2, 2],
    )
    .unwrap();
    let y = Tensor::new(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
        &[2, 2, 2, 2],
    )
    .unwrap();

    let c = l2::matmul(&x, &y)?;
    println!("{:?}", c);

    Ok(())
}
