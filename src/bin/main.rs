use l2;
use l2::tensor::*;

fn main() -> Result<(), l2::errors::TensorError> {
    // let t = Tensor {
    //     data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    //     shape: vec![2, 2, 2],
    //     strides: vec![4, 2, 1],
    // };

    // let x = t.slice(&[[0, 2], [0, 2], [0, 1]]);

    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let c = l2::sum(&a, -1)?;

    println!("{:?}", c);
    println!("{:?}", a);

    Ok(())
}
