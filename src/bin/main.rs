use l2;
use l2::tensor::*;

fn main() -> Result<(), l2::errors::TensorError> {
    // let t = Tensor {
    //     data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    //     shape: vec![2, 2, 2],
    //     strides: vec![4, 2, 1],
    // };

    // let x = t.slice(&[[0, 2], [0, 2], [0, 1]]);

    // let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    // let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();

    // let a = Tensor::zeros(&[3, 64, 64]).unwrap();
    // let b = Tensor::zeros(&[3, 64, 64]).unwrap();

    // let c = a.matmul(&b)?;
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

    let c = x.matmul(&y).unwrap();
    println!("{:?}", c);
    // println!("{:?}", a);

    Ok(())
}
