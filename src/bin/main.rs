use l2;
use l2::tensor::*;

fn main() -> Result<(), l2::errors::TensorError> {
    let x = Tensor::new(vec![-2.0], &[1])?;
    let y = Tensor::new(vec![5.0], &[1])?;

    let q = &x + &y;

    let z = Tensor::new(vec![-4.0], &[1])?;

    let f = &q * &z;

    let derivative = Tensor::new_no_grad(vec![1.0], &[1])?;

    f.backward(&derivative);

    println!("{:?}", f);

    Ok(())
}

// Tensor { data: [-12.0], shape: [1], strides: [1], track_grad: true,
//     lhs_parent: Some(Tensor { data: [3.0], shape: [1], strides: [1], track_grad: true,
//             lhs_parent: Some(Tensor { data: [-2.0], shape: [1], strides: [1], track_grad: true, lhs_parent: None, rhs_parent: None, create_op: None,
//                     derivative: RefCell { value: Some([-4.0]) } }),
//             rhs_parent: Some(Tensor { data: [5.0], shape: [1], strides: [1], track_grad: true, lhs_parent: None, rhs_parent: None, create_op: None,
//                     derivative: RefCell { value: Some([-4.0]) } }),
//             create_op: Some(Add),
//             derivative: RefCell { value: Some([-4.0]) }
//     }),
//     rhs_parent: Some(Tensor { data: [-4.0], shape: [1], strides: [1], track_grad: true, lhs_parent: None, rhs_parent: None, create_op: None,
//          derivative: RefCell { value: Some([3.0]) }
//     }),
//     create_op: Some(Mul),
//     derivative: RefCell { value: Some([1.0]) } }
