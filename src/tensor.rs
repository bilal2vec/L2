use std::cell::RefCell;
use std::cmp::Ordering;
use std::f32::consts::E;
use std::ops::{Add, Div, Mul, Sub};

use rand::distributions::{Distribution, Uniform};
use rand_distr::Normal;

use crate::errors::TensorError;
use crate::ops::{DimOp, TensorOp};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_tensor_from_vec() {
        let _t = Tensor {
            data: vec![0.0; 16],
            shape: vec![16],
            strides: vec![1],
            track_grad: false,
            lhs_parent: None,
            rhs_parent: None,
            create_op: None,
            derivative: RefCell::new(vec![0.0; 16]),
        };
    }

    #[test]
    fn allocate_tensor_zeros() {
        let _t = Tensor::zeros(&[2, 4]).unwrap();
    }

    #[test]
    #[should_panic]
    fn try_allocate_tensor_no_shape() {
        let _t = Tensor::zeros(&[]).unwrap();
    }

    #[test]
    #[should_panic]
    fn try_allocate_tensor_too_many_dims() {
        let _t = Tensor::zeros(&[2, 2, 2, 2, 2]).unwrap();
    }

    #[test]
    #[should_panic]
    fn try_slice_tensor_too_large_slice() {
        let t = Tensor::zeros(&[2, 2]).unwrap();
        let _x = t.slice(&[[0, 1], [0, 1], [0, 1]]).unwrap();
    }

    #[test]
    fn try_slice_tensor_negative_1() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let x = t.slice(&[[0, -1], [0, 1]]).unwrap();

        assert!((x.data == vec![1.0, 3.0]) && (x.shape == &[2]));
    }

    #[test]
    fn try_slice_fewer_dims() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let x = t.slice(&[[0, 1]]).unwrap();

        assert!((x.data == vec![1.0, 2.0]) && (x.shape == vec![2]));
    }

    #[test]
    #[should_panic]
    fn try_slice_tensor_start_greater_than_stop() {
        let t = Tensor::zeros(&[2, 2]).unwrap();
        let _x = t.slice(&[[0, 1], [1, 0]]).unwrap();
    }

    #[test]
    #[should_panic]
    fn try_slice_tensor_stop_greater_than_shape() {
        let t = Tensor::zeros(&[2, 2]).unwrap();
        let _x = t.slice(&[[0, 1], [0, 3]]).unwrap();
    }

    #[test]
    fn slice_tensor_1d_element() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();

        let x = t.slice(&[[0, 1]]).unwrap();

        assert!((x.data == vec![1.0]) && (x.shape == vec![1]) && (x.strides == vec![1]))
    }
    #[test]
    fn slice_tensor_2d_element() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let x = t.slice(&[[0, 1], [0, 1]]).unwrap();

        assert!((x.data == vec![1.0]) && (x.shape == vec![1]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_2d_row() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let x = t.slice(&[[0, 1], [0, 2]]).unwrap();

        assert!((x.data == vec![1.0, 2.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_2d_col() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let x = t.slice(&[[0, 2], [0, 1]]).unwrap();

        assert!((x.data == vec![1.0, 3.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_3d_element() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]).unwrap();

        let x = t.slice(&[[0, 1], [0, 1], [0, 1]]).unwrap();

        assert!((x.data == vec![1.0]) && (x.shape == vec![1]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_3d_row() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]).unwrap();

        let x = t.slice(&[[0, 1], [0, 1], [0, 2]]).unwrap();

        assert!((x.data == vec![1.0, 2.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_3d_col() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]).unwrap();

        let x = t.slice(&[[0, 1], [0, 2], [0, 1]]).unwrap();

        assert!((x.data == vec![1.0, 3.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_3d_channel() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]).unwrap();

        let x = t.slice(&[[0, 2], [0, 1], [0, 1]]).unwrap();

        assert!((x.data == vec![1.0, 5.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_3d_chunk() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]).unwrap();

        let x = t.slice(&[[0, 2], [0, 2], [0, 1]]).unwrap();

        assert!(
            (x.data == vec![1.0, 3.0, 5.0, 7.0])
                && (x.shape == &[2, 2])
                && (x.strides == vec![2, 1])
        )
    }
    #[test]
    fn slice_tensor_4d_element() {
        let t = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 2, 2],
        )
        .unwrap();

        let x = t.slice(&[[0, 1], [0, 1], [0, 1], [0, 1]]).unwrap();

        assert!((x.data == vec![1.0]) && (x.shape == vec![1]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_4d_row() {
        let t = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 2, 2],
        )
        .unwrap();

        let x = t.slice(&[[0, 1], [0, 1], [0, 1], [0, 2]]).unwrap();

        assert!((x.data == vec![1.0, 2.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_4d_col() {
        let t = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 2, 2],
        )
        .unwrap();

        let x = t.slice(&[[0, 1], [0, 1], [0, 2], [0, 1]]).unwrap();

        assert!((x.data == vec![1.0, 3.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_4d_channel() {
        let t = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 2, 2],
        )
        .unwrap();

        let x = t.slice(&[[0, 1], [0, 2], [0, 1], [0, 1]]).unwrap();

        assert!((x.data == vec![1.0, 5.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_4d_batch() {
        let t = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 2, 2],
        )
        .unwrap();

        let x = t.slice(&[[0, 2], [0, 1], [0, 1], [0, 1]]).unwrap();

        assert!((x.data == vec![1.0, 9.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_4d_chunk() {
        let t = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 2, 2],
        )
        .unwrap();

        let x = t.slice(&[[0, 2], [0, 2], [0, 1], [0, 1]]).unwrap();

        assert!(
            (x.data == vec![1.0, 5.0, 9.0, 13.0])
                && (x.shape == &[2, 2])
                && (x.strides == vec![2, 1])
        )
    }

    #[test]
    fn view_tensor() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let x = t.view(&[4]).unwrap();

        assert!((x.data == vec![1.0, 2.0, 3.0, 4.0]) && (x.shape == vec![4]));
    }

    #[test]
    fn view_tensor_neg_1() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let _x = t.view(&[-1]).unwrap();
    }

    #[test]
    fn view_tensor_neg_2() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let _x = t.view(&[2, -1]).unwrap();
    }

    #[test]
    fn view_tensor_neg_3() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let _x = t.view(&[1, -1]).unwrap();
    }

    #[test]
    fn view_tensor_neg_4() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let _x = t.view(&[4, -1]).unwrap();
    }

    #[test]
    #[should_panic]
    fn view_tensor_should_panic() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let _x = t.view(&[1, 1, 1, 1]).unwrap();
    }

    #[test]
    #[should_panic]
    fn view_tensor_neg_should_panic() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let _x = t.view(&[-1, -1]).unwrap();
    }

    #[test]
    #[should_panic]
    fn view_tensor_neg_should_panic_2() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let _x = t.view(&[6, -1]).unwrap();
    }

    #[test]
    fn elementwise_add_op() {
        let a = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[4]).unwrap();
        let b = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[4]).unwrap();

        let c = &a + &b;

        assert!((c.data == vec![4.0, 6.0, 8.0, 10.0]) && (c.shape == vec![4]))
    }

    #[test]
    fn elementwise_sub_op() {
        let a = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]).unwrap();
        let b = Tensor::new(vec![2.0, 3.0], &[2]).unwrap();

        let c = &a - &b;

        assert!((c.data == vec![0.0, 0.0, 2.0, 2.0]) && (c.shape == vec![2, 2]))
    }

    #[test]
    fn elementwise_mul_op() {
        let a = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]).unwrap();
        let b = Tensor::new(vec![2.0, 3.0], &[2]).unwrap();

        let c = &a * &b;

        assert!((c.data == vec![4.0, 9.0, 8.0, 15.0]) && (c.shape == vec![2, 2]))
    }

    #[test]
    fn elementwise_div_op() {
        let a = Tensor::new(vec![2.0, 4.0, 6.0, 8.0], &[2, 2]).unwrap();
        let b = Tensor::new(vec![2.0, 4.0], &[2]).unwrap();

        let c = &a / &b;

        assert!((c.data == vec![1.0, 1.0, 3.0, 2.0]) && (c.shape == vec![2, 2]))
    }

    #[test]
    fn broadcast_shapes() {
        let a = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[1, 4]).unwrap();
        let b = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[4]).unwrap();

        let c = &a + &b;

        assert!((c.data == vec![4.0, 6.0, 8.0, 10.0]) && (c.shape == vec![1, 4]));
    }

    #[test]
    fn broadcast_dims() {
        let a = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]).unwrap();
        let b = Tensor::new(vec![2.0, 3.0], &[2]).unwrap();

        let c = &a + &b;

        assert!((c.data == vec![4.0, 6.0, 6.0, 8.0]) && (c.shape == vec![2, 2]));
    }

    #[test]
    fn broadcast_shapes_and_dims() {
        let a = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[1, 2, 2]).unwrap();
        let b = Tensor::new(vec![2.0, 3.0], &[2]).unwrap();

        let c = &a + &b;

        assert!((c.data == vec![4.0, 6.0, 6.0, 8.0]) && (c.shape == vec![1, 2, 2]));
    }

    #[test]
    #[should_panic]
    fn broadcast_should_panic() {
        let a = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[4]).unwrap();
        let b = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]).unwrap();

        let _c = &a + &b;
    }

    #[test]
    fn pow() {
        let a = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[4]).unwrap();

        let b = a.pow(2).unwrap();

        assert!((b.data == vec![4.0, 9.0, 16.0, 25.0]) && (b.shape == vec![4]))
    }

    #[test]
    fn sum_1d() {
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[16],
        )
        .unwrap();

        let y = x.sum(0).unwrap();

        assert!((y.data == vec![136.0]) && (y.shape == vec![1]))
    }

    #[test]
    fn sum_2d() {
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 8],
        )
        .unwrap();

        let y = x.sum(1).unwrap();

        assert!((y.data == vec![36.0, 100.0]) && (y.shape == vec![2]))
    }

    #[test]
    fn sum_3d() {
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 4],
        )
        .unwrap();

        let y = x.sum(1).unwrap();

        assert!(
            (y.data == vec![6.0, 8.0, 10.0, 12.0, 22.0, 24.0, 26.0, 28.0])
                && (y.shape == vec![2, 4])
        )
    }

    #[test]
    fn sum_4d() {
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 2, 2],
        )
        .unwrap();

        let y = x.sum(-1).unwrap();

        assert!(
            (y.data == vec![3.0, 7.0, 11.0, 15.0, 19.0, 23.0, 27.0, 31.0])
                && (y.shape == vec![2, 2, 2])
        )
    }

    #[test]
    fn mean() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        let y = x.mean(-1).unwrap();

        assert!((y.data == vec![2.0, 5.0]) && (y.shape == vec![2]))
    }

    #[test]
    fn max() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        let y = x.max(-1).unwrap();

        assert!((y.data == vec![3.0, 6.0]) && (y.shape == vec![2]))
    }

    #[test]
    fn min() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        let y = x.min(-1).unwrap();

        assert!((y.data == vec![1.0, 4.0]) && (y.shape == vec![2]))
    }

    #[test]
    fn argmax() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        let y = x.argmax(-1).unwrap();

        assert!((y.data == vec![2.0, 2.0]) && (y.shape == vec![2]))
    }

    #[test]
    fn argmin() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        let y = x.argmin(-1).unwrap();

        assert!((y.data == vec![0.0, 0.0]) && (y.shape == vec![2]))
    }

    #[test]
    fn matmul_2d() {
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 8],
        )
        .unwrap();
        let y = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[8, 2],
        )
        .unwrap();

        let z = x.matmul(&y).unwrap();

        assert!((z.data == vec![372.0, 408.0, 884.0, 984.0]) && (z.shape == vec![2, 2]))
    }

    #[test]
    fn matmul_3d() {
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 4],
        )
        .unwrap();
        let y = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 4, 2],
        )
        .unwrap();

        let z = x.matmul(&y).unwrap();

        assert!(
            (z.data == vec![50.0, 60.0, 114.0, 140.0, 514.0, 556.0, 706.0, 764.0])
                && (z.shape == vec![2, 2, 2])
        )
    }

    #[test]
    fn matmul_4d() {
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 2, 2],
        )
        .unwrap();
        let y = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 2, 2],
        )
        .unwrap();

        let z = x.matmul(&y).unwrap();

        assert!(
            (z.data
                == vec![
                    7.0, 10.0, 15.0, 22.0, 67.0, 78.0, 91.0, 106.0, 191.0, 210.0, 231.0, 254.0,
                    379.0, 406.0, 435.0, 466.0
                ])
                && (z.shape == vec![2, 2, 2, 2])
        )
    }

    #[test]
    fn concat_1d() {
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[16],
        )
        .unwrap();
        let y = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[16],
        )
        .unwrap();

        let z = x.concat(&y, -1).unwrap();

        assert!(
            (z.data
                == vec![
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                    15.0, 16.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                    13.0, 14.0, 15.0, 16.0,
                ])
                && (z.shape == vec![32])
        )
    }

    #[test]
    fn concat_2d() {
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 8],
        )
        .unwrap();
        let y = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 8],
        )
        .unwrap();

        let z = x.concat(&y, -1).unwrap();

        assert!(
            (z.data
                == vec![
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                    9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 9.0, 10.0, 11.0, 12.0, 13.0,
                    14.0, 15.0, 16.0,
                ])
                && (z.shape == vec![2, 16])
        )
    }

    #[test]
    fn concat_3d() {
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 4],
        )
        .unwrap();
        let y = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 4],
        )
        .unwrap();

        let z = x.concat(&y, -1).unwrap();

        assert!(
            (z.data
                == vec![
                    1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0,
                    9.0, 10.0, 11.0, 12.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 13.0,
                    14.0, 15.0, 16.0
                ])
                && (z.shape == vec![2, 2, 8])
        )
    }

    #[test]
    fn concat_4d() {
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 2, 2],
        )
        .unwrap();
        let y = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 2, 2],
        )
        .unwrap();

        let z = x.concat(&y, -1).unwrap();

        assert!(
            (z.data
                == vec![
                    1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 6.0, 5.0, 6.0, 7.0, 8.0, 7.0, 8.0,
                    9.0, 10.0, 9.0, 10.0, 11.0, 12.0, 11.0, 12.0, 13.0, 14.0, 13.0, 14.0, 15.0,
                    16.0, 15.0, 16.0
                ])
                && (z.shape == vec![2, 2, 2, 4])
        )
    }

    #[test]
    fn transpose_1d() {
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[16],
        )
        .unwrap();

        let z = x.transpose().unwrap();

        assert!(
            (z.data
                == vec![
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                    15.0, 16.0,
                ])
                && (z.shape == vec![16])
        )
    }

    #[test]
    fn transpose_2d() {
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 8],
        )
        .unwrap();

        let z = x.transpose().unwrap();

        assert!(
            (z.data
                == vec![
                    1.0, 9.0, 2.0, 10.0, 3.0, 11.0, 4.0, 12.0, 5.0, 13.0, 6.0, 14.0, 7.0, 15.0,
                    8.0, 16.0
                ])
                && (z.shape == vec![8, 2])
        )
    }

    #[test]
    fn transpose_3d() {
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 4],
        )
        .unwrap();

        let z = x.transpose().unwrap();

        assert!(
            (z.data
                == vec![
                    1.0, 9.0, 5.0, 13.0, 2.0, 10.0, 6.0, 14.0, 3.0, 11.0, 7.0, 15.0, 4.0, 12.0,
                    8.0, 16.0
                ])
                && (z.shape == vec![4, 2, 2])
        )
    }

    #[test]
    fn transpose_4d() {
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 2, 2],
        )
        .unwrap();

        let z = x.transpose().unwrap();

        assert!(
            (z.data
                == vec![
                    1.0, 9.0, 5.0, 13.0, 3.0, 11.0, 7.0, 15.0, 2.0, 10.0, 6.0, 14.0, 4.0, 12.0,
                    8.0, 16.0
                ])
                && (z.shape == vec![2, 2, 2, 2])
        )
    }

    #[test]
    fn clone() {
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 2, 2],
        )
        .unwrap();

        let y = x.clone().unwrap();

        assert!((x.data == y.data) && (x.shape == y.shape))
    }
}

#[derive(Debug, PartialEq)]
pub struct Tensor<'a> {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,

    track_grad: bool,

    lhs_parent: Option<&'a Tensor<'a>>,
    rhs_parent: Option<&'a Tensor<'a>>,
    create_op: Option<TensorOp>,
    derivative: RefCell<Vec<f32>>,
}

impl<'a> Tensor<'a> {
    fn calc_tensor_len_from_shape(shape: &[usize]) -> usize {
        let mut length = 1;
        for i in shape {
            length *= i;
        }

        length
    }

    fn calc_strides_from_shape(shape: &[usize]) -> Vec<usize> {
        let mut strides = Vec::with_capacity(shape.len());

        let mut current_stride = 1;
        for i in shape.iter().rev() {
            strides.insert(0, current_stride);
            current_stride *= i;
        }

        strides
    }

    fn calc_shape_from_slice(slice: &[[usize; 2]]) -> Vec<usize> {
        // can't preallocate vec length because we don't know its length at compile time
        let mut slice_shape = Vec::new();

        for idx in slice {
            if idx[1] - idx[0] > 1 {
                slice_shape.push(idx[1] - idx[0]);
            }
        }

        if slice_shape.len() == 0 {
            slice_shape.push(1);
        }

        slice_shape
    }

    fn get_physical_idx(logical_indices: &[usize], strides: &[usize]) -> usize {
        let mut physical_idx = 0;

        for (i, idx) in logical_indices.iter().enumerate() {
            physical_idx += idx * strides[i];
        }

        physical_idx
    }

    fn one_dimension_slice(&self, logical_indices: &[[usize; 2]], slice_len: usize) -> Vec<f32> {
        let mut new_data = Vec::with_capacity(slice_len);

        for i in logical_indices[0][0]..logical_indices[0][1] {
            new_data.push(self.data[Tensor::get_physical_idx(&[i], &self.strides)]);
        }

        new_data
    }

    fn two_dimension_slice(&self, logical_indices: &[[usize; 2]], slice_len: usize) -> Vec<f32> {
        let mut new_data = Vec::with_capacity(slice_len);

        for i in logical_indices[0][0]..logical_indices[0][1] {
            for j in logical_indices[1][0]..logical_indices[1][1] {
                new_data.push(self.data[Tensor::get_physical_idx(&[i, j], &self.strides)]);
            }
        }

        new_data
    }

    fn three_dimension_slice(&self, logical_indices: &[[usize; 2]], slice_len: usize) -> Vec<f32> {
        let mut new_data = Vec::with_capacity(slice_len);

        for i in logical_indices[0][0]..logical_indices[0][1] {
            for j in logical_indices[1][0]..logical_indices[1][1] {
                for k in logical_indices[2][0]..logical_indices[2][1] {
                    new_data.push(self.data[Tensor::get_physical_idx(&[i, j, k], &self.strides)]);
                }
            }
        }

        new_data
    }

    fn four_dimension_slice(&self, logical_indices: &[[usize; 2]], slice_len: usize) -> Vec<f32> {
        let mut new_data = Vec::with_capacity(slice_len);

        for i in logical_indices[0][0]..logical_indices[0][1] {
            for j in logical_indices[1][0]..logical_indices[1][1] {
                for k in logical_indices[2][0]..logical_indices[2][1] {
                    for m in logical_indices[3][0]..logical_indices[3][1] {
                        new_data.push(
                            self.data[Tensor::get_physical_idx(&[i, j, k, m], &self.strides)],
                        );
                    }
                }
            }
        }

        new_data
    }

    fn validate_logical_indices<'b>(
        &self,
        logical_indices: &'b [[usize; 2]],
    ) -> Result<&'b [[usize; 2]], TensorError> {
        if logical_indices.len() != self.shape.len() {
            Err(TensorError::SliceError)
        } else {
            for i in 0..logical_indices.len() {
                if logical_indices[i][0] >= logical_indices[i][1]
                    || logical_indices[i][1] > self.shape[i]
                {
                    return Err(TensorError::SliceError);
                }
            }
            Ok(logical_indices)
        }
    }

    fn process_indices(&self, indices: &[[isize; 2]]) -> Vec<[usize; 2]> {
        let mut indices = indices.to_vec();
        let mut processed_indices: Vec<[usize; 2]> = Vec::with_capacity(indices.len());

        let diff = self.shape.len() - indices.len();
        if diff > 0 {
            for _ in 0..diff {
                indices.push([0, -1]);
            }
        }

        for i in 0..indices.len() {
            let start: usize = indices[i][0] as usize;
            let stop: usize = if indices[i][1] == -1 {
                self.shape[i]
            } else {
                indices[i][1] as usize
            };

            processed_indices.push([start, stop]);
        }

        processed_indices
    }

    fn process_view(&self, shape: &[isize]) -> Result<Vec<usize>, TensorError> {
        match shape.iter().filter(|&&val| val == -1).count() {
            0 => Ok(shape.iter().map(|&val| val as usize).collect()),
            1 => {
                let tensor_len = Tensor::calc_tensor_len_from_shape(&self.shape);

                let mut remainder_len = 1;
                for val in shape.iter() {
                    if *val != -1 {
                        remainder_len *= *val as usize;
                    }
                }

                // can't return an error
                let neg_1_idx = shape.iter().position(|&val| val == -1).unwrap();

                let mut shape: Vec<usize> = shape.iter().map(|&val| val as usize).collect();
                shape[neg_1_idx] = tensor_len / remainder_len;

                Ok(shape)
            }
            _ => Err(TensorError::ViewError),
        }
    }

    fn broadcast(
        lhs_shape: &Vec<usize>,
        rhs_shape: &Vec<usize>,
    ) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>), TensorError> {
        let lhs_shape = if lhs_shape.len() < rhs_shape.len() {
            let ones = vec![1; rhs_shape.len() - lhs_shape.len()];
            [&ones[..], &lhs_shape[..]].concat()
        } else {
            lhs_shape.clone()
        };

        let rhs_shape = if rhs_shape.len() < lhs_shape.len() {
            let ones = vec![1; lhs_shape.len() - rhs_shape.len()];
            [&ones[..], &rhs_shape[..]].concat()
        } else {
            rhs_shape.clone()
        };

        let mut broadcasted_shape: Vec<usize> = Vec::with_capacity(lhs_shape.len());
        let mut broadcasted_lhs_strides: Vec<usize> = Tensor::calc_strides_from_shape(&lhs_shape);
        let mut broadcasted_rhs_strides: Vec<usize> = Tensor::calc_strides_from_shape(&rhs_shape);

        for (i, (&lhs, &rhs)) in lhs_shape.iter().zip(rhs_shape.iter()).enumerate() {
            if lhs == rhs {
                broadcasted_shape.push(lhs);
            } else if lhs == 1 {
                broadcasted_shape.push(rhs);
                broadcasted_lhs_strides[i] = 0;
            } else if rhs == 1 {
                broadcasted_shape.push(lhs);
                broadcasted_rhs_strides[i] = 0;
            } else {
                return Err(TensorError::BroadcastError);
            }
        }

        Ok((
            broadcasted_shape,
            broadcasted_lhs_strides,
            broadcasted_rhs_strides,
        ))
    }

    fn op(lhs: &f32, rhs: &f32, op: &TensorOp) -> Result<f32, TensorError> {
        match op {
            TensorOp::Add => Ok(lhs + rhs),
            TensorOp::Sub => Ok(lhs - rhs),
            TensorOp::Mul => Ok(lhs * rhs),
            TensorOp::Div => Ok(lhs / rhs),
        }
    }

    fn one_dimension_tensor_op(
        &self,
        other: &Tensor,
        lhs_strides: &[usize],
        rhs_strides: &[usize],
        new_shape: &Vec<usize>,
        op: &TensorOp,
    ) -> Vec<f32> {
        let mut new_data: Vec<f32> =
            Vec::with_capacity(Tensor::calc_tensor_len_from_shape(new_shape));

        for i in 0..new_shape[0] {
            let op_result = Tensor::op(
                &self.data[Tensor::get_physical_idx(&[i], &lhs_strides)],
                &other.data[Tensor::get_physical_idx(&[i], &rhs_strides)],
                &op,
            )
            .unwrap();

            new_data.push(op_result);
        }

        new_data
    }

    fn two_dimension_tensor_op(
        &self,
        other: &Tensor,
        lhs_strides: &[usize],
        rhs_strides: &[usize],
        new_shape: &Vec<usize>,
        op: &TensorOp,
    ) -> Vec<f32> {
        let mut new_data: Vec<f32> =
            Vec::with_capacity(Tensor::calc_tensor_len_from_shape(new_shape));

        for i in 0..new_shape[0] {
            for j in 0..new_shape[1] {
                let op_result = Tensor::op(
                    &self.data[Tensor::get_physical_idx(&[i, j], &lhs_strides)],
                    &other.data[Tensor::get_physical_idx(&[i, j], &rhs_strides)],
                    &op,
                )
                .unwrap();

                new_data.push(op_result);
            }
        }

        new_data
    }

    fn three_dimension_tensor_op(
        &self,
        other: &Tensor,
        lhs_strides: &[usize],
        rhs_strides: &[usize],
        new_shape: &Vec<usize>,
        op: &TensorOp,
    ) -> Vec<f32> {
        let mut new_data: Vec<f32> =
            Vec::with_capacity(Tensor::calc_tensor_len_from_shape(new_shape));

        for i in 0..new_shape[0] {
            for j in 0..new_shape[1] {
                for k in 0..new_shape[2] {
                    let op_result = Tensor::op(
                        &self.data[Tensor::get_physical_idx(&[i, j, k], &lhs_strides)],
                        &other.data[Tensor::get_physical_idx(&[i, j, k], &rhs_strides)],
                        &op,
                    )
                    .unwrap();

                    new_data.push(op_result);
                }
            }
        }

        new_data
    }

    fn four_dimension_tensor_op(
        &self,
        other: &Tensor,
        lhs_strides: &[usize],
        rhs_strides: &[usize],
        new_shape: &Vec<usize>,
        op: &TensorOp,
    ) -> Vec<f32> {
        let mut new_data: Vec<f32> =
            Vec::with_capacity(Tensor::calc_tensor_len_from_shape(new_shape));

        for i in 0..new_shape[0] {
            for j in 0..new_shape[1] {
                for k in 0..new_shape[2] {
                    for m in 0..new_shape[3] {
                        let op_result = Tensor::op(
                            &self.data[Tensor::get_physical_idx(&[i, j, k, m], &lhs_strides)],
                            &other.data[Tensor::get_physical_idx(&[i, j, k, m], &rhs_strides)],
                            &op,
                        )
                        .unwrap();

                        new_data.push(op_result);
                    }
                }
            }
        }

        new_data
    }

    fn process_dims(
        idxs: &mut Vec<[isize; 2]>,
        dim: usize,
        current_dim: usize,
        current_idx: usize,
    ) {
        if dim == current_dim {
            idxs[current_dim] = [0, -1];
        } else {
            idxs[current_dim] = [current_idx as isize, current_idx as isize + 1];
        }
    }

    fn dim_op(lhs: &Tensor, op: &DimOp) -> Result<f32, TensorError> {
        match op {
            DimOp::Sum => Ok(lhs.data.iter().sum()),
            DimOp::Mean => {
                let sum: f32 = lhs.data.iter().sum();
                Ok(sum / lhs.data.len() as f32)
            }
            DimOp::Max => {
                let max = lhs
                    .data
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .map(|(_, val)| val);

                match max {
                    Some(x) => Ok(*x),
                    None => Err(TensorError::OpError),
                }
            }
            DimOp::Min => {
                let min = lhs
                    .data
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .map(|(_, val)| val);

                match min {
                    Some(x) => Ok(*x),
                    None => Err(TensorError::OpError),
                }
            }
            DimOp::Argmax => {
                let argmax = lhs
                    .data
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .map(|(i, _)| i);

                match argmax {
                    Some(x) => Ok(x as f32),
                    None => Err(TensorError::OpError),
                }
            }
            DimOp::Argmin => {
                let argmin = lhs
                    .data
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .map(|(i, _)| i);

                match argmin {
                    Some(x) => Ok(x as f32),
                    None => Err(TensorError::OpError),
                }
            }
        }
    }

    fn one_dimension_dimension_op(
        &self,
        new_dim: usize,
        new_shape: &Vec<usize>,
        op: &DimOp,
    ) -> Result<Vec<f32>, TensorError> {
        let mut new_data: Vec<f32> =
            Vec::with_capacity(Tensor::calc_tensor_len_from_shape(new_shape));

        let mut idxs: Vec<[isize; 2]> = vec![[0, 0]; self.shape.len()];

        let dim_zero_bounds = if new_dim == 0 { 1 } else { self.shape[0] };
        for i in 0..dim_zero_bounds {
            Tensor::process_dims(&mut idxs, new_dim, 0, i);

            new_data.push(Tensor::dim_op(&self.slice(&idxs)?, op)?);
        }

        Ok(new_data)
    }

    fn two_dimension_dimension_op(
        &self,
        new_dim: usize,
        new_shape: &Vec<usize>,
        op: &DimOp,
    ) -> Result<Vec<f32>, TensorError> {
        let mut new_data: Vec<f32> =
            Vec::with_capacity(Tensor::calc_tensor_len_from_shape(new_shape));

        let mut idxs: Vec<[isize; 2]> = vec![[0, 0]; self.shape.len()];

        let dim_zero_bounds = if new_dim == 0 { 1 } else { self.shape[0] };
        for i in 0..dim_zero_bounds {
            Tensor::process_dims(&mut idxs, new_dim, 0, i);

            let dim_one_bounds = if new_dim == 1 { 1 } else { self.shape[1] };
            for j in 0..dim_one_bounds {
                Tensor::process_dims(&mut idxs, new_dim, 1, j);

                new_data.push(Tensor::dim_op(&self.slice(&idxs)?, op)?);
            }
        }

        Ok(new_data)
    }

    fn three_dimension_dimension_op(
        &self,
        new_dim: usize,
        new_shape: &Vec<usize>,
        op: &DimOp,
    ) -> Result<Vec<f32>, TensorError> {
        let mut new_data: Vec<f32> =
            Vec::with_capacity(Tensor::calc_tensor_len_from_shape(new_shape));

        let mut idxs: Vec<[isize; 2]> = vec![[0, 0]; self.shape.len()];

        let dim_zero_bounds = if new_dim == 0 { 1 } else { self.shape[0] };
        for i in 0..dim_zero_bounds {
            Tensor::process_dims(&mut idxs, new_dim, 0, i);

            let dim_one_bounds = if new_dim == 1 { 1 } else { self.shape[1] };
            for j in 0..dim_one_bounds {
                Tensor::process_dims(&mut idxs, new_dim, 1, j);

                let dim_two_bounds = if new_dim == 2 { 1 } else { self.shape[2] };
                for k in 0..dim_two_bounds {
                    Tensor::process_dims(&mut idxs, new_dim, 2, k);

                    new_data.push(Tensor::dim_op(&self.slice(&idxs)?, op)?);
                }
            }
        }

        Ok(new_data)
    }

    fn four_dimension_dimension_op(
        &self,
        new_dim: usize,
        new_shape: &Vec<usize>,
        op: &DimOp,
    ) -> Result<Vec<f32>, TensorError> {
        let mut new_data: Vec<f32> =
            Vec::with_capacity(Tensor::calc_tensor_len_from_shape(new_shape));

        let mut idxs: Vec<[isize; 2]> = vec![[0, 0]; self.shape.len()];

        let dim_zero_bounds = if new_dim == 0 { 1 } else { self.shape[0] };
        for i in 0..dim_zero_bounds {
            Tensor::process_dims(&mut idxs, new_dim, 0, i);

            let dim_one_bounds = if new_dim == 1 { 1 } else { self.shape[1] };
            for j in 0..dim_one_bounds {
                Tensor::process_dims(&mut idxs, new_dim, 1, j);

                let dim_two_bounds = if new_dim == 2 { 1 } else { self.shape[2] };
                for k in 0..dim_two_bounds {
                    Tensor::process_dims(&mut idxs, new_dim, 2, k);

                    let dim_three_bounds = if new_dim == 3 { 1 } else { self.shape[3] };
                    for m in 0..dim_three_bounds {
                        Tensor::process_dims(&mut idxs, new_dim, 3, m);

                        new_data.push(Tensor::dim_op(&self.slice(&idxs)?, op)?);
                    }
                }
            }
        }

        Ok(new_data)
    }

    fn dimension_op(&self, dim: isize, op: DimOp) -> Result<Tensor, TensorError> {
        let new_dim = if dim == -1 {
            self.shape.len() - 1 as usize
        } else if (dim >= 0) && (dim < self.shape.len() as isize) {
            dim as usize
        } else {
            return Err(TensorError::DimError);
        };

        let mut new_shape: Vec<usize> = Vec::new();
        for i in 0..self.shape.len() {
            if i != new_dim {
                new_shape.push(self.shape[i]);
            }
        }

        if new_shape.len() == 0 {
            new_shape.push(1);
        }

        let new_data = match self.shape.len() {
            1 => self.one_dimension_dimension_op(new_dim, &new_shape, &op),
            2 => self.two_dimension_dimension_op(new_dim, &new_shape, &op),
            3 => self.three_dimension_dimension_op(new_dim, &new_shape, &op),
            4 => self.four_dimension_dimension_op(new_dim, &new_shape, &op),
            _ => Err(TensorError::MaxDimsError),
        }?;

        Tensor::new(new_data, &new_shape)
    }

    fn tensor_op<'b>(&'b self, other: &'b Tensor, op: TensorOp) -> Result<Tensor<'b>, TensorError> {
        let (new_shape, lhs_strides, rhs_strides) = Tensor::broadcast(&self.shape, &other.shape)?;

        let new_data = match new_shape.len() {
            1 => Ok(self.one_dimension_tensor_op(
                &other,
                &lhs_strides,
                &rhs_strides,
                &new_shape,
                &op,
            )),
            2 => Ok(self.two_dimension_tensor_op(
                &other,
                &lhs_strides,
                &rhs_strides,
                &new_shape,
                &op,
            )),
            3 => Ok(self.three_dimension_tensor_op(
                &other,
                &lhs_strides,
                &rhs_strides,
                &new_shape,
                &op,
            )),
            4 => Ok(self.four_dimension_tensor_op(
                &other,
                &lhs_strides,
                &rhs_strides,
                &new_shape,
                &op,
            )),
            _ => Err(TensorError::MaxDimsError),
        }?;

        if self.track_grad && other.track_grad {
            Tensor::new_with_parents(new_data, &new_shape, self, other, op)
        } else {
            Tensor::new_no_grad(new_data, &new_shape)
        }
    }

    fn validate_tensors(lhs: &Tensor, rhs: &Tensor) -> Result<Vec<usize>, TensorError> {
        let lhs_shape = lhs.shape.clone();
        let rhs_shape = rhs.shape.clone();

        if (lhs_shape.len() == rhs_shape.len())
            && (lhs_shape[lhs_shape.len() - 1] == rhs_shape[rhs_shape.len() - 2])
        {
            let mut new_shape = lhs_shape.clone();
            new_shape[lhs_shape.len() - 1] = rhs_shape[rhs_shape.len() - 1];

            Ok(new_shape)
        } else {
            Err(TensorError::MatmulShapeError)
        }
    }

    fn two_dimension_matmul(
        lhs: &Tensor,
        rhs: &Tensor,
        dim: usize,
    ) -> Result<Vec<f32>, TensorError> {
        let mut new_data: Vec<f32> = Vec::new();

        for i in 0..dim as isize {
            let a = lhs.slice(&[[i, i + 1], [0, -1]])?;
            let a = a.view(&[-1, 1])?;

            let c: Tensor = &a * rhs;
            let d = c.sum(0)?;

            new_data.append(&mut d.data.clone());
        }

        Ok(new_data)
    }

    fn three_dimension_matmul(
        &self,
        rhs: &Tensor,
        dim: usize,
        dims: &Vec<usize>,
    ) -> Result<Vec<f32>, TensorError> {
        let mut idxs: Vec<[isize; 2]> = vec![[0, -1]; 3];

        let lhs_shape: Vec<isize> = self.shape[self.shape.len() - 2..]
            .to_vec()
            .iter()
            .map(|&val| val as isize)
            .collect();
        let rhs_shape: Vec<isize> = rhs.shape[rhs.shape.len() - 2..]
            .to_vec()
            .iter()
            .map(|&val| val as isize)
            .collect();

        let mut new_data: Vec<f32> = Vec::new();

        for i in 0..dims[0] {
            idxs[0] = [i as isize, i as isize + 1];

            let mut temp = Tensor::two_dimension_matmul(
                &self.slice(&idxs)?.view(&lhs_shape)?,
                &rhs.slice(&idxs)?.view(&rhs_shape)?,
                dim,
            )?;
            new_data.append(&mut temp);
        }

        Ok(new_data)
    }

    fn four_dimension_matmul(
        &self,
        rhs: &Tensor,
        dim: usize,
        dims: &Vec<usize>,
    ) -> Result<Vec<f32>, TensorError> {
        let mut idxs: Vec<[isize; 2]> = vec![[0, -1]; 4];

        let lhs_shape: Vec<isize> = self.shape[self.shape.len() - 2..]
            .to_vec()
            .iter()
            .map(|&val| val as isize)
            .collect();
        let rhs_shape: Vec<isize> = rhs.shape[rhs.shape.len() - 2..]
            .to_vec()
            .iter()
            .map(|&val| val as isize)
            .collect();

        let mut new_data: Vec<f32> = Vec::new();

        for i in 0..dims[0] {
            idxs[0] = [i as isize, i as isize + 1];

            for j in 0..dims[1] {
                idxs[1] = [j as isize, j as isize + 1];

                let mut temp = Tensor::two_dimension_matmul(
                    &self.slice(&idxs)?.view(&lhs_shape)?,
                    &rhs.slice(&idxs)?.view(&rhs_shape)?,
                    dim,
                )?;
                new_data.append(&mut temp);
            }
        }

        Ok(new_data)
    }

    fn one_dimension_concat(
        &self,
        rhs: &Tensor,
        dim: usize,
        shape: &Vec<usize>,
    ) -> Result<Vec<f32>, TensorError> {
        let mut new_data: Vec<f32> = Vec::with_capacity(Tensor::calc_tensor_len_from_shape(shape));

        let mut idxs: Vec<[isize; 2]> = vec![[0, 0]; self.shape.len()];

        let dim_zero_bounds = if dim == 0 { 1 } else { self.shape[0] };
        for i in 0..dim_zero_bounds {
            Tensor::process_dims(&mut idxs, dim, 0, i);

            new_data.extend(self.slice(&idxs)?.data);
            new_data.extend(rhs.slice(&idxs)?.data);
        }

        Ok(new_data)
    }

    fn two_dimension_concat(
        &self,
        rhs: &Tensor,
        dim: usize,
        shape: &Vec<usize>,
    ) -> Result<Vec<f32>, TensorError> {
        let mut new_data: Vec<f32> = Vec::with_capacity(Tensor::calc_tensor_len_from_shape(shape));

        let mut idxs: Vec<[isize; 2]> = vec![[0, 0]; self.shape.len()];

        let dim_zero_bounds = if dim == 0 { 1 } else { self.shape[0] };
        for i in 0..dim_zero_bounds {
            Tensor::process_dims(&mut idxs, dim, 0, i);

            let dim_one_bounds = if dim == 1 { 1 } else { self.shape[1] };
            for j in 0..dim_one_bounds {
                Tensor::process_dims(&mut idxs, dim, 1, j);

                new_data.extend(self.slice(&idxs)?.data);
                new_data.extend(rhs.slice(&idxs)?.data);
            }
        }

        Ok(new_data)
    }

    fn three_dimension_concat(
        &self,
        rhs: &Tensor,
        dim: usize,
        shape: &Vec<usize>,
    ) -> Result<Vec<f32>, TensorError> {
        let mut new_data: Vec<f32> = Vec::with_capacity(Tensor::calc_tensor_len_from_shape(shape));

        let mut idxs: Vec<[isize; 2]> = vec![[0, 0]; self.shape.len()];

        let dim_zero_bounds = if dim == 0 { 1 } else { self.shape[0] };
        for i in 0..dim_zero_bounds {
            Tensor::process_dims(&mut idxs, dim, 0, i);

            let dim_one_bounds = if dim == 1 { 1 } else { self.shape[1] };
            for j in 0..dim_one_bounds {
                Tensor::process_dims(&mut idxs, dim, 1, j);

                let dim_two_bounds = if dim == 2 { 1 } else { self.shape[2] };
                for k in 0..dim_two_bounds {
                    Tensor::process_dims(&mut idxs, dim, 2, k);

                    new_data.extend(self.slice(&idxs)?.data);
                    new_data.extend(rhs.slice(&idxs)?.data);
                }
            }
        }

        Ok(new_data)
    }

    fn four_dimension_concat(
        &self,
        rhs: &Tensor,
        dim: usize,
        shape: &Vec<usize>,
    ) -> Result<Vec<f32>, TensorError> {
        let mut new_data: Vec<f32> = Vec::with_capacity(Tensor::calc_tensor_len_from_shape(shape));

        let mut idxs: Vec<[isize; 2]> = vec![[0, 0]; self.shape.len()];

        let dim_zero_bounds = if dim == 0 { 1 } else { self.shape[0] };
        for i in 0..dim_zero_bounds {
            Tensor::process_dims(&mut idxs, dim, 0, i);

            let dim_one_bounds = if dim == 1 { 1 } else { self.shape[1] };
            for j in 0..dim_one_bounds {
                Tensor::process_dims(&mut idxs, dim, 1, j);

                let dim_two_bounds = if dim == 2 { 1 } else { self.shape[2] };
                for k in 0..dim_two_bounds {
                    Tensor::process_dims(&mut idxs, dim, 2, k);

                    let dim_three_bounds = if dim == 3 { 1 } else { self.shape[3] };
                    for m in 0..dim_three_bounds {
                        Tensor::process_dims(&mut idxs, dim, 3, m);

                        new_data.extend(self.slice(&idxs)?.data);
                        new_data.extend(rhs.slice(&idxs)?.data);
                    }
                }
            }
        }

        Ok(new_data)
    }

    fn one_dimension_transpose(
        &self,
        shape: &Vec<usize>,
        strides: &Vec<usize>,
    ) -> Result<Vec<f32>, TensorError> {
        let mut new_data: Vec<f32> = Vec::with_capacity(Tensor::calc_tensor_len_from_shape(shape));

        for i in 0..self.shape[0] {
            new_data.push(self.data[Tensor::get_physical_idx(&[i], strides)]);
        }

        Ok(new_data)
    }

    fn two_dimension_transpose(
        &self,
        shape: &Vec<usize>,
        strides: &Vec<usize>,
    ) -> Result<Vec<f32>, TensorError> {
        let mut new_data: Vec<f32> = Vec::with_capacity(Tensor::calc_tensor_len_from_shape(shape));

        for i in 0..shape[0] {
            for j in 0..shape[1] {
                new_data.push(self.data[Tensor::get_physical_idx(&[i, j], strides)]);
            }
        }

        Ok(new_data)
    }

    fn three_dimension_transpose(
        &self,
        shape: &Vec<usize>,
        strides: &Vec<usize>,
    ) -> Result<Vec<f32>, TensorError> {
        let mut new_data: Vec<f32> = Vec::with_capacity(Tensor::calc_tensor_len_from_shape(shape));

        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    new_data.push(self.data[Tensor::get_physical_idx(&[i, j, k], strides)]);
                }
            }
        }

        Ok(new_data)
    }

    fn four_dimension_transpose(
        &self,
        shape: &Vec<usize>,
        strides: &Vec<usize>,
    ) -> Result<Vec<f32>, TensorError> {
        let mut new_data: Vec<f32> = Vec::with_capacity(Tensor::calc_tensor_len_from_shape(shape));

        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    for m in 0..shape[3] {
                        new_data.push(self.data[Tensor::get_physical_idx(&[i, j, k, m], strides)]);
                    }
                }
            }
        }

        Ok(new_data)
    }

    fn new_with_parents<'b>(
        data: Vec<f32>,
        shape: &[usize],
        lhs_parent: &'b Tensor,
        rhs_parent: &'b Tensor,
        op: TensorOp,
    ) -> Result<Tensor<'b>, TensorError> {
        if data.len() == Tensor::calc_tensor_len_from_shape(shape)
            && shape.len() > 0
            && shape.len() < 5
        {
            Ok(Tensor {
                data,
                shape: shape.to_vec(),
                strides: Tensor::calc_strides_from_shape(shape),
                track_grad: true,
                lhs_parent: Some(lhs_parent),
                rhs_parent: Some(rhs_parent),
                create_op: Some(op),
                derivative: RefCell::new(vec![0.0; Tensor::calc_tensor_len_from_shape(shape)]),
            })
        } else {
            Err(TensorError::InvalidTensor)
        }
    }

    pub fn new<'b>(data: Vec<f32>, shape: &[usize]) -> Result<Tensor<'b>, TensorError> {
        if data.len() == Tensor::calc_tensor_len_from_shape(shape)
            && shape.len() > 0
            && shape.len() < 5
        {
            Ok(Tensor {
                data,
                shape: shape.to_vec(),
                strides: Tensor::calc_strides_from_shape(shape),
                track_grad: true,
                lhs_parent: None,
                rhs_parent: None,
                create_op: None,
                derivative: RefCell::new(vec![0.0; Tensor::calc_tensor_len_from_shape(shape)]),
            })
        } else {
            Err(TensorError::InvalidTensor)
        }
    }

    pub fn new_no_grad<'b>(data: Vec<f32>, shape: &[usize]) -> Result<Tensor<'b>, TensorError> {
        if data.len() == Tensor::calc_tensor_len_from_shape(shape)
            && shape.len() > 0
            && shape.len() < 5
        {
            Ok(Tensor {
                data,
                shape: shape.to_vec(),
                strides: Tensor::calc_strides_from_shape(shape),
                track_grad: false,
                lhs_parent: None,
                rhs_parent: None,
                create_op: None,
                derivative: RefCell::new(vec![0.0; Tensor::calc_tensor_len_from_shape(shape)]),
            })
        } else {
            Err(TensorError::InvalidTensor)
        }
    }
    pub fn zeros<'b>(shape: &[usize]) -> Result<Tensor<'b>, TensorError> {
        Tensor::new(vec![0.0; Tensor::calc_tensor_len_from_shape(shape)], shape)
    }

    pub fn normal<'b>(shape: &[usize], mean: f32, std: f32) -> Result<Tensor<'b>, TensorError> {
        let normal = Normal::new(mean, std).unwrap();
        let mut rng = rand::thread_rng();

        let tensor_len = Tensor::calc_tensor_len_from_shape(shape);
        let mut new_data: Vec<f32> = Vec::with_capacity(tensor_len);

        for _ in 0..tensor_len {
            new_data.push(normal.sample(&mut rng));
        }

        Tensor::new(new_data, shape)
    }

    pub fn uniform<'b>(shape: &[usize], low: f32, high: f32) -> Result<Tensor<'b>, TensorError> {
        let uniform = Uniform::from(low..high);
        let mut rng = rand::thread_rng();

        let tensor_len = Tensor::calc_tensor_len_from_shape(shape);
        let mut new_data: Vec<f32> = Vec::with_capacity(tensor_len);

        for _ in 0..tensor_len {
            new_data.push(uniform.sample(&mut rng));
        }

        Tensor::new(new_data, shape)
    }

    pub fn slice(&self, logical_indices: &[[isize; 2]]) -> Result<Self, TensorError> {
        let logical_indices = self.process_indices(logical_indices);

        let logical_indices = self.validate_logical_indices(&logical_indices)?;

        // converting to a slice b/c can't move `new_shape` to tensor and pass a reference to it to `Tensor::calc_strides_from_shape
        let new_shape = Tensor::calc_shape_from_slice(logical_indices);
        let slice_len = Tensor::calc_tensor_len_from_shape(&new_shape);

        let new_data = match logical_indices.len() {
            1 => Ok(self.one_dimension_slice(logical_indices, slice_len)),
            2 => Ok(self.two_dimension_slice(logical_indices, slice_len)),
            3 => Ok(self.three_dimension_slice(logical_indices, slice_len)),
            4 => Ok(self.four_dimension_slice(logical_indices, slice_len)),
            _ => Err(TensorError::MaxDimsError),
        }?;

        Tensor::new(new_data, &new_shape)
    }

    pub fn view(&self, shape: &[isize]) -> Result<Self, TensorError> {
        let shape = self.process_view(shape)?;

        match Tensor::calc_tensor_len_from_shape(&self.shape)
            == Tensor::calc_tensor_len_from_shape(&shape)
        {
            true => Tensor::new(self.data.clone(), &shape),
            false => Err(TensorError::ViewError),
        }
    }

    pub fn pow(&self, exp: usize) -> Result<Tensor, TensorError> {
        let new_data = self.data.iter().map(|val| val.powi(exp as i32)).collect();

        Tensor::new(new_data, &self.shape)
    }

    pub fn sqrt(&self) -> Result<Tensor, TensorError> {
        let new_data = self.data.iter().map(|val| val.sqrt()).collect();

        Tensor::new(new_data, &self.shape)
    }

    pub fn exp(&self) -> Result<Tensor, TensorError> {
        let new_data = self.data.iter().map(|val| val.exp()).collect();

        Tensor::new(new_data, &self.shape)
    }

    pub fn log10(&self) -> Result<Tensor, TensorError> {
        let new_data = self.data.iter().map(|val| val.log10()).collect();

        Tensor::new(new_data, &self.shape)
    }

    pub fn log(&self) -> Result<Tensor, TensorError> {
        let new_data = self.data.iter().map(|val| val.log(E)).collect();

        Tensor::new(new_data, &self.shape)
    }

    pub fn abs(&self) -> Result<Tensor, TensorError> {
        let new_data = self.data.iter().map(|val| val.abs()).collect();

        Tensor::new(new_data, &self.shape)
    }

    pub fn sin(&self) -> Result<Tensor, TensorError> {
        let new_data = self.data.iter().map(|val| val.sin()).collect();

        Tensor::new(new_data, &self.shape)
    }

    pub fn cos(&self) -> Result<Tensor, TensorError> {
        let new_data = self.data.iter().map(|val| val.cos()).collect();

        Tensor::new(new_data, &self.shape)
    }

    pub fn tan(&self) -> Result<Tensor, TensorError> {
        let new_data = self.data.iter().map(|val| val.tan()).collect();

        Tensor::new(new_data, &self.shape)
    }

    pub fn sum(&self, dim: isize) -> Result<Tensor, TensorError> {
        self.dimension_op(dim, DimOp::Sum)
    }

    pub fn mean(&self, dim: isize) -> Result<Tensor, TensorError> {
        self.dimension_op(dim, DimOp::Mean)
    }

    pub fn max(&self, dim: isize) -> Result<Tensor, TensorError> {
        self.dimension_op(dim, DimOp::Max)
    }

    pub fn min(&self, dim: isize) -> Result<Tensor, TensorError> {
        self.dimension_op(dim, DimOp::Min)
    }

    pub fn argmax(&self, dim: isize) -> Result<Tensor, TensorError> {
        self.dimension_op(dim, DimOp::Argmax)
    }

    pub fn argmin(&self, dim: isize) -> Result<Tensor, TensorError> {
        self.dimension_op(dim, DimOp::Argmin)
    }

    pub fn matmul(&self, rhs: &Tensor) -> Result<Tensor, TensorError> {
        let new_shape = Tensor::validate_tensors(self, &rhs)?;

        let batch_dims = new_shape[0..new_shape.len() - 2].to_vec();

        let matmul_dim = new_shape[new_shape.len() - 2];

        let new_data = match new_shape.len() {
            1 => Err(TensorError::MatmulShapeError),
            2 => Tensor::two_dimension_matmul(&self, &rhs, matmul_dim),
            3 => self.three_dimension_matmul(&rhs, matmul_dim, &batch_dims),
            4 => self.four_dimension_matmul(&rhs, matmul_dim, &batch_dims),
            _ => Err(TensorError::MaxDimsError),
        }?;

        Tensor::new(new_data, &new_shape)
    }

    pub fn concat(&self, rhs: &Tensor, dim: isize) -> Result<Tensor, TensorError> {
        let concat_dim = if dim == -1 {
            self.shape.len() - 1 as usize
        } else if (dim >= 0) && (dim < self.shape.len() as isize) {
            dim as usize
        } else {
            return Err(TensorError::ShapeError);
        };

        if self.shape.len() != rhs.shape.len() {
            return Err(TensorError::ShapeError);
        }

        let mut new_shape: Vec<usize> = Vec::with_capacity(self.shape.len());
        for i in 0..self.shape.len() {
            if i != concat_dim {
                if self.shape[i] != rhs.shape[i] {
                    return Err(TensorError::ShapeError);
                }
                new_shape.push(self.shape[i]);
            } else {
                new_shape.push(self.shape[i] + rhs.shape[i]);
            }
        }

        let new_data = match self.shape.len() {
            1 => self.one_dimension_concat(&rhs, concat_dim, &new_shape),
            2 => self.two_dimension_concat(&rhs, concat_dim, &new_shape),
            3 => self.three_dimension_concat(&rhs, concat_dim, &new_shape),
            4 => self.four_dimension_concat(&rhs, concat_dim, &new_shape),
            _ => Err(TensorError::MaxDimsError),
        }?;

        Tensor::new(new_data, &new_shape)
    }

    pub fn transpose(&self) -> Result<Tensor, TensorError> {
        let mut transposed_shape = self.shape.clone();
        let mut transposed_strides = self.strides.clone();

        transposed_shape.reverse();
        transposed_strides.reverse();

        let new_data = match self.shape.len() {
            1 => self.one_dimension_transpose(&transposed_shape, &transposed_strides),
            2 => self.two_dimension_transpose(&transposed_shape, &transposed_strides),
            3 => self.three_dimension_transpose(&transposed_shape, &transposed_strides),
            4 => self.four_dimension_transpose(&transposed_shape, &transposed_strides),
            _ => Err(TensorError::MaxDimsError),
        }?;

        Tensor::new(new_data, &transposed_shape)
    }

    pub fn clone(&self) -> Result<Tensor, TensorError> {
        Tensor::new(self.data.clone(), &self.shape)
    }

    pub fn clone_no_grad(&self) -> Result<Tensor, TensorError> {
        Tensor::new_no_grad(self.data.clone(), &self.shape)
    }

    fn grad(&self) {
        match self.lhs_parent {
            Some(t) => {
                let d_lhs = match self.create_op {
                    Some(TensorOp::Add) => {
                        Ok(vec![1.0; Tensor::calc_tensor_len_from_shape(&self.shape)])
                    }
                    Some(TensorOp::Sub) => {
                        Ok(vec![1.0; Tensor::calc_tensor_len_from_shape(&self.shape)])
                    }
                    Some(TensorOp::Mul) => Ok(self.rhs_parent.unwrap().data.clone()),
                    Some(TensorOp::Div) => Ok(self.rhs_parent.unwrap().data.clone()), //not correct
                    None => Err(TensorError::ShapeError),
                }
                .unwrap();

                let d_lhs: Vec<f32> = d_lhs
                    .iter()
                    .zip(self.derivative.borrow().clone())
                    .map(|(a, b)| a * b)
                    .collect();

                let d_lhs_prev = t.derivative.borrow().clone();
                *t.derivative.borrow_mut() =
                    d_lhs.iter().zip(&d_lhs_prev).map(|(a, b)| a + b).collect();
            }
            None => (),
        }

        match self.rhs_parent {
            Some(t) => {
                let d_rhs = match self.create_op {
                    Some(TensorOp::Add) => {
                        Ok(vec![1.0; Tensor::calc_tensor_len_from_shape(&self.shape)])
                    }
                    Some(TensorOp::Sub) => {
                        Ok(vec![1.0; Tensor::calc_tensor_len_from_shape(&self.shape)])
                    }
                    Some(TensorOp::Mul) => Ok(self.lhs_parent.unwrap().data.clone()),
                    Some(TensorOp::Div) => Ok(self.lhs_parent.unwrap().data.clone()), //not correct
                    None => Err(TensorError::ShapeError),
                }
                .unwrap();

                let d_rhs: Vec<f32> = d_rhs
                    .iter()
                    .zip(self.derivative.borrow().clone())
                    .map(|(a, b)| a * b)
                    .collect();

                let d_rhs_prev = t.derivative.borrow().clone();
                *t.derivative.borrow_mut() =
                    d_rhs.iter().zip(&d_rhs_prev).map(|(a, b)| a + b).collect();
            }
            None => (),
        }
    }

    pub fn backward(&self) {
        // from https://github.com/evcu/numpy_autograd/blob/master/my_autograd.py#L147
        let mut seen: Vec<&Tensor> = Vec::new();
        let mut sorted: Vec<&Tensor> = Vec::new();

        fn topological_sort<'a>(
            vr: &'a Tensor,
            seen: &mut Vec<&Tensor<'a>>,
            sorted: &mut Vec<&Tensor<'a>>,
        ) {
            if seen.contains(&vr) || (vr.lhs_parent.is_none() && vr.rhs_parent.is_none()) {
                return;
            } else {
                seen.push(vr);

                if vr.lhs_parent.is_some() {
                    topological_sort(vr.lhs_parent.unwrap(), seen, sorted);
                }
                if vr.rhs_parent.is_some() {
                    topological_sort(vr.rhs_parent.unwrap(), seen, sorted);
                }

                sorted.push(vr);
            }
        }

        topological_sort(&self, &mut seen, &mut sorted);

        sorted.reverse();

        *sorted[0].derivative.borrow_mut() =
            vec![1.0; Tensor::calc_tensor_len_from_shape(&sorted[0].shape)];

        for t in sorted.iter() {
            t.grad()
        }
    }
}

impl<'a> Add for &'a Tensor<'a> {
    type Output = Tensor<'a>;

    fn add(self, other: &'a Tensor) -> Tensor<'a> {
        match self.tensor_op(other, TensorOp::Add) {
            Ok(t) => t,
            Err(e) => panic!("{}", e),
        }
    }
}

impl<'a> Sub for &'a Tensor<'a> {
    type Output = Tensor<'a>;

    fn sub(self, other: &'a Tensor) -> Tensor<'a> {
        match self.tensor_op(other, TensorOp::Sub) {
            Ok(t) => t,
            Err(e) => panic!("{}", e),
        }
    }
}

impl<'a> Mul for &'a Tensor<'a> {
    type Output = Tensor<'a>;

    fn mul(self, other: &'a Tensor) -> Tensor<'a> {
        match self.tensor_op(other, TensorOp::Mul) {
            Ok(t) => t,
            Err(e) => panic!("{}", e),
        }
    }
}

impl<'a> Div for &'a Tensor<'a> {
    type Output = Tensor<'a>;

    fn div(self, other: &'a Tensor) -> Tensor<'a> {
        match self.tensor_op(other, TensorOp::Div) {
            Ok(t) => t,
            Err(e) => panic!("{}", e),
        }
    }
}
