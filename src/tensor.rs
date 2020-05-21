use std::f32::consts::E;
use std::ops::{Add, Div, Mul, Sub};

use crate::errors::TensorError;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_tensor_from_vec() {
        let _t = Tensor {
            data: vec![0.0; 16],
            shape: vec![16],
            strides: vec![1],
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

        assert!((x.data == vec![1.0, 3.0]) && (x.shape == vec![2]));
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
        let t = Tensor {
            data: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![4],
            strides: vec![1],
        };

        let x = t.slice(&[[0, 1]]).unwrap();

        assert!((x.data == vec![1.0]) && (x.shape == vec![1]) && (x.strides == vec![1]))
    }
    #[test]
    fn slice_tensor_2d_element() {
        let t = Tensor {
            data: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![2, 2],
            strides: vec![2, 1],
        };

        let x = t.slice(&[[0, 1], [0, 1]]).unwrap();

        assert!((x.data == vec![1.0]) && (x.shape == vec![1]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_2d_row() {
        let t = Tensor {
            data: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![2, 2],
            strides: vec![2, 1],
        };

        let x = t.slice(&[[0, 1], [0, 2]]).unwrap();

        assert!((x.data == vec![1.0, 2.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_2d_col() {
        let t = Tensor {
            data: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![2, 2],
            strides: vec![2, 1],
        };

        let x = t.slice(&[[0, 2], [0, 1]]).unwrap();

        assert!((x.data == vec![1.0, 3.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_3d_element() {
        let t = Tensor {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            shape: vec![2, 2, 2],
            strides: vec![4, 2, 1],
        };

        let x = t.slice(&[[0, 1], [0, 1], [0, 1]]).unwrap();

        assert!((x.data == vec![1.0]) && (x.shape == vec![1]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_3d_row() {
        let t = Tensor {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            shape: vec![2, 2, 2],
            strides: vec![4, 2, 1],
        };

        let x = t.slice(&[[0, 1], [0, 1], [0, 2]]).unwrap();

        assert!((x.data == vec![1.0, 2.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_3d_col() {
        let t = Tensor {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            shape: vec![2, 2, 2],
            strides: vec![4, 2, 1],
        };

        let x = t.slice(&[[0, 1], [0, 2], [0, 1]]).unwrap();

        assert!((x.data == vec![1.0, 3.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_3d_channel() {
        let t = Tensor {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            shape: vec![2, 2, 2],
            strides: vec![4, 2, 1],
        };

        let x = t.slice(&[[0, 2], [0, 1], [0, 1]]).unwrap();

        assert!((x.data == vec![1.0, 5.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_3d_chunk() {
        let t = Tensor {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            shape: vec![2, 2, 2],
            strides: vec![4, 2, 1],
        };

        let x = t.slice(&[[0, 2], [0, 2], [0, 1]]).unwrap();

        assert!(
            (x.data == vec![1.0, 3.0, 5.0, 7.0])
                && (x.shape == vec![2, 2])
                && (x.strides == vec![2, 1])
        )
    }
    #[test]
    fn slice_tensor_4d_element() {
        let t = Tensor {
            data: vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            shape: vec![2, 2, 2, 2],
            strides: vec![8, 4, 2, 1],
        };

        let x = t.slice(&[[0, 1], [0, 1], [0, 1], [0, 1]]).unwrap();

        assert!((x.data == vec![1.0]) && (x.shape == vec![1]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_4d_row() {
        let t = Tensor {
            data: vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            shape: vec![2, 2, 2, 2],
            strides: vec![8, 4, 2, 1],
        };

        let x = t.slice(&[[0, 1], [0, 1], [0, 1], [0, 2]]).unwrap();

        assert!((x.data == vec![1.0, 2.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_4d_col() {
        let t = Tensor {
            data: vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            shape: vec![2, 2, 2, 2],
            strides: vec![8, 4, 2, 1],
        };

        let x = t.slice(&[[0, 1], [0, 1], [0, 2], [0, 1]]).unwrap();

        assert!((x.data == vec![1.0, 3.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_4d_channel() {
        let t = Tensor {
            data: vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            shape: vec![2, 2, 2, 2],
            strides: vec![8, 4, 2, 1],
        };

        let x = t.slice(&[[0, 1], [0, 2], [0, 1], [0, 1]]).unwrap();

        assert!((x.data == vec![1.0, 5.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_4d_batch() {
        let t = Tensor {
            data: vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            shape: vec![2, 2, 2, 2],
            strides: vec![8, 4, 2, 1],
        };

        let x = t.slice(&[[0, 2], [0, 1], [0, 1], [0, 1]]).unwrap();

        assert!((x.data == vec![1.0, 9.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_4d_chunk() {
        let t = Tensor {
            data: vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            shape: vec![2, 2, 2, 2],
            strides: vec![8, 4, 2, 1],
        };

        let x = t.slice(&[[0, 2], [0, 2], [0, 1], [0, 1]]).unwrap();

        assert!(
            (x.data == vec![1.0, 5.0, 9.0, 13.0])
                && (x.shape == vec![2, 2])
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

        let x = Tensor::new(vec![4.0, 6.0, 8.0, 10.0], &[4]).unwrap();

        assert_eq!(c, x);
    }

    #[test]
    fn elementwise_sub_op() {
        let a = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]).unwrap();
        let b = Tensor::new(vec![2.0, 3.0], &[2]).unwrap();

        let c = &a - &b;

        let x = Tensor::new(vec![0.0, 0.0, 2.0, 2.0], &[2, 2]).unwrap();

        assert_eq!(c, x);
    }

    #[test]
    fn elementwise_mul_op() {
        let a = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]).unwrap();
        let b = Tensor::new(vec![2.0, 3.0], &[2]).unwrap();

        let c = &a * &b;

        let x = Tensor::new(vec![4.0, 9.0, 8.0, 15.0], &[2, 2]).unwrap();

        assert_eq!(c, x);
    }

    #[test]
    fn elementwise_div_op() {
        let a = Tensor::new(vec![2.0, 4.0, 6.0, 8.0], &[2, 2]).unwrap();
        let b = Tensor::new(vec![2.0, 4.0], &[2]).unwrap();

        let c = &a / &b;

        let x = Tensor::new(vec![1.0, 1.0, 3.0, 2.0], &[2, 2]).unwrap();

        assert_eq!(c, x);
    }

    #[test]
    fn broadcast_shapes() {
        let a = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[1, 4]).unwrap();
        let b = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[4]).unwrap();

        let c = &a + &b;

        let x = Tensor::new(vec![4.0, 6.0, 8.0, 10.0], &[1, 4]).unwrap();

        assert!((x == c) && (c.shape == vec![1, 4]));
    }

    #[test]
    fn broadcast_dims() {
        let a = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]).unwrap();
        let b = Tensor::new(vec![2.0, 3.0], &[2]).unwrap();

        let c = &a + &b;

        let x = Tensor::new(vec![4.0, 6.0, 6.0, 8.0], &[2, 2]).unwrap();

        assert!((x == c) && (c.shape == vec![2, 2]));
    }

    #[test]
    fn broadcast_shapes_and_dims() {
        let a = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[1, 2, 2]).unwrap();
        let b = Tensor::new(vec![2.0, 3.0], &[2]).unwrap();

        let c = &a + &b;

        let x = Tensor::new(vec![4.0, 6.0, 6.0, 8.0], &[1, 2, 2]).unwrap();

        assert!((x == c) && (c.shape == vec![1, 2, 2]));
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
}

#[derive(Debug, PartialEq)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl Tensor {
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

    fn validate_logical_indices<'a>(
        &self,
        logical_indices: &'a [[usize; 2]],
    ) -> Result<&'a [[usize; 2]], TensorError> {
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
    fn op(lhs: &f32, rhs: &f32, op: &str) -> Result<f32, TensorError> {
        match op {
            "+" => Ok(lhs + rhs),
            "-" => Ok(lhs - rhs),
            "*" => Ok(lhs * rhs),
            "/" => Ok(lhs / rhs),
            _ => Err(TensorError::OpNotSupportedError),
        }
    }
    fn one_dimension_tensor_op(
        &self,
        other: &Tensor,
        lhs_strides: &[usize],
        rhs_strides: &[usize],
        new_shape: &Vec<usize>,
        op: &str,
    ) -> Vec<f32> {
        let mut new_data: Vec<f32> =
            Vec::with_capacity(Tensor::calc_tensor_len_from_shape(new_shape));

        for i in 0..new_shape[0] {
            let op_result = Tensor::op(
                &self.data[Tensor::get_physical_idx(&[i], &lhs_strides)],
                &other.data[Tensor::get_physical_idx(&[i], &rhs_strides)],
                op,
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
        op: &str,
    ) -> Vec<f32> {
        let mut new_data: Vec<f32> =
            Vec::with_capacity(Tensor::calc_tensor_len_from_shape(new_shape));

        for i in 0..new_shape[0] {
            for j in 0..new_shape[1] {
                let op_result = Tensor::op(
                    &self.data[Tensor::get_physical_idx(&[i, j], &lhs_strides)],
                    &other.data[Tensor::get_physical_idx(&[i, j], &rhs_strides)],
                    op,
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
        op: &str,
    ) -> Vec<f32> {
        let mut new_data: Vec<f32> =
            Vec::with_capacity(Tensor::calc_tensor_len_from_shape(new_shape));

        for i in 0..new_shape[0] {
            for j in 0..new_shape[1] {
                for k in 0..new_shape[2] {
                    let op_result = Tensor::op(
                        &self.data[Tensor::get_physical_idx(&[i, j, k], &lhs_strides)],
                        &other.data[Tensor::get_physical_idx(&[i, j, k], &rhs_strides)],
                        op,
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
        op: &str,
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
                            op,
                        )
                        .unwrap();

                        new_data.push(op_result);
                    }
                }
            }
        }

        new_data
    }
    fn tensor_op(&self, other: &Tensor, op: &str) -> Result<Tensor, TensorError> {
        let (new_shape, lhs_strides, rhs_strides) = Tensor::broadcast(&self.shape, &other.shape)?;

        let new_data = match new_shape.len() {
            1 => Ok(self.one_dimension_tensor_op(
                &other,
                &lhs_strides,
                &rhs_strides,
                &new_shape,
                op,
            )),
            2 => Ok(self.two_dimension_tensor_op(
                &other,
                &lhs_strides,
                &rhs_strides,
                &new_shape,
                op,
            )),
            3 => Ok(self.three_dimension_tensor_op(
                &other,
                &lhs_strides,
                &rhs_strides,
                &new_shape,
                op,
            )),
            4 => Ok(self.four_dimension_tensor_op(
                &other,
                &lhs_strides,
                &rhs_strides,
                &new_shape,
                op,
            )),
            _ => Err(TensorError::OpError),
        }?;

        Tensor::new(new_data, &new_shape)
    }

    pub fn new(data: Vec<f32>, shape: &[usize]) -> Result<Tensor, TensorError> {
        if data.len() == Tensor::calc_tensor_len_from_shape(shape)
            && shape.len() > 0
            && shape.len() < 5
        {
            Ok(Tensor {
                data,
                shape: shape.to_vec(),
                strides: Tensor::calc_strides_from_shape(shape),
            })
        } else {
            Err(TensorError::InvalidTensor)
        }
    }

    pub fn zeros(shape: &[usize]) -> Result<Self, TensorError> {
        Tensor::new(vec![0.0; Tensor::calc_tensor_len_from_shape(shape)], shape)
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
            _ => Err(TensorError::SliceError),
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

    // not returning result since low chance of any errors
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

    // pub fn sum(&self, dim: isize) -> Tensor {}
}

impl Add for &Tensor {
    type Output = Tensor;

    fn add(self, other: &Tensor) -> Tensor {
        match self.tensor_op(other, "+") {
            Ok(t) => t,
            Err(e) => panic!("{}", e),
        }
    }
}
impl Sub for &Tensor {
    type Output = Tensor;

    fn sub(self, other: &Tensor) -> Tensor {
        match self.tensor_op(other, "-") {
            Ok(t) => t,
            Err(e) => panic!("{}", e),
        }
    }
}
impl Mul for &Tensor {
    type Output = Tensor;

    fn mul(self, other: &Tensor) -> Tensor {
        match self.tensor_op(other, "*") {
            Ok(t) => t,
            Err(e) => panic!("{}", e),
        }
    }
}
impl Div for &Tensor {
    type Output = Tensor;

    fn div(self, other: &Tensor) -> Tensor {
        match self.tensor_op(other, "/") {
            Ok(t) => t,
            Err(e) => panic!("{}", e),
        }
    }
}
