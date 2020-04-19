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
        let _t = Tensor::zeros(&[2, 4]);
    }

    #[test]
    #[should_panic(expected = "Shape cannot be empty")]
    fn try_allocate_tensor_no_shape() {
        let _t = Tensor::zeros(&[]);
    }

    #[test]
    #[should_panic(expected = "We currently only support Tensors with up to 4 dimensions")]
    fn try_allocate_tensor_too_many_dims() {
        let _t = Tensor::zeros(&[2, 2, 2, 2, 2]);
    }
    #[test]
    #[should_panic(expected = "Cannot create a Tensor with a shape of zero for a dimension")]
    fn try_allocate_tensor_zero_shape() {
        let _t = Tensor::zeros(&[2, 0, 2]);
    }

    #[test]
    fn slice_tensor_1d_element() {
        let t = Tensor {
            data: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![4],
            strides: vec![1],
        };

        let x = t.slice(&[[0, 1]]);

        assert!((x.data == vec![1.0]) && (x.shape == vec![1]) && (x.strides == vec![1]))
    }
    #[test]
    fn slice_tensor_2d_element() {
        let t = Tensor {
            data: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![2, 2],
            strides: vec![2, 1],
        };

        let x = t.slice(&[[0, 1], [0, 1]]);

        assert!((x.data == vec![1.0]) && (x.shape == vec![1]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_2d_row() {
        let t = Tensor {
            data: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![2, 2],
            strides: vec![2, 1],
        };

        let x = t.slice(&[[0, 1], [0, 2]]);

        assert!((x.data == vec![1.0, 2.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_2d_col() {
        let t = Tensor {
            data: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![2, 2],
            strides: vec![2, 1],
        };

        let x = t.slice(&[[0, 2], [0, 1]]);

        assert!((x.data == vec![1.0, 3.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_3d_element() {
        let t = Tensor {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            shape: vec![2, 2, 2],
            strides: vec![4, 2, 1],
        };

        let x = t.slice(&[[0, 1], [0, 1], [0, 1]]);

        assert!((x.data == vec![1.0]) && (x.shape == vec![1]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_3d_row() {
        let t = Tensor {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            shape: vec![2, 2, 2],
            strides: vec![4, 2, 1],
        };

        let x = t.slice(&[[0, 1], [0, 1], [0, 2]]);

        assert!((x.data == vec![1.0, 2.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_3d_col() {
        let t = Tensor {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            shape: vec![2, 2, 2],
            strides: vec![4, 2, 1],
        };

        let x = t.slice(&[[0, 1], [0, 2], [0, 1]]);

        assert!((x.data == vec![1.0, 3.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_3d_channel() {
        let t = Tensor {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            shape: vec![2, 2, 2],
            strides: vec![4, 2, 1],
        };

        let x = t.slice(&[[0, 2], [0, 1], [0, 1]]);

        assert!((x.data == vec![1.0, 5.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_3d_chunk() {
        let t = Tensor {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            shape: vec![2, 2, 2],
            strides: vec![4, 2, 1],
        };

        let x = t.slice(&[[0, 2], [0, 2], [0, 1]]);

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

        let x = t.slice(&[[0, 1], [0, 1], [0, 1], [0, 1]]);

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

        let x = t.slice(&[[0, 1], [0, 1], [0, 1], [0, 2]]);

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

        let x = t.slice(&[[0, 1], [0, 1], [0, 2], [0, 1]]);

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

        let x = t.slice(&[[0, 1], [0, 2], [0, 1], [0, 1]]);

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

        let x = t.slice(&[[0, 2], [0, 1], [0, 1], [0, 1]]);

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

        let x = t.slice(&[[0, 2], [0, 2], [0, 1], [0, 1]]);

        assert!(
            (x.data == vec![1.0, 5.0, 9.0, 13.0])
                && (x.shape == vec![2, 2])
                && (x.strides == vec![2, 1])
        )
    }
}
#[derive(Debug)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl Tensor {
    fn validate_shape(shape: &[usize]) -> Result<&[usize], TensorError> {
        if shape.len() == 0 {
            Err(TensorError::EmptyShapeError())
        } else if shape.len() > 4 {
            Err(TensorError::TooManyDimensionsError())
        } else {
            match shape.iter().min() {
                //shape is a usize, so the compiler won't let us have a negative shape for a dimension
                Some(min) => match min {
                    min if min > &0 => Ok(shape),
                    _ => Err(TensorError::ZeroShapeError()),
                },
                None => Err(TensorError::EmptyShapeError()),
            }
        }
    }

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

    fn get_physical_idx(&self, logical_indices: &[usize]) -> usize {
        let mut physical_idx = 0;

        for (i, idx) in logical_indices.iter().enumerate() {
            physical_idx += idx * self.strides[i];
        }

        physical_idx
    }

    fn one_dimension_slice(&self, logical_indices: &[[usize; 2]], slice_len: usize) -> Vec<f32> {
        let mut new_data = Vec::with_capacity(slice_len);

        for i in logical_indices[0][0]..logical_indices[0][1] {
            new_data.push(self.data[self.get_physical_idx(&[i])]);
        }

        new_data
    }

    fn two_dimension_slice(&self, logical_indices: &[[usize; 2]], slice_len: usize) -> Vec<f32> {
        let mut new_data = Vec::with_capacity(slice_len);

        for i in logical_indices[0][0]..logical_indices[0][1] {
            for j in logical_indices[1][0]..logical_indices[1][1] {
                new_data.push(self.data[self.get_physical_idx(&[i, j])]);
            }
        }

        new_data
    }

    fn three_dimension_slice(&self, logical_indices: &[[usize; 2]], slice_len: usize) -> Vec<f32> {
        let mut new_data = Vec::with_capacity(slice_len);

        for i in logical_indices[0][0]..logical_indices[0][1] {
            for j in logical_indices[1][0]..logical_indices[1][1] {
                for k in logical_indices[2][0]..logical_indices[2][1] {
                    new_data.push(self.data[self.get_physical_idx(&[i, j, k])]);
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
                        new_data.push(self.data[self.get_physical_idx(&[i, j, k, m])]);
                    }
                }
            }
        }

        new_data
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let shape = Tensor::validate_shape(shape);

        match shape {
            Ok(s) => Tensor {
                data: vec![0.0; Tensor::calc_tensor_len_from_shape(s)],
                shape: s.to_vec(),
                strides: Tensor::calc_strides_from_shape(s),
            },
            Err(e) => panic!("{}", e),
        }
    }

    pub fn slice(&self, logical_indices: &[[usize; 2]]) -> Tensor {
        // converting to a slice b/c can't move `new_shape` to tensor and pass a reference to it to `Tensor::calc_strides_from_shape()`
        let new_shape: &[usize] = &Tensor::calc_shape_from_slice(logical_indices)[..];
        let slice_len = Tensor::calc_tensor_len_from_shape(&new_shape);

        let new_data = match logical_indices.len() {
            1 => self.one_dimension_slice(logical_indices, slice_len),
            2 => self.two_dimension_slice(logical_indices, slice_len),
            3 => self.three_dimension_slice(logical_indices, slice_len),
            4 => self.four_dimension_slice(logical_indices, slice_len),
            _ => panic!("Invalid slice"),
        };

        Tensor {
            data: new_data,
            shape: new_shape.to_vec(),
            strides: Tensor::calc_strides_from_shape(&new_shape),
        }
    }
}
