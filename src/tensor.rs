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

    pub fn slice(&self, logical_indices: &[[isize; 2]]) -> Result<Self, TensorError> {
        let logical_indices = &self.process_indices(logical_indices)[..];

        match self.validate_logical_indices(logical_indices) {
            Ok(idxs) => {
                // converting to a slice b/c can't move `new_shape` to tensor and pass a reference to it to `Tensor::calc_strides_from_shape
                let new_shape: &[usize] = &Tensor::calc_shape_from_slice(idxs)[..];
                let slice_len = Tensor::calc_tensor_len_from_shape(&new_shape);

                let new_data = match idxs.len() {
                    1 => Ok(self.one_dimension_slice(idxs, slice_len)),
                    2 => Ok(self.two_dimension_slice(idxs, slice_len)),
                    3 => Ok(self.three_dimension_slice(idxs, slice_len)),
                    4 => Ok(self.four_dimension_slice(idxs, slice_len)),
                    _ => Err(TensorError::SliceError),
                };

                match new_data {
                    Ok(data) => Tensor::new(data, new_shape),
                    Err(e) => Err(e),
                }
            }
            Err(e) => Err(e),
        }
    }
}
