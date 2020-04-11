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
    fn slice_tensor_1d() {
        let t = Tensor {
            data: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![4],
            strides: vec![1],
        };

        let x = t.slice(&[[0, 2]]);

        assert!((x.data == vec![1.0, 2.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }
    #[test]
    fn slice_tensor_2d_element() {
        let t = Tensor {
            data: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![2, 2],
            strides: vec![2, 1],
        };

        let x = t.slice(&[[0, 1], [0, 1]]);

        assert!((x.data == vec![1.0]) && (x.shape == vec![1, 1]) && (x.strides == vec![1, 1]))
    }

    #[test]
    fn slice_tensor_2d_row() {
        let t = Tensor {
            data: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![2, 2],
            strides: vec![2, 1],
        };

        let x = t.slice(&[[0, 1], [0, 2]]);

        assert!((x.data == vec![1.0, 2.0]) && (x.shape == vec![1, 2]) && (x.strides == vec![2, 1]))
    }

    #[test]
    fn slice_tensor_2d_col() {
        let t = Tensor {
            data: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![2, 2],
            strides: vec![2, 1],
        };

        let x = t.slice(&[[0, 2], [0, 1]]);

        assert!((x.data == vec![1.0, 3.0]) && (x.shape == vec![2, 1]) && (x.strides == vec![1, 1]))
    }

    #[test]
    fn slice_tensor_2d_all() {
        let t = Tensor {
            data: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![2, 2],
            strides: vec![2, 1],
        };

        let x = t.slice(&[[0, 2], [0, 2]]);

        assert!(
            (x.data == vec![1.0, 2.0, 3.0, 4.0])
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
    fn calc_tensor_len_from_shape(shape: &[usize]) -> usize {
        let mut length = 1;
        for i in shape {
            length *= i;
        }

        length
    }

    fn calc_strides_from_shape(shape: &[usize]) -> Vec<usize> {
        let mut strides = Vec::new();

        let mut current_stride = 1;
        for i in shape.iter().rev() {
            strides.insert(0, current_stride);
            current_stride *= i;
        }

        strides
    }

    fn calc_shape_from_slice(slice: &[[usize; 2]]) -> Vec<usize> {
        let mut slice_shape = Vec::new();

        for idx in slice {
            slice_shape.push(idx[1] - idx[0]);
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

    fn one_dimension_slice(&self, logical_indices: &[[usize; 2]]) -> Vec<f32> {
        let mut new_data = Vec::new();

        for i in logical_indices[0][0]..logical_indices[0][1] {
            new_data.push(self.data[self.get_physical_idx(&[i])]);
        }

        new_data
    }

    fn two_dimension_slice(&self, logical_indices: &[[usize; 2]]) -> Vec<f32> {
        let mut new_data = Vec::new();

        for i in logical_indices[0][0]..logical_indices[0][1] {
            for j in logical_indices[1][0]..logical_indices[1][1] {
                new_data.push(self.data[self.get_physical_idx(&[i, j])]);
            }
        }

        new_data
    }

    pub fn zeros(shape: &[usize]) -> Self {
        Tensor {
            data: vec![0.0; Tensor::calc_tensor_len_from_shape(shape)],
            shape: shape.to_vec(),
            strides: Tensor::calc_strides_from_shape(shape),
        }
    }

    pub fn slice(&self, logical_indices: &[[usize; 2]]) -> Tensor {
        let new_data = match logical_indices.len() {
            1 => self.one_dimension_slice(logical_indices),
            2 => self.two_dimension_slice(logical_indices),
            _ => panic!("Invalid slice"),
        };

        // converting to a slice b/c can't move `new_shape` to tensor and pass a reference to it to `Tensor::calc_strides_from_shape()`
        let new_shape: &[usize] = &Tensor::calc_shape_from_slice(logical_indices)[..];

        Tensor {
            data: new_data,
            shape: new_shape.to_vec(),
            strides: Tensor::calc_strides_from_shape(&new_shape),
        }
    }
}
