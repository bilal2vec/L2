use std::ops::Index;

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
    fn index_single_value() {
        let t = Tensor::zeros(&[2, 4]);

        let x = t[&[0]];

        assert_eq!(x, 0.0);
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

    //fix
    fn get_physical_idx(logical_idx: &[usize]) -> usize {
        let mut physical_idx = 0;

        physical_idx += logical_idx[0];

        physical_idx
    }

    pub fn zeros(shape: &[usize]) -> Self {
        Tensor {
            data: vec![0.0; Tensor::calc_tensor_len_from_shape(shape)],
            shape: shape.to_vec(),
            strides: Tensor::calc_strides_from_shape(shape),
        }
    }
}

impl Index<&[usize]> for Tensor {
    type Output = f32;

    fn index(&self, indices: &[usize]) -> &Self::Output {
        &self.data[Tensor::get_physical_idx(indices)]
    }
}
