#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_tensor_from_vec() {
        let _t = Tensor {
            data: vec![0.0; 16],
        };
    }

    #[test]
    fn allocate_tensor_zeros() {
        let _t = Tensor::zeros(16);
    }
}

pub struct Tensor {
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn zeros(len: usize) -> Self {
        Tensor {
            data: vec![0.0; len],
        }
    }
}
