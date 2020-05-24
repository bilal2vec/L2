pub mod errors;
mod ops;
pub mod tensor;

use errors::TensorError;
use tensor::Tensor;

#[cfg(test)]
mod tests {
    use super::tensor::*;
    use super::*;

    #[test]
    fn test_add() {
        let a = Tensor::new(vec![2.0, 3.0], &[2]).unwrap();
        let b = Tensor::new(vec![2.0, 3.0], &[2]).unwrap();

        let c = add(&a, &b);

        assert!((c.data == vec![4.0, 6.0]) && (c.shape == vec![2]))
    }

    #[test]
    fn test_subtract() {
        let a = Tensor::new(vec![2.0, 3.0], &[2]).unwrap();
        let b = Tensor::new(vec![2.0, 3.0], &[2]).unwrap();

        let c = sub(&a, &b);

        assert!((c.data == vec![0.0, 0.0]) && (c.shape == vec![2]))
    }
    #[test]
    fn test_mul() {
        let a = Tensor::new(vec![2.0, 3.0], &[2]).unwrap();
        let b = Tensor::new(vec![2.0, 3.0], &[2]).unwrap();

        let c = mul(&a, &b);

        assert!((c.data == vec![4.0, 9.0]) && (c.shape == vec![2]))
    }

    #[test]
    fn test_div() {
        let a = Tensor::new(vec![2.0, 3.0], &[2]).unwrap();
        let b = Tensor::new(vec![2.0, 3.0], &[2]).unwrap();

        let c = div(&a, &b);

        assert!((c.data == vec![1.0, 1.0]) && (c.shape == vec![2]))
    }

    #[test]
    fn test_pow() {
        let a = Tensor::new(vec![2.0, 3.0], &[2]).unwrap();

        let c = pow(&a, 2).unwrap();

        assert!((c.data == vec![4.0, 9.0]) && (c.shape == vec![2]))
    }

    #[test]
    fn test_sum() {
        let a = Tensor::new(vec![2.0, 3.0], &[2]).unwrap();

        let c = sum(&a, 0).unwrap();

        assert!((c.data == vec![5.0]) && (c.shape == vec![1]))
    }

    #[test]
    fn test_mean() {
        let a = Tensor::new(vec![2.0, 3.0], &[2]).unwrap();

        let c = mean(&a, 0).unwrap();

        assert!((c.data == vec![2.5]) && (c.shape == vec![1]))
    }
    #[test]
    fn test_max() {
        let a = Tensor::new(vec![2.0, 3.0], &[2]).unwrap();

        let c = max(&a, 0).unwrap();

        assert!((c.data == vec![3.0]) && (c.shape == vec![1]))
    }
    #[test]
    fn test_min() {
        let a = Tensor::new(vec![2.0, 3.0], &[2]).unwrap();

        let c = min(&a, 0).unwrap();

        assert!((c.data == vec![2.0]) && (c.shape == vec![1]))
    }

    #[test]
    fn test_argmax() {
        let a = Tensor::new(vec![2.0, 3.0], &[2]).unwrap();

        let c = argmax(&a, 0).unwrap();

        assert!((c.data == vec![1.0]) && (c.shape == vec![1]))
    }
    #[test]
    fn test_argmin() {
        let a = Tensor::new(vec![2.0, 3.0], &[2]).unwrap();

        let c = argmin(&a, 0).unwrap();

        assert!((c.data == vec![0.0]) && (c.shape == vec![1]))
    }
}

pub fn add(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    lhs + rhs
}

pub fn sub(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    lhs - rhs
}

pub fn mul(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    lhs * rhs
}

pub fn div(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    lhs / rhs
}

pub fn pow(lhs: &Tensor, exp: usize) -> Result<Tensor, TensorError> {
    lhs.pow(exp)
}

pub fn sqrt(lhs: &Tensor) -> Result<Tensor, TensorError> {
    lhs.sqrt()
}

pub fn exp(lhs: &Tensor) -> Result<Tensor, TensorError> {
    lhs.exp()
}

pub fn log10(lhs: &Tensor) -> Result<Tensor, TensorError> {
    lhs.log10()
}

pub fn log(lhs: &Tensor) -> Result<Tensor, TensorError> {
    lhs.log()
}

pub fn abs(lhs: &Tensor) -> Result<Tensor, TensorError> {
    lhs.abs()
}

pub fn sin(lhs: &Tensor) -> Result<Tensor, TensorError> {
    lhs.sin()
}

pub fn cos(lhs: &Tensor) -> Result<Tensor, TensorError> {
    lhs.cos()
}

pub fn tan(lhs: &Tensor) -> Result<Tensor, TensorError> {
    lhs.tan()
}

pub fn sum(lhs: &Tensor, dim: isize) -> Result<Tensor, TensorError> {
    lhs.sum(dim)
}

pub fn mean(lhs: &Tensor, dim: isize) -> Result<Tensor, TensorError> {
    lhs.mean(dim)
}

pub fn max(lhs: &Tensor, dim: isize) -> Result<Tensor, TensorError> {
    lhs.max(dim)
}

pub fn min(lhs: &Tensor, dim: isize) -> Result<Tensor, TensorError> {
    lhs.min(dim)
}

pub fn argmax(lhs: &Tensor, dim: isize) -> Result<Tensor, TensorError> {
    lhs.argmax(dim)
}

pub fn argmin(lhs: &Tensor, dim: isize) -> Result<Tensor, TensorError> {
    lhs.argmin(dim)
}
