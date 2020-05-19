pub mod errors;
pub mod tensor;

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

        let c = pow(&a, 2);

        assert!((c.data == vec![4.0, 9.0]) && (c.shape == vec![2]))
    }
}

pub fn add(lhs: &tensor::Tensor, rhs: &tensor::Tensor) -> tensor::Tensor {
    lhs + rhs
}

pub fn sub(lhs: &tensor::Tensor, rhs: &tensor::Tensor) -> tensor::Tensor {
    lhs - rhs
}

pub fn mul(lhs: &tensor::Tensor, rhs: &tensor::Tensor) -> tensor::Tensor {
    lhs * rhs
}

pub fn div(lhs: &tensor::Tensor, rhs: &tensor::Tensor) -> tensor::Tensor {
    lhs / rhs
}

pub fn pow(lhs: &tensor::Tensor, exp: usize) -> tensor::Tensor {
    lhs.pow(exp)
}

pub fn sqrt(lhs: &tensor::Tensor) -> tensor::Tensor {
    lhs.sqrt()
}

pub fn exp(lhs: &tensor::Tensor) -> tensor::Tensor {
    lhs.exp()
}

pub fn log10(lhs: &tensor::Tensor) -> tensor::Tensor {
    lhs.log10()
}

pub fn log(lhs: &tensor::Tensor) -> tensor::Tensor {
    lhs.log()
}

pub fn abs(lhs: &tensor::Tensor) -> tensor::Tensor {
    lhs.abs()
}

pub fn sin(lhs: &tensor::Tensor) -> tensor::Tensor {
    lhs.sin()
}

pub fn cos(lhs: &tensor::Tensor) -> tensor::Tensor {
    lhs.cos()
}

pub fn tan(lhs: &tensor::Tensor) -> tensor::Tensor {
    lhs.tan()
}
