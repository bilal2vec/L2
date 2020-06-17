use std::fmt;

#[derive(Debug, PartialEq)]
pub enum TensorOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl fmt::Display for TensorOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorOp::Add => write!(f, "Add"),
            TensorOp::Sub => write!(f, "Subtract"),
            TensorOp::Mul => write!(f, "Multiply"),
            TensorOp::Div => write!(f, "Divide"),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum DimOp {
    Sum(isize),
    Mean(isize),
    Max(isize),
    Min(isize),
    Argmax(isize),
    Argmin(isize),
}

impl fmt::Display for DimOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DimOp::Sum(dim) => write!(f, "Sum: Dim {}", *dim),
            DimOp::Mean(dim) => write!(f, "Mean: Dim {}", *dim),
            DimOp::Max(dim) => write!(f, "Max: Dim {}", *dim),
            DimOp::Min(dim) => write!(f, "Min: Dim {}", *dim),
            DimOp::Argmax(dim) => write!(f, "Argmax: Dim {}", *dim),
            DimOp::Argmin(dim) => write!(f, "Argmin: Dim {}", *dim),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum Ops {
    TensorOp(TensorOp),
    DimOp(DimOp),
    Pow(f32),
    Sqrt,
    Exp,
    Log10,
    Log,
    Abs,
    Sin,
    Cos,
    Tan,
    Matmul,
    Slice(Vec<[usize; 2]>),
    Transpose,
    View,
    Concat((usize, usize)),
}

impl fmt::Display for Ops {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Ops::TensorOp(tensor_op) => write!(f, "{}", tensor_op),
            Ops::DimOp(dim_op) => write!(f, "{}", dim_op),
            Ops::Pow(pow) => write!(f, "Pow: {}", pow),
            Ops::Sqrt => write!(f, "Sqrt"),
            Ops::Exp => write!(f, "Exp"),
            Ops::Log10 => write!(f, "Log10"),
            Ops::Log => write!(f, "Log e"),
            Ops::Abs => write!(f, "Abs"),
            Ops::Sin => write!(f, "Sin"),
            Ops::Cos => write!(f, "Cos"),
            Ops::Tan => write!(f, "Tan"),
            Ops::Matmul => write!(f, "Matmul"),
            Ops::Slice(idxs) => write!(f, "Slice: {:?}", idxs),
            Ops::Transpose => write!(f, "Transpose"),
            Ops::View => write!(f, "View"),
            Ops::Concat((concat_dim, _concat_dim_shape)) => {
                write!(f, "Concat: Dim: {}", concat_dim)
            }
        }
    }
}
