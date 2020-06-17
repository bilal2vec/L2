#[derive(Debug, PartialEq)]
pub enum TensorOp {
    Add,
    Sub,
    Mul,
    Div,
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
