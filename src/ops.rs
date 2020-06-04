#[derive(Debug, PartialEq)]
pub enum TensorOp {
    Add,
    Sub,
    Mul,
    Div,
}

pub enum DimOp {
    Sum,
    Mean,
    Max,
    Min,
    Argmax,
    Argmin,
}
