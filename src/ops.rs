#[derive(Debug, PartialEq)]
pub enum TensorOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, PartialEq)]
pub enum DimOp {
    Sum,
    Mean,
    Max,
    Min,
    Argmax,
    Argmin,
}

#[derive(Debug, PartialEq)]
pub enum Ops {
    TensorOp(TensorOp),
    DimOp(DimOp),
}
