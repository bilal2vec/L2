use std::error;
use std::fmt;

#[derive(Debug, Clone)]
pub enum TensorError {
    EmptyShapeError(),
    TooManyDimensionsError(),
    ZeroShapeError(),

    SliceError(),
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TensorError::EmptyShapeError() => write!(f, "Shape cannot be empty"),
            TensorError::TooManyDimensionsError() => write!(
                f,
                "We currently only support Tensors with up to 4 dimensions"
            ),
            TensorError::ZeroShapeError() => write!(
                f,
                "Cannot create a Tensor with a shape of zero for a dimension"
            ),
            TensorError::SliceError() => write!(f, "Invalid slice for Tensor"),
        }
    }
}

// This is important for other errors to wrap this one.
impl error::Error for TensorError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        // Generic error, underlying cause isn't tracked.
        None
    }
}
