use std::error;
use std::fmt;

#[derive(Debug, Clone)]
pub enum TensorError {
    InvalidTensor,
    SliceError,
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TensorError::InvalidTensor => write!(f, "Invalid parameters for Tensor"),
            TensorError::SliceError => write!(f, "Invalid slice for Tensor"),
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
