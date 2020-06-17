use std::error;
use std::fmt;

#[derive(Debug, Clone)]
pub enum TensorError {
    MaxDimsError,
    InvalidTensor,
    SliceError,
    ViewError,
    BroadcastError,
    OpError,
    DimError,
    MatmulShapeError,
    ShapeError,
    GradError,
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TensorError::MaxDimsError => write!(f, "L2 currently only supports tensors with up to 4 dimensions"),
            TensorError::InvalidTensor => write!(f, "Invalid parameters for Tensor"),
            TensorError::SliceError => write!(f, "Invalid slice for Tensor"),
            TensorError::ViewError => write!(f, "Invalid view shape for Tensor"),
            TensorError::BroadcastError => write!(f, "Shapes are not broadcastable"),
            TensorError::OpError => write!(f, "Tensors cannot be operated on"),
            TensorError::DimError => write!(f, "Tensors cannot be operated on over the given dimension"),
            TensorError::MatmulShapeError => write!(
                f,
                "Tensors must have at least two dimensions and have same shape in all dims except the last dimension"
            ),
            TensorError::ShapeError => write!(f, "Tensors must have the same shape in all dims except the last dimension"),
            TensorError::GradError => write!(f, "Error while computing .backward()"),
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
