use std::cell::RefCell;
use std::cmp::Ordering;
use std::f32::consts::E;
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

extern crate blas_src;
use blas::*;

use rand::distributions::{Distribution, Uniform};
use rand_distr::Normal;

use crate::errors::TensorError;
use crate::ops::{DimOp, Ops, TensorOp};

#[derive(Debug, PartialEq)]
pub struct Tensor<'a> {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,

    track_grad: bool,

    lhs_parent: Option<&'a Tensor<'a>>,
    rhs_parent: Option<&'a Tensor<'a>>,
    create_op: Option<Ops>,
    derivative: RefCell<Vec<f32>>,
}

impl<'a> fmt::Display for Tensor<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn recurse(tensor: &Tensor, level: usize) -> String {
            let indent = "  ".to_string().repeat(level);

            let lhs = match tensor.lhs_parent {
                Some(t) => recurse(t, level + 1),
                None => "None".to_string(),
            };

            let rhs = match tensor.rhs_parent {
                Some(t) => recurse(t, level + 1),
                None => "None".to_string(),
            };

            let op = match &tensor.create_op {
                Some(t) => format!("{}", t),
                None => "None".to_string(),
            };

            format!(
                "\n{}Value: {:?} \n{}Shape: {:?} \n{}Lhs: {} \n{}Rhs: {} \n{}Op: {} \n{}TrackGrad: {:?} \n{}Derivative: {:?}",
                indent,
                tensor.data,
                indent,
                tensor.shape,
                indent,
                lhs,
                indent,
                rhs,
                indent,
                op,
                indent,
                tensor.track_grad,
                indent,
                *(tensor.derivative.borrow())
            )
        }

        let graph = recurse(self, 0);

        write!(f, "{}", graph)
    }
}

impl<'a> Clone for Tensor<'a> {
    fn clone(&self) -> Self {
        Tensor::new(self.data.clone(), &self.shape).unwrap()
    }
}

impl<'a> Tensor<'a> {
    fn calc_tensor_len_from_shape(shape: &[usize]) -> usize {
        let mut length = 1;
        for i in shape {
            length *= i;
        }

        length
    }

    fn calc_strides_from_shape(shape: &[usize]) -> Vec<usize> {
        let mut strides = Vec::with_capacity(shape.len());

        let mut current_stride = 1;
        for i in shape.iter().rev() {
            strides.insert(0, current_stride);
            current_stride *= i;
        }

        strides
    }

    fn calc_shape_from_slice(slice: &[[usize; 2]]) -> Vec<usize> {
        let mut slice_shape = Vec::with_capacity(slice.len());

        for idx in slice {
            if idx[1] - idx[0] > 1 {
                slice_shape.push(idx[1] - idx[0]);
            }
        }

        if slice_shape.is_empty() {
            slice_shape.push(1);
        }

        slice_shape
    }

    fn get_physical_idx(logical_indices: &[usize], strides: &[usize]) -> usize {
        let mut physical_idx = 0;

        for (i, idx) in logical_indices.iter().enumerate() {
            physical_idx += idx * strides[i];
        }

        physical_idx
    }

    fn validate_logical_indices<'b>(
        &self,
        logical_indices: &'b [[usize; 2]],
    ) -> Result<&'b [[usize; 2]], TensorError> {
        if (logical_indices.len() != self.shape.len())
            || logical_indices.is_empty()
            || (logical_indices.len() > 4)
        {
            Err(TensorError::SliceError)
        } else if logical_indices.is_empty() || (logical_indices.len() > 4) {
            Err(TensorError::MaxDimsError)
        } else {
            for (i, logical_index) in logical_indices.iter().enumerate() {
                if logical_index[0] >= logical_index[1] || logical_index[1] > self.shape[i] {
                    return Err(TensorError::SliceError);
                }
            }
            Ok(logical_indices)
        }
    }

    fn process_indices(&self, indices: &[[isize; 2]]) -> Vec<[usize; 2]> {
        let mut indices = indices.to_vec();
        let mut processed_indices: Vec<[usize; 2]> = Vec::with_capacity(indices.len());

        let diff = self.shape.len() - indices.len();
        if diff > 0 {
            for _ in 0..diff {
                indices.push([0, -1]);
            }
        }

        for (i, idx) in indices.iter().enumerate() {
            let start: usize = idx[0] as usize;
            let stop: usize = if idx[1] == -1 {
                self.shape[i]
            } else {
                idx[1] as usize
            };

            processed_indices.push([start, stop]);
        }

        processed_indices
    }

    fn process_view(&self, shape: &[isize]) -> Result<Vec<usize>, TensorError> {
        match shape.iter().filter(|&&val| val == -1).count() {
            0 => Ok(shape.iter().map(|&val| val as usize).collect()),
            1 => {
                let tensor_len = Tensor::calc_tensor_len_from_shape(&self.shape);

                let mut remainder_len = 1;
                for val in shape.iter() {
                    if *val != -1 {
                        remainder_len *= *val as usize;
                    }
                }

                // can't return an error
                let neg_1_idx = shape.iter().position(|&val| val == -1).unwrap();

                let mut shape: Vec<usize> = shape.iter().map(|&val| val as usize).collect();
                shape[neg_1_idx] = tensor_len / remainder_len;

                Ok(shape)
            }
            _ => Err(TensorError::ViewError),
        }
    }

    #[allow(clippy::ptr_arg, clippy::type_complexity)]
    fn broadcast(
        lhs_shape: &Vec<usize>,
        rhs_shape: &Vec<usize>,
    ) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>), TensorError> {
        let lhs_shape = if lhs_shape.len() < rhs_shape.len() {
            let ones = vec![1; rhs_shape.len() - lhs_shape.len()];
            [&ones[..], &lhs_shape[..]].concat()
        } else {
            lhs_shape.clone()
        };

        let rhs_shape = if rhs_shape.len() < lhs_shape.len() {
            let ones = vec![1; lhs_shape.len() - rhs_shape.len()];
            [&ones[..], &rhs_shape[..]].concat()
        } else {
            rhs_shape.clone()
        };

        let mut broadcasted_shape: Vec<usize> = Vec::with_capacity(lhs_shape.len());
        let mut broadcasted_lhs_strides: Vec<usize> = Tensor::calc_strides_from_shape(&lhs_shape);
        let mut broadcasted_rhs_strides: Vec<usize> = Tensor::calc_strides_from_shape(&rhs_shape);

        for (i, (&lhs, &rhs)) in lhs_shape.iter().zip(rhs_shape.iter()).enumerate() {
            if lhs == rhs {
                broadcasted_shape.push(lhs);
            } else if lhs == 1 {
                broadcasted_shape.push(rhs);
                broadcasted_lhs_strides[i] = 0;
            } else if rhs == 1 {
                broadcasted_shape.push(lhs);
                broadcasted_rhs_strides[i] = 0;
            } else {
                return Err(TensorError::BroadcastError);
            }
        }

        Ok((
            broadcasted_shape,
            broadcasted_lhs_strides,
            broadcasted_rhs_strides,
        ))
    }

    fn op(lhs: &f32, rhs: &f32, op: &Ops) -> Result<f32, TensorError> {
        match op {
            Ops::TensorOp(TensorOp::Add) => Ok(lhs + rhs),
            Ops::TensorOp(TensorOp::Sub) => Ok(lhs - rhs),
            Ops::TensorOp(TensorOp::Mul) => Ok(lhs * rhs),
            Ops::TensorOp(TensorOp::Div) => Ok(lhs / rhs),
            _ => Err(TensorError::OpError),
        }
    }

    fn process_dims(
        idxs: &mut Vec<[isize; 2]>,
        dim: usize,
        current_dim: usize,
        current_idx: usize,
    ) {
        if dim == current_dim {
            idxs[current_dim] = [0, -1];
        } else {
            idxs[current_dim] = [current_idx as isize, current_idx as isize + 1];
        }
    }

    fn dim_op(lhs: &Tensor, op: &Ops) -> Result<f32, TensorError> {
        match op {
            Ops::DimOp(DimOp::Sum(_dim)) => Ok(lhs.data.iter().sum()),
            Ops::DimOp(DimOp::Mean(_dim)) => {
                let sum: f32 = lhs.data.iter().sum();
                Ok(sum / lhs.data.len() as f32)
            }
            Ops::DimOp(DimOp::Max(_dim)) => {
                let max = lhs
                    .data
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .map(|(_, val)| val);

                match max {
                    Some(x) => Ok(*x),
                    None => Err(TensorError::OpError),
                }
            }
            Ops::DimOp(DimOp::Min(_dim)) => {
                let min = lhs
                    .data
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .map(|(_, val)| val);

                match min {
                    Some(x) => Ok(*x),
                    None => Err(TensorError::OpError),
                }
            }
            Ops::DimOp(DimOp::Argmax(_dim)) => {
                let argmax = lhs
                    .data
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .map(|(i, _)| i);

                match argmax {
                    Some(x) => Ok(x as f32),
                    None => Err(TensorError::OpError),
                }
            }
            Ops::DimOp(DimOp::Argmin(_dim)) => {
                let argmin = lhs
                    .data
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .map(|(i, _)| i);

                match argmin {
                    Some(x) => Ok(x as f32),
                    None => Err(TensorError::OpError),
                }
            }
            _ => Err(TensorError::OpError),
        }
    }

    fn dimension_op(&self, dim: isize, op: Ops) -> Result<Tensor, TensorError> {
        let new_dim = if dim == -1 {
            self.shape.len() - 1 as usize
        } else if (dim >= 0) && (dim < self.shape.len() as isize) {
            dim as usize
        } else {
            return Err(TensorError::DimError);
        };

        let mut new_shape: Vec<usize> = Vec::new();
        for i in 0..self.shape.len() {
            if i != new_dim {
                new_shape.push(self.shape[i]);
            }
        }

        if new_shape.is_empty() {
            new_shape.push(1);
        }

        if self.shape.is_empty() || (self.shape.len() > 4) {
            return Err(TensorError::MaxDimsError);
        }

        let mut new_data: Vec<f32> =
            Vec::with_capacity(Tensor::calc_tensor_len_from_shape(&new_shape));

        let mut idxs: Vec<[isize; 2]> = vec![[0, 0]; self.shape.len()];

        let dim_zero_bounds = if new_dim == 0 { 1 } else { self.shape[0] };
        for i in 0..dim_zero_bounds {
            Tensor::process_dims(&mut idxs, new_dim, 0, i);

            if self.shape.len() == 1 {
                new_data.push(Tensor::dim_op(&self.slice(&idxs)?, &op)?);
            } else {
                let dim_one_bounds = if new_dim == 1 { 1 } else { self.shape[1] };
                for j in 0..dim_one_bounds {
                    Tensor::process_dims(&mut idxs, new_dim, 1, j);

                    if self.shape.len() == 2 {
                        new_data.push(Tensor::dim_op(&self.slice(&idxs)?, &op)?);
                    } else {
                        let dim_two_bounds = if new_dim == 2 { 1 } else { self.shape[2] };
                        for k in 0..dim_two_bounds {
                            Tensor::process_dims(&mut idxs, new_dim, 2, k);

                            if self.shape.len() == 3 {
                                new_data.push(Tensor::dim_op(&self.slice(&idxs)?, &op)?);
                            } else {
                                let dim_three_bounds = if new_dim == 3 { 1 } else { self.shape[3] };
                                for m in 0..dim_three_bounds {
                                    Tensor::process_dims(&mut idxs, new_dim, 3, m);

                                    new_data.push(Tensor::dim_op(&self.slice(&idxs)?, &op)?);
                                }
                            }
                        }
                    }
                }
            }
        }

        Tensor::new_with_parents(new_data, &new_shape, Some(&self), None, op)
    }

    fn tensor_op<'b>(&'b self, other: &'b Tensor, op: Ops) -> Result<Tensor<'b>, TensorError> {
        let (new_shape, lhs_strides, rhs_strides) = Tensor::broadcast(&self.shape, &other.shape)?;

        if new_shape.is_empty() || (new_shape.len() > 4) {
            return Err(TensorError::MaxDimsError);
        }

        let mut new_data: Vec<f32> =
            Vec::with_capacity(Tensor::calc_tensor_len_from_shape(&new_shape));

        for i in 0..new_shape[0] {
            if new_shape.len() == 1 {
                let op_result = Tensor::op(
                    &self.data[Tensor::get_physical_idx(&[i], &lhs_strides)],
                    &other.data[Tensor::get_physical_idx(&[i], &rhs_strides)],
                    &op,
                )?;

                new_data.push(op_result);
            } else {
                for j in 0..new_shape[1] {
                    if new_shape.len() == 2 {
                        let op_result = Tensor::op(
                            &self.data[Tensor::get_physical_idx(&[i, j], &lhs_strides)],
                            &other.data[Tensor::get_physical_idx(&[i, j], &rhs_strides)],
                            &op,
                        )?;

                        new_data.push(op_result);
                    } else {
                        for k in 0..new_shape[2] {
                            if new_shape.len() == 3 {
                                let op_result = Tensor::op(
                                    &self.data[Tensor::get_physical_idx(&[i, j, k], &lhs_strides)],
                                    &other.data[Tensor::get_physical_idx(&[i, j, k], &rhs_strides)],
                                    &op,
                                )?;

                                new_data.push(op_result);
                            } else {
                                for m in 0..new_shape[3] {
                                    let op_result = Tensor::op(
                                        &self.data
                                            [Tensor::get_physical_idx(&[i, j, k, m], &lhs_strides)],
                                        &other.data
                                            [Tensor::get_physical_idx(&[i, j, k, m], &rhs_strides)],
                                        &op,
                                    )?;

                                    new_data.push(op_result);
                                }
                            }
                        }
                    }
                }
            }
        }

        if self.track_grad && other.track_grad {
            Tensor::new_with_parents(new_data, &new_shape, Some(self), Some(other), op)
        } else {
            Tensor::new_no_grad(new_data, &new_shape)
        }
    }

    fn validate_tensors(lhs: &Tensor, rhs: &Tensor) -> Result<Vec<usize>, TensorError> {
        let lhs_shape = lhs.shape.clone();
        let rhs_shape = rhs.shape.clone();

        if (lhs_shape.len() == rhs_shape.len())
            && (lhs_shape[lhs_shape.len() - 1] == rhs_shape[rhs_shape.len() - 2])
        {
            let mut new_shape = lhs_shape.clone();
            new_shape[lhs_shape.len() - 1] = rhs_shape[rhs_shape.len() - 1];

            Ok(new_shape)
        } else {
            Err(TensorError::MatmulShapeError)
        }
    }

    // lots of transposing since blas expects data in col-major order; can be cleaned up a lot
    #[allow(clippy::many_single_char_names)]
    fn two_dimension_matmul(lhs: &Tensor, rhs: &Tensor, out: &mut Vec<f32>) {
        let lhs = lhs.transpose().unwrap();
        let rhs = rhs.transpose().unwrap();

        let a: Vec<f64> = lhs.data.iter().map(|val| *val as f64).collect();
        let b: Vec<f64> = rhs.data.iter().map(|val| *val as f64).collect();

        let mut c: Vec<f64> =
            vec![0.0; Tensor::calc_tensor_len_from_shape(&[lhs.shape[1], rhs.shape[0]])];

        let (m, n, k) = (
            lhs.shape[1] as i32,
            rhs.shape[0] as i32,
            lhs.shape[0] as i32,
        );

        unsafe {
            dgemm(b'N', b'N', m, n, k, 1.0, &a, m, &b, k, 1.0, &mut c, m);
        }

        let c = c.iter().map(|val| *val as f32).collect();
        let c = Tensor::new(c, &[rhs.shape[0], lhs.shape[1]]).unwrap();
        let c = c.transpose().unwrap();

        let mut c = c.data;

        out.append(&mut c);
    }

    fn new_with_parents<'b>(
        data: Vec<f32>,
        shape: &[usize],
        lhs_parent: Option<&'b Tensor>,
        rhs_parent: Option<&'b Tensor>,
        op: Ops,
    ) -> Result<Tensor<'b>, TensorError> {
        if data.len() == Tensor::calc_tensor_len_from_shape(shape)
            && !shape.is_empty()
            && shape.len() < 5
        {
            Ok(Tensor {
                data,
                shape: shape.to_vec(),
                strides: Tensor::calc_strides_from_shape(shape),
                track_grad: true,
                lhs_parent,
                rhs_parent,
                create_op: Some(op),
                derivative: RefCell::new(vec![0.0; Tensor::calc_tensor_len_from_shape(shape)]),
            })
        } else {
            Err(TensorError::InvalidTensor)
        }
    }

    pub fn new<'b>(data: Vec<f32>, shape: &[usize]) -> Result<Tensor<'b>, TensorError> {
        if data.len() == Tensor::calc_tensor_len_from_shape(shape)
            && !shape.is_empty()
            && shape.len() < 5
        {
            Ok(Tensor {
                data,
                shape: shape.to_vec(),
                strides: Tensor::calc_strides_from_shape(shape),
                track_grad: true,
                lhs_parent: None,
                rhs_parent: None,
                create_op: None,
                derivative: RefCell::new(vec![0.0; Tensor::calc_tensor_len_from_shape(shape)]),
            })
        } else {
            Err(TensorError::InvalidTensor)
        }
    }

    pub fn new_no_grad<'b>(data: Vec<f32>, shape: &[usize]) -> Result<Tensor<'b>, TensorError> {
        if data.len() == Tensor::calc_tensor_len_from_shape(shape)
            && !shape.is_empty()
            && shape.len() < 5
        {
            Ok(Tensor {
                data,
                shape: shape.to_vec(),
                strides: Tensor::calc_strides_from_shape(shape),
                track_grad: false,
                lhs_parent: None,
                rhs_parent: None,
                create_op: None,
                derivative: RefCell::new(vec![0.0; Tensor::calc_tensor_len_from_shape(shape)]),
            })
        } else {
            Err(TensorError::InvalidTensor)
        }
    }
    pub fn zeros<'b>(shape: &[usize]) -> Result<Tensor<'b>, TensorError> {
        Tensor::new(vec![0.0; Tensor::calc_tensor_len_from_shape(shape)], shape)
    }

    pub fn normal<'b>(shape: &[usize], mean: f32, std: f32) -> Result<Tensor<'b>, TensorError> {
        let normal = Normal::new(mean, std).unwrap();
        let mut rng = rand::thread_rng();

        let tensor_len = Tensor::calc_tensor_len_from_shape(shape);
        let mut new_data: Vec<f32> = Vec::with_capacity(tensor_len);

        for _ in 0..tensor_len {
            new_data.push(normal.sample(&mut rng));
        }

        Tensor::new(new_data, shape)
    }

    pub fn uniform<'b>(shape: &[usize], low: f32, high: f32) -> Result<Tensor<'b>, TensorError> {
        let uniform = Uniform::from(low..high);
        let mut rng = rand::thread_rng();

        let tensor_len = Tensor::calc_tensor_len_from_shape(shape);
        let mut new_data: Vec<f32> = Vec::with_capacity(tensor_len);

        for _ in 0..tensor_len {
            new_data.push(uniform.sample(&mut rng));
        }

        Tensor::new(new_data, shape)
    }

    pub fn slice(&'a self, logical_indices: &[[isize; 2]]) -> Result<Self, TensorError> {
        let logical_indices = self.process_indices(logical_indices);

        let logical_indices = self.validate_logical_indices(&logical_indices)?;

        // converting to a slice b/c can't move `new_shape` to tensor and pass a reference to it to `Tensor::calc_strides_from_shape
        let new_shape = Tensor::calc_shape_from_slice(logical_indices);
        let slice_len = Tensor::calc_tensor_len_from_shape(&new_shape);

        let mut new_data = Vec::with_capacity(slice_len);

        for i in logical_indices[0][0]..logical_indices[0][1] {
            if logical_indices.len() == 1 {
                new_data.push(self.data[Tensor::get_physical_idx(&[i], &self.strides)]);
            } else {
                for j in logical_indices[1][0]..logical_indices[1][1] {
                    if logical_indices.len() == 2 {
                        new_data.push(self.data[Tensor::get_physical_idx(&[i, j], &self.strides)]);
                    } else {
                        for k in logical_indices[2][0]..logical_indices[2][1] {
                            if logical_indices.len() == 3 {
                                new_data.push(
                                    self.data[Tensor::get_physical_idx(&[i, j, k], &self.strides)],
                                );
                            } else {
                                for m in logical_indices[3][0]..logical_indices[3][1] {
                                    new_data.push(
                                        self.data[Tensor::get_physical_idx(
                                            &[i, j, k, m],
                                            &self.strides,
                                        )],
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        Tensor::new_with_parents(
            new_data,
            &new_shape,
            Some(&self),
            None,
            Ops::Slice(logical_indices.to_vec()),
        )
    }

    pub fn view(&'a self, shape: &[isize]) -> Result<Self, TensorError> {
        let shape = self.process_view(shape)?;

        match Tensor::calc_tensor_len_from_shape(&self.shape)
            == Tensor::calc_tensor_len_from_shape(&shape)
        {
            true => {
                Tensor::new_with_parents(self.data.clone(), &shape, Some(&self), None, Ops::View)
            }
            false => Err(TensorError::ViewError),
        }
    }

    pub fn pow(&self, exp: f32) -> Result<Tensor, TensorError> {
        let new_data = self.data.iter().map(|val| val.powf(exp)).collect();

        Tensor::new_with_parents(new_data, &self.shape, Some(&self), None, Ops::Pow(exp))
    }

    pub fn sqrt(&self) -> Result<Tensor, TensorError> {
        let new_data = self.data.iter().map(|val| val.sqrt()).collect();

        Tensor::new_with_parents(new_data, &self.shape, Some(&self), None, Ops::Sqrt)
    }

    pub fn exp(&self) -> Result<Tensor, TensorError> {
        let new_data = self.data.iter().map(|val| val.exp()).collect();

        Tensor::new_with_parents(new_data, &self.shape, Some(&self), None, Ops::Exp)
    }

    pub fn log10(&self) -> Result<Tensor, TensorError> {
        let new_data = self.data.iter().map(|val| val.log10()).collect();

        Tensor::new_with_parents(new_data, &self.shape, Some(&self), None, Ops::Log10)
    }

    pub fn log(&self) -> Result<Tensor, TensorError> {
        let new_data = self.data.iter().map(|val| val.log(E)).collect();

        Tensor::new_with_parents(new_data, &self.shape, Some(&self), None, Ops::Log)
    }

    pub fn abs(&self) -> Result<Tensor, TensorError> {
        let new_data = self.data.iter().map(|val| val.abs()).collect();

        Tensor::new_with_parents(new_data, &self.shape, Some(&self), None, Ops::Abs)
    }

    pub fn sin(&self) -> Result<Tensor, TensorError> {
        let new_data = self.data.iter().map(|val| val.sin()).collect();

        Tensor::new_with_parents(new_data, &self.shape, Some(&self), None, Ops::Sin)
    }

    pub fn cos(&self) -> Result<Tensor, TensorError> {
        let new_data = self.data.iter().map(|val| val.cos()).collect();

        Tensor::new_with_parents(new_data, &self.shape, Some(&self), None, Ops::Cos)
    }

    pub fn tan(&self) -> Result<Tensor, TensorError> {
        let new_data = self.data.iter().map(|val| val.tan()).collect();

        Tensor::new_with_parents(new_data, &self.shape, Some(&self), None, Ops::Tan)
    }

    pub fn sum(&self, dim: isize) -> Result<Tensor, TensorError> {
        self.dimension_op(dim, Ops::DimOp(DimOp::Sum(dim)))
    }

    pub fn mean(&self, dim: isize) -> Result<Tensor, TensorError> {
        self.dimension_op(dim, Ops::DimOp(DimOp::Mean(dim)))
    }

    pub fn max(&self, dim: isize) -> Result<Tensor, TensorError> {
        self.dimension_op(dim, Ops::DimOp(DimOp::Max(dim)))
    }

    pub fn min(&self, dim: isize) -> Result<Tensor, TensorError> {
        self.dimension_op(dim, Ops::DimOp(DimOp::Min(dim)))
    }

    pub fn argmax(&self, dim: isize) -> Result<Tensor, TensorError> {
        self.dimension_op(dim, Ops::DimOp(DimOp::Argmax(dim)))
    }

    pub fn argmin(&self, dim: isize) -> Result<Tensor, TensorError> {
        self.dimension_op(dim, Ops::DimOp(DimOp::Argmin(dim)))
    }

    pub fn matmul(&'a self, rhs: &'a Tensor) -> Result<Tensor, TensorError> {
        let new_shape = Tensor::validate_tensors(self, &rhs)?;

        let batch_dims = new_shape[0..new_shape.len() - 2].to_vec();

        if (new_shape.len() <= 1) || (new_shape.len() > 4) {
            return Err(TensorError::MaxDimsError);
        }

        let mut new_data = Vec::with_capacity(Tensor::calc_tensor_len_from_shape(&new_shape));

        if new_shape.len() == 2 {
            Tensor::two_dimension_matmul(&self, rhs, &mut new_data)
        } else {
            let mut idxs: Vec<[isize; 2]> = vec![[0, -1]; new_shape.len()];

            let lhs_shape: Vec<isize> = self.shape[self.shape.len() - 2..]
                .to_vec()
                .iter()
                .map(|&val| val as isize)
                .collect();
            let rhs_shape: Vec<isize> = rhs.shape[rhs.shape.len() - 2..]
                .to_vec()
                .iter()
                .map(|&val| val as isize)
                .collect();

            for i in 0..batch_dims[0] {
                idxs[0] = [i as isize, i as isize + 1];

                if new_shape.len() == 4 {
                    for j in 0..batch_dims[1] {
                        idxs[1] = [j as isize, j as isize + 1];

                        Tensor::two_dimension_matmul(
                            &self.slice(&idxs)?.view(&lhs_shape)?,
                            &rhs.slice(&idxs)?.view(&rhs_shape)?,
                            &mut new_data,
                        );
                    }
                } else {
                    Tensor::two_dimension_matmul(
                        &self.slice(&idxs)?.view(&lhs_shape)?,
                        &rhs.slice(&idxs)?.view(&rhs_shape)?,
                        &mut new_data,
                    );
                }
            }
        }

        Tensor::new_with_parents(new_data, &new_shape, Some(&self), Some(&rhs), Ops::Matmul)
    }

    pub fn concat(&self, rhs: &'a Tensor, dim: isize) -> Result<Tensor, TensorError> {
        let concat_dim = if dim == -1 {
            self.shape.len() - 1 as usize
        } else if (dim >= 0) && (dim < self.shape.len() as isize) {
            dim as usize
        } else {
            return Err(TensorError::ShapeError);
        };

        if self.shape.len() != rhs.shape.len() {
            return Err(TensorError::ShapeError);
        }

        let mut new_shape: Vec<usize> = Vec::with_capacity(self.shape.len());
        for i in 0..self.shape.len() {
            if i != concat_dim {
                if self.shape[i] != rhs.shape[i] {
                    return Err(TensorError::ShapeError);
                }
                new_shape.push(self.shape[i]);
            } else {
                new_shape.push(self.shape[i] + rhs.shape[i]);
            }
        }

        if new_shape.is_empty() || (new_shape.len() > 4) {
            return Err(TensorError::MaxDimsError);
        }

        let mut new_data: Vec<f32> =
            Vec::with_capacity(Tensor::calc_tensor_len_from_shape(&new_shape));

        let mut idxs: Vec<[isize; 2]> = vec![[0, 0]; self.shape.len()];

        let dim_zero_bounds = if concat_dim == 0 { 1 } else { self.shape[0] };
        for i in 0..dim_zero_bounds {
            Tensor::process_dims(&mut idxs, concat_dim, 0, i);

            if self.shape.len() == 1 {
                new_data.extend(self.slice(&idxs)?.data);
                new_data.extend(rhs.slice(&idxs)?.data);
            } else {
                let dim_one_bounds = if concat_dim == 1 { 1 } else { self.shape[1] };
                for j in 0..dim_one_bounds {
                    Tensor::process_dims(&mut idxs, concat_dim, 1, j);

                    if self.shape.len() == 2 {
                        new_data.extend(self.slice(&idxs)?.data);
                        new_data.extend(rhs.slice(&idxs)?.data);
                    } else {
                        let dim_two_bounds = if concat_dim == 2 { 1 } else { self.shape[2] };
                        for k in 0..dim_two_bounds {
                            Tensor::process_dims(&mut idxs, concat_dim, 2, k);

                            if self.shape.len() == 3 {
                                new_data.extend(self.slice(&idxs)?.data);
                                new_data.extend(rhs.slice(&idxs)?.data);
                            } else {
                                let dim_three_bounds =
                                    if concat_dim == 3 { 1 } else { self.shape[3] };
                                for m in 0..dim_three_bounds {
                                    Tensor::process_dims(&mut idxs, concat_dim, 3, m);

                                    new_data.extend(self.slice(&idxs)?.data);
                                    new_data.extend(rhs.slice(&idxs)?.data);
                                }
                            }
                        }
                    }
                }
            }
        }

        Tensor::new_with_parents(
            new_data,
            &new_shape,
            Some(&self),
            Some(rhs),
            Ops::Concat((concat_dim, self.shape[concat_dim])),
        )
    }

    pub fn transpose(&self) -> Result<Tensor, TensorError> {
        let mut transposed_shape = self.shape.clone();
        let mut transposed_strides = self.strides.clone();

        transposed_shape.reverse();
        transposed_strides.reverse();

        let mut new_data: Vec<f32> =
            Vec::with_capacity(Tensor::calc_tensor_len_from_shape(&transposed_shape));

        for i in 0..transposed_shape[0] {
            if transposed_shape.len() == 1 {
                new_data.push(self.data[Tensor::get_physical_idx(&[i], &transposed_strides)]);
            } else {
                for j in 0..transposed_shape[1] {
                    if transposed_shape.len() == 2 {
                        new_data.push(
                            self.data[Tensor::get_physical_idx(&[i, j], &transposed_strides)],
                        );
                    } else {
                        for k in 0..transposed_shape[2] {
                            if transposed_shape.len() == 3 {
                                new_data.push(
                                    self.data
                                        [Tensor::get_physical_idx(&[i, j, k], &transposed_strides)],
                                );
                            } else {
                                for m in 0..transposed_shape[3] {
                                    new_data.push(
                                        self.data[Tensor::get_physical_idx(
                                            &[i, j, k, m],
                                            &transposed_strides,
                                        )],
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        Tensor::new_with_parents(
            new_data,
            &transposed_shape,
            Some(&self),
            None,
            Ops::Transpose,
        )
    }

    pub fn clone_no_grad(&self) -> Result<Tensor, TensorError> {
        Tensor::new_no_grad(self.data.clone(), &self.shape)
    }

    fn grad(&self) {
        let d = Tensor::new(self.derivative.borrow().clone(), &self.shape).unwrap();

        if let Some(t) = self.lhs_parent {
            let d_lhs = match &self.create_op {
                Some(Ops::TensorOp(TensorOp::Add)) => Tensor::new(
                    vec![1.0; Tensor::calc_tensor_len_from_shape(&self.shape)],
                    &self.shape,
                ),
                Some(Ops::TensorOp(TensorOp::Sub)) => Tensor::new(
                    vec![1.0; Tensor::calc_tensor_len_from_shape(&self.shape)],
                    &self.shape,
                ),
                Some(Ops::TensorOp(TensorOp::Mul)) => Ok(self.rhs_parent.unwrap().clone()),
                Some(Ops::TensorOp(TensorOp::Div)) => {
                    let temp = self.rhs_parent.unwrap().pow(-1.0).unwrap();

                    Tensor::new(temp.data.clone(), &temp.shape)
                }
                Some(Ops::Pow(exp)) => {
                    let n = Tensor::new(vec![*exp as f32], &[1]).unwrap();

                    let temp = t.pow(exp - 1.0).unwrap();
                    let temp = &n * &temp;

                    Tensor::new(temp.data, &temp.shape)
                }
                Some(Ops::Sqrt) => {
                    let one_half = Tensor::new(vec![0.5], &[1]).unwrap();

                    let temp = t.pow(-0.5).unwrap();
                    let temp = &one_half * &temp;

                    Tensor::new(temp.data, &temp.shape)
                }
                Some(Ops::Exp) => Ok(self.clone()),
                Some(Ops::Log10) => {
                    let one = Tensor::new(vec![1.0], &[1]).unwrap();
                    let lna = Tensor::new(vec![(10.0 as f32).log(E)], &[1]).unwrap();

                    let temp = t * &lna;
                    let temp = &one / &temp;

                    Tensor::new(temp.data, &temp.shape)
                }
                Some(Ops::Log) => {
                    let one = Tensor::new(vec![1.0], &[1]).unwrap();

                    let temp = &one / t;

                    Tensor::new(temp.data, &temp.shape)
                }
                Some(Ops::Abs) => {
                    let temp = t
                        .data
                        .iter()
                        .map(|val| {
                            if *val > 0.0 {
                                1.0
                            } else if *val < 0.0 {
                                -1.0
                            } else {
                                0.0
                            }
                        })
                        .collect();

                    Tensor::new(temp, &t.shape)
                }
                Some(Ops::Sin) => t.cos(),
                Some(Ops::Cos) => {
                    let neg1 = Tensor::new(vec![-1.0], &[1]).unwrap();

                    let temp = t.sin().unwrap();
                    let temp = &neg1 * &temp;

                    Tensor::new(temp.data, &temp.shape)
                }
                Some(Ops::Tan) => {
                    let one = Tensor::new(vec![1.0], &[1]).unwrap();

                    let temp = t.cos().unwrap();
                    let temp = temp.pow(2.0).unwrap();
                    let temp = &one / &temp;

                    Tensor::new(temp.data, &temp.shape)
                }
                Some(Ops::Matmul) => self.rhs_parent.unwrap().transpose(),
                Some(Ops::Slice(_idxs)) => Tensor::zeros(&[1]), // this is never used
                Some(Ops::Transpose) => Tensor::zeros(&[1]),    // this is never used
                Some(Ops::View) => Tensor::zeros(&[1]),         // this is never used
                Some(Ops::Concat((_concat_dim, _concat_dim_size))) => Tensor::zeros(&[1]), // this is never used
                Some(Ops::DimOp(DimOp::Sum(_dim))) => Tensor::new(vec![1.0], &[1]),
                Some(Ops::DimOp(DimOp::Mean(dim))) => {
                    let dim = if *dim == -1 {
                        t.shape.len() - 1
                    } else {
                        *dim as usize
                    };

                    Tensor::new(vec![1.0 / t.shape[dim] as f32], &[1])
                }
                _ => Err(TensorError::GradError),
            }
            .unwrap();

            let d_lhs = match &self.create_op {
                Some(Ops::Matmul) => d.matmul(&d_lhs).unwrap(),
                Some(Ops::Slice(idxs)) => {
                    let mut t_grad = vec![0.0; Tensor::calc_tensor_len_from_shape(&t.shape)];

                    let mut d_grad = d.data.iter();

                    for i in idxs[0][0]..idxs[0][1] {
                        if idxs.len() == 1 {
                            t_grad[Tensor::get_physical_idx(&[i], &t.strides)] =
                                *(d_grad.next().unwrap());
                        } else {
                            for j in idxs[1][0]..idxs[1][1] {
                                if idxs.len() == 2 {
                                    t_grad[Tensor::get_physical_idx(&[i, j], &t.strides)] =
                                        *(d_grad.next().unwrap());
                                } else {
                                    for k in idxs[2][0]..idxs[2][1] {
                                        if idxs.len() == 3 {
                                            t_grad[Tensor::get_physical_idx(
                                                &[i, j, k],
                                                &t.strides,
                                            )] = *(d_grad.next().unwrap());
                                        } else {
                                            for m in idxs[3][0]..idxs[3][1] {
                                                t_grad[Tensor::get_physical_idx(
                                                    &[i, j, k, m],
                                                    &t.strides,
                                                )] = *(d_grad.next().unwrap());
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    Tensor::new(t_grad, &t.shape).unwrap()
                }
                Some(Ops::Transpose) => d.transpose().unwrap(),
                Some(Ops::View) => {
                    let new_shape: Vec<isize> = t.shape.iter().map(|val| *val as isize).collect();

                    d.view(&new_shape).unwrap()
                }
                Some(Ops::Concat((concat_dim, concat_dim_size))) => {
                    let mut idxs = vec![[0, -1]; d.shape.len()];
                    idxs[*concat_dim] = [0, *concat_dim_size as isize];

                    d.slice(&idxs).unwrap()
                }
                Some(Ops::DimOp(DimOp::Sum(dim))) => {
                    let mut new_shape: Vec<isize> =
                        d.shape.clone().iter().map(|val| *val as isize).collect();

                    let temp;
                    if *dim == -1 {
                        temp = new_shape.len() as isize;
                    } else {
                        temp = *dim;
                    }

                    new_shape.insert(temp as usize, 1);

                    let temp = &d_lhs * &d;
                    let temp = temp.view(&new_shape).unwrap();

                    Tensor::new(temp.data, &temp.shape).unwrap()
                }
                Some(Ops::DimOp(DimOp::Mean(dim))) => {
                    let mut new_shape: Vec<isize> =
                        d.shape.clone().iter().map(|val| *val as isize).collect();

                    let temp;
                    if *dim == -1 {
                        temp = new_shape.len() as isize;
                    } else {
                        temp = *dim;
                    }

                    new_shape.insert(temp as usize, 1);

                    let temp = &d_lhs * &d;
                    let temp = temp.view(&new_shape).unwrap();

                    Tensor::new(temp.data, &temp.shape).unwrap()
                }
                _ => &d_lhs * &d,
            };

            let d_lhs_prev = Tensor::new(t.derivative.borrow().clone(), &t.shape).unwrap();

            let d_lhs = &d_lhs + &d_lhs_prev;
            *t.derivative.borrow_mut() = d_lhs.data;
        }

        if let Some(t) = self.rhs_parent {
            let d_rhs = match self.create_op {
                Some(Ops::TensorOp(TensorOp::Add)) => Tensor::new(
                    vec![1.0; Tensor::calc_tensor_len_from_shape(&self.shape)],
                    &self.shape,
                ),
                Some(Ops::TensorOp(TensorOp::Sub)) => Tensor::new(
                    vec![1.0; Tensor::calc_tensor_len_from_shape(&self.shape)],
                    &self.shape,
                ),
                Some(Ops::TensorOp(TensorOp::Mul)) => Ok(self.lhs_parent.unwrap().clone()),
                Some(Ops::TensorOp(TensorOp::Div)) => {
                    let neg1 = Tensor::new(vec![-1.0], &[1]).unwrap();
                    let t_powed = t.pow(-2.0).unwrap();

                    let temp = &neg1 * self.lhs_parent.unwrap();
                    let temp = &temp * &t_powed;

                    Tensor::new(temp.data.clone(), &temp.shape)
                }
                Some(Ops::Matmul) => self.lhs_parent.unwrap().transpose(),
                Some(Ops::Concat((_concat_dim, _concat_dim_size))) => Tensor::zeros(&[1]), // this is never used
                _ => Err(TensorError::GradError),
            }
            .unwrap();

            let d_rhs = match &self.create_op {
                Some(Ops::Matmul) => d_rhs.matmul(&d).unwrap(),
                Some(Ops::Concat((concat_dim, concat_dim_size))) => {
                    let mut idxs = vec![[0, -1]; d.shape.len()];
                    idxs[*concat_dim] = [*concat_dim_size as isize, -1];

                    d.slice(&idxs).unwrap()
                }

                _ => &d_rhs * &d,
            };

            let d_rhs_prev = Tensor::new(t.derivative.borrow().clone(), &t.shape).unwrap();
            let d_rhs = &d_rhs + &d_rhs_prev;
            *t.derivative.borrow_mut() = d_rhs.data;
        }
    }

    pub fn backward(&self) {
        // from https://github.com/evcu/numpy_autograd/blob/master/my_autograd.py#L147
        let mut seen: Vec<&Tensor> = Vec::new();
        let mut sorted: Vec<&Tensor> = Vec::new();

        fn topological_sort<'a>(
            vr: &'a Tensor,
            seen: &mut Vec<&Tensor<'a>>,
            sorted: &mut Vec<&Tensor<'a>>,
        ) {
            if !seen.contains(&vr) && (vr.lhs_parent.is_some() || vr.rhs_parent.is_some()) {
                seen.push(vr);

                if vr.lhs_parent.is_some() {
                    topological_sort(vr.lhs_parent.unwrap(), seen, sorted);
                }
                if vr.rhs_parent.is_some() {
                    topological_sort(vr.rhs_parent.unwrap(), seen, sorted);
                }

                sorted.push(vr);
            }
        }

        topological_sort(&self, &mut seen, &mut sorted);

        sorted.reverse();

        *sorted[0].derivative.borrow_mut() =
            vec![1.0; Tensor::calc_tensor_len_from_shape(&sorted[0].shape)];

        for t in sorted.iter() {
            t.grad()
        }
    }

    pub fn clear(&self) {
        *(self.derivative.borrow_mut()) =
            vec![0.0; Tensor::calc_tensor_len_from_shape(&self.shape)];

        if let Some(t) = self.lhs_parent {
            t.clear();
        }

        if let Some(t) = self.rhs_parent {
            t.clear();
        }
    }
}

impl<'a> Add for &'a Tensor<'a> {
    type Output = Tensor<'a>;

    fn add(self, other: &'a Tensor) -> Tensor<'a> {
        match self.tensor_op(other, Ops::TensorOp(TensorOp::Add)) {
            Ok(t) => t,
            Err(e) => panic!("{}", e),
        }
    }
}

impl<'a> Sub for &'a Tensor<'a> {
    type Output = Tensor<'a>;

    fn sub(self, other: &'a Tensor) -> Tensor<'a> {
        match self.tensor_op(other, Ops::TensorOp(TensorOp::Sub)) {
            Ok(t) => t,
            Err(e) => panic!("{}", e),
        }
    }
}

impl<'a> Mul for &'a Tensor<'a> {
    type Output = Tensor<'a>;

    fn mul(self, other: &'a Tensor) -> Tensor<'a> {
        match self.tensor_op(other, Ops::TensorOp(TensorOp::Mul)) {
            Ok(t) => t,
            Err(e) => panic!("{}", e),
        }
    }
}

impl<'a> Div for &'a Tensor<'a> {
    type Output = Tensor<'a>;

    fn div(self, other: &'a Tensor) -> Tensor<'a> {
        match self.tensor_op(other, Ops::TensorOp(TensorOp::Div)) {
            Ok(t) => t,
            Err(e) => panic!("{}", e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_tensor_from_vec() {
        let _t = Tensor {
            data: vec![0.0; 16],
            shape: vec![16],
            strides: vec![1],
            track_grad: false,
            lhs_parent: None,
            rhs_parent: None,
            create_op: None,
            derivative: RefCell::new(vec![0.0; 16]),
        };
    }

    #[test]
    fn allocate_tensor_zeros() {
        let _t = Tensor::zeros(&[2, 4]).unwrap();
    }

    #[test]
    #[should_panic]
    fn try_allocate_tensor_no_shape() {
        let _t = Tensor::zeros(&[]).unwrap();
    }

    #[test]
    #[should_panic]
    fn try_allocate_tensor_too_many_dims() {
        let _t = Tensor::zeros(&[2, 2, 2, 2, 2]).unwrap();
    }

    #[test]
    #[should_panic]
    fn try_slice_tensor_too_large_slice() {
        let t = Tensor::zeros(&[2, 2]).unwrap();
        let _x = t.slice(&[[0, 1], [0, 1], [0, 1]]).unwrap();
    }

    #[test]
    fn try_slice_tensor_negative_1() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let x = t.slice(&[[0, -1], [0, 1]]).unwrap();

        assert!((x.data == vec![1.0, 3.0]) && (*x.shape == [2]));
    }

    #[test]
    fn try_slice_fewer_dims() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let x = t.slice(&[[0, 1]]).unwrap();

        assert!((x.data == vec![1.0, 2.0]) && (x.shape == vec![2]));
    }

    #[test]
    #[should_panic]
    fn try_slice_tensor_start_greater_than_stop() {
        let t = Tensor::zeros(&[2, 2]).unwrap();
        let _x = t.slice(&[[0, 1], [1, 0]]).unwrap();
    }

    #[test]
    #[should_panic]
    fn try_slice_tensor_stop_greater_than_shape() {
        let t = Tensor::zeros(&[2, 2]).unwrap();
        let _x = t.slice(&[[0, 1], [0, 3]]).unwrap();
    }

    #[test]
    fn slice_tensor_1d_element() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();

        let x = t.slice(&[[0, 1]]).unwrap();

        assert!((x.data == vec![1.0]) && (x.shape == vec![1]) && (x.strides == vec![1]))
    }
    #[test]
    fn slice_tensor_2d_element() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let x = t.slice(&[[0, 1], [0, 1]]).unwrap();

        assert!((x.data == vec![1.0]) && (x.shape == vec![1]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_2d_row() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let x = t.slice(&[[0, 1], [0, 2]]).unwrap();

        assert!((x.data == vec![1.0, 2.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_2d_col() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let x = t.slice(&[[0, 2], [0, 1]]).unwrap();

        assert!((x.data == vec![1.0, 3.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_3d_element() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]).unwrap();

        let x = t.slice(&[[0, 1], [0, 1], [0, 1]]).unwrap();

        assert!((x.data == vec![1.0]) && (x.shape == vec![1]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_3d_row() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]).unwrap();

        let x = t.slice(&[[0, 1], [0, 1], [0, 2]]).unwrap();

        assert!((x.data == vec![1.0, 2.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_3d_col() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]).unwrap();

        let x = t.slice(&[[0, 1], [0, 2], [0, 1]]).unwrap();

        assert!((x.data == vec![1.0, 3.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_3d_channel() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]).unwrap();

        let x = t.slice(&[[0, 2], [0, 1], [0, 1]]).unwrap();

        assert!((x.data == vec![1.0, 5.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_3d_chunk() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]).unwrap();

        let x = t.slice(&[[0, 2], [0, 2], [0, 1]]).unwrap();

        assert!(
            (x.data == vec![1.0, 3.0, 5.0, 7.0])
                && (*x.shape == [2, 2])
                && (x.strides == vec![2, 1])
        )
    }
    #[test]
    fn slice_tensor_4d_element() {
        let t = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 2, 2],
        )
        .unwrap();

        let x = t.slice(&[[0, 1], [0, 1], [0, 1], [0, 1]]).unwrap();

        assert!((x.data == vec![1.0]) && (x.shape == vec![1]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_4d_row() {
        let t = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 2, 2],
        )
        .unwrap();

        let x = t.slice(&[[0, 1], [0, 1], [0, 1], [0, 2]]).unwrap();

        assert!((x.data == vec![1.0, 2.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_4d_col() {
        let t = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 2, 2],
        )
        .unwrap();

        let x = t.slice(&[[0, 1], [0, 1], [0, 2], [0, 1]]).unwrap();

        assert!((x.data == vec![1.0, 3.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_4d_channel() {
        let t = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 2, 2],
        )
        .unwrap();

        let x = t.slice(&[[0, 1], [0, 2], [0, 1], [0, 1]]).unwrap();

        assert!((x.data == vec![1.0, 5.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_4d_batch() {
        let t = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 2, 2],
        )
        .unwrap();

        let x = t.slice(&[[0, 2], [0, 1], [0, 1], [0, 1]]).unwrap();

        assert!((x.data == vec![1.0, 9.0]) && (x.shape == vec![2]) && (x.strides == vec![1]))
    }

    #[test]
    fn slice_tensor_4d_chunk() {
        let t = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 2, 2],
        )
        .unwrap();

        let x = t.slice(&[[0, 2], [0, 2], [0, 1], [0, 1]]).unwrap();

        assert!(
            (x.data == vec![1.0, 5.0, 9.0, 13.0])
                && (*x.shape == [2, 2])
                && (x.strides == vec![2, 1])
        )
    }

    #[test]
    fn view_tensor() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let x = t.view(&[4]).unwrap();

        assert!((x.data == vec![1.0, 2.0, 3.0, 4.0]) && (x.shape == vec![4]));
    }

    #[test]
    fn view_tensor_neg_1() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let _x = t.view(&[-1]).unwrap();
    }

    #[test]
    fn view_tensor_neg_2() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let _x = t.view(&[2, -1]).unwrap();
    }

    #[test]
    fn view_tensor_neg_3() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let _x = t.view(&[1, -1]).unwrap();
    }

    #[test]
    fn view_tensor_neg_4() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let _x = t.view(&[4, -1]).unwrap();
    }

    #[test]
    #[should_panic]
    fn view_tensor_should_panic() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let _x = t.view(&[1, 1, 1, 1]).unwrap();
    }

    #[test]
    #[should_panic]
    fn view_tensor_neg_should_panic() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let _x = t.view(&[-1, -1]).unwrap();
    }

    #[test]
    #[should_panic]
    fn view_tensor_neg_should_panic_2() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let _x = t.view(&[6, -1]).unwrap();
    }

    #[test]
    fn elementwise_add_op() {
        let a = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[4]).unwrap();
        let b = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[4]).unwrap();

        let c = &a + &b;

        assert!((c.data == vec![4.0, 6.0, 8.0, 10.0]) && (c.shape == vec![4]))
    }

    #[test]
    fn elementwise_sub_op() {
        let a = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]).unwrap();
        let b = Tensor::new(vec![2.0, 3.0], &[2]).unwrap();

        let c = &a - &b;

        assert!((c.data == vec![0.0, 0.0, 2.0, 2.0]) && (c.shape == vec![2, 2]))
    }

    #[test]
    fn elementwise_mul_op() {
        let a = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]).unwrap();
        let b = Tensor::new(vec![2.0, 3.0], &[2]).unwrap();

        let c = &a * &b;

        assert!((c.data == vec![4.0, 9.0, 8.0, 15.0]) && (c.shape == vec![2, 2]))
    }

    #[test]
    fn elementwise_div_op() {
        let a = Tensor::new(vec![2.0, 4.0, 6.0, 8.0], &[2, 2]).unwrap();
        let b = Tensor::new(vec![2.0, 4.0], &[2]).unwrap();

        let c = &a / &b;

        assert!((c.data == vec![1.0, 1.0, 3.0, 2.0]) && (c.shape == vec![2, 2]))
    }

    #[test]
    fn broadcast_shapes() {
        let a = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[1, 4]).unwrap();
        let b = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[4]).unwrap();

        let c = &a + &b;

        assert!((c.data == vec![4.0, 6.0, 8.0, 10.0]) && (c.shape == vec![1, 4]));
    }

    #[test]
    fn broadcast_dims() {
        let a = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]).unwrap();
        let b = Tensor::new(vec![2.0, 3.0], &[2]).unwrap();

        let c = &a + &b;

        assert!((c.data == vec![4.0, 6.0, 6.0, 8.0]) && (c.shape == vec![2, 2]));
    }

    #[test]
    fn broadcast_shapes_and_dims() {
        let a = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[1, 2, 2]).unwrap();
        let b = Tensor::new(vec![2.0, 3.0], &[2]).unwrap();

        let c = &a + &b;

        assert!((c.data == vec![4.0, 6.0, 6.0, 8.0]) && (c.shape == vec![1, 2, 2]));
    }

    #[test]
    #[should_panic]
    fn broadcast_should_panic() {
        let a = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[4]).unwrap();
        let b = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]).unwrap();

        let _c = &a + &b;
    }

    #[test]
    fn pow() {
        let a = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], &[4]).unwrap();

        let b = a.pow(2.0).unwrap();

        assert!((b.data == vec![4.0, 9.0, 16.0, 25.0]) && (b.shape == vec![4]))
    }

    #[test]
    fn sum_1d() {
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[16],
        )
        .unwrap();

        let y = x.sum(0).unwrap();

        assert!((y.data == vec![136.0]) && (y.shape == vec![1]))
    }

    #[test]
    fn sum_2d() {
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 8],
        )
        .unwrap();

        let y = x.sum(1).unwrap();

        assert!((y.data == vec![36.0, 100.0]) && (y.shape == vec![2]))
    }

    #[test]
    fn sum_3d() {
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 4],
        )
        .unwrap();

        let y = x.sum(1).unwrap();

        assert!(
            (y.data == vec![6.0, 8.0, 10.0, 12.0, 22.0, 24.0, 26.0, 28.0])
                && (y.shape == vec![2, 4])
        )
    }

    #[test]
    fn sum_4d() {
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 2, 2],
        )
        .unwrap();

        let y = x.sum(-1).unwrap();

        assert!(
            (y.data == vec![3.0, 7.0, 11.0, 15.0, 19.0, 23.0, 27.0, 31.0])
                && (y.shape == vec![2, 2, 2])
        )
    }

    #[test]
    fn mean() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        let y = x.mean(-1).unwrap();

        assert!((y.data == vec![2.0, 5.0]) && (y.shape == vec![2]))
    }

    #[test]
    fn max() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        let y = x.max(-1).unwrap();

        assert!((y.data == vec![3.0, 6.0]) && (y.shape == vec![2]))
    }

    #[test]
    fn min() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        let y = x.min(-1).unwrap();

        assert!((y.data == vec![1.0, 4.0]) && (y.shape == vec![2]))
    }

    #[test]
    fn argmax() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        let y = x.argmax(-1).unwrap();

        assert!((y.data == vec![2.0, 2.0]) && (y.shape == vec![2]))
    }

    #[test]
    fn argmin() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        let y = x.argmin(-1).unwrap();

        assert!((y.data == vec![0.0, 0.0]) && (y.shape == vec![2]))
    }

    #[test]
    fn matmul_2d() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let y = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]).unwrap();

        let z = x.matmul(&y).unwrap();

        assert!((z.data == vec![30.0, 36.0, 42.0, 66.0, 81.0, 96.0]) && (z.shape == vec![2, 3]))
    }

    #[test]
    fn matmul_3d() {
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 4],
        )
        .unwrap();
        let y = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 4, 2],
        )
        .unwrap();

        let z = x.matmul(&y).unwrap();

        assert!(
            (z.data == vec![50.0, 60.0, 114.0, 140.0, 514.0, 556.0, 706.0, 764.0])
                && (z.shape == vec![2, 2, 2])
        )
    }

    #[test]
    fn matmul_4d() {
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 2, 2],
        )
        .unwrap();
        let y = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 2, 2],
        )
        .unwrap();

        let z = x.matmul(&y).unwrap();

        assert!(
            (z.data
                == vec![
                    7.0, 10.0, 15.0, 22.0, 67.0, 78.0, 91.0, 106.0, 191.0, 210.0, 231.0, 254.0,
                    379.0, 406.0, 435.0, 466.0
                ])
                && (z.shape == vec![2, 2, 2, 2])
        )
    }

    #[test]
    fn concat_1d() {
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[16],
        )
        .unwrap();
        let y = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[16],
        )
        .unwrap();

        let z = x.concat(&y, -1).unwrap();

        assert!(
            (z.data
                == vec![
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                    15.0, 16.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                    13.0, 14.0, 15.0, 16.0,
                ])
                && (z.shape == vec![32])
        )
    }

    #[test]
    fn concat_2d() {
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 8],
        )
        .unwrap();
        let y = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 8],
        )
        .unwrap();

        let z = x.concat(&y, -1).unwrap();

        assert!(
            (z.data
                == vec![
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                    9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 9.0, 10.0, 11.0, 12.0, 13.0,
                    14.0, 15.0, 16.0,
                ])
                && (z.shape == vec![2, 16])
        )
    }

    #[test]
    fn concat_3d() {
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 4],
        )
        .unwrap();
        let y = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 4],
        )
        .unwrap();

        let z = x.concat(&y, -1).unwrap();

        assert!(
            (z.data
                == vec![
                    1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0,
                    9.0, 10.0, 11.0, 12.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 13.0,
                    14.0, 15.0, 16.0
                ])
                && (z.shape == vec![2, 2, 8])
        )
    }

    #[test]
    fn concat_4d() {
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 2, 2],
        )
        .unwrap();
        let y = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 2, 2],
        )
        .unwrap();

        let z = x.concat(&y, -1).unwrap();

        assert!(
            (z.data
                == vec![
                    1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 6.0, 5.0, 6.0, 7.0, 8.0, 7.0, 8.0,
                    9.0, 10.0, 9.0, 10.0, 11.0, 12.0, 11.0, 12.0, 13.0, 14.0, 13.0, 14.0, 15.0,
                    16.0, 15.0, 16.0
                ])
                && (z.shape == vec![2, 2, 2, 4])
        )
    }

    #[test]
    fn transpose_1d() {
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[16],
        )
        .unwrap();

        let z = x.transpose().unwrap();

        assert!(
            (z.data
                == vec![
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                    15.0, 16.0,
                ])
                && (z.shape == vec![16])
        )
    }

    #[test]
    fn transpose_2d() {
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 8],
        )
        .unwrap();

        let z = x.transpose().unwrap();

        assert!(
            (z.data
                == vec![
                    1.0, 9.0, 2.0, 10.0, 3.0, 11.0, 4.0, 12.0, 5.0, 13.0, 6.0, 14.0, 7.0, 15.0,
                    8.0, 16.0
                ])
                && (z.shape == vec![8, 2])
        )
    }

    #[test]
    fn transpose_3d() {
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 4],
        )
        .unwrap();

        let z = x.transpose().unwrap();

        assert!(
            (z.data
                == vec![
                    1.0, 9.0, 5.0, 13.0, 2.0, 10.0, 6.0, 14.0, 3.0, 11.0, 7.0, 15.0, 4.0, 12.0,
                    8.0, 16.0
                ])
                && (z.shape == vec![4, 2, 2])
        )
    }

    #[test]
    fn transpose_4d() {
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 2, 2],
        )
        .unwrap();

        let z = x.transpose().unwrap();

        assert!(
            (z.data
                == vec![
                    1.0, 9.0, 5.0, 13.0, 3.0, 11.0, 7.0, 15.0, 2.0, 10.0, 6.0, 14.0, 4.0, 12.0,
                    8.0, 16.0
                ])
                && (z.shape == vec![2, 2, 2, 2])
        )
    }

    #[test]
    fn clone() {
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[2, 2, 2, 2],
        )
        .unwrap();

        let y = x.clone();

        assert!((x.data == y.data) && (x.shape == y.shape))
    }

    #[test]
    fn backwards() {
        let x = Tensor::new(vec![-2.0], &[1]).unwrap();
        let y = Tensor::new(vec![5.0], &[1]).unwrap();

        let q = &x + &y;

        let z = Tensor::new(vec![-4.0], &[1]).unwrap();

        let out = &q * &z;

        out.backward();

        assert!(
            (out.derivative == RefCell::new(vec![1.0])
                && (z.derivative == RefCell::new(vec![3.0]))
                && (q.derivative == RefCell::new(vec![-4.0]))
                && (x.derivative == RefCell::new(vec![-4.0]))
                && (y.derivative == RefCell::new(vec![-4.0])))
        )
    }

    #[allow(clippy::many_single_char_names)]
    #[test]
    fn backwards_shared_tensor() {
        let a = Tensor::new(vec![2.0], &[1]).unwrap();
        let b = Tensor::new(vec![3.0], &[1]).unwrap();
        let c = Tensor::new(vec![4.0], &[1]).unwrap();

        let e = &a * &b;
        let f = &e * &c;

        let out = &a + &f;

        out.backward();

        assert!(
            (out.derivative == RefCell::new(vec![1.0]))
                && (a.derivative == RefCell::new(vec![13.0]))
                && (b.derivative == RefCell::new(vec![8.0]))
                && (c.derivative == RefCell::new(vec![6.0]))
        )
    }

    #[test]
    fn backwards_add() {
        let a = Tensor::new(vec![2.0], &[1]).unwrap();
        let b = Tensor::new(vec![3.0], &[1]).unwrap();

        let c = &a + &b;

        c.backward();

        assert!(
            (c.derivative == RefCell::new(vec![1.0]))
                && (a.derivative == RefCell::new(vec![1.0]))
                && (b.derivative == RefCell::new(vec![1.0]))
        );
    }

    #[test]
    fn backwards_subtract() {
        let a = Tensor::new(vec![2.0], &[1]).unwrap();
        let b = Tensor::new(vec![3.0], &[1]).unwrap();

        let c = &a - &b;

        c.backward();

        assert!(
            (c.derivative == RefCell::new(vec![1.0]))
                && (a.derivative == RefCell::new(vec![1.0]))
                && (b.derivative == RefCell::new(vec![1.0]))
        );
    }

    #[test]
    fn backwards_multiply() {
        let a = Tensor::new(vec![2.0], &[1]).unwrap();
        let b = Tensor::new(vec![3.0], &[1]).unwrap();

        let c = &a * &b;

        c.backward();

        assert!(
            (c.derivative == RefCell::new(vec![1.0]))
                && (a.derivative == RefCell::new(vec![3.0]))
                && (b.derivative == RefCell::new(vec![2.0]))
        );
    }

    #[test]
    fn backwards_divide() {
        let a = Tensor::new(vec![2.0], &[1]).unwrap();
        let b = Tensor::new(vec![3.0], &[1]).unwrap();

        let c = &a / &b;

        c.backward();

        assert!(
            (c.derivative == RefCell::new(vec![1.0]))
                && (a.derivative == RefCell::new(vec![1.0 / 3.0]))
                && (b.derivative == RefCell::new(vec![-1.0 * 2.0 * ((3.0 as f32).powi(-2))]))
        );
    }

    #[test]
    fn backwards_pow() {
        let a = Tensor::new(vec![-3.0], &[1]).unwrap();

        let b = a.pow(2.0).unwrap();

        b.backward();

        assert!(
            (b.derivative == RefCell::new(vec![1.0]) && (a.derivative == RefCell::new(vec![-6.0])))
        )
    }

    #[test]
    fn backwards_sqrt() {
        let a = Tensor::new(vec![3.0], &[1]).unwrap();

        let b = a.sqrt().unwrap();

        b.backward();

        assert!(
            (b.derivative == RefCell::new(vec![1.0])
                && (a.derivative == RefCell::new(vec![0.5 * ((3.0 as f32).powf(-0.5))])))
        )
    }

    #[test]
    fn backwards_exp() {
        let a = Tensor::new(vec![3.0], &[1]).unwrap();

        let b = a.exp().unwrap();

        b.backward();

        assert!(
            (b.derivative == RefCell::new(vec![1.0])
                && (a.derivative == RefCell::new(vec![(3.0 as f32).exp()])))
        )
    }

    #[test]
    fn backwards_log10() {
        let a = Tensor::new(vec![3.0], &[1]).unwrap();

        let b = a.log10().unwrap();

        b.backward();

        assert!(
            (b.derivative == RefCell::new(vec![1.0])
                && (a.derivative == RefCell::new(vec![1.0 / (3.0 * (10.0 as f32).log(E))])))
        )
    }

    #[test]
    fn backwards_log() {
        let a = Tensor::new(vec![3.0], &[1]).unwrap();

        let b = a.log().unwrap();

        b.backward();

        assert!(
            (b.derivative == RefCell::new(vec![1.0])
                && (a.derivative == RefCell::new(vec![1.0 / 3.0])))
        )
    }

    #[test]
    fn backwards_abs_pos() {
        let a = Tensor::new(vec![3.0], &[1]).unwrap();

        let b = a.abs().unwrap();

        b.backward();

        assert!(
            (b.derivative == RefCell::new(vec![1.0]) && (a.derivative == RefCell::new(vec![1.0])))
        )
    }

    #[test]
    fn backwards_abs_neg() {
        let a = Tensor::new(vec![-3.0], &[1]).unwrap();

        let b = a.abs().unwrap();

        b.backward();

        assert!(
            (b.derivative == RefCell::new(vec![1.0]) && (a.derivative == RefCell::new(vec![-1.0])))
        )
    }

    #[test]
    fn backwards_abs_zero() {
        let a = Tensor::new(vec![0.0], &[1]).unwrap();

        let b = a.abs().unwrap();

        b.backward();

        assert!(
            (b.derivative == RefCell::new(vec![1.0]) && (a.derivative == RefCell::new(vec![0.0])))
        )
    }

    #[test]
    fn backwards_sin() {
        let a = Tensor::new(vec![2.0], &[1]).unwrap();

        let b = a.sin().unwrap();

        b.backward();

        assert!(
            (b.derivative == RefCell::new(vec![1.0])
                && (a.derivative == RefCell::new(vec![(2.0 as f32).cos()])))
        )
    }

    #[test]
    fn backwards_cos() {
        let a = Tensor::new(vec![2.0], &[1]).unwrap();

        let b = a.cos().unwrap();

        b.backward();

        assert!(
            (b.derivative == RefCell::new(vec![1.0])
                && (a.derivative == RefCell::new(vec![-1.0 * (2.0 as f32).sin()])))
        )
    }

    #[test]
    fn backwards_tan() {
        let a = Tensor::new(vec![2.0], &[1]).unwrap();

        let b = a.tan().unwrap();

        b.backward();

        assert!(
            (b.derivative == RefCell::new(vec![1.0])
                && (a.derivative == RefCell::new(vec![1.0 / (2.0 as f32).cos().powi(2)])))
        )
    }

    #[test]
    fn backwards_matmul() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();

        let c = a.matmul(&b).unwrap();

        c.backward();

        assert!(
            (c.derivative == RefCell::new(vec![1.0, 1.0, 1.0, 1.0]))
                && (a.derivative == RefCell::new(vec![3.0, 7.0, 11.0, 3.0, 7.0, 11.0]))
                && (b.derivative == RefCell::new(vec![5.0, 5.0, 7.0, 7.0, 9.0, 9.0]))
        )
    }

    #[test]
    fn backwards_slice() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        let b = a.slice(&[[0, 1], [0, 1]]).unwrap();

        b.backward();

        assert!(
            (b.derivative == RefCell::new(vec![1.0]))
                && (a.derivative == RefCell::new(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        )
    }

    #[test]
    fn backwards_slice_2() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        let b = a.slice(&[[0, 1], [0, -1]]).unwrap();

        b.backward();

        assert!(
            (b.derivative == RefCell::new(vec![1.0, 1.0, 1.0]))
                && (a.derivative == RefCell::new(vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))
        )
    }

    #[test]
    fn backwards_slice_3() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        let b = a.slice(&[[0, -1], [0, 1]]).unwrap();

        b.backward();

        assert!(
            (b.derivative == RefCell::new(vec![1.0, 1.0]))
                && (a.derivative == RefCell::new(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]))
        )
    }

    #[test]
    fn backwards_slice_4() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        let b = a.slice(&[[0, 2], [0, 2]]).unwrap();

        b.backward();

        assert!(
            (b.derivative == RefCell::new(vec![1.0, 1.0, 1.0, 1.0]))
                && (a.derivative == RefCell::new(vec![1.0, 1.0, 0.0, 1.0, 1.0, 0.0]))
        )
    }

    #[test]
    fn backwards_transpose() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        let y = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();

        let temp = x.transpose().unwrap();

        let z = temp.matmul(&y).unwrap();

        z.backward();

        assert!(
            (x.derivative == RefCell::new(vec![3.0, 3.0, 7.0, 7.0, 11.0, 11.0]))
                && (y.derivative == RefCell::new(vec![3.0, 3.0, 7.0, 7.0, 11.0, 11.0]))
        )
    }

    #[test]
    fn backwards_view() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        let y = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();

        let temp = x.view(&[2, 3]).unwrap();

        let z = temp.matmul(&y).unwrap();

        z.backward();

        assert!(
            (x.derivative == RefCell::new(vec![3.0, 7.0, 11.0, 3.0, 7.0, 11.0]))
                && (y.derivative == RefCell::new(vec![5.0, 5.0, 7.0, 7.0, 9.0, 9.0]))
        )
    }

    #[test]
    fn backwards_concat() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        let y = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();

        let one = Tensor::new(vec![1.0], &[1]).unwrap();
        let two = Tensor::new(vec![2.0], &[1]).unwrap();

        let xx = &one * &x;
        let yy = &two * &y;

        let z = xx.concat(&yy, -1).unwrap();

        z.backward();

        assert!(
            (x.derivative == RefCell::new(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
                && (y.derivative == RefCell::new(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0]))
        )
    }

    #[test]
    fn backwards_sum() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();

        let y = x.sum(-1).unwrap();

        y.backward();

        assert!(
            (x.derivative == RefCell::new(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
                && (y.derivative == RefCell::new(vec![1.0, 1.0, 1.0]))
        )
    }

    #[test]
    fn backwards_mean() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        let y = x.mean(0).unwrap();

        y.backward();

        assert!((x.derivative == RefCell::new(vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5])))
    }

    #[test]
    fn backwards_clear() {
        let x = Tensor::new(vec![-2.0], &[1]).unwrap();
        let y = Tensor::new(vec![5.0], &[1]).unwrap();

        let q = &x + &y;

        q.backward();
        q.clear();

        assert!(
            (q.derivative == RefCell::new(vec![0.0]))
                && (x.derivative == RefCell::new(vec![0.0]))
                && (y.derivative == RefCell::new(vec![0.0]))
        )
    }
}
