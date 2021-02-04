//! # L2
//!# What is L2?
//!
//!> L2 is named after the L2 or Euclidean distance, a popular distance function in deep learning
//!
//!L2 is a Pytorch-style Tensor+Autograd library written in the Rust programming language. It contains a multidimensional array class, `Tensor`, with support for strided arrays, numpy-style array slicing,
//!broadcasting, and most major math operations (including fast, BLAS-accelerated matrix multiplication!). On top of this, L2 has a built-in efficient graph-based autograd engine that keeps track of all
//!operations performed on a tensor and topologically sorts and traverses the graph to compute the gradients.
//!
//!I also made a more simplified C++ version of l2 last year, which you can take a look at [here](https://github.com/bilal2vec/L2/tree/c%2B%2B)
//!
//!# Example
//!
//!```rust
//!use l2::tensor::*;
//!
//!fn main() -> Result<(), l2::errors::TensorError> {
//!    let x: Tensor = Tensor::normal(&[2, 4], 0.0, 1.0)?;
//!    let y: Tensor = Tensor::normal(&[4, 1], 0.0, 1.0)?;
//!
//!    let z: Tensor = l2::matmul(&x, &y)?;
//!
//!    z.backward();
//!
//!    println!("{}", z);
//!
//!    Ok(())
//!}
//!```
//!
//!# Design choices
//!
//!I made L2 to get better at using Rust and to learn more about how libraries like Pytorch and Tensorflow work behind the scenes, so don't expect this library to be production-ready :)
//!
//!L2 is surprisingly fast especially since I didn't try very hard to optimize all the operators, it's usually only less than one order of magnitude slower than Pytorch in most of the benchmarks that I ran. L2 //!only supports a cpu backend at the moment since I'm not familiar enough with rust to start working with CUDA and cudnn. So far, l2 doesn't have any Pytorch-style abstractions like the Parameter, Layer, or
//!Module classes. There might still be some bugs in the transpose operators and calling `.backward()` on tensors with more dimensions. I was interested in using Rust's [Const Generics](https://github.com/
//!rust-lang/rfcs/blob/master/text/2000-const-generics.md) to run compile-time shape checks but I decided to leave it until some other time.
//!
//!# Contributing
//!
//!This repository is still a work in progress, so if you find a bug, think there is something missing, or have any suggestions for new features, feel free to open an issue or a pull request. Feel free to use
//!the library or code from it in your own projects, and if you feel that some code used in this project hasn't been properly accredited, please open an issue.
//!
//!# Authors
//!
//!-   _Bilal Khan_
//!
//!# License
//!
//!This project is licensed under the MIT License - see the license file for details
//!
//!# Acknowledgements
//!
//!The fast.ai deep learning from the foundations course (https://course.fast.ai/part2) teaches a lot about how to make your own deep learning library
//!
//!Some of the resources that I found useful when working on this library include:
//!
//!-   http://blog.ezyang.com/2019/05/pytorch-internals/
//!-   https://pytorch.org/tutorials/beginner/nn_tutorial.html
//!-   https://eisenjulian.github.io/deep-learning-in-100-lines/
//!-   https://medium.com/@florian.caesar/how-to-create-a-machine-learning-framework-from-scratch-in-491-steps-93428369a4eb
//!-   https://medium.com/@johan.mabille/how-we-wrote-xtensor-1-n-n-dimensional-containers-f79f9f4966a7
//!-   https://erikpartridge.com/2019-03/rust-ml-simd-blas-lapack
//!-   https://medium.com/@GolDDranks/things-rust-doesnt-let-you-do-draft-f596a3c740a5
//!-   https://datascience.stackexchange.com/questions/20139/gradients-for-bias-terms-in-backpropagation
//!-   https://cs231n.github.io/optimization-2/
//!-   https://cs231n.github.io/neural-networks-case-study/#grad
//!-   https://stackoverflow.com/questions/38082835/backpropagation-in-gradient-descent-for-neural-networks-vs-linear-regression
//!-   https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b
//!-   https://stackoverflow.com/questions/38082835/backpropagation-in-gradient-descent-for-neural-networks-vs-linear-regression
//!-   https://github.com/karpathy/micrograd
//!-   https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation
//!    -   https://github.com/ibab/rust-ad
//!    -   https://github.com/Rufflewind/revad/blob/eb3978b3ccdfa8189f3ff59d1ecee71f51c33fd7/revad.py
//!    -   https://github.com/srirambandi/ai
//!-   https://discuss.pytorch.org/t/is-pytorch-autograd-tape-based/13992/3
//!-   https://www.reddit.com/r/MachineLearning/comments/8ep130/d_how_does_autograd_work/
//!-   https://github.com/mattjj/autodidact
//!-   https://github.com/karpathy/recurrentjs
//!-   https://github.com/karpathy/randomfun
//!-   https://medium.com/@ralphmao95/simple-autograd-implementation-understand-automatic-differentiation-hand-by-hand-9e86f6d703ab
//!-   https://evcu.github.io/ml/autograd/
//!-   https://blog.paperspace.com/pytorch-101-understanding-graphs-and-automatic-differentiation/
//!-   https://github.com/maciejkula/wyrm
//!-   https://medium.com/@maciejkula/building-an-autodifferentiation-library-9ccf32c7a658
//!-   https://github.com/evcu/numpy_autograd/blob/master/my_autograd.py#L147
//!-   https://github.com/evcu/numpy_autograd/blob/master/Autograd.ipynb
//!-   https://cs231n.github.io/optimization-2/
//!-   https://github.com/explosion/thinc
//!-   https://github.com/joelgrus/joelnet
//!-   https://github.com/QuantStack/xtensor
//!-   https://github.com/ThinkingTransistor/Sigma
//!-   https://github.com/mratsim/Arraymancer
//!-   https://github.com/siekmanj/sieknet
//!-   https://github.com/siekmanj/sieknet_2.0
//!-   https://github.com/Daniel-Liu-c0deb0t/Java-Machine-Learning
//!-   https://github.com/karpathy/micrograd
//!
//!This README is based on:
//!
//!-   https://github.com/bilal2vec/pytorch_zoo
//!-   https://github.com/bilal2vec/grover
//!-   https://github.com/rish-16/gpt2client
//!-   https://github.com/mxbi/mlcrate
//!-   https://github.com/athityakumar/colorls
//!-   https://github.com/amitmerchant1990/electron-markdownify
//!
//!I used carbon.now.sh with the "Shades of Purple" theme for the screenshot at the beginning of this README
//!
//!This project contains ~4300 lines of code
pub mod errors;
mod ops;
pub mod tensor;

use errors::TensorError;
use tensor::Tensor;

pub fn add<'a>(lhs: &'a Tensor, rhs: &'a Tensor) -> Tensor<'a> {
    lhs + rhs
}

pub fn sub<'a>(lhs: &'a Tensor, rhs: &'a Tensor) -> Tensor<'a> {
    lhs - rhs
}

pub fn mul<'a>(lhs: &'a Tensor, rhs: &'a Tensor) -> Tensor<'a> {
    lhs * rhs
}

pub fn div<'a>(lhs: &'a Tensor, rhs: &'a Tensor) -> Tensor<'a> {
    lhs / rhs
}

pub fn pow<'a>(lhs: &'a Tensor, exp: f32) -> Result<Tensor<'a>, TensorError> {
    lhs.pow(exp)
}

pub fn sqrt<'a>(lhs: &'a Tensor) -> Result<Tensor<'a>, TensorError> {
    lhs.sqrt()
}

pub fn exp<'a>(lhs: &'a Tensor) -> Result<Tensor<'a>, TensorError> {
    lhs.exp()
}

pub fn log10<'a>(lhs: &'a Tensor) -> Result<Tensor<'a>, TensorError> {
    lhs.log10()
}

pub fn log<'a>(lhs: &'a Tensor) -> Result<Tensor<'a>, TensorError> {
    lhs.log()
}

pub fn abs<'a>(lhs: &'a Tensor) -> Result<Tensor<'a>, TensorError> {
    lhs.abs()
}

pub fn sin<'a>(lhs: &'a Tensor) -> Result<Tensor<'a>, TensorError> {
    lhs.sin()
}

pub fn cos<'a>(lhs: &'a Tensor) -> Result<Tensor<'a>, TensorError> {
    lhs.cos()
}

pub fn tan<'a>(lhs: &'a Tensor) -> Result<Tensor<'a>, TensorError> {
    lhs.tan()
}

pub fn sum<'a>(lhs: &'a Tensor, dim: isize) -> Result<Tensor<'a>, TensorError> {
    lhs.sum(dim)
}

pub fn mean<'a>(lhs: &'a Tensor, dim: isize) -> Result<Tensor<'a>, TensorError> {
    lhs.mean(dim)
}

pub fn max<'a>(lhs: &'a Tensor, dim: isize) -> Result<Tensor<'a>, TensorError> {
    lhs.max(dim)
}

pub fn min<'a>(lhs: &'a Tensor, dim: isize) -> Result<Tensor<'a>, TensorError> {
    lhs.min(dim)
}

pub fn argmax<'a>(lhs: &'a Tensor, dim: isize) -> Result<Tensor<'a>, TensorError> {
    lhs.argmax(dim)
}

pub fn argmin<'a>(lhs: &'a Tensor, dim: isize) -> Result<Tensor<'a>, TensorError> {
    lhs.argmin(dim)
}

pub fn matmul<'a>(lhs: &'a Tensor, rhs: &'a Tensor) -> Result<Tensor<'a>, TensorError> {
    lhs.matmul(rhs)
}

pub fn concat<'a>(lhs: &'a Tensor, rhs: &'a Tensor, dim: isize) -> Result<Tensor<'a>, TensorError> {
    lhs.concat(&rhs, dim)
}

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

        let c = pow(&a, 2.0).unwrap();

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

    #[test]
    fn test_matmul() {
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

        let z = matmul(&x, &y).unwrap();

        assert!(
            (z.data == vec![50.0, 60.0, 114.0, 140.0, 514.0, 556.0, 706.0, 764.0])
                && (z.shape == vec![2, 2, 2])
        )
    }

    #[test]
    fn test_concat() {
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

        let z = concat(&x, &y, -1).unwrap();

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
}
