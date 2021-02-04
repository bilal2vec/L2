<h1 align='center'>
    l2 • 🤖
</h1>

<h4 align='center'>
    A Pytorch-style Tensor+Autograd library written in Rust
</h4>

<p align='center'>
    <a href="">
        <img src="https://github.com/bilal2vec/l2/workflows/Rust/badge.svg" alt="Rust: CI">
    </a>
    <a href="https://opensource.org/licenses/MIT">
        <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
    </a>
    <a href="https://crates.io/crates/l2">
        <img alt="crates.io l2 badge" src="http://meritbadge.herokuapp.com/l2">
    </a>
    <a href=" https://docs.rs/l2">
        <img alt="docs.rs l2 badge" src="https://docs.rs/l2/badge.svg">
    </a>
</p>

<p align='center'>
    <a href='#installation'>Installation</a> •
    <a href='#contributing'>Contributing</a> •
    <a href='#authors'>Authors</a> •
    <a href='#license'>License</a> •
    <a href='#acknowledgements'>Acknowledgements</a>
</p>

<div>
    <img src="./screenshot.png" />
</div>

<p align='center'><strong>Made by <a href='https://github.com/bilal2vec'>Bilal Khan</a> • https://bilal.software</strong></p>

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

-   [What is l2?](#what-is-l2)
-   [Quick start](#quick-start)
-   [Design choices](#design-choices)
-   [Contributing](#contributing)
-   [Authors](#authors)
-   [License](#license)
-   [Acknowledgements](#acknowledgements)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# What is l2?

> l2 is named after the l2 or Euclidean distance, a popular distance function in deep learning

l2 is a Pytorch-style Tensor+Autograd library written in Rust. It contains a multidimensional array class, `Tensor`, with support for strided arrays, numpy-style array slicing, broadcasting, and most major math operations (including fast, BLAS-accelerated matrix multiplication!). On top of this, l2 has a built-in efficient graph-based autograd engine that keeps track of all operations performed on a tensor and topologically sorts and traverses the graph to compute the gradients.

I also made a more simplified C++ version of l2 last year, which you can take a look at [here](https://github.com/bilal2vec/L2/tree/c%2B%2B)

# Quick start

Add `l2 = "1.0.3"` to your `Cargo.toml` file and add the following to `main.rs`

> Note: L2 will by default use Apple's `acclerate` BLAS library on macOS
> You can also change the BLAS library that you want to use yourself. Take a look at the [`blas-src`](https://crates.io/crates/blas-src) crate for more information

```rust
use l2::tensor::*;

let x: Tensor = Tensor::normal(&[2, 4], 0.0, 1.0)?;
let y: Tensor = Tensor::normal(&[4, 1], 0.0, 1.0)?;

let z: Tensor = l2::matmul(&x, &y)?;

z.backward();

println!("{}", z);

```

# Design choices

I made l2 to get better at using Rust and to learn more about how libraries like Pytorch and Tensorflow work behind the scenes, so don't expect this library to be production-ready :)

l2 is surprisingly fast especially since I didn't try very hard to optimize all the operators, it's usually only less than one order of magnitude slower than Pytorch in most of the benchmarks that I ran. l2 only supports a cpu backend at the moment since I'm not familiar enough with rust to start working with CUDA and cudnn. So far, l2 doesn't have any Pytorch-style abstractions like the Parameter, Layer, or Module classes. There might still be some bugs in the transpose operators and calling `.backward()` on tensors with more dimensions. I was interested in using Rust's [Const Generics](https://github.com/rust-lang/rfcs/blob/master/text/2000-const-generics.md) to run compile-time shape checks but I decided to leave it until some other time.

# Contributing

This repository is still a work in progress, so if you find a bug, think there is something missing, or have any suggestions for new features, feel free to open an issue or a pull request. Feel free to use the library or code from it in your own projects, and if you feel that some code used in this project hasn't been properly accredited, please open an issue.

# Authors

-   _Bilal Khan_

# License

This project is licensed under the MIT License - see the [license](LICENSE) file for details

# Acknowledgements

The fast.ai deep learning from the foundations course (https://course.fast.ai/part2) teaches a lot about how to make your own deep learning library

Some of the resources that I found useful when working on this library include:

-   http://blog.ezyang.com/2019/05/pytorch-internals/
-   https://pytorch.org/tutorials/beginner/nn_tutorial.html
-   https://eisenjulian.github.io/deep-learning-in-100-lines/
-   https://medium.com/@florian.caesar/how-to-create-a-machine-learning-framework-from-scratch-in-491-steps-93428369a4eb
-   https://medium.com/@johan.mabille/how-we-wrote-xtensor-1-n-n-dimensional-containers-f79f9f4966a7
-   https://erikpartridge.com/2019-03/rust-ml-simd-blas-lapack
-   https://medium.com/@GolDDranks/things-rust-doesnt-let-you-do-draft-f596a3c740a5
-   https://datascience.stackexchange.com/questions/20139/gradients-for-bias-terms-in-backpropagation
-   https://cs231n.github.io/optimization-2/
-   https://cs231n.github.io/neural-networks-case-study/#grad
-   https://stackoverflow.com/questions/38082835/backpropagation-in-gradient-descent-for-neural-networks-vs-linear-regression
-   https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b
-   https://stackoverflow.com/questions/38082835/backpropagation-in-gradient-descent-for-neural-networks-vs-linear-regression
-   https://github.com/karpathy/micrograd
-   https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation
    -   https://github.com/ibab/rust-ad
    -   https://github.com/Rufflewind/revad/blob/eb3978b3ccdfa8189f3ff59d1ecee71f51c33fd7/revad.py
    -   https://github.com/srirambandi/ai
-   https://discuss.pytorch.org/t/is-pytorch-autograd-tape-based/13992/3
-   https://www.reddit.com/r/MachineLearning/comments/8ep130/d_how_does_autograd_work/
-   https://github.com/mattjj/autodidact
-   https://github.com/karpathy/recurrentjs
-   https://github.com/karpathy/randomfun
-   https://medium.com/@ralphmao95/simple-autograd-implementation-understand-automatic-differentiation-hand-by-hand-9e86f6d703ab
-   https://evcu.github.io/ml/autograd/
-   https://blog.paperspace.com/pytorch-101-understanding-graphs-and-automatic-differentiation/
-   https://github.com/maciejkula/wyrm
-   https://medium.com/@maciejkula/building-an-autodifferentiation-library-9ccf32c7a658
-   https://github.com/evcu/numpy_autograd/blob/master/my_autograd.py#L147
-   https://github.com/evcu/numpy_autograd/blob/master/Autograd.ipynb
-   https://cs231n.github.io/optimization-2/
-   https://github.com/explosion/thinc
-   https://github.com/joelgrus/joelnet
-   https://github.com/QuantStack/xtensor
-   https://github.com/ThinkingTransistor/Sigma
-   https://github.com/mratsim/Arraymancer
-   https://github.com/siekmanj/sieknet
-   https://github.com/siekmanj/sieknet_2.0
-   https://github.com/Daniel-Liu-c0deb0t/Java-Machine-Learning
-   https://github.com/karpathy/micrograd

This README is based on:

-   https://github.com/bilal2vec/pytorch_zoo
-   https://github.com/bilal2vec/grover
-   https://github.com/rish-16/gpt2client
-   https://github.com/mxbi/mlcrate
-   https://github.com/athityakumar/colorls
-   https://github.com/amitmerchant1990/electron-markdownify

I used carbon.now.sh with the "Shades of Purple" theme for the screenshot at the beginning of this README

This project contains ~4300 lines of code
