<h1 align='center'>
    L2 • 🤖
</h1>

<h4 align='center'>
    A tensor, linear algebra, and deep learning library in C++, using only the standard library
</h4>

<p align='center'>
    <a href="https://forthebadge.com">
        <img src="https://forthebadge.com/images/badges/made-with-c-plus-plus.svg" alt="forthebadge">
    </a>
    <a href="https://github.com/prettier/prettier">
        <img src="https://img.shields.io/badge/code_style-prettier-ff69b4.svg?style=flat-square" alt="code style: prettier" />
    </a>
    <a href="https://opensource.org/licenses/MIT">
        <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
    </a>
    <a href="http://makeapullrequest.com">
        <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square" alt="PRs Welcome">
    </a>
    <a href="https://github.com/bkkaggle/L2/pulls">
        <img alt="GitHub pull requests" src="https://img.shields.io/github/issues-pr/bkkaggle/L2">
    </a>
</p>

<p align='center'>
    <a href='#installation'>Installation</a> •
    <a href='#documentation'>Documentation</a> •
    <a href='#contributing'>Contributing</a> •
    <a href='#authors'>Authors</a> •
    <a href='#license'>License</a>
</p>

<div>
    <img src="./screenshot.png" />
</div>

<p align='center'><strong>Made by <a href='https://github.com/bkkaggle'>Bilal Khan</a> • https://bkkaggle.github.io</strong></p>

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

-   [Quick start](#quick-start)
-   [Installation](#installation)
-   [Documentation](#documentation)
    -   [Tensor](#tensor)
    -   [Parameter](#parameter)
    -   [nn](#nn)
        -   [Linear](#linear)
        -   [Sigmoid](#sigmoid)
        -   [Sequential](#sequential)
    -   [Loss](#loss)
        -   [MSE](#mse)
    -   [Optimizer](#optimizer)
        -   [SGD](#sgd)
    -   [Trainer](#trainer)
-   [Contributing](#contributing)
-   [Authors](#authors)
-   [License](#license)
-   [Acknowledgements](#acknowledgements)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Quick start

```cpp
L2::Tensor<double> x = L2::Tensor<double>({100, 10}).normal(0, 1);

L2::Tensor<double> w = L2::Tensor<double>({10, 1}).normal(0, 1);
L2::Tensor<double> b = L2::Tensor<double>({1}).normal(0, 1);

L2::Tensor<double> y = L2::matmul(x, w) + b;

L2::nn::loss::MSE<double> *criterion = new L2::nn::loss::MSE<double>();
L2::nn::optimizer::SGD<double> *optimizer = new L2::nn::optimizer::SGD<double>(0.05);

L2::nn::Sequential<double> *sequential = new L2::nn::Sequential<double>({
    new L2::nn::Linear<double>(10, 1) //
});

L2::trainer::Trainer<double> trainer = L2::trainer::Trainer<double>(sequential, criterion, optimizer);

trainer.fit(x, y, 10, 10);

L2::Tensor<double> y_hat = trainer.predict(x);

y_hat.print();
```

# design choices

# Installation

-   main.cpp
-   as library

# Documentation

-   public methods
-   example per heading

#### [L2::Tensor\<double>({3, 3})](./include/tensor.h#L13)

#### [L2::Parameter\<double>(tensor)](./include/parameter.h#L8)

#### nn

[L2::\<double>()](./include/)

##### [L2::nn::Linear\<double>(c_in=32, c_out=64)](./include/nn.h#L14)

##### Sigmoid

##### Sequential

#### Loss

##### MSE

#### Optimizer

##### SGD

#### Trainer

# Contributing

This repository is still a work in progress, so if you find a bug, think there is something missing, or have any suggestions for new features, feel free to open an issue or a pull request. Feel free to use the library or code from it in your own projects, and if you feel that some code used in this project hasn't been properly accredited, please open an issue.

# Authors

-   _Bilal Khan_ - _Initial work_

# License

This project is licensed under the MIT License - see the [license](LICENSE) file for details

# Acknowledgements

The fast.ai deep learning from the foundations course (https://course.fast.ai/part2) teaches a lot about how to make your own deep learning library

Some of the blog posts I used when writing this library include:

-   http://blog.ezyang.com/2019/05/pytorch-internals/
-   https://pytorch.org/tutorials/beginner/nn_tutorial.html
-   https://eisenjulian.github.io/deep-learning-in-100-lines/
-   https://medium.com/@florian.caesar/how-to-create-a-machine-learning-framework-from-scratch-in-491-steps-93428369a4eb
-   https://medium.com/@johan.mabille/how-we-wrote-xtensor-1-n-n-dimensional-containers-f79f9f4966a7

Other deep learning libraries from scratch include:

-   https://github.com/explosion/thinc
-   https://github.com/joelgrus/joelnet
-   https://github.com/QuantStack/xtensor
-   https://github.com/ThinkingTransistor/Sigma

This README is based on:

-   https://github.com/bkkaggle/pytorch_zoo
-   https://github.com/bkkaggle/grover
-   https://github.com/rish-16/gpt2client
-   https://github.com/mxbi/mlcrate
-   https://github.com/athityakumar/colorls
-   https://github.com/amitmerchant1990/electron-markdownify

I used carbon.now.sh with the "Shades of Purple" theme for the screenshot at the beginning of this README
