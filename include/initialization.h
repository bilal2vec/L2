#pragma once

#include <iostream>
#include <cmath> // std::sqrt

#include "tensor.h"

namespace L2::nn::init
{

template <class T>
Tensor<T> zeros(Tensor<T> tensor)
{
    return tensor.zeros();
}

template <class T>
Tensor<T> kaiming_uniform(Tensor<T> tensor, int c_in)
{
    double bounds = std::sqrt(6.0 / c_in);
    return tensor.uniform(-bounds, bounds);
}

template <class T>
Tensor<T> kaiming_normal(Tensor<T> tensor, int c_in)
{
    double stddev = std::sqrt(2.0 / c_in);
    return tensor.normal(0, stddev);
}

template <class T>
Tensor<T> xavier_uniform(Tensor<T> tensor, int c_in, int c_out)
{
    double bounds = std::sqrt(6.0 / (c_in + c_out));
    return tensor.uniform(-bounds, bounds);
}

template <class T>
Tensor<T> xavier_normal(Tensor<T> tensor, int c_in, int c_out)
{
    double stddev = std::sqrt(2.0 / (c_in + c_out));
    return tensor.normal(0, stddev);
}

} // namespace L2::nn::init