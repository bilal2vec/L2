#pragma once

#include "tensor.h"

namespace L2
{
template <class T>
Tensor<T> pow(Tensor<T> tensor, T exp)
{
    return tensor.pow(exp);
}

template <class T>
Tensor<T> sqrt(Tensor<T> tensor)
{
    return tensor.sqrt();
}

template <class T>
Tensor<T> exp(Tensor<T> tensor)
{
    return tensor.exp();
}

template <class T>
Tensor<T> log(Tensor<T> tensor)
{
    return tensor.log();
}

template <class T>
Tensor<T> log10(Tensor<T> tensor)
{
    return tensor.log10();
}

template <class T>
Tensor<T> abs(Tensor<T> tensor)
{
    return tensor.abs();
}

template <class T>
Tensor<T> sin(Tensor<T> tensor)
{
    return tensor.sin();
}

template <class T>
Tensor<T> cos(Tensor<T> tensor)
{
    return tensor.cos();
}

template <class T>
Tensor<T> tan(Tensor<T> tensor)
{
    return tensor.tan();
}

} // namespace L2
