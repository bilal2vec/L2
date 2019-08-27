#include "loss.h"

#include "tensor.h"
#include "math.h"

namespace L2::nn::loss
{

template <class T>
Tensor<T> MSE<T>::forward(Tensor<T> pred, Tensor<T> label)
{
    Loss<T>::cached = pred - label;

    return L2::pow(Loss<T>::cached, 2.0).mean();
}

template <class T>
Tensor<T> MSE<T>::backward()
{
    return (Loss<T>::cached * 2) / Loss<T>::cached.length();
}

template class MSE<double>;
} // namespace L2::nn::loss