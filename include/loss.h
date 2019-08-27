#pragma once

#include <iostream>

#include "tensor.h"

namespace L2::nn::loss
{

template <class T>
class Loss
{
protected:
    Tensor<T> cached;

public:
    virtual Tensor<T> forward(Tensor<T> pred, Tensor<T> label)
    {
        return pred;
    }

    virtual Tensor<T> backward()
    {
        return Tensor<T>();
    }
};

template <class T>
class MSE : public Loss<T>
{
public:
    Tensor<T> forward(Tensor<T> pred, Tensor<T> label);
    Tensor<T> backward();
};

} // namespace L2::nn::loss