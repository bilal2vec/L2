#pragma once

#include <iostream>
#include <vector>

#include "tensor.h"
#include "parameter.h"
#include "optimizer.h"

namespace L2
{
template <class T>
class Layer
{
protected:
    Tensor<T> cached;

public:
    std::vector<Parameter<T>> parameters;

    Parameter<T> build_param(Tensor<T> tensor);

    // override in all derived classes
    virtual void update(L2::nn::optimizer::Optimizer<T> *optimizer)
    {
        for (int i = 0; i < parameters.size(); ++i)
        {
            parameters[i] = optimizer->update(parameters[i]);
        }
    }

    // override in all derived classes
    virtual Tensor<T> forward(Tensor<T> tensor)
    {
        return tensor;
    }

    // override in all derived classes
    virtual Tensor<T> backward(Tensor<T> derivative)
    {
        return derivative;
    }
};
} // namespace L2