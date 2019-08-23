#pragma once

#include <iostream>
#include <vector>

#include "tensor.h"
#include "parameter.h"

namespace L2
{
template <class T>
class Layer
{
protected:
    Tensor<T> cached_input;

public:
    std::vector<Parameter<T>> parameters;

    Parameter<T> build_param(Tensor<T> tensor);
    void update(); // IMPLEMENT after optimizer

    // Override in all derived classes
    Tensor<T> forward(Tensor<T> tensor);
    // Override in all derived classes
    Tensor<T> backward(Tensor<T> derivative);
};
} // namespace L2