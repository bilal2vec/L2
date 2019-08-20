#pragma once

#include "tensor.h"

namespace L2
{
template <class T>
class Parameter
{
public:
    Tensor<T> tensor;
    Tensor<T> grad;

    Parameter();
    Parameter(Tensor<T> t);
};
} // namespace L2