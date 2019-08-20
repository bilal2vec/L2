#pragma once

#include "tensor.h"

namespace L2
{
template <class T>
class Parameter
{
public:
    L2::Tensor<T> tensor;
    L2::Tensor<T> grad;

    Parameter(L2::Tensor<T> t);
};
} // namespace L2