#include <iostream>

#include "parameter.h"
#include "tensor.h"

namespace L2
{
template <class T>
Parameter<T>::Parameter() : tensor({1}), grad(tensor.get_shape()) {}

template <class T>
Parameter<T>::Parameter(Tensor<T> t) : tensor(t), grad(tensor.get_shape()) {}

template class Parameter<double>;
} // namespace L2