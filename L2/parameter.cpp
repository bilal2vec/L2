#include <iostream>

#include "parameter.h"
#include "tensor.h"

namespace L2
{
template <class T>
Parameter<T>::Parameter(L2::Tensor<T> t) : tensor(t), grad(t.get_shape()) {}

template class Parameter<double>;
} // namespace L2