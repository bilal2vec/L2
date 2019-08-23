#include "layer.h"

#include "tensor.h"
#include "parameter.h"

namespace L2
{
template <class T>
Parameter<T> Layer<T>::build_param(Tensor<T> tensor)
{
    Parameter<T> param = Parameter<T>(tensor);
    parameters.push_back(param);

    return param;
}

template <class T>
void Layer<T>::update()
{
    1 + 1;
}

// Override in all derived classes
template <class T>
Tensor<T> Layer<T>::forward(Tensor<T> tensor)
{
    return tensor;
}

// Override in all derived classes
template <class T>
Tensor<T> Layer<T>::backward(Tensor<T> derivative)
{
    return derivative;
}

template class Layer<double>;
} // namespace L2