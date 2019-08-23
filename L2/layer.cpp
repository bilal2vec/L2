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

template class Layer<double>;
} // namespace L2