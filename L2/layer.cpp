#include "layer.h"

#include "tensor.h"
#include "parameter.h"
#include "optimizer.h"

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
void Layer<T>::update(L2::nn::optimizer::Optimizer<T> *optimizer)
{
    for (int i = 0; i < parameters.size(); ++i)
    {
        parameters[i] = optimizer->update(parameters[i]);
    }
}

template class Layer<double>;
} // namespace L2