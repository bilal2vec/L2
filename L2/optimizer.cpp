#include "optimizer.h"

#include "parameter.h"

namespace L2::nn::optimizer
{

template <class T>
SGD<T>::SGD(double lr)
{
    Optimizer<T>::lr = lr;
}

template <class T>
Parameter<T> SGD<T>::update(Parameter<T> param)
{
    param.tensor -= param.grad * Optimizer<T>::lr;
    param.reset();

    return param;
}

template class Optimizer<double>;
template class SGD<double>;
} // namespace L2::nn::optimizer
