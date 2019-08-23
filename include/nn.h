#pragma once

#include <iostream>
#include <vector>

#include "layer.h"
#include "parameter.h"

namespace L2::nn
{
template <class T>
class Linear : public Layer<T>
{
private:
    Parameter<T> weights;
    Parameter<T> bias;

public:
    Linear(int c_in, int c_out);

    Tensor<T> forward(Tensor<T> tensor);
    Tensor<T> backward(Tensor<T> derivative);
};

template <class T>
class Sequential : public Layer<T>
{
private:
    std::vector<Layer<T>> layers;

public:
    Sequential(std::vector<Layer<T>> layers);

    Tensor<T> forward(Tensor<T> tensor);
    Tensor<T> backward(Tensor<T> derivative);
};

} // namespace L2::nn