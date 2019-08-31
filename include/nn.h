#pragma once

#include <iostream>
#include <vector>

#include "tensor.h"
#include "parameter.h"
#include "layer.h"
#include "optimizer.h"

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

    void update(L2::nn::optimizer::Optimizer<T> *optimizer);
};

template <class T>
class Sigmoid : public Layer<T>
{
public:
    Sigmoid();

    Tensor<T> forward(Tensor<T> tensor);
    Tensor<T> backward(Tensor<T> derivative);

    void update(L2::nn::optimizer::Optimizer<T> *optimizer);
};

template <class T>
class Sequential
{
private:
    std::vector<Layer<T> *> layers;

public:
    Sequential(std::vector<Layer<T> *> layers);
    ~Sequential();

    Tensor<T> forward(Tensor<T> tensor);
    Tensor<T> backward(Tensor<T> derivative);

    void update(L2::nn::optimizer::Optimizer<T> *optimizer);
};

} // namespace L2::nn