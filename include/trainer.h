#pragma once

#include <iostream>

#include "nn.h"
#include "loss.h"
#include "optimizer.h"

namespace L2::trainer
{

template <class T>
class Trainer
{
private:
    L2::nn::Sequential<T> *sequential;
    L2::nn::loss::Loss<T> *criterion;
    L2::nn::optimizer::Optimizer<T> *optim;

    Tensor<double> fit_batch(Tensor<T> x, L2::Tensor<T> y);

public:
    Trainer(L2::nn::Sequential<T> *model, L2::nn::loss::Loss<T> *loss, L2::nn::optimizer::Optimizer<T> *optimizer);

    void fit(Tensor<T> x, Tensor<T> y, int epochs, int batch_size);

    Tensor<T> predict(Tensor<T> x);
};
} // namespace L2::trainer