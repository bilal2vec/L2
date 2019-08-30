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

public:
    Trainer(L2::nn::Sequential<T> *model, L2::nn::loss::Loss<T> *loss, L2::nn::optimizer::Optimizer<T> *optimizer);

    void fit(L2::Tensor<T> x, L2::Tensor<T> y, int epochs, int batch_size);

    L2::Tensor<double> fit_batch(L2::Tensor<T> x, L2::Tensor<T> y);
};
} // namespace L2::trainer