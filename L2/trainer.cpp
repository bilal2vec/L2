#include "trainer.h"

#include "tensor.h"
#include "nn.h"
#include "loss.h"
#include "optimizer.h"

namespace L2::trainer
{
template <class T>
Trainer<T>::Trainer(L2::nn::Sequential<T> *model, L2::nn::loss::Loss<T> *loss, L2::nn::optimizer::Optimizer<T> *optimizer) : sequential(model), criterion(loss), optim(optimizer) {}

template <class T>
void Trainer<T>::fit(L2::Tensor<T> x, L2::Tensor<T> y, int epochs, int batch_size)
{
    // number of elements in x
    int length = x({{0, -1}, {0, 1}}).length();
    int n_batches = int(length / batch_size);

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        L2::Tensor<double> loss;

        for (int batch = 0; batch < n_batches; ++batch)
        {
            int start = batch * batch_size;
            int stop = start + batch_size;

            L2::Tensor<T> x_batch = x({{start, stop}});
            L2::Tensor<T> y_batch = y({{start, stop}});

            // change shape of (batch_size) to (batch_size, 1)
            if (y_batch.get_shape().size() == 1)
            {
                y_batch.view_({batch_size, y.get_shape()[1]});
            }

            loss += fit_batch(x_batch, y_batch);
        }

        loss /= n_batches;

        std::cout << "Epoch: " << epoch << " | loss: " << loss.get_data()[0] << "\n";
    }
}

template <class T>
L2::Tensor<double> Trainer<T>::fit_batch(L2::Tensor<T> x, L2::Tensor<T> y)
{
    L2::Tensor<double> y_hat = sequential->forward(x);

    L2::Tensor<double> loss = criterion->forward(y_hat, y);
    L2::Tensor<double> derivative = criterion->backward();

    L2::Tensor<double> zz = sequential->backward(derivative);

    sequential->update(optim);

    return loss;
}

template class Trainer<double>;
} // namespace L2::trainer