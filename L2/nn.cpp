#include <vector>
#include <algorithm>

#include "nn.h"
#include "math.h"
#include "initialization.h"

namespace L2::nn
{
template <class T>
Linear<T>::Linear(int c_in, int c_out)
{
    Tensor<T> w = Tensor<T>({c_in, c_out});
    Tensor<T> b = Tensor<T>({c_out});

    w = L2::nn::init::kaiming_uniform(w, c_in);
    b = L2::nn::init::zeros(b);

    weights = Layer<T>::build_param(w);
    bias = Layer<T>::build_param(b);
}

template <class T>
Tensor<T> Linear<T>::forward(Tensor<T> tensor)
{
    Layer<T>::cached = tensor;
    return L2::matmul(tensor, weights.tensor) + bias.tensor;
}

template <class T>
Tensor<T> Linear<T>::backward(Tensor<T> derivative)
{
    weights.grad += L2::matmul(Layer<T>::cached.transpose(), derivative);
    bias.grad += L2::sum(derivative, 0);

    return L2::matmul(derivative, weights.tensor.transpose());
}

template <class T>
Sigmoid<T>::Sigmoid() {}

template <class T>
Tensor<T> Sigmoid<T>::forward(Tensor<T> tensor)
{
    L2::Tensor<T> S = L2::pow(L2::exp(tensor * -1.0) + 1, -1.0);
    Layer<T>::cached = S;
    return S;
}

template <class T>
Tensor<T> Sigmoid<T>::backward(Tensor<T> derivative)
{
    return derivative * Layer<T>::cached * ((Layer<T>::cached - 1) * -1.0);
}

template <class T>
Sequential<T>::Sequential(std::vector<Layer<T> *> input)
{
    layers = input;

    for (Layer<T> *layer : layers)
    {
        std::vector<Parameter<T>> params = layer->parameters;
        Layer<T>::parameters.insert(Layer<T>::parameters.end(), params.begin(), params.end());
    }
}

template <class T>
Sequential<T>::~Sequential()
{
    for (auto layer : layers)
    {
        delete layer;
    }
}

template <class T>
Tensor<T> Sequential<T>::forward(Tensor<T> tensor)
{
    for (Layer<T> *layer : layers)
    {
        tensor = layer->forward(tensor);
    }

    return tensor;
}

template <class T>
Tensor<T> Sequential<T>::backward(Tensor<T> derivative)
{
    std::vector<Layer<T> *> reverse_layers(layers.rbegin(), layers.rend());

    for (Layer<T> *layer : reverse_layers)
    {
        derivative = layer->backward(derivative);
    }

    return derivative;
}

template class Linear<double>;
template class Sigmoid<double>;

template class Sequential<double>;
} // namespace L2::nn