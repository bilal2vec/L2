#include <vector>
#include <algorithm>

#include "nn.h"
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
    Layer<T>::cached_input = tensor;
    return L2::matmul(tensor, weights.tensor) + bias.tensor;
}

template <class T>
Tensor<T> Linear<T>::backward(Tensor<T> derivative)
{
    weights.grad += L2::matmul(Layer<T>::cached_input.transpose(), derivative);
    bias.grad += L2::sum(derivative, 0);

    return L2::matmul(derivative, weights.tensor.transpose());
}

template <class T>
Sequential<T>::Sequential(std::vector<Layer<T>> input)
{
    layers = input;

    // object slicing means that only the layer part of Linear gets saved
    // need to use polymorphism
    // virtual base class
    // for that, it means that we need to convert the vector of layers to vectors of pointers in the constructor to save it in layers and dereference the pointers to get the params

    // then in forward, go over each pointer, dereference it, polymorphism makes sure that the forward method we call is on linear even though in the for loop it is defined as a Layer (https://stackoverflow.com/questions/2391679/why-do-we-need-virtual-functions-in-c)
    for (Layer<T> layer : layers)
    {
        std::vector<Parameter<T>> params = layer.parameters;
        Layer<T>::parameters.insert(Layer<T>::parameters.end(), params.begin(), params.end());
    }
}

template <class T>
Tensor<T> Sequential<T>::forward(Tensor<T> tensor)
{
    for (Layer<T> &layer : layers)
    {
        Linear<T> layer2 = static_cast<Linear<T> &>(layer);
        tensor = layer2.forward(tensor);
    }

    return tensor;
}

template <class T>
Tensor<T> Sequential<T>::backward(Tensor<T> derivative)
{
    return derivative;
}

template class Linear<double>;
template class Sequential<double>;
} // namespace L2::nn