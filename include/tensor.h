#pragma once

#include <vector>
#include <random>
#include <tuple>
#include <string>

#include "index.h"

namespace L2
{
template <class T>
class Tensor
{
private:
    std::vector<T> data;
    std::vector<int> shape;
    std::vector<int> strides;

    template <typename T2>
    void _print(std::vector<T2> x)
    {
        std::cout << "\n";

        for (T2 i : x)
        {
            std::cout << i << ", ";
        }

        std::cout << "\n\n";
    }

    int get_physical_idx(std::vector<int> indices);
    int get_physical_idx(std::vector<int> indices, std::vector<int> strides);
    int shape_to_n_elements(std::vector<int> shape);
    bool valid_shape(std::vector<int> shape, std::vector<int> new_shape);

    std::vector<index> process_index(std::vector<index> indices, std::vector<int> shape);
    std::vector<index> process_dims(std::vector<index> indices, int dim, int current_dim, int i);

    std::vector<int> get_shape(std::vector<index> indices);
    std::vector<int> get_strides(std::vector<int> shape);

    std::vector<int> expand_shape(std::vector<int> shape, int diff);
    std::vector<int> broadcast_shape(std::vector<int> shape_a, std::vector<int> shape_b);

    std::tuple<std::vector<int>, std::vector<int>> broadcast_strides(std::vector<int> lhs_shape, std::vector<int> rhs_shape, std::vector<int> new_shape);

    T operation(T lhs, T rhs, std::string op);
    T operation(T lhs, std::string op);

    T sum(Tensor<T> tensor);
    T dimension_operation(Tensor<T> tensor, std::string op);

    Tensor<T> tensor_elementwise_op(Tensor<T> other, std::string op);
    Tensor<T> scalar_elementwise_op(T other, std::string op);
    Tensor<T> tensor_op(std::string op);
    Tensor<T> dimension_op(int dim, std::string op);

public:
    Tensor(std::vector<int> shape);
    Tensor(std::vector<T> x, std::vector<int> shape);

    Tensor<T> operator()(std::vector<index> indices);

    Tensor<T> operator+(T other);
    Tensor<T> operator+(Tensor<T> other);

    Tensor<T> operator-(T other);
    Tensor<T> operator-(Tensor<T> other);

    Tensor<T> operator*(T other);
    Tensor<T> operator*(Tensor<T> other);

    Tensor<T> operator/(T other);
    Tensor<T> operator/(Tensor<T> other);

    Tensor<T> pow(T other);

    Tensor<T> sqrt();
    Tensor<T> exp();
    Tensor<T> log();
    Tensor<T> log10();

    Tensor<T> abs();
    Tensor<T> sin();
    Tensor<T> cos();
    Tensor<T> tan();

    Tensor<T> sum();
    Tensor<T> sum(int dim);

    Tensor<T> mean();
    Tensor<T> mean(int dim);

    Tensor<T> max();
    Tensor<T> max(int dim);

    Tensor<T> min();
    Tensor<T> min(int dim);

    Tensor<T> argmax();
    Tensor<T> argmax(int dim);

    Tensor<T> argmin();
    Tensor<T> argmin(int dim);

    Tensor<T> cat(Tensor<T> other, int dim);

    Tensor<T> normal_(double mean = 0, double stddev = 1);
    Tensor<T> uniform_(double low = 0, double high = 1);

    void view(std::vector<int> new_shape);
    Tensor<T> clone();
    std::vector<int> get_shape();
    std::vector<T> get_data();

    std::string type();

    void print();
};

template <class T>
Tensor<T> sum(Tensor<T> tensor)
{
    return tensor.sum();
}

template <class T>
Tensor<T> sum(Tensor<T> tensor, int dim)
{
    return tensor.sum(dim);
}

template <class T>
Tensor<T> mean(Tensor<T> tensor)
{
    return tensor.mean();
}

template <class T>
Tensor<T> mean(Tensor<T> tensor, int dim)
{
    return tensor.mean(dim);
}

template <class T>
Tensor<T> max(Tensor<T> tensor)
{
    return tensor.max();
}

template <class T>
Tensor<T> max(Tensor<T> tensor, int dim)
{
    return tensor.max(dim);
}

template <class T>
Tensor<T> min(Tensor<T> tensor)
{
    return tensor.min();
}

template <class T>
Tensor<T> min(Tensor<T> tensor, int dim)
{
    return tensor.min(dim);
}

template <class T>
Tensor<T> argmax(Tensor<T> tensor)
{
    return tensor.argmax();
}

template <class T>
Tensor<T> argmax(Tensor<T> tensor, int dim)
{
    return tensor.argmax(dim);
}

template <class T>
Tensor<T> argmin(Tensor<T> tensor)
{
    return tensor.argmin();
}

template <class T>
Tensor<T> argmin(Tensor<T> tensor, int dim)
{
    return tensor.argmin(dim);
}

template <class T>
Tensor<T> cat(std::vector<Tensor<T>> tensors, int dim)
{
    // check that all tensors should have the same dimensions except in the dimension dim and same number of dimensions
    //

    std::vector<T> new_data;
    //get new shape
    std::vector<int> new_shape = tensors[0].get_shape();

    for (int i = 1; i < tensors.size(); ++i)
    {
        new_shape[dim] += tensors[i].get_shape()[dim];
    }

    int length = static_cast<int>(new_shape.size());
    std::vector<index> idxs;

    for (int i = 0; i < length; ++i)
    {
        idxs.push_back({0, 0});
    }

    for (int i = 0; i < (dim == 0 ? 1 : new_shape[0]); ++i)
    {
        if (dim == 0)
        {

            idxs[0] = {0, i + 1};
            for (Tensor<T> tensor : tensors)
            {
                std::vector<T> z = tensor.get_data();

                new_data.insert(new_data.end(), z.begin(), z.end());
            }
        }
        else
        {
            idxs[0] = {i, i + 1};
            for (int j = 0; j < (dim == 1 ? 1 : new_shape[1]); ++j)
            {
                if (dim == 1)
                {
                    idxs[1] = {0, -1};

                    for (Tensor<T> tensor : tensors)
                    {
                        // get row
                        std::vector<T> z = tensor(idxs).get_data();

                        // insert at end of row
                        new_data.insert(new_data.end(), z.begin(), z.end());
                    }
                }
                else
                {
                }
            }
        }
    }

    return Tensor<T>(new_data, new_shape);

    // go over new shape
    //      get slice from

    // insert elements into vector, expanding it

    // if dim is dim
    //      select row/col of each tensor {0, -1} on that dim
    //      concat data and append

    // concating axis 1
    //

    // cat all elements in dim of tensors

    //dim 0 means to add the array after the current one
    // dim1 means to add the first row of the array after the current one
}
} // namespace L2
