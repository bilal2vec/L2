#include <iostream>
#include <vector>
#include <cmath>     // std::abs
#include <algorithm> //std::max
#include <tuple>     //std::tuple
#include <string>

#include "tensor.h"

namespace L2
{
template <class T>
int Tensor<T>::get_physical_idx(std::vector<int> indices)
{
    int physical_idx = 0;

    for (int i = 0; i < static_cast<int>(indices.size()); ++i)
    {
        physical_idx += (indices[i] * strides[i]);
    }

    return physical_idx;
}

template <class T>
int Tensor<T>::get_physical_idx(std::vector<int> indices, std::vector<int> strides)
{
    int physical_idx = 0;

    for (int i = 0; i < static_cast<int>(indices.size()); ++i)
    {
        physical_idx += (indices[i] * strides[i]);
    }

    return physical_idx;
}

template <class T>
int Tensor<T>::sizes_to_n_elements(std::vector<int> sizes)
{
    int sum = 1;

    for (int size : sizes)
    {
        sum *= size;
    }

    return sum;
}

template <class T>
bool Tensor<T>::valid_sizes(std::vector<int> new_sizes)
{
    int current_sizes_sum = sizes_to_n_elements(sizes);
    int new_sizes_sum = sizes_to_n_elements(new_sizes);

    if (current_sizes_sum == new_sizes_sum)
    {
        return true;
    }
    else
    {
        return false;
    }
}

template <class T>
std::vector<index> Tensor<T>::process_index(std::vector<index> indices)
{
    for (int i = 0; i < indices.size(); ++i)
    {
        if (indices[i].stop == -1)
        {
            indices[i].stop = sizes[i];
        }
    }

    return indices;
}

template <class T>
std::vector<int> Tensor<T>::get_sizes(std::vector<index> indices)
{
    std::vector<int> slice_sizes;

    for (index idx : indices)
    {
        // Add number of elements in dimension to slice_sizes if not 1
        if ((idx.stop - idx.start) > 1)
        {
            slice_sizes.push_back(idx.stop - idx.start);
        }
    }

    // slice_size is 1 if selecting single element from tensor
    if (static_cast<int>(slice_sizes.size()) == 0)
    {
        slice_sizes.push_back(1);
    }

    return slice_sizes;
}

template <class T>
std::vector<int> Tensor<T>::get_strides(std::vector<int> sizes)
{
    std::vector<int> strides;

    // from: https://github.com/ThinkingTransistor/Sigma/blob/fe645441eb523996d3bfc6de9ad72e814a146195/Sigma.Core/MathAbstract/NDArrayUtils.cs#L24
    int current_stride = 1;
    for (int i = static_cast<int>(sizes.size()) - 1; i >= 0; --i)
    {
        strides.insert(strides.begin(), current_stride);
        current_stride *= sizes[i];
    }
    return strides;
}

template <class T>
std::vector<int> Tensor<T>::expand_sizes(std::vector<int> size, int size_diff)
{
    for (int i = 0; i < size_diff; ++i)
    {
        size.insert(size.begin(), 1);
    }
    return size;
}

template <class T>
std::vector<int> Tensor<T>::broadcast_sizes(std::vector<int> sizes_a, std::vector<int> sizes_b)
{
    // choose max of each element in sizes as new size
    std::vector<int> new_sizes;
    for (int i = 0; i < sizes_a.size(); ++i)
    {
        new_sizes.push_back(std::max(sizes_a[i], sizes_b[i]));
    }

    return new_sizes;
}

template <class T>
std::tuple<std::vector<int>, std::vector<int>> Tensor<T>::broadcast_strides(std::vector<int> lhs_sizes, std::vector<int> rhs_sizes, std::vector<int> new_sizes)
{

    std::vector<int> lhs_strides = get_strides(lhs_sizes);
    std::vector<int> rhs_strides = get_strides(rhs_sizes);

    for (int i = 0; i < new_sizes.size(); ++i)
    {
        if (lhs_sizes[i] != new_sizes[i])
        {
            lhs_strides[i] = 0;
        }

        if (rhs_sizes[i] != new_sizes[i])
        {
            rhs_strides[i] = 0;
        }
    }

    return {lhs_strides, rhs_strides};
}

template <class T>
T Tensor<T>::operation(T lhs, T rhs, std::string op)
{
    if (op == "+")
    {
        return lhs + rhs;
    }
    else if (op == "-")
    {
        return lhs - rhs;
    }
    else if (op == "*")
    {
        return lhs * rhs;
    }
    else if (op == "/")
    {
        return lhs / rhs;
    }
}

template <class T>
Tensor<T> Tensor<T>::tensor_elementwise_op(Tensor<T> other, std::string op)
{
    // expand lhs and rhs sizes to have the same number of elements
    int size_diff = size().size() - other.size().size();

    std::vector<int> lhs_sizes = (size_diff < 0) ? expand_sizes(size(), std::abs(size_diff)) : size();
    std::vector<int> rhs_sizes = (size_diff > 0) ? expand_sizes(other.size(), std::abs(size_diff)) : other.size();

    // broadcast lhs_sizes and rhs_sizes to get the broadcasted size
    std::vector<int> new_sizes = broadcast_sizes(lhs_sizes, rhs_sizes);

    // update strides and zero out broadcasted dimensions
    std::vector<int> lhs_strides;
    std::vector<int> rhs_strides;
    std::tie(lhs_strides, rhs_strides) = broadcast_strides(lhs_sizes, rhs_sizes, new_sizes);

    // perform operation on data element wise and save
    std::vector<T> new_data;

    int length = static_cast<int>(new_sizes.size());
    for (int i = 0; i < new_sizes[0]; ++i)
    {
        if (length > 1)
        {
            for (int j = 0; j < new_sizes[1]; ++j)
            {
                if (length > 2)
                {
                    for (int k = 0; k < new_sizes[2]; ++k)
                    {
                        if (length > 3)
                        {
                            for (int m = 0; m < new_sizes[3]; ++m)
                            {
                                T op_result = operation(data[get_physical_idx({i, j, k, m}, lhs_strides)], other.data[get_physical_idx({i, j, k, m}, rhs_strides)], op);
                                new_data.push_back(op_result);
                            }
                        }
                        else
                        {
                            T op_result = operation(data[get_physical_idx({i, j, k}, lhs_strides)], other.data[get_physical_idx({i, j, k}, rhs_strides)], op);
                            new_data.push_back(op_result);
                        }
                    }
                }
                else
                {
                    T op_result = operation(data[get_physical_idx({i, j}, lhs_strides)], other.data[get_physical_idx({i, j}, rhs_strides)], op);
                    new_data.push_back(op_result);
                }
            }
        }
        else
        {
            T op_result = operation(data[get_physical_idx({i}, lhs_strides)], other.data[get_physical_idx({i}, rhs_strides)], op);
            new_data.push_back(op_result);
        }
    }

    return Tensor<T>(new_data, new_sizes);
}

template <class T>
Tensor<T> Tensor<T>::scalar_elementwise_op(T other, std::string op)
{
    std::vector<T> new_data;

    for (int i = 0; i < data.size(); ++i)
    {
        new_data.push_back(operation(data[i], other, op));
    }

    return Tensor<T>(new_data, sizes);
}

template <class T>
Tensor<T>::Tensor(std::vector<int> sizes) : data(sizes_to_n_elements(sizes), 0), sizes(sizes), strides(get_strides(sizes)) {}

template <class T>
Tensor<T>::Tensor(std::vector<T> x, std::vector<int> sizes) : data(x), sizes(sizes), strides(get_strides(sizes)) {}

// tensor indexing and slicing
template <class T>
Tensor<T> Tensor<T>::operator()(std::vector<index> indices)
{
    indices = process_index(indices);

    std::vector<T> slice;
    std::vector<int> slice_sizes = get_sizes(indices);

    int length = static_cast<int>(sizes.size());

    // quick hardcoded solution for up to 4 dimensions
    for (int i = indices[0].start; i < indices[0].stop; ++i)
    {
        if (length > 1)
        {
            for (int j = indices[1].start; j < indices[1].stop; ++j)
            {
                if (length > 2)
                {
                    for (int k = indices[2].start; k < indices[2].stop; ++k)
                    {
                        if (length > 3)
                        {
                            for (int m = indices[3].start; m < indices[3].stop; ++m)
                            {
                                slice.push_back(data[get_physical_idx({i, j, k, m})]);
                            }
                        }
                        else
                        {
                            slice.push_back(data[get_physical_idx({i, j, k})]);
                        }
                    }
                }
                else
                {
                    slice.push_back(data[get_physical_idx({i, j})]);
                }
            }
        }
        else
        {
            slice.push_back(data[get_physical_idx({i})]);
        }
    }
    return Tensor<T>(slice, slice_sizes);
}

template <class T>
Tensor<T> Tensor<T>::operator+(T other)
{
    return scalar_elementwise_op(other, "+");
}

template <class T>
Tensor<T> Tensor<T>::operator+(Tensor<T> other)
{
    return tensor_elementwise_op(other, "+");
}

template <class T>
Tensor<T> Tensor<T>::operator-(T other)
{
    return scalar_elementwise_op(other, "-");
}

template <class T>
Tensor<T> Tensor<T>::operator-(Tensor<T> other)
{
    return tensor_elementwise_op(other, "-");
}

template <class T>
Tensor<T> Tensor<T>::operator*(T other)
{
    return scalar_elementwise_op(other, "*");
}

template <class T>
Tensor<T> Tensor<T>::operator*(Tensor<T> other)
{
    return tensor_elementwise_op(other, "*");
}

template <class T>
Tensor<T> Tensor<T>::operator/(T other)
{
    return scalar_elementwise_op(other, "/");
}

template <class T>
Tensor<T> Tensor<T>::operator/(Tensor<T> other)
{
    return tensor_elementwise_op(other, "/");
}

template <class T>
Tensor<T> Tensor<T>::normal_(double mean, double stddev)
{
    std::random_device random_device{};
    std::mt19937 pseudorandom_generator{random_device()};

    std::normal_distribution<double> distribution{mean, stddev};

    std::vector<T> normal_tensor;

    for (int i = 0; i < data.size(); ++i)
    {
        double sample = distribution(pseudorandom_generator);

        normal_tensor.push_back(sample);
    }

    return Tensor<T>(normal_tensor, sizes);
}

template <class T>
Tensor<T> Tensor<T>::uniform_(double low, double high)
{
    std::random_device random_device{};
    std::mt19937 pseudorandom_generator{random_device()};

    std::uniform_real_distribution<double> distribution{low, high};

    std::vector<T> uniform_tensor;

    for (int i = 0; i < data.size(); ++i)
    {
        double sample = distribution(pseudorandom_generator);

        uniform_tensor.push_back(sample);
    }

    return Tensor<T>(uniform_tensor, sizes);
}

template <class T>
void Tensor<T>::view(std::vector<int> new_sizes)
{
    if (valid_sizes(new_sizes))
    {
        sizes = new_sizes;
        strides = get_strides(sizes);
    }
}

template <class T>
std::vector<int> Tensor<T>::size()
{
    return sizes;
}

template <class T>
void Tensor<T>::print()
{
    std::cout << "data:\n";
    _print(data);
    std::cout << "size:\n";
    _print(sizes);
    std::cout << "strides:\n";
    _print(strides);
}

template class Tensor<int>;
template class Tensor<float>;
template class Tensor<double>;
}