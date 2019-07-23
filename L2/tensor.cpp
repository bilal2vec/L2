#include <iostream>
#include <vector>
#include <cmath>     // std::abs
#include <algorithm> //std::max

#include "tensor.h"

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
bool Tensor<T>::broadcastable_with(std::vector<int> new_sizes)
{
    std::vector<int> current_sizes = size();
    int size_diff = size().size() - new_sizes.size();

    // expand current_sizes to match length of new_sizes
    if (size_diff > 0)
    {
        new_sizes = expand_sizes(new_sizes, std::abs(size_diff));
    }
    // expand new_sizes to match length of current_sizes
    else if (size_diff < 0)
    {
        current_sizes = expand_sizes(size(), std::abs(size_diff));
    }

    for (int i = current_sizes.size() - 1; i >= 0; --i)
    {
        if ((current_sizes[i] != 1) && (new_sizes[i] != 1) && (current_sizes[i] != new_sizes[i]))
        {
            return false;
        }
    }
    return true;
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
    std::vector<T> new_data;

    for (int i = 0; i < data.size(); ++i)
    {
        new_data.push_back(data[i] + other);
    }

    return Tensor<T>(new_data, sizes);
}

template <class T>
Tensor<T> Tensor<T>::operator+(Tensor<T> other)
{
    if (size() == other.size())
    {
        std::vector<T> new_data;

        for (int i = 0; i < data.size(); ++i)
        {
            new_data.push_back(data[i] + other.data[i]);
        }

        return Tensor<T>(new_data, sizes);
    }
    else if (broadcastable_with(other.size()))
    {
        // stride should be 0 on broadcasted dimensions

        // find what the size of the resulting tensor should be
        // clone both tensors a and b
        // change their sizes to match the size of the resulting tensor
        // change the elements of strides that have been broadcasted in sizes to be 0
        // create a new vector to hold the results
        // iterate over the new sizes
        // for each lowest level for loop, use get_physical_idx to convert the symbolic idxs to actual idxs and add the elements from both tensors
        // return a new tensor

        Tensor<T> a = Tensor<T>(data, size());
        Tensor<T> b = Tensor<T>(other.data, other.size());

        int size_diff = a.size().size() - b.size().size();

        // expand current_sizes to match length of new_sizes
        if (size_diff > 0)
        {
            b.sizes = expand_sizes(b.size(), std::abs(size_diff));
        }
        // expand new_sizes to match length of current_sizes
        else if (size_diff < 0)
        {
            a.sizes = expand_sizes(a.size(), std::abs(size_diff));
        }

        a.strides = get_strides(a.sizes);
        b.strides = get_strides(b.sizes);

        std::vector<int> new_size = broadcast_sizes(a.size(), b.size());

        for (int i = 0; i < new_size.size(); ++i)
        {
            if (a.size()[i] != new_size[i])
            {
                a.strides[i] = 0;
            }

            if (b.size()[i] != new_size[i])
            {
                b.strides[i] = 0;
            }
        }

        a.sizes = new_size;
        b.sizes = new_size;

        std::vector<T> new_data;

        int length = static_cast<int>(new_size.size());
        for (int i = 0; i < new_size[0]; ++i)
        {
            if (length > 1)
            {
                for (int j = 0; j < new_size[1]; ++j)
                {
                    if (length > 2)
                    {
                        for (int k = 0; k < new_size[2]; ++k)
                        {
                            if (length > 3)
                            {
                                for (int m = 0; m < new_size[3]; ++m)
                                {
                                    new_data.push_back(a.data[a.get_physical_idx({i, j, k, m})] + b.data[b.get_physical_idx({i, j, k, m})]);
                                }
                            }
                            else
                            {
                                new_data.push_back(a.data[a.get_physical_idx({i, j, k})] + b.data[b.get_physical_idx({i, j, k})]);
                            }
                        }
                    }
                    else
                    {
                        new_data.push_back(a.data[a.get_physical_idx({i, j})] + b.data[b.get_physical_idx({i, j})]);
                    }
                }
            }
            else
            {
                new_data.push_back(a.data[a.get_physical_idx({i})] + b.data[b.get_physical_idx({i})]);
            }
        }

        return Tensor<T>(new_data, new_size);
    }
    else
    {
        std::cout << "False";
        return Tensor<T>(data, sizes);
    }
}

template <class T>
Tensor<T> Tensor<T>::operator-(T other)
{
    std::vector<T> new_data;

    for (int i = 0; i < data.size(); ++i)
    {
        new_data.push_back(data[i] - other);
    }

    return Tensor<T>(new_data, sizes);
}

template <class T>
Tensor<T> Tensor<T>::operator-(Tensor<T> other)
{
    if (size() == other.size())
    {
        std::vector<T> new_data;

        for (int i = 0; i < data.size(); ++i)
        {
            new_data.push_back(data[i] - other.data[i]);
        }

        return Tensor<T>(new_data, sizes);
    }
}

template <class T>
Tensor<T> Tensor<T>::operator*(T other)
{
    std::vector<T> new_data;

    for (int i = 0; i < data.size(); ++i)
    {
        new_data.push_back(data[i] * other);
    }

    return Tensor<T>(new_data, sizes);
}

template <class T>
Tensor<T> Tensor<T>::operator*(Tensor<T> other)
{
    if (size() == other.size())
    {
        std::vector<T> new_data;

        for (int i = 0; i < data.size(); ++i)
        {
            new_data.push_back(data[i] * other.data[i]);
        }

        return Tensor<T>(new_data, sizes);
    }
}

template <class T>
Tensor<T> Tensor<T>::operator/(T other)
{
    std::vector<T> new_data;

    for (int i = 0; i < data.size(); ++i)
    {
        new_data.push_back(data[i] / other);
    }

    return Tensor<T>(new_data, sizes);
}

template <class T>
Tensor<T> Tensor<T>::operator/(Tensor<T> other)
{
    if (size() == other.size())
    {
        std::vector<T> new_data;

        for (int i = 0; i < data.size(); ++i)
        {
            new_data.push_back(data[i] / other.data[i]);
        }

        return Tensor<T>(new_data, sizes);
    }
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