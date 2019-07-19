#include <iostream>
#include <random>

#include "tensor.h"
#include "index.h"

Tensor::Tensor(double x) : data(1, x), sizes(1, 1), strides(1, 1) {}

Tensor::Tensor(std::vector<int> sizes) : data(sizes_to_n_elements(sizes), 0), sizes(sizes), strides(get_strides(sizes)) {}

Tensor::Tensor(std::vector<double> x, std::vector<int> sizes) : data(x), sizes(sizes), strides(get_strides(sizes)) {}

int Tensor::get_physical_idx(std::vector<int> indices)
{
    int physical_idx = 0;

    for (int i = 0; i < static_cast<int>(indices.size()); ++i)
    {
        physical_idx += (indices[i] * strides[i]);
    }

    return physical_idx;
}

int Tensor::sizes_to_n_elements(std::vector<int> sizes)
{
    int sum = 1;

    for (int size : sizes)
    {
        sum *= size;
    }

    return sum;
}

bool Tensor::valid_sizes(std::vector<int> new_sizes)
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

std::vector<index> Tensor::process_index(std::vector<index> indices)
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

std::vector<int> Tensor::get_sizes(std::vector<index> indices)
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

std::vector<int> Tensor::get_strides(std::vector<int> sizes)
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

// tensor indexing and slicing
Tensor Tensor::operator()(std::vector<index> indices)
{
    indices = process_index(indices);

    std::vector<double> slice;
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
    return Tensor(slice, slice_sizes);
}

Tensor Tensor::normal_(double mean, double stddev)
{
    std::random_device random_device{};
    std::mt19937 pseudorandom_generator{random_device()};

    std::normal_distribution<double> distribution{mean, stddev};

    std::vector<double> normal_tensor;

    for (int i = 0; i < data.size(); ++i)
    {
        double sample = distribution(pseudorandom_generator);

        normal_tensor.push_back(sample);
    }

    return Tensor(normal_tensor, sizes);
}

Tensor Tensor::uniform_(double low, double high)
{
    std::random_device random_device{};
    std::mt19937 pseudorandom_generator{random_device()};

    std::uniform_real_distribution<double> distribution{low, high};

    std::vector<double> uniform_tensor;

    for (int i = 0; i < data.size(); ++i)
    {
        double sample = distribution(pseudorandom_generator);

        uniform_tensor.push_back(sample);
    }

    return Tensor(uniform_tensor, sizes);
}

void Tensor::view(std::vector<int> new_sizes)
{
    if (valid_sizes(new_sizes))
    {
        sizes = new_sizes;
        strides = get_strides(sizes);
    }
}

void Tensor::print()
{
    std::cout << "data:\n";
    _print(data);
    std::cout << "size:\n";
    _print(sizes);
    std::cout << "strides:\n";
    _print(strides);
}