#include <iostream>

#include "tensor.h"
#include "index.h"

Tensor::Tensor(double x) : data(1, x), sizes(1, 1), strides(1, 1) {}

Tensor::Tensor(std::vector<double> x) : data(x), sizes(1, static_cast<int>(x.size())), strides(1, 1) {}
Tensor::Tensor(std::vector<double> x, std::vector<int> size) : data(x), sizes(size), strides(get_strides(sizes)) {}
Tensor::Tensor(std::vector<double> x, std::initializer_list<int> size) : data(x), sizes(size), strides(get_strides(sizes)) {}

int Tensor::get_physical_idx(std::vector<int> indices)
{
    int physical_idx = 0;

    for (int i = 0; i < static_cast<int>(indices.size()); ++i)
    {
        physical_idx += (indices[i] * strides[i]);
    }

    return physical_idx;
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

void Tensor::print()
{
    std::cout << "data:\n";
    _print(data);
    std::cout << "size:\n";
    _print(sizes);
    std::cout << "strides:\n";
    _print(strides);
}