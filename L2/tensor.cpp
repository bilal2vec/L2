#include <iostream>
#include <vector>
#include <cmath>     // std::abs, std::pow
#include <algorithm> //std::max, std::max_element, std::min_element
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
int Tensor<T>::shape_to_n_elements(std::vector<int> shapes)
{
    int sum = 1;

    for (int shape : shapes)
    {
        sum *= shape;
    }

    return sum;
}

template <class T>
bool Tensor<T>::valid_shape(std::vector<int> shape, std::vector<int> new_shape)
{
    int current_shapes_sum = shape_to_n_elements(shape);
    int new_shapes_sum = shape_to_n_elements(new_shape);

    if (current_shapes_sum == new_shapes_sum)
    {
        return true;
    }
    else
    {
        return false;
    }
}

template <class T>
std::vector<index> Tensor<T>::process_index(std::vector<index> indices, std::vector<int> shape)
{
    for (int i = 0; i < indices.size(); ++i)
    {
        if (indices[i].stop == -1)
        {
            indices[i].stop = shape[i];
        }
    }

    return indices;
}

template <class T>
std::vector<index> Tensor<T>::process_dims(std::vector<index> indices, int dim, int current_dim, int i)
{
    if (dim == current_dim)
    {
        indices[current_dim] = {0, -1};
    }
    else
    {
        indices[current_dim] = {i, i + 1};
    }

    return indices;
}

template <class T>
std::vector<int> Tensor<T>::get_shape(std::vector<index> indices)
{
    std::vector<int> slice_shape;

    for (index idx : indices)
    {
        // Add number of elements in dimension to slice_shape if not 1
        if ((idx.stop - idx.start) > 1)
        {
            slice_shape.push_back(idx.stop - idx.start);
        }
    }

    // slice_size is 1 if selecting single element from tensor
    if (static_cast<int>(slice_shape.size()) == 0)
    {
        slice_shape.push_back(1);
    }

    return slice_shape;
}

template <class T>
std::vector<int> Tensor<T>::get_strides(std::vector<int> shape)
{
    std::vector<int> strides;

    // from: https://github.com/ThinkingTransistor/Sigma/blob/fe645441eb523996d3bfc6de9ad72e814a146195/Sigma.Core/MathAbstract/NDArrayUtils.cs#L24
    int current_stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i)
    {
        strides.insert(strides.begin(), current_stride);
        current_stride *= shape[i];
    }
    return strides;
}

template <class T>
std::vector<int> Tensor<T>::expand_shape(std::vector<int> shape, int diff)
{
    for (int i = 0; i < diff; ++i)
    {
        shape.insert(shape.begin(), 1);
    }
    return shape;
}

template <class T>
std::vector<int> Tensor<T>::broadcast_shape(std::vector<int> shape_a, std::vector<int> shape_b)
{
    // choose max of each element in shape_a and shape_b as new size
    std::vector<int> new_shape;
    for (int i = 0; i < shape_a.size(); ++i)
    {
        new_shape.push_back(std::max(shape_a[i], shape_b[i]));
    }

    return new_shape;
}

template <class T>
std::tuple<std::vector<int>, std::vector<int>> Tensor<T>::broadcast_strides(std::vector<int> lhs_shape, std::vector<int> rhs_shape, std::vector<int> new_shape)
{

    std::vector<int> lhs_strides = get_strides(lhs_shape);
    std::vector<int> rhs_strides = get_strides(rhs_shape);

    for (int i = 0; i < new_shape.size(); ++i)
    {
        if (lhs_shape[i] != new_shape[i])
        {
            lhs_strides[i] = 0;
        }

        if (rhs_shape[i] != new_shape[i])
        {
            rhs_strides[i] = 0;
        }
    }

    return {lhs_strides, rhs_strides};
}

template <class T>
std::vector<T> Tensor<T>::concat_tensors(std::vector<Tensor<T>> tensors, std::vector<index> idxs, std::vector<T> vector)
{
    for (Tensor<T> tensor : tensors)
    {
        std::vector<T> temp = tensor(idxs).get_data();

        vector.insert(vector.end(), temp.begin(), temp.end());
    }
    return vector;
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
    else if (op == "pow")
    {
        return std::pow(lhs, rhs);
    }
}

template <class T>
T Tensor<T>::operation(T lhs, std::string op)
{
    if (op == "sqrt")
    {
        return std::sqrt(lhs);
    }
    else if (op == "exp")
    {
        return std::exp(lhs);
    }
    else if (op == "log")
    {
        return std::log(lhs);
    }
    else if (op == "log10")
    {
        return std::log10(lhs);
    }
    else if (op == "abs")
    {
        return std::abs(lhs);
    }
    else if (op == "sin")
    {
        return std::sin(lhs);
    }
    else if (op == "cos")
    {
        return std::cos(lhs);
    }
    else if (op == "tan")
    {
        return std::tan(lhs);
    }
}

template <class T>
T Tensor<T>::sum(Tensor<T> tensor)
{
    T new_data;

    for (T el : tensor.data)
    {
        new_data += el;
    }

    return new_data;
}

template <class T>
T Tensor<T>::dimension_operation(Tensor<T> tensor, std::string op)
{
    if (op == "sum")
    {
        return sum(tensor);
    }
    else if (op == "mean")
    {
        return sum(tensor) / tensor.data.size();
    }
    else if (op == "max")
    {
        return *std::max_element(tensor.data.begin(), tensor.data.end());
    }
    else if (op == "min")
    {
        return *std::min_element(tensor.data.begin(), tensor.data.end());
    }
    else if (op == "argmax")
    {
        return std::distance(tensor.data.begin(), std::max_element(tensor.data.begin(), tensor.data.end()));
    }
    else if (op == "argmin")
    {
        return std::distance(tensor.data.begin(), std::min_element(tensor.data.begin(), tensor.data.end()));
    }
}

template <class T>
Tensor<T> Tensor<T>::tensor_elementwise_op(Tensor<T> other, std::string op)
{
    // expand lhs and rhs sizes to have the same number of elements
    int diff = get_shape().size() - other.get_shape().size();

    std::vector<int> lhs_shape = (diff < 0) ? expand_shape(get_shape(), std::abs(diff)) : get_shape();
    std::vector<int> rhs_shape = (diff > 0) ? expand_shape(other.get_shape(), std::abs(diff)) : other.get_shape();

    // broadcast lhs_shape and rhs_shape to get the broadcasted size
    std::vector<int> new_shape = broadcast_shape(lhs_shape, rhs_shape);

    // update strides and zero out broadcasted dimensions
    std::vector<int> lhs_strides;
    std::vector<int> rhs_strides;
    std::tie(lhs_strides, rhs_strides) = broadcast_strides(lhs_shape, rhs_shape, new_shape);

    // perform operation on data element wise and save
    std::vector<T> new_data;

    int length = static_cast<int>(new_shape.size());
    for (int i = 0; i < new_shape[0]; ++i)
    {
        if (length > 1)
        {
            for (int j = 0; j < new_shape[1]; ++j)
            {
                if (length > 2)
                {
                    for (int k = 0; k < new_shape[2]; ++k)
                    {
                        if (length > 3)
                        {
                            for (int m = 0; m < new_shape[3]; ++m)
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

    return Tensor<T>(new_data, new_shape);
}

template <class T>
Tensor<T> Tensor<T>::scalar_elementwise_op(T other, std::string op)
{
    std::vector<T> new_data;

    for (int i = 0; i < data.size(); ++i)
    {
        new_data.push_back(operation(data[i], other, op));
    }

    return Tensor<T>(new_data, get_shape());
}

template <class T>
Tensor<T> Tensor<T>::tensor_op(std::string op)
{
    std::vector<T> new_data;

    for (int i = 0; i < data.size(); ++i)
    {
        new_data.push_back(operation(data[i], op));
    }

    return Tensor<T>(new_data, get_shape());
}

template <class T>
Tensor<T> Tensor<T>::dimension_op(int dim, std::string op)
{
    std::vector<T> new_data;

    std::vector<int> current_shape = get_shape();
    int length = static_cast<int>(current_shape.size());
    dim = (dim == -1) ? current_shape.size() - 1 : dim;

    std::vector<index> idxs;

    for (int i = 0; i < length; ++i)
    {
        idxs.push_back({0, 0});
    }

    for (int i = 0; i < (dim == 0 ? 1 : current_shape[0]); ++i)
    {

        idxs = process_dims(idxs, dim, 0, i);

        if (length > 1)
        {
            for (int j = 0; j < (dim == 1 ? 1 : current_shape[1]); ++j)
            {
                idxs = process_dims(idxs, dim, 1, j);

                if (length > 2)
                {
                    for (int k = 0; k < (dim == 2 ? 1 : current_shape[2]); ++k)
                    {
                        idxs = process_dims(idxs, dim, 2, k);
                        if (length > 3)
                        {
                            for (int m = 0; m < (dim == 3 ? 1 : current_shape[3]); ++m)
                            {

                                idxs = process_dims(idxs, dim, 3, m);

                                T op_result = dimension_operation((*this)(idxs), op);
                                new_data.push_back(op_result);
                            }
                        }
                        else
                        {
                            T op_result = dimension_operation((*this)(idxs), op);
                            new_data.push_back(op_result);
                        }
                    }
                }
                else
                {
                    T op_result = dimension_operation((*this)(idxs), op);
                    new_data.push_back(op_result);
                }
            }
        }
        else
        {
            T op_result = dimension_operation((*this)(idxs), op);
            new_data.push_back(op_result);
        }
    }

    current_shape.erase(current_shape.begin() + dim);

    return Tensor<T>(new_data, current_shape);
}

template <class T>
std::vector<T> Tensor<T>::matmul_(Tensor<T> lhs, Tensor<T> rhs, int dim)
{

    if (lhs.get_shape().size() == 1 && rhs.get_shape().size() == 1)
    {
        lhs = lhs.view({1, lhs.get_shape()[0]});
        rhs = rhs.view({rhs.get_shape()[0], 1});
    }

    std::vector<T> new_data;

    // efficient matmul from https://github.com/fastai/course-v3/blob/master/nbs/dl2/01_matmul.ipynb
    for (int i = 0; i < dim; ++i)
    {
        Tensor<T> a = lhs({{i, i + 1}, {0, -1}});
        a.shape.push_back(1);

        Tensor<T> c = (a * rhs).sum(0);

        new_data.insert(new_data.end(), c.data.begin(), c.data.end());
    }

    return new_data;
}

template <class T>
Tensor<T>::Tensor(std::vector<int> shape) : data(shape_to_n_elements(shape), 0), shape(shape), strides(get_strides(shape)) {}

template <class T>
Tensor<T>::Tensor(std::vector<T> x, std::vector<int> shape) : data(x), shape(shape), strides(get_strides(shape)) {}

// tensor indexing and slicing
template <class T>
Tensor<T> Tensor<T>::operator()(std::vector<index> indices)
{
    indices = process_index(indices, get_shape());

    std::vector<T> slice;
    std::vector<int> slice_shape = get_shape(indices);

    int length = static_cast<int>(get_shape().size());

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
    return Tensor<T>(slice, slice_shape);
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
Tensor<T> Tensor<T>::pow(T other)
{
    return scalar_elementwise_op(other, "pow");
}

template <class T>
Tensor<T> Tensor<T>::sqrt()
{
    return tensor_op("sqrt");
}

template <class T>
Tensor<T> Tensor<T>::exp()
{
    return tensor_op("exp");
}

template <class T>
Tensor<T> Tensor<T>::log()
{
    return tensor_op("log");
}

template <class T>
Tensor<T> Tensor<T>::log10()
{
    return tensor_op("log10");
}

template <class T>
Tensor<T> Tensor<T>::abs()
{
    return tensor_op("abs");
}

template <class T>
Tensor<T> Tensor<T>::sin()
{
    return tensor_op("sin");
}

template <class T>
Tensor<T> Tensor<T>::cos()
{
    return tensor_op("cos");
}

template <class T>
Tensor<T> Tensor<T>::tan()
{
    return tensor_op("tan");
}

template <class T>
Tensor<T> Tensor<T>::sum()
{
    T new_data;

    for (T el : data)
    {
        new_data += el;
    }

    return Tensor<T>({new_data}, {1});
}

template <class T>
Tensor<T> Tensor<T>::sum(int dim)
{
    return dimension_op(dim, "sum");
}

template <class T>
Tensor<T> Tensor<T>::mean()
{
    Tensor<T> tensor = sum();
    tensor.data[0] /= shape_to_n_elements(get_shape());

    return tensor;
}

template <class T>
Tensor<T> Tensor<T>::mean(int dim)
{
    return dimension_op(dim, "mean");
}

template <class T>
Tensor<T> Tensor<T>::max()
{
    T max = *std::max_element(data.begin(), data.end());

    return Tensor<T>({max}, {1});
}

template <class T>
Tensor<T> Tensor<T>::max(int dim)
{
    return dimension_op(dim, "max");
}

template <class T>
Tensor<T> Tensor<T>::min()
{
    T min = *std::min_element(data.begin(), data.end());

    return Tensor<T>({min}, {1});
}

template <class T>
Tensor<T> Tensor<T>::min(int dim)
{
    return dimension_op(dim, "min");
}

template <class T>
Tensor<T> Tensor<T>::argmax()
{
    T argmax = std::distance(data.begin(), std::max_element(data.begin(), data.end()));

    return Tensor<T>({argmax}, {1});
}

template <class T>
Tensor<T> Tensor<T>::argmax(int dim)
{
    return dimension_op(dim, "argmax");
}

template <class T>
Tensor<T> Tensor<T>::argmin()
{
    T argmin = std::distance(data.begin(), std::min_element(data.begin(), data.end()));

    return Tensor<T>({argmin}, {1});
}

template <class T>
Tensor<T> Tensor<T>::argmin(int dim)
{
    return dimension_op(dim, "argmin");
}

template <class T>
Tensor<T> Tensor<T>::cat(std::vector<Tensor<T>> tensors, int dim)
{
    std::vector<T> new_data;

    std::vector<int> new_shape = tensors[0].get_shape();
    for (int i = 1; i < tensors.size(); ++i)
    {
        new_shape[dim] += tensors[i].get_shape()[dim];
    }

    int length = static_cast<int>(new_shape.size());

    std::vector<index> idxs;
    for (int i = 0; i < length; ++i)
    {
        idxs.push_back({0, -1});
    }

    for (int i = 0; i < (dim == 0 ? 1 : new_shape[0]); ++i)
    {

        idxs = process_dims(idxs, dim, 0, i);

        if (dim == 0)
        {
            new_data = concat_tensors(tensors, idxs, new_data);
        }
        else
        {
            for (int j = 0; j < (dim == 1 ? 1 : new_shape[1]); ++j)
            {
                idxs = process_dims(idxs, dim, 1, j);

                if (dim == 1)
                {
                    new_data = concat_tensors(tensors, idxs, new_data);
                }
                else
                {
                    for (int k = 0; k < (dim == 2 ? 1 : new_shape[2]); ++k)
                    {
                        idxs = process_dims(idxs, dim, 2, k);

                        if (dim == 2)
                        {
                            new_data = concat_tensors(tensors, idxs, new_data);
                        }
                        else
                        {
                            for (int m = 0; m < (dim == 3 ? 1 : new_shape[3]); ++m)
                            {
                                idxs = process_dims(idxs, dim, 3, m);

                                new_data = concat_tensors(tensors, idxs, new_data);
                            }
                        }
                    }
                }
            }
        }
    }

    return Tensor<T>(new_data, new_shape);
}

template <class T>
Tensor<T> Tensor<T>::matmul(Tensor<T> rhs)
{
    Tensor<T> lhs = (*this);

    std::vector<T> new_data;
    std::vector<int> new_shape = lhs.get_shape();
    new_shape[new_shape.size() - 1] = rhs.get_shape()[new_shape.size() - 1];
    int length = new_shape.size();

    std::vector<int> batch_dims(new_shape.begin(), new_shape.end() - 2);

    std::vector<index> idxs;
    for (int i = 0; i < length; ++i)
    {
        idxs.push_back({0, -1});
    }

    if (length > 2)
    {
        for (int i = 0; i < batch_dims[0]; ++i)
        {
            idxs[0] = {i, i + 1};
            if (length > 3)
            {
                for (int j = 0; j < batch_dims[1]; ++j)
                {
                    idxs[1] = {j, j + 1};

                    std::vector<T> temp = matmul_(lhs(idxs), rhs(idxs), new_shape[new_shape.size() - 2]);
                    new_data.insert(new_data.end(), temp.begin(), temp.end());
                }
            }
            else
            {
                std::vector<T> temp = matmul_(lhs(idxs), rhs(idxs), new_shape[new_shape.size() - 2]);
                new_data.insert(new_data.end(), temp.begin(), temp.end());
            }
        }
    }
    else
    {
        std::vector<T> temp = matmul_(lhs(idxs), rhs(idxs), new_shape[new_shape.size() - 2]);
        new_data.insert(new_data.end(), temp.begin(), temp.end());
    }

    return Tensor<T>(new_data, new_shape);
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

    return Tensor<T>(normal_tensor, get_shape());
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

    return Tensor<T>(uniform_tensor, get_shape());
}

template <class T>
Tensor<T> Tensor<T>::view(std::vector<int> new_shape)
{
    if (valid_shape(get_shape(), new_shape))
    {
        return Tensor<T>(data, new_shape);
    }
}

template <class T>
Tensor<T> Tensor<T>::clone()
{
    return Tensor(data, shape);
}

template <class T>
std::vector<int> Tensor<T>::get_shape()
{
    return shape;
}

template <class T>
std::vector<T> Tensor<T>::get_data()
{
    return data;
}

template <class T>
std::string Tensor<T>::type()
{
    std::string type_name = typeid(T).name();

    if (type_name == "d")
    {
        return "double";
    }
    else if (type_name == "f")
    {
        return "float";
    }
    else if (type_name == "i")
    {
        return "int";
    }
    else
    {
        return type_name;
    }
}

template <class T>
void Tensor<T>::print()
{
    std::cout << "data:\n";
    _print(data);
    std::cout << "size:\n";
    _print(shape);
    std::cout << "strides:\n";
    _print(strides);
    std::cout << "dtype:\n\n"
              << type() << "\n";
}

template class Tensor<int>;
template class Tensor<float>;
template class Tensor<double>;
} // namespace L2