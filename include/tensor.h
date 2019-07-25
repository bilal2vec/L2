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

    std::vector<int> get_shape(std::vector<index> indices);
    std::vector<int> get_strides(std::vector<int> shape);

    std::vector<int> expand_shape(std::vector<int> shape, int diff);
    std::vector<int> broadcast_shape(std::vector<int> shape_a, std::vector<int> shape_b);

    std::tuple<std::vector<int>, std::vector<int>> broadcast_strides(std::vector<int> lhs_shape, std::vector<int> rhs_shape, std::vector<int> new_shape);

    T operation(T lhs, T rhs, std::string op);

    Tensor<T> tensor_elementwise_op(Tensor<T> other, std::string op);
    Tensor<T> scalar_elementwise_op(T other, std::string op);

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

    Tensor<T> normal_(double mean = 0, double stddev = 1);
    Tensor<T> uniform_(double low = 0, double high = 1);

    void view(std::vector<int> new_shape);
    std::vector<int> get_shape();

    void print();
};
}