#pragma once

#include <vector>
#include <random>

#include "index.h"

template <class T>
class Tensor
{
private:
    std::vector<T> data;
    std::vector<int> sizes;
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
    int sizes_to_n_elements(std::vector<int> sizes);
    bool valid_sizes(std::vector<int> new_sizes);

    std::vector<int> expand_sizes(std::vector<int> size, int size_diff);
    std::vector<int> broadcast_sizes(std::vector<int> sizes_a, std::vector<int> sizes_b);
    bool broadcastable_with(std::vector<int> new_sizes);

    std::vector<index> process_index(std::vector<index> indices);

    std::vector<int> get_sizes(std::vector<index> indices);
    std::vector<int> get_strides(std::vector<int> sizes);

public:
    Tensor(std::vector<int> sizes);
    Tensor(std::vector<T> x, std::vector<int> sizes);

    Tensor<T> operator()(std::vector<index> indices);

    Tensor<T> operator+(T othen);
    Tensor<T> operator+(Tensor<T> othen);

    Tensor<T> operator-(T othen);
    Tensor<T> operator-(Tensor<T> othen);

    Tensor<T> operator*(T othen);
    Tensor<T> operator*(Tensor<T> othen);

    Tensor<T> operator/(T othen);
    Tensor<T> operator/(Tensor<T> othen);

    Tensor<T> normal_(double mean = 0, double stddev = 1);
    Tensor<T> uniform_(double low = 0, double high = 1);

    void view(std::vector<int> new_sizes);
    std::vector<int> size();

    void print();
};