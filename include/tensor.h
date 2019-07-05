#pragma once

#include <vector>

#include "index.h"

class Tensor
{
private:
    std::vector<double> data;
    std::vector<int> sizes;
    std::vector<int> strides;

    template <typename T>
    void _print(std::vector<T> x)
    {
        std::cout << "\n";

        for (double i : x)
        {
            std::cout << i << ", ";
        }

        std::cout << "\n\n";
    }

    int get_physical_idx(std::vector<int> indices);

    std::vector<int> get_sizes(std::vector<index> indices);
    std::vector<int> get_strides(std::vector<int> sizes);

public:
    Tensor(double x);
    Tensor(std::vector<double> x);
    Tensor(std::vector<double> x, std::vector<int> sizes);
    Tensor(std::vector<double> x, std::initializer_list<int> sizes);

    Tensor operator()(std::vector<index> indices);

    void print();
};