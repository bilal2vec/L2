#pragma once

#include <vector>
#include <random>

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
    int sizes_to_n_elements(std::vector<int> sizes);
    bool valid_sizes(std::vector<int> new_sizes);

    std::vector<index> process_index(std::vector<index> indices);

    std::vector<int> get_sizes(std::vector<index> indices);
    std::vector<int> get_strides(std::vector<int> sizes);

public:
    Tensor(double x);

    Tensor(std::vector<int> sizes);
    Tensor(std::vector<double> x, std::vector<int> sizes);

    Tensor operator()(std::vector<index> indices);

    Tensor normal_(double mean = 0, double stddev = 1);
    Tensor uniform_(double low = 0, double high = 1);

    void view(std::vector<int> new_sizes);

    void print();
};