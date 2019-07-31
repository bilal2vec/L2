#include <iostream>
#include <vector>

#include "include/L2.h"

int main()
{

    std::vector<double> a{1, 2, 3, 4, 5, 6};
    std::vector<int> b{1, 2, 3, 4, 5, 6};

    L2::Tensor<double> x = L2::Tensor<double>(a, {2, 3});
    L2::Tensor<int> y = L2::Tensor<int>(b, {2, 3});

    L2::Tensor<double> z = x + y;

    z.print();

    return 0;
}