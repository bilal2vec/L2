#include <iostream>
#include <vector>

#include "include/L2.h"

int main()
{

    std::vector<double> a{1, 2, 3, 4, 5, 6, 7, 8};

    L2::Tensor<double> c = L2::Tensor<double>(a, {2, 2, 2});

    L2::Tensor<double> e = c.transpose();

    e.print();

    return 0;
}