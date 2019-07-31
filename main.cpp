#include <iostream>
#include <vector>

#include "include/L2.h"

int main()
{

    std::vector<double> a{1, 2, 3, 4, 5, 6};

    L2::Tensor<double> b = L2::Tensor<double>(a, {2, 3});

    L2::Tensor<double> c = b({{1, 2}});

    c.print();

    return 0;
}