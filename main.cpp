#include <iostream>
#include <vector>

#include "include/L2.h"

int main()
{

    std::vector<double> x{1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0};
    L2::Tensor<double> y = L2::Tensor<double>(x, {2, 2, 2});

    y.print();

    L2::Tensor<double> z = y.argmax();

    z.print();

    return 0;
}