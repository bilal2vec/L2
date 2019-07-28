#include <iostream>
#include <vector>

#include "include/L2.h"

int main()
{

    std::vector<double> x{1, 2, 3, 4, 5, 6, 7, 8};

    L2::Tensor<double> xx = L2::Tensor<double>(x, {2, 2, 2});
    L2::Tensor<double> yy = L2::Tensor<double>(x, {2, 2, 2});

    std::vector<L2::Tensor<double>> zz{xx, yy};

    L2::Tensor<double> z = L2::cat(zz, 2);

    z.print();

    return 0;
}