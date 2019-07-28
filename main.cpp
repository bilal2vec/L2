#include <iostream>
#include <vector>

#include "include/L2.h"

int main()
{

    std::vector<double> x{0.6580, -1.0969, -0.4614, -0.1034, -0.5790, 0.1497};

    L2::Tensor<double> xx = L2::Tensor<double>(x, {2, 3});
    L2::Tensor<double> yy = L2::Tensor<double>(x, {2, 3});

    std::vector<L2::Tensor<double>> zz{xx, yy};

    L2::Tensor<double> z = L2::cat(zz, 1);
    // L2::Tensor<double> z = xx.cat(yy, 0);

    z.print();

    return 0;
}