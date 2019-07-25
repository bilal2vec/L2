#include <iostream>
#include <vector>

#include "include/L2.h"

int main()
{

    L2::Tensor<double> y = L2::Tensor<double>({3, 3}).normal_(10, 1);
    L2::Tensor<double> z = L2::Tensor<double>({1, 3}).normal_(10, 1);

    // L2::Tensor<int> zz = y / z;

    y.print();

    L2::Tensor<double> zz = L2::exp(y);
    zz = L2::log(y);
    zz = L2::log10(y);
    zz = L2::abs(y);
    zz = L2::sin(y);
    zz = L2::cos(y);
    zz = L2::tan(y);

    zz.print();

    return 0;
}