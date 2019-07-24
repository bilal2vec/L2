#include <iostream>
#include <vector>

#include "include/L2.h"

int main()
{

    L2::Tensor<double> y = L2::Tensor<double>({3, 1}).normal_(0, 1);
    L2::Tensor<double> z = L2::Tensor<double>({1, 3}).normal_(0, 1);

    L2::Tensor<double> zz = y / z;

    zz.print();

    return 0;
}