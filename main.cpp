#include <iostream>
#include <vector>

#include "include/L2.h"

int main()
{
    Tensor<double> y = Tensor<double>({3, 1}).normal_(0, 1);
    Tensor<double> z = Tensor<double>({1, 3}).normal_(0, 1);

    Tensor<double> zz = y * z;

    zz.print();

    return 0;
}