#include <iostream>
#include <vector>

#include "include/L2.h"

int main()
{

    L2::Tensor<double> y = L2::Tensor<double>({3, 3}).normal_(10, 1);

    y.print();

    L2::Tensor<double> z = y.mean();

    z.print();

    return 0;
}