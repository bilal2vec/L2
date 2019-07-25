#include <iostream>
#include <vector>

#include "include/L2.h"

int main()
{

    L2::Tensor<float> y = L2::Tensor<float>({2, 3, 3}).normal_(10, 1);
    L2::Tensor<float> z = L2::Tensor<float>({1, 3}).normal_(10, 1);

    L2::Tensor<float> zz = y / z;

    zz.print();

    return 0;
}