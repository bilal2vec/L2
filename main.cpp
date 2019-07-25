#include <iostream>
#include <vector>

#include "include/L2.h"

int main()
{

    L2::Tensor<int> y = L2::Tensor<int>({2, 3, 3}).normal_(10, 1);
    L2::Tensor<int> z = L2::Tensor<int>({1, 3}).normal_(10, 1);

    L2::Tensor<int> zz = y / z;

    // zz.print();
    std::cout << zz.type() << "\n";

    return 0;
}