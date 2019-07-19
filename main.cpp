#include <iostream>
#include <vector>

#include "include/L2.h"

int main()
{

    // 1d array
    // std::vector<double> x{1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9};

    // Tensor y(x, {9});

    // Tensor z = y({{0}});

    // z.print();

    // 2d array
    // std::vector<double> x{1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9};

    // Tensor y(x, {3, 3});

    // Tensor z = y({{0, 2}, {0, 3}}); // row
    // // Tensor z = y({{0, 3}, {0, 1}}); // col
    // // Tensor z = y({{0, 3}, {0, 3}}); // all
    // // Tensor z = y({{0, 1}, {0, 1}}); // one

    // z.print();

    // z.view({1, 6});

    // z.print();

    // 3d array
    // std::vector<double> x{1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8};

    // Tensor y(x, {2, 2, 2});

    // Tensor z = y({{1, 2}, {0, 2}, {0, 2}});

    // z.print();

    // Tensor y({3, 3});

    // y.print();

    // Tensor y = Tensor({3, 3}).normal_(10, 0.1);
    Tensor y = Tensor({3, 3}).uniform_(0, 1);

    y.print();

    return 0;
}