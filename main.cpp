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

    // y.print();

    // Tensor<double> y = Tensor<double>({3, 3}).normal_(10, 0.1);

    // std::vector<int> z = y.size();

    // for (int i : z)
    // {
    //     std::cout << i << ", ";
    // }

    // y.print();

    // Tensor<double> z = y / 3;

    // z.print();

    // std::vector<double> a{
    //     1,
    //     1,
    //     1,
    //     1,
    //     1,
    //     1,
    //     1,
    //     1,
    //     1,
    // };
    // std::vector<double> b{0, 1, 2};

    // Tensor<double> y = Tensor<double>(a, {3, 3}); // 2, 2
    // Tensor<double> z = Tensor<double>(b, {3});    // 1, 2

    // y.print();
    // z.print();

    // Tensor<double> zz = y + z;

    // std::cout << "\n\n";

    // zz.print();

    // x.print();
    // y.print();
    // z.print();

    std::vector<double> a{0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<double> b{0, 1, 2, 3};

    Tensor<double> y = Tensor<double>(a, {1, 2, 2, 2}); // 2, 2
    Tensor<double> z = Tensor<double>(b, {2, 2});       // 1, 2

    Tensor<double> zz = y + z;

    zz.print();

    // std::vector<double> a{0, 1, 2};
    // std::vector<double> b{5};

    // Tensor<double> y = Tensor<double>(a, {3}); // 2, 2
    // Tensor<double> z = Tensor<double>(b, {1}); // 1, 2

    // Tensor<double> zz = y + z;

    // zz.print();

    return 0;
}