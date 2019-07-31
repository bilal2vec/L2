#include <iostream>
#include <vector>

#include "include/L2.h"

int main()
{

    std::vector<double> a{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<double> b{1, 2, 3, 4, 5, 6, 7, 8};

    //ok
    // L2::Tensor<double> c = L2::Tensor<double>({2, 2});
    // c.print();

    // ok
    // L2::Tensor<double> c = L2::Tensor<double>(a, {2, 2, 2});
    // c.print();

    // ok
    // L2::Tensor<double> c = L2::Tensor<double>(a, {2, 2, 2});
    // L2::Tensor<double> d = c({{0, 1}});
    // d.print();

    // L2::Tensor<double> c = L2::Tensor<double>(a, {2, 2, 2});
    // L2::Tensor<double> d = c({{0, 1}, {1, 2}});
    // d.print();

    // ok
    // L2::Tensor<double> c = L2::Tensor<double>(a, {2, 2, 2});
    // L2::Tensor<double> d = c + 2;
    // L2::Tensor<double> d = c - 2;
    // L2::Tensor<double> d = c * 2;
    // L2::Tensor<double> d = c / 2;
    // d.print();

    // ok
    // L2::Tensor<double> c = L2::Tensor<double>(a, {2, 2, 2});
    // L2::Tensor<double> d = c({{0, 1}});
    // d.print();
    // L2::Tensor<double> e = c + d;
    // L2::Tensor<double> e = c - d;
    // L2::Tensor<double> e = c * d;
    // L2::Tensor<double> e = c / d;
    // e.print();

    // ok
    // L2::Tensor<double> c = L2::Tensor<double>(a, {2, 2, 2});
    // L2::Tensor<double> d = c.pow(1);
    // L2::Tensor<double> d = c.sqrt();
    // L2::Tensor<double> d = c.log();
    // L2::Tensor<double> d = c.log10();
    // L2::Tensor<double> d = c.abs();
    // L2::Tensor<double> d = c.sin();
    // L2::Tensor<double> d = c.cos();
    // L2::Tensor<double> d = c.tan();
    // d.print();

    // ok
    // L2::Tensor<double> c = L2::Tensor<double>(a, {2, 2, 2});
    // L2::Tensor<double> d = L2::sum(c);
    // L2::Tensor<double> d = L2::sum(c, -1);
    // L2::Tensor<double> d = L2::mean(c);
    // L2::Tensor<double> d = L2::mean(c, 1);
    // L2::Tensor<double> d = L2::max(c);
    // L2::Tensor<double> d = L2::max(c, 1);
    // L2::Tensor<double> d = L2::min(c);
    // L2::Tensor<double> d = L2::min(c, 1);
    // L2::Tensor<double> d = L2::argmax(c);
    // L2::Tensor<double> d = L2::argmax(c, 1);
    // L2::Tensor<double> d = L2::argmin(c);
    // L2::Tensor<double> d = L2::argmin(c, 1);
    // d.print();

    //
    L2::Tensor<double> c = L2::Tensor<double>(a, {2, 2, 2});
    L2::Tensor<double> d = L2::Tensor<double>(b, {2, 2, 2});
    // L2::Tensor<double> e = L2::cat({c, d}, 0);
    // L2::Tensor<double> e = L2::cat({c, d}, 1);
    L2::Tensor<double> e = L2::cat({c, d}, 2);
    e.print();

    return 0;
}