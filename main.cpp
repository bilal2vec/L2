#include <iostream>
#include <vector>
#include <exception>

#include "include/L2.h"

int main()
{
    try
    {
        L2::Tensor<double> x = L2::Tensor<double>({100, 10}).normal(0, 1);

        L2::Tensor<double> w = L2::Tensor<double>({10, 1}).normal(0, 1);
        L2::Tensor<double> b = L2::Tensor<double>({1}).normal(0, 1);

        L2::Tensor<double> y = L2::matmul(x, w) + b;

        L2::nn::loss::MSE<double> *criterion = new L2::nn::loss::MSE<double>();
        L2::nn::optimizer::SGD<double> *optimizer = new L2::nn::optimizer::SGD<double>(0.05);

        L2::nn::Sequential<double> *sequential = new L2::nn::Sequential<double>({
            new L2::nn::Linear<double>(10, 1) //
        });

        L2::trainer::Trainer<double> trainer = L2::trainer::Trainer<double>(sequential, criterion, optimizer);

        trainer.fit(x, y, 10, 10);

        std::cout << "hllo";

        // L2::nn::loss::MSE<double> criterion = L2::nn::loss::MSE<double>();
        // L2::nn::optimizer::SGD<double> *optimizer = new L2::nn::optimizer::SGD<double>(0.1);

        // L2::nn::Sequential<double> sequential = L2::nn::Sequential<double>({
        //     new L2::nn::Linear<double>(10, 1) //
        // });

        // L2::Tensor<double> y_hat = sequential.forward(x);

        // L2::Tensor<double> loss = criterion.forward(y_hat, y);
        // L2::Tensor<double> derivative = criterion.backward();

        // L2::Tensor<double> zz = sequential.backward(derivative);

        // loss.print();

        // sequential.update(optimizer);

        // std::cout << "hello";
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << "\n";
    }

    // std::vector<double> a{1, 2, 3, 4, 5, 6, 7, 8};
    // std::vector<double> b{1, 2, 3, 4, 5, 6, 7, 8};

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

    // ok
    // L2::Tensor<double> c = L2::Tensor<double>(a, {2, 2, 2});
    // L2::Tensor<double> d = L2::Tensor<double>(b, {2, 2, 2});
    // L2::Tensor<double> e = L2::cat({c, d}, 0);
    // L2::Tensor<double> e = L2::cat({c, d}, 1);
    // L2::Tensor<double> e = L2::cat({c, d}, -1);
    // e.print();

    // ok
    // L2::Tensor<double> c = L2::Tensor<double>(a, {4, 2});
    // L2::Tensor<double> d = L2::Tensor<double>(b, {2, 4});
    // L2::Tensor<double> e = L2::matmul(c, d);
    // L2::Tensor<double> c = L2::Tensor<double>(a, {2, 4, 1});
    // L2::Tensor<double> d = L2::Tensor<double>(b, {2, 1, 4});
    // L2::Tensor<double> e = L2::matmul(c, d);
    // L2::Tensor<double> c = L2::Tensor<double>(a, {2, 2, 2});
    // L2::Tensor<double> d = L2::Tensor<double>(b, {2, 2, 2});
    // L2::Tensor<double> e = L2::matmul(c, d);
    // e.print();

    // ok
    // L2::Tensor<double> c = L2::Tensor<double>({4, 4}).normal_(5, 0.1);
    // L2::Tensor<double> c = L2::Tensor<double>({4, 4}).uniform_(5, 0.1);
    // c.print();

    // ok
    // L2::Tensor<double> c = L2::Tensor<double>(a, {4, 2});
    // L2::Tensor<double> d = c.view({-1});
    // L2::Tensor<double> d = c.view({2, 2, 2});
    // d.print();

    // ok
    // L2::Tensor<double> c = L2::Tensor<double>(a, {4, 2});
    // L2::Tensor<double> d = c.unsqueeze(0);
    // L2::Tensor<double> d = c.unsqueeze(1);
    // L2::Tensor<double> d = c.unsqueeze(-1);
    // L2::Tensor<double> d = c.unsqueeze(2);
    // d.print();

    // ok
    // L2::Tensor<double> c = L2::Tensor<double>(a, {4, 2});
    // L2::Tensor<double> d = c.transpose();
    // d.print();

    // ok
    // L2::Tensor<double> c = L2::Tensor<double>(a, {4, 2});
    // L2::Tensor<double> d = c.clone();
    // d.print();

    return 0;
}