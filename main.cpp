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

        L2::Tensor<double> y_hat = trainer.predict(x);

        y_hat.print();
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << "\n";
    }

    return 0;
}