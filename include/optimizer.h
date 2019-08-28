#pragma once

#include <iostream>

#include "parameter.h"

namespace L2::nn::optimizer
{

template <class T>
class Optimizer
{
protected:
    double lr;

public:
    virtual Parameter<T> update(Parameter<T> param)
    {
        return param;
    }
};

template <class T>
class SGD : public Optimizer<T>
{
public:
    SGD(double lr);

    Parameter<T> update(Parameter<T> param);
};

} // namespace L2::nn::optimizer