#include <iostream>
#include <exception>

struct InvalidShapeException : public std::exception
{
    const char *what() const throw()
    {
        return "Invalid shape";
    }
};

struct InvalidOperationException : public std::exception
{
    const char *what() const throw()
    {
        return "Invalid operation";
    }
};

struct TooManyDimenstionsException : public std::exception
{
    const char *what() const throw()
    {
        return "Too many dimensions, max 4";
    }
};

struct ForbiddenTypeException : public std::exception
{
    const char *what() const throw()
    {
        return "Type of Tensor is not allowed, only Tensors of type <double> are allowed";
    }
};