# Notes

-   no ints or primitive types in generic parameters
-   https://www.lpalmieri.com/

## Tensor

### slicing

-   use enum to store diff between two types of indices
    -   [start, end)
    -   -1

### Todo

-   change col-major to row-major
-   impl iterator
-   separate the view and data parts of tensor to not have to copy everything
    -   use pointers?

### Done

-   use vectors to store data

### Notes

-   vector's overhead shouldn't be much
-   maybe set vec to not allocate extra memory
-   vec is contigous in memory
-   naive slicing is too slow
-   numpy sometimes makes a copy when slicing to keep stuff contigous
-   naive approach slows down with size
-   it looks like numpy will make a copy when slicing
-   dont' use `[start:stop]` syntax for rust, use `.slice(!vec[start:stop])`
-   you should be able to do:
    -   `x[1:2, 3:4] = y`
        -   this has to be inplace
            -   return a tensor containing a vector of mut references to values in original tensor?
            -   return vec of mut references to original data
        -   bad idea, instead:
            -   use strides and slice start and stop points to allow access to the right parts of the data vec
            -   return a mut reference to the tensor
    -   `y = x[1:2, 3:4]`
        -   this can be a copy or a view
        -   should be able to do either easily
