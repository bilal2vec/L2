# L2

A C++ deep learning library

## ToDo

-   tensors
-   error handling and checking

## Done

-   handle errors when dividing ints
-   change some code to use iterators over vectors instead of for loops
-   use shape() more
-   more static functions
-   sum_sizes => shape_to_n_elements
-   namespace
-   try removing broadcastable_with
-   raise error on when using normal\_ or uniform\_ bool **not making bool tensors anymore**

## Tensor

-   move code into separate functions
-   handle higher dimensional tensors
-   improve API **done**

    -   allow not passing in one or more start or stops in index
    -   If indices aren't passed, all items in that dimension should be included
        -   do this by setting defaults in index to -1 then processing them to be the size of that dimension
    -   use structs for indexing params and storing sizes and shapes
    -   indexing should be with array of initilizer lists; each list specifies start and end positions for a dimension
    -   generalize functions to work with an arbitary number of dimensions
    -   infer stride in constructor

-   indexing **done**

    -   (row_begin, row_end] and (col_begin, col_end]
    -   How to return slices from an array instead of just single values?

        -   stride code: https://github.com/ThinkingTransistor/Sigma/blob/fe645441eb523996d3bfc6de9ad72e814a146195/Sigma.Core/MathAbstract/NDArrayUtils.cs#L24

    -   use chars for indices with ":" for all; cast all others to ints?

-   implementation details **done**

    -   store data, sizes, and strides in separate structs
    -   dimensions

-   reshape **done**

-   tensor.zeros/random **done**

    -   normal and uniform distributions
    -   specify means and stddevs

-   dtype **done**

-   broadcasting **done**

    -   both tensors need broadcasting; (2, 1, 2) (1, 2, 1)
    -   generalize broadcasting function for multiple operators
    -   stride should be 0 on broadcasted dimensions

    -   find what the size of the resulting tensor should be
    -   clone both tensors a and b
    -   change their sizes to match the size of the resulting tensor
    -   change the elements of strides that have been broadcasted in sizes to be 0
    -   create a new vector to hold the results
    -   iterate over the new sizes
    -   for each lowest level for loop, use get_physical_idx to convert the symbolic idxs to actual idxs and add the elements from both tensors
    -   return a new tensor

*   operators **done**

    -   same sizes and broadcasting

*   \+ \- / \* (by scalars, tensors, etc) **done**

*   operability between tensors of different types
*   return dtype **done**
*   return size **done**

    -   use structs to store sizes and strides **won't do**

*   pow(), sqrt(), exp(), log(), log10(), abs(), cos(), sin(), tan() **done**

-   .sum() **done**
-   .mean() **done**
-   sum and mean over dims **done**
    -   -1 dim **done**
-   max(), argmax(), min(), argmin(), **done**

-   mat dot bmm

    -   for bmm
    -   make sure all dims are same except last two
    -   iterate over all dimensions except the last two dimensions
    -   for each of these array slice lhs and rhs to get dims of (a, b) and (b, c)
    -   matmul the array slice and append to new_data

-   in place operations on slice
-   in slicing allow not passing all indices

-   view that returns new tensor so chaining can work **done**
-   add an unsqueeze(dim) function **done**
-   transpose

-   change dtype

*   concat **done**

    -   go over new shape
    -   get slice from
    -   insert elements into vector, expanding it

    -   if dim is dim
        -   select row/col of each tensor {0, -1} on that dim
        -   concat data and append
    -   concating axis 1
    -   cat all elements in dim of tensors
    -   dim 0 means to add the array after the current one
    -   dim1 means to add the first row of the array after the current one

*   store the data as a pointer and return a pointer whenever indexing
*   also handle cloning tensors **done**
*   copying **done**

## things to know

-   template classes must be defined and implemented in headers
    -   https://stackoverflow.com/questions/1724036/splitting-templated-c-classes-into-hpp-cpp-files-is-it-possible

## Structure

-   .h files in include/
-   .cpp files in L2/
-   compiled library is in lib/
-   use #pragma once in header files
-   for additional files, add the path to the cpp file to add_library in CMakeLists.txt

## Debugger

-   https://code.visualstudio.com/docs/cpp/config-wsl
-   https://github.com/microsoft/vscode-cpptools/issues/2998 (external_console: false)
-   https://github.com/microsoft/vscode-cpptools/issues/2998 ("pipeProgram": "bash.exe")
-   compile with -g flag to use debugger
-   can't use debugger on library files

## Running

### manually

-   g++ -g -o build/L2.o -c L2/L2.cpp
-   ar rcs build/libL2.a build/L2.o
-   g++ -g -o main.out main.cpp -Lbuild -lL2

### cmake

-   "ctrl-shift-b" or:

-   mkdir build
-   cd build
-   cmake .. -DCMAKE_INSTALL_PREFIX=../
-   make
-   make install
-   cd ..
-   g++ -g -o main.out main.cpp -Llib -lL2
