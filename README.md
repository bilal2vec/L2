# L2

A C++ deep learning library

## ToDo

-   namespace
-   tensors
-   error handling and checking

## Done

-   sum_sizes => sizes_to_n_elements

## Tensor

-   move code into separate functions
-   handle higher dimensional tensors
-   improve API

    -   allow not passing in one or more start or stops in index
    -   If indices aren't passed, all items in that dimension should be included
        -   do this by setting defaults in index to -1 then processing them to be the size of that dimension
    -   use structs for indexing params and storing sizes and shapes
    -   indexing should be with array of initilizer lists; each list specifies start and end positions for a dimension
    -   generalize functions to work with an arbitary number of dimensions
    -   infer stride in constructor

-   indexing

    -   (row_begin, row_end] and (col_begin, col_end]
    -   How to return slices from an array instead of just single values?

        -   stride code: https://github.com/ThinkingTransistor/Sigma/blob/fe645441eb523996d3bfc6de9ad72e814a146195/Sigma.Core/MathAbstract/NDArrayUtils.cs#L24

    -   use chars for indices with ":" for all; cast all others to ints?

-   implementation details

    -   store data, sizes, and strides in separate structs
    -   dimensions

-   reshape

-   tensor.zeros/random **done**

    -   normal and uniform distributions
    -   specify means and stddevs

-   dtype

-   operators
-   \+ \- / \* (by scalars, tensors, etc)
-   mat dot

-   store the data as a pointer and return a pointer whenever indexing
-   also handle cloning tensors
-   copying

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
