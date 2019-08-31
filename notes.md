## ideas

-   autodiff shouldn't rely on how Tensor is implemented

## ToDo

-   check numerical correctness **ongoing**
-   shape checks on layers.forward() and backward() _v2_
-   better error handling _v2_
-   make Tensor::cat a static method? _v2_
-   cant do <double> <op> <Tensor> _v2_
-   documentation

## Library notes

-   Parameter class **done**

    -   tensor
    -   grad

-   init class **done**

-   Layer class **done**

    -   Sequential **done**

        -   object slicing means that only the layer part of Linear gets saved
        -   need to use polymorphism
        -   virtual base class
        -   for that, it means that we need to convert the vector of layers to vectors of pointers in the constructor to save it in layers and dereference the pointers to get the params
        -   then in forward, go over each pointer, dereference it, polymorphism makes sure that the forward method we call is on linear even though in the for loop it is defined as a Layer (https://stackoverflow.com/questions/2391679/why-do-we-need-virtual-functions-in-c)
        -   backward()
            -   sequential's parameters are initialized by copying over the the parameters from it's layers. this becomes a problem when it's layer's parameter's gradients change but don't change Sequential's own parameter's gradients
            -   so to fix, go over layers, copying over all the parameters to overwrite sequential's parameters

    -   linear **done**

        -   backward()
            -   changes to weight.grad and bias.grad don't change the grads of the elements in the parameters vector which is what gets used in sequential
            -   manually copied over weights and bias to Linear's parameters to fix

    -   conv
    -   rnn
    -   activations

        -   sigmoid **done**
        -   relu
        -   dropout

    -   parameters **done**
    -   forward() **done**
    -   backward() **done**
    -   destructor **done**

    -   problem: **fixed**
    -   sync changes to Sequential's parameters to each of it's layer's parameters
    -   in backprop, layer params are copied to sequential
    -   sequential params are updated
    -   layer params are now out of date
    -   on the next forward and backprop, layer params haven't been changed, they're identical since init

    -   Fixes

        -   1: **better and works**
        -   change sequential to not store it's own params
        -   remove extra code in backward()
        -   add a custom update() for it where it just calls update on all it's layers
        -   this means: no param syncing required

        -   problem

            -   in linear::backward(), weights and bias are never modified and are copied to layer::paramters on each backward() to save the gradient

            -   fix **works**
            -   make layer::update virtual and create a update() method for each derived class that calls layer::update then copies layer::parameters to weights and bias

        -   2:
        -   create custom update() that copies over new params to each of it's layers

*   loss class **done**

    -   all loss classes are derived from the Loss class
    -   no forward() or backward()
    -   instead, operator() which returns a tuple; loss and derivative

    -   mse **done**
    -   crossentropy
    -   bce
    -   bce with logits

*   optimizer class **done**

    -   call()

*   train class **done**

    -   takes

        -   nn
        -   loss
        -   optimizer
            -   learning rate
        -   dataset
        -   dataloader
        -   predict **done**

    -   drops last batch if batch_size % length != 0

*   dataset class _v2_

    -   for now simple 2 class classification using 2 gaussian clusters
    -   stores data and iterate

*   dataloader class _v2_

    -   takes
        -   dataset
        -   batch size
        -   shuffle

*   tensors _v2_

    -   store data as pointers
    -   copy semantics

*   make it more like xtensor _v2_
    -   https://medium.com/@wolfv/the-julia-challenge-in-c-21272d36c002
    -   https://medium.com/@johan.mabille/how-we-wrote-xtensor-1-n-n-dimensional-containers-f79f9f4966a7
    -   https://medium.com/@johan.mabille/how-we-wrote-xtensor-2-n-access-operators-57e8e3852263
    -   https://medium.com/@johan.mabille/how-we-wrote-xtensor-3-n-the-constructors-65a177260638
    -   https://medium.com/@johan.mabille/how-we-wrote-xtensor-4-n-value-semantics-6baa6856d313

## Done

-   error handling and checking
-   only print out part of long tensors
-   cat tensors wrong shape
-   handle errors when dividing ints
-   change some code to use iterators over vectors instead of for loops
-   use shape() more
-   more static functions
-   sum_sizes => shape_to_n_elements
-   namespace
-   try removing broadcastable_with
-   raise error on when using normal\_ or uniform\_ bool **not making bool tensors anymore**
-   default constructor for tensor;
-   inplace operations on tensors **done**
    -   append inplace ops with "\_" to distinguish them
-   create initializers
-   check sigmoid implementation to make sure it's sigmoiding each el seperately **it is**
-   check if reset() works **does**
-   refactor code to use namespaced keywords **wont do**
-   backward() in sequential shouldn't return anything **wont do**

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

-   mat bmm

    -   for bmm
    -   make sure all dims are same except last two
    -   iterate over all dimensions except the last two dimensions
    -   for each of these array slice lhs and rhs to get dims of (a, b) and (b, c)
    -   matmul the array slice and append to new_data

-   dot **wont do**

-   in place operations on slice **wont do**
-   in slicing allow not passing all indices **done**

-   view that returns new tensor so chaining can work **done**
-   add an unsqueeze(dim) function **done**
-   transpose **done**

-   change dtype **wont do, only double tensors**
    -   operations between int, float and double tensors
    -   casting tensors to types

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
