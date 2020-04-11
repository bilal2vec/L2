# Notes

-   no ints or primitive types in generic parameters
    -   there is an rfc, but it was opened in 2017
-   https://www.lpalmieri.com/

## Tensor

### slicing

-   use enum to store diff between two types of indices
    -   [start, end)
    -   -1

### Todo

-   change col-major to row-major
-   impl iterator
-   pre-allocate size of new vectors
-   add checks for slicing and tensor creation
-   right now, `calc_shape_from_slice` will slice a tensor of shape `[2, 2, 2]` to `[1, 1, 1]` if slicing it on `[[0, 1], [0, 1], [0, 1]]`

### Done

-   use vectors to store data
-   separate the view and data parts of tensor to not have to copy everything?
    -   use pointers?
    -   cant use (mut) reference because you need to initialize data somehow, it must belong to the tensor that created it
    -   tensor must be able to mutate data to do in-place operations
    -   option to do in-place or copy
        -   autograd needs copy
            -   https://discuss.pytorch.org/t/how-in-place-operation-affect-autograd/31173
        -   but some ops like `x[:10] = 0` should be able to be done inplace
    -   so don't need to separate data and tensor, but keep in mind for future if you want to do more (or multiple) inplace ops
    -   make two, `slice` and `slice_mut`
    -   just make copies immutable to make it simple

### Notes

-   vector's overhead shouldn't be much
-   maybe set vec to not allocate extra memory
-   vec is contigous in memory
-   naive slicing is too slow
-   numpy sometimes makes a copy when slicing to keep stuff contigous
-   naive approach slows down with size
-   it looks like numpy will make a copy when slicing
-   dont' use `[start:stop]` syntax for rust, use `.slice(!vec[start:stop])`
-   you should be able to do: **will make copies to make it simple**
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
    -   how ndarray does it https://stackoverflow.com/questions/50400966/is-there-a-rust-ndarray-equivalent-for-numpy-arithmetic-on-a-slice
-   numpy notes
    -   http://scipy-lectures.org/advanced/advanced_numpy/#indexing-scheme-strides
    -   https://ipython-books.github.io/45-understanding-the-internals-of-numpy-to-avoid-unnecessary-array-copying/
-   ndarray heap (https://users.rust-lang.org/t/ndarray-stack-and-heap-memory-and-overhead/25254)

## Resources

-   https://dev.to/erikinapeartree/rust-for-machine-learning-simd-blas-and-lapack-4dcg
-   https://docs.rs/rayon/1.3.0/rayon/
-   https://www.google.com/search?q=rust+ndarray+simd&oq=rust+ndarray+simd&aqs=chrome..69i57.3773j0j7&sourceid=chrome&ie=UTF-8
