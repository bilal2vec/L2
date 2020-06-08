# Notes

-   no ints or primitive types in generic parameters
    -   there is an rfc, but it was opened in 2017
-   https://www.lpalmieri.com/

## Tensor

### Todo

-   printing of tensors and graph
-   combine 1,2,3,4d ops into one function
-   impl == on tensors
-   figure out macros and crates
-   autograd
    -   figure out if you want to use tensor ops in backward or vec ops
        -   decide what to do with new_with_parents
    -   tests
    -   ops
        -   slice
        -   view
        -   pow
        -   sqrt
        -   exp
        -   log10
        -   log
        -   abs
        -   sin
        -   cos
        -   tan
        -   sum
        -   mean
        -   max
        -   min
        -   argmax
        -   argmin
        -   matmul
        -   concat
        -   transpose
        -   div
-   compare to np and ndarray
-   redo error handling
-   const generics for compile time errors

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
-   right now, `calc_shape_from_slice` will slice a tensor of shape `[2, 2, 2]` to `[1, 1, 1]` if slicing it on `[[0, 1], [0, 1], [0, 1]]`
    -   this means that strides for 3+ d tensors can be wrong
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
-   pre-allocate size of new vectors
-   add checks for slicing and tensor creation
-   validation should happen at initialization
-   fix error handling and structure
-   panic!("Invalid slice")
-   should panicing be done in main?
-   use smarter indexing
    -   -1 for all values
    -   not needing to say either first or last vals
    -   slicing with an empty slice will return a copy of the orignal tensor
-   reduce number of &[..]
-   reshape
-   benchmarks for previous stuff
-   broadcasting
-   ops
    -   should be like `let c = &a + &b;` and `let c = l2::add(a, b);`
    -   elementwise ops are comparable to numpy until ~ 4096x4096
    -   other should be reference
    -   ops won't return Result<Tensor, TensorError>
        -   that would mean that you would need to run `let x = (a + b).unwrap();`
        -   will panic instead
    -   all ops are tensor-tensor ops
    -   element-wise ops
-   self-ops
    -   pow
    -   sqrt
    -   exp
    -   log
        -   e
        -   10
    -   abs
    -   sin
    -   cos
    -   tan
-   other ops
    -   _dim ops are slower than numpy (30us vs 300us), numpy seems to cache stuff_
    -   _argmax and argmin return f32 tensors_
    -   over dim or all dims
    -   sum
    -   mean
    -   max
    -   min
    -   argmax
    -   argmin
-   use enum for ops?
-   matmul
    -   _about 100x slower than numpy_
    -   _wont implement broadcasting on matmul_
    -   batch matmul
-   check errors
-   concat
-   transpose
-   clone
-   normal
-   uniform
-   autodiff
    -   accumulate gradients
    -   hold reference to parents
        -   lhs and rhs
    -   hold gradient
    -   know grad functions for each op
    -   account for creators
    -   do tensor-tensor op without grad tracking for backend
    -   derivative is `RefCell<Option<Tensor>>`
    -   `borrow_mut()` on derivative and assign to it
    -   use `Rc::new(Tensor::new(Rc::clone(&t))),` so references can be to more than one tensor?
    -   dont need to use option?
        -   no
    -   use a wrapper for grad tracking tensors?
        -   save memory on normal tensors
    -   mark nodes as evaluated?
        -   prevent having to recurse through shared graph multiple times
        -   topological sort
            -   done
        -   in backward mutate lhs and rhs parent's grad not own
            -   works

```
   a
  / \
 *   *
/     \
b      c
\     /
 +   +
  \ /
   d

da =  dd/db + dd/dc
but this will recompute the backwards pass for all the graph above a
topological sorting accumulates gradients for a before going further up the computation graph
```

### wont do

-   change col-major to row-major
-   impl iterator
-   replace indices slice with enum
-   tensor::new shouldn't return result
-   use enum to store diff between two types of indices
    -   [start, end)
    -   -1
-   prevent having to reallocate memory on each backwards pass
    -   clear unneeded memory as soon as you can?

### Notes

-   L2 is competitive with numpy for small tensors
-   L2 copies slices since thats whats needed for autograd
    -   by default, numpy returns a view
-   takes about 6s to slice all elements from a 64x64x64x64 tensor
-   speed of slicing/allocating cannot be optimized. Numpy takes about 2x the time that l2 does because l2 will always copy a slice. Numpy's native slices are views, but copies are needed for autograd.

## Resources

-   https://dev.to/erikinapeartree/rust-for-machine-learning-simd-blas-and-lapack-4dcg
-   https://docs.rs/rayon/1.3.0/rayon/
-   https://www.google.com/search?q=rust+ndarray+simd&oq=rust+ndarray+simd&aqs=chrome..69i57.3773j0j7&sourceid=chrome&ie=UTF-8
-   https://stackoverflow.com/questions/39477684/should-i-avoid-unwrap-in-production-application/39478185#39478185

### backprop

-   https://datascience.stackexchange.com/questions/20139/gradients-for-bias-terms-in-backpropagation
-   https://cs231n.github.io/optimization-2/
-   https://cs231n.github.io/neural-networks-case-study/#grad
-   https://stackoverflow.com/questions/38082835/backpropagation-in-gradient-descent-for-neural-networks-vs-linear-regression
-   https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b
-   https://stackoverflow.com/questions/38082835/backpropagation-in-gradient-descent-for-neural-networks-vs-linear-regression
-   https://github.com/bkkaggle/L2/tree/master#acknowledgements

### Autodiff

-   https://github.com/karpathy/micrograd
-   https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation
    -   https://github.com/ibab/rust-ad
    -   https://github.com/Rufflewind/revad/blob/eb3978b3ccdfa8189f3ff59d1ecee71f51c33fd7/revad.py
    -   https://github.com/srirambandi/ai
-   https://discuss.pytorch.org/t/is-pytorch-autograd-tape-based/13992/3
-   https://www.reddit.com/r/MachineLearning/comments/8ep130/d_how_does_autograd_work/
-   https://github.com/mattjj/autodidact
-   https://github.com/karpathy/recurrentjs
-   https://github.com/karpathy/randomfun
-   https://medium.com/@ralphmao95/simple-autograd-implementation-understand-automatic-differentiation-hand-by-hand-9e86f6d703ab
-   https://evcu.github.io/ml/autograd/
-   https://blog.paperspace.com/pytorch-101-understanding-graphs-and-automatic-differentiation/
-   https://github.com/maciejkula/wyrm
-   https://medium.com/@maciejkula/building-an-autodifferentiation-library-9ccf32c7a658
-   https://github.com/evcu/numpy_autograd/blob/master/my_autograd.py#L147
-   https://github.com/evcu/numpy_autograd/blob/master/Autograd.ipynb
-   https://cs231n.github.io/optimization-2/

### Rust

-   https://nora.codes/post/what-is-rusts-unsafe/

### SIMD

-   https://opensourceweekly.org/issues/7/
