# Tl;dr

---

This blog post shows you, step-by-step, how to build a fast [PyTorch](https://pytorch.org/)-style machine learning library in the [Rust programming language](https://www.rust-lang.org/). This blog post is based on a library called [L2](https://github.com/bilal2vec/L2) that I finished working on a while ago.

I [compiled](#resources) quite a long list of blog posts, articles, and GitHub repos that I found useful when I was working on L2, so take a look at that if that's the type of stuff you're interested in.

---

**Disclaimers**:

L2 is a small project I was working on during the summer before uni for fun [^1], so don't expect it to be production-ready or bug-free. [^2]

I'm going to assume that everyone who's reading this knows or uses Rust relatively well and is familiar with how PyTorch and TF work at a high level. If you want to learn about these topics or just brush up on some things that you aren't 💯 clear on, try looking through my [resources](#resources) section.

L2 is surprisingly fast especially since I didn't try very hard to optimize all the operators, it's usually only less than one order of magnitude slower than PyTorch in most of the benchmarks that I ran. L2 only supports a CPU backend at the moment since I'm not familiar enough with Rust to start working with CUDA and cuDNN. So far, it doesn't have any PyTorch-style high level abstractions that are really useful for machine learning like PyTorch's `Parameter`, `Layer`, or `Module` classes. There might still be some bugs in the transpose operators and calling `.backward()` on broadcasted tensors. The autograd system won't automatically clear unused buffers once they've been used so this won't be as memory efficient as PyTorch.

I wrote dozens of tests and benchmarks to make sure that L2 was working properly when I was developing it. I'm going to be omitting tests in this blog post and instead just going to show some example code in `src/bin/main.rs`.

**If you just want to skip to the code part of the tutorial, click [here](#baseline)**

# Background

---

Last summer [^3], I [wrote](https://github.com/bilal2vec/L2/tree/c%2B%2B) a machine learning library as a way of getting better at using C++. The library wasn't really that advanced (I didn't have an autograd system like PyTorch does, instead I just did the backprop calculations by hand for each layer) or very fast (I pretty much passed everything by value and didn't really put a focus on making my code fast and performant), but it was a good way at getting a lot of experience working with a lower level language like c++ that I'd never used before and I learned a lot about how machine learning libraries like Pytorch and Tensorflow work behind the scenes.

This summer, I did a complete rewrite of L2, this time in Rust, with a focus on making it as close to Pytorch as I could (speed and feature wise) and got to learn about and implement a lot of interesting and cool features that are used in all the popular machine learning libraries today.

I'm pretty satisfied with how L2 turned out, here's the pitch I wrote for it on my GitHub repo:

L2 is a Pytorch-style Tensor+Autograd library written in Rust. It contains a multidimensional array class, `Tensor`, with support for strided arrays, numpy-style array slicing, broadcasting, and most major math operations (including fast, BLAS-accelerated matrix multiplication!). On top of this, L2 has a built-in efficient graph-based autograd engine that keeps track of all operations performed on a tensor and topologically sorts and traverses the graph to compute the gradients.

I'm also pretty happy with how the user-facing API of the library turned out:

---

```rust
use l2::tensor::*;

let x: Tensor = Tensor::normal(&[2, 4], 0.0, 1.0)?;
let y: Tensor = Tensor::normal(&[4, 1], 0.0, 1.0)?;

let z: Tensor = l2::matmul(&x, &y)?;

z.backward();

println!("{}", z);
```

---

# Let's get started

---

So let's get started. I'm pretty much just copying down the installation instructions from the official [get started](https://www.rust-lang.org/learn/get-started) guide, so take a look at that if you want.

Install rustup to your computer:

---

```bash
bilal@Bilals-MacBook-Pro ~ % curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

---

Switch the default rust version to nightly:

---

```bash
bilal@Bilals-MacBook-Pro ~ % rustup default nightly
```

---

I'll be using my preferred text editor [VScode](https://code.visualstudio.com/) in this post, but feel free to use whatever editor you prefer.

I highly recommend using the (soon to become) official Rust extension for VScode, [Rust-analyzer](https://marketplace.visualstudio.com/items?itemName=matklad.rust-analyzer) instead of the old [RLS](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust) extension. Just install it from the marketplace and you should be ready to go.

Create a new Rust library called `l2` with:

---

```bash
bilal@Bilals-MacBook-Pro ~ % cargo new l2 --lib
```

---

Install clippy (Rust's official linter):

_You can take a look at all the lint rules and how to fix each one [here](https://rust-lang.github.io/rust-clippy/master/index.html)_

---

```bash
bilal@Bilals-MacBook-Pro ~ % rustup component add clippy
```

---

And change rust-analyzer to use clippy as its default linter by creating a `.vscode/settings.json` file and pasting this in it.

---

```json
{
	"rust-analyzer.checkOnSave.command": "clippy"
}
```

---

For debugging support, I use the [Code-LLDB](https://marketplace.visualstudio.com/items?itemName=vadimcn.vscode-lldb) extension, so install that as well.

create a `.vscode/launch.json` file and paste this into it:

---

```json
{
	"version": "0.2.0",
	"configurations": [
		{
			"type": "lldb",
			"request": "launch",
			"name": "Debug",
			"program": "${workspaceRoot}/target/debug/main",
			"args": [],
			"cwd": "${workspaceRoot}"
		}
	]
}
```

---

Add a rust binary target `src/bin/main.rs` that will be linked against our library at `src/lib.rs`. Your project should now have a directory structure like this:

---

```plain
.git/
.vscode/
    settings.json
    launch.json
src/
    bin/
        main.rs
    lib.rs
target/
.gitignore
Cargo.lock
Cargo.toml
```

---

We'll code up the library in `src/lib.rs` and any other files in the `src/` directory. We'll use `src/bin/main.rs` to interact with L2 as you would when using it in your own project.

---

# A simple baseline

---

Ok, so let's start by creating a simple `Tensor` struct and defining a few simple operations on it.

A `Tensor` is really just a multidimensional array. For this library, we'll keep it simple and restrict tensors to have at most $2$ dimensions (You'll see why later).

The simplest way to store a multidimensional array of say, dimensions $m \times n$ would be to create an array of length $m$ that holds a pointers to $m$ distinct arrays of length $n$, each holding the elements of a single row. This would be the simplest way to represent a `Tensor` but isn't really optimal when you need to create and process large `Tensors` quickly.

Most (if not all) people use _strided arrays_, where elements of a multidimensional array are layed out contigously in memory (the $m \times n$ `Tensor` would then be represented as single array of length $m * n$).

Take a look at http://blog.ezyang.com/2019/05/pytorch-internals/ for a good in-depth look into how PyTorch uses strided arrays. I'll summarize the main parts below:

Say you have a $2 \times 2$ `Tensor` like this:

$$
\begin{bmatrix}
   1 & 2 \\
   3 & 4
\end{bmatrix}
$$

If you wanted to represent this as a strided array, you could either store them in row-major or column-major order, storing either values from a single row or column contigously in memory (the same idea would still apply if you have a `Tensor` of more dimensions):

$$
\text {row-major:}
\begin{bmatrix}
   1, 2, 3, 4
\end{bmatrix}
$$

$$
\text {column-major:}
\begin{bmatrix}
   1, 3, 2, 4
\end{bmatrix}
$$

Most machine learning libraries like Numpy, PyTorch, and Tensorflow store Tensors in row-major order by default. This lets you quickly access the next element in the same row just by moving one element to the right in the `Tensor`. Column-major order isn't as commonly used, the only time I had to use it when I was integrating a BLAS library written in very optimized Fortran into L2 in order to use its super fast matrix multiplication implementations (using BLAS sped up my matrix multiplication code by about 200 times IIRC).

The choice of whether to store your data in column-major or row-major order depends on whether you prefer to have contigous access to elements in the first or last dimensions of your `Tensor`. For example, if you store a batch of $N$ three-channel image in a `Tensor` of dimensions ($256$, $256$, $3$), you would be able to either access the channels or the batch dimension contigously (i.e. have the elements in that dimension be next to each other in memory) depending on whether it's stored in row-major or column-major order.

The _stride_ for each dimension of a _strided array_ is the number of elements you want to skip between neighboring elements of a `Tensor` in a particular dimension. For example, our original `Tensor` of shape $\begin{bmatrix} 2, 2 \end{bmatrix}$ has strides of $\begin{bmatrix} 2, 1 \end{bmatrix}$.

This means that if we want to advance one element in the column dimension (from the element $1$ to the element $3$) of the _logical_ `Tensor`, we need to advance $2$ elements at a time in the _strided_ `Tensor`.

$$
\text {Logical Tensor:}

\begin{bmatrix}
   1 & \color{gray} 2 \\
   3 & \color{gray} 4
\end{bmatrix}
$$

$$
\text {Strided Tensor:}
\begin{bmatrix}
   1, \color{gray} 2, \color{white} 3, \color{gray} 4
\end{bmatrix}
$$

The same would be true for the other dimensions as well. If we want to advance one element in the row dimension (from the element $1$ to the element $2$) of the _logical_ `Tensor`, we would advance $1$ element in the _strided_ `Tensor`.

$$
\text {Logical Tensor:}

\begin{bmatrix}
   1 &  2 \\
   \color{gray} 3 & \color{gray} 4
\end{bmatrix}
$$

$$
\text {Strided Tensor:}
\begin{bmatrix}
   1,  2, \color{gray} 3, 4
\end{bmatrix}
$$

If we wanted to get the _physical_ location in memory of a specific element in the `Tensor` from the _logical_ location, we can simply "multiply each index with the respective stride for that dimension, and sum them all together" [^4]. So for an example, if we want to get the _physical_ index of the element at the _logical_ indices $[ 1, 1]$, we would calculate it like this:

$$
\text{logical index: } [\color{red} 1, \color{blue} 1 \color{white}] \ \text{strides: } [\color{red} 2, \color{blue} 1 \color{white}]
$$

<br />

$$
\text{physical index} = {\color{blue} 1} {\color{white} *} {\color{blue} 1} {\color{white} +} {\color{red} 1} {\color{white} *} {\color{red} 2}
$$

$$
\text{physical index} =  1 + 2 = 3
$$

$$
\text {element at [1, 1]} =
\begin{bmatrix}
   \color{gray} 1,  2,  3, \color{white} 4
\end{bmatrix}
$$

---

So now that we have that out of the way, let's start writing some code.

In this section, we'll make a basic `Tensor` struct the just creates and stores a strided array. We'll also take advantage of Rust's excellent error handling primitives to add robust error handling and add pretty printing of our `Tensors`.

Let's make a new file at `src/tensor.rs` to house our `Tensor` struct.

---

```rust
use crate::errors::TensorError;

use std::fmt;

#[derive(Debug, PartialEq)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

// Pretty print Tensors
impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let graph = format!("Value: {:?} \nShape: {:?}",
            self.data, self.shape);

        write!(f, "{}", graph)
    }
}

impl<'a> Clone for Tensor<'a> {
    fn clone(&self) -> Self {
        Tensor::new(self.data.clone(), &self.shape).unwrap()
    }
}

impl Tensor {
    // Calculate the number of elements in a tensor from the shape
    fn calc_tensor_len_from_shape(shape: &[usize]) -> usize {
        let mut length = 1;
        for i in shape {
            length *= i;
        }

        length
    }

    // calculate the strides for each dimension from the shape
    fn calc_strides_from_shape(shape: &[usize]) -> Vec<usize> {
        let mut strides = Vec::with_capacity(shape.len());

        let mut current_stride = 1;
        for i in shape.iter().rev() {
            strides.insert(0, current_stride);
            current_stride *= i;
        }

        strides
    }

    // Create a new tensor from some data with a specific shape
    pub fn new(data: Vec<f32>, shape: &[usize]) -> Result<Tensor,
        TensorError> {
        if data.len() == Tensor::calc_tensor_len_from_shape(shape)
            && !shape.is_empty()
            && shape.len() < 3
        {
            Ok(Tensor {
                data,
                shape: shape.to_vec(),
                strides: Tensor::calc_strides_from_shape(shape),
            })
        } else {
            Err(TensorError::InvalidTensor)
        }
    }
}
```

---

Let's add the error handling struct `TensorError` to `src/errors.rs`

---

```rust
// src/errors.rs

use std::error;
use std::fmt;

#[derive(Debug, Clone)]
pub enum TensorError {
    MaxDimsError,
    InvalidTensor,
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TensorError::MaxDimsError => write!(
                f,
                "L2 currently only supports
                tensors with up to 2 dimensions"
            ),
            TensorError::InvalidTensor =>
            write!(f, "Invalid parameters for Tensor"),
        }
    }
}

// This is important for other errors to wrap this one.
impl error::Error for TensorError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        // Generic error, underlying cause isn't tracked.
        None
    }
}
```

---

Add the relevant imports to `src/lib.rs` and `src/bin/main.rs`

---

```rust
// src/lib.rs

pub mod errors;
pub mod tensor;
```

---

And finally, lets test out our library by creating a simple `Tensor` in `src/bin/main.rs`

---

```rust
// src/bin/main.rs

use l2::errors::*;
use l2::tensor::*;

fn main() -> Result<(), TensorError> {
    let x = Tensor::new(vec![1.0, 2.0, 3.0], &[3])?;

    println!("{}", x);

    Ok(())
}
```

---

and run `cargo run` to see the output.

---

```bash
bilal@Bilals-MacBook-Pro l2 % cargo run
   Compiling l2 v0.1.0 (/Users/bilal/Desktop/l2)
    Finished dev [unoptimized + debuginfo] target(s) in 1.02s
     Running `target/debug/main`

Value: [1.0, 2.0, 3.0]
Shape: [3]
```

---

🎉! you now have a very simple machine learning library. Now that we have the general structure of the library set up, I\'ll be speeding up the pace of this blog post.

---

# Broadcasting

---

Storing a bunch of values in a `Tensor` is useless if we can't operate over them.

Before we can create some `Tensor`—`Tensor` operations, we need to implement _broadcasting_. I won't go into what exactly broadcasting is here, since there are a lot of better explanations out there. Numpy's [documentation](https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules) on their broadcasting rules is a good technical explanation.

One thing the numpy docs don't go into is how to implement broadcasting. I struggled with how to best implement it when I was making my original C++ version of the library last year [^5], but I eventually settled on the pretty simple and efficient solution of adding dimensions of size $1$ to the tensor with the fewer number of dimensions to make their shapes broadcastable, then setting the shapes and strides of a broadcasted dimension to $1$ and $0$ respectively. By doing it this way, the `Tensor` would use the same value across all values of a specific dimension.

---

```rust
// src/tensor.rs

impl Tensor {

    ...

    #[allow(clippy::ptr_arg, clippy::type_complexity)]
    fn broadcast(
        lhs_shape: &Vec<usize>,
        rhs_shape: &Vec<usize>,
    ) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>), TensorError> {

        // prepend lhs_shape with ones if the length of it is smaller than rhs_shape
        let lhs_shape = if lhs_shape.len() < rhs_shape.len() {
            let ones = vec![1; rhs_shape.len() - lhs_shape.len()];
            [&ones[..], &lhs_shape[..]].concat()
        } else {
            lhs_shape.clone()
        };

        // prepend rhs_shape with ones if the length of it is smaller than lhs_shape
        let rhs_shape = if rhs_shape.len() < lhs_shape.len() {
            let ones = vec![1; lhs_shape.len() - rhs_shape.len()];
            [&ones[..], &rhs_shape[..]].concat()
        } else {
            rhs_shape.clone()
        };

        let mut broadcasted_shape: Vec<usize> =
            Vec::with_capacity(lhs_shape.len());
        let mut broadcasted_lhs_strides: Vec<usize> =
            Tensor::calc_strides_from_shape(&lhs_shape);
        let mut broadcasted_rhs_strides: Vec<usize> =
            Tensor::calc_strides_from_shape(&rhs_shape);

        // go over each dimension of lhs and rhs
        for (i, (&lhs, &rhs)) in lhs_shape.iter()
            .zip(rhs_shape.iter()).enumerate() {
            // if both dimensions are the same,
            // the dimension of the broadcasted shape
            // for this dimension doesn't change
            if lhs == rhs {
                broadcasted_shape.push(lhs);

            // if the size of this dimension of lhs
            // is 1, set the strides of lhs for that
            // dimension to 0
            } else if lhs == 1 {
                broadcasted_shape.push(rhs);
                broadcasted_lhs_strides[i] = 0;

            // if the size of this dimension of rhs
            // is 1, set the strides of rhs for
            // that dimension to 0
            } else if rhs == 1 {
                broadcasted_shape.push(lhs);
                broadcasted_rhs_strides[i] = 0;

            // return an error if the tensors
            // aren't broadcastable
            } else {
                return Err(TensorError::BroadcastError);
            }
        }

        Ok((
            broadcasted_shape,
            broadcasted_lhs_strides,
            broadcasted_rhs_strides,
        ))
    }
```

```rust
// src/errors.rs

pub enum TensorError {
    ...
    BroadcastError,
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ...
            TensorError::BroadcastError =>
                write!(f, "Shapes are not broadcastable"),
        }
    }
}
```

---

Now that we've implemented broadcasting, we'll add some operations over `Tensor`s in the next section so we can try it out.

---

# Ops

---

Let's start by defining a struct `Ops` that we'll use to keep track of what operation should be performed on a tensor.

We'll be storing the `Tensor`—`Tensor` ops in an enum called `TensorOp`, but we'll wrap that in the `Ops` enum so we can add more different kinds of ops in the future (slicing, matmuls, transposes, etc).

---

```rust
// src/ops.rs

use std::fmt;

#[derive(Debug, PartialEq)]
pub enum TensorOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl fmt::Display for TensorOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorOp::Add => write!(f, "Add"),
            TensorOp::Sub => write!(f, "Subtract"),
            TensorOp::Mul => write!(f, "Multiply"),
            TensorOp::Div => write!(f, "Divide"),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum Ops {
    TensorOp(TensorOp),
}

impl fmt::Display for Ops {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Ops::TensorOp(tensor_op) =>
                write!(f, "{}", tensor_op),
        }
    }
}
```

---

And now let's add an `OpError` variant to our `TensorError` enum

---

```rust
// src/errors.rs

pub enum TensorError {
    ...
    OpError,
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ...
            TensorError::OpError =>
                write!(f, "Tensors cannot be operated on"),
        }
    }
}

```

---

Now that we have an `Ops` enum that we can use, let's integrate it into `Tensor`

---

```rust
// src/tensor.rs

use std::ops::{Add, Div, Mul, Sub};

use crate::ops::{Ops, TensorOp};

impl Tensor {

    ...

    // calculate the physical index of an element
    // from a `Tensor`'s logical indices and strides
    fn get_physical_idx(logical_indices: &[usize],
        strides: &[usize]) -> usize {
        let mut physical_idx = 0;

        for (i, idx) in logical_indices.iter().enumerate() {
            physical_idx += idx * strides[i];
        }

        physical_idx
    }

    // perform op on lhs and rhs
    fn op(lhs: &f32, rhs: &f32, op: &Ops) ->
        Result<f32, TensorError> {
        match op {
            Ops::TensorOp(TensorOp::Add) =>
                Ok(lhs + rhs),
            Ops::TensorOp(TensorOp::Sub) =>
                Ok(lhs - rhs),
            Ops::TensorOp(TensorOp::Mul) =>
                Ok(lhs * rhs),
            Ops::TensorOp(TensorOp::Div) =>
                Ok(lhs / rhs),
            _ => Err(TensorError::OpError),
        }
    }

    fn tensor_op(&self, other: &Tensor, op: Ops) ->
        Result<Tensor, TensorError> {
        // broadcast tensors
        let (new_shape, lhs_strides, rhs_strides) =
            Tensor::broadcast(&self.shape, &other.shape)?;

        if new_shape.is_empty() || (new_shape.len() > 3) {
            return Err(TensorError::MaxDimsError);
        }

        // allocate a new vector for the result of the op
        let mut new_data: Vec<f32> =
            Vec::with_capacity(Tensor::
                calc_tensor_len_from_shape(&new_shape));

        // call `Tensor::op()` on each element in the tensor
        for i in 0..new_shape[0] {
            if new_shape.len() == 1 {
                let op_result = Tensor::op(
                    &self.data[Tensor::
                        get_physical_idx(&[i], &lhs_strides)],
                    &other.data[Tensor::
                        get_physical_idx(&[i], &rhs_strides)],
                    &op,
                )?;

                new_data.push(op_result);
            } else {
                for j in 0..new_shape[1] {
                    let op_result = Tensor::op(
                        &self.data[Tensor::
                            get_physical_idx(&[i, j], &lhs_strides)],
                        &other.data[Tensor::
                            get_physical_idx(&[i, j], &rhs_strides)],
                        &op,
                    )?;

                    new_data.push(op_result);
                }
            }
        }

        Tensor::new(new_data, &new_shape)
    }
}
```

---

Let's also overload Rust's built-in `Add`, `Sub`, `Mul`, and `Div` traits for `Tensor` so we can use the native plus and minus operators on `Tensor`s: `let c: Tensor = a + b;`

---

```rust
// src/tensor.rs

...

impl Add for &Tensor {
    type Output = Tensor;

    fn add(self, other: &Tensor) -> Tensor {
        match self.tensor_op(other,
            Ops::TensorOp(TensorOp::Add)) {
            Ok(t) => t,
            Err(e) => panic!("{}", e),
        }
    }
}

impl Sub for &Tensor {
    type Output = Tensor;

    fn sub(self, other: &Tensor) -> Tensor {
        match self.tensor_op(other,
            Ops::TensorOp(TensorOp::Sub)) {
            Ok(t) => t,
            Err(e) => panic!("{}", e),
        }
    }
}

impl Mul for &Tensor {
    type Output = Tensor;

    fn mul(self, other: &Tensor) -> Tensor {
        match self.tensor_op(other,
            Ops::TensorOp(TensorOp::Mul)) {
            Ok(t) => t,
            Err(e) => panic!("{}", e),
        }
    }
}

impl Div for &Tensor {
    type Output = Tensor;

    fn div(self, other: &Tensor) -> Tensor {
        match self.tensor_op(other,
            Ops::TensorOp(TensorOp::Div)) {
            Ok(t) => t,
            Err(e) => panic!("{}", e),
        }
    }
}
```

---

Now that we have the ops implemented, all we need to do now is to add `ops.rs` as a module in `lib.rs`

---

```rust
// src/lib.rs

mod ops;
```

---

and let's try it out:

---

```rust
// src/bin/main.rs

fn main() -> Result<(), TensorError> {
    let a = Tensor::new(vec![1.0, 2.0, 3.0], &[1, 3])?;
    let b = Tensor::new(vec![1.0, 2.0, 3.0], &[3])?;

    let c = &a * &b;

    println!("{}", c);

    ...
```

---

Just run `cargo run` in your terminal to see the results:

---

```bash
bilal@Bilals-MacBook-Pro l2 % cargo run
    Finished dev [unoptimized + debuginfo] target(s) in 0.75s
    Running `target/debug/main`
Value: [1.0, 4.0, 9.0]
Shape: [1, 3]
```

---

# Autograd

---

We need to implement one more operator before we can start working on our autograd system: `.pow()`

---

```rust
// src/ops.rs
#[derive(Debug, PartialEq)]
pub enum Ops {
    ...
    Pow(f32),
}

impl fmt::Display for Ops {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) ->
        fmt::Result {
        match self {
            ...
            Ops::Pow(pow) => write!(f, "Pow: {}", pow),
        }
    }
}
```

```rust
// src/tensor.rs

impl Tensor {

    ...

    pub fn pow(&self, exp: f32) ->
        Result<Tensor, TensorError> {
        let new_data = self.data.iter()
            .map(|val| val.powf(exp)).collect();

        Tensor::new(new_data, &self.shape)
    }
}
```

---

Now that that's out of the way, lets go on to the fun stuff: Autograd. We'll be implementing a simple but efficient graph based autograd system similar to what PyTorch uses.

Every `Tensor` struct will hold field(s) that hold references to its parent(s) as well as a field holding the op that was used to create it and a `Vec<f32>` to store its gradient.

Since we're using Rust, a language famous for its focus on guaranteeing memory safety at compile time, we'll need to put a little bit of thought into how to implement all this. A `Tensor` may or may not have either one or two immutable references to its parent `Tensors` and also may or may not have been created using an `Op`. We also need a way to compute a `Tensor`'s gradient wrt to its children.

To make everything simple, we'll wrap the gradient of a `Tensor` in a `RefCell` so we can safely change its value by calling `.borrow_mut()` without needing to keep a mutable reference to it. _Keeping a mutable reference might not be possible if one `Tensor` has two distinct children — Rust only allows one mutable reference to be in scope at a time._

Let's get started by adding a few field to our original `Tensor` struct:

---

```rust
// src/tensor.rs

use std::cell::RefCell;

pub struct Tensor {
    ...

    track_grad: bool,

    lhs_parent: Option<&Tensor>,
    rhs_parent: Option<&Tensor>,
    create_op: Option<Ops>,
    derivative: RefCell<Vec<f32>>,
}
```

---

If you add this and press `⌘-S`, you'll probably see that rust-analyzer starts throwing out dozens of warnings and errors. Now that we're storing references to other `Tensor`s inside our `Tensor`, we need to add lifetime parameters to our struct so the Rust compiler can make sure that these references don't go out of scope during any part of our program.

If you're using VSCode with rust-analyzer like I am, fixing lifetime errors in Rust is pretty painless when the compiler literally guides you through it and tells you where the problem is, why it exists, and how to fix it :)

Here's a diff showing the changes that I had to make:

---

```rust
// src/tensor.rs

-pub struct Tensor {
+pub struct Tensor<'a> {
    ...

-    lhs_parent: Option<&Tensor,
-    rhs_parent: Option<&Tensor,
+    lhs_parent: Option<&'a Tensor<'a>>,
+    rhs_parent: Option<&'a Tensor<'a>>,
}

-impl fmt::Display for Tensor {
+impl<'a> fmt::Display for Tensor<'a> {
    ...
}

-impl Clone for Tensor {
+impl<'a> Clone for Tensor<'a> {
    ...
}

-impl Tensor {
+impl<'a> Tensor<'a> {
    ...

-    pub fn new(data: Vec<f32>, shape: &[usize])
-        -> Result<Tensor, TensorError> {
+    pub fn new<'b>(data: Vec<f32>, shape: &[usize])
+        -> Result<Tensor<'b>, TensorError> {
        if data.len() == Tensor::calc_tensor_len_from_shape(shape)
            && !shape.is_empty()
            && shape.len() < 3
        {
            Ok(Tensor {
                data,
                shape: shape.to_vec(),
                strides: Tensor::calc_strides_from_shape(shape),
+                track_grad: true,
+                create_op: None,
+                derivative: RefCell::new(
+                    vec![0.0; Tensor::calc_tensor_len_from_shape(shape)]),
+                lhs_parent: None,
+                rhs_parent: None,
            })
        } else {
            Err(TensorError::InvalidTensor)
        }
    }

}

-impl Add for &Tensor {
-    type Output = Tensor;
+impl<'a> Add for &'a Tensor<'a> {
+    type Output = Tensor<'a>;

-    fn add(self, other: &Tensor) -> Tensor {
+    fn add(self, other: &'a Tensor) -> Tensor<'a> {
        ...
    }
}

-impl Sub for &Tensor {
-    type Output = Tensor;
+impl<'a> Sub for &'a Tensor<'a> {
+    type Output = Tensor<'a>;

-    fn sub(self, other: &Tensor) -> Tensor {
+    fn sub(self, other: &'a Tensor) -> Tensor<'a> {
        ...
    }
}

-impl Mul for &Tensor {
-    type Output = Tensor;
+impl<'a> Mul for &'a Tensor<'a> {
+    type Output = Tensor<'a>;

-    fn mul(self, other: &Tensor) -> Tensor {
+    fn mul(self, other: &'a Tensor) -> Tensor<'a> {
        ...
    }
}

-impl Div for &Tensor {
-    type Output = Tensor;
+impl<'a> Div for &'a Tensor<'a> {
+    type Output = Tensor<'a>;

-    fn div(self, other: &Tensor) -> Tensor {
+    fn div(self, other: &'a Tensor) -> Tensor<'a> {
        ...
    }
}
```

---

_Note: you might notice that you don't need to declare a lifetime parameter on `other` in the `impl` blocks for `Add`, `Sub`, `Mul`, and `Div`. I'm including the lifetime parameters here since we'll need to add them in the next step since the output of `Tensor::tensorop()` will store a reference to `other` as one of its parents. This means that lifetime parameters will be needed to make sure that the reference to the parent remains valid for the full lifetime of the output._

Now that we've satisfied the Rust compiler, let's modify `Tensor::tensor_op()` and `Tensor::pow()` to use the new struct fields we just added.

---

```rust
// src/tensor.rs

impl<'a> Tensor<'a> {
    ...

-    fn tensor_op(&self, other: &Tensor, op: Ops)
-       -> Result<Tensor, TensorError> {
+    fn tensor_op(&'a self, other: &'a Tensor, op: Ops)
+        -> Result<Tensor, TensorError> {

        ...

-       Tensor::new(new_data, &new_shape)
+        Ok(Tensor {
+            data: new_data,
+            shape: new_shape.to_vec(),
+            strides: Tensor::
+                calc_strides_from_shape(&new_shape),
+            track_grad: true,
+            create_op: Some(op),
+            derivative: RefCell::new(
+                vec![0.0; Tensor::calc_tensor_len_from_shape(&new_shape)]),
+            lhs_parent: Some(self),
+            rhs_parent: Some(other),
+        })

    }

    ...

    pub fn pow(&self, exp: f32) -> Result<Tensor, TensorError> {

        ...

-       Tensor::new(new_data, &new_shape)
+        Ok(Tensor {
+            data: new_data,
+            shape: self.shape.to_vec(),
+            strides: Tensor::calc_strides_from_shape(&self.shape),
+            track_grad: true,
+            create_op: Some(Ops::Pow(exp)),
+            derivative: RefCell::new(
+                vec![0.0; Tensor::calc_tensor_len_from_shape(&self.shape)]),
+            lhs_parent: Some(self),
+            rhs_parent: None,
+        })
    }
}

```

---

Ok, we're halfway there! We can now represent a sequence of operations as a computation graph. Let's update our pretty-printing code to print out the structure of our internal representation (IR) of the computation graph.

This probably isn't the most elegant way of implementing this but it works and I'm not motivated enough right now to try and improve it.

```rust
// src/tensor.rs

impl<'a> fmt::Display for Tensor<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn recurse(tensor: &Tensor, level: usize) -> String {
            let indent = "  ".to_string().repeat(level);

            let lhs = match tensor.lhs_parent {
                Some(t) => recurse(t, level + 1),
                None => "None".to_string(),
            };

            let rhs = match tensor.rhs_parent {
                Some(t) => recurse(t, level + 1),
                None => "None".to_string(),
            };

            let op = match &tensor.create_op {
                Some(t) => format!("{}", t),
                None => "None".to_string(),
            };

            format!(
                "\n{}Value: {:?} \n{}Shape: {:?} \n{}Lhs: {} \n{}Rhs: {} \n{}Op: {} \n{}TrackGrad: {:?} \n{}Derivative: {:?}",
                indent,
                tensor.data,
                indent,
                tensor.shape,
                indent,
                lhs,
                indent,
                rhs,
                indent,
                op,
                indent,
                tensor.track_grad,
                indent,
                *(tensor.derivative.borrow())
            )
        }

        let graph = recurse(self, 0);

        write!(f, "{}", graph)
    }
}
```

```rust
// src/bin/main.rs

fn main() -> Result<(), TensorError> {
    let a = Tensor::new(vec![1.0, 2.0, 3.0], &[1, 3])?;
    let b = Tensor::new(vec![1.0, 2.0, 3.0], &[3])?;

    let c = &a * &b;

    let d = c.pow(2.0)?;

    println!("{}", d);

    Ok(())
}
```

---

Let's run this and take a look at the resulting IR:

---

```bash
bilal@Bilals-MacBook-Pro l2 % cargo run
Value: [1.0, 16.0, 81.0]
Shape: [1, 3]
Lhs:
  Value: [1.0, 4.0, 9.0]
  Shape: [1, 3]
  Lhs:
    Value: [1.0, 2.0, 3.0]
    Shape: [1, 3]
    Lhs: None
    Rhs: None
    Op: None
    TrackGrad: true
    Derivative: [0.0, 0.0, 0.0]
  Rhs:
    Value: [1.0, 2.0, 3.0]
    Shape: [3]
    Lhs: None
    Rhs: None
    Op: None
    TrackGrad: true
    Derivative: [0.0, 0.0, 0.0]
  Op: Multiply
  TrackGrad: true
  Derivative: [0.0, 0.0, 0.0]
Rhs: None
Op: Pow: 2
TrackGrad: true
Derivative: [0.0, 0.0, 0.0]
```

---

Maybe it's not the nicest looking graph, but it works well for when you're trying to visually verify that your gradients are being calculated correctly.

---

Now that we have a computation graph, we need to find a way to backpropogate through it.

The simplest way would be to recursively call a function named `backward()` on the tensor you want to calculate the gradient with respect to. `backward()` would first take the gradient of the current tensor (the gradient of the output tensor would be with respect to itself so its gradient is $1$) then use it to calculate (and accumulate, if necessary) the gradient of its parent(s) before calling `.backward()` on the parent `Tensor`(s) to recursively calculate the gradient for the entire computation graph.

There are a couple of problems with this:

First, recursively calling `.backward()` on the entire computation graph would be very memory-inefficient.

Second, if the computation graph has multiple branches (like in a Resnet), the backwards pass over the computation graph will have to be computed multiple times as the gradients for the parent `Tensor` of each branch in the network are accumulated. Doing it this way would have make computing the backwards pass _very_ slow and inefficient.

Luckily, there is a better way of doing this. If we topologically sort and reverse the graph so that all the `Tensor`s are ordered in a way so that the gradients for all child `Tensor`s of a certain `Tensor` have already been computed and the gradient for the current `Tensor` has already been accumulated (if necessary), we won't have to re-compute any parts of the graph.

Let's see how we could implement this in Rust:

---

```rust
// src/tensor.rs

impl<'a> Tensor<'a> {

    ...

    pub fn backward(&self) {
        // from https://github.com/evcu/numpy_autograd/blob/master/my_autograd.py#L147
        let mut seen: Vec<&Tensor> = Vec::new();
        let mut sorted: Vec<&Tensor> = Vec::new();

        fn topological_sort<'a>(
            vr: &'a Tensor,
            seen: &mut Vec<&Tensor<'a>>,
            sorted: &mut Vec<&Tensor<'a>>,
        ) {
            if !seen.contains(&vr) && (vr.lhs_parent.is_some()
                    || vr.rhs_parent.is_some()) {
                seen.push(vr);

                if vr.lhs_parent.is_some() {
                    topological_sort(vr.lhs_parent.unwrap(),
                        seen, sorted);
                }
                if vr.rhs_parent.is_some() {
                    topological_sort(vr.rhs_parent.unwrap(),
                        seen, sorted);
                }

                sorted.push(vr);
            }
        }

        // Topologically sort the computation graph
        topological_sort(&self, &mut seen, &mut sorted);

        // reverse it
        sorted.reverse();

        // Set the derivative of the output of the computation
        // graph to itself to equal 1 (usually the derivative
        // of the loss wrt itself)
        *sorted[0].derivative.borrow_mut() = vec![1.0;
            Tensor::calc_tensor_len_from_shape(&sorted[0].shape)];

        for t in sorted.iter() {
            t.grad()
        }
    }
}
```

---

The `.grad()` function doens't exist yet, but its purpose is to take the gradient of the current `Tensor` `t` and use it to compute the gradients of its parent(s). Since we wrapped the `derivative` field of `Tensor` in a `RefCell()`, we can use something like `*lhs_parent.borrow_mut() = gradient;` to safely mutate the parent's gradient.

Here's how I did it:

---

```rust
// src/tensor.rs

impl<'a> Tensor<'a> {
    fn grad(&self) {
        // get the gradient of the derivative of self wrt output
        // d_x/d_loss
        let d = Tensor::new(self.derivative.borrow().clone(),
            &self.shape).unwrap();

        // if lhs_parent exists
        if let Some(t) = self.lhs_parent {

            // calculate the gradient of lhs_parent wrt x
            // d_lhs/d_x
            let d_lhs = match &self.create_op {
                Some(Ops::TensorOp(TensorOp::Add)) =>
                    Tensor::new(vec![1.0;
                        Tensor::calc_tensor_len_from_shape(&self.shape)],
                        &self.shape,
                ),
                Some(Ops::TensorOp(TensorOp::Sub)) =>
                    Tensor::new(
                        vec![1.0;
                        Tensor::calc_tensor_len_from_shape(&self.shape)],
                        &self.shape,
                ),
                Some(Ops::TensorOp(TensorOp::Mul)) =>
                    Ok(self.rhs_parent.unwrap().clone()),
                Some(Ops::TensorOp(TensorOp::Div)) => {
                    let temp = self.rhs_parent.unwrap()
                        .pow(-1.0).unwrap();

                    Tensor::new(temp.data.clone(), &temp.shape)
                }
                _ => Err(TensorError::GradError),
            }
            .unwrap();

            // calculate the gradient of lhs_parent wrt loss
            // d_lhs/d_loss = d_lhs/d_x * d_x/d_loss
            let d_lhs = match self.create_op {
                _ => &d_lhs * &d,
            };

            // accumulate the gradient of d_lhs/d_loss if necessary
            let d_lhs_prev =
                Tensor::new(t.derivative.borrow().clone(), &t.shape).unwrap();
            let d_lhs = &d_lhs + &d_lhs_prev;

            // assign to the derivative of the parent
            *t.derivative.borrow_mut() = d_lhs.data;
        }

        // if rhs_parent exists
        if let Some(t) = self.rhs_parent {

            // calculate the gradient of rhs_parent wrt x
            // d_rhs/d_x
            let d_rhs = match self.create_op {
                Some(Ops::TensorOp(TensorOp::Add)) => Tensor::new(
                    vec![1.0; Tensor::calc_tensor_len_from_shape(&self.shape)],
                    &self.shape,
                ),
                Some(Ops::TensorOp(TensorOp::Sub)) => Tensor::new(
                    vec![1.0; Tensor::calc_tensor_len_from_shape(&self.shape)],
                    &self.shape,
                ),
                Some(Ops::TensorOp(TensorOp::Mul)) =>
                    Ok(self.lhs_parent.unwrap().clone()),
                Some(Ops::TensorOp(TensorOp::Div)) => {
                    let neg1 = Tensor::new(vec![-1.0], &[1]).unwrap();
                    let t_powed = t.pow(-2.0).unwrap();

                    let temp = &neg1 * self.lhs_parent.unwrap();
                    let temp = &temp * &t_powed;

                    Tensor::new(temp.data.clone(), &temp.shape)
                }
                _ => Err(TensorError::GradError),
            }
            .unwrap();

            // calculate the gradient of rhs_parent wrt loss
            // d_rhs/d_loss = d_rhs/d_x * d_x/d_loss
            let d_rhs = match self.create_op {
                _ => &d_rhs * &d,
            };

            // accumulate the gradient of d_rhs/d_loss if necessary
            let d_rhs_prev =
                Tensor::new(t.derivative.borrow().clone(), &t.shape).unwrap();
            let d_rhs = &d_rhs + &d_rhs_prev;

            // assign to the derivative of the parent
            *t.derivative.borrow_mut() = d_rhs.data;
        }
    }
}
```

---

That should be pretty much it. Try it out:

```rust
// src/bin/main.rs

fn main() -> Result<(), TensorError> {
    let a = Tensor::new(vec![2.0], &[1]).unwrap();
    let b = Tensor::new(vec![3.0], &[1]).unwrap();

    let c = &a * &b;

    c.backward();

    println!("{}", c);

    Ok(())
}
```

```bash
bilal@Bilals-MacBook-Pro l2 % cargo run
Value: [6.0]
Shape: [1]
Lhs:
  Value: [2.0]
  Shape: [1]
  Lhs: None
  Rhs: None
  Op: None
  TrackGrad: true
  Derivative: [3.0]
Rhs:
  Value: [3.0]
  Shape: [1]
  Lhs: None
  Rhs: None
  Op: None
  TrackGrad: true
  Derivative: [2.0]
Op: Multiply
TrackGrad: true
Derivative: [1.0]
```

🎉, you now have a semi-complete autograd engine!

---

# Advanced Ops

---

Let's add support for fast matrix multiplications with BLAS.

_todo_ talk about blas

First up, lets implement the `transpose()` operator

---

```rust
// src/ops.rs

#[derive(Debug, PartialEq)]
pub enum Ops {
    ...
    Transpose,
}

impl fmt::Display for Ops {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ...
            Ops::Transpose => write!(f, "Transpose"),
        }
    }
}
```

```rust
// src/tensor.rs

impl<'a> Tensor<'a> {
    fn grad(&self) {
        if let Some(t) = self.lhs_parent {
            ...
            Some(Ops::Transpose) => Tensor::new(vec![1.0], &[1]), // dummy value
        }
        .unwrap();

        let d_lhs = match self.create_op {
            Some(Ops::Transpose) => d.transpose().unwrap(),
            _ => &d_lhs * &d,
        };
    }

    pub fn transpose(&self) -> Result<Tensor, TensorError> {
        let mut transposed_shape = self.shape.clone();
        let mut transposed_strides = self.strides.clone();

        transposed_shape.reverse();
        transposed_strides.reverse();

        let mut new_data: Vec<f32> =
            Vec::with_capacity(
                Tensor::calc_tensor_len_from_shape(&transposed_shape));

        for i in 0..transposed_shape[0] {
            if transposed_shape.len() == 1 {
                new_data.push(self.data[Tensor::
                    get_physical_idx(&[i], &transposed_strides)]
                );
            } else {
                for j in 0..transposed_shape[1] {
                    new_data.push(self.data[Tensor::
                        get_physical_idx(&[i, j], &transposed_strides)]);
                }
            }
        }

        Ok(Tensor {
            data: new_data,
            shape: transposed_shape.to_vec(),
            strides: Tensor::calc_strides_from_shape(&transposed_shape),
            track_grad: true,
            create_op: Some(Ops::Transpose),
            derivative: RefCell::new(vec![
                0.0;
                Tensor::calc_tensor_len_from_shape(&transposed_shape)
            ]),
            lhs_parent: Some(self),
            rhs_parent: None,
        })
    }
}
```

---

Now that we have this, let's add matmul support.

First up, let's add a BLAS crate to `Cargo.toml`. Note that I'm using Apple's accelerate as the BLAS library backend since its already installed on my Macbook pro, but you can change it to use [another](https://crates.io/crates/blas-src) BLAS library if you want.

---

```toml
[dependencies]
blas = "0.20.0"
blas-src = { version = "0.6", features = ["accelerate"] }
```

---

Let's add some Ops and errors for matmul

---

```rust
// src/ops.rs

#[derive(Debug, PartialEq)]
pub enum Ops {
    ...
    Matmul,
}

impl fmt::Display for Ops {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ...
            Ops::Matmul => write!(f, "Matmul"),
        }
    }
}
```

```rust
// src/errors.rs
#[derive(Debug, Clone)]
pub enum TensorError {
    ...
    MatmulShapeError,
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ...
            TensorError::MatmulShapeError => write!(
                f,
                "Tensors must be two dimensions each and must be matrix multipliable"
            ),
        }
    }
}

```

---

Let's add the matrix multiplication code

---

```rust
impl<'a> Tensor<'a> {
    #[allow(clippy::many_single_char_names)]
    fn two_dimension_matmul(lhs: &Tensor, rhs: &Tensor, out: &mut Vec<f32>) {
        let lhs = lhs.transpose().unwrap();
        let rhs = rhs.transpose().unwrap();

        let a: Vec<f64> = lhs.data.iter().map(|val| *val as f64).collect();
        let b: Vec<f64> = rhs.data.iter().map(|val| *val as f64).collect();

        let mut c: Vec<f64> =
            vec![0.0; Tensor::calc_tensor_len_from_shape(&[lhs.shape[1],
                rhs.shape[0]])];

        let (m, n, k) = (
            lhs.shape[1] as i32,
            rhs.shape[0] as i32,
            lhs.shape[0] as i32,
        );

        unsafe {
            dgemm(b'N', b'N', m, n, k, 1.0, &a, m, &b, k, 1.0, &mut c, m);
        }

        let c = c.iter().map(|val| *val as f32).collect();
        let c = Tensor::new(c, &[rhs.shape[0], lhs.shape[1]]).unwrap();
        let c = c.transpose().unwrap();

        let mut c = c.data;

        out.append(&mut c);
    }

    pub fn matmul(&'a self, rhs: &'a Tensor) -> Result<Tensor, TensorError> {
        let new_shape = Tensor::validate_tensors(self, &rhs)?;

        if (new_shape.len() <= 1) || (new_shape.len() > 2) {
            return Err(TensorError::MaxDimsError);
        }

        let mut new_data = Vec::with_capacity(Tensor::
            calc_tensor_len_from_shape(&new_shape));

        if new_shape.len() == 2 {
            Tensor::two_dimension_matmul(&self, rhs, &mut new_data)
        } else {
            return Err(TensorError::MatmulShapeError);
        }

        Ok(Tensor {
            data: new_data,
            shape: new_shape.to_vec(),
            strides: Tensor::calc_strides_from_shape(&new_shape),
            track_grad: true,
            create_op: Some(Ops::Matmul),
            derivative: RefCell::new(vec![0.0; Tensor::
                calc_tensor_len_from_shape(&new_shape)]),
            lhs_parent: Some(self),
            rhs_parent: Some(rhs),
        })
    }
}
```

---

Now that we have that, let's add autograd support for matmul

---

```rust
// src/tensor.rs
impl<'a> Tensor<'a> {
    fn grad(&self) {

        if let Some(t) = self.lhs_parent {
            let d_lhs = match &self.create_op {
                ...
                Some(Ops::Matmul) => self.rhs_parent.unwrap().transpose(),
            }
            .unwrap();

            let d_lhs = match self.create_op {
                ...
                Some(Ops::Matmul) => d.matmul(&d_lhs).unwrap(),
                _ => &d_lhs * &d,
            };
        }

        if let Some(t) = self.rhs_parent {
            let d_rhs = match self.create_op {
                ...
                Some(Ops::Matmul) => self.lhs_parent.unwrap().transpose(),
            }
            .unwrap();

            let d_rhs = match self.create_op {
                ...
                Some(Ops::Matmul) => d_rhs.matmul(&d).unwrap(),
            };

            let d_rhs_prev =
                Tensor::new(t.derivative.borrow().clone(), &t.shape).unwrap();
            let d_rhs = &d_rhs + &d_rhs_prev;
            *t.derivative.borrow_mut() = d_rhs.data;
        }
    }
}
```

---

Let's try it out:

```rust
// src/main.rs
fn main() -> Result<(), TensorError> {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();

    let c = a.matmul(&b).unwrap();

    c.backward();

    println!("{}", c);

    Ok(())
}
```

```bash
bilal@Bilals-MacBook-Pro l2 % cargo run
   Compiling l2 v0.1.0 (/Users/bilal/Desktop/l2)
    Finished dev [unoptimized + debuginfo] target(s) in 1.46s
     Running `target/debug/main`

Value: [22.0, 28.0, 49.0, 64.0]
Shape: [2, 2]
Lhs:
  Value: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  Shape: [2, 3]
  Lhs: None
  Rhs: None
  Op: None
  TrackGrad: true
  Derivative: [3.0, 7.0, 11.0, 3.0, 7.0, 11.0]
Rhs:
  Value: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  Shape: [3, 2]
  Lhs: None
  Rhs: None
  Op: None
  TrackGrad: true
  Derivative: [5.0, 5.0, 7.0, 7.0, 9.0, 9.0]
Op: Matmul
TrackGrad: true
Derivative: [1.0, 1.0, 1.0, 1.0]
```

---

Well thats pretty much it for the first draft. Ill see about adding more stuff when I redo this whole post.

---

# Future Work

---

-   rust arrays vs vec
    -   const generics
-   jax
-   compiler in rust

---

# Conclusions

---

_todo_

-   benchmarks
-   subsections
-   gradient vs derivative
-   standardize code snippets
-   move implementing ops to beginning
-   naive matmul
-   slicing?

---

# Resources

---

---

# References

---

[^1]: I guess the fact that I like to spend my last free summer working on a side project says a lot about me :p
[^2]: I'm almost certain that there are a few bugs in how I handle backpropogation through broadcasted tensors
[^3]: That's the summer of 2019, for those of you reading this in the near or not so near future :)
[^4]: https://blog.ezyang.com/2019/05/pytorch-internals/'>http://blog.ezyang.com/2019/05/pytorch-internals
[^5]: In my defense, I was pretty bad at algorithmy stuff back then
