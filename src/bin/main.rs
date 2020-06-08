use l2;
use l2::tensor::*;

fn main() -> Result<(), l2::errors::TensorError> {
    let x = Tensor::new(vec![-2.0], &[1])?;
    let y = Tensor::new(vec![5.0], &[1])?;

    let q = &x + &y;

    let z = Tensor::new(vec![-4.0], &[1])?;

    let g = &q * &z;

    g.backward();

    println!("{}", g);

    // let a = Tensor::new(vec![2.0], &[1])?;
    // let b = Tensor::new(vec![3.0], &[1])?;
    // let c = Tensor::new(vec![4.0], &[1])?;

    // let e = &a * &b;
    // let f = &e * &c;

    // let g = &a + &f;

    // let h = Tensor::new(vec![5.0], &[1])?;
    // let i = Tensor::new(vec![6.0], &[1])?;
    // let j = Tensor::new(vec![7.0], &[1])?;

    // let k = &g * &h;

    // let m = &k * &i;
    // let n = &m + &j;

    // let o = &h + &n;

    // o.backward();

    // println!("{:?}", o);

    Ok(())
}
