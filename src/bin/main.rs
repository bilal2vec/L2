// use l2;
use l2::tensor::*;

#[allow(clippy::many_single_char_names)]
fn main() -> Result<(), l2::errors::TensorError> {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
    let y = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();

    let one = Tensor::new(vec![1.0], &[1]).unwrap();
    let two = Tensor::new(vec![2.0], &[1]).unwrap();

    let temp = Tensor::new(vec![3.0], &[1]).unwrap();

    let t2 = &one * &temp;

    let xx = &t2 * &x;
    let yy = &two * &y;

    let z = xx.concat(&yy, -1)?;

    z.backward();

    println!("{:#?}", z);

    // let z = Tensor::new(vec![-4.0], &[1])?;

    // let out = &q * &z;

    // out.backward();

    // println!("{}", out);

    // let a = Tensor::new(vec![2.0], &[1])?;
    // let b = Tensor::new(vec![3.0], &[1])?;
    // let c = Tensor::new(vec![4.0], &[1])?;

    // let e = &a * &b;
    // let f = &e * &c;

    // let out = &a + &f;

    // out.backward();

    // println!("{}", out);

    // let h = Tensor::new(vec![5.0], &[1])?;
    // let i = Tensor::new(vec![6.0], &[1])?;
    // let j = Tensor::new(vec![7.0], &[1])?;

    // let k = &g * &h;

    // let m = &k * &i;
    // let n = &m + &j;

    // let o = &h + &n;

    // o.backward();

    // println!("{}", o);

    Ok(())
}
