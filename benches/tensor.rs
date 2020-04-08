#![feature(test)]

extern crate test;

#[cfg(test)]
mod tests {
    use test::Bencher;

    use l2::tensor::*;

    #[bench]
    fn bench_allocate_tensor_zeros(b: &mut Bencher) {
        b.iter(|| {
            let _t = Tensor::zeros(&[256, 256]);
        });
    }
}
