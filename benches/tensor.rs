#![feature(test)]

extern crate test;

#[cfg(test)]
mod tests {
    use test::Bencher;

    use l2::tensor::*;

    #[bench]
    fn bench_allocate_tensor(b: &mut Bencher) {
        b.iter(|| {
            let data = vec![0.0; 1424 * 1024];
            let _t = Tensor { data };
        });
    }

    #[bench]
    fn bench_allocate_tensor_zeros(b: &mut Bencher) {
        b.iter(|| {
            let _t = Tensor::zeros(1024 * 1024);
        });
    }
}
