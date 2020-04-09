#![feature(test)]

extern crate test;

#[cfg(test)]
mod tests {
    use test::Bencher;

    use l2::tensor::*;
    #[bench]
    fn bench_allocate_tensor_zeros(b: &mut Bencher) {
        b.iter(|| {
            let _t = Tensor::zeros(&[512 * 512]);
        });
    }

    #[bench]
    fn bench_slice_1d_tensor_small(b: &mut Bencher) {
        let t = Tensor::zeros(&[512 * 512]);
        b.iter(|| {
            let _x = t.slice(&[[0, 128]]);
        })
    }

    #[bench]
    fn bench_slice_1d_tensor_large(b: &mut Bencher) {
        let t = Tensor::zeros(&[512 * 512]);
        b.iter(|| {
            let _x = t.slice(&[[0, 512]]);
        })
    }

    #[bench]
    fn bench_slice_1d_tensor_xl(b: &mut Bencher) {
        let t = Tensor::zeros(&[1024 * 1024]);
        b.iter(|| {
            let _x = t.slice(&[[0, 512 * 512]]);
        })
    }
}
