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
    fn bench_slice_1d_tensor_all(b: &mut Bencher) {
        let t = Tensor::zeros(&[512 * 512]);
        b.iter(|| {
            let _x = t.slice(&[[0, 512 * 512]]);
        })
    }

    #[bench]
    fn bench_slice_2d_tensor_row(b: &mut Bencher) {
        let t = Tensor::zeros(&[512, 512]);
        b.iter(|| {
            let _x = t.slice(&[[0, 1], [0, 512]]);
        })
    }

    #[bench]
    fn bench_slice_2d_tensor_col(b: &mut Bencher) {
        let t = Tensor::zeros(&[512, 512]);
        b.iter(|| {
            let _x = t.slice(&[[0, 512], [0, 1]]);
        })
    }

    #[bench]
    fn bench_slice_2d_tensor_small_chunk(b: &mut Bencher) {
        let t = Tensor::zeros(&[512, 512]);
        b.iter(|| {
            let _x = t.slice(&[[0, 64], [0, 64]]);
        })
    }

    #[bench]
    fn bench_slice_2d_tensor_large_chunk(b: &mut Bencher) {
        let t = Tensor::zeros(&[512, 512]);
        b.iter(|| {
            let _x = t.slice(&[[0, 128], [0, 128]]);
        })
    }

    #[bench]
    fn bench_slice_2d_tensor_xl_chunk(b: &mut Bencher) {
        let t = Tensor::zeros(&[512, 512]);
        b.iter(|| {
            let _x = t.slice(&[[0, 256], [0, 256]]);
        })
    }

    #[bench]
    fn bench_slice_2d_tensor_all(b: &mut Bencher) {
        let t = Tensor::zeros(&[512, 512]);
        b.iter(|| {
            let _x = t.slice(&[[0, 512], [0, 512]]);
        })
    }
}
