#![feature(test)]

extern crate test;

#[cfg(test)]
mod tests {
    use test::Bencher;

    use l2::tensor::*;
    #[bench]
    fn bench_allocate_1d_tensor(b: &mut Bencher) {
        b.iter(|| {
            let _t = Tensor::zeros(&[64 * 64]).unwrap();
        });
    }

    #[bench]
    fn bench_slice_1d_tensor_chunk(b: &mut Bencher) {
        let t = Tensor::zeros(&[64 * 64]).unwrap();
        b.iter(|| {
            let _x = t.slice(&[[0, 64]]).unwrap();
        })
    }

    #[bench]
    fn bench_allocate_2d_tensor(b: &mut Bencher) {
        b.iter(|| {
            let _t = Tensor::zeros(&[64, 64]).unwrap();
        });
    }

    #[bench]
    fn bench_slice_2d_tensor_row(b: &mut Bencher) {
        let t = Tensor::zeros(&[64, 64]).unwrap();
        b.iter(|| {
            let _x = t.slice(&[[0, 1], [0, 64]]).unwrap();
        })
    }

    #[bench]
    fn bench_slice_2d_tensor_col(b: &mut Bencher) {
        let t = Tensor::zeros(&[64, 64]).unwrap();
        b.iter(|| {
            let _x = t.slice(&[[0, 64], [0, 1]]).unwrap();
        })
    }

    #[bench]
    fn bench_slice_2d_tensor_chunk(b: &mut Bencher) {
        let t = Tensor::zeros(&[64, 64]).unwrap();
        b.iter(|| {
            let _x = t.slice(&[[0, 16], [0, 16]]).unwrap();
        })
    }
    #[bench]
    fn bench_allocate_3d_tensor(b: &mut Bencher) {
        b.iter(|| {
            let _t = Tensor::zeros(&[64, 64, 64]).unwrap();
        });
    }
    #[bench]
    fn bench_slice_3d_tensor_row(b: &mut Bencher) {
        let t = Tensor::zeros(&[64, 64, 64]).unwrap();
        b.iter(|| {
            let _x = t.slice(&[[0, 1], [0, 1], [0, 64]]).unwrap();
        })
    }

    #[bench]
    fn bench_slice_3d_tensor_col(b: &mut Bencher) {
        let t = Tensor::zeros(&[64, 64, 64]).unwrap();
        b.iter(|| {
            let _x = t.slice(&[[0, 1], [0, 64], [0, 1]]).unwrap();
        })
    }

    #[bench]
    fn bench_slice_3d_tensor_channel(b: &mut Bencher) {
        let t = Tensor::zeros(&[64, 64, 64]).unwrap();
        b.iter(|| {
            let _x = t.slice(&[[0, 64], [0, 1], [0, 1]]).unwrap();
        })
    }

    #[bench]
    fn bench_slice_3d_tensor_chunk(b: &mut Bencher) {
        let t = Tensor::zeros(&[64, 64, 64]).unwrap();
        b.iter(|| {
            let _x = t.slice(&[[0, 16], [0, 16], [0, 16]]).unwrap();
        })
    }

    #[bench]
    fn bench_allocate_4d_tensor_small(b: &mut Bencher) {
        b.iter(|| {
            let _t = Tensor::zeros(&[16, 16, 16, 16]).unwrap();
        });
    }
    #[bench]
    fn bench_slice_4d_tensor_row(b: &mut Bencher) {
        let t = Tensor::zeros(&[64, 64, 64, 64]).unwrap();
        b.iter(|| {
            let _x = t.slice(&[[0, 1], [0, 1], [0, 1], [0, 64]]).unwrap();
        })
    }

    #[bench]
    fn bench_slice_4d_tensor_col(b: &mut Bencher) {
        let t = Tensor::zeros(&[64, 64, 64, 64]).unwrap();
        b.iter(|| {
            let _x = t.slice(&[[0, 1], [0, 1], [0, 64], [0, 1]]).unwrap();
        })
    }

    #[bench]
    fn bench_slice_4d_tensor_channel(b: &mut Bencher) {
        let t = Tensor::zeros(&[64, 64, 64, 64]).unwrap();
        b.iter(|| {
            let _x = t.slice(&[[0, 1], [0, 64], [0, 1], [0, 1]]).unwrap();
        })
    }

    #[bench]
    fn bench_slice_4d_tensor_batch(b: &mut Bencher) {
        let t = Tensor::zeros(&[64, 64, 64, 64]).unwrap();
        b.iter(|| {
            let _x = t.slice(&[[0, 64], [0, 1], [0, 1], [0, 1]]).unwrap();
        })
    }

    #[bench]
    fn bench_slice_4d_tensor_chunk(b: &mut Bencher) {
        let t = Tensor::zeros(&[64, 64, 64, 64]).unwrap();
        b.iter(|| {
            let _x = t.slice(&[[0, 16], [0, 16], [0, 16], [0, 16]]).unwrap();
        })
    }
}
