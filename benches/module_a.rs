#![feature(test)]

extern crate test;

#[cfg(test)]
mod tests {
    use test::Bencher;

    use l2::module_abc::module_a::A;

    #[bench]
    fn bench_test_a(b: &mut Bencher) {
        b.iter(|| {
            let _first = A {
                a_1: String::from("Hello"),
                a_2: 1,
                a_3: 2,
            };
        });
    }

    #[bench]
    fn bench_test_run_a(b: &mut Bencher) {
        let first = A {
            a_1: String::from("Hello"),
            a_2: 1,
            a_3: 2,
        };

        b.iter(|| {
            first.calculate_a();
        });
    }
}
