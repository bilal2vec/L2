# Benches

## 6f963e5ac1c224f641c4d8d69a4a574506648cbf

### Benchmark

running 17 tests
test tests::bench_allocate_1d_tensor ... bench: 418 ns/iter (+/- 40)
test tests::bench_allocate_2d_tensor ... bench: 410 ns/iter (+/- 24)
test tests::bench_allocate_3d_tensor ... bench: 20,043 ns/iter (+/- 1,269)
test tests::bench_allocate_4d_tensor_small ... bench: 5,236 ns/iter (+/- 501)
test tests::bench_slice_1d_tensor_chunk ... bench: 543 ns/iter (+/- 26)
test tests::bench_slice_2d_tensor_chunk ... bench: 822 ns/iter (+/- 58)
test tests::bench_slice_2d_tensor_col ... bench: 566 ns/iter (+/- 61)
test tests::bench_slice_2d_tensor_row ... bench: 534 ns/iter (+/- 44)
test tests::bench_slice_3d_tensor_channel ... bench: 772 ns/iter (+/- 39)
test tests::bench_slice_3d_tensor_chunk ... bench: 8,376 ns/iter (+/- 546)
test tests::bench_slice_3d_tensor_col ... bench: 582 ns/iter (+/- 64)
test tests::bench_slice_3d_tensor_row ... bench: 568 ns/iter (+/- 78)
test tests::bench_slice_4d_tensor_batch ... bench: 875 ns/iter (+/- 46)
test tests::bench_slice_4d_tensor_channel ... bench: 774 ns/iter (+/- 68)
test tests::bench_slice_4d_tensor_chunk ... bench: 144,760 ns/iter (+/- 7,562)
test tests::bench_slice_4d_tensor_col ... bench: 604 ns/iter (+/- 68)
test tests::bench_slice_4d_tensor_row ... bench: 575 ns/iter (+/- 65)

## e1c07224ee46e820790d6e030dd1876c5f286c40

### Commits

-   Add support to slice with -1 and not need to specify all dims
-   add more view support

### Benchmark

test tests::bench_allocate_1d_tensor ... bench: 428 ns/iter (+/- 40)
test tests::bench_allocate_2d_tensor ... bench: 437 ns/iter (+/- 89)
test tests::bench_allocate_3d_tensor ... bench: 19,967 ns/iter (+/- 1,066)
test tests::bench_allocate_4d_tensor_small ... bench: 5,292 ns/iter (+/- 505)
test tests::bench_slice_1d_tensor_chunk ... bench: 716 ns/iter (+/- 95)
test tests::bench_slice_2d_tensor_chunk ... bench: 1,135 ns/iter (+/- 110)
test tests::bench_slice_2d_tensor_col ... bench: 819 ns/iter (+/- 87)
test tests::bench_slice_2d_tensor_col_neg_1 ... bench: 877 ns/iter (+/- 68)
test tests::bench_slice_2d_tensor_row ... bench: 823 ns/iter (+/- 65)
test tests::bench_slice_3d_tensor_channel ... bench: 1,061 ns/iter (+/- 138)
test tests::bench_slice_3d_tensor_chunk ... bench: 8,599 ns/iter (+/- 534)
test tests::bench_slice_3d_tensor_col ... bench: 865 ns/iter (+/- 58)
test tests::bench_slice_3d_tensor_row ... bench: 867 ns/iter (+/- 68)
test tests::bench_slice_3d_tensor_row_automatic_slicing ... bench: 1,146 ns/iter (+/- 150)
test tests::bench_slice_3d_tensor_row_neg_1 ... bench: 844 ns/iter (+/- 99)
test tests::bench_slice_4d_tensor_batch ... bench: 1,133 ns/iter (+/- 70)
test tests::bench_slice_4d_tensor_channel ... bench: 1,051 ns/iter (+/- 142)
test tests::bench_slice_4d_tensor_chunk ... bench: 142,424 ns/iter (+/- 11,338)
test tests::bench_slice_4d_tensor_col ... bench: 897 ns/iter (+/- 125)
test tests::bench_slice_4d_tensor_row ... bench: 843 ns/iter (+/- 71)
test tests::bench_view_2d_to_1d ... bench: 641 ns/iter (+/- 41)
test tests::bench_view_2d_to_1d_automatic_expanding ... bench: 654 ns/iter (+/- 64)
