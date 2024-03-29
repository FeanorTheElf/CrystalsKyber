use std::arch::x86_64::*;

use super::util;

pub unsafe fn constant_i32<const C: i32>() -> __m256i
{
    _mm256_set1_epi32(C)
}

pub unsafe fn constant_u32<const C: u32>() -> __m256i
{
    _mm256_set1_epi32(std::mem::transmute(C))
}

pub unsafe fn constant_f32<const C: f32>() -> __m256
{
    _mm256_set1_ps(C)
}

pub unsafe fn constant_zero() -> __m256i
{
    _mm256_setzero_si256()
}

/// Transposes a matrix, where the entries of each row are vectorized, i.e. 8 elements are grouped in
/// one avx vector. In the result matrix, also the entries of each row will be grouped in avx vectors
/// the same way.
/// 
/// COL_COUNT: the count of columns in the matrix (the count of entry columns, not of vector columns).
/// 
/// VEC_COUNT: the count of all avx vectors in the matrix. Must be divisible by COL_COUNT.
pub unsafe fn transpose_vectorized_matrix<const COL_COUNT: usize, const VEC_COUNT: usize>(value: [__m256i; VEC_COUNT]) -> [__m256i; VEC_COUNT]
{
    const VEC_SIZE: usize = 8;
    assert!(VEC_COUNT % COL_COUNT == 0);

    let vector_count_per_col: usize = VEC_COUNT / COL_COUNT;
    let indices: __m256i = _mm256_setr_epi32(
        0, 
        COL_COUNT as i32, 
        COL_COUNT as i32 * 2, 
        COL_COUNT as i32 * 3,
        COL_COUNT as i32 * 4,
        COL_COUNT as i32 * 5,
        COL_COUNT as i32 * 6,
        COL_COUNT as i32 * 7);
    
    let matrix_begin: *const i32 = std::mem::transmute(value.as_ptr());
    return util::create_array_it(&mut util::cartesian(0..COL_COUNT, 0..vector_count_per_col).map(
            |(result_row, result_col): (usize, usize)|
        {
            let vector_begin: *const i32 = matrix_begin.offset((result_row + result_col * COL_COUNT * VEC_SIZE) as isize);
            return _mm256_i32gather_epi32(vector_begin, indices, 4);
        }
    ));
}

#[inline(always)]
pub unsafe fn compose<const IN: usize, const OUT: usize>(x: [i32; IN]) -> [__m256i; OUT]
{
    std::mem::transmute_copy::<[i32; IN], [__m256i; OUT]>(&x)
}

#[inline(always)]
pub unsafe fn decompose<const IN: usize, const OUT: usize>(x: [__m256i; IN]) -> [i32; OUT]
{
    std::mem::transmute_copy::<[__m256i; IN], [i32; OUT]>(&x)
}

#[inline(always)]
pub unsafe fn eq(x: __m256i, y: __m256i) -> bool
{
    let equality: __m256i = _mm256_cmpeq_epi32(x, y);
    let bitmask: i32 = _mm256_movemask_epi8(equality);
    return bitmask == !0;
}

#[test]
fn test_transpose() {
    unsafe {
        let matrix: [__m256i; 4] = compose::<32, 4>([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]);
        let transposed = decompose::<4, 32>(transpose_vectorized_matrix::<4, 4>(matrix));
        let expected: [i32; 32] = [0, 4, 8, 12, 16, 20, 24, 28, 1, 5, 9, 13, 17, 21, 25, 29, 2, 6, 10, 14, 18, 22, 26, 30, 3, 7, 11, 15, 19, 23, 27, 31];
        assert_eq!(expected, transposed);
    }
}

#[test]
fn test_transpose_greater() {
    unsafe {
        let matrix: [__m256i; 8] = compose::<64, 8>([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 
            29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]);
        let transposed = decompose::<8, 64>(transpose_vectorized_matrix::<4, 8>(matrix));
        let expected: [i32; 64] = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61,
            2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63];
        assert_eq!(expected[0..32], transposed[0..32]);
        assert_eq!(expected[32..64], transposed[32..64]);
    }
}