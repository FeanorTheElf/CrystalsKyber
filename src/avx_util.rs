use std::arch::x86_64::*;
use std::mem::MaybeUninit;

use super::util;

pub unsafe fn constant_i32<const C: i32>() -> __m256i
{
    _mm256_set1_epi32(C)
}

pub unsafe fn constant_f32<const C: f32>() -> __m256
{
    _mm256_set1_ps(C)
}

pub unsafe fn constant_zero() -> __m256i
{
    _mm256_setzero_si256()
}

pub unsafe fn rising_indices() -> __m256i
{
    _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)
}

pub unsafe fn transpose_vectorized_matrix<const COL_COUNT: usize, const VEC_COUNT: usize>(value: [__m256i; VEC_COUNT]) -> [__m256i; VEC_COUNT]
{
    const VEC_SIZE: usize = 8;

    assert_eq!(VEC_COUNT, value.len());
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
    return util::create_array_it(util::cartesian(0..COL_COUNT, 0..vector_count_per_col).map(
            |(result_row, result_col): (usize, usize)|
        {
            let vector_begin: *const i32 = matrix_begin.offset((result_row + result_col * COL_COUNT * VEC_SIZE) as isize);
            return _mm256_i32gather_epi32(vector_begin, indices, 4);
        }
    ));
}

pub unsafe fn horizontal_sum(x: __m256i) -> i32
{
    let low4: __m128i = _mm256_extractf128_si256(x, 0);
    let high4: __m128i = _mm256_extractf128_si256(x, 1);
    let sum4: __m128i = _mm_add_epi32(low4, high4);
    let low2: i64 = _mm_extract_epi64(sum4, 0);
    let high2: i64 = _mm_extract_epi64(sum4, 1);
    let sum2: i64 = low2 + high2;
    let sum: i32 = ((sum2 >> 32) + (sum2 & 0xFFFFFFFF)) as i32;
    return sum;
}

pub unsafe fn shift_left(amount: usize, x: __m256i) -> __m256i
{
    let i: [i32; 8] = util::shift_left(amount, [0, 1, 2, 3, 4, 5, 6, 7]);
    let index: __m256i = _mm256_setr_epi32(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7]);
    return _mm256_permutevar8x32_epi32(x, index);
}

#[inline(always)]
pub unsafe fn compose<const IN: usize, const OUT: usize>(x: [i32; IN]) -> [__m256i; OUT]
{
    assert_eq!(IN, OUT * 8);
    return util::create_array(|i| _mm256_setr_epi32(
        x[i * 8 + 0], x[i * 8 + 1], x[i * 8 + 2], x[i * 8 + 3], 
        x[i * 8 + 4], x[i * 8 + 5], x[i * 8 + 6], x[i * 8 + 7]));
}

#[inline(always)]
pub unsafe fn decompose<const IN: usize, const OUT: usize>(x: [__m256i; IN]) -> [i32; OUT]
{
    assert_eq!(IN * 8, OUT);
    let mut result: MaybeUninit<[i32; OUT]> = MaybeUninit::uninit();
    for i in 0..IN {
        let current_ptr = (*result.as_mut_ptr()).as_mut_ptr().offset((i * 8) as isize);
        std::ptr::write(current_ptr.offset(0), _mm256_extract_epi32(x[i], 0));
        std::ptr::write(current_ptr.offset(1), _mm256_extract_epi32(x[i], 1));
        std::ptr::write(current_ptr.offset(2), _mm256_extract_epi32(x[i], 2));
        std::ptr::write(current_ptr.offset(3), _mm256_extract_epi32(x[i], 3));
        std::ptr::write(current_ptr.offset(4), _mm256_extract_epi32(x[i], 4));
        std::ptr::write(current_ptr.offset(5), _mm256_extract_epi32(x[i], 5));
        std::ptr::write(current_ptr.offset(6), _mm256_extract_epi32(x[i], 6));
        std::ptr::write(current_ptr.offset(7), _mm256_extract_epi32(x[i], 7));
    }
    return result.assume_init();
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