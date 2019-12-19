use std::arch::x86_64::*;
use std::ops::{ Add, Mul, Sub, AddAssign, MulAssign, SubAssign };
use std::cmp::{ PartialEq, Eq };
use std::fmt::{ Debug, Display, Formatter };

use super::zq::Zq;

#[cfg(test)]
use super::zq::ZERO;

pub const Q: i32 = 7681;

#[derive(Clone, Copy)]
pub struct Zq8
{
    // 8 x 32bit integer
    data: __m256i
}

unsafe fn avx_q() -> __m256i { _mm256_set1_epi32 (Q) }
unsafe fn avx_negative_q() -> __m256i { _mm256_set1_epi32 (-Q) }
unsafe fn avx_q_minus_one() -> __m256i { _mm256_set1_epi32 (Q-1) }
static ONE_OVER_Q: f32 = 1. / 7681.;
unsafe fn avx_one_over_q() -> __m256 { _mm256_set1_ps(ONE_OVER_Q) }
unsafe fn avx_zero() -> __m256i { _mm256_setzero_si256() }

impl Zq8 
{
    fn zero() -> Zq8
    {
        Zq8 {
            data: unsafe { _mm256_setzero_si256() }
        }
    }

    fn get<const index: i32>(&self) -> Zq
    {
        Zq::from(unsafe { _mm256_extract_epi32(self.data, index) } as u16)
    }

    fn set<const index: i32>(&mut self, value: Zq)
    {
        unsafe {
            self.data = _mm256_blend_epi32(self.data, _mm256_set1_epi32(value.representative_pos() as i32), 1 << index);
        }
    }
}

impl From<(Zq, Zq, Zq, Zq, Zq, Zq, Zq, Zq)> for Zq8
{
    fn from(value: (Zq, Zq, Zq, Zq, Zq, Zq, Zq, Zq)) -> Zq8
    {
        Zq8 {
            data: unsafe { _mm256_setr_epi32(
                value.0.representative_pos() as i32, 
                value.1.representative_pos() as i32, 
                value.2.representative_pos() as i32, 
                value.3.representative_pos() as i32, 
                value.4.representative_pos() as i32, 
                value.5.representative_pos() as i32, 
                value.6.representative_pos() as i32, 
                value.7.representative_pos() as i32) 
            }
        }
    }
}

impl<'a> AddAssign<&'a Zq8> for Zq8
{
    fn add_assign(&mut self, rhs: &'a Zq8)
    {
        unsafe {
            let sum = _mm256_add_epi32(self.data, rhs.data);
            let too_great = _mm256_cmpgt_epi32(sum, avx_q_minus_one());
            self.data = _mm256_add_epi32(sum, _mm256_and_si256(too_great, avx_negative_q()));
        }
    }
}

impl<'a> SubAssign<&'a Zq8> for Zq8
{
    fn sub_assign(&mut self, rhs: &'a Zq8)
    {
        unsafe {
            let difference = _mm256_sub_epi32(self.data, rhs.data);
            let too_small = _mm256_cmpgt_epi32(avx_zero(), difference);
            self.data = _mm256_add_epi32(difference, _mm256_and_si256(too_small, avx_q()));
        }
    }
}

impl<'a> MulAssign<&'a Zq8> for Zq8
{
    #[no_mangle]
    fn mul_assign(&mut self, rhs: &'a Zq8)
    {
        // use floating point arithmetic with correction
        // in order to perform modulo using multiplication
        // (this requires the upper bits of a 26 + 24 bit product)
        // the floating point arithmetic gives us at least
        // the upper bits of a 24 + 24 bit product, so correction
        // is required (see test_modulo_q)
        unsafe {
            let product: __m256i = _mm256_mullo_epi32(self.data, rhs.data);
            let product_float: __m256 = _mm256_cvtepi32_ps(product); 
            let quotient: __m256 = _mm256_mul_ps(product_float, avx_one_over_q());
            let rounded_quotient: __m256i = _mm256_cvttps_epi32(quotient);
            let rest: __m256i = _mm256_sub_epi32(product, _mm256_mullo_epi32(rounded_quotient, avx_q()));
            // apply correction: rest is now in -7681 to 2 * 7680 - 1
            let too_small = _mm256_cmpgt_epi32(avx_zero(), rest);
            let too_big = _mm256_cmpgt_epi32(rest, avx_q_minus_one());
            let correction = _mm256_or_si256(_mm256_and_si256(too_small, avx_q()), _mm256_and_si256(too_big, avx_negative_q()));
            self.data = _mm256_add_epi32(rest, correction);
        }
    }
}

#[test]
fn test_get_set() {
    let mut vector = Zq8::zero();
    assert_eq!(ZERO, vector.get::<0_i32>());
    assert_eq!(ZERO, vector.get::<1_i32>());
    assert_eq!(ZERO, vector.get::<2_i32>());
    assert_eq!(ZERO, vector.get::<3_i32>());
    assert_eq!(ZERO, vector.get::<4_i32>());
    assert_eq!(ZERO, vector.get::<5_i32>());
    assert_eq!(ZERO, vector.get::<6_i32>());
    assert_eq!(ZERO, vector.get::<7_i32>());

    vector.set::<1_i32>(Zq::from(1_u16));
    vector.set::<2_i32>(Zq::from(2_u16));
    vector.set::<3_i32>(Zq::from(3_u16));
    vector.set::<4_i32>(Zq::from(4_u16));
    vector.set::<5_i32>(Zq::from(5_u16));
    vector.set::<6_i32>(Zq::from(6_u16));
    vector.set::<7_i32>(Zq::from(7_u16));

    assert_eq!(0, vector.get::<0_i32>().representative_pos());
    assert_eq!(1, vector.get::<1_i32>().representative_pos());
    assert_eq!(2, vector.get::<2_i32>().representative_pos());
    assert_eq!(3, vector.get::<3_i32>().representative_pos());
    assert_eq!(4, vector.get::<4_i32>().representative_pos());
    assert_eq!(5, vector.get::<5_i32>().representative_pos());
    assert_eq!(6, vector.get::<6_i32>().representative_pos());
    assert_eq!(7, vector.get::<7_i32>().representative_pos());
}

#[test]
fn test_modulo_q() {
    for x in 0..7680*7680 {
        let p: i32 = x as i32;
        let f: f32 = p as f32;
        let q: f32 = f * ONE_OVER_Q;
        let r: i32 = q.floor() as i32;
        let mut m: i32 = p - r * 7681;
        if m < 0 {
            m += 7681;
        } else if m >= 7681 {
            m -= 7681;
        }
        assert!((x as i32) % 7681 == m, "Expected {}, got {} at for {}", (x as i32) % 7681, m, x);
    }
}